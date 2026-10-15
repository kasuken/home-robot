"""The deterministic interaction policy (#39): whether Laika may start talking on her own.

The policy decides WHETHER to interact; the model only decides WHAT to say. The proactive path
(#40) calls `decide()` before it generates anything, and the agent has no tool that starts an
interaction or changes the cooldown, so the model cannot talk itself past the policy. The user
can always talk to Laika: only her own initiative is limited here.

`decide()` has no randomness and reads only the injected clocks, the presence tracker and what
the policy was told, so the same inputs give the same decision.

State, first match wins:

    SLEEPING         inside `quiet_hours` (local wall time, may cross midnight)
    DO_NOT_DISTURB   the user asked for it and it has not expired
    BUSY             a conversation (or a USER/CONVERSATION lease) holds the body, or the user
                     said they are busy and that has not expired
    AWAY             nobody seen for `away_after_s`
    AVAILABLE        a person is near now and has been for `settle_s`
    UNKNOWN          anything else (nobody seen yet, just left, just arrived)

The only rule for V0.6: allowed when the state is AVAILABLE, no interaction (a chat, or an earlier
proactive attempt) happened within the cooldown, there is something worth saying (a pending
question, a low battery, or an interest score at or above the threshold within `interest_window_s`; for an
encounter, #49, a short hello counts too) and fewer than the hourly limit of proactive attempts were made
in the last hour. Each refusal names its reason.

The personality (#41) scales the numbers the rule uses, never the states: `effective_tuning()`
turns `cooldown_s`, `max_per_hour` and `settle_s` into the `Tuning` the policy reads. DND, quiet
hours, busy and safety are not touched by any of it.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Callable

from laika.config import InteractionConfig, parse_quiet_hours
from laika.events import Event, EventBus, EventType
from laika.personality import (
    MIN_COOLDOWN_S,
    Personality,
    cooldown_factor,
    max_per_hour_for,
    present_floor_s,
    settle_factor,
)

if TYPE_CHECKING:
    from laika.presence import PresenceTracker
    from laika.store import Store

log = logging.getLogger(__name__)

NAMESPACE = "interaction"
DND_KEY = "do_not_disturb_until"
BUSY_KEY = "busy_until"
MIN_MINUTES = 5
MAX_MINUTES = 720
HOUR_S = 3600.0
MAX_TRACKED = 64
# A conversation whose INTERACTION_COMPLETED never arrived must not block her for ever.
MAX_CONVERSATION_S = 3600.0
REASON_ALLOWED = "allowed"


class Availability(str, Enum):
    AVAILABLE = "available"
    BUSY = "busy"
    DO_NOT_DISTURB = "do_not_disturb"
    AWAY = "away"
    SLEEPING = "sleeping"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Decision:
    allowed: bool
    # "allowed", or why not: disabled, sleeping, do_not_disturb, busy, away, unknown,
    # cooldown, nothing_to_say, rate_limit.
    reason: str
    state: Availability


@dataclass(frozen=True)
class Context:
    """What the caller knows that the policy cannot see for itself."""

    pending_questions: int = 0
    conversation_active: bool = False
    # An idle encounter (#49): she met this person while out and about, so a short hello is something
    # to say. Only the encounter handler sets it; every other rule still applies.
    encounter: bool = False


@dataclass(frozen=True)
class Tuning:
    """The numbers the rule uses, after the personality (see `effective_tuning`)."""

    cooldown_s: float
    max_per_hour: int
    settle_s: float
    min_present_s: float

    @property
    def present_s(self) -> float:
        """How long a person must have been near before she may speak."""
        return max(self.settle_s, self.min_present_s)


def effective_tuning(config: InteractionConfig, personality: Personality) -> Tuning:
    """`[interaction]` scaled by initiative and interruption tolerance, within these bounds:

    cooldown_s      * 0.5 (initiative 100) to * 2 (initiative 0), never below ten minutes
                    (or cooldown_s itself when that is shorter; 0 stays 0)
    max_per_hour    1 to 4 from initiative, never above [interaction] max_per_hour
    settle_s        * 0.5 (tolerance 100) to * 2 (tolerance 0)
    min_present_s   30 s when tolerance is below 20, else 0
    """
    scaled = config.cooldown_s * cooldown_factor(personality.initiative)
    return Tuning(
        cooldown_s=max(min(config.cooldown_s, MIN_COOLDOWN_S), scaled),
        max_per_hour=min(config.max_per_hour, max_per_hour_for(personality.initiative)),
        settle_s=config.settle_s * settle_factor(personality.interruption_tolerance),
        min_present_s=present_floor_s(personality.interruption_tolerance),
    )


class InteractionPolicy:
    def __init__(
        self,
        config: InteractionConfig,
        presence: PresenceTracker | None = None,
        *,
        events: EventBus | None = None,
        store: Store | None = None,
        threshold: float = 0.35,
        pending_questions: Callable[[], int] | None = None,
        body_busy: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        personality: Personality | None = None,
        battery_low: Callable[[], bool] | None = None,
    ) -> None:
        self.config = config
        self.personality = personality if personality is not None else Personality()
        self.tuning = effective_tuning(config, self.personality)
        self.presence = presence
        self.events = events
        self.store = store
        self.threshold = threshold
        self._pending_questions = pending_questions
        self._body_busy = body_busy
        # A low battery (#50) is something worth saying, within every rule below (never a shortcut past them).
        self._battery_low = battery_low
        self._clock = clock
        self._wall = wall_clock
        self._quiet = parse_quiet_hours(config.quiet_hours)
        self._lock = threading.Lock()
        self._created = clock()
        # (kind, clock time), kind: "started", "completed" or "attempt"; bounded.
        self._recent: deque[tuple[str, float]] = deque(maxlen=MAX_TRACKED)
        self._attempts: deque[float] = deque(maxlen=MAX_TRACKED)
        self._interest: deque[tuple[float, float]] = deque(maxlen=MAX_TRACKED)
        self._conversation_since: float | None = None
        self._dnd_until: float | None = None  # wall clock, so it survives a restart
        self._busy_until: float | None = None
        self._last_logged: str | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self._load()

    # --- lifecycle ---------------------------------------------------------------

    def start(self) -> None:
        if self.events is not None and self._unsubscribe is None:
            self._unsubscribe = self.events.subscribe(
                self._on_event,
                EventType.INTERACTION_STARTED,
                EventType.INTERACTION_COMPLETED,
                EventType.INTERESTING_CHANGE_DETECTED,
            )

    def close(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    # --- inputs ------------------------------------------------------------------

    def _on_event(self, event: Event) -> None:
        now = self._clock()
        with self._lock:
            if event.type is EventType.INTERACTION_STARTED:
                self._recent.append(("started", now))
                self._conversation_since = now
            elif event.type is EventType.INTERACTION_COMPLETED:
                self._recent.append(("completed", now))
                self._conversation_since = None
            elif event.type is EventType.INTERESTING_CHANGE_DETECTED:
                try:
                    self._interest.append((now, float(event.data.get("total", 0.0))))
                except (TypeError, ValueError):
                    pass

    def note_interest(self, total: float) -> None:
        """An interest score was seen now (the same thing InterestingChangeDetected tells it)."""
        with self._lock:
            self._interest.append((self._clock(), float(total)))

    def record_attempt(self) -> None:
        """The proactive path calls this when it starts an interaction: it counts for cooldown and rate."""
        now = self._clock()
        with self._lock:
            self._attempts.append(now)
            self._recent.append(("attempt", now))

    def recent_interactions(self) -> list[tuple[str, float]]:
        """The latest interaction events as (kind, clock time), oldest first."""
        with self._lock:
            return list(self._recent)

    # --- what the user asked for -------------------------------------------------

    @staticmethod
    def _check_minutes(minutes: float) -> float:
        if isinstance(minutes, bool) or not isinstance(minutes, (int, float)) or not math.isfinite(minutes):
            raise ValueError("minutes must be a number")
        if not MIN_MINUTES <= minutes <= MAX_MINUTES:
            raise ValueError(f"minutes must be between {MIN_MINUTES} and {MAX_MINUTES}, got {minutes:g}")
        return float(minutes)

    def set_do_not_disturb(self, minutes: float) -> float:
        """Stay quiet for `minutes`. Never shortens a longer one already set; returns the wall time it ends."""
        length = self._check_minutes(minutes) * 60.0
        with self._lock:
            until = max(self._wall() + length, self._active(self._dnd_until) or 0.0)
            self._dnd_until = until
        self._save(DND_KEY, until)
        log.info("do not disturb until %s", datetime.fromtimestamp(until).isoformat(timespec="minutes"))
        return until

    def clear_do_not_disturb(self) -> bool:
        with self._lock:
            was = self._active(self._dnd_until) is not None
            self._dnd_until = None
        self._save(DND_KEY, None)
        return was

    def set_busy(self, minutes: float) -> float:
        """The user is busy: Laika does not start anything, but the user can still talk to her."""
        length = self._check_minutes(minutes) * 60.0
        with self._lock:
            until = max(self._wall() + length, self._active(self._busy_until) or 0.0)
            self._busy_until = until
        self._save(BUSY_KEY, until)
        return until

    def clear_busy(self) -> bool:
        with self._lock:
            was = self._active(self._busy_until) is not None
            self._busy_until = None
        self._save(BUSY_KEY, None)
        return was

    @property
    def do_not_disturb_until(self) -> float | None:
        with self._lock:
            return self._active(self._dnd_until)

    @property
    def busy_until(self) -> float | None:
        with self._lock:
            return self._active(self._busy_until)

    def _active(self, until: float | None) -> float | None:
        """`until` while it is still in the future (caller holds the lock or only reads)."""
        return until if until is not None and until > self._wall() else None

    def _save(self, key: str, value: float | None) -> None:
        if self.store is not None:
            self.store.kv_put(NAMESPACE, key, value)

    def _load(self) -> None:
        if self.store is None:
            return
        for key, attribute in ((DND_KEY, "_dnd_until"), (BUSY_KEY, "_busy_until")):
            value = self.store.kv_get(NAMESPACE, key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                setattr(self, attribute, float(value))

    # --- the decision ------------------------------------------------------------

    def state(self, context: Context | None = None) -> Availability:
        context = context or self._context()
        with self._lock:
            return self._state(context, self._clock())

    def decide(self, context: Context | None = None) -> Decision:
        """May Laika start an interaction now? Reads only; call `record_attempt()` when she does."""
        context = context or self._context()
        now = self._clock()
        with self._lock:
            state = self._state(context, now)
            decision = self._rule(state, context, now)
        self._log(decision)
        return decision

    def _context(self) -> Context:
        try:
            questions = int(self._pending_questions()) if self._pending_questions else 0
        except Exception:  # noqa: BLE001 - a broken provider means "nothing to say", never a crash
            questions = 0
        try:
            busy = bool(self._body_busy()) if self._body_busy else False
        except Exception:  # noqa: BLE001 - unknown is treated as busy: the safe side
            busy = True
        return Context(pending_questions=questions, conversation_active=busy)

    def _in_quiet_hours(self) -> bool:
        if self._quiet is None:
            return False
        moment = datetime.fromtimestamp(self._wall())
        minute = moment.hour * 60 + moment.minute
        start, end = self._quiet
        if start < end:
            return start <= minute < end
        return minute >= start or minute < end

    def _conversation_running(self, now: float) -> bool:
        since = self._conversation_since
        return since is not None and now - since < MAX_CONVERSATION_S

    def _state(self, context: Context, now: float) -> Availability:
        if self._in_quiet_hours():
            return Availability.SLEEPING
        if self._active(self._dnd_until) is not None:
            return Availability.DO_NOT_DISTURB
        if (
            context.conversation_active
            or self._conversation_running(now)
            or self._active(self._busy_until) is not None
        ):
            return Availability.BUSY
        presence = self.presence
        if presence is None:
            return Availability.UNKNOWN
        present, since, last_seen = presence.present, presence.since, presence.last_seen
        if present and since is not None:
            return Availability.AVAILABLE if now - since >= self.tuning.present_s else Availability.UNKNOWN
        reference = last_seen if last_seen is not None else self._created
        if now - reference >= self.config.away_after_s:
            return Availability.AWAY
        return Availability.UNKNOWN

    def _rule(self, state: Availability, context: Context, now: float) -> Decision:
        if not self.config.enabled:
            return Decision(False, "disabled", state)
        if state is not Availability.AVAILABLE:
            return Decision(False, state.value, state)
        last = max((t for _, t in self._recent), default=None)
        if last is not None and now - last < self.tuning.cooldown_s:
            return Decision(False, "cooldown", state)
        interesting = any(
            total >= self.threshold and now - t <= self.config.interest_window_s
            for t, total in self._interest
        )
        has_something = interesting or context.pending_questions > 0 or context.encounter
        if not has_something and not self._battery_needs_saying():
            return Decision(False, "nothing_to_say", state)
        recent_attempts = sum(1 for t in self._attempts if now - t < HOUR_S)
        if recent_attempts >= self.tuning.max_per_hour:
            return Decision(False, "rate_limit", state)
        return Decision(True, REASON_ALLOWED, state)

    def _battery_needs_saying(self) -> bool:
        try:
            return bool(self._battery_low()) if self._battery_low else False
        except Exception:  # noqa: BLE001 - a broken provider means "nothing to say"
            return False

    def _log(self, decision: Decision) -> None:
        log.debug("interaction policy: %s (%s, %s)", decision.allowed, decision.reason, decision.state.value)
        with self._lock:
            changed = decision.reason != self._last_logged
            self._last_logged = decision.reason
        if changed:
            log.info("interaction policy: %s (%s)", "allowed" if decision.allowed else "refused", decision.reason)
