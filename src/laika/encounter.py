"""Meeting people while she is out and about (#49).

A person appearing while an autonomous activity holds the body is an encounter. `ProactiveInteraction`
hands every arrival to `Encounters.on_person_detected`, which takes it only when:

* the holder is EXPLORATION, or CURIOSITY (investigating, looking out of a window): the activities that
  are "out and about". An idle Laika, a conversation, a user command, a search or a follow-up are not:
  the arrival is handled as before (the recheck loop and the interaction policy), and an encounter
  never pre-empts any of them;
* the policy is not SLEEPING (quiet hours), DO_NOT_DISTURB or BUSY: then nothing happens at all, not
  even a pause or a look, and the autonomy pause refuses the body anyway.

Then one thread meets the person:

1. take the body. EXPLORATION is pre-empted at SOCIAL priority (resumable: it waits and carries on or
   ends by its own time afterwards; the arbiter stops the wheels). A CURIOSITY activity is short and more
   important than SOCIAL, so it is never pre-empted: the encounter waits for it to finish (up to a
   minute, while the person is still there);
2. turn the head to where the person was seen;
3. wait until the policy considers the person settled (at most ten seconds: a longer settle time means
   she asks at once and the policy says no);
4. ask the policy, with `Context(encounter=True)`: for an encounter a short hello is something to say.
   Cooldown, settle time, DND, quiet hours, busy, the hourly limit and the pause all still apply;
5. allowed: maybe one short approach move, then a proactive conversation (#40) that takes over the
   lease this encounter holds. It opens with a pending question, else something she noticed (a low
   battery, a recent interesting change), else a short hello. Not allowed: no words, at most a look and
   a nod (not when sleeping, in do-not-disturb or busy), then the lease goes back and exploration resumes.

She never chases a person: if they are gone during the approach, the move stops and that is the end.
The approach is her own driving: the autonomous switch, the pause, the budget, the no-go rooms and the
battery all apply (`self_directed=True`), and she never gets closer than `stop_cm`. Who the person is
never matters: she cannot tell people apart. Docs: docs/behaviours.md, "Meeting people".
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from contextlib import ExitStack
from typing import TYPE_CHECKING, Callable

from laika.behaviour import Lease, Priority
from laika.commands import Direction, InvalidCommand, LookCommand, MoveCommand, ResultStatus
from laika.config import APPROACH_MOVE_CM, EncounterConfig, InteractionConfig
from laika.events import Event, EventBus, EventType
from laika.explore import NAME as EXPLORE_NAME
from laika.interaction_policy import Availability, Context, Decision, InteractionPolicy
from laika.investigate import NAME as INVESTIGATE_NAME, _heartbeat as keep_alive
from laika.outside import NAME as WINDOW_NAME
from laika.proactive import REASON as QUESTION_REASON, ProactiveRequest, ProactiveTarget

if TYPE_CHECKING:
    from laika.behaviour import Arbiter
    from laika.investigate import PendingQuestions
    from laika.memory import Memory
    from laika.presence import PresenceTracker
    from laika.robot import Robot
    from laika.safety import SafetyController

log = logging.getLogger(__name__)

NAME = "encounter"
POLL_S = 0.05
MAX_SETTLE_WAIT_S = 10.0  # waiting longer than this for the settle time is not worth holding her up
DEFER_MAX_S = 60.0  # how long she waits for a short curiosity activity to finish
TAKE_WAIT_S = 35.0  # the conversation takes a handed-over request within 30 s or drops it (PROACTIVE_MAX_AGE_S)
CHAT_MAX_S = 3600.0
GESTURE_HOLD_S = 1.0
APPROACH_SPEED = 20  # one short move; the safety layer clamps both
APPROACH_MS = 800
AHEAD_DEG = 20.0  # only a person roughly ahead is driven towards
MAX_INTERESTS = 8

# The states in which an encounter does nothing at all.
SILENT_STATES = frozenset({Availability.SLEEPING, Availability.DO_NOT_DISTURB, Availability.BUSY})
# Refusals for which a look and a nod are fine (silent and harmless): she is just not going to talk now.
GESTURE_REASONS = frozenset({"cooldown", "nothing_to_say", "rate_limit", "unknown"})

# lease name -> (what the episode says she was doing, what she tells the model and says if it fails)
ACTIVITIES = {
    EXPLORE_NAME: ("exploring", "stavo facendo un giro"),
    INVESTIGATE_NAME: ("investigating something", "stavo guardando una cosa"),
    WINDOW_NAME: ("looking out of the window", "stavo guardando fuori dalla finestra"),
    # The idle selector's own leases (#47).
    "idle-rest": ("resting", "stavo riposando"),
    "idle-follow-up": ("looking for something", "stavo cercando una cosa"),
    "idle-check-home": ("checking on who was home", "stavo dando un'occhiata in casa"),
}
DEFAULT_ACTIVITY = ("out and about", "stavo facendo un giro")
LOW_BATTERY_NOTE = "ti si sta scaricando la batteria e presto dovrai riposare e ricaricarti"
MAX_NOTE_CHARS = 120


def greeting_line(doing: str) -> str:
    """The Italian hello she says when the model cannot word it."""
    return f"Oh, ciao. {doing[:1].upper()}{doing[1:]}."


class Encounters:
    def __init__(
        self,
        config: EncounterConfig,
        interaction: InteractionConfig,
        policy: InteractionPolicy,
        presence: PresenceTracker,
        arbiter: Arbiter,
        safety: SafetyController,
        conversation: ProactiveTarget,
        questions: PendingQuestions,
        events: EventBus,
        robot: Robot,
        *,
        actions: object | None = None,
        memory: Memory | None = None,
        battery_low: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.interaction = interaction
        self.policy = policy
        self.presence = presence
        self.arbiter = arbiter
        self.safety = safety
        self.conversation = conversation
        self.questions = questions
        self.events = events
        self.robot = robot
        self.actions = actions
        self.memory = memory
        self._battery_low = battery_low
        self._clock = clock
        self._lock = threading.Lock()
        self._handling = False
        self._thread: threading.Thread | None = None
        self._closed = threading.Event()
        self._interests: deque[tuple[float, float, str]] = deque(maxlen=MAX_INTERESTS)
        self._unsubscribe: Callable[[], None] | None = None
        self.count = 0  # encounters started since startup (logs and tests)

    # --- lifecycle ---------------------------------------------------------------

    def start(self) -> None:
        if self._unsubscribe is None:
            self._unsubscribe = self.events.subscribe(self._on_event, EventType.INTERESTING_CHANGE_DETECTED)

    def close(self, timeout_s: float = 3.0) -> None:
        self._closed.set()
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout_s)

    @property
    def handling(self) -> bool:
        """True while she is meeting someone: the recheck loop leaves that person to the encounter."""
        return self._handling

    def _on_event(self, event: Event) -> None:
        """Remember what she noticed lately: it can be the thing she says to someone she meets."""
        data = event.data
        try:
            total = float(data.get("total", 0.0))
        except (TypeError, ValueError):
            return
        reasons = [str(r) for r in (data.get("reasons") or [])][:2]
        if reasons:
            where = f" ({data['location']})" if data.get("location") else ""
            self._interests.append((self._clock(), total, ("; ".join(reasons) + where)[:MAX_NOTE_CHARS]))

    # --- the arrival -------------------------------------------------------------

    def on_person_detected(self, data: dict) -> bool:
        """A person arrived. True when this is an encounter and the thread has taken it over.

        Quick and non-blocking: it runs on the event thread. Anything that is not clearly "out and about"
        is left to the normal path.
        """
        if not self.config.enabled or self._closed.is_set() or not self.interaction.enabled:
            return False
        holder = self.arbiter.holder
        if holder is None or holder.priority not in (Priority.EXPLORATION, Priority.CURIOSITY):
            return False
        if self.safety.paused:
            return False
        if self.policy.state(Context(conversation_active=self._body_busy())) in SILENT_STATES:
            return False
        with self._lock:
            if self._handling:
                return False
            self._handling = True
            self.count += 1
            thread = threading.Thread(
                target=self._run, args=(dict(data), holder.name), name="laika-encounter", daemon=True
            )
            self._thread = thread
        thread.start()
        return True

    def _body_busy(self) -> bool:
        if getattr(self.conversation, "awake", False):
            return True
        holder = self.arbiter.holder
        return holder is not None and holder.priority <= Priority.CONVERSATION

    # --- one encounter -------------------------------------------------------------

    def _run(self, data: dict, activity: str) -> None:
        label, doing = ACTIVITIES.get(activity, DEFAULT_ACTIVITY)
        outcome, reason = "quiet", "error"
        lease: Lease | None = None
        handed = False
        beat = ExitStack()
        try:
            print(f"[laika] someone appeared while she was {label}")
            lease, reason = self._take_body()
            if lease is None:
                return
            beat.enter_context(keep_alive(lease, self.arbiter.lease_timeout_s))
            self._look_at_person(data, lease)
            if reason := self._settle(lease):
                return
            if self.safety.paused:
                reason = "paused"
                return
            decision = self._decide()
            if not decision.allowed:
                reason = decision.reason
                self._gesture(lease, decision)
                return
            moved = self._approach(lease)
            if not lease.active:
                reason = self._gone()
                return
            if not self.presence.present:
                reason = "left"  # never chase: the move was stopped, and that is all
                return
            if moved:
                decision = self._decide()  # she is somewhere else now: ask again
                if not decision.allowed:
                    reason = decision.reason
                    self._gesture(lease, decision)
                    return
            request = self._request(data, lease, doing)
            self.policy.record_attempt()
            if not self.conversation.request_proactive(request):
                reason = "busy"
                return
            handed = True
            outcome, reason = self._follow(request, lease, beat)
        except Exception:  # noqa: BLE001 - an encounter must end cleanly and give the body back
            log.exception("encounter failed")
            reason = "error"
        finally:
            beat.close()
            if lease is not None and not handed:
                self._look_forward(lease)
                lease.release()  # whatever she interrupted gets the body back
            self._record(label, outcome, reason, data)
            with self._lock:
                self._handling = False

    def _take_body(self) -> tuple[Lease | None, str]:
        """The body at SOCIAL priority: EXPLORATION is pre-empted; anything more important is waited for."""
        give_up = time.monotonic() + DEFER_MAX_S
        while not self._closed.is_set():
            if not self.presence.present:
                return None, "left"
            if self.safety.paused:
                return None, "paused"
            holder = self.arbiter.holder
            if holder is not None and (holder.priority <= Priority.CONVERSATION or holder.priority == Priority.SOCIAL):
                return None, "busy"  # a chat, a command or another conversation: never pre-empted
            if holder is None or holder.priority > Priority.CURIOSITY:
                lease = self.arbiter.request(NAME, Priority.SOCIAL, resumable=False)
                if lease is not None:
                    return lease, ""
            if time.monotonic() >= give_up:
                return None, "busy"
            self._closed.wait(POLL_S)
        return None, "closed"

    def _look_at_person(self, data: dict, lease: Lease) -> None:
        pan, tilt = self._direction(data)
        try:
            command = LookCommand(pan=pan, tilt=tilt)
        except InvalidCommand:
            return
        if lease.active:
            self.robot.look(command.pan, command.tilt)

    def _direction(self, data: dict) -> tuple[float, float]:
        direction = getattr(self.presence, "direction", None)
        if direction is not None:
            return direction
        try:
            return float(data.get("pan", 0.0) or 0.0), float(data.get("tilt", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0, 0.0

    def _settle(self, lease: Lease) -> str:
        """Wait (a little) for the policy's settle time. "" when it is time to ask, else why she stops."""
        since = self.presence.since
        if since is None:
            return "left"
        ready = since + self.policy.tuning.present_s
        if ready - self._clock() > MAX_SETTLE_WAIT_S:
            return ""  # too long to wait: ask now, the policy says "unknown" and she stays quiet
        give_up = time.monotonic() + MAX_SETTLE_WAIT_S
        while self._clock() < ready and time.monotonic() < give_up:
            if self._closed.is_set() or not lease.active:
                return self._gone()
            if not self.presence.present:
                return "left"
            self._closed.wait(POLL_S)
        if self._closed.is_set() or not lease.active:
            return self._gone()
        return "" if self.presence.present else "left"

    def _gone(self) -> str:
        """Why the body is no longer hers: the user paused her, or something more important took it."""
        return "paused" if self.safety.paused else "preempted"

    def _decide(self) -> Decision:
        return self.policy.decide(
            Context(
                pending_questions=len(self.questions.pending()),
                conversation_active=self._body_busy(),
                encounter=self.config.greet_when_nothing_to_say,
            )
        )

    # --- no words ------------------------------------------------------------------

    def _gesture(self, lease: Lease, decision: Decision) -> None:
        """A look and a nod, silently, when the refusal is only "not now". Never when she must be quiet."""
        if not self.config.gesture_when_quiet or decision.reason not in GESTURE_REASONS:
            return
        end = time.monotonic() + GESTURE_HOLD_S
        while time.monotonic() < end and lease.active and not self._closed.is_set():
            self._closed.wait(POLL_S)
        play = getattr(self.actions, "play", None)
        if play is None or not lease.active:
            return
        try:
            play("nod")
            self.actions.wait_done(timeout=5)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 - a gesture is a nicety
            log.exception("encounter gesture failed")

    def _look_forward(self, lease: Lease) -> None:
        if lease.active:
            try:
                self.robot.look(0, 0)
            except Exception:  # noqa: BLE001
                log.exception("could not put the head back")

    # --- coming closer -------------------------------------------------------------

    def _approach(self, lease: Lease) -> bool:
        """One short move towards a person who is roughly ahead and far. True when the wheels moved."""
        config, safety = self.config, self.safety
        if not config.approach or not (safety.enabled and safety.allow_autonomous):
            return False
        pan, _tilt = self._direction({})
        if abs(pan) > AHEAD_DEG or not lease.active or not self.presence.present:
            return False
        distance = self.robot.distance_cm()
        if distance is None or distance < config.approach_distance_cm:
            return False
        if distance - APPROACH_MOVE_CM < config.stop_cm:
            return False  # one move would take her closer than she may
        done = threading.Event()

        def watch() -> None:
            while not done.wait(0.03):
                if not self.presence.present or not lease.active:
                    safety.stop("stopped")  # they walked away: no following
                    return

        watcher = threading.Thread(target=watch, name="laika-encounter-watch", daemon=True)
        watcher.start()
        try:
            command = MoveCommand(direction=Direction.FORWARD, speed=APPROACH_SPEED, duration_ms=APPROACH_MS)
            result = safety.execute(command, lease=lease, self_directed=True)
            lease.kick()
        finally:
            done.set()
            watcher.join(1.0)
        moved = result is not None and result.result is ResultStatus.SUCCESS
        print(f"[laika] encounter: came closer ({'moved' if moved else (result.reason if result else 'no result')})")
        return moved

    # --- talking -------------------------------------------------------------------

    def _request(self, data: dict, lease: Lease, doing: str) -> ProactiveRequest:
        """What she opens with: a pending question, else something she noticed, else a hello."""
        pan, tilt = self._direction(data)
        room = self.presence.room or data.get("room")
        common = dict(trigger="encounter", pan=pan, tilt=tilt, encounter=True, lease=lease)
        pending = self.questions.pending()
        if pending:
            question = pending[0]
            return ProactiveRequest(
                question.id, question.text, QUESTION_REASON, room=room or question.room, **common
            )
        hello = greeting_line(doing)
        if (noticed := self._noticed()) is not None:
            return ProactiveRequest(
                "", hello, "encounter_noticed", room=room, kind="noticed", detail=noticed, **common
            )
        return ProactiveRequest(
            "", hello, "encounter_greeting", room=room, kind="greeting", detail=doing, **common
        )

    def _noticed(self) -> str | None:
        """Something she can say she noticed: a low battery, or an interesting change that is still recent."""
        try:
            if self._battery_low is not None and self._battery_low():
                return LOW_BATTERY_NOTE
        except Exception:  # noqa: BLE001
            pass
        now = self._clock()
        for when, total, text in reversed(self._interests):
            if total >= self.policy.threshold and now - when <= self.interaction.interest_window_s:
                return text
        return None

    def _follow(self, request: ProactiveRequest, lease: Lease, beat: ExitStack) -> tuple[str, str]:
        """The conversation has the request: keep the lease alive until it takes it, then wait for the end."""
        give_up = time.monotonic() + TAKE_WAIT_S
        while not request.taken.wait(POLL_S):
            if self._closed.is_set() or time.monotonic() >= give_up:
                lease.release()
                return "quiet", "dropped"
        beat.close()  # the conversation keeps it alive from here
        end = time.monotonic() + CHAT_MAX_S
        while not request.finished.wait(POLL_S):
            if self._closed.is_set() or time.monotonic() >= end:
                break
        return ("talked", "") if request.spoke.is_set() else ("quiet", "no_words")

    # --- remembering ----------------------------------------------------------------

    def _record(self, label: str, outcome: str, reason: str, data: dict) -> None:
        """A "met someone while ..." episode. The subjects carry the outcome, so the report can count them."""
        why = "" if outcome == "talked" else f" ({reason.replace('_', ' ')})"
        text = "talked" if outcome == "talked" else f"stayed quiet{why}"
        print(f"[laika] encounter ended: {text}")
        if self.memory is None:
            return
        subjects = ["encounter", outcome] + ([reason] if reason else [])
        try:
            self.memory.episodes.record(
                "encounter", f"Met someone while {label}: {text}", subjects, self.presence.room or data.get("room")
            )
        except Exception:  # noqa: BLE001 - a note, never a reason to fail
            log.exception("could not record the encounter")
