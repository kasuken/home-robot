"""What Laika does when nothing happens (#47): a deterministic selector, not the model.

After `[idle] idle_after_s` with nobody talking to her and no behaviour holding the body, `IdleSelector`
chooses one activity, runs it (bounded by `max_activity_s`) and waits `between_activities_s` before it
chooses again. The five activities are small adapters with `available(context)`, `utility(context)` and
`run(...)`: explore (driving where it is allowed, head only otherwise: the Explorer decides which), look
out of a window, follow up (point at / look for an object from a pending question or an unresolved
change, never asking anything: asking is the interaction policy's), check on who is home (a head scan for
a person, a count only) and rest (stay still, an occasional slow look around, always possible).

The formula, one line per activity and every factor documented here and logged at debug level:

    utility = weight * time_of_day * recency * personality * need * (mode, budget, presence, battery)

* weight: `[idle] <activity>_weight` (explore 1.0, look_outside 0.6, follow_up 0.8, check_home 0.4, rest 0.5);
* time of day (local hour): night 22-5, early morning 5-8, day 8-18, evening 18-22. Exploring, looking
  out and following up are at their best in the day (x1.1 to x1.2) and nearly off at night (x0.1 to x0.3);
  rest is x2.5 at night, x1.5 early in the morning, x0.8 in the day; checking on the house is best in the
  evening (x1.3). Quiet hours, do not disturb and a busy user (the interaction policy's SLEEPING,
  DO_NOT_DISTURB, BUSY) leave rest as the only choice;
* recency (variety): 1.0 for an activity that has not run yet, else 0.25 right after it ran, recovering
  in a straight line to 1.0 over an hour. An activity is also never chosen more than `max_repeats` times in
  a row (unless nothing else is possible);
* personality: curiosity scales explore, look_outside and follow_up from x0.6 (0) to x1.4 (100);
  initiative scales check_home the same way. Rest is not scaled;
* need: follow_up is x(1 + 0.5 per pending question + 0.4 per recent unresolved change), at most three of
  each; it is not possible without one of them;
* explore: x0.5 when it would only look around (autonomous driving is off, or she must not drive now),
  and, while driving, x(minutes of driving budget left / 2), between 0.3 and 1, when a driving budget is set;
* presence: when someone is near she does not check on the house (they are there), follow-ups stay
  head-only, and rest is x2; exploring still drives (slowly, behind the safety checks), so she meets people;
* battery: LOW or UNKNOWN take away explore (no driving activities) and make rest x2.5; CRITICAL leaves rest
  only (she stays still, head included; with the arbiter refusing leases she rests without one).

An activity whose time-of-day factor is below `NOT_NOW` (explore, look out, follow up and check on the
house at night) is not an option then, so at night only rest is left. The highest utility wins; ties are broken by a generator seeded from `[idle] seed`, so the same inputs and the same seed give the
same choice. The pause ("resta ferma") stops all of it, rest included.

Each activity runs under its own lease (the explorer, the window watcher and the searcher take theirs; rest
takes an EXPLORATION lease and never moves the wheels) so the wake word, an encounter, safety and the pause
pre-empt it. The wake word also ends the activity for good and the selector waits `idle_after_s` again.
Every run becomes an `idle` episode ("Idle: explored for 2 minutes"), kept in memory and returned by
`what_changed`; a run of rests is added up and recorded as one episode.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping

from laika.battery import BatteryState
from laika.behaviour import Arbiter, Lease, Priority
from laika.config import IDLE_ACTIVITIES, IdleConfig
from laika.events import Event, EventBus, EventType
from laika.interaction_policy import Availability
from laika.investigate import _heartbeat
from laika.personality import Personality

if TYPE_CHECKING:
    from laika.changes import ChangeTracker
    from laika.explore import Explorer
    from laika.investigate import PendingQuestions
    from laika.memory import Memory
    from laika.outside import WindowWatcher
    from laika.presence import PresenceTracker
    from laika.robot import Robot
    from laika.safety import SafetyController
    from laika.search import Searcher
    from laika.world_model import WorldModel
    from laika.interaction_policy import InteractionPolicy

log = logging.getLogger(__name__)

NAME = "idle"
EXPLORE, LOOK_OUTSIDE, FOLLOW_UP, CHECK_HOME, REST = IDLE_ACTIVITIES
REST_LEASE = "idle-rest"
FOLLOW_UP_LEASE = "idle-follow-up"
CHECK_HOME_LEASE = "idle-check-home"

POLL_S = 1.0  # how often the selector looks (the thread), and how often a running activity checks it may go on
ACTIVITY_POLL_S = 0.2
NOT_NOW = 0.35  # a time-of-day factor below this takes the activity away (rest is always possible)
RECENT_FLOOR = 0.25  # the recency factor right after an activity ran ...
RECOVERY_S = 3600.0  # ... and how long it takes to be 1.0 again
CHANGE_WINDOW_S = 3600.0  # a change she noticed is "unresolved" for this long
FOLLOW_UP_COOLDOWN_S = 1800.0  # the same object is not followed up again within this
MAX_TARGETS = 3
BUDGET_COMFORT_S = 120.0  # this much driving budget left (two minutes) is plenty for one activity
MIN_BUDGET_FACTOR = 0.3
HEAD_ONLY_FACTOR = 0.5
LOW_BATTERY_REST = 2.5
PRESENT_REST = 2.0
REST_LOOK_EVERY_S = 180.0  # a rest looks around slowly this often
REST_LOOK_PANS = (-30.0, 30.0)
REST_LOOK_HOLD_S = 2.0
REST_FLUSH_S = 900.0  # a run of rests is recorded as one episode, at least this often
HISTORY = 64
JOIN_TIMEOUT_S = 5.0
# Her own words for the episodes' subjects: "idle", the activity, how it ended, how long (seconds).
EPISODE_KIND = "idle"

PERIODS = ("night", "early_morning", "day", "evening")
# Per activity: night, early morning, day, evening.
TIME_FACTORS: dict[str, tuple[float, float, float, float]] = {
    EXPLORE: (0.2, 0.5, 1.2, 0.8),
    LOOK_OUTSIDE: (0.1, 0.8, 1.2, 0.6),
    FOLLOW_UP: (0.3, 0.6, 1.1, 0.9),
    CHECK_HOME: (0.2, 0.8, 1.0, 1.3),
    REST: (2.5, 1.5, 0.8, 1.2),
}
# The states of the interaction policy in which she only rests.
QUIET_STATES = frozenset({Availability.SLEEPING, Availability.DO_NOT_DISTURB, Availability.BUSY})
# How an activity ended: "done", "interrupted" (something more important took the body), "refused"
# (could not start) or "failed" (started and did nothing).
RECORDED = ("done", "interrupted")


# --- the formula ---------------------------------------------------------------------------------


def period_of(hour: float) -> str:
    if hour >= 22 or hour < 5:
        return "night"
    if hour < 8:
        return "early_morning"
    return "day" if hour < 18 else "evening"


def time_factor(activity: str, period: str) -> float:
    return TIME_FACTORS[activity][PERIODS.index(period)]


def recency_factor(since_s: float | None) -> float:
    """1.0 when it never ran; 0.25 right after, back to 1.0 over an hour."""
    if since_s is None:
        return 1.0
    return RECENT_FLOOR + (1.0 - RECENT_FLOOR) * min(1.0, max(0.0, since_s) / RECOVERY_S)


def scale_factor(value: int) -> float:
    """0.6 at 0, 1.0 at 50, 1.4 at 100."""
    return 0.6 + 0.8 * value / 100


def personality_factor(activity: str, personality: Personality) -> float:
    if activity in (EXPLORE, LOOK_OUTSIDE, FOLLOW_UP):
        return scale_factor(personality.curiosity)
    if activity == CHECK_HOME:
        return scale_factor(personality.initiative)
    return 1.0


def need_factor(questions: int, changes: int) -> float:
    return 1.0 + 0.5 * min(questions, MAX_TARGETS) + 0.4 * min(changes, MAX_TARGETS)


def budget_factor(left_s: float | None) -> float:
    """1.0 without a driving budget; else the minutes of driving left over two, between 0.3 and 1."""
    if left_s is None:
        return 1.0
    return max(MIN_BUDGET_FACTOR, min(1.0, left_s / BUDGET_COMFORT_S))


@dataclass(frozen=True)
class IdleContext:
    """Everything the utility formula reads, taken once so one choice is made from one picture."""

    now: float
    hour: float
    period: str
    state: Availability | None = None
    battery: BatteryState = BatteryState.OK
    driving: bool = False  # a driving activity would really drive now
    present: bool = False
    budget_left_s: float | None = None
    personality: Personality = field(default_factory=Personality)
    question_objects: tuple[str, ...] = ()
    change_objects: tuple[str, ...] = ()
    windows_due: int = 0
    since: Mapping[str, float | None] = field(default_factory=dict)
    streak: tuple[str | None, int] = (None, 0)
    max_repeats: int = 2

    @property
    def quiet(self) -> bool:
        return self.state in QUIET_STATES

    @property
    def rest_only(self) -> bool:
        return self.quiet or self.battery is BatteryState.CRITICAL

    @property
    def battery_low(self) -> bool:
        return self.battery in (BatteryState.LOW, BatteryState.UNKNOWN)


@dataclass
class Outcome:
    status: str  # done, interrupted, refused, failed
    seconds: float = 0.0
    summary: str = ""
    reason: str = ""


@dataclass
class Run:
    """What an activity may use while it runs: the time limit, whether to stop, and the clock."""

    max_s: float
    stop: Callable[[], bool]
    clock: Callable[[], float]
    sleep: Callable[[float], None]


def duration_phrase(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        count = max(1, round(seconds / 5) * 5) if seconds >= 5 else max(1, round(seconds))
        return f"{count} second{'s' if count != 1 else ''}"
    if seconds < 90 * 60:
        count = round(seconds / 60)
        return f"{count} minute{'s' if count != 1 else ''}"
    count = round(seconds / 3600)
    return f"{count} hour{'s' if count != 1 else ''}"


# --- the activities ------------------------------------------------------------------------------


class Activity:
    name = ""

    def __init__(self, config: IdleConfig) -> None:
        self.enabled = bool(getattr(config, self.name))
        self.weight = float(getattr(config, f"{self.name}_weight"))

    def possible(self, ctx: IdleContext) -> bool:
        """The activity's own conditions (what it needs and what stops it)."""
        return True

    def why_not(self, ctx: IdleContext) -> str:
        """Why `available` is False, in words for the console."""
        if not self.enabled or self.weight <= 0:
            return "switched off in [idle]"
        if self.name != REST and ctx.rest_only:
            return "quiet time (quiet hours, do not disturb, busy or a critical battery)"
        if self.name != REST and time_factor(self.name, ctx.period) < NOT_NOW:
            return f"not at this time of day ({ctx.period})"
        return "not possible now"

    def available(self, ctx: IdleContext) -> bool:
        if not self.enabled or self.weight <= 0:
            return False
        if self.name != REST and (ctx.rest_only or time_factor(self.name, ctx.period) < NOT_NOW):
            return False
        return self.possible(ctx)

    def factors(self, ctx: IdleContext) -> dict[str, float]:
        return {
            "time": time_factor(self.name, ctx.period),
            "recency": recency_factor(ctx.since.get(self.name)),
            "personality": personality_factor(self.name, ctx.personality),
        }

    def utility(self, ctx: IdleContext) -> float:
        """0.0 when it is not available; else the weight times every factor."""
        if not self.available(ctx):
            return 0.0
        return self.weight * math.prod(self.factors(ctx).values())

    def run(self, ctx: IdleContext, run: Run) -> Outcome:
        raise NotImplementedError


@contextmanager
def _leased(arbiter: Arbiter, name: str, priority: Priority) -> Iterator[Lease | None]:
    """A lease for the length of the block, kept alive; None when the arbiter refuses."""
    lease = arbiter.request(name, priority, resumable=False)
    if lease is None:
        yield None
        return
    try:
        with _heartbeat(lease, arbiter.lease_timeout_s):
            yield lease
    finally:
        lease.release()


class ExploreActivity(Activity):
    name = EXPLORE

    def __init__(self, config: IdleConfig, explorer: Explorer | None) -> None:
        super().__init__(config)
        self.explorer = explorer

    def possible(self, ctx: IdleContext) -> bool:
        explorer = self.explorer
        if explorer is None or not explorer.config.enabled:
            return False
        # Someone near does not stop her: she drives slowly, safety stops her short of legs, and meeting
        # people while she explores is the point (the encounter talks to them).
        return ctx.battery is BatteryState.OK  # no driving activities when the battery is low or not known

    def why_not(self, ctx: IdleContext) -> str:
        explorer = self.explorer
        if explorer is None or not explorer.config.enabled:
            return "[explore] enabled is false"
        if ctx.battery is not BatteryState.OK:
            return f"the battery is {ctx.battery.value}"
        return super().why_not(ctx)

    def factors(self, ctx: IdleContext) -> dict[str, float]:
        factors = super().factors(ctx)
        if ctx.driving:
            factors["budget"] = budget_factor(ctx.budget_left_s)
        else:
            factors["head_only"] = HEAD_ONLY_FACTOR
        return factors

    def run(self, ctx: IdleContext, run: Run) -> Outcome:
        explorer = self.explorer
        assert explorer is not None
        if not explorer.start(duration_s=run.max_s):
            return Outcome("refused", reason=explorer.refusal)
        began = run.clock()
        interrupted = False
        while explorer.active:
            if run.stop():
                interrupted = True
                break
            if run.clock() - began >= run.max_s:
                break
            run.sleep(ACTIVITY_POLL_S)
        explorer.stop()  # idempotent: ends the session and gives the body back
        seconds = run.clock() - began
        if not interrupted and not (explorer.steps or explorer.scans):
            return Outcome("failed", seconds, reason=explorer.end_reason or "nothing happened")
        how = "explored" if explorer.mode == "driving" else "explored with the head only"
        return Outcome("interrupted" if interrupted else "done", seconds, f"Idle: {how} for {duration_phrase(seconds)}")


class LookOutsideActivity(Activity):
    name = LOOK_OUTSIDE

    def __init__(self, config: IdleConfig, watcher: WindowWatcher | None) -> None:
        super().__init__(config)
        self.watcher = watcher

    def possible(self, ctx: IdleContext) -> bool:
        return self.watcher is not None and ctx.windows_due > 0 and ctx.battery is not BatteryState.CRITICAL

    def run(self, ctx: IdleContext, run: Run) -> Outcome:
        watcher = self.watcher
        assert watcher is not None
        due = watcher.due_windows()
        if not due:
            return Outcome("refused", reason="no window is due")
        window = due[0]
        began = run.clock()
        result = watcher.watch(window)  # its own CURIOSITY lease, a few seconds, the vision model's answer
        seconds = run.clock() - began
        where = f"the {window.room.replace('_', ' ')} window"
        if result.status in ("done", "not_seen"):
            return Outcome("done", seconds, f"Idle: looked out of {where}")
        if result.status == "aborted":
            return Outcome("interrupted", seconds, f"Idle: looked out of {where}")
        return Outcome("refused" if result.status == "refused" else "failed", seconds, reason=result.reason)


class FollowUpActivity(Activity):
    name = FOLLOW_UP

    def __init__(self, config: IdleConfig, searcher: Searcher | None, arbiter: Arbiter, search_max_s: float) -> None:
        super().__init__(config)
        self.searcher = searcher
        self.arbiter = arbiter
        self.search_max_s = search_max_s
        self.last_target: str | None = None

    def possible(self, ctx: IdleContext) -> bool:
        has_target = bool(ctx.question_objects or ctx.change_objects)
        return self.searcher is not None and has_target and ctx.battery is not BatteryState.CRITICAL

    def factors(self, ctx: IdleContext) -> dict[str, float]:
        factors = super().factors(ctx)
        factors["need"] = need_factor(len(ctx.question_objects), len(ctx.change_objects))
        return factors

    def target(self, ctx: IdleContext) -> str | None:
        return next(iter(ctx.question_objects + ctx.change_objects), None)

    def run(self, ctx: IdleContext, run: Run) -> Outcome:
        searcher = self.searcher
        target = self.target(ctx)
        assert searcher is not None and target is not None
        self.last_target = target
        # She only drives where an activity may: not with someone near, and never when she should not.
        head_only = not ctx.driving or ctx.present or ctx.battery_low
        began = run.clock()
        with _leased(self.arbiter, FOLLOW_UP_LEASE, Priority.CURIOSITY) as lease:
            if lease is None:
                return Outcome("refused", reason="the body is busy, paused or the battery is empty")
            result = searcher.find_object(
                target,
                lease=lease,
                max_s=min(run.max_s, self.search_max_s),
                should_stop=run.stop,
                head_only=head_only,
            )
        seconds = run.clock() - began
        if not result.get("ok"):
            reason = str(result.get("reason") or result.get("error") or "")
            if reason in ("preempted", "cancelled") or run.stop():
                return Outcome("interrupted", seconds, f"Idle: looked for the {target}")
            return Outcome("refused" if reason == "busy" else "failed", seconds, reason=reason)
        ending = " and found it" if result.get("found") else ""
        return Outcome("done", seconds, f"Idle: looked for the {target}{ending}")


class CheckHomeActivity(Activity):
    name = CHECK_HOME

    def __init__(self, config: IdleConfig, searcher: Searcher | None, arbiter: Arbiter, search_max_s: float) -> None:
        super().__init__(config)
        self.searcher = searcher
        self.arbiter = arbiter
        self.search_max_s = search_max_s

    def possible(self, ctx: IdleContext) -> bool:
        return self.searcher is not None and not ctx.present and ctx.battery is not BatteryState.CRITICAL

    def run(self, ctx: IdleContext, run: Run) -> Outcome:
        searcher = self.searcher
        assert searcher is not None
        began = run.clock()
        with _leased(self.arbiter, CHECK_HOME_LEASE, Priority.CURIOSITY) as lease:
            if lease is None:
                return Outcome("refused", reason="the body is busy, paused or the battery is empty")
            # A look around the room where she is: the head only, and people are counted, never identified.
            result = searcher.find_person(
                lease=lease, max_s=min(run.max_s, self.search_max_s), should_stop=run.stop, head_only=True
            )
        seconds = run.clock() - began
        if not result.get("ok"):
            reason = str(result.get("reason") or result.get("error") or "")
            if reason in ("preempted", "cancelled") or run.stop():
                return Outcome("interrupted", seconds, "Idle: checked who was home")
            return Outcome("refused" if reason == "busy" else "failed", seconds, reason=reason)
        people = int(result.get("people") or 0) if result.get("found") else 0
        seen = "nobody seen" if people == 0 and not result.get("found") else (
            f"saw {max(1, people)} {'person' if max(1, people) == 1 else 'people'}"
        )
        return Outcome("done", seconds, f"Idle: checked who was home ({seen})")


class RestActivity(Activity):
    name = REST

    def __init__(self, config: IdleConfig, arbiter: Arbiter, robot: Robot | None, clock: Callable[[], float]) -> None:
        super().__init__(config)
        self.arbiter = arbiter
        self.robot = robot
        self._last_look = clock()

    def factors(self, ctx: IdleContext) -> dict[str, float]:
        factors = super().factors(ctx)
        factors["presence"] = PRESENT_REST if ctx.present else 1.0
        factors["battery"] = LOW_BATTERY_REST if ctx.battery_low else 1.0
        return factors

    def run(self, ctx: IdleContext, run: Run) -> Outcome:
        began = run.clock()
        lease = self.arbiter.request(REST_LEASE, Priority.EXPLORATION, resumable=False)
        if lease is None and ctx.battery is not BatteryState.CRITICAL:
            return Outcome("refused", reason="the body is busy or paused")
        try:
            # Without a lease (a critical battery: the arbiter refuses) she just stays still, head included.
            interrupted = self._rest(lease, run, began)
        finally:
            if lease is not None:
                lease.release()
        seconds = run.clock() - began
        return Outcome("interrupted" if interrupted else "done", seconds, f"Idle: rested for {duration_phrase(seconds)}")

    def _rest(self, lease: Lease | None, run: Run, began: float) -> bool:
        """Stay still until time is up. True when something more important ended it."""
        while run.clock() - began < run.max_s:
            if run.stop() or (lease is not None and not lease.active):
                return True
            if lease is not None:
                lease.kick()
                if run.clock() - self._last_look >= REST_LOOK_EVERY_S:
                    self._last_look = run.clock()
                    if self._look_around(lease, run):
                        return True
            run.sleep(ACTIVITY_POLL_S)
        return False

    def _look_around(self, lease: Lease, run: Run) -> bool:
        """A slow look to the left and to the right, then back. True when it was cut short."""
        robot = self.robot
        if robot is None:
            return False
        original = robot.head_pose()
        try:
            for pan in REST_LOOK_PANS:
                if run.stop() or not lease.active:
                    return True
                robot.look(pan, 0.0)
                lease.kick()
                if self._hold(lease, run):
                    return True
        except Exception:  # noqa: BLE001 - a head failure must not end her rest badly
            log.exception("could not look around while resting")
        finally:
            if lease.active:
                try:
                    robot.look(*original)
                except Exception:  # noqa: BLE001
                    log.exception("could not restore the head after resting")
        return False

    def _hold(self, lease: Lease, run: Run) -> bool:
        until = run.clock() + REST_LOOK_HOLD_S
        while run.clock() < until:
            if run.stop() or not lease.active:
                return True
            lease.kick()
            run.sleep(ACTIVITY_POLL_S)
        return False


# --- the selector --------------------------------------------------------------------------------


@dataclass(frozen=True)
class IdleRun:
    """One choice and what came of it (kept for logs and tests)."""

    activity: str
    scores: Mapping[str, float]
    outcome: Outcome
    summary: str = ""


class IdleSelector:
    def __init__(
        self,
        config: IdleConfig,
        *,
        arbiter: Arbiter,
        safety: SafetyController,
        explorer: Explorer | None = None,
        windows: WindowWatcher | None = None,
        searcher: Searcher | None = None,
        questions: PendingQuestions | None = None,
        changes: ChangeTracker | None = None,
        presence: PresenceTracker | None = None,
        policy: InteractionPolicy | None = None,
        world: WorldModel | None = None,
        memory: Memory | None = None,
        personality: Personality | None = None,
        robot: Robot | None = None,
        events: EventBus | None = None,
        search_max_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        rng: random.Random | None = None,
        sleep: Callable[[float], None] | None = None,
        poll_s: float = POLL_S,
    ) -> None:
        self.config = config
        self.arbiter = arbiter
        self.safety = safety
        self.explorer = explorer
        self.windows = windows
        self.questions = questions
        self.changes = changes
        self.presence = presence
        self.policy = policy
        self.world = world
        self.memory = memory
        self.personality = personality if personality is not None else Personality()
        self.poll_s = poll_s
        self._clock = clock
        self._wall = wall_clock
        self._rng = rng if rng is not None else random.Random(config.seed or time.time_ns())
        self._stop = threading.Event()
        self._sleep = sleep if sleep is not None else (lambda seconds: self._stop.wait(seconds))
        self._lock = threading.Lock()
        self._interrupt = threading.Event()  # a conversation began (an event), so an activity must end
        self._last_activity = clock()  # when somebody last needed her (a chat, a behaviour, the pause)
        self._next_at = 0.0  # no new choice before this
        self._last_run: dict[str, float] = {}
        self._followed: dict[str, float] = {}
        self._streak: tuple[str | None, int] = (None, 0)
        self._rest_s = 0.0
        self._rest_at = 0.0
        self._thread: threading.Thread | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self.activities: dict[str, Activity] = {
            EXPLORE: ExploreActivity(config, explorer),
            LOOK_OUTSIDE: LookOutsideActivity(config, windows),
            FOLLOW_UP: FollowUpActivity(config, searcher, arbiter, search_max_s),
            CHECK_HOME: CheckHomeActivity(config, searcher, arbiter, search_max_s),
            REST: RestActivity(config, arbiter, robot, clock),
        }
        self.history: deque[IdleRun] = deque(maxlen=HISTORY)
        self.counts: dict[str, int] = {name: 0 for name in IDLE_ACTIVITIES}
        if events is not None:
            self._unsubscribe = events.subscribe(
                self._on_event, EventType.INTERACTION_STARTED, EventType.INTERACTION_COMPLETED
            )

    # --- lifecycle ---------------------------------------------------------------

    def start(self) -> None:
        """Start the selector's thread. Nothing when it is switched off or `idle_after_s` is 0."""
        if self._thread is not None or not self.config.enabled or self.config.idle_after_s <= 0:
            return
        self._thread = threading.Thread(target=self._run, name="laika-idle", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._interrupt.set()
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(JOIN_TIMEOUT_S)
        self.flush()

    def _run(self) -> None:
        while not self._stop.wait(self.poll_s):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - the selector must survive anything
                log.exception("idle selector failed")

    def _on_event(self, event: Event) -> None:
        with self._lock:
            self._last_activity = self._clock()
        if event.type is EventType.INTERACTION_STARTED:
            self._interrupt.set()

    # --- one look: idle or not, and what to do ---------------------------------------

    def tick(self) -> IdleRun | None:
        """Choose and run one activity if she is idle now. Blocks for as long as the activity runs."""
        config = self.config
        if not config.enabled or config.idle_after_s <= 0 or self._stop.is_set():
            return None
        now = self._clock()
        if self._busy():
            self._touch(now)
            return None
        with self._lock:
            idle_for = now - self._last_activity
        if idle_for < config.idle_after_s or now < self._next_at:
            return None
        ctx = self.context(now)
        scores = self.scores(ctx)
        name = self.choose(ctx, scores)
        if name is None:
            self._next_at = now + config.between_activities_s
            return None
        ranked = ", ".join(f"{n} {s:.2f}" for n, s in sorted(scores.items(), key=lambda item: -item[1]) if s > 0)
        explore = self.activities.get(EXPLORE)
        why = f"; explore off: {explore.why_not(ctx)}" if explore is not None and not scores.get(EXPLORE) else ""
        if not why and explore is not None and not ctx.driving and self.explorer is not None:
            why = f"; explore head-only: {self.explorer.drive_block()}"
        print(f"[laika] idle: chose {name} ({ranked}{why})")
        log.debug("idle factors: %s", {n: self.activities[n].factors(ctx) for n in scores if scores[n] > 0})
        return self._run_activity(name, ctx, scores)

    def _busy(self) -> bool:
        """Somebody needs the body or asked her to stay still: she is not idle."""
        return (
            self.arbiter.holder is not None
            or bool(self.arbiter.waiting)
            or self.safety.paused
        )

    def _touch(self, now: float) -> None:
        with self._lock:
            self._last_activity = max(self._last_activity, now)

    def _conversation(self) -> bool:
        """The body is held by a chat or a user command (or something safer): an activity must give way."""
        holder = self.arbiter.holder
        return holder is not None and holder.priority <= Priority.CONVERSATION

    def _stopped(self) -> bool:
        return self._stop.is_set() or self._interrupt.is_set() or self._conversation() or self.safety.paused

    # --- the context and the choice ------------------------------------------------------

    def context(self, now: float | None = None) -> IdleContext:
        now = self._clock() if now is None else now
        wall = self._wall()
        moment = datetime.fromtimestamp(wall)
        hour = moment.hour + moment.minute / 60
        explorer = self.explorer
        driving = explorer is not None and explorer.config.enabled and explorer.driving
        left = self.safety.drive_budget.snapshot().get("left_s")
        questions, changes = self._targets(now, wall)
        return IdleContext(
            now=now,
            hour=hour,
            period=period_of(hour),
            state=self._state(),
            battery=self.safety.battery_state,
            driving=bool(driving),
            present=self._present(),
            budget_left_s=left,
            personality=self.personality,
            question_objects=questions,
            change_objects=changes,
            windows_due=len(self.windows.due_windows()) if self.windows is not None else 0,
            since={name: (now - self._last_run[name] if name in self._last_run else None) for name in IDLE_ACTIVITIES},
            streak=self._streak,
            max_repeats=self.config.max_repeats,
        )

    def _state(self) -> Availability | None:
        if self.policy is None:
            return None
        try:
            return self.policy.state()
        except Exception:  # noqa: BLE001 - unknown is treated as quiet: she rests
            log.exception("could not read the interaction state")
            return Availability.BUSY

    def _present(self) -> bool:
        try:
            return bool(self.presence.present) if self.presence is not None else False
        except Exception:  # noqa: BLE001
            return False

    def _targets(self, now: float, wall: float) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """The objects a follow-up could be about: waiting questions first, then recent changes."""

        def fresh(name: str) -> bool:
            last = self._followed.get(name)
            return last is None or now - last >= FOLLOW_UP_COOLDOWN_S

        questions: list[str] = []
        if self.questions is not None:
            questions = [q.object for q in self.questions.pending() if fresh(q.object)]
        changed: list[str] = []
        if self.changes is not None:
            changed = [
                c.object
                for c in self.changes.recent(20)
                if wall - c.timestamp <= CHANGE_WINDOW_S and fresh(c.object) and c.object not in questions
            ]
        return (
            tuple(dict.fromkeys(questions))[:MAX_TARGETS],
            tuple(dict.fromkeys(changed))[:MAX_TARGETS],
        )

    def scores(self, ctx: IdleContext) -> dict[str, float]:
        """The utility of every activity, 0.0 for one that is not available (explainable: see the docstring)."""
        return {name: self.activities[name].utility(ctx) for name in IDLE_ACTIVITIES}

    def choose(self, ctx: IdleContext, scores: Mapping[str, float] | None = None) -> str | None:
        """The activity with the highest utility, or None when nothing is possible. Ties: the seeded generator."""
        scores = self.scores(ctx) if scores is None else scores
        eligible = [name for name in IDLE_ACTIVITIES if scores[name] > 0]
        if not eligible:
            return None
        streak_name, streak_count = ctx.streak
        varied = [n for n in eligible if not (n == streak_name and streak_count >= ctx.max_repeats)]
        pool = varied or eligible  # nothing else is possible: the same one again is fine
        best = max(round(scores[n], 4) for n in pool)
        tied = [n for n in pool if round(scores[n], 4) == best]
        return tied[0] if len(tied) == 1 else self._rng.choice(tied)

    # --- running one ------------------------------------------------------------------------

    def _run_activity(self, name: str, ctx: IdleContext, scores: Mapping[str, float]) -> IdleRun:
        activity = self.activities[name]
        self._interrupt.clear()
        max_s = max(1.0, float(self.config.max_activity_s))
        if name == EXPLORE and self.explorer is not None:
            max_s = min(max_s, self.explorer.config.max_duration_s)
        run = Run(max_s, self._stopped, self._clock, self._sleep)
        try:
            outcome = activity.run(ctx, run)
        except Exception:  # noqa: BLE001 - one bad activity must not stop her life
            log.exception("idle activity %s failed", name)
            outcome = Outcome("failed", reason="error")
        now = self._clock()
        self._next_at = now + self.config.between_activities_s
        if outcome.status in RECORDED:
            self._last_run[name] = now
            self.counts[name] += 1
            self._streak = (name, self._streak[1] + 1 if self._streak[0] == name else 1)
            if name == FOLLOW_UP:
                self._mark_followed(activity, now)
            self._record(name, outcome)
        else:
            self._last_run[name] = now  # a refused or empty try is not retried at once
        if outcome.status == "interrupted" or self._interrupt.is_set():
            with self._lock:  # the wake word ends it for good: wait the whole idle time again
                self._last_activity = now
        entry = IdleRun(name, dict(scores), outcome, outcome.summary)
        self.history.append(entry)
        print(f"[laika] idle: {outcome.summary or name + ' ' + outcome.status + (': ' + outcome.reason if outcome.reason else '')}")
        return entry

    def _mark_followed(self, activity: Activity, now: float) -> None:
        target = getattr(activity, "last_target", None)
        if target:
            self._followed[target] = now

    # --- remembering ------------------------------------------------------------------------

    def _record(self, name: str, outcome: Outcome) -> None:
        """Keep the run as an `idle` episode. A run of rests is added up and kept as one."""
        if self.memory is None:
            return
        if name == REST:
            self._rest_s += outcome.seconds
            self._rest_at = self._wall()
            if outcome.status == "interrupted" or self._rest_s >= REST_FLUSH_S:
                self.flush(outcome.status)
            return
        self.flush()
        summary = outcome.summary + (" (interrupted)" if outcome.status == "interrupted" else "")
        self._episode(name, outcome.status, outcome.seconds, summary, self._wall())

    def flush(self, status: str = "done") -> None:
        """Record the rests added up so far as one episode."""
        seconds, self._rest_s = self._rest_s, 0.0
        if seconds <= 0 or self.memory is None:
            return
        suffix = " (interrupted)" if status == "interrupted" else ""
        summary = f"Idle: rested for {duration_phrase(seconds)}{suffix}"
        self._episode(REST, status, seconds, summary, self._rest_at or self._wall())

    def _episode(self, name: str, status: str, seconds: float, summary: str, when: float) -> None:
        assert self.memory is not None
        room = self.world.current_room if self.world is not None else None
        try:
            self.memory.episodes.record(
                EPISODE_KIND, summary, ["idle", name, status, f"{round(seconds)}s"], room, when
            )
        except Exception:  # noqa: BLE001 - remembering is a note, never a reason to stop
            log.exception("could not record the idle episode")

    def snapshot(self) -> dict[str, Any]:
        """What she has been doing, for logs and tests."""
        return {"counts": dict(self.counts), "last": self.history[-1].activity if self.history else None}
