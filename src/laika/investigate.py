"""Investigation: react to something interesting without always interrupting someone (#29).

When the curiosity engine (#28) finds an observation interesting, the `Investigator`
looks into it on its own worker thread, one investigation at a time, under a CURIOSITY
lease (which pre-empts exploration; exploration resumes afterwards):

1. inspect: point the head at where the observation was made;
2. take another picture (reason "investigate") and wait for its observation, to confirm the
   object is still there and refine the confidence;
3. move closer, only if the object was roughly ahead and the way ahead is clear: at most
   `max_approach_moves` short forward moves. This follows the autonomous-driving switch
   even for a user-requested investigation, so with it off (the robot's default) the step
   is skipped; then take another picture;
4. conclude: an object that is new, unknown or doubtful becomes a queued question for later;
   otherwise the sighting is remembered as a memory fact.

It never speaks. Questions wait in `PendingQuestions`; the agent sees them while a
conversation is active and the `answer_question` tool stores the answer. When to interrupt
somebody is the interaction policy's job (V0.6).

Triggers are `InterestingChangeDetected` events, the explorer's `on_scan` seam
(`consider_scan`) and the `investigate` tool (`investigate_now`). Bounded: a queue of 3 where
the newest replaces the oldest, an hourly cap, a per-object cooldown and a time cap per
investigation. Details: docs/behaviours.md, "Investigation".
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterator

from laika.behaviour import Arbiter, Lease, Preempted, Priority
from laika.camera import Camera
from laika.commands import Direction, MoveCommand, ResultStatus
from laika.config import InvestigateConfig
from laika.events import Event, EventBus, EventType
from laika.observations import LOG_SIZE, Observation, Observer, normalise
from laika.robot import Robot
from laika.safety import SafetyController
from laika.scan import EnvironmentScan
from laika.world_model import WorldModel

if TYPE_CHECKING:
    from laika.memory import Memory

log = logging.getLogger(__name__)

NAME = "investigation"
CAPTURE_REASON = "investigate"
QUEUE_SIZE = 3
MAX_QUESTIONS = 5
QUESTION_TTL_S = 24 * 3600.0
MAX_DURATION_S = 60.0  # one investigation, whatever the vision model's speed
BODY_WAIT_S = 10.0  # how long a queued investigation waits for the body (a chat has it)
BODY_POLL_S = 0.1
SETTLE_S = 0.5  # the head needs a moment before the picture
OBSERVATION_WAIT_S = 2.0
APPROACH_MAX_PAN_DEG = 20.0  # only an object roughly ahead is worth driving towards
APPROACH_SPEED = 20
APPROACH_MS = 800  # one short move; the safety layer clamps both
HOUR_S = 3600.0
MAX_COOLDOWNS = 100
HANDLED_IDS = 64
HISTORY = 16
JOIN_TIMEOUT_S = 3.0
TOOL_WAIT_S = 3.0
VAGUE_NAMES = frozenset({"unknown", "object", "thing"})
# The reasons the scorer gives for an object, most telling first (docs/curiosity.md).
FOCUS_PREFIXES = ("new object: ", "moved: ", "unknown object: ", "unusual location: ")
NEW_PREFIXES = ("new object: ", "unknown object: ")

# Italian for the simulator's rooms; anything else is just "in casa".
ROOM_ITALIAN = {
    "living_room": "in soggiorno",
    "kitchen": "in cucina",
    "office": "in studio",
    "bedroom": "in camera da letto",
}


def room_phrase(room: str | None) -> str:
    """"in soggiorno", "in cucina"... or "in casa" for a room Laika has no word for."""
    return ROOM_ITALIAN.get(room or "", "in casa")


def question_text(object_name: str, room: str | None) -> str:
    """The question, from a template (the model never writes it)."""
    return f"Ho visto una cosa nuova {room_phrase(room)}: {object_name}. Che cos'è?"


# --- pending questions --------------------------------------------------------------


@dataclass(frozen=True)
class PendingQuestion:
    id: str
    object: str
    room: str | None
    text: str
    created: float
    observation_id: str | None = None


class PendingQuestions:
    """Questions Laika wants to ask when the moment is right. Bounded, one per object, expiring."""

    def __init__(
        self,
        max_items: int = MAX_QUESTIONS,
        ttl_s: float = QUESTION_TTL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_items = max(1, max_items)
        self.ttl_s = ttl_s
        self._clock = clock
        self._lock = threading.Lock()
        self._items: list[PendingQuestion] = []
        self._next_id = 1

    def add(
        self, object_name: str, room: str | None, observation_id: str | None = None
    ) -> PendingQuestion | None:
        """Queue a question about `object_name`; None when one about it is already waiting."""
        key = normalise(object_name)
        with self._lock:
            self._expire()
            if any(item.object == key for item in self._items):
                return None
            question = PendingQuestion(
                f"q-{self._next_id}", key, room, question_text(key, room), self._clock(), observation_id
            )
            self._next_id += 1
            self._items.append(question)
            del self._items[: -self.max_items]  # the oldest goes first
            return question

    def pending(self) -> list[PendingQuestion]:
        """The waiting questions, oldest first."""
        with self._lock:
            self._expire()
            return list(self._items)

    def get(self, question_id: str) -> PendingQuestion | None:
        return next((q for q in self.pending() if q.id == question_id), None)

    def remove(self, question_id: str) -> bool:
        with self._lock:
            before = len(self._items)
            self._items = [q for q in self._items if q.id != question_id]
            return len(self._items) < before

    def __len__(self) -> int:
        return len(self.pending())

    def _expire(self) -> None:
        now = self._clock()
        self._items = [q for q in self._items if now - q.created < self.ttl_s]


# --- targets and results ------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """What to look into: one object, and where the head pointed when it was seen."""

    object: str
    location: str | None
    observation_id: str | None = None
    pan: float | None = None
    tilt: float = 0.0
    reasons: tuple[str, ...] = ()
    total: float = 0.0
    # Never seen before (or unknown): the outcome is a question, not just a memory.
    new: bool = False
    neighbours: tuple[str, ...] = ()


def focus_object(reasons: list[str] | tuple[str, ...], names: list[str]) -> str | None:
    """The object an interest score is about, read back from its reasons."""
    ordered = sorted(names, key=len, reverse=True)
    for prefix in FOCUS_PREFIXES:
        for reason in reasons:
            if reason.startswith(prefix):
                rest = reason[len(prefix):]
                for name in ordered:
                    if rest == name or rest.startswith(name + " "):
                        return name
    return None


@dataclass
class Investigation:
    object: str
    room: str | None
    source: str  # "event", "scan" or "tool"
    # "done", "not_seen", "aborted", "timeout", "error"
    status: str = "done"
    reason: str = ""
    confirmed: bool = False
    confidence: float | None = None
    description: str = ""
    captures: int = 0
    approached: int = 0
    # Why she did not (or could not) move closer, e.g. "autonomous_disabled".
    drive_note: str = ""
    question: PendingQuestion | None = None
    question_pending: bool = False  # one about this object was already waiting
    remembered: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """What the investigate tool tells the model."""
        data: dict[str, Any] = {
            "ok": self.status in ("done", "not_seen"),
            "object": self.object,
            "room": self.room,
            "found": self.confirmed,
            "pictures": self.captures,
            "moved_closer": self.approached,
            "question_queued": self.question is not None,
            "remembered": self.remembered is not None,
        }
        if self.confidence is not None:
            data["confidence"] = self.confidence
        if self.description:
            data["description"] = self.description
        if self.drive_note:
            data["drive_note"] = self.drive_note
        if self.question_pending:
            data["note"] = "a question about it was already waiting"
        if self.status == "aborted":
            data["reason"] = self.reason or "preempted"
            data["error"] = f"investigation interrupted: {self.reason or 'preempted'}"
        elif self.status in ("timeout", "error"):
            data["error"] = f"investigation {self.status}"
        return data


class _TimedOut(Exception):
    pass


@contextmanager
def _heartbeat(lease: Lease, timeout_s: float) -> Iterator[None]:
    """Keep the lease alive while this thread blocks on a camera or a vision request."""
    done = threading.Event()
    period = max(0.05, timeout_s / 4)

    def beat() -> None:
        while not done.wait(period):
            lease.kick()

    thread = threading.Thread(target=beat, name="laika-investigate-beat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join(1.0)


# --- the investigator -----------------------------------------------------------------


class Investigator:
    def __init__(
        self,
        arbiter: Arbiter,
        safety: SafetyController,
        robot: Robot,
        camera: Camera,
        observer: Observer,
        world: WorldModel,
        memory: Memory,
        events: EventBus | None,
        config: InvestigateConfig,
        *,
        threshold: float = 0.35,
        unknown_confidence: float = 0.5,
        questions: PendingQuestions | None = None,
        clock: Callable[[], float] = time.monotonic,
        settle_s: float = SETTLE_S,
        observation_wait_s: float = OBSERVATION_WAIT_S,
        body_wait_s: float = BODY_WAIT_S,
        max_duration_s: float = MAX_DURATION_S,
    ) -> None:
        self.arbiter = arbiter
        self.safety = safety
        self.robot = robot
        self.camera = camera
        self.observer = observer
        self.observations = observer.log
        self.world = world
        self.memory = memory
        self.events = events
        self.config = config
        self.threshold = threshold
        self.unknown_confidence = unknown_confidence
        self.questions = questions or PendingQuestions(ttl_s=config.question_ttl_h * 3600.0, clock=clock)
        self._clock = clock
        self.settle_s = max(0.0, settle_s)
        self.observation_wait_s = max(0.0, observation_wait_s)
        self.body_wait_s = body_wait_s
        self.max_duration_s = max_duration_s
        self._cond = threading.Condition()
        self._queue: deque[Target] = deque(maxlen=QUEUE_SIZE)
        self._handled: deque[str] = deque(maxlen=HANDLED_IDS)
        self._starts: deque[float] = deque()
        self._cooldowns: dict[str, float] = {}
        self._one = threading.Lock()  # one investigation at a time, from any thread
        self._busy = False
        self._closed = False
        self._thread: threading.Thread | None = None
        self._unsubscribe: Callable[[], None] | None = None
        # What was done (for logs and tests).
        self.dropped = 0
        self.results: deque[Investigation] = deque(maxlen=HISTORY)

    # --- lifecycle ----------------------------------------------------------------

    def start(self) -> None:
        with self._cond:
            if self._thread is not None or self._closed:
                return
            self._thread = threading.Thread(target=self._run, name="laika-investigate", daemon=True)
            self._thread.start()
        if self.events is not None:
            self._unsubscribe = self.events.subscribe(self._on_event, EventType.INTERESTING_CHANGE_DETECTED)

    def close(self, timeout_s: float = JOIN_TIMEOUT_S) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        with self._cond:
            self._closed = True
            self._queue.clear()
            self._cond.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout_s)

    def wait_idle(self, timeout_s: float = 5.0) -> bool:
        """Wait until nothing is queued or running (for tests). False on timeout."""
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while self._queue or self._busy:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
        return True

    # --- triggers ----------------------------------------------------------------

    def _on_event(self, event: Event) -> None:
        """Bus thread: only look the observation up and queue it."""
        data = event.data
        observation = self.observations.get(str(data.get("observation_id", "")))
        target = self._target(
            observation,
            observation.id if observation else data.get("observation_id"),
            data.get("location"),
            list(data.get("objects", ())),
            tuple(data.get("reasons", ())),
            float(data.get("total", 0.0)),
        )
        if target is not None:
            self.enqueue(target)

    def consider_scan(self, scan: EnvironmentScan) -> str:
        """The explorer's `on_scan` seam: queue what a scan found interesting, then carry on.

        The investigation pre-empts exploration through the arbiter and exploration resumes
        when it ends, so this never stops the explorer.
        """
        for position in scan.positions:
            observation = position.observation
            score = self.observations.score_of(observation.id) if observation is not None else None
            if observation is None or score is None or score.total < self.threshold:
                continue
            target = self._target(
                observation,
                observation.id,
                observation.location,
                observation.object_names,
                tuple(score.reasons),
                score.total,
            )
            if target is not None:
                self.enqueue(target)
        return "continue"

    def _target(
        self,
        observation: Observation | None,
        observation_id: str | None,
        location: str | None,
        names: list[str],
        reasons: tuple[str, ...],
        total: float,
    ) -> Target | None:
        focus = focus_object(reasons, names)
        if focus is None:
            return None  # a person, or nothing nameable: nobody to ask about
        return Target(
            object=focus,
            location=location,
            observation_id=observation_id,
            pan=observation.camera_pan if observation else None,
            tilt=observation.camera_tilt if observation else 0.0,
            reasons=reasons,
            total=total,
            new=any(reason == prefix + focus for reason in reasons for prefix in NEW_PREFIXES),
            neighbours=tuple(n for n in names if n != focus),
        )

    def enqueue(self, target: Target) -> bool:
        """Queue a target. False when off, already seen, queued or cooling down. Never blocks."""
        with self._cond:
            if self._closed or not self.config.enabled:
                return False
            if target.observation_id and target.observation_id in self._handled:
                return False
            if any(queued.object == target.object for queued in self._queue):
                return False
            if not self._cooled(target.object, self._clock()):
                return False
            if target.observation_id:
                self._handled.append(target.observation_id)
            if len(self._queue) == self._queue.maxlen:
                self.dropped += 1  # the newest replaces the oldest
            self._queue.append(target)
            self._cond.notify_all()
            return True

    def _cooled(self, object_name: str, now: float) -> bool:
        last = self._cooldowns.get(object_name)
        return last is None or now - last >= self.config.object_cooldown_s

    # --- the worker ---------------------------------------------------------------

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._closed:
                    self._cond.wait()
                if self._closed:
                    return
                target = self._queue.popleft()
                self._busy = True
            try:
                self._run_target(target)
            except Exception:  # noqa: BLE001 - the worker must survive anything
                log.exception("investigation failed")
            finally:
                with self._cond:
                    self._busy = False
                    self._cond.notify_all()

    def _run_target(self, target: Target) -> None:
        while not self._one.acquire(timeout=0.2):  # a tool-run investigation is using the body
            if self._closed:
                return
        try:
            now = self._clock()
            with self._cond:
                while self._starts and now - self._starts[0] >= HOUR_S:
                    self._starts.popleft()
                if len(self._starts) >= self.config.max_per_hour:
                    print(f"[laika] investigation of {target.object} skipped: hourly limit reached")
                    return
                if not self._cooled(target.object, now):
                    return
            lease = self._acquire()
            if lease is None:
                print(f"[laika] investigation of {target.object} skipped: the body is busy")
                return
            with self._cond:
                self._starts.append(now)
                self._cooldowns[target.object] = now
                while len(self._cooldowns) > MAX_COOLDOWNS:
                    self._cooldowns.pop(next(iter(self._cooldowns)))
            try:
                result = self._investigate(target, lease, "event")
            finally:
                lease.release()
            self.results.append(result)
        finally:
            self._one.release()

    def _acquire(self) -> Lease | None:
        """A CURIOSITY lease (pre-empting exploration), waiting a little while a chat has the body."""
        give_up = self._clock() + self.body_wait_s
        while not self._closed:
            lease = self.arbiter.request(NAME, Priority.CURIOSITY, resumable=False)
            if lease is not None:
                return lease
            if self._clock() >= give_up:
                return None
            time.sleep(BODY_POLL_S)
        return None

    # --- the investigation ------------------------------------------------------------

    def _investigate(self, target: Target, lease: Lease, source: str) -> Investigation:
        result = Investigation(target.object, target.location, source)
        deadline = self._clock() + self.max_duration_s
        original = self.robot.head_pose()
        print(f"[laika] investigating {target.object} in {_words(target.location)}")
        try:
            with _heartbeat(lease, self.arbiter.lease_timeout_s):
                self._inspect(target, result, lease, deadline)
        except Preempted as gone:
            result.status, result.reason = "aborted", gone.reason
        except _TimedOut:
            result.status = "timeout"
        except Exception:  # noqa: BLE001 - a head or camera failure ends this investigation only
            log.exception("investigation of %s failed", target.object)
            result.status = "error"
        finally:
            if lease.active:
                try:
                    self.robot.look(*original)
                except Exception:  # noqa: BLE001
                    log.exception("could not restore the head after investigating")
        print(f"[laika] investigation of {target.object}: {_outcome(result)}")
        return result

    def _step(self, lease: Lease, deadline: float) -> None:
        """Between steps: still ours, still alive, still within the time cap."""
        lease.check()
        lease.kick()
        if self._clock() > deadline:
            raise _TimedOut

    def _inspect(self, target: Target, result: Investigation, lease: Lease, deadline: float) -> None:
        self._step(lease, deadline)
        observation = self._look_again(target, result, lease, deadline)
        seen = _find(observation, target.object)
        if seen is None:
            result.status = "not_seen"
            return
        result.confirmed, result.confidence = True, seen.confidence
        for _ in range(self.config.max_approach_moves):
            if not self._approach(target, result, lease, deadline):
                break
            observation = self._look_again(target, result, lease, deadline)
            seen = _find(observation, target.object)
            if seen is None:
                result.confirmed = False
                result.status = "not_seen"
                return
            result.confidence = seen.confidence
        assert observation is not None
        self._step(lease, deadline)  # pre-empted while the last picture was read: conclude nothing
        result.description = observation.description
        self._conclude(target, result, observation, seen.confidence)

    def _look_again(
        self, target: Target, result: Investigation, lease: Lease, deadline: float
    ) -> Observation | None:
        """Aim at the target, take a picture and wait (bounded) for what Laika makes of it."""
        self._step(lease, deadline)
        if target.pan is not None and self.robot.head_pose() != (target.pan, target.tilt):
            self.robot.look(target.pan, target.tilt)  # also after a scan that was cut short moved it
            time.sleep(self.settle_s)
            self._step(lease, deadline)
        image = self.camera.capture(CAPTURE_REASON)
        if image is not None and self.camera.reused:  # asked too soon: wait out the camera's interval
            time.sleep(max(self.camera.min_interval_s, self.settle_s) + 0.05)
            self._step(lease, deadline)
            image = self.camera.capture(CAPTURE_REASON)
            if image is not None and self.camera.reused:
                image = None
        if image is None:
            return None
        result.captures += 1
        wait = min(self.observation_wait_s, max(0.0, deadline - self._clock()))
        return self.observer.wait_for(image, wait)

    def _approach(self, target: Target, result: Investigation, lease: Lease, deadline: float) -> bool:
        """One short move towards the object, if it is ahead and the way is clear. True if it moved."""
        self._step(lease, deadline)
        # Moving closer is Laika's own decision even when the user asked her to investigate
        # (from the tool, under the conversation lease), so it follows the autonomous-driving
        # switch too: on the real robot it stays off until the V0.2 hardware check passes.
        if not self.safety.allow_autonomous:
            result.drive_note = "autonomous_disabled"
            return False
        if target.pan is None or abs(target.pan) > APPROACH_MAX_PAN_DEG:
            result.drive_note = "not ahead"
            return False
        distance = self.robot.distance_cm()
        if distance is None or distance <= self.config.approach_stop_cm:
            result.drive_note = "too close or no reading"
            return False
        command = MoveCommand(direction=Direction.FORWARD, speed=APPROACH_SPEED, duration_ms=APPROACH_MS)
        outcome = self.safety.execute(command, lease=lease, self_directed=True)
        lease.kick()
        if outcome is None or outcome.result is not ResultStatus.SUCCESS:
            result.drive_note = (outcome.reason or outcome.result.value) if outcome else "no result"
            return False
        result.approached += 1
        return True

    def _conclude(
        self, target: Target, result: Investigation, observation: Observation, confidence: float
    ) -> None:
        """A question for something unknown or doubtful; otherwise remember where it was."""
        known = self.world.where_is(target.object)
        user_named = known is not None and (known.note is not None or known.source == "conversation")
        doubtful = confidence < self.unknown_confidence or target.object in VAGUE_NAMES
        room = observation.location or target.location
        if not user_named and (target.new or doubtful):
            question = self.questions.add(target.object, room, observation.id)
            result.question = question
            result.question_pending = question is None
            if question is not None:
                self.memory.episodes.record(
                    "question", f"Wondered what the {target.object} is: question queued",
                    [target.object], room, observation.timestamp,
                )
            return
        fact = _fact(target.object, observation, room)
        # A new sighting replaces the last one: a usual place told by the user is a different source.
        self.memory.facts.remember(fact, target.object, "location", source="investigation", confidence=confidence)
        result.remembered = fact

    # --- the investigate tool ----------------------------------------------------------

    def investigate_now(self, object_name: str | None = None, *, lease: Lease | None = None) -> dict[str, Any]:
        """Investigate for the investigate tool: the named object, or the latest interesting sighting.

        Runs on the caller's thread under the caller's lease. The agent's turn holds the
        CONVERSATION lease, which nothing here may pre-empt, and the user asked for it, so its
        moves are not autonomous. Not counted against the hourly cap or the cooldown.
        """
        target, problem = self._tool_target(object_name)
        if target is None:
            return {"ok": False, "error": problem}
        if not self._one.acquire(timeout=TOOL_WAIT_S):
            return {"ok": False, "error": "already investigating something else"}
        try:
            # Without an arbiter lease (no conversation) a detached, user-level one stands in.
            active = lease or Lease(threading.Condition(), NAME, Priority.USER_COMMAND, resumable=False)
            result = self._investigate(target, active, "tool")
        finally:
            self._one.release()
        self.results.append(result)
        return result.to_dict()

    def _tool_target(self, object_name: str | None) -> tuple[Target | None, str]:
        recent = self.observations.recent(LOG_SIZE)
        if object_name and object_name.strip():
            key = normalise(object_name)
            known = self.world.where_is(key)
            observation = next((o for o in recent if key in o.object_names), None)
            if observation is not None:
                return Target(key, observation.location, observation.id, observation.camera_pan,
                              observation.camera_tilt, new=known is None), ""
            room = known.room if known is not None else self.world.current_room
            return Target(key, room, new=known is None), ""
        for observation in recent:
            score = self.observations.score_of(observation.id)
            if score is not None and score.total >= self.threshold and observation.objects:
                target = self._target(observation, observation.id, observation.location,
                                      observation.object_names, tuple(score.reasons), score.total)
                if target is not None:
                    return target, ""
        for observation in recent:
            if observation.objects:
                name = observation.objects[0].name
                return Target(name, observation.location, observation.id, observation.camera_pan,
                              observation.camera_tilt), ""
        return None, "nothing to investigate yet: nothing interesting has been seen"


def _find(observation: Observation | None, name: str):
    if observation is None:
        return None
    return next((o for o in observation.objects if o.name == name), None)


def _words(room: str | None) -> str:
    return room.replace("_", " ") if room else "an unknown room"


def _fact(name: str, observation: Observation, room: str | None) -> str:
    others = [n for n in observation.object_names if n != name]
    fact = f"Saw {name}"
    if others:
        fact += f" next to {others[0]}"
    if room:
        fact += f" in the {_words(room)}"
    return fact


def _outcome(result: Investigation) -> str:
    if result.status == "aborted":
        return f"interrupted ({result.reason})"
    if result.status != "done":
        return result.status.replace("_", " ")
    if result.question is not None:
        return "question queued"
    if result.question_pending:
        return "question already waiting"
    return "remembered"
