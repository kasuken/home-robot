"""Looking for someone or something, and moving closer to it (#43).

`Searcher` backs the `find_object`, `find_person` and `approach` tools. It runs on the agent's thread
under the conversation's lease (the user asked for it), one call at a time. The steps, cheapest first:

1. memory: `WorldModel.where_is` / `usual_room`, the newest episodes and location facts say where the
   thing was last seen. For a person: the presence tracker (point the head where the person was and
   confirm with a picture) and the latest sighting in the observation log;
2. a head scan where she is (`Scanner`, wheels never move) when she is in the room it was seen in,
   does not know where she is, has nothing in memory, or may not drive at all;
3. goal-directed exploration (`Explorer.run_goal`) when she may drive: she wanders and scans, and the
   exploration stops as soon as a scan shows the target. It is bounded by `max_s`.

Whatever finds it, the head is pointed at it and the result says how it was found, in which room and
camera direction. There is no map: "go to the kitchen" is not possible, so a thing last seen in another
room is reported as such, honestly, never navigated to. People are counted, never identified.

Hard rule: every wheel movement she decides on herself (exploring, approaching) follows
`safety.allow_autonomous`, even though the user asked and the conversation lease is not autonomous for
the controller. With the switch off (the robot's default) a search is memory plus one head scan.
Docs: docs/behaviours.md, "Searching and approaching".

Object names match fold-insensitively (case, accents, underscores) and by whole words, so "backpack"
finds "black backpack" and "coffee machine" finds "coffee_machine". Synonyms are out of scope.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator

from laika.behaviour import Arbiter, Lease, Priority
from laika.commands import Direction, MoveCommand, ResultStatus
from laika.config import SearchConfig
from laika.explore import ExploreDecision, Explorer
from laika.memory import Memory, iso_minutes, relative_time
from laika.observations import LOG_SIZE, Observation, Observer, fold
from laika.robot import Robot
from laika.safety import SafetyController
from laika.scan import EnvironmentScan, PositionResult, ScanBusy, Scanner, label_for
from laika.world_model import KnownObject, WorldModel

if TYPE_CHECKING:
    from laika.presence import PresenceTracker

log = logging.getLogger(__name__)

NAME = "search"
CAPTURE_REASON = "search"
SETTLE_S = 0.5  # the head needs a moment before a picture
OBSERVATION_WAIT_S = 2.0
MIN_EXPLORE_S = 3.0  # less time than this left: not worth starting to drive
MAX_EPISODES = 2
MAX_FACTS = 2
AHEAD_DEG = 20.0  # only a target roughly ahead is driven towards without turning first
APPROACH_SPEED = 20  # one short move; the safety layer clamps both
APPROACH_MS = 800
TURN_STEERING = 30  # one turning move: full-ish steering, forward
TURN_ESTIMATE_DEG = 30.0  # roughly how far it turns the nose; the next picture checks it
SIGHTING_MAX_AGE_S = 300.0  # approach uses a sighting at most this old
PERSON_WORDS = frozenset({"person", "persona", "people", "persone", "someone", "qualcuno", "human"})
# A move that ended for one of these means nothing else should be tried this turn.
ABORT_REASONS = frozenset({"emergency_stop", "watchdog", "ai_unavailable", "closed", "preempted", "lease_expired"})


def name_matches(wanted: str, seen: str) -> bool:
    """Same thing by name: equal when folded, or one is a run of whole words of the other."""
    a, b = fold(wanted), fold(seen)
    if not a or not b:
        return False
    return a == b or f" {a} " in f" {b} " or f" {b} " in f" {a} "


class _Stopped(Exception):
    """The turn was cancelled or the lease taken away: stop at once."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class _Goal:
    kind: str  # "object" or "person"
    label: str

    def confidence_in(self, observation: Observation | None) -> float | None:
        """How sure this observation shows the goal, or None when it does not."""
        if observation is None:
            return None
        if self.kind == "person":
            return observation.confidence if observation.people > 0 else None
        seen = [o.confidence for o in observation.objects if name_matches(self.label, o.name)]
        return max(seen) if seen else None


@dataclass(frozen=True)
class _Hit:
    how: str
    room: str | None
    pan: float
    tilt: float
    people: int = 0


@dataclass
class _Run:
    """One call: whose body it is, how to tell it was cancelled, and when time is up."""

    lease: Lease
    should_stop: Callable[[], bool]
    deadline: float

    def stopped(self) -> bool:
        return not self.lease.active or self.should_stop()

    def check(self) -> None:
        if not self.lease.active:
            raise _Stopped("preempted")
        if self.should_stop():
            raise _Stopped("cancelled")

    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())


@dataclass
class SearchResult:
    goal: _Goal
    found: bool = False
    how: str | None = None
    where: dict[str, Any] | None = None
    people: int = 0
    last_known: dict[str, Any] | None = None
    searched: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    drove: bool = False
    autonomous_driving: bool = False
    explored: dict[str, Any] | None = None
    known_room: str | None = None

    def hit(self, hit: _Hit) -> None:
        self.found, self.how = True, hit.how
        self.where = {
            "room": hit.room,
            "direction": label_for(hit.pan),
            "pan": hit.pan,
            "tilt": hit.tilt,
            "head_pointed": True,
        }
        self.people = hit.people

    def to_dict(self) -> dict[str, Any]:
        """What the tool tells the model: compact, with relative times like `recall`."""
        data: dict[str, Any] = {
            "ok": True,
            "target": self.goal.label,
            "found": self.found,
            "how": self.how if self.found else ("memory" if self.last_known else None),
            "searched": self.searched,
            "autonomous_driving": self.autonomous_driving,
            "drove": self.drove,
        }
        if self.found:
            data["where"] = self.where
            if self.goal.kind == "person":
                data["people"] = self.people
        elif self.last_known:
            data["confirmed"] = False
        if self.last_known:
            data["last_known"] = self.last_known
        if self.explored:
            data["explored"] = self.explored
        data["notes"] = self.notes
        return data


class Searcher:
    def __init__(
        self,
        robot: Robot,
        safety: SafetyController,
        arbiter: Arbiter,
        scanner: Scanner,
        explorer: Explorer | None,
        world: WorldModel,
        memory: Memory,
        presence: PresenceTracker | None,
        observer: Observer | None,
        config: SearchConfig,
        clock: Callable[[], float] = time.time,
        *,
        settle_s: float = SETTLE_S,
        observation_wait_s: float = OBSERVATION_WAIT_S,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.robot = robot
        self.safety = safety
        self.arbiter = arbiter
        self.scanner = scanner
        self.explorer = explorer
        self.world = world
        self.memory = memory
        self.presence = presence
        self.observer = observer
        self.config = config
        self._clock = clock
        self.settle_s = max(0.0, settle_s)
        self.observation_wait_s = max(0.0, observation_wait_s)
        self._sleep = sleep
        self._one = threading.Lock()

    # --- the tools ------------------------------------------------------------------

    def find_object(
        self,
        name: str,
        *,
        lease: Lease | None = None,
        max_s: float | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Look for a named object: memory, a head scan, then (if she may drive) exploring."""
        label = " ".join(name.split())
        if not fold(label) or fold(label) in PERSON_WORDS:
            return {"ok": False, "error": "say which object to look for (for a person use find_person)"}
        goal = _Goal("object", label)
        return self._guarded(lease, should_stop, lambda run: self._find(goal, run), max_s)

    def find_person(
        self,
        *,
        lease: Lease | None = None,
        max_s: float | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Look for *a* person: presence, a head scan, then exploring. People are counted, never identified."""
        goal = _Goal("person", "person")
        return self._guarded(lease, should_stop, lambda run: self._find(goal, run), max_s)

    def approach(
        self,
        target: str,
        *,
        lease: Lease | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Move closer to what was seen last (an object name or "person"), a few short safe moves."""
        label = " ".join(target.split())
        goal = _Goal("person", "person") if fold(label) in PERSON_WORDS else _Goal("object", label)
        base: dict[str, Any] = {"ok": False, "target": goal.label, "moved": False}
        # Her own decision to drive, even when the user asked: it follows the autonomous switch.
        if not self.safety.allow_autonomous:
            return {
                **base,
                "reason": "autonomous_disabled",
                "error": "autonomous driving is off: she cannot come closer by herself",
            }
        if not self.safety.enabled:
            return {**base, "reason": "driving_disabled", "error": "the wheels are disabled"}
        return self._guarded(lease, should_stop, lambda run: self._approach(goal, run, base))

    # --- running one call ------------------------------------------------------------

    def _guarded(
        self,
        lease: Lease | None,
        should_stop: Callable[[], bool] | None,
        work: Callable[[_Run], dict[str, Any]],
        max_s: float | None = None,
    ) -> dict[str, Any]:
        if not self._one.acquire(blocking=False):
            return {"ok": False, "error": "already searching or approaching"}
        try:
            with self._body(lease) as body:
                if body is None:
                    return {"ok": False, "reason": "busy", "error": "the body is busy with something more important"}
                run = _Run(body, should_stop or (lambda: False), time.monotonic() + self._budget(max_s))
                try:
                    run.check()
                    return work(run)
                except _Stopped as gone:
                    self.safety.stop("stopped")
                    print(f"[laika] search interrupted: {gone.reason}")
                    return {"ok": False, "reason": gone.reason, "error": f"interrupted: {gone.reason}"}
        finally:
            self._one.release()

    @contextmanager
    def _body(self, lease: Lease | None) -> Iterator[Lease | None]:
        """The caller's lease, or (no conversation) one asked from the arbiter for the length of the call."""
        if lease is not None:
            yield lease
            return
        own = self.arbiter.request(NAME, Priority.CONVERSATION, resumable=False)
        try:
            yield own
        finally:
            if own is not None:
                own.release()

    def _budget(self, max_s: float | None) -> float:
        wanted = self.config.default_max_s if max_s is None else float(max_s)
        return max(1.0, min(wanted, self.config.max_max_s))

    # --- finding ----------------------------------------------------------------------

    def _find(self, goal: _Goal, run: _Run) -> dict[str, Any]:
        result = SearchResult(goal)
        print(f"[laika] searching for {goal.label}, up to {run.remaining():g} s")
        self._recall(goal, result)
        drive = self._may_drive(result)
        if goal.kind == "person":
            self._check_presence(goal, run, result)
        if not result.found and self._scan_here(goal, result, drive):
            self._scan(goal, run, result)
        if not result.found and drive:
            self._explore(goal, run, result)
        if not result.found:
            self._say_not_found(result)
        if goal.kind == "person":
            result.notes.append("she cannot tell people apart: she looks for a person, any person, never a name")
        print(f"[laika] search for {goal.label}: {'found by ' + (result.how or '') if result.found else 'not found'}")
        return result.to_dict()

    def _may_drive(self, result: SearchResult) -> bool:
        """Whether she may drive to look. Notes why not, so she can say she only looked around."""
        explorer = self.explorer
        if not self.safety.allow_autonomous:
            result.notes.append("autonomous_disabled: the wheels stay still, she only looks around from where she is")
        elif not self.safety.enabled:
            result.notes.append("driving_disabled: the wheels are off, she only looks around from where she is")
        elif explorer is None or not explorer.config.enabled:
            result.notes.append("exploration_disabled: she only looks around from where she is")
        else:
            result.autonomous_driving = True
        return result.autonomous_driving

    def _scan_here(self, goal: _Goal, result: SearchResult, drive: bool) -> bool:
        """A head scan first, unless she knows it is in another room and can go and look there."""
        current = self.world.current_room
        if not drive or goal.kind == "person" or current is None or result.known_room is None:
            return True
        return result.known_room == current

    def _scan(self, goal: _Goal, run: _Run, result: SearchResult) -> None:
        run.check()
        try:
            scan = self.scanner.scan(keep_going=lambda: not run.stopped())
        except ScanBusy:
            result.notes.append("a scan was already running, so she did not look around now")
            return
        run.check()  # cut short by a cancel: the partial scan is not used
        result.searched.append("head_scan")
        if not scan.ok:
            result.notes.append("the camera gave nothing to look at")
            return
        hit = self._hit_in(goal, scan, "scan")
        if hit is None:
            result.notes.append(f"not in view from here ({scan.summary})")
            return
        result.hit(hit)
        self.robot.look(hit.pan, hit.tilt)

    def _explore(self, goal: _Goal, run: _Run, result: SearchResult) -> None:
        explorer = self.explorer
        assert explorer is not None  # _may_drive checked
        remaining = run.remaining()
        if remaining < MIN_EXPLORE_S:
            result.notes.append("no time left to explore")
            return
        found: list[_Hit] = []

        def on_scan(scan: EnvironmentScan) -> ExploreDecision:
            if run.stopped():
                return ExploreDecision("stop", "interrupted")
            hit = self._hit_in(goal, scan, "exploration")
            if hit is None:
                return ExploreDecision()
            found.append(hit)
            return ExploreDecision("stop", "found")

        ended = explorer.run_goal(run.lease, remaining, on_scan, should_stop=run.stopped)
        result.searched.append("exploration")
        result.drove = explorer.steps > 0
        result.explored = {"ended": ended, "steps": explorer.steps, "scans": explorer.scans}
        run.check()
        if found:
            result.hit(found[0])
            self.robot.look(found[0].pan, found[0].tilt)
        elif ended == "duration":
            result.notes.append("time ran out while exploring")
        elif ended != "decision":
            result.notes.append(f"exploration ended early: {ended}")

    def _hit_in(self, goal: _Goal, scan: EnvironmentScan, how: str) -> _Hit | None:
        """The scan position that shows the goal best (None if none does)."""
        best: tuple[float, PositionResult] | None = None
        for position in scan.positions:
            confidence = goal.confidence_in(position.observation)
            if confidence is not None and (best is None or confidence > best[0]):
                best = (confidence, position)
        if best is None:
            return None
        position = best[1]
        observation = position.observation
        assert observation is not None
        return _Hit(how, observation.location or scan.location, position.pan, position.tilt, observation.people)

    def _check_presence(self, goal: _Goal, run: _Run, result: SearchResult) -> None:
        """Someone is known to be near: look where they were and confirm with a picture."""
        presence = self.presence
        if presence is None or not presence.present:
            return
        pan, tilt = presence.direction
        original = self.robot.head_pose()
        observation = self._look_and_see(run, pan, tilt)
        result.searched.append("presence")
        if goal.confidence_in(observation) is not None:
            assert observation is not None
            result.hit(_Hit("presence", observation.location or presence.room, pan, tilt, observation.people))
            return
        self.robot.look(*original)
        result.notes.append("someone was near a moment ago but is not where she saw them")

    def _say_not_found(self, result: SearchResult) -> None:
        room, current = result.known_room, self.world.current_room
        if room is not None and room != current:
            result.notes.append(
                f"last seen in the {room.replace('_', ' ')}, but she cannot navigate to a room yet: "
                "she can only look around where she is or wander and look"
            )

    # --- memory -----------------------------------------------------------------------

    def _recall(self, goal: _Goal, result: SearchResult) -> None:
        now = self._clock()
        if goal.kind == "person":
            self._recall_person(now, result)
        else:
            self._recall_object(goal, now, result)

    def _recall_person(self, now: float, result: SearchResult) -> None:
        for observation in self._recent():
            if observation.people > 0:
                result.last_known = {
                    "room": observation.location,
                    "when": iso_minutes(observation.timestamp),
                    "ago": relative_time(observation.timestamp, now),
                    "people": observation.people,
                }
                result.known_room = observation.location
                return
        result.notes.append("she has not seen anyone lately")

    def _recall_object(self, goal: _Goal, now: float, result: SearchResult) -> None:
        known = self._known(goal.label)
        name = known.name if known is not None else goal.label
        usual = self.world.usual_room(name) if known is not None else None
        episodes = self.memory.episodes.about(name, MAX_EPISODES)
        facts = [f for f in self.memory.facts.search(name, MAX_FACTS) if f.kind == "location"]
        if known is None and not episodes and not facts:
            result.notes.append("she has never seen it and was never told where it is")
            return
        last: dict[str, Any] = {}
        if known is not None:
            last.update(
                room=known.room,
                when=iso_minutes(known.last_seen),
                ago=relative_time(known.last_seen, now),
                source=known.source,
            )
            if known.note:
                last["note"] = known.note
        if usual is not None:
            last["usual_room"] = usual
        if facts:
            last["told"] = facts[0].text
        if episodes:
            last["latest"] = [
                {"ago": relative_time(e.timestamp, now), "summary": e.summary, "room": e.location}
                for e in episodes
            ]
        result.last_known = last
        result.known_room = (
            (known.room if known is not None else None)
            or usual
            or next((e.location for e in episodes if e.location), None)
        )

    def _known(self, label: str) -> KnownObject | None:
        exact = self.world.where_is(label)
        if exact is not None:
            return exact
        matches = [
            known
            for known in (KnownObject.from_dict(raw) for raw in self.world.to_dict()["objects"])
            if name_matches(label, known.name)
        ]
        return max(matches, key=lambda k: k.last_seen, default=None)

    def _recent(self) -> list[Observation]:
        return self.observer.log.recent(LOG_SIZE) if self.observer is not None else []

    # --- approaching --------------------------------------------------------------------

    def _approach(self, goal: _Goal, run: _Run, base: dict[str, Any]) -> dict[str, Any]:
        sighting, problem = self._sighting(goal)
        if sighting is None:
            return {**base, "reason": "not_seen_recently", "error": problem}
        out = {**base, "ok": True, "moves": 0, "turned": False, "reached": False}
        pan, tilt = sighting.camera_pan, sighting.camera_tilt
        reason = self._turn_and_go(goal, run, out, pan, tilt)
        out["reason"] = reason
        out["distance_cm"] = self.robot.distance_cm()
        out["moved"] = out["moves"] > 0 or out["turned"]
        if reason == "close_enough":
            out["reached"] = True
        out["ok"] = out["reached"] or out["moved"]
        if reason in ABORT_REASONS or not out["ok"]:
            out["ok"] = False
            out["error"] = f"she could not come closer: {reason.replace('_', ' ')}"
        print(f"[laika] approach {goal.label}: {reason} ({out['moves']} moves)")
        return out

    def _sighting(self, goal: _Goal) -> tuple[Observation | None, str]:
        """The newest observation of the goal, if it is recent and from the room she is in."""
        now = self._clock()
        current = self.world.current_room
        for observation in self._recent():
            if goal.confidence_in(observation) is None:
                continue
            if now - observation.timestamp > SIGHTING_MAX_AGE_S:
                break
            if current is not None and observation.location not in (None, current):
                return None, "she saw it in another room and cannot navigate there: use find_object here"
            return observation, ""
        return None, "she has not seen it lately: look for it first (find_object or find_person)"

    def _turn_and_go(self, goal: _Goal, run: _Run, out: dict[str, Any], pan: float, tilt: float) -> str:
        """Turn once if the target is not ahead, then up to `approach_moves` short forward moves."""
        if abs(pan) > AHEAD_DEG:
            sign = 1.0 if pan > 0 else -1.0  # a positive pan is to the right, so is positive steering
            if (refused := self._drive(run, TURN_STEERING * sign)) is not None:
                return refused
            out["turned"] = True
            pan -= sign * TURN_ESTIMATE_DEG
            if goal.confidence_in(self._look_and_see(run, pan, tilt)) is None:
                return "lost_sight"
            if abs(pan) > AHEAD_DEG:
                return "not_ahead"
        else:
            self.robot.look(pan, tilt)
        stop_cm = self.config.approach_stop_cm
        for _ in range(self.config.approach_moves):
            run.check()
            distance = self.robot.distance_cm()
            if distance is None:
                return "no_reading"
            if distance <= stop_cm:
                return "close_enough"
            if (refused := self._drive(run, 0.0)) is not None:
                return refused
            out["moves"] += 1
            if goal.confidence_in(self._look_and_see(run, pan, tilt)) is None:
                return "lost_sight"
        distance = self.robot.distance_cm()
        return "close_enough" if distance is not None and distance <= stop_cm else "max_moves"

    def _drive(self, run: _Run, steering: float) -> str | None:
        """One short forward move through the safety layer. None when it ran; else why it did not."""
        run.check()
        command = MoveCommand(
            direction=Direction.FORWARD, speed=APPROACH_SPEED, duration_ms=APPROACH_MS, steering=steering
        )
        outcome = self.safety.execute(command, lease=run.lease)
        run.lease.kick()
        if outcome is None:
            return "no_result"
        if outcome.result is ResultStatus.SUCCESS:
            return None
        return outcome.reason or outcome.result.value

    # --- pictures -----------------------------------------------------------------------

    def _look_and_see(self, run: _Run, pan: float, tilt: float) -> Observation | None:
        """Point the head, take a picture and wait (bounded) for what she makes of it."""
        run.check()
        self.robot.look(pan, tilt)
        self._sleep(self.settle_s)
        run.check()
        image = self._capture()
        if image is None or self.observer is None:
            return None
        return self.observer.wait_for(image, self.observation_wait_s)

    def _capture(self):
        camera = self.scanner.camera
        image = camera.capture(CAPTURE_REASON)
        if image is not None and camera.reused:  # asked too soon: wait out the camera's interval
            self._sleep(max(camera.min_interval_s, self.settle_s) + 0.05)
            image = camera.capture(CAPTURE_REASON)
            if image is not None and camera.reused:
                return None
        return image
