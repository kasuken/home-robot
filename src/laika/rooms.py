"""Which room Laika is in, from what she sees (#53, docs/behaviours.md "Room recognition").

On the robot nothing tells her the room, so the vision model guesses it in every frame
(`Observation.room_guess`) and the `RoomTracker` decides from many guesses, not from one. A single
frame of a doorway or a dark corner must not move her from the kitchen to the hallway.

How a frame votes. Its evidence for a room is the model's guess confidence for that room plus
landmark evidence: every object in the frame that the world model has seen usually in a room
(`WorldModel.usual_room`: at least two sightings there) adds LANDMARK_WEIGHT times its confidence
to that room, at most LANDMARK_CAP per room. So a sofa and a television vote for the living room even
when the model's own guess is weak or wrong. The evidence is capped at 1. The frame votes for the room
with the most evidence when that reaches `min_confidence`; otherwise it abstains.

How the votes decide. The tracker keeps the last MAX_VOTES frames from the last `window_s` seconds.
Newer frames count more: the frame `i` steps back weighs RECENCY**i. A room's score is the sum of its
votes' evidence times that weight. The room changes only when another room has at least `min_votes`
votes in the window and its score beats the current room's and every other room's by `switch_margin`
(hysteresis). When the newest `min_votes + 1` frames all abstain (a doorway or a dark corner is
a frame or two; this is a stretch with nothing to go on) she is in an unknown room: the room is
dropped, nothing is blocked by house rules and the observation has no location until she is sure
again. The confidence of the current room is its score divided by the sum of all the weights, so
abstaining frames lower it.

Where the room goes. On the robot the tracker is the one source of truth: it calls
`world.enter_room` and publishes `RoomEntered` (room, previous, confidence, source "vision") once per
change. In the simulator the ground truth from `SimulatedRobot` stays in charge and the tracker runs in
shadow mode: it decides the same way, changes nothing, and each observation event records what it
decided next to the true room, which is where the trial report takes its accuracy from.

Naming. "Questa è la camera di Leonardo" teaches the world model that name for the current room (an
alias kept with the room, used by the house rules and by recall); "no, questa è la cucina" corrects
a wrong decision. `name_current_room` is the one code path for the `name_room` tool and the memory selector.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from laika.events import EventBus, EventType, emit
from laika.house_rules import room_word
from laika.observations import Observation, fold
from laika.world_model import clean_alias

if TYPE_CHECKING:
    from laika.config import RoomsConfig
    from laika.floorplan import FloorPlan
    from laika.memory import Fact, Memory
    from laika.world_model import WorldModel

log = logging.getLogger(__name__)

MAX_VOTES = 5
RECENCY = 0.7
LANDMARK_WEIGHT = 0.3
LANDMARK_CAP = 0.6
SOURCE_VISION = "vision"
# With a floor plan, a room that is not next to the current one is a jump: it needs this much more margin
# and one more vote (nobody crosses two doors between two frames).
JUMP_EXTRA_MARGIN = 0.4
SOURCE_USER = "user"
SOURCE_TAG = "tag"
# The room types the vision prompt offers besides the rooms the world model knows.
GENERIC_ROOMS = ("living_room", "kitchen", "bedroom", "office", "bathroom", "entrance")
# English words a model may use for the same room, so one room is not known under three ids.
SYNONYMS = {
    "hallway": "entrance", "hall": "entrance", "corridor": "entrance", "foyer": "entrance",
    "lounge": "living_room", "sitting_room": "living_room", "livingroom": "living_room",
    "study": "office", "home_office": "office", "workroom": "office",
    "bedrooms": "bedroom", "master_bedroom": "bedroom",
    "restroom": "bathroom", "toilet": "bathroom", "wc": "bathroom",
}
# Things that move around the house say nothing about the room.
NOT_LANDMARKS = frozenset("person people man woman child kid baby dog cat pet".split())


def resolve_room(raw: str, world: WorldModel | None = None) -> str:
    """One id for a guess: a name people gave a room, a synonym or an Italian word mapped to the world
    model's name, else the guess itself."""
    text = raw.strip()
    if world is not None and (named := world.room_for_alias(text)):
        return named
    key = SYNONYMS.get(text, text)
    return room_word(key.replace("_", " ")) or key


def room_slug(name: str) -> str:
    """A room id for a name people chose: "camera di Leonardo" -> camera_di_leonardo."""
    return "_".join(fold(name.replace("'", "").replace("’", "")).split())


def prompt_rooms(world: WorldModel) -> list[str]:
    """The rooms the vision prompt lists: the ones the world model knows, with the names people gave them."""
    rooms = []
    for room in world.rooms():
        aliases = world.aliases_of(room)
        rooms.append(f"{room} (the household calls it: {', '.join(aliases)})" if aliases else room)
    return rooms


@dataclass(frozen=True)
class RoomDecision:
    """What the tracker made of one frame."""

    room: str | None  # the room after this frame (the tracker's, shadow or not)
    previous: str | None
    changed: bool
    confidence: float
    guess: str | None  # the model's guess, resolved to a room id
    guess_confidence: float
    frame_room: str | None  # the room this frame alone voted for, if it was confident
    drives: bool
    truth: str | None = None

    @property
    def location(self) -> str | None:
        """Where the frame was taken: the tracker's room, else the frame's own confident vote."""
        return self.room or self.frame_room

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "guess": self.guess, "guess_confidence": round(self.guess_confidence, 3),
            "decided": self.room, "confidence": round(self.confidence, 3), "changed": self.changed,
            "shadow": not self.drives,
        }
        if self.truth is not None:
            data["truth"] = self.truth
        return data


@dataclass
class RoomStats:
    frames: int = 0
    unknown_frames: int = 0  # frames after which she did not know the room
    changes: int = 0
    to_unknown: int = 0
    corrections: int = 0  # the user said it is another room
    truth_checked: int = 0
    truth_correct: int = 0

    @property
    def accuracy(self) -> float | None:
        return self.truth_correct / self.truth_checked if self.truth_checked else None


@dataclass
class _Vote:
    timestamp: float
    room: str | None
    weight: float


class RoomTracker:
    """Decides the current room from a sliding window of frames. Thread-safe."""

    def __init__(
        self,
        world: WorldModel,
        bus: EventBus | None,
        config: RoomsConfig,
        *,
        drive: bool = True,
        truth: Callable[[], str | None] | None = None,
        plan: FloorPlan | None = None,
    ) -> None:
        self.plan = plan
        self.world = world
        self.bus = bus
        self.config = config
        self.drive = drive
        self.truth = truth
        self.stats = RoomStats()
        self._lock = threading.Lock()
        self._window: deque[_Vote] = deque(maxlen=MAX_VOTES)
        self._room: str | None = None

    @property
    def current_room(self) -> str | None:
        """The room she is confident she is in, or None (unknown)."""
        with self._lock:
            return self._room

    def observe(self, observation: Observation) -> RoomDecision:
        """Count one frame's guess and landmarks, and change the room when the window says so."""
        guess = resolve_room(observation.room_guess, self.world) if observation.room_guess else None
        evidence = self._evidence(observation, guess)
        frame_room, weight = None, 0.0
        if evidence:
            best = max(evidence, key=lambda room: (evidence[room], room == guess))
            if evidence[best] >= self.config.min_confidence:
                frame_room, weight = best, evidence[best]
        with self._lock:
            when = observation.timestamp
            while self._window and when - self._window[0].timestamp > self.config.window_s:
                self._window.popleft()
            self._window.append(_Vote(when, frame_room, weight))
            previous = self._room
            new, confidence = self._decide()
            changed = new != previous
            if changed:
                self._room = new
            self.stats.frames += 1
            if self._room is None:
                self.stats.unknown_frames += 1
            if changed:
                self.stats.changes += 1 if new else 0
                self.stats.to_unknown += 0 if new else 1
            decided = self._room
        truth = self.truth() if self.truth is not None else None
        if truth is not None:
            self.stats.truth_checked += 1
            self.stats.truth_correct += 1 if decided == truth else 0
        if changed:
            self._apply(decided, previous, confidence, SOURCE_VISION, when)
        return RoomDecision(
            room=decided, previous=previous, changed=changed, confidence=confidence if decided else 0.0,
            guess=guess, guess_confidence=observation.room_confidence if guess else 0.0,
            frame_room=frame_room, drives=self.drive, truth=truth,
        )

    def tag_fix(self, room: str, timestamp: float | None = None) -> bool:
        """A printed tag says she is in `room` (tags.py): as sure as a person saying so. Returns whether the
        room changed. The plausibility of the move is not asked: a tag is not a guess."""
        return self.correct(room, timestamp, source=SOURCE_TAG)

    def correct(self, room: str, timestamp: float | None = None, source: str = SOURCE_USER) -> bool:
        """A person said this is `room`: believe it now and make the window agree, so the next frames do not
        undo it at once. Returns whether the room was different. In shadow mode only the count changes."""
        when = time.time() if timestamp is None else timestamp
        with self._lock:
            previous = self._room
            self._room = room
            self._window.clear()
            self._window.extend([_Vote(when, room, 1.0), _Vote(when, room, 1.0)])
            self.stats.corrections += 1
        if previous != room:
            self._apply(room, previous, 1.0, source, when)
        return previous != room

    def _evidence(self, observation: Observation, guess: str | None) -> dict[str, float]:
        scores: dict[str, float] = {}
        if guess:
            scores[guess] = observation.room_confidence
        landmarks: dict[str, float] = {}
        for seen in observation.objects:
            if seen.name in NOT_LANDMARKS:
                continue
            usual = self.world.usual_room(seen.name)
            if usual:
                landmarks[usual] = landmarks.get(usual, 0.0) + LANDMARK_WEIGHT * seen.confidence
        for room, weight in landmarks.items():
            scores[room] = scores.get(room, 0.0) + min(weight, LANDMARK_CAP)
        return {room: min(1.0, value) for room, value in scores.items()}

    def _decide(self) -> tuple[str | None, float]:
        """The room after the window changed, and how sure she is. Caller holds the lock."""
        votes = list(self._window)
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        scale = 0.0
        for steps_back, vote in enumerate(reversed(votes)):
            factor = RECENCY**steps_back
            scale += factor
            if vote.room is not None:
                totals[vote.room] = totals.get(vote.room, 0.0) + vote.weight * factor
                counts[vote.room] = counts.get(vote.room, 0) + 1
        current = self._room
        if totals:
            leader = max(totals, key=lambda room: totals[room])
            rivals = max((total for room, total in totals.items() if room != leader), default=0.0)
            jump = (
                self.plan is not None and current is not None
                and not self.plan.plausible_move(current, leader)
            )
            if (
                leader != current
                and counts[leader] >= self.config.min_votes + (1 if jump else 0)
                and totals[leader] - rivals >= self.config.switch_margin + (JUMP_EXTRA_MARGIN if jump else 0.0)
            ):
                return leader, min(1.0, totals[leader] / scale)
        if current is not None:
            needed = min(self.config.min_votes + 1, MAX_VOTES)
            if len(votes) >= needed and all(v.room is None for v in votes[-needed:]):
                return None, 0.0
            return current, min(1.0, totals.get(current, 0.0) / scale)
        return None, 0.0

    def _apply(self, room: str | None, previous: str | None, confidence: float, source: str, when: float) -> None:
        """Make the change real: the world model, then RoomEntered. Nothing in shadow mode."""
        if not self.drive:
            log.info("rooms (shadow): %s -> %s", previous, room)
            return
        self.world.enter_room(room, timestamp=when)
        if room is not None:
            log.info("rooms: %s -> %s (%.2f, %s)", previous, room, confidence, source)
            emit(self.bus, EventType.ROOM_ENTERED, room=room, previous=previous,
                 confidence=round(confidence, 3), source=source)
        else:
            log.info("rooms: %s -> unknown", previous)


@dataclass(frozen=True)
class RoomNaming:
    """What `name_current_room` did: "named" (a new name for the room she is in), "known" (she already
    knew it), "corrected" (it is another room than she thought) or "created" (a room she did not know)."""

    action: str
    room: str
    name: str
    previous: str | None = None
    labels: list[str] = field(default_factory=list)
    fact: Fact | None = None  # what was kept in memory, when something was


def name_current_room(
    world: WorldModel,
    memory: Memory | None,
    rooms: RoomTracker | None,
    name: str,
    now: float,
    *,
    said: str | None = None,
) -> RoomNaming:
    """Learn the name people use for the room she is in, or take a correction.

    A name that already means a room (an alias, or an Italian room word such as "la cucina") says which
    room this is: when that is not the room she thought, the tracker is corrected. Any other name becomes
    an alias of the current room, or a new room when she does not know where she is. Raises ValueError
    when `name` cannot be a room name.
    """
    alias = clean_alias(name)
    if alias is None:
        raise ValueError(f"{name!r} is not a usable room name")
    current = world.current_room
    target = world.room_for_alias(alias) or room_word(alias)
    if target is not None:
        action = "known" if target == current else ("corrected" if current else "created")
        room = target
    elif current is not None:
        action, room = "named", current
    else:
        action, room = "created", room_slug(alias)
    if action in ("corrected", "created"):
        if rooms is not None:
            rooms.correct(room, now)
        else:
            world.enter_room(room, timestamp=now)
    if target is None:
        world.add_room_alias(room, alias, now)
    fact = None
    if action != "known" and memory is not None:
        label = world.room_label(room)
        fact = memory.facts.remember(said or f"La stanza si chiama {alias}", alias, "general", source="user")
        summary = (
            f"The user said this is the {label}, not the {current.replace('_', ' ')}"
            if action == "corrected" and current else f"The user called this room: {alias}"
        )
        memory.episodes.record("conversation", summary, [alias], room, now)
    return RoomNaming(action, room, alias, current, world.aliases_of(room), fact)
