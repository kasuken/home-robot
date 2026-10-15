"""Curiosity: how interesting an observation is, and when Laika may announce it (#28).

`InterestScorer` scores one observation against the world model as it was BEFORE that
observation is applied: the Observer scores before it publishes ObservationCreated, and
the world model learns from that event afterwards. The scorer only reads. It has no
clock and no randomness, so the same inputs always give the same score and reasons.
Moved, new and unusual objects come from the `ChangeDetector` (changes.py, #36).

The total is a weighted average of nine inputs, each 0 to 1:

    total = sum(weight * component) / sum(weight)

Dividing by the weights keeps the total in 0 to 1 whatever scale the weights use.

`InterestGate` turns a score into `InterestingChangeDetected` when the total reaches the
threshold, once per reason per cooldown. It never decides to act: whether Laika looks,
asks or moves is the interaction policy's job (#29), which reads the score.
Inputs, defaults and the heuristics: docs/curiosity.md.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from laika.changes import Change, ChangeDetector
from laika.config import CURIOSITY_INPUTS, CuriosityConfig
from laika.events import EventBus, EventType, emit
from laika.observations import Observation, same_view
from laika.world_model import KnownObject, WorldModel

if TYPE_CHECKING:
    from laika.memory import Memory

INPUTS = CURIOSITY_INPUTS
# Names that say nothing about what the object is.
VAGUE_NAME = re.compile(r"\b(unknown|object|thing)\b")
MAX_REMEMBERED_REASONS = 256


@dataclass(frozen=True)
class InterestScore:
    total: float
    components: dict[str, float]
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": round(self.total, 3),
            "components": {name: round(value, 3) for name, value in self.components.items()},
            "reasons": list(self.reasons),
        }


def _share(count: int, total: int) -> float:
    return count / total if total else 0.0


def _words(room: str) -> str:
    return room.replace("_", " ")


class InterestScorer:
    """Scores observations against the world model and the remembered facts. Reads only."""

    def __init__(
        self,
        world: WorldModel,
        memory: Memory | None = None,
        config: CuriosityConfig | None = None,
        detector: ChangeDetector | None = None,
    ) -> None:
        self.world = world
        self.memory = memory
        self.config = config if config is not None else CuriosityConfig()
        # Used when the caller does not pass the changes it already detected (see `score`).
        self.detector = detector if detector is not None else ChangeDetector(world, None, self.config)
        self._weights = {name: float(getattr(self.config, name)) for name in INPUTS}

    def score(
        self,
        observation: Observation,
        *,
        previous: Observation | None = None,
        changes: list[Change] | None = None,
    ) -> InterestScore:
        """Score `observation`. `previous` is the earlier observation of the same view, if any.

        `changes` are the observation's changes from the ChangeDetector (changes.py), detected
        before the world model learned the frame; they are detected here when not given.
        """
        if changes is None:
            changes = self.detector.detect(observation, previous)
        before = {obj.name: self.world.where_is(obj.name) for obj in observation.objects}
        parts = {
            "novelty": self._novelty(observation, before),
            "environmental_change": self._environmental_change(changes),
            "uncertainty": self._uncertainty(observation),
            "person": self._person(observation),
            "unknown_object": self._unknown_object(observation, before),
            "unusual_location": self._unusual_location(observation, changes),
            "movement": self._movement(observation, previous),
            "sound": (0.0, []),
            "user_relevance": self._user_relevance(observation, before),
        }
        components = {name: min(1.0, max(0.0, parts[name][0])) for name in INPUTS}
        reasons = sorted({reason for name in INPUTS for reason in parts[name][1]})
        weight_sum = sum(self._weights.values())
        weighted = sum(self._weights[name] * components[name] for name in INPUTS)
        total = min(1.0, max(0.0, weighted / weight_sum)) if weight_sum > 0 else 0.0
        return InterestScore(total=total, components=components, reasons=reasons)

    # Each input returns (value 0 to 1, reasons for it).

    def _novelty(self, observation: Observation, before: dict[str, KnownObject | None]):
        new = [obj.name for obj in observation.objects if before[obj.name] is None]
        return _share(len(new), len(observation.objects)), [f"new object: {name}" for name in new]

    def _environmental_change(self, changes: list[Change]):
        # A change is an object last known in another room (moved here), or a never-seen object
        # in a familiar room (it arrived). One change is enough: it is not diluted by the other,
        # unchanged objects in view. A room seen for the first time has no changes, only novelty.
        reasons = [change.reason for change in changes if change.kind in ("moved", "new")]
        return (1.0 if reasons else 0.0), reasons

    def _uncertainty(self, observation: Observation):
        value = 1.0 - observation.confidence
        return value, (["uncertain view"] if value > 0 else [])

    def _person(self, observation: Observation):
        return (1.0, ["person nearby"]) if observation.people > 0 else (0.0, [])

    def _unknown_object(self, observation: Observation, before: dict[str, KnownObject | None]):
        # An object a person explained (it has a note) is known, however unsure the camera is of it.
        threshold = self.config.unknown_object_confidence
        unknown = [
            obj.name
            for obj in observation.objects
            if (obj.confidence < threshold or VAGUE_NAME.search(obj.name))
            and not (before[obj.name] is not None and before[obj.name].note)
        ]
        return _share(len(unknown), len(observation.objects)), [f"unknown object: {n}" for n in unknown]

    def _unusual_location(self, observation: Observation, changes: list[Change]):
        unusual = [change.reason for change in changes if change.kind == "unusual"]
        return _share(len(unusual), len(observation.objects)), unusual

    def _movement(self, observation: Observation, previous: Observation | None):
        # Compared with an earlier look at the same view: what appeared, vanished or changed in number.
        if previous is None or not same_view(previous, observation):
            return 0.0, []
        now = {obj.name for obj in observation.objects}
        then = {obj.name for obj in previous.objects}
        people_changed = previous.people != observation.people
        changes = len(now ^ then) + int(people_changed)
        slots = len(now | then) + int(people_changed)
        reasons = [f"appeared: {name}" for name in sorted(now - then)]
        reasons += [f"disappeared: {name}" for name in sorted(then - now)]
        if people_changed:
            reasons.append(f"people count changed: {previous.people} to {observation.people}")
        return _share(changes, slots), reasons

    def _user_relevance(self, observation: Observation, before: dict[str, KnownObject | None]):
        # Mentioned in a remembered fact (its subject or its words, as recall works) or carrying a note from the user.
        hits = []
        for obj in observation.objects:
            known = before[obj.name]
            if (known is not None and known.note) or self._remembered(obj.name):
                hits.append(f"relevant to the user: {obj.name}")
        items = len(observation.objects)
        if observation.location is not None:
            items += 1
            if self._remembered(_words(observation.location)):
                hits.append(f"relevant to the user: {_words(observation.location)}")
        return _share(len(hits), items), hits

    def _remembered(self, text: str) -> bool:
        return self.memory is not None and self.memory.mentions(text)


class InterestGate:
    """Publishes InterestingChangeDetected for scores at or above the threshold.

    A reason already announced within `cooldown_s` does not count again: the event fires
    only when at least one of the reasons is new, and the fresh reasons start their cooldown.
    """

    def __init__(
        self,
        config: CuriosityConfig,
        bus: EventBus | None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.bus = bus
        self._clock = clock
        self._lock = threading.Lock()
        self._announced: dict[str, float] = {}

    def consider(self, observation: Observation, score: InterestScore) -> bool:
        """Publish the event if the score is interesting and not all of it is cooling down."""
        if score.total < self.config.threshold:
            return False
        now = self._clock()
        cooldown = self.config.cooldown_s
        with self._lock:
            self._forget_expired(now)
            fresh = [reason for reason in score.reasons if reason not in self._announced]
            if not fresh:
                return False
            if cooldown > 0:
                for reason in fresh:
                    self._announced.pop(reason, None)
                    self._announced[reason] = now
                while len(self._announced) > MAX_REMEMBERED_REASONS:
                    self._announced.pop(next(iter(self._announced)))
        emit(
            self.bus,
            EventType.INTERESTING_CHANGE_DETECTED,
            observation_id=observation.id,
            location=observation.location,
            total=round(score.total, 3),
            reasons=list(score.reasons),
            objects=observation.object_names,
        )
        return True

    def _forget_expired(self, now: float) -> None:
        cooldown = self.config.cooldown_s
        if cooldown <= 0:
            self._announced.clear()
            return
        expired = [reason for reason, when in self._announced.items() if now - when >= cooldown]
        for reason in expired:
            del self._announced[reason]
