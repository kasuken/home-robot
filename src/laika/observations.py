"""Structured observations: what Laika saw, validated locally before anything uses it.

A vision model (issue #22) answers in free-form JSON, which is untrusted input:
`parse_model_observation` checks every field and rejects anything malformed with
`InvalidObservation`, so only a well-formed `Observation` ever reaches the world
model or memory. The simulator builds the same type from its ground-truth scene.
The `Observer` turns each captured frame into an observation on its own worker
thread, so a slow vision call never blocks the caller or the safety layer.
Persistence comes later (#32); `ObservationLog` only keeps the latest ones. When the
`Observer` has a scorer (#28, docs/curiosity.md), each observation is scored before it is
published, so the score is computed against the world model as it was before this frame.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

from laika.events import EventBus, EventType, emit
from laika.robot import ImageObservation

if TYPE_CHECKING:
    from laika.curiosity import InterestGate, InterestScore, InterestScorer

log = logging.getLogger(__name__)

MAX_DESCRIPTION_CHARS = 400
MAX_OBJECTS = 20
MAX_NAME_CHARS = 40
MAX_NOTE_CHARS = 100
MAX_PEOPLE = 20
MAX_MODEL_TEXT_CHARS = 20_000
LOG_SIZE = 50
QUEUE_SIZE = 8
# Two observations show the same view when they are of the same location and this close in pan.
PREVIOUS_PAN_DEG = 20.0
REJECTION_HISTORY = 50
DONE_HISTORY = 16
POSITIONS = ("left", "centre", "right")
SOURCES = ("vision", "simulator")

_FENCE = re.compile(r"^```[A-Za-z0-9_-]*\s*\n?(.*?)\n?\s*```$", re.DOTALL)


class InvalidObservation(ValueError):
    """Model output (or stored data) that is not a valid observation."""


def same_view(a: Observation, b: Observation) -> bool:
    """Same location and roughly the same camera direction."""
    return a.location == b.location and abs(a.camera_pan - b.camera_pan) <= PREVIOUS_PAN_DEG


def find_previous(candidates: Iterable[Observation], observation: Observation) -> Observation | None:
    """The newest of `candidates` (newest first) that shows the same view as `observation`."""
    for candidate in candidates:
        if candidate.id != observation.id and same_view(candidate, observation):
            return candidate
    return None


def normalise(name: str) -> str:
    """One spelling for a name: trimmed, single-spaced, lowercase."""
    return " ".join(name.split()).lower()


@dataclass(frozen=True)
class SeenObject:
    name: str
    confidence: float = 1.0
    position: str | None = None
    note: str | None = None

    def to_json(self) -> str | dict[str, Any]:
        """A plain name when there is nothing else to say, as in the PRD example."""
        if self.confidence == 1.0 and self.position is None and self.note is None:
            return self.name
        data: dict[str, Any] = {"name": self.name, "confidence": self.confidence}
        if self.position is not None:
            data["position"] = self.position
        if self.note is not None:
            data["note"] = self.note
        return data


@dataclass(frozen=True)
class Observation:
    description: str
    objects: tuple[SeenObject, ...] = ()
    confidence: float = 1.0
    location: str | None = None
    people: int = 0
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: f"obs-{uuid.uuid4().hex[:8]}")
    camera_pan: float = 0.0
    camera_tilt: float = 0.0
    image_path: str | None = None
    source: str = "vision"

    @property
    def object_names(self) -> list[str]:
        return [o.name for o in self.objects]

    def to_dict(self) -> dict[str, Any]:
        """The PRD's observation JSON plus what the robot knows about the frame."""
        return {
            "id": self.id,
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(timespec="seconds"),
            "location": self.location,
            "description": self.description,
            "objects": [o.to_json() for o in self.objects],
            "confidence": self.confidence,
            "people": self.people,
            "cameraPan": self.camera_pan,
            "cameraTilt": self.camera_tilt,
            "imagePath": self.image_path,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Observation:
        """Rebuild from `to_dict` output (or the bare PRD example); raises InvalidObservation."""
        if not isinstance(raw, Mapping):
            raise InvalidObservation("an observation must be an object")
        fields = _validate_content(raw)
        source = raw.get("source", "vision")
        if source not in SOURCES:
            raise InvalidObservation(f"unknown source {source!r}")
        return cls(
            **fields,
            location=_location(raw.get("location")),
            timestamp=_timestamp(raw.get("timestamp")),
            id=_obs_id(raw.get("id")),
            camera_pan=_number(raw.get("cameraPan", 0.0), "cameraPan"),
            camera_tilt=_number(raw.get("cameraTilt", 0.0), "cameraTilt"),
            image_path=_optional_str(raw.get("imagePath"), "imagePath"),
            source=source,
        )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(value: Any, label: str) -> float:
    if not _is_number(value) or not math.isfinite(value):
        raise InvalidObservation(f"{label} must be a finite number")
    return float(value)


def _unit(value: Any, label: str) -> float:
    number = _number(value, label)
    if not 0.0 <= number <= 1.0:
        raise InvalidObservation(f"{label} must be between 0 and 1, got {number}")
    return number


def _optional_str(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidObservation(f"{label} must be a string")
    return value


def _location(value: Any) -> str | None:
    text = _optional_str(value, "location")
    return (normalise(text) or None) if text else None


def _timestamp(value: Any) -> float:
    if value is None:
        return time.time()
    if _is_number(value):
        return _number(value, "timestamp")
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            pass
    raise InvalidObservation("timestamp must be an ISO date or a number")


def _obs_id(value: Any) -> str:
    if value is None:
        return f"obs-{uuid.uuid4().hex[:8]}"
    if not isinstance(value, str) or not re.fullmatch(r"obs-[0-9a-f]{1,32}", value):
        raise InvalidObservation("id must look like obs-<hex>")
    return value


def _seen_object(raw: Any, default_confidence: float = 1.0) -> SeenObject:
    if isinstance(raw, str):
        name, confidence, position, note = raw, default_confidence, None, None
    elif isinstance(raw, Mapping):
        if "name" not in raw:
            raise InvalidObservation("an object needs a name")
        name = raw["name"]
        confidence = _unit(raw.get("confidence", default_confidence), "object confidence")
        position = raw.get("position")
        note = raw.get("note")
        extra = set(raw) - {"name", "confidence", "position", "note"}
        if extra:
            log.debug("ignoring unknown object keys: %s", sorted(extra))
    else:
        raise InvalidObservation("each object must be a name or an object")
    if not isinstance(name, str):
        raise InvalidObservation("an object name must be a string")
    name = normalise(name)
    if not 1 <= len(name) <= MAX_NAME_CHARS:
        raise InvalidObservation(f"an object name must have 1 to {MAX_NAME_CHARS} characters")
    if position is not None and position not in POSITIONS:
        raise InvalidObservation(f"unknown position {position!r}, expected one of {', '.join(POSITIONS)}")
    if note is not None:
        if not isinstance(note, str):
            raise InvalidObservation("an object note must be a string")
        note = " ".join(note.split()) or None
        if note is not None and len(note) > MAX_NOTE_CHARS:
            raise InvalidObservation(f"an object note is longer than {MAX_NOTE_CHARS} characters")
    return SeenObject(name, confidence, position, note)


def _validate_content(raw: Mapping[str, Any], *, inherit_confidence: bool = False) -> dict[str, Any]:
    """Check description, objects, confidence and people; the shared core of every parser.

    With `inherit_confidence` (fresh model output), an object without its own confidence
    gets the observation's, so a doubtful guess never becomes a certainty. Stored
    observations keep 1.0, which is how `SeenObject.to_dict` serialises plain names.
    """
    missing = [key for key in ("description", "objects", "confidence") if key not in raw]
    if missing:
        raise InvalidObservation(f"missing keys: {', '.join(missing)}")
    description = raw["description"]
    if not isinstance(description, str):
        raise InvalidObservation("description must be a string")
    description = " ".join(description.split())
    if not description:
        raise InvalidObservation("description is empty")
    if len(description) > MAX_DESCRIPTION_CHARS:
        raise InvalidObservation(f"description is longer than {MAX_DESCRIPTION_CHARS} characters")
    items = raw["objects"]
    if not isinstance(items, (list, tuple)):
        raise InvalidObservation("objects must be a list")
    if len(items) > MAX_OBJECTS:
        raise InvalidObservation(f"too many objects: {len(items)} (limit {MAX_OBJECTS})")
    confidence = _unit(raw["confidence"], "confidence")
    default = confidence if inherit_confidence else 1.0
    objects: dict[str, SeenObject] = {}
    for item in items:
        seen = _seen_object(item, default)
        objects.setdefault(seen.name, seen)
    people = raw.get("people", 0)
    if isinstance(people, bool) or not isinstance(people, int) or not 0 <= people <= MAX_PEOPLE:
        raise InvalidObservation(f"people must be a whole number from 0 to {MAX_PEOPLE}")
    return {
        "description": description,
        "objects": tuple(objects.values()),
        "confidence": confidence,
        "people": people,
    }


def _load_json(text: str) -> Any:
    if len(text) > MAX_MODEL_TEXT_CHARS:
        raise InvalidObservation("model output is too long")
    body = text.strip()
    fenced = _FENCE.match(body)
    if fenced:
        body = fenced.group(1).strip()
    try:
        return json.loads(body)
    except (ValueError, RecursionError) as error:
        raise InvalidObservation(f"not valid JSON: {error}") from None


def parse_model_observation(
    output: str | Mapping[str, Any], *, image: ImageObservation, location: str | None
) -> Observation:
    """Validate untrusted model output into an Observation for `image`, or raise InvalidObservation.

    The model only supplies description, objects, confidence and optionally people;
    time, place, camera pose and the saved image come from the robot, never the model.
    """
    raw = _load_json(output) if isinstance(output, str) else output
    if not isinstance(raw, Mapping):
        raise InvalidObservation("model output must be a JSON object")
    fields = _validate_content(raw, inherit_confidence=True)
    extra = set(raw) - {"description", "objects", "confidence", "people"}
    if extra:
        log.debug("ignoring unknown observation keys: %s", sorted(extra))
    return Observation(
        **fields,
        location=_location(location),
        timestamp=image.timestamp,
        camera_pan=image.camera_pan,
        camera_tilt=image.camera_tilt,
        source="vision",
    )


def observation_from_scene(image: ImageObservation, location: str | None) -> Observation:
    """An observation from the simulator's ground truth: certain, and people counted apart."""
    people = set(image.people)
    names = [normalise(n) for n in image.scene if n not in people]
    names = [n for n in dict.fromkeys(names) if 1 <= len(n) <= MAX_NAME_CHARS][:MAX_OBJECTS]
    seen = [*names, *image.people]
    description = f"I can see {', '.join(seen)}." if seen else "I can't see anything recognisable."
    return Observation(
        description=description[:MAX_DESCRIPTION_CHARS],
        objects=tuple(SeenObject(n) for n in names),
        confidence=1.0,
        location=_location(location),
        people=min(len(people), MAX_PEOPLE),
        timestamp=image.timestamp,
        camera_pan=image.camera_pan,
        camera_tilt=image.camera_tilt,
        source="simulator",
    )


class ObservationLog:
    """The latest observations and their interest scores, bounded and thread-safe. Persistence comes with #32."""

    def __init__(self, max_items: int = LOG_SIZE) -> None:
        self._items: deque[tuple[Observation, InterestScore | None]] = deque(maxlen=max(1, max_items))
        self._lock = threading.Lock()

    def add(self, observation: Observation, score: InterestScore | None = None) -> None:
        with self._lock:
            self._items.append((observation, score))

    def recent(self, n: int = 5) -> list[Observation]:
        """The newest `n`, newest first."""
        with self._lock:
            return [observation for observation, _ in reversed(self._items)][: max(0, n)]

    def last(self) -> Observation | None:
        with self._lock:
            return self._items[-1][0] if self._items else None

    def get(self, observation_id: str) -> Observation | None:
        """A logged observation by id, if it is still in the log."""
        with self._lock:
            for observation, _ in reversed(self._items):
                if observation.id == observation_id:
                    return observation
        return None

    def score_of(self, observation_id: str) -> InterestScore | None:
        """The interest score kept with a logged observation, if it is still in the log."""
        with self._lock:
            for observation, score in reversed(self._items):
                if observation.id == observation_id:
                    return score
        return None

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


# Seam for #22: given a frame and the room, ask a vision model; None when it cannot.
Describe = Callable[[ImageObservation, "str | None"], "Observation | None"]


@dataclass(frozen=True)
class _Job:
    image: ImageObservation
    location: str | None
    path: str | None


class Observer:
    """Turns captured frames into validated observations on one worker thread.

    The queue is small and drops the oldest frame when full. Frames without ground
    truth produce nothing until a `describe` callable (the vision call) is given.
    """

    def __init__(
        self,
        bus: EventBus | None,
        observations: ObservationLog | None = None,
        *,
        describe: Describe | None = None,
        default_location: Callable[[], str | None] | None = None,
        queue_size: int = QUEUE_SIZE,
        scorer: InterestScorer | None = None,
        gate: InterestGate | None = None,
    ) -> None:
        self.bus = bus
        self.log = observations if observations is not None else ObservationLog()
        self.describe = describe
        self.default_location = default_location
        self.queue_size = max(1, queue_size)
        # Curiosity (#28): the scorer runs before ObservationCreated, the gate after it.
        self.scorer = scorer
        self.gate = gate
        # Recent rejections as reasons, for logs and tests.
        self.rejections: deque[str] = deque(maxlen=REJECTION_HISTORY)
        self.dropped = 0
        self._cond = threading.Condition()
        self._queue: deque[_Job] = deque()
        self._current: _Job | None = None
        self._done: deque[tuple[ImageObservation, Observation | None]] = deque(maxlen=DONE_HISTORY)
        self._thread: threading.Thread | None = None
        self._closed = False

    def attach(self, camera: Any) -> None:
        camera.on_capture.append(self.submit)

    def submit(
        self, image: ImageObservation, location: str | None = None, path: Path | str | None = None
    ) -> None:
        """Queue a frame. Never blocks and never raises."""
        if not (image.simulated or self.describe is not None):
            return
        try:
            job = _Job(image, location, str(path) if path else None)
            with self._cond:
                if self._closed:
                    return
                if len(self._queue) >= self.queue_size:
                    old = self._queue.popleft()
                    self.dropped += 1
                    self._done.append((old.image, None))
                self._queue.append(job)
                if self._thread is None:
                    self._thread = threading.Thread(target=self._run, name="observer", daemon=True)
                    self._thread.start()
                self._cond.notify_all()
        except Exception:  # noqa: BLE001 - observing must never break the camera
            log.exception("could not queue a frame for observation")

    def wait_for(self, image: ImageObservation, timeout_s: float = 2.0) -> Observation | None:
        """The observation made from `image`, or None if there is none (yet) after `timeout_s`."""
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._cond:
            while True:
                for done_image, observation in self._done:
                    if done_image is image:
                        return observation
                waiting = self._current is not None and self._current.image is image
                waiting = waiting or any(job.image is image for job in self._queue)
                remaining = deadline - time.monotonic()
                if not waiting or remaining <= 0:
                    return None
                self._cond.wait(remaining)

    def close(self, timeout_s: float = 2.0) -> None:
        with self._cond:
            self._closed = True
            self._queue.clear()
            self._cond.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout_s)

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._closed:
                    self._cond.wait()
                if self._closed:
                    return
                job = self._current = self._queue.popleft()
            observation = self._process(job)
            with self._cond:
                self._done.append((job.image, observation))
                self._current = None
                self._cond.notify_all()

    def _process(self, job: _Job) -> Observation | None:
        location = job.location
        if location is None and self.default_location is not None:
            location = self.default_location()
        try:
            if job.image.simulated:
                observation = observation_from_scene(job.image, location)
            else:
                if self.describe is None:
                    return None
                observation = self.describe(job.image, location)
                if observation is None:
                    return None
            if job.path and observation.image_path is None:
                observation = replace(observation, image_path=job.path)
        except InvalidObservation as error:
            self.rejections.append(str(error))
            print(f"[laika] observation rejected: {error}", flush=True)
            return None
        except Exception:  # noqa: BLE001 - a failing vision call must not kill the worker
            log.exception("could not make an observation")
            return None
        score = self._score(observation)
        self.log.add(observation, score)
        data = observation.to_dict()
        if score is not None:
            data["interest"] = score.to_dict()
        emit(self.bus, EventType.OBSERVATION_CREATED, **data)
        if score is not None and self.gate is not None:
            self._gate(observation, score)
        return observation

    def _score(self, observation: Observation) -> InterestScore | None:
        if self.scorer is None:
            return None
        previous = find_previous(self.log.recent(LOG_SIZE), observation)
        try:
            return self.scorer.score(observation, previous=previous)
        except Exception:  # noqa: BLE001 - scoring must never drop the observation
            log.exception("could not score an observation")
            return None

    def _gate(self, observation: Observation, score: InterestScore) -> None:
        assert self.gate is not None
        try:
            self.gate.consider(observation, score)
        except Exception:  # noqa: BLE001 - announcing must never break the worker
            log.exception("could not announce an interesting observation")
