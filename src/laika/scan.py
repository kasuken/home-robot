"""Scanning the surroundings by moving the camera, never the wheels.

`Scanner.scan` looks left, centre and right, takes one frame at each position,
and merges what the observer made of them into one `EnvironmentScan`. All frames
are captured first and only then awaited, so the head keeps moving while the
vision model works on the earlier frames. The head returns to where it was, even
when something fails. The head goes through `robot.look`, the same primitive the
look tools use; the safety controller only guards wheels.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from laika.camera import Camera
from laika.observations import Observation, Observer
from laika.robot import ImageObservation, Robot

log = logging.getLogger(__name__)

# (pan, tilt): negative pan is left.
DEFAULT_POSITIONS: tuple[tuple[float, float], ...] = ((-60, 5), (0, 5), (60, 5))
DEFAULT_SETTLE_S = 0.6
DEFAULT_OBSERVATION_WAIT_S = 2.0
MAX_TOTAL_WAIT_S = 60.0
SIDE_THRESHOLD_DEG = 15.0
SCAN_REASON = "scan"


def label_for(pan: float) -> str:
    """The side of the robot a camera pan looks at."""
    if pan < -SIDE_THRESHOLD_DEG:
        return "left"
    if pan > SIDE_THRESHOLD_DEG:
        return "right"
    return "centre"


@dataclass(frozen=True)
class PositionResult:
    label: str
    pan: float
    tilt: float
    observation: Observation | None = None
    # Why there is no observation: "camera unavailable", "no observation", ...
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"position": self.label, "pan": self.pan, "tilt": self.tilt}
        if self.observation is None:
            data["note"] = self.reason or "no observation"
            return data
        data["description"] = self.observation.description
        data["objects"] = self.observation.object_names
        data["people"] = self.observation.people
        return data


@dataclass(frozen=True)
class MergedObject:
    name: str
    seen_at: tuple[str, ...]  # camera directions, in scan order
    confidence: float
    in_frame: tuple[str, ...] = ()  # positions the model gave inside the frame

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "seen_at": list(self.seen_at),
            "confidence": self.confidence,
        }
        if self.in_frame:
            data["in_frame"] = list(self.in_frame)
        return data


@dataclass(frozen=True)
class EnvironmentScan:
    positions: tuple[PositionResult, ...]
    objects: tuple[MergedObject, ...] = ()
    people: int = 0
    location: str | None = None
    confidence: float = 0.0
    summary: str = ""
    observed: int = 0  # how many positions produced an observation

    @property
    def ok(self) -> bool:
        return self.observed > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "location": self.location,
            "positions": [p.to_dict() for p in self.positions],
            "objects": [o.to_dict() for o in self.objects],
            "people": self.people,
            "confidence": self.confidence,
            "summary": self.summary,
        }


def merge(results: list[PositionResult]) -> EnvironmentScan:
    """One description from the per-position results. People are a max: one person can span frames."""
    seen: dict[str, dict[str, Any]] = {}
    observed = [(r.label, r.observation) for r in results if r.observation is not None]
    for label, observation in observed:
        for obj in observation.objects:
            entry = seen.setdefault(obj.name, {"at": [], "conf": 0.0, "in": []})
            if label not in entry["at"]:
                entry["at"].append(label)
            entry["conf"] = max(entry["conf"], obj.confidence)
            if obj.position and obj.position not in entry["in"]:
                entry["in"].append(obj.position)
    objects = tuple(
        MergedObject(name, tuple(e["at"]), e["conf"], tuple(e["in"])) for name, e in seen.items()
    )
    people = max((o.people for _, o in observed), default=0)
    locations = Counter(o.location for _, o in observed if o.location)
    location = locations.most_common(1)[0][0] if locations else None
    confidence = round(sum(o.confidence for _, o in observed) / len(observed), 3) if observed else 0.0
    return EnvironmentScan(
        positions=tuple(results),
        objects=objects,
        people=people,
        location=location,
        confidence=confidence,
        summary=_summary(results, location, people),
        observed=len(observed),
    )


def _summary(results: list[PositionResult], location: str | None, people: int) -> str:
    parts = []
    for result in results:
        if result.observation is None:
            parts.append(f"{result.label}: {result.reason or 'no observation'}")
        elif result.observation.objects:
            parts.append(f"{result.label}: {', '.join(result.observation.object_names)}")
        else:
            parts.append(f"{result.label}: nothing recognised")
    text = "; ".join(parts) + "."
    if people:
        text += f" People seen: {people}."
    if location:
        text += f" Location: {location.replace('_', ' ')}."
    return text


class ScanBusy(RuntimeError):
    """A scan is already running."""


class Scanner:
    def __init__(
        self,
        robot: Robot,
        camera: Camera,
        observer: Observer | None = None,
        *,
        positions: tuple[tuple[float, float], ...] = DEFAULT_POSITIONS,
        settle_s: float = DEFAULT_SETTLE_S,
        observation_wait_s: float = DEFAULT_OBSERVATION_WAIT_S,
        max_total_wait_s: float = MAX_TOTAL_WAIT_S,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.robot = robot
        self.camera = camera
        self.observer = observer
        self.positions = positions
        self.settle_s = max(0.0, settle_s)
        self.observation_wait_s = max(0.0, observation_wait_s)
        self.max_total_wait_s = max_total_wait_s
        self._sleep = sleep
        self._lock = threading.Lock()

    def scan(self) -> EnvironmentScan:
        """Look at each position, then merge. Raises ScanBusy if a scan is already running."""
        if not self._lock.acquire(blocking=False):
            raise ScanBusy("a scan is already running")
        try:
            return self._scan()
        finally:
            self._lock.release()

    def _scan(self) -> EnvironmentScan:
        original = self.robot.head_pose()
        frames: list[tuple[float, float, ImageObservation | None]] = []
        try:
            for pan, tilt in self.positions:
                self.robot.look(pan, tilt)
                self._sleep(self.settle_s)
                frames.append((pan, tilt, self._fresh_frame()))
            results = self._observe(frames)
        finally:
            try:
                self.robot.look(*original)
            except Exception:  # noqa: BLE001 - restoring must not hide the original error
                log.exception("could not restore the head after a scan")
        scan = merge(results)
        log.info("environment scanned: %s", scan.summary)
        return scan

    def _fresh_frame(self) -> ImageObservation | None:
        """A new frame, waiting once for the camera's minimum interval; None if it stays limited."""
        image = self.camera.capture(SCAN_REASON)
        if image is not None and self.camera.reused:
            self._sleep(max(self.camera.min_interval_s, self.settle_s))
            image = self.camera.capture(SCAN_REASON)
            if image is not None and self.camera.reused:
                return None
        return image

    def _observe(
        self, frames: list[tuple[float, float, ImageObservation | None]]
    ) -> list[PositionResult]:
        budget = min(self.observation_wait_s * len(frames), self.max_total_wait_s)
        deadline = time.monotonic() + budget
        results = []
        for pan, tilt, image in frames:
            if image is None:
                results.append(PositionResult(label_for(pan), pan, tilt, reason="camera unavailable"))
                continue
            observation = None
            if self.observer is not None:
                observation = self.observer.wait_for(image, max(0.0, deadline - time.monotonic()))
            results.append(
                PositionResult(
                    label_for(image.camera_pan),
                    image.camera_pan,
                    image.camera_tilt,
                    observation,
                    None if observation else "no observation",
                )
            )
        return results
