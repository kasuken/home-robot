"""AprilTag position fixes: a printed tag at a doorway tells her, for sure, where she is.

Laika has no odometry, so nothing in her knows where she is except what the vision model guesses. A tag is the
opposite of a guess: a printed black-and-white square (family tag36h11, see `docs/apriltags.md`) that a
detector reads locally, with its id, from the pixels. The plan of the house (`floorplan.py`) says where each
id is: which doorway it marks, and which room she is in when she sees it (`PlanTag`). Seeing one near enough
makes that room certain (`RoomTracker.tag_fix`), the same way a person's "questa è la cucina" does.

Nothing leaves the robot: the detector runs on the frames the camera already captured, on its own worker
thread, one frame at a time (an older waiting frame is dropped). It never touches the wheels or the safety
monitor. A tag far away, small or oblique is ignored: `max_fix_cm` bounds how far a tag counts.

The detector is OpenCV's ArUco module (`cv2.aruco`, dictionary AprilTag 36h11), imported lazily: Laika runs
without it (no tag fixes) and the tests inject a fake. The distance is a pinhole estimate from the tag's
printed size and the camera's field of view, so it is only as good as `[tags] tag_size_cm` and `fov_deg`
(a guess for the PiCar-X camera until calibrated; docs/apriltags.md says how to check).
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol, Sequence

if TYPE_CHECKING:
    from laika.config import TagsConfig
    from laika.floorplan import FloorPlan
    from laika.robot import ImageObservation
    from laika.rooms import RoomTracker

log = logging.getLogger(__name__)

Corners = Sequence[tuple[float, float]]


@dataclass(frozen=True)
class RawTag:
    """What a detector found: the id and the four corners in pixels."""

    tag_id: int
    corners: Corners


@dataclass(frozen=True)
class TagSighting:
    tag_id: int
    width_px: float
    bearing_deg: float  # positive: to the right of the camera's axis
    distance_cm: float  # the pinhole estimate


class TagDetector(Protocol):
    def detect(self, jpeg: bytes) -> list[RawTag]: ...


class OpenCVTagDetector:
    """tag36h11 with OpenCV. Raises ImportError when OpenCV (4.7 or newer) is not installed."""

    def __init__(self) -> None:
        import cv2  # lazy: only the robot needs it
        import numpy

        self._cv2 = cv2
        self._numpy = numpy
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self._detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    def detect(self, jpeg: bytes) -> list[RawTag]:
        gray = self._cv2.imdecode(self._numpy.frombuffer(jpeg, self._numpy.uint8), self._cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return []
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return []
        return [
            RawTag(int(tag_id), [(float(x), float(y)) for x, y in corner.reshape(4, 2)])
            for tag_id, corner in zip(ids.flatten(), corners)
        ]


def sighting(raw: RawTag, image_width: int, fov_deg: float, tag_cm: float) -> TagSighting | None:
    """Width, bearing and distance of a detected tag, or None when its corners make no sense."""
    if len(raw.corners) != 4 or image_width <= 0:
        return None
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = raw.corners
    width_px = (math.hypot(x1 - x0, y1 - y0) + math.hypot(x2 - x3, y2 - y3)) / 2
    if width_px < 1:
        return None
    focal_px = (image_width / 2) / math.tan(math.radians(fov_deg) / 2)
    centre_x = (x0 + x1 + x2 + x3) / 4
    bearing = math.degrees(math.atan2(centre_x - image_width / 2, focal_px))
    return TagSighting(raw.tag_id, width_px, bearing, tag_cm * focal_px / width_px)


class TagReader:
    """Reads the tags in captured frames and turns the near ones into room fixes."""

    def __init__(
        self,
        plan: FloorPlan,
        rooms: RoomTracker,
        detector: TagDetector,
        config: TagsConfig,
        *,
        on_sighting: Callable[[TagSighting], None] | None = None,
    ) -> None:
        self.plan = plan
        self.rooms = rooms
        self.detector = detector
        self.config = config
        self.on_sighting = on_sighting
        self.frames = 0
        self.fixes = 0
        self.ignored = 0  # tags the plan does not know, or that were too far
        self.last: list[TagSighting] = []
        self._cond = threading.Condition()
        self._queue: deque[ImageObservation] = deque(maxlen=1)  # an older waiting frame is dropped
        self._thread: threading.Thread | None = None
        self._closed = False

    def attach(self, camera: Any) -> None:
        camera.on_capture.append(self.submit)

    def submit(self, image: ImageObservation, location: str | None = None, path: Any = None) -> None:
        """Queue a frame. Never blocks and never raises."""
        if image.jpeg is None:
            return
        with self._cond:
            if self._closed:
                return
            self._queue.append(image)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="laika-tags", daemon=True)
                self._thread.start()
            self._cond.notify()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        thread = self._thread
        if thread is not None:
            thread.join(2.0)

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._closed:
                    self._cond.wait()
                if self._closed:
                    return
                image = self._queue.popleft()
            try:
                self.read(image)
            except Exception:  # noqa: BLE001 - a bad frame must not stop the reader
                log.exception("tag reading failed")

    def read(self, image: ImageObservation) -> list[TagSighting]:
        """Detect the tags in one frame and apply the fixes. Called on the worker thread (and by tests)."""
        assert image.jpeg is not None
        self.frames += 1
        seen = [
            s for raw in self.detector.detect(image.jpeg)
            if (s := sighting(raw, image.width, self.config.fov_deg, self.config.tag_size_cm)) is not None
        ]
        self.last = seen
        for tag in sorted(seen, key=lambda s: s.distance_cm):
            known = self.plan.tags.get(tag.tag_id)
            if known is None or tag.distance_cm > self.config.max_fix_cm:
                self.ignored += 1
                continue
            if self.on_sighting is not None:
                self.on_sighting(tag)
            if self.rooms.tag_fix(known.seen_from, image.timestamp):
                self.fixes += 1
                log.info("tag %d at %.0f cm: she is in the %s", tag.tag_id, tag.distance_cm, known.seen_from)
            break  # the nearest known tag decides
        return seen
