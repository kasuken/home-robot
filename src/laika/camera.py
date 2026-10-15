"""The camera service: the one place frames are requested, saved and announced.

Callers (the `capture_image` tool, later vision and exploration) never talk to
`robot.capture_image()` directly. This service keeps the camera on request only:
a minimum interval between real captures and a per-minute cap mean a model that
loops cannot turn the camera into continuous video or fill the SD card. A request
that hits a limit gets the previous observation back instead of a new frame.
Hardware errors never reach the caller: it gets None and one HardwareUnavailable event.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

from laika.events import EventBus, EventType, emit
from laika.robot import ImageObservation, Robot
from laika.storage import ImageStore

log = logging.getLogger(__name__)

DEFAULT_MIN_INTERVAL_S = 0.5
DEFAULT_MAX_PER_MINUTE = 30
RATE_WINDOW_S = 60.0


class Camera:
    def __init__(
        self,
        robot: Robot,
        store: ImageStore | None = None,
        events: EventBus | None = None,
        *,
        min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
        max_per_minute: int = DEFAULT_MAX_PER_MINUTE,
        save_images: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.robot = robot
        self.store = store
        self.events = events
        self.min_interval_s = max(0.0, min_interval_s)
        self.max_per_minute = max_per_minute  # 0 or less: no cap
        self.save_images = save_images
        self._clock = clock
        self._lock = threading.Lock()
        self._times: deque[float] = deque()
        self._last_time: float | None = None
        self._hardware_ok = True
        # Only the latest observation is kept in memory; older frames live on disk.
        self.last: ImageObservation | None = None
        self.last_path: Path | None = None
        # True when the latest `capture()` returned the previous frame because of a limit.
        self.reused = False

    def capture(self, reason: str = "request") -> ImageObservation | None:
        """One frame, or the previous one when asked too soon, or None when the camera failed."""
        with self._lock:
            self.reused = False
            now = self._clock()
            if self.last is not None and self._limited(now):
                self.reused = True
                return self.last
            try:
                observation = self.robot.capture_image()
            except Exception as error:  # noqa: BLE001 - the camera must never break the agent
                log.warning("camera failed: %s", error)
                if self._hardware_ok:
                    self._hardware_ok = False
                    emit(
                        self.events,
                        EventType.HARDWARE_UNAVAILABLE,
                        error=str(error),
                        device="camera",
                    )
                return None
            self._hardware_ok = True
            self._last_time = now
            self._times.append(now)
            path = None
            if self.save_images and self.store is not None:
                path = self.store.save(observation)
            self.last = observation
            self.last_path = path
            data = {
                "path": str(path) if path else None,
                "width": observation.width,
                "height": observation.height,
                "pan": observation.camera_pan,
                "tilt": observation.camera_tilt,
                "scene": list(observation.scene),
                "reason": reason,
            }
            current_room = getattr(self.robot, "current_room", None)
            if callable(current_room):
                data["room"] = current_room()
            emit(self.events, EventType.IMAGE_CAPTURED, **data)
            return observation

    def _limited(self, now: float) -> bool:
        if self._last_time is not None and now - self._last_time < self.min_interval_s:
            return True
        while self._times and now - self._times[0] >= RATE_WINDOW_S:
            self._times.popleft()
        return self.max_per_minute > 0 and len(self._times) >= self.max_per_minute
