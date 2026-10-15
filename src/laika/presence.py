"""Knowing when a person is near (#38), from a head count only.

`PresenceTracker` listens to `ObservationCreated`. A frame whose observation counts people and
whose confidence reaches `min_confidence` marks "someone is here" and publishes `PersonDetected`
on the change from absent to present. `PersonLost` follows when no such frame arrived for
`lost_after_s`: checked on every observation and by a small timer thread, so it fires even when
no more frames come. A frame that shows nobody does not end presence by itself (one blurry
frame must not flap), and a low-confidence frame is ignored altogether.

Privacy: the tracker keeps a count, the room and where the head pointed. No faces, names or
images. The vision prompt already forbids identifying anyone (docs/behaviours.md, "People").
The interaction policy (#39) reads `present`, `count`, `last_seen` and `since`.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

from laika.config import PresenceConfig
from laika.events import Event, EventBus, EventType, emit

log = logging.getLogger(__name__)

MAX_TIMER_WAIT_S = 1.0
CLOSE_TIMEOUT_S = 2.0


class PresenceTracker:
    def __init__(
        self,
        events: EventBus,
        config: PresenceConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.events = events
        self.config = config
        self._clock = clock
        self._cond = threading.Condition()
        self._present = False
        self._count = 0
        self._last_seen: float | None = None
        self._since: float | None = None  # clock time of the arrival
        self._since_wall: float | None = None
        self._confidence = 0.0
        self._room: str | None = None
        self._pan = 0.0
        self._tilt = 0.0
        self._closed = False
        self._thread: threading.Thread | None = None
        self._unsubscribe: Callable[[], None] | None = None

    # --- state for the interaction policy ---------------------------------------

    @property
    def present(self) -> bool:
        with self._cond:
            return self._present

    @property
    def count(self) -> int:
        """People in the latest frame that showed any; 0 when nobody is known to be near."""
        with self._cond:
            return self._count if self._present else 0

    @property
    def last_seen(self) -> float | None:
        """Clock time of the latest frame that showed a person (kept after they left)."""
        with self._cond:
            return self._last_seen

    @property
    def since(self) -> float | None:
        """Clock time the current visit began; None when nobody is near."""
        with self._cond:
            return self._since if self._present else None

    # --- lifecycle ---------------------------------------------------------------

    def start(self) -> None:
        with self._cond:
            if self._thread is not None or self._closed:
                return
            self._thread = threading.Thread(target=self._run, name="presence", daemon=True)
            self._thread.start()
        self._unsubscribe = self.events.subscribe(self._on_event, EventType.OBSERVATION_CREATED)

    def close(self, timeout_s: float = CLOSE_TIMEOUT_S) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        with self._cond:
            self._closed = True
            self._cond.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout_s)

    # --- updates -----------------------------------------------------------------

    def _on_event(self, event: Event) -> None:
        self.observe(event.data)

    def observe(self, data: dict[str, Any] | Any) -> None:
        """Take one observation (its `to_dict` shape). Never raises."""
        try:
            people = int(data.get("people", 0) or 0)
            confidence = float(data.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError, AttributeError):
            return
        with self._cond:
            if self._closed:
                return
            now = self._clock()
            # First close a visit that already ran out, so a return after a long gap is a new arrival.
            self._expire(now)
            if people > 0 and confidence >= self.config.min_confidence:
                self._last_seen = now
                self._count = people
                self._confidence = confidence
                self._room = data.get("location")
                self._pan = float(data.get("cameraPan", 0.0) or 0.0)
                self._tilt = float(data.get("cameraTilt", 0.0) or 0.0)
                if not self._present:
                    self._present = True
                    self._since = now
                    self._since_wall = time.time()
                    # Publishing never blocks, so doing it under the lock keeps the events in order
                    # when the timer thread and the event thread race.
                    emit(self.events, EventType.PERSON_DETECTED, **self._data(now))
                    self._cond.notify_all()

    def check(self) -> None:
        """Publish PersonLost if the person has not been seen for `lost_after_s`."""
        with self._cond:
            self._expire(self._clock())

    def _expire(self, now: float) -> None:
        if not self._present or self._last_seen is None:
            return
        if now - self._last_seen < self.config.lost_after_s:
            return
        emit(self.events, EventType.PERSON_LOST, **self._data(now))
        self._present = False
        self._count = 0
        self._since = None

    def _data(self, now: float) -> dict[str, Any]:
        assert self._since is not None
        return {
            "count": self._count,
            "confidence": self._confidence,
            "room": self._room,
            "pan": self._pan,
            "tilt": self._tilt,
            "since": self._since_wall,
            "duration_s": round(max(0.0, now - self._since), 3),
        }

    def _run(self) -> None:
        """Wake when the person may have left; sleep until then, with no polling while nobody is near."""
        while True:
            with self._cond:
                while not self._closed and not self._present:
                    self._cond.wait()
                if self._closed:
                    return
                remaining = self.config.lost_after_s - (self._clock() - (self._last_seen or 0.0))
                if remaining > 0:
                    # Capped so a clock that is not the wall clock (tests) is still noticed.
                    self._cond.wait(min(remaining, MAX_TIMER_WAIT_S))
                    continue
                self._expire(self._clock())
