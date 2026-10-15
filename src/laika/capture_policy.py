"""When Laika takes pictures on her own, and how often.

The agent's `capture_image` tool goes straight to the camera; this policy only adds
the automatic captures: after a completed move, when a person comes into view, when
something interesting changes, during an investigation, on a timer while exploring and, so she can
notice someone walking in (#38), on a slow idle "presence" timer.
Every automatic frame may become a vision request, so each trigger has an on/off
switch and a cooldown, and all automatic triggers share one per-minute cap.

Captures run on the policy's own worker thread. Triggers are only recorded in a
one-slot pending slot (a newer trigger replaces an older one), so a burst of events
never piles up and the event bus or the safety monitor never waits for the camera.

Presence pictures: every `presence_interval_s` while nobody is known to be near, every
`presence_check_s` while someone is (to notice they left), and none while a conversation is
active. They share the per-minute cap and the camera and vision limits like every other trigger.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable

from laika.camera import Camera
from laika.config import CaptureConfig, PresenceConfig
from laika.events import Event, EventBus, EventType

log = logging.getLogger(__name__)

EXPLORATION = "exploration"
WINDOW_S = 60.0
CLOSE_TIMEOUT_S = 2.0
EVENT_TRIGGERS = {
    EventType.MOVEMENT_COMPLETED: "after_movement",
    EventType.PERSON_DETECTED: "person",
    EventType.INTERESTING_CHANGE_DETECTED: "interesting",
}
PRESENCE = "presence"
# The presence timer is re-read at least this often, so a change of state is noticed without an event.
MAX_WAIT_S = 1.0
TRIGGERS = ("after_movement", "person", "interesting", "investigate", EXPLORATION, PRESENCE)


@dataclass
class TriggerStats:
    captured: int = 0
    # The camera gave back the previous frame because of its own interval or cap (not counted as a capture).
    reused: int = 0
    failed: int = 0
    skipped_cooldown: int = 0
    skipped_rate: int = 0
    disabled: int = 0
    # A newer trigger took the pending slot before this one ran.
    replaced: int = 0


class CapturePolicy:
    def __init__(
        self,
        camera: Camera,
        events: EventBus,
        config: CaptureConfig,
        *,
        presence: PresenceConfig | None = None,
        person_present: Callable[[], bool] | None = None,
        conversation_active: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.camera = camera
        self.events = events
        self.config = config
        # Without a presence config the idle presence pictures are off.
        self.presence = presence
        self._person_present = person_present or (lambda: False)
        self._conversation_active = conversation_active or (lambda: False)
        self._clock = clock
        self._cond = threading.Condition()
        self._pending: str | None = None
        self._busy = False
        self._closed = False
        self._exploring = False
        self._next_exploration = 0.0
        self._last_presence = 0.0
        self._thread: threading.Thread | None = None
        self._unsubscribe: Callable[[], None] | None = None
        # Only the worker thread touches these two.
        self._last_capture: dict[str, float] = {}
        self._window: deque[float] = deque()
        self.stats: dict[str, TriggerStats] = {name: TriggerStats() for name in TRIGGERS}

    def start(self) -> None:
        with self._cond:
            if self._thread is not None or self._closed:
                return
            self._last_presence = self._clock()
            self._thread = threading.Thread(target=self._run, name="capture-policy", daemon=True)
            self._thread.start()
        self._unsubscribe = self.events.subscribe(self._on_event, *EVENT_TRIGGERS)

    def close(self, timeout_s: float = CLOSE_TIMEOUT_S) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        with self._cond:
            self._closed = True
            self._pending = None
            self._exploring = False
            self._cond.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout_s)
        log.info("capture policy stopped: %s", {name: asdict(s) for name, s in self.stats.items()})

    def investigate(self) -> None:
        """Ask for a capture during an investigation (#29). Not blocked by cooldowns, still capped."""
        self._offer("investigate")

    def set_exploring(self, exploring: bool) -> None:
        """While exploring (#27), one capture every `exploration_interval_s`."""
        with self._cond:
            if self._closed:
                return
            if exploring and not self._exploring:
                self._next_exploration = self._clock() + self.config.exploration_interval_s
            if not exploring and self._pending == EXPLORATION:
                self._pending = None
            self._exploring = bool(exploring)
            self._cond.notify_all()

    def flush(self, timeout_s: float = 2.0) -> bool:
        """Wait until no trigger is pending or running (for tests). False on timeout."""
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while self._pending is not None or self._busy:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
        return True

    def _on_event(self, event: Event) -> None:
        if event.data.get("source") == "outside":
            return  # something changed out of the window (#48): there is nothing in the room to photograph
        self._offer(EVENT_TRIGGERS[event.type])

    def _offer(self, reason: str) -> None:
        """Record a trigger without waiting for the camera. Never raises."""
        with self._cond:
            if self._closed:
                return
            if not self._switched_on(reason):
                self.stats[reason].disabled += 1
                return
            if self._pending is not None:
                self.stats[self._pending].replaced += 1
            self._pending = reason
            self._cond.notify_all()

    def _switched_on(self, reason: str) -> bool:
        config = self.config
        if reason == "after_movement":
            return config.after_movement
        if reason == "person":
            return config.on_person
        if reason == "interesting":
            return config.on_interesting
        return True

    def _cooldown_s(self, reason: str) -> float:
        config = self.config
        return {
            "after_movement": config.after_movement_cooldown_s,
            "person": config.person_cooldown_s,
            "interesting": config.interesting_cooldown_s,
        }.get(reason, 0.0)

    def _exploration_due(self) -> bool:
        return self._exploring and self._clock() >= self._next_exploration

    def _presence_interval_s(self) -> float:
        """Seconds between presence pictures right now; 0 when they are off."""
        if self.presence is None:
            return 0.0
        if self._person_present():
            return self.presence.presence_check_s
        return self.presence.presence_interval_s

    def _presence_due(self) -> bool:
        interval = self._presence_interval_s()
        return interval > 0 and self._clock() - self._last_presence >= interval

    def _wait_s(self) -> float | None:
        """How long the worker may sleep: until the next timer tick, or until woken."""
        waits = []
        if self._exploring:
            waits.append(max(0.0, self._next_exploration - self._clock()))
        interval = self._presence_interval_s()
        if interval > 0:
            waits.append(min(MAX_WAIT_S, max(0.0, self._last_presence + interval - self._clock())))
        elif self.presence is not None and (
            self.presence.presence_interval_s > 0 or self.presence.presence_check_s > 0
        ):
            waits.append(MAX_WAIT_S)  # this state is off but the other one is on: look again soon
        return min(waits) if waits else None

    def _run(self) -> None:
        while True:
            with self._cond:
                while (
                    not self._closed
                    and self._pending is None
                    and not self._exploration_due()
                    and not self._presence_due()
                ):
                    self._cond.wait(self._wait_s())
                if self._closed:
                    return
                if self._pending is not None:
                    reason, self._pending = self._pending, None
                elif self._exploration_due():
                    reason = EXPLORATION
                    self._next_exploration = self._clock() + self.config.exploration_interval_s
                else:
                    reason = PRESENCE
                    self._last_presence = self._clock()
                    if self._conversation_active():
                        continue  # nobody should be photographed mid-conversation; try again later
                self._busy = True
            try:
                self._capture(reason)
            except Exception:  # noqa: BLE001 - a failing capture must not stop the worker
                log.exception("automatic capture (%s) failed", reason)
            finally:
                with self._cond:
                    self._busy = False
                    self._cond.notify_all()

    def _capture(self, reason: str) -> None:
        stats = self.stats[reason]
        now = self._clock()
        last = self._last_capture.get(reason)
        if last is not None and now - last < self._cooldown_s(reason):
            stats.skipped_cooldown += 1
            return
        while self._window and now - self._window[0] >= WINDOW_S:
            self._window.popleft()
        if len(self._window) >= self.config.max_auto_per_minute:
            stats.skipped_rate += 1
            return
        previous = self.camera.last
        image = self.camera.capture(reason)
        if image is None:
            stats.failed += 1
        elif image is previous:
            stats.reused += 1
        else:
            self._last_capture[reason] = now
            self._window.append(now)
            with self._cond:
                self._last_presence = now  # a fresh frame is as good as a presence check
            stats.captured += 1
            log.debug("automatic capture: %s", reason)
