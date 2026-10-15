"""Local event dispatcher: robot activity publishes events, the agent reacts.

One daemon thread delivers events to subscribers in publish order. `publish`
only appends to a bounded queue under a short lock, so the safety layer can
publish without ever waiting for a subscriber. Because a single thread serves
all subscribers, a slow subscriber delays the other subscribers (and makes the
queue drop its oldest events) but never the publisher. Subscribers must not
be needed for safety: safety decisions are made before the event is published.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

log = logging.getLogger(__name__)

DEFAULT_QUEUE_SIZE = 256
CLOSE_TIMEOUT_S = 2.0


class EventType(str, Enum):
    HARDWARE_READY = "HardwareReady"
    HARDWARE_UNAVAILABLE = "HardwareUnavailable"
    MOVEMENT_STARTED = "MovementStarted"
    MOVEMENT_COMPLETED = "MovementCompleted"
    MOVEMENT_FAILED = "MovementFailed"
    PERSON_DETECTED = "PersonDetected"
    PERSON_LOST = "PersonLost"
    IMAGE_CAPTURED = "ImageCaptured"
    OBSTACLE_DETECTED = "ObstacleDetected"
    CLIFF_DETECTED = "CliffDetected"
    ROOM_ENTERED = "RoomEntered"
    OBSERVATION_CREATED = "ObservationCreated"
    INTERESTING_CHANGE_DETECTED = "InterestingChangeDetected"
    INTERACTION_STARTED = "InteractionStarted"
    INTERACTION_COMPLETED = "InteractionCompleted"
    AI_ENDPOINT_UNAVAILABLE = "AIEndpointUnavailable"
    AI_ENDPOINT_RECOVERED = "AIEndpointRecovered"
    LOCAL_WATCHDOG_EXPIRED = "LocalWatchdogExpired"


def _new_event_id() -> str:
    return f"evt-{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True)
class Event:
    """`data` holds JSON-friendly values only, so events can go straight to logs."""

    type: EventType
    data: Mapping[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=_new_event_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "data": dict(self.data),
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Event:
        return cls(
            type=EventType(raw["type"]),
            data=dict(raw.get("data", {})),
            timestamp=float(raw["timestamp"]),
            id=str(raw["id"]),
        )


Handler = Callable[[Event], None]


class EventBus:
    def __init__(self, maxsize: int = DEFAULT_QUEUE_SIZE) -> None:
        self.maxsize = max(1, maxsize)
        self.dropped = 0
        self._queue: deque[Event] = deque()
        self._cond = threading.Condition()
        self._subscribers: list[tuple[Handler, frozenset[EventType]]] = []
        self._thread: threading.Thread | None = None
        self._closed = False
        self._delivering = False

    def subscribe(self, handler: Handler, *types: EventType) -> Callable[[], None]:
        """Call `handler` for the given types (all events if none). Returns an unsubscribe function."""
        entry = (handler, frozenset(types))
        with self._cond:
            self._subscribers = [*self._subscribers, entry]

        def unsubscribe() -> None:
            with self._cond:
                self._subscribers = [s for s in self._subscribers if s is not entry]

        return unsubscribe

    def publish(self, event: Event) -> None:
        """Queue an event. Never blocks on subscribers and never raises."""
        try:
            with self._cond:
                if self._closed:
                    return
                if len(self._queue) >= self.maxsize:
                    self._queue.popleft()
                    self.dropped += 1
                self._queue.append(event)
                self._cond.notify_all()
        except Exception:  # noqa: BLE001 - publishing must never break the caller
            log.exception("could not publish %s", getattr(event, "type", event))

    def emit(self, type: EventType, **data: Any) -> None:
        """Build and publish an event from keyword data."""
        self.publish(Event(type, data))

    def start(self) -> None:
        with self._cond:
            if self._thread is not None or self._closed:
                return
            self._thread = threading.Thread(target=self._run, name="event-bus", daemon=True)
            self._thread.start()

    def close(self, timeout_s: float = CLOSE_TIMEOUT_S) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout_s)

    def drain(self, timeout_s: float = 2.0) -> bool:
        """Wait until every queued event was delivered (for tests). False on timeout."""
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while self._queue or self._delivering:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
        return True

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._closed:
                    self._cond.wait()
                if not self._queue:
                    return
                event = self._queue.popleft()
                subscribers = self._subscribers
                self._delivering = True
            try:
                for handler, types in subscribers:
                    if types and event.type not in types:
                        continue
                    try:
                        handler(event)
                    except Exception:  # noqa: BLE001 - one bad handler must not stop delivery
                        log.exception("event handler failed for %s", event.type.value)
            finally:
                with self._cond:
                    self._delivering = False
                    self._cond.notify_all()


def emit(bus: EventBus | None, type: EventType, **data: Any) -> None:
    """Publish when a bus is configured; a no-op otherwise."""
    if bus is not None:
        bus.emit(type, **data)


def log_events_to_console(event: Event) -> None:
    print(f"[event] {event.to_json()}", flush=True)


class AIEndpointMonitor:
    """Turns request outcomes into AIEndpointUnavailable/Recovered transitions.

    Kept outside `ChatClient` so the AI client stays free of robot concerns.
    The optional callbacks run synchronously on the transition, before the
    event is published: safety reactions must not depend on the event bus.
    """

    def __init__(
        self,
        bus: EventBus | None,
        on_unavailable: Callable[[], None] | None = None,
        on_recovered: Callable[[], None] | None = None,
    ) -> None:
        self.bus = bus
        self.on_unavailable = on_unavailable
        self.on_recovered = on_recovered
        self.available = True
        self._lock = threading.Lock()

    def failed(self, error: object) -> None:
        with self._lock:
            if not self.available:
                return
            self.available = False
        _call(self.on_unavailable)
        emit(self.bus, EventType.AI_ENDPOINT_UNAVAILABLE, error=str(error))

    def succeeded(self) -> None:
        with self._lock:
            if self.available:
                return
            self.available = True
        _call(self.on_recovered)
        emit(self.bus, EventType.AI_ENDPOINT_RECOVERED)


def _call(callback: Callable[[], None] | None) -> None:
    if callback is None:
        return
    try:
        callback()
    except Exception:  # noqa: BLE001 - a failing reaction must not break the caller
        log.exception("AI endpoint callback failed")
