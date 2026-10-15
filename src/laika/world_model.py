"""Semantic map of the house, grown from what Laika sees and is told (docs/prd.md, World Model).

Nothing about the house is predefined: rooms and objects appear when the robot
enters a room, sees something, or someone tells it. The model is kept in memory and
mirrored to the store by `WorldModelPersister` (#33), so it survives a restart. Events
arrive on the event-bus thread while the conversation reads the model, so every method
takes one lock. Change listeners are called after the lock is released, one at a time
and in the order the changes were made.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

from laika.events import Event, EventBus, EventType
from laika.observations import normalise

if TYPE_CHECKING:
    from laika.store import Store

log = logging.getLogger(__name__)

MAX_OBJECTS = 500
MAX_PREVIOUS_ROOMS = 5
SUMMARY_MAX_CHARS = 600
VISION = "vision"
CONVERSATION = "conversation"
VISION_CONFIDENCE = 0.9
CONVERSATION_CONFIDENCE = 0.8

# A listener gets ("object", KnownObject), ("deleted", name) or ("room", name, first_seen, last_visit).
ChangeListener = Callable[..., None]


def _room_key(room: str | None) -> str | None:
    key = normalise(room) if room else ""
    return key or None


@dataclass
class KnownObject:
    name: str
    room: str | None
    first_seen: float
    last_seen: float
    times_seen: int = 0
    source: str = VISION
    note: str | None = None
    confidence: float = 1.0
    previous_rooms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "room": self.room,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "times_seen": self.times_seen,
            "source": self.source,
            "note": self.note,
            "confidence": self.confidence,
            "previous_rooms": list(self.previous_rooms),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> KnownObject:
        return cls(
            name=str(raw["name"]),
            room=_room_key(raw.get("room")),
            first_seen=float(raw["first_seen"]),
            last_seen=float(raw["last_seen"]),
            times_seen=int(raw.get("times_seen", 0)),
            source=str(raw.get("source", VISION)),
            note=raw.get("note"),
            confidence=float(raw.get("confidence", 1.0)),
            previous_rooms=[str(room) for room in raw.get("previous_rooms", [])][-MAX_PREVIOUS_ROOMS:],
        )


@dataclass
class RoomRecord:
    name: str
    first_seen: float
    last_visit: float

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "first_seen": self.first_seen, "last_visit": self.last_visit}


def _move(obj: KnownObject, room: str | None) -> None:
    """Keep the latest room; the one it leaves goes into a short history."""
    if room is None or room == obj.room:
        return
    if obj.room is not None:
        obj.previous_rooms.append(obj.room)
        del obj.previous_rooms[:-MAX_PREVIOUS_ROOMS]
    obj.room = room


class WorldModel:
    def __init__(self, max_objects: int = MAX_OBJECTS) -> None:
        self.max_objects = max(1, max_objects)
        self._lock = threading.Lock()
        # Held while a mutation and its notifications run, so listeners see changes in order.
        # Never taken by readers, and `_lock` is never held while a listener runs.
        self._notify_lock = threading.RLock()
        self._listeners: list[ChangeListener] = []
        # Least recently seen first, so eviction drops the stalest object.
        self._objects: OrderedDict[str, KnownObject] = OrderedDict()
        # Rooms in the order they were first known.
        self._rooms: dict[str, RoomRecord] = {}
        self._current_room: str | None = None

    @property
    def current_room(self) -> str | None:
        with self._lock:
            return self._current_room

    def on_change(self, listener: ChangeListener) -> Callable[[], None]:
        """Call `listener` after each change. Returns a function that removes it.

        A listener that raises is logged and the others still run; the model is unaffected.
        """
        with self._notify_lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._notify_lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return unsubscribe

    def _notify(self, changes: list[tuple[Any, ...]]) -> None:
        """Run the listeners for `changes`. Caller holds `_notify_lock` and not `_lock`."""
        for change in changes:
            for listener in list(self._listeners):
                try:
                    listener(*change)
                except Exception:  # noqa: BLE001 - a listener must not break the model
                    log.exception("world model listener failed on a %s change", change[0])

    def enter_room(self, room: str | None, timestamp: float | None = None) -> None:
        key = _room_key(room)
        when = time.time() if timestamp is None else timestamp
        with self._notify_lock:
            changes: list[tuple[Any, ...]] = []
            with self._lock:
                self._current_room = key
                if key is not None:
                    self._touch_room(key, when, changes)
            self._notify(changes)

    def observe(
        self,
        objects: Iterable[str],
        room: str | None,
        timestamp: float | None = None,
        confidence: float = VISION_CONFIDENCE,
    ) -> None:
        when = time.time() if timestamp is None else timestamp
        key_room = _room_key(room)
        with self._notify_lock:
            changes: list[tuple[Any, ...]] = []
            with self._lock:
                if key_room is not None:
                    self._touch_room(key_room, when, changes)
                for name in dict.fromkeys(normalise(raw) for raw in objects):
                    if not name:
                        continue
                    obj = self._objects.get(name)
                    if obj is None:
                        obj = KnownObject(
                            name=name, room=key_room, first_seen=when, last_seen=when,
                            source=VISION, confidence=confidence,
                        )
                        self._objects[name] = obj
                    else:
                        _move(obj, key_room)
                        obj.last_seen = when
                        obj.source = VISION
                        obj.confidence = confidence
                    obj.times_seen += 1
                    self._objects.move_to_end(name)
                    changes.append(("object", _copy(obj)))
                changes.extend(self._evict())
            self._notify(changes)

    def learn(
        self,
        name: str,
        room: str | None = None,
        note: str | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Record something a person told Laika. It is not a sighting, so `times_seen` stays put."""
        key = normalise(name)
        if not key:
            return
        when = time.time() if timestamp is None else timestamp
        key_room = _room_key(room)
        with self._notify_lock:
            changes: list[tuple[Any, ...]] = []
            with self._lock:
                if key_room is not None:
                    self._touch_room(key_room, when, changes)
                obj = self._objects.get(key)
                if obj is None:
                    obj = KnownObject(
                        name=key, room=key_room, first_seen=when, last_seen=when,
                        source=CONVERSATION, confidence=CONVERSATION_CONFIDENCE,
                    )
                    self._objects[key] = obj
                else:
                    _move(obj, key_room)
                    obj.last_seen = when
                    obj.source = CONVERSATION
                    obj.confidence = CONVERSATION_CONFIDENCE
                if note and note.strip():
                    obj.note = note.strip()
                self._objects.move_to_end(key)
                changes.append(("object", _copy(obj)))
                changes.extend(self._evict())
            self._notify(changes)

    def load(self, objects: Iterable[Mapping[str, Any]], rooms: Iterable[Mapping[str, Any]]) -> None:
        """Replace the contents with stored data, silently: no listener is called and no time changes.

        Meant for startup, before the persister is attached. The current room is not
        restored: Laika does not know where she is until she sees or enters a room.
        Unreadable entries are logged and skipped.
        """
        with self._lock:
            self._objects.clear()
            self._rooms.clear()
            self._current_room = None
            for raw in rooms:
                try:
                    key = _room_key(raw.get("name"))
                    if key is not None and key not in self._rooms:
                        self._rooms[key] = RoomRecord(
                            key, float(raw["first_seen"]), float(raw["last_visit"])
                        )
                except (AttributeError, KeyError, TypeError, ValueError):
                    log.warning("world model: skipping an unreadable room %r", raw)
            for item in objects:
                try:
                    obj = KnownObject.from_dict(item)
                except (AttributeError, KeyError, TypeError, ValueError):
                    log.warning("world model: skipping an unreadable object %r", item)
                    continue
                self._objects[obj.name] = obj
            self._evict()  # silently: the stored rows beyond the limit are trimmed from memory only

    def where_is(self, name: str) -> KnownObject | None:
        with self._lock:
            obj = self._objects.get(normalise(name))
            return _copy(obj) if obj is not None else None

    def objects_in(self, room: str) -> list[KnownObject]:
        key = _room_key(room)
        with self._lock:
            return [_copy(o) for o in reversed(list(self._objects.values())) if o.room == key]

    def rooms(self) -> list[str]:
        with self._lock:
            return list(self._rooms)

    def summary(self, max_chars: int = SUMMARY_MAX_CHARS) -> str:
        """Plain English for a prompt, cut to `max_chars`."""
        with self._lock:
            lines: list[str] = []
            if self._current_room is not None:
                lines.append(f"Laika is in the {_words(self._current_room)}.")
            else:
                lines.append("Laika does not know which room it is in.")
            if not self._rooms:
                lines.append("The house has not been explored yet.")
            objects = list(reversed(list(self._objects.values())))
            for room in self._rooms:
                names = [_describe(o) for o in objects if o.room == room]
                listed = ", ".join(names) if names else "nothing known yet"
                lines.append(f"{_words(room).capitalize()}: {listed}.")
            loose = [_describe(o) for o in objects if o.room is None]
            if loose:
                lines.append(f"Objects with no known room: {', '.join(loose)}.")
        text = " ".join(lines)
        if len(text) > max_chars:
            text = text[: max(0, max_chars - 3)].rstrip() + "..."
        return text

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "current_room": self._current_room,
                "rooms": [room.to_dict() for room in self._rooms.values()],
                "objects": [o.to_dict() for o in self._objects.values()],
            }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], max_objects: int = MAX_OBJECTS) -> WorldModel:
        world = cls(max_objects)
        world.load(raw.get("objects", []), raw.get("rooms", []))
        world._current_room = _room_key(raw.get("current_room"))
        return world

    # Callers hold `_lock`. Each returns the changes to notify, in order.

    def _touch_room(self, key: str, when: float, changes: list[tuple[Any, ...]]) -> None:
        room = self._rooms.get(key)
        if room is None:
            room = RoomRecord(key, when, when)
            self._rooms[key] = room
        elif when <= room.last_visit:
            return
        else:
            room.last_visit = when
        changes.append(("room", key, room.first_seen, room.last_visit))

    def _evict(self) -> list[tuple[Any, ...]]:
        deleted: list[tuple[Any, ...]] = []
        while len(self._objects) > self.max_objects:
            name, _ = self._objects.popitem(last=False)
            deleted.append(("deleted", name))
        return deleted


def _copy(obj: KnownObject) -> KnownObject:
    return replace(obj, previous_rooms=list(obj.previous_rooms))


def _words(name: str) -> str:
    return name.replace("_", " ")


def _describe(obj: KnownObject) -> str:
    return f"{_words(obj.name)} ({obj.note})" if obj.note else _words(obj.name)


class WorldModelPersister:
    """Writes every world model change to the store (#33). The store queues the writes without blocking."""

    def __init__(self, world: WorldModel, store: Store) -> None:
        self.store = store
        self._unsubscribe = world.on_change(self._on_change)

    def _on_change(self, kind: str, *args: Any) -> None:
        if kind == "object":
            self.store.put_world_object(args[0].to_dict())
        elif kind == "deleted":
            self.store.delete_world_object(args[0])
        elif kind == "room":
            self.store.put_world_room(*args)

    def close(self) -> None:
        self._unsubscribe()


class WorldModelUpdater:
    """Feeds the world model from the event bus. Handlers run on the bus thread."""

    def __init__(self, world: WorldModel, bus: EventBus) -> None:
        self.world = world
        self.bus = bus
        self._unsubscribe = bus.subscribe(
            self._on_event, EventType.ROOM_ENTERED, EventType.OBSERVATION_CREATED
        )

    def close(self) -> None:
        self._unsubscribe()

    def _on_event(self, event: Event) -> None:
        if event.type is EventType.ROOM_ENTERED:
            self.world.enter_room(event.data.get("room"), timestamp=event.timestamp)
            return
        # An observation carries the room it was made in; fall back to the tracked room.
        room = event.data.get("location") or self.world.current_room
        names = [o["name"] if isinstance(o, dict) else o for o in event.data.get("objects", ())]
        confidence = event.data.get("confidence", VISION_CONFIDENCE)
        self.world.observe(names, room, timestamp=event.timestamp, confidence=confidence)
