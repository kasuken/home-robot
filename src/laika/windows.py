"""Where the windows are (#48): a small registry of windows with their room and the camera direction.

The world model keeps one object per name, so it cannot tell two windows apart. This registry can: a window
is a room plus the head pan (and tilt) it was seen at, and two windows of one room are told apart by that
direction. They come from two places:

* vision: an observation that lists a "window" (or "finestra") in a room adds or refreshes the window at the
  pan the frame was taken at;
* people: "la finestra è dietro il divano", "in soggiorno c'è una finestra a sinistra", or the `remember` tool
  with the subject window. A direction in the words is only a hint (left = -45, right = 45); the next time
  vision sees a window in that room the direction is corrected.

The pan is relative to where she stood when she saw it: without navigation (#45) she does not know her heading,
so the direction is good for the spot she looked from. The registry is persisted in the store's key/value table
(no migration needed) and bounded. `WindowWatcher` (outside.py) uses it to know where to point the head.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping

from laika.events import Event, EventBus, EventType
from laika.observations import fold, normalise

if TYPE_CHECKING:
    from laika.memory import Fact, Memory
    from laika.store import Store
    from laika.world_model import WorldModel

log = logging.getLogger(__name__)

NAMESPACE = "windows"
KEY = "registry"
WINDOW = "window"
# What vision and people call a window (folded); "bay window" and the like end in "window".
WINDOW_NAMES = frozenset({"window", "windows", "finestra", "finestre", "finestrone", "portafinestra"})
# Two sightings in one room within this many degrees of pan are the same window.
SAME_WINDOW_DEG = 30.0
MAX_WINDOWS = 12
MAX_PER_ROOM = 4
# Pan hints from the user's words (degrees; negative is left).
SIDE_PAN = {"left": -45.0, "right": 45.0, "centre": 0.0}
DEFAULT_TILT = 0.0  # where a told window is looked at until a sighting says otherwise
SOURCES = ("vision", "user")


def is_window_name(name: str) -> bool:
    """Whether an object name (any language) means a window."""
    folded = fold(name)
    return folded in WINDOW_NAMES or folded.endswith(" window")


def canonical_name(name: str) -> str:
    """"finestra" and "Windows" are both "window"; anything else is just normalised."""
    return WINDOW if is_window_name(name) else normalise(name)


@dataclass
class Window:
    id: str
    room: str
    # Where the head points to see it, None until a direction is known (the watcher then looks ahead).
    pan: float | None
    tilt: float
    source: str
    first_seen: float
    last_seen: float
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "room": self.room, "pan": self.pan, "tilt": self.tilt, "source": self.source,
            "first_seen": self.first_seen, "last_seen": self.last_seen, "note": self.note,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Window:
        pan = raw.get("pan")
        source = str(raw.get("source", "vision"))
        return cls(
            id=str(raw["id"]),
            room=normalise(str(raw["room"])),
            pan=None if pan is None else float(pan),
            tilt=float(raw.get("tilt", DEFAULT_TILT)),
            source=source if source in SOURCES else "vision",
            first_seen=float(raw["first_seen"]),
            last_seen=float(raw["last_seen"]),
            note=raw.get("note"),
        )

    @property
    def label(self) -> str:
        """For people and logs: "window in the living room (pan 60)"."""
        where = f"window in the {self.room.replace('_', ' ')}"
        return where if self.pan is None else f"{where} (pan {self.pan:g})"


class WindowRegistry:
    """The windows of the house. Thread-safe, bounded, written through to the store when there is one."""

    def __init__(
        self,
        store: Store | None = None,
        bus: EventBus | None = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self._clock = clock
        self._lock = threading.Lock()
        self._windows: dict[str, Window] = {}
        self._unsubscribe: Callable[[], None] | None = None
        if bus is not None:
            self._unsubscribe = bus.subscribe(self._on_event, EventType.OBSERVATION_CREATED)

    def close(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    def load(self) -> int:
        """Fill from the store at startup, silently. Returns how many windows were loaded."""
        if self.store is None:
            return 0
        saved = self.store.kv_get(NAMESPACE, KEY, [])
        with self._lock:
            self._windows.clear()
            for raw in saved if isinstance(saved, list) else []:
                try:
                    window = Window.from_dict(raw)
                except (AttributeError, KeyError, TypeError, ValueError):
                    log.warning("windows: skipping an unreadable window %r", raw)
                    continue
                self._windows[window.id] = window
            return len(self._windows)

    # --- reading ------------------------------------------------------------------

    def all(self) -> list[Window]:
        with self._lock:
            return [Window(**w.to_dict()) for w in self._windows.values()]

    def in_room(self, room: str | None) -> list[Window]:
        """The windows of `room`, the ones seen most recently first."""
        key = normalise(room) if room else ""
        if not key:
            return []
        with self._lock:
            found = [Window(**w.to_dict()) for w in self._windows.values() if w.room == key]
        return sorted(found, key=lambda w: -w.last_seen)

    def get(self, window_id: str) -> Window | None:
        with self._lock:
            window = self._windows.get(window_id)
            return Window(**window.to_dict()) if window else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._windows)

    # --- learning ------------------------------------------------------------------

    def note_window(
        self,
        room: str | None,
        pan: float | None,
        tilt: float | None = None,
        *,
        source: str = "vision",
        when: float | None = None,
        note: str | None = None,
    ) -> Window | None:
        """A window in `room` at head direction `pan`: added, or merged into the one already there.

        A window in the same room within SAME_WINDOW_DEG of pan is the same one (a window with no known
        direction matches any window of the room); a sighting corrects the direction a person only hinted at.
        None when there is no room, or when the house already has as many windows as it keeps.
        """
        key = normalise(room) if room else ""
        if not key:
            return None
        source = source if source in SOURCES else "vision"
        now = self._clock() if when is None else when
        with self._lock:
            window = self._match(key, pan)
            if window is None:
                if len(self._windows) >= MAX_WINDOWS or sum(w.room == key for w in self._windows.values()) >= MAX_PER_ROOM:
                    return None
                window = Window(
                    f"win-{uuid.uuid4().hex[:6]}", key, pan, DEFAULT_TILT if tilt is None else tilt, source, now, now
                )
                self._windows[window.id] = window
            else:
                window.last_seen = max(window.last_seen, now)
                if pan is not None and (source == "vision" or window.pan is None):
                    window.pan = pan  # what the camera saw beats what a person hinted
                    if tilt is not None:
                        window.tilt = tilt
                if source == "vision":
                    window.source = "vision"
            if note:
                window.note = note.strip()[:120] or window.note
            result = Window(**window.to_dict())
            snapshot = [w.to_dict() for w in self._windows.values()]
        self._save(snapshot)
        return result

    def _match(self, room: str, pan: float | None) -> Window | None:
        """The window of `room` that `pan` points at (lock held): the nearest within SAME_WINDOW_DEG."""
        candidates = [w for w in self._windows.values() if w.room == room]
        if pan is None:
            return max(candidates, key=lambda w: w.last_seen, default=None)
        unknown = [w for w in candidates if w.pan is None]
        near = [w for w in candidates if w.pan is not None and abs(w.pan - pan) <= SAME_WINDOW_DEG]
        if near:
            return min(near, key=lambda w: abs((w.pan or 0.0) - pan))
        return unknown[0] if unknown else None

    def forget_room(self, room: str) -> int:
        """Drop the windows of `room` (it was merged or renamed away). Returns how many."""
        key = normalise(room)
        with self._lock:
            gone = [w.id for w in self._windows.values() if w.room == key]
            for window_id in gone:
                del self._windows[window_id]
            snapshot = [w.to_dict() for w in self._windows.values()]
        if gone:
            self._save(snapshot)
        return len(gone)

    def _save(self, snapshot: list[dict[str, Any]]) -> None:
        if self.store is not None:
            self.store.kv_put(NAMESPACE, KEY, snapshot)

    # --- vision --------------------------------------------------------------------

    def _on_event(self, event: Event) -> None:
        """Bus thread: an observation with a window in it. The camera direction is where it was seen."""
        data = event.data
        if not any(is_window_name(_name(item)) for item in list(data.get("objects", ()))[:50]):
            return
        pan, tilt = data.get("cameraPan"), data.get("cameraTilt")
        try:
            self.note_window(
                data.get("location"),
                float(pan) if isinstance(pan, (int, float)) and not isinstance(pan, bool) else None,
                float(tilt) if isinstance(tilt, (int, float)) and not isinstance(tilt, bool) else None,
                source="vision",
                when=event.timestamp,
            )
        except Exception:  # noqa: BLE001 - learning a window must never break the bus
            log.exception("could not learn a window from an observation")


def _name(item: Any) -> str:
    name = item.get("name") if isinstance(item, Mapping) else item
    return name if isinstance(name, str) else ""


# --- what people say ---------------------------------------------------------------------------

_VERB = re.compile(r"\bfinestr(?:a|e|one)\s+(?:e|sta|si trova|stanno)\b|\bc e\s+(?:una|la|anche una)\s+finestra\b")
_PLACE = re.compile(
    r"\b(?:a sinistra|a destra|di fronte|davanti|dietro|accanto|vicino|sopra|sotto|in fondo|di lato|"
    r"in|nel|nella|nello|nell)\b"
)
_SIDES = (
    ("left", re.compile(r"\ba sinistra\b|\bsul lato sinistro\b")),
    ("right", re.compile(r"\ba destra\b|\bsul lato destro\b")),
    ("centre", re.compile(r"\bdi fronte\b|\bdavanti a te\b|\bdritto\b")),
)


def parse_window_statement(folded: str) -> tuple[str, float | None] | None:
    """What a folded Italian sentence says about a window: (the words, a pan hint), or None.

    It must say a window is somewhere ("la finestra è dietro il divano", "c'è una finestra a sinistra"),
    not how it is ("la finestra è aperta"). The room, when it names one, is the caller's to read.
    """
    if not _VERB.search(folded) or not _PLACE.search(folded):
        return None
    for side, pattern in _SIDES:
        if pattern.search(folded):
            return side, SIDE_PAN[side]
    return "somewhere", None


def teach_window(
    registry: WindowRegistry,
    world: WorldModel,
    memory: Memory,
    *,
    room: str,
    text: str,
    pan: float | None,
    now: float,
    confidence: float = 1.0,
) -> tuple[Window | None, Fact]:
    """Keep a window a person told her about: the registry, the world model, a fact and an episode.

    The one path for the `remember` tool and the memory selector. The fact's subject names the room, so a
    second window in another room does not replace the first fact (a location fact replaces by subject).
    """
    window = registry.note_window(room, pan, source="user", when=now, note=None)
    world.learn(WINDOW, room=room, timestamp=now)
    subject = f"{WINDOW} {room.replace('_', ' ')}"
    fact = memory.facts.remember(text, subject, "location", source="user", confidence=confidence)
    memory.episodes.record("conversation", f"User said: {text}", [WINDOW, subject], room, now)
    return window, fact
