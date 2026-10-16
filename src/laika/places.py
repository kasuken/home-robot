"""What Laika learns about the places she explores, kept for later (the owner's ask after V1.1).

The world model already keeps every object and the room it was seen in. `PlaceNotes` adds what a room is
like, so a later conversation, search or exploration can use it:

* how the room looks: the latest vision descriptions taken there (a few, deduplicated, each clipped);
* which rooms it leads to: every RoomEntered with a `previous` room counts one passage between the two;
* how often she explored it and when last, and how many people she met there.

It listens on the event bus (cheap, on the bus thread) and keeps one record per room in the store's `kv`
table (namespace `places`), so it survives a restart without a schema change.

An exploration session (`begin` / `end`, called by the `Explorer`) also becomes an `explored` episode in
memory: the rooms visited, the objects seen for the first time, the people met, how far it went
("Explored the kitchen and the living room for 9 minutes: new: plant, lamp; met 1 person"). `what_changed`
and "cosa hai fatto oggi?" read those episodes.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping

from laika.events import Event, EventBus, EventType
from laika.observations import normalise

if TYPE_CHECKING:
    from laika.memory import Memory
    from laika.store import Store
    from laika.world_model import WorldModel

log = logging.getLogger(__name__)

NAMESPACE = "places"
MAX_LOOKS = 3  # descriptions kept per room
MAX_LOOK_CHARS = 160
MAX_NEIGHBOURS = 6
MAX_ROOMS = 40
SUMMARY_MAX_CHARS = 700
MAX_NEW_LISTED = 8


def _words(name: str) -> str:
    return name.replace("_", " ")


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _ago(seconds: float) -> str:
    if seconds < 3600:
        return f"{max(1, int(seconds // 60))} min ago"
    if seconds < 2 * 86400:
        return f"{int(seconds // 3600)} h ago"
    return f"{int(seconds // 86400)} days ago"


def _duration(seconds: float) -> str:
    minutes = int(round(seconds / 60))
    return f"{minutes} minute{'s' if minutes != 1 else ''}" if minutes else f"{int(seconds)} seconds"


@dataclass
class Place:
    room: str
    looks: list[str] = field(default_factory=list)
    neighbours: dict[str, int] = field(default_factory=dict)
    explored: int = 0
    last_explored: float | None = None
    people_met: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "room": self.room, "looks": list(self.looks), "neighbours": dict(self.neighbours),
            "explored": self.explored, "last_explored": self.last_explored, "people_met": self.people_met,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Place:
        last = raw.get("last_explored")
        return cls(
            room=str(raw["room"]),
            looks=[str(s) for s in raw.get("looks") or ()][-MAX_LOOKS:],
            neighbours={str(k): int(v) for k, v in dict(raw.get("neighbours") or {}).items()},
            explored=int(raw.get("explored") or 0),
            last_explored=float(last) if isinstance(last, (int, float)) else None,
            people_met=int(raw.get("people_met") or 0),
        )

    def neighbours_by_use(self) -> list[str]:
        return [room for room, _ in sorted(self.neighbours.items(), key=lambda item: -item[1])]


@dataclass
class _Session:
    started: float
    known_before: frozenset[str]
    rooms: list[str] = field(default_factory=list)
    seen: dict[str, str | None] = field(default_factory=dict)  # object -> room it was seen in
    people: int = 0


class PlaceNotes:
    def __init__(
        self,
        bus: EventBus | None = None,
        store: Store | None = None,
        *,
        world: WorldModel | None = None,
        memory: Memory | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self.world = world
        self.memory = memory
        self._clock = clock
        self._lock = threading.Lock()
        self._places: dict[str, Place] = {}
        self._session: _Session | None = None
        self._load()
        self._unsubscribe = (
            bus.subscribe(self._on_event, EventType.ROOM_ENTERED, EventType.OBSERVATION_CREATED)
            if bus is not None
            else None
        )

    def close(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    # --- what she knows --------------------------------------------------------

    def place(self, room: str) -> Place | None:
        key = normalise(room) if room else ""
        with self._lock:
            place = self._places.get(key)
            return Place.from_dict(place.to_dict()) if place is not None else None

    def places(self) -> list[Place]:
        with self._lock:
            return [Place.from_dict(p.to_dict()) for p in self._places.values()]

    def summary(self, max_chars: int = SUMMARY_MAX_CHARS) -> str:
        """Plain English for a prompt: how each room looks and what it leads to. '' when nothing is known."""
        now = self._clock()
        lines = []
        with self._lock:
            places = sorted(self._places.values(), key=lambda p: -(p.last_explored or 0))
            for place in places:
                parts = []
                if place.looks:
                    parts.append(f"looks like: {place.looks[-1]}")
                if place.neighbours:
                    parts.append("leads to " + ", ".join(_words(r) for r in place.neighbours_by_use()[:3]))
                if place.last_explored is not None:
                    parts.append(f"explored {place.explored}x, last {_ago(max(0.0, now - place.last_explored))}")
                if parts:
                    lines.append(f"{_words(place.room).capitalize()}: {'; '.join(parts)}.")
        return _clip(" ".join(lines), max_chars) if lines else ""

    def to_dict(self) -> dict[str, Any]:
        """For a tool: every room with its notes."""
        now = self._clock()
        out = []
        for place in self.places():
            entry: dict[str, Any] = {"room": _words(place.room), "looks": place.looks}
            entry["leads_to"] = [_words(r) for r in place.neighbours_by_use()]
            entry["explored"] = place.explored
            if place.last_explored is not None:
                entry["last_explored"] = _ago(max(0.0, now - place.last_explored))
            entry["people_met"] = place.people_met
            out.append(entry)
        return {"rooms": out}

    # --- an exploration session ------------------------------------------------

    def begin(self) -> None:
        """An exploration starts: count what she sees and where she goes from now on."""
        known = frozenset(o.get("name") for o in self.world.to_dict()["objects"]) if self.world else frozenset()
        session = _Session(self._clock(), known)
        if self.world is not None and self.world.current_room:
            session.rooms.append(self.world.current_room)
        with self._lock:
            self._session = session

    def end(self, *, steps: int = 0, scans: int = 0, driving: bool = True, reason: str = "") -> str | None:
        """The exploration ended: mark the rooms explored and record one `explored` episode. Its summary, or None."""
        now = self._clock()
        with self._lock:
            session, self._session = self._session, None
            if session is None:
                return None
            rooms = list(dict.fromkeys(session.rooms + [r for r in session.seen.values() if r]))
            for room in rooms:
                place = self._get(room)
                place.explored += 1
                place.last_explored = now
            touched = [self._places[r] for r in rooms if r in self._places]
        for place in touched:
            self._save(place)
        if not (steps or scans):
            return None  # nothing happened: nothing worth remembering
        new = [name for name in session.seen if name not in session.known_before]
        summary = self._sentence(rooms, new, session.people, now - session.started, driving, steps)
        if self.memory is not None:
            try:
                self.memory.episodes.record(
                    "explored", summary, [*rooms, *new[:MAX_NEW_LISTED]], rooms[-1] if rooms else None, now
                )
            except Exception:  # noqa: BLE001 - a note, never a reason for the exploration to fail
                log.exception("could not record the exploration")
        print(f"[laika] places: {summary}")
        return summary

    @staticmethod
    def _sentence(rooms: list[str], new: list[str], people: int, seconds: float, driving: bool, steps: int) -> str:
        where = (
            " and ".join(f"the {_words(r)}" for r in rooms) if len(rooms) <= 2
            else ", ".join(f"the {_words(r)}" for r in rooms[:-1]) + f" and the {_words(rooms[-1])}"
        ) if rooms else "the house (room not recognised)"
        how = "" if driving else " with the head only"
        text = f"Explored {where}{how} for {_duration(seconds)}"
        if driving and steps:
            text += f" ({steps} steps)"
        details = []
        if new:
            listed = ", ".join(_words(n) for n in new[:MAX_NEW_LISTED])
            more = f" and {len(new) - MAX_NEW_LISTED} more" if len(new) > MAX_NEW_LISTED else ""
            details.append(f"new: {listed}{more}")
        else:
            details.append("nothing new")
        if people:
            details.append(f"met {people} {'person' if people == 1 else 'people'}")
        return f"{text}: {'; '.join(details)}"

    # --- the event bus ----------------------------------------------------------

    def _on_event(self, event: Event) -> None:
        try:
            if event.type is EventType.ROOM_ENTERED:
                self._entered(event.data.get("room"), event.data.get("previous"))
            elif event.type is EventType.OBSERVATION_CREATED:
                self._observed(event.data)
        except Exception:  # noqa: BLE001 - the bus thread must survive a bad event
            log.exception("place notes failed on %s", event.type)

    def _entered(self, room: Any, previous: Any) -> None:
        room = normalise(room) if isinstance(room, str) and room.strip() else None
        previous = normalise(previous) if isinstance(previous, str) and previous.strip() else None
        if room is None:
            return
        changed = []
        with self._lock:
            if self._session is not None:
                self._session.rooms.append(room)
            if previous is not None and previous != room:
                for here, there in ((room, previous), (previous, room)):
                    place = self._get(here)
                    place.neighbours[there] = place.neighbours.get(there, 0) + 1
                    if len(place.neighbours) > MAX_NEIGHBOURS:
                        victim = min((r for r in place.neighbours if r != there), key=lambda r: place.neighbours[r])
                        del place.neighbours[victim]
                    changed.append(place)
        for place in changed:
            self._save(place)

    def _observed(self, data: Mapping[str, Any]) -> None:
        room = data.get("location")
        room = normalise(room) if isinstance(room, str) and room.strip() else None
        people = data.get("people") if isinstance(data.get("people"), int) else 0
        names = []
        for raw in data.get("objects") or ():
            name = raw.get("name") if isinstance(raw, Mapping) else raw
            if isinstance(name, str) and name.strip():
                names.append(normalise(name))
        with self._lock:
            session = self._session
            if session is not None:
                for name in names:
                    session.seen.setdefault(name, room)
                session.people = max(session.people, people)
            if room is None:
                return
            place = self._get(room)
            description = data.get("description")
            if isinstance(description, str) and description.strip():
                look = _clip(description, MAX_LOOK_CHARS)
                if look not in place.looks:
                    place.looks.append(look)
                    del place.looks[:-MAX_LOOKS]
            if session is not None and people:
                place.people_met += 1
        self._save(place)

    # --- storage ------------------------------------------------------------------

    def _get(self, room: str) -> Place:
        """The record for `room`, made if needed (lock held)."""
        place = self._places.get(room)
        if place is None:
            if len(self._places) >= MAX_ROOMS:
                # Room names come from the vision model: a runaway list keeps the rooms explored most recently.
                oldest = min(self._places.values(), key=lambda p: p.last_explored or 0)
                del self._places[oldest.room]
            place = self._places[room] = Place(room)
        return place

    def _save(self, place: Place) -> None:
        if self.store is None:
            return
        with self._lock:
            data = place.to_dict()
        try:
            self.store.kv_put(NAMESPACE, place.room, data)
        except Exception:  # noqa: BLE001 - losing a note is better than failing the bus
            log.exception("could not save the notes on %s", place.room)

    def _load(self) -> None:
        if self.store is None:
            return
        try:
            items = self.store.kv_items(NAMESPACE)
        except Exception:  # noqa: BLE001
            log.exception("could not load the place notes")
            return
        for key, raw in items.items():
            try:
                place = Place.from_dict(raw)
            except (KeyError, TypeError, ValueError):
                log.warning("places: skipping an unreadable record for %r", key)
                continue
            self._places[place.room] = place
        while len(self._places) > MAX_ROOMS:
            oldest = min(self._places.values(), key=lambda p: p.last_explored or 0)
            del self._places[oldest.room]
