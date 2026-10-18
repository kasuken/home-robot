"""The plan of the house: which rooms there are, what people call them and which doors join them.

Laika cannot measure where she is (no odometry), so a metric plan would have nothing to attach to. What a plan
can give her is the *shape* of the house, a graph of rooms:

- names: every name people use for a room is taught to the world model, so "la cucina", "la camera dei
  bambini" and the vision model's room ids are the same room (and the vision prompt lists them);
- plausibility: a move from one room to a room that is not next to it needs stronger evidence from the
  vision model (`FloorPlan.plausible_move`, used by the room tracker). Nobody teleports;
- routes: "how do I get to the kitchen?" is answered from the graph (`route`, the `house_route` tool), and the
  plan is in the model's prompt (`summary`), so "where is the bathroom?" has an answer;
- rules: rooms the plan marks `no_go` (a balcony) join the house rules, and routes never pass through them;
- a home spot: the room where she rests.

The plan is a local TOML file (`[house] plan`, `config/house-plan.toml`, ignored by git: it describes a home,
and the repository is public). `config/house-plan.example.toml` shows the format with an invented flat. A
missing file means no plan and nothing changes; a broken one is reported at start and ignored.
"""

from __future__ import annotations

import re
import tomllib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from laika.world_model import WorldModel

MAX_ROOMS = 40
MAX_NAMES = 5  # what the world model keeps per room (world_model.MAX_ROOM_ALIASES)
MAX_NAME_CHARS = 60
MAX_ABOUT_CHARS = 400
MAX_SUMMARY_CHARS = 1800
ID_FORMAT = re.compile(r"[a-z][a-z0-9_]{0,39}")


@dataclass(frozen=True)
class PlanRoom:
    id: str
    names: tuple[str, ...]  # Italian, as people say them; the first one is how she says it
    about: str  # English, for the prompt
    area_m2: float | None
    no_go: bool
    links: frozenset[str]  # rooms with a door or an opening to this one (both directions)

    @property
    def spoken(self) -> str:
        return self.names[0] if self.names else self.id.replace("_", " ")


def _fold(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]|_", " ", text.lower()).split())


class FloorPlan:
    """A graph of rooms. Immutable after loading, so safe to share between threads."""

    def __init__(self, rooms: list[PlanRoom], *, name: str = "", home_room: str | None = None, notes: str = "") -> None:
        self.name = name
        self.notes = notes
        self.rooms: dict[str, PlanRoom] = {room.id: room for room in rooms}
        self.home_room = home_room
        self._names: dict[str, str] = {}
        for room in rooms:
            for text in (room.id.replace("_", " "), *room.names):
                self._names.setdefault(_fold(text), room.id)

    def __len__(self) -> int:
        return len(self.rooms)

    def has(self, room: str | None) -> bool:
        return room in self.rooms if room else False

    def resolve(self, text: str) -> str | None:
        """The room id for an id or any of its names ("la cucina" is not matched: pass the name, articles
        are dropped here)."""
        folded = _fold(text)
        words = folded.split()
        while words and words[0] in {"il", "lo", "la", "l", "i", "gli", "le", "un", "una", "in", "nel", "nella", "the"}:
            words.pop(0)
        return self._names.get(" ".join(words)) or self._names.get(folded)

    def neighbours(self, room: str) -> list[str]:
        found = self.rooms.get(room)
        return sorted(found.links) if found else []

    def adjacent(self, a: str, b: str) -> bool:
        return b in self.rooms[a].links if a in self.rooms and b in self.rooms else False

    def plausible_move(self, a: str | None, b: str | None) -> bool:
        """Can she go from `a` to `b` in one step? True when either room is not on the plan (nothing is known)."""
        if not a or not b or a == b or a not in self.rooms or b not in self.rooms:
            return True
        return self.adjacent(a, b)

    def no_go_rooms(self) -> list[str]:
        return [room.id for room in self.rooms.values() if room.no_go]

    def route(self, start: str, goal: str) -> list[str] | None:
        """The rooms to cross from `start` to `goal` (both included), never through a no-go room; None when
        there is no way (or either room is unknown)."""
        if start not in self.rooms or goal not in self.rooms:
            return None
        if start == goal:
            return [start]
        previous: dict[str, str | None] = {start: None}
        queue = deque([start])
        while queue:
            here = queue.popleft()
            for there in sorted(self.rooms[here].links):
                if there in previous or (self.rooms[there].no_go and there != goal):
                    continue
                previous[there] = here
                if there == goal:
                    path = [there]
                    while previous[path[-1]] is not None:
                        path.append(previous[path[-1]])  # type: ignore[arg-type]
                    return path[::-1]
                queue.append(there)
        return None

    def route_description(self, start: str | None, goal: str) -> dict[str, Any]:
        """For the `house_route` tool: plain English, nothing moves."""
        target = self.rooms[goal]
        result: dict[str, Any] = {"ok": True, "to": target.spoken, "about": target.about}
        if start is None or start not in self.rooms:
            result["from"] = "unknown (she does not know which room she is in)"
            result["neighbours"] = [self.rooms[r].spoken for r in self.neighbours(goal)]
            return result
        result["from"] = self.rooms[start].spoken
        path = self.route(start, goal)
        if path is None:
            result["route"] = "no way through the plan that avoids the off-limits rooms"
        else:
            result["route"] = " -> ".join(self.rooms[r].spoken for r in path)
            result["doors"] = len(path) - 1
        if target.no_go:
            result["note"] = "this room is off limits for her: she only looks at it from outside"
        return result

    def aliases(self) -> list[tuple[str, str]]:
        return [(room.id, name) for room in self.rooms.values() for name in room.names]

    def summary(self, max_chars: int = MAX_SUMMARY_CHARS) -> str:
        """Plain English for the prompt: every room, its names, what it joins; the home spot and the no-go rooms."""
        lines = []
        if self.name or self.notes:
            lines.append(" ".join(part for part in (self.name, self.notes) if part))
        for room in self.rooms.values():
            parts = [f"{room.id} (says: {', '.join(room.names[:3])})" if room.names else room.id]
            if room.area_m2:
                parts.append(f"{room.area_m2:g} m2")
            if room.about:
                parts.append(room.about)
            if room.links:
                parts.append("joins " + ", ".join(sorted(room.links)))
            if room.no_go:
                parts.append("off limits for her wheels")
            lines.append("- " + "; ".join(parts))
        if self.home_room in self.rooms:
            lines.append(f"Her home spot (where she rests) is in the {self.home_room}.")
        text = "\n".join(lines)
        return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."


# --- loading -------------------------------------------------------------------------------------------


def parse_plan(data: Mapping[str, Any], source: str = "house plan") -> FloorPlan:
    """A FloorPlan from parsed TOML. Raises ValueError naming the problem."""
    header = data.get("plan", {})
    if not isinstance(header, Mapping):
        raise ValueError(f"{source}: [plan] must be a table")
    raw_rooms = data.get("room", [])
    if not isinstance(raw_rooms, list) or not raw_rooms:
        raise ValueError(f"{source}: needs at least one [[room]]")
    if len(raw_rooms) > MAX_ROOMS:
        raise ValueError(f"{source}: at most {MAX_ROOMS} rooms")
    links: dict[str, set[str]] = {}
    drafts: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rooms, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{source}: room {index} must be a table")
        room_id = raw.get("id")
        if not isinstance(room_id, str) or not ID_FORMAT.fullmatch(room_id):
            raise ValueError(f"{source}: room {index} needs an id like living_room (lowercase letters, digits, _)")
        if room_id in links:
            raise ValueError(f"{source}: room id {room_id!r} is used twice")
        names = raw.get("names", [])
        if (
            not isinstance(names, list)
            or len(names) > MAX_NAMES
            or not all(isinstance(n, str) and 0 < len(n.strip()) <= MAX_NAME_CHARS for n in names)
        ):
            raise ValueError(f"{source}: [{room_id}] names must be a list of at most {MAX_NAMES} names")
        about = raw.get("about", "")
        if not isinstance(about, str) or len(about) > MAX_ABOUT_CHARS:
            raise ValueError(f"{source}: [{room_id}] about must be text up to {MAX_ABOUT_CHARS} characters")
        area = raw.get("area_m2")
        if area is not None and (isinstance(area, bool) or not isinstance(area, (int, float)) or not 0.5 <= area <= 1000):
            raise ValueError(f"{source}: [{room_id}] area_m2 must be a number from 0.5 to 1000")
        no_go = raw.get("no_go", False)
        if not isinstance(no_go, bool):
            raise ValueError(f"{source}: [{room_id}] no_go must be true or false")
        joined = raw.get("links", [])
        if not isinstance(joined, list) or not all(isinstance(r, str) for r in joined):
            raise ValueError(f"{source}: [{room_id}] links must be a list of room ids")
        links[room_id] = set()
        drafts.append(
            {"id": room_id, "names": tuple(n.strip() for n in names), "about": about.strip(),
             "area": float(area) if area is not None else None, "no_go": no_go, "joined": joined}
        )
    for draft in drafts:
        for other in draft["joined"]:
            if other not in links:
                raise ValueError(f"{source}: [{draft['id']}] links to {other!r}, which is not a room of the plan")
            if other != draft["id"]:
                links[draft["id"]].add(other)
                links[other].add(draft["id"])
    home = header.get("home_room")
    if home is not None and home not in links:
        raise ValueError(f"{source}: [plan] home_room {home!r} is not a room of the plan")
    name, notes = header.get("name", ""), header.get("notes", "")
    if not isinstance(name, str) or not isinstance(notes, str) or len(name) + len(notes) > MAX_ABOUT_CHARS:
        raise ValueError(f"{source}: [plan] name and notes must be short text")
    rooms = [
        PlanRoom(d["id"], d["names"], d["about"], d["area"], d["no_go"], frozenset(links[d["id"]]))
        for d in drafts
    ]
    return FloorPlan(rooms, name=name.strip(), home_room=home, notes=notes.strip())


def load_plan(path: Path) -> FloorPlan | None:
    """The plan in `path`; None when there is no such file. Raises ValueError for a file that cannot be used."""
    if not path.is_file():
        return None
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"{path.name}: {error}") from error
    return parse_plan(data, path.name)


def seed_world(world: WorldModel, plan: FloorPlan) -> None:
    """Teach the world model every room of the plan and the names people use for it."""
    for room_id, name in plan.aliases():
        world.add_room_alias(room_id, name)
