"""A small 2D house for the simulator: pure data and geometry, no robot code.

Units are centimetres and degrees. x grows east and y grows north; heading 0
points east and positive headings turn counter-clockwise. Rooms are rectangles,
walls are line segments (a doorway is a gap in a wall), things and people are
circles, and cliffs (a staircase) are rectangles of floor that is not there.

The `House` is shared between the simulated robot and the test that scripts
it, so every access goes through one lock.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

MAX_RANGE_CM = 400.0
CAMERA_FOV_DEG = 60.0
CAMERA_RANGE_CM = 500.0
PERSON_RADIUS_CM = 20.0

Segment = tuple[float, float, float, float]


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    heading: float = 0.0


@dataclass(frozen=True)
class Room:
    name: str
    x1: float
    y1: float
    x2: float
    y2: float

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x < self.x2 and self.y1 <= y < self.y2


@dataclass
class Thing:
    name: str
    x: float
    y: float
    radius: float
    room: str = ""


@dataclass
class Person:
    name: str
    x: float
    y: float
    radius: float = PERSON_RADIUS_CM


@dataclass(frozen=True)
class Cliff:
    name: str
    x1: float
    y1: float
    x2: float
    y2: float

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


def wall_with_gaps(x1: float, y1: float, x2: float, y2: float, *gaps: tuple[float, float]) -> list[Segment]:
    """Split an axis-aligned wall into segments, leaving a doorway at each gap.

    A gap is (start, end) along the wall: x for a horizontal wall, y for a vertical one.
    """
    horizontal = y1 == y2
    low, high = (x1, x2) if horizontal else (y1, y2)
    segments: list[Segment] = []
    cursor = low
    for start, end in sorted(gaps):
        if start > cursor:
            segments.append(_along(horizontal, x1, y1, cursor, start))
        cursor = max(cursor, end)
    if cursor < high:
        segments.append(_along(horizontal, x1, y1, cursor, high))
    return segments


def _along(horizontal: bool, x: float, y: float, start: float, end: float) -> Segment:
    return (start, y, end, y) if horizontal else (x, start, x, end)


def _normalize(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _ray_segment(ox: float, oy: float, dx: float, dy: float, seg: Segment) -> float | None:
    x1, y1, x2, y2 = seg
    ex, ey = x2 - x1, y2 - y1
    denom = dx * ey - dy * ex
    if abs(denom) < 1e-9:
        return None
    t = ((x1 - ox) * ey - (y1 - oy) * ex) / denom
    u = ((x1 - ox) * dy - (y1 - oy) * dx) / denom
    return t if t >= 0 and 0 <= u <= 1 else None


def _ray_circle(ox: float, oy: float, dx: float, dy: float, cx: float, cy: float, r: float) -> float | None:
    fx, fy = ox - cx, oy - cy
    b = fx * dx + fy * dy
    c = fx * fx + fy * fy - r * r
    if c <= 0:
        return 0.0
    disc = b * b - c
    if disc < 0:
        return None
    t = -b - math.sqrt(disc)
    return t if t >= 0 else None


def _distance_to_segment(px: float, py: float, seg: Segment) -> float:
    x1, y1, x2, y2 = seg
    ex, ey = x2 - x1, y2 - y1
    length_sq = ex * ex + ey * ey
    t = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((px - x1) * ex + (py - y1) * ey) / length_sq))
    return math.hypot(px - (x1 + t * ex), py - (y1 + t * ey))


class House:
    def __init__(
        self,
        rooms: list[Room],
        walls: list[Segment],
        things: list[Thing] | None = None,
        people: list[Person] | None = None,
        cliffs: list[Cliff] | None = None,
        start: Pose | None = None,
    ) -> None:
        self.rooms = list(rooms)
        self.walls = list(walls)
        self.things = list(things or [])
        self.people = list(people or [])
        self.cliffs = list(cliffs or [])
        if start is None:
            start = Pose(rooms[0].x1 + 50, rooms[0].y1 + 50) if rooms else Pose(0, 0)
        self.start = start
        self._lock = threading.RLock()

    # --- lookup ---------------------------------------------------------

    def room_at(self, x: float, y: float) -> Room | None:
        with self._lock:
            return next((room for room in self.rooms if room.contains(x, y)), None)

    def thing(self, name: str) -> Thing:
        with self._lock:
            for thing in self.things:
                if thing.name == name:
                    return thing
        raise KeyError(name)

    def is_over_cliff(self, x: float, y: float) -> bool:
        with self._lock:
            return any(cliff.contains(x, y) for cliff in self.cliffs)

    # --- environmental changes -------------------------------------------

    def add_thing(self, thing: Thing) -> None:
        with self._lock:
            if not thing.room:
                room = self.room_at(thing.x, thing.y)
                thing.room = room.name if room else ""
            self.things.append(thing)

    def remove_thing(self, name: str) -> None:
        with self._lock:
            self.things.remove(self.thing(name))

    def move_thing(self, name: str, x: float, y: float) -> None:
        with self._lock:
            thing = self.thing(name)
            thing.x, thing.y = x, y
            room = self.room_at(x, y)
            thing.room = room.name if room else ""

    def add_person(self, person: Person) -> None:
        with self._lock:
            self.people.append(person)

    def remove_person(self, name: str) -> None:
        with self._lock:
            self.people = [p for p in self.people if p.name != name]

    def move_person(self, name: str, x: float, y: float) -> None:
        with self._lock:
            for person in self.people:
                if person.name == name:
                    person.x, person.y = x, y
                    return
        raise KeyError(name)

    # --- geometry ---------------------------------------------------------

    def ray_distance(self, x: float, y: float, heading: float, max_range: float = MAX_RANGE_CM) -> float:
        """Distance along one ray to the nearest wall, thing or person; `max_range` when clear."""
        dx, dy = math.cos(math.radians(heading)), math.sin(math.radians(heading))
        best = max_range
        with self._lock:
            for seg in self.walls:
                hit = _ray_segment(x, y, dx, dy, seg)
                if hit is not None:
                    best = min(best, hit)
            for body in (*self.things, *self.people):
                hit = _ray_circle(x, y, dx, dy, body.x, body.y, body.radius)
                if hit is not None:
                    best = min(best, hit)
        return best

    def collides(self, x: float, y: float, radius: float) -> bool:
        """True when a round body of `radius` at (x, y) touches a wall, thing or person."""
        with self._lock:
            if any(_distance_to_segment(x, y, seg) < radius for seg in self.walls):
                return True
            return any(math.hypot(x - b.x, y - b.y) < b.radius + radius for b in (*self.things, *self.people))

    def visible(
        self,
        x: float,
        y: float,
        direction: float,
        fov: float = CAMERA_FOV_DEG,
        max_range: float = CAMERA_RANGE_CM,
    ) -> tuple[list[Thing], list[Person]]:
        """What a camera at (x, y) looking along `direction` sees. Walls block the view."""
        with self._lock:
            things = [t for t in self.things if self._sees(x, y, direction, fov, max_range, t)]
            people = [p for p in self.people if self._sees(x, y, direction, fov, max_range, p)]
        return things, people

    def _sees(
        self, x: float, y: float, direction: float, fov: float, max_range: float, body: Thing | Person
    ) -> bool:
        dist = math.hypot(body.x - x, body.y - y)
        if dist > max_range:
            return False
        if dist == 0:
            return True
        bearing = math.degrees(math.atan2(body.y - y, body.x - x))
        slack = math.degrees(math.atan2(body.radius, dist))
        if abs(_normalize(bearing - direction)) > fov / 2 + slack:
            return False
        dx, dy = (body.x - x) / dist, (body.y - y) / dist
        for seg in self.walls:
            hit = _ray_segment(x, y, dx, dy, seg)
            if hit is not None and hit < dist:
                return False
        return True


def default_house() -> House:
    """The PRD example: living room, office, kitchen and bedroom, with a staircase."""
    rooms = [
        Room("living_room", 0, 0, 500, 400),
        Room("kitchen", 500, 0, 800, 400),
        Room("office", 0, 400, 500, 700),
        Room("bedroom", 500, 400, 800, 700),
    ]
    walls = [
        *wall_with_gaps(0, 0, 800, 0),
        *wall_with_gaps(0, 700, 800, 700),
        *wall_with_gaps(0, 0, 0, 700),
        *wall_with_gaps(800, 0, 800, 700),
        *wall_with_gaps(500, 0, 500, 400, (150, 250)),
        *wall_with_gaps(500, 400, 500, 700, (500, 600)),
        *wall_with_gaps(0, 400, 500, 400, (200, 300)),
        *wall_with_gaps(500, 400, 800, 400, (600, 700)),
    ]
    things = [
        Thing("sofa", 130, 200, 40, "living_room"),
        Thing("television", 30, 200, 20, "living_room"),
        Thing("keyboard", 420, 60, 25, "living_room"),
        Thing("guitars", 300, 40, 20, "living_room"),
        Thing("backpack", 170, 270, 15, "living_room"),
        Thing("desk", 100, 600, 50, "office"),
        Thing("monitor", 100, 660, 15, "office"),
        Thing("chair", 160, 520, 25, "office"),
        Thing("table", 650, 250, 50, "kitchen"),
        Thing("coffee_machine", 770, 40, 15, "kitchen"),
    ]
    cliffs = [Cliff("stairs", 420, 300, 500, 400)]
    return House(rooms, walls, things, cliffs=cliffs, start=Pose(300, 200, 0))
