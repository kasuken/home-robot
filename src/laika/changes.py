"""Environment changes: what is different from what Laika knew (#36, docs/prd.md "Observations").

`ChangeDetector.detect` compares one observation with the world model as it was BEFORE that
observation is applied (the same ordering rule as the scorer: the Observer detects before it
publishes ObservationCreated, and the world model learns from that event afterwards), and with
the previous look at the same view. It only reads. Four kinds:

* "moved":   the object was last known in another room (`previous_location`), now in `location`;
* "new":     never seen before, in a room Laika has known for `familiar_room_after_s`;
* "missing": in the previous look at the same view, absent now. Only judged on a confident
  observation, so a blurry frame does not make things vanish;
* "unusual": seen in a room that is not its usual room (`previous_location` is the usual room).

`ChangeTracker` keeps the changes in a bounded list and records the "moved", "new" and "missing"
ones as episodes, so `what_changed` can answer after a restart. The same change for the same object
is recorded once, until that object's state changes (the world model learns from the bus thread, so
two quick looks can both see the same change).
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

from laika.config import CuriosityConfig
from laika.observations import LOG_SIZE, Observation, ObservationLog, find_previous, normalise
from laika.world_model import WorldModel

if TYPE_CHECKING:
    from laika.memory import EpisodicMemory

log = logging.getLogger(__name__)

KINDS = ("moved", "new", "missing", "unusual")
# Kinds that are remembered as episodes ("unusual" is a score input, not an event).
EPISODE_CHANGES = ("moved", "new", "missing")
# Below this confidence a frame is too uncertain to say something vanished.
MISSING_MIN_CONFIDENCE = 0.6
MAX_RECENT_CHANGES = 100
MAX_TRACKED_OBJECTS = 500


def _words(room: str) -> str:
    return room.replace("_", " ")


@dataclass(frozen=True)
class Change:
    """One difference. `location` is where the object is now (for "missing": where it was)."""

    kind: str
    object: str
    location: str
    previous_location: str | None
    timestamp: float
    observation_id: str

    @property
    def text(self) -> str:
        """Short English, also the episode summary."""
        name, here = self.object, _words(self.location)
        if self.kind == "moved":
            return f"{name} moved from {_words(self.previous_location or 'unknown room')} to {here}"
        if self.kind == "new":
            return f"{name} appeared in {here}"
        if self.kind == "missing":
            return f"{name} is no longer in {here}"
        return f"{name} in {here}, usually {_words(self.previous_location or 'elsewhere')}"

    @property
    def reason(self) -> str:
        """The curiosity reason string for this change."""
        name, here = self.object, _words(self.location)
        if self.kind == "moved":
            return f"moved: {name} from {_words(self.previous_location or 'unknown room')} to {here}"
        if self.kind == "new":
            return f"new in {here}: {name}"
        if self.kind == "missing":
            return f"missing from {here}: {name}"
        return f"unusual location: {name} in {here} (usually {_words(self.previous_location or '')})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "object": self.object,
            "location": self.location,
            "previous_location": self.previous_location,
            "timestamp": self.timestamp,
            "observation_id": self.observation_id,
            "text": self.text,
        }

    def signature(self) -> tuple[str, str, str | None]:
        return (self.kind, self.location, self.previous_location)


class ChangeDetector:
    """Finds the changes in an observation. Pure: reads the world model and the log, writes nothing."""

    def __init__(
        self,
        world: WorldModel,
        log: ObservationLog | None = None,
        config: CuriosityConfig | None = None,
    ) -> None:
        self.world = world
        self.log = log
        self.config = config if config is not None else CuriosityConfig()

    def detect(self, observation: Observation, previous: Observation | None = None) -> list[Change]:
        """The changes `observation` shows. `previous` is the earlier look at the same view;
        when not given it is taken from the log. Call this before the world model learns the frame."""
        room = observation.location
        if room is None:
            return []
        if previous is None and self.log is not None:
            previous = find_previous(self.log.recent(LOG_SIZE), observation)
        familiar = self.world.is_familiar(room, observation.timestamp - self.config.familiar_room_after_s)
        changes: list[Change] = []
        seen: set[str] = set()

        def add(kind: str, name: str, before: str | None) -> None:
            changes.append(Change(kind, name, room, before, observation.timestamp, observation.id))

        for seen_object in observation.objects:
            name = normalise(seen_object.name)
            if not name or name in seen:
                continue
            seen.add(name)
            known = self.world.where_is(name)
            if known is None:
                if familiar:
                    add("new", name, None)
                continue
            if known.room is not None and known.room != room:
                add("moved", name, known.room)
            usual = self.world.usual_room(name)
            if usual is not None and usual != room:
                add("unusual", name, usual)
        if previous is not None and observation.confidence >= MISSING_MIN_CONFIDENCE:
            for name in sorted({normalise(o.name) for o in previous.objects} - seen):
                if name:
                    add("missing", name, None)
        return changes


class ChangeTracker:
    """Recent changes in memory and the "moved", "new" and "missing" ones as episodes. Thread-safe."""

    def __init__(self, episodes: EpisodicMemory | None = None, max_items: int = MAX_RECENT_CHANGES) -> None:
        self.episodes = episodes
        self._lock = threading.Lock()
        self._recent: deque[Change] = deque(maxlen=max(1, max_items))
        # Per object, the signature of the last change recorded for it.
        self._last: OrderedDict[str, tuple[str, str, str | None]] = OrderedDict()

    def record(self, changes: list[Change], present: Iterable[str] = ()) -> list[Change]:
        """Keep and remember the changes that are not repeats. Returns those that were recorded.

        `present` are the objects in view: one that was missing and is back has a new state, so
        its next disappearance is a change again.
        """
        fresh: list[Change] = []
        with self._lock:
            for name in present:
                key = normalise(name)
                if self._last.get(key, ("",))[0] == "missing":
                    del self._last[key]
            for change in changes:
                if change.kind not in EPISODE_CHANGES:
                    continue
                if self._last.get(change.object) == change.signature():
                    continue
                self._last[change.object] = change.signature()
                self._last.move_to_end(change.object)
                while len(self._last) > MAX_TRACKED_OBJECTS:
                    self._last.popitem(last=False)
                self._recent.append(change)
                fresh.append(change)
        for change in fresh:
            self._remember(change)
        return fresh

    def recent(self, n: int = 20) -> list[Change]:
        """The newest `n` recorded changes, newest first."""
        with self._lock:
            return list(reversed(self._recent))[: max(0, n)]

    def _remember(self, change: Change) -> None:
        if self.episodes is None:
            return
        try:
            if change.kind == "moved":
                self.episodes.record_moved(
                    change.object, change.previous_location, change.location, change.timestamp, change.text
                )
            elif change.kind == "new":
                self.episodes.record_new(change.object, change.location, change.timestamp, change.text)
            else:
                self.episodes.record_missing(change.object, change.location, change.timestamp, change.text)
        except Exception:  # noqa: BLE001 - remembering must never drop the observation
            log.exception("could not record a change")
