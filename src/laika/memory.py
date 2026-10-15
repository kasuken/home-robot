"""What Laika knows and what happened: semantic and episodic memory (#34, docs/prd.md "Memory").

`SemanticMemory` holds facts people told her or she worked out ("the black backpack stays near the
entrance"). `EpisodicMemory` holds events ("seen", "asked", "answered", ...). Both live in memory
within a bound and are mirrored to the store (`Store.put_fact`, `Store.put_episode`) through its
world-priority queue, so they survive a restart and are not dropped for a flood of observations.

Facts: text is kept as the user said it, in any language. Replacing and adding follow one rule:

* subject + kind "location" or "ownership": a new fact REPLACES the old one with the same
  subject, kind and source (a usual place or an owner changes, it does not pile up); the id and
  creation time stay;
* any other kind, or no subject: the fact is ADDED, and one with the same text (compared folded) only
  has its time and confidence refreshed.

At most `max_facts` are kept; beyond that the least recently updated one is evicted (and deleted from
the store). Nothing else removes a fact except `forget`.

Episodes: the newest `episode_window` are kept in memory; the store keeps the history within
`[storage] max_episodes` and `max_episode_days`. `EpisodeRecorder` turns observations into "seen"
episodes, at most one per object and room per interval, so a camera staring at a sofa stays cheap.
All matching is case-, accent- and underscore-insensitive (`observations.fold`).
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

from laika.events import Event, EventBus, EventType
from laika.observations import fold, normalise

if TYPE_CHECKING:
    from laika.store import Store

log = logging.getLogger(__name__)

MAX_FACTS = 500
EPISODE_WINDOW = 200
MAX_TEXT_CHARS = 300
MAX_SUMMARY_CHARS = 200
SEEN_EPISODE_INTERVAL_S = 3600.0
MAX_SEEN_KEYS = 2000
MAX_SEEN_PER_OBSERVATION = 20

FACT_KINDS = ("location", "ownership", "preference", "rule", "general")
FACT_SOURCES = ("user", "vision", "investigation")
EPISODE_KINDS = ("seen", "moved", "conversation", "question", "answer", "rule")
REPLACING_KINDS = frozenset({"location", "ownership"})
# Words that say nothing about which fact is relevant to a message (Italian and English).
STOPWORDS = frozenset(
    "il lo la le gli un una uno di a da in con su per tra fra che chi cosa come dove quando quale "
    "e o ma se non si mi ti ci vi sono sei era ero del della dei delle al alla ai alle nel nella "
    "the an of to is are was were do does did where what who how when it this that and or in on at "
    "my me you your i ho hai ha hanno mio mia tuo tua suo sua ieri oggi".split()
)


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[^\W_]+", fold(text)) if len(w) >= 3 and w not in STOPWORDS}


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


# --- facts ------------------------------------------------------------------


@dataclass
class Fact:
    id: str
    text: str
    subject: str | None
    kind: str
    source: str
    created: float
    updated: float
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "text": self.text, "subject": self.subject, "kind": self.kind,
            "source": self.source, "created": self.created, "updated": self.updated,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Fact:
        subject = raw.get("subject")
        kind, source = str(raw.get("kind", "general")), str(raw.get("source", "user"))
        return cls(
            id=str(raw["id"]),
            text=_clip(str(raw["text"]), MAX_TEXT_CHARS),
            subject=normalise(subject) if isinstance(subject, str) and subject.strip() else None,
            kind=kind if kind in FACT_KINDS else "general",
            source=source if source in FACT_SOURCES else "user",
            created=float(raw["created"]),
            updated=float(raw["updated"]),
            confidence=float(raw.get("confidence", 1.0)),
        )


class SemanticMemory:
    """Bounded, thread-safe facts, written through to the store when there is one."""

    def __init__(
        self,
        store: Store | None = None,
        *,
        max_facts: int = MAX_FACTS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self.max_facts = max(1, max_facts)
        self._clock = clock
        self._lock = threading.Lock()
        self._facts: OrderedDict[str, Fact] = OrderedDict()  # least recently updated first

    def load(self) -> int:
        """Fill from the store at startup, silently. Returns how many facts were loaded."""
        if self.store is None:
            return 0
        with self._lock:
            self._facts.clear()
            for raw in self.store.load_facts(self.max_facts):
                try:
                    fact = Fact.from_dict(raw)
                except (KeyError, TypeError, ValueError):
                    log.warning("memory: skipping an unreadable fact %r", raw)
                    continue
                self._facts[fact.id] = fact
            return len(self._facts)

    def remember(
        self,
        text: str,
        subject: str | None = None,
        kind: str | None = None,
        source: str = "user",
        confidence: float = 1.0,
    ) -> Fact:
        """Store a fact, replacing or adding as the module docstring says. Returns the stored fact."""
        text = _clip(text, MAX_TEXT_CHARS)
        key = normalise(subject) if subject and subject.strip() else None
        kind = kind if kind in FACT_KINDS else "general"
        source = source if source in FACT_SOURCES else "user"
        now = self._clock()
        with self._lock:
            same = self._matching(text, key, kind, source)
            if same:
                fact, *extras = same
                fact.text, fact.updated, fact.confidence = text, now, confidence
                for extra in extras:
                    self._drop(extra.id)
                self._facts.move_to_end(fact.id)
            else:
                fact = Fact(f"f-{uuid.uuid4().hex[:8]}", text, key, kind, source, now, now, confidence)
                self._facts[fact.id] = fact
            self._persist(fact)
            while len(self._facts) > self.max_facts:
                self._drop(next(iter(self._facts)))
            return Fact(**fact.to_dict())

    def _matching(self, text: str, key: str | None, kind: str, source: str) -> list[Fact]:
        if key is not None and kind in REPLACING_KINDS:
            return [f for f in self._facts.values()
                    if f.subject == key and f.kind == kind and f.source == source]
        folded = fold(text)
        return [f for f in self._facts.values() if fold(f.text) == folded and f.subject == key]

    def _persist(self, fact: Fact) -> None:
        if self.store is not None:
            self.store.put_fact(fact.to_dict())

    def _drop(self, fact_id: str) -> None:
        if self._facts.pop(fact_id, None) is not None and self.store is not None:
            self.store.delete_fact(fact_id)

    def forget(self, target: str) -> list[Fact]:
        """Remove the fact with this id, or every fact about this subject. Returns what was removed."""
        wanted = target.strip()
        key = fold(wanted)
        with self._lock:
            found = [f for f in self._facts.values()
                     if f.id == wanted or (f.subject is not None and fold(f.subject) == key)]
            for fact in found:
                self._drop(fact.id)
            return found

    def search(self, query: str, limit: int = 8) -> list[Fact]:
        """Facts for `query`: subject matches first, then text substrings; newest first within each."""
        needle = fold(query)
        if not needle:
            return []
        ranked: list[tuple[int, float, Fact]] = []
        with self._lock:
            for fact in self._facts.values():
                rank = self._rank(fact, needle)
                if rank is not None:
                    ranked.append((rank, -fact.updated, Fact(**fact.to_dict())))
        ranked.sort(key=lambda item: item[:2])
        return [fact for _, _, fact in ranked[: max(0, limit)]]

    @staticmethod
    def _rank(fact: Fact, needle: str) -> int | None:
        subject = fold(fact.subject) if fact.subject else ""
        if subject and subject == needle:
            return 0
        if subject and (needle in subject or (len(subject) >= 3 and subject in needle)):
            return 1
        if needle in fold(fact.text):
            return 2
        return None

    def mentions(self, text: str) -> bool:
        """Does any fact concern `text` (as its subject or in its words)? Used for user relevance."""
        needle = fold(text)
        if not needle:
            return False
        with self._lock:
            return any(self._rank(f, needle) is not None for f in self._facts.values())

    def relevant(self, message: str, limit: int = 5) -> list[Fact]:
        """The facts that share the most meaningful words with `message`, best first."""
        words = _words(message)
        if not words:
            return []
        scored: list[tuple[int, float, Fact]] = []
        with self._lock:
            for fact in self._facts.values():
                score = 2 * len(words & _words(fact.subject or "")) + len(words & _words(fact.text))
                if score:
                    scored.append((score, fact.updated, Fact(**fact.to_dict())))
        scored.sort(key=lambda item: (-item[0], -item[1]))
        return [fact for _, _, fact in scored[: max(0, limit)]]

    def all(self) -> list[Fact]:
        with self._lock:
            return [Fact(**f.to_dict()) for f in self._facts.values()]

    def __len__(self) -> int:
        with self._lock:
            return len(self._facts)


# --- episodes ---------------------------------------------------------------


@dataclass(frozen=True)
class Episode:
    id: str
    timestamp: float
    kind: str
    summary: str
    subjects: tuple[str, ...] = ()
    location: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "timestamp": self.timestamp, "kind": self.kind, "summary": self.summary,
            "subjects": list(self.subjects), "location": self.location,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Episode:
        return cls(
            id=str(raw["id"]),
            timestamp=float(raw["timestamp"]),
            kind=str(raw["kind"]),
            summary=str(raw["summary"]),
            subjects=tuple(str(s) for s in raw.get("subjects", ())),
            location=raw.get("location"),
        )

    def concerns(self, needle: str) -> bool:
        """Is `needle` (already folded) one of the subjects, or part of one?"""
        for subject in self.subjects:
            folded = fold(subject)
            if needle in folded or (len(folded) >= 3 and folded in needle):
                return True
        return False


class EpisodicMemory:
    """Events with a time. A recent window is in memory; the store keeps the history."""

    def __init__(
        self,
        store: Store | None = None,
        *,
        window: int = EPISODE_WINDOW,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self._clock = clock
        self._lock = threading.Lock()
        self._recent: deque[Episode] = deque(maxlen=max(1, window))

    def load(self) -> int:
        """Fill the window from the store at startup. Returns how many episodes were loaded."""
        if self.store is None:
            return 0
        with self._lock:
            self._recent.clear()
            for raw in self.store.recent_episodes(self._recent.maxlen or EPISODE_WINDOW):
                try:
                    self._recent.append(Episode.from_dict(raw))
                except (KeyError, TypeError, ValueError):
                    log.warning("memory: skipping an unreadable episode %r", raw)
            return len(self._recent)

    def record(
        self,
        kind: str,
        summary: str,
        subjects: Iterable[str] = (),
        location: str | None = None,
        timestamp: float | None = None,
    ) -> Episode:
        """Remember that something happened. `summary` is short English; `subjects` are names."""
        names = tuple(dict.fromkeys(n for n in (normalise(s) for s in subjects if s) if n))
        room = normalise(location) if location and location.strip() else None
        episode = Episode(
            f"ep-{uuid.uuid4().hex[:10]}",
            self._clock() if timestamp is None else timestamp,
            kind if kind in EPISODE_KINDS else "conversation",
            _clip(summary, MAX_SUMMARY_CHARS),
            names,
            room,
        )
        with self._lock:
            self._recent.append(episode)
        if self.store is not None:
            self.store.put_episode(episode.to_dict())
        return episode

    def record_moved(
        self,
        name: str,
        from_room: str | None,
        to_room: str | None,
        timestamp: float | None = None,
    ) -> Episode:
        """An object changed room. The seam for environment-change detection (#36)."""
        where = f" from the {_room(from_room)}" if from_room else ""
        there = f" to the {_room(to_room)}" if to_room else ""
        return self.record(
            "moved", f"{name} moved{where}{there}", [name], to_room, timestamp
        )

    def recent(self, limit: int = 20) -> list[Episode]:
        """The newest episodes in the window, newest first."""
        with self._lock:
            return list(reversed(self._recent))[: max(0, limit)]

    def about(self, subject: str, limit: int = 5) -> list[Episode]:
        """The newest episodes concerning `subject`, newest first: the window plus the stored history."""
        needle = fold(subject)
        if not needle:
            return []
        with self._lock:
            found = {e.id: e for e in self._recent if e.concerns(needle)}
        if self.store is not None:
            for raw in self.store.episodes_about(needle, limit):
                try:
                    episode = Episode.from_dict(raw)
                except (KeyError, TypeError, ValueError):
                    continue
                found.setdefault(episode.id, episode)
        newest = sorted(found.values(), key=lambda e: e.timestamp, reverse=True)
        return newest[: max(0, limit)]

    def __len__(self) -> int:
        with self._lock:
            return len(self._recent)


def _room(name: str | None) -> str:
    return (name or "unknown room").replace("_", " ")


class Memory:
    """Semantic and episodic memory together, over one store (or none, for tests and no-store runs)."""

    def __init__(
        self,
        store: Store | None = None,
        *,
        max_facts: int = MAX_FACTS,
        episode_window: int = EPISODE_WINDOW,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.facts = SemanticMemory(store, max_facts=max_facts, clock=clock)
        self.episodes = EpisodicMemory(store, window=episode_window, clock=clock)

    def load(self) -> tuple[int, int]:
        """Restore both from the store: (facts, episodes in the window)."""
        return self.facts.load(), self.episodes.load()

    def mentions(self, text: str) -> bool:
        return self.facts.mentions(text)

    def __len__(self) -> int:
        return len(self.facts)


# --- recording sightings ----------------------------------------------------


class EpisodeRecorder:
    """Records a "seen" episode per object and room, at most once per interval (bus thread, cheap)."""

    def __init__(
        self,
        episodes: EpisodicMemory,
        bus: EventBus,
        interval_s: float = SEEN_EPISODE_INTERVAL_S,
    ) -> None:
        self.episodes = episodes
        self.interval_s = max(0.0, interval_s)
        self._last: OrderedDict[tuple[str, str | None], float] = OrderedDict()
        self._unsubscribe = bus.subscribe(self._on_event, EventType.OBSERVATION_CREATED)

    def close(self) -> None:
        self._unsubscribe()

    def _on_event(self, event: Event) -> None:
        room = event.data.get("location")
        room = normalise(room) if isinstance(room, str) and room.strip() else None
        for raw in list(event.data.get("objects", ()))[:MAX_SEEN_PER_OBSERVATION]:
            name = raw.get("name") if isinstance(raw, Mapping) else raw
            if not isinstance(name, str) or not name.strip():
                continue
            name = normalise(name)
            key = (name, room)
            last = self._last.get(key)
            if last is not None and 0 <= event.timestamp - last < self.interval_s:
                continue
            self._last[key] = event.timestamp
            self._last.move_to_end(key)
            while len(self._last) > MAX_SEEN_KEYS:
                self._last.popitem(last=False)
            where = f" in the {_room(room)}" if room else ""
            self.episodes.record("seen", f"Saw {name}{where}", [name, *([room] if room else [])],
                                 room, event.timestamp)


# --- telling the time -------------------------------------------------------


def iso_minutes(timestamp: float) -> str:
    """Absolute local time, minute precision, with the UTC offset: 2026-10-15T14:30+02:00."""
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="minutes")


def relative_time(timestamp: float, now: float) -> str:
    """An English hint ("2 hours ago", "yesterday") the model can phrase in Italian."""
    delta = now - timestamp
    if delta < 0:
        return "in the future"
    if delta < 60:
        return "just now"
    if delta < 3600:
        minutes = int(delta // 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    then, today = datetime.fromtimestamp(timestamp), datetime.fromtimestamp(now)
    days = (today.date() - then.date()).days
    if days == 0 or delta < 6 * 3600:
        hours = int(delta // 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    if days == 1:
        return "yesterday"
    if days < 14:
        return f"{days} days ago"
    if days < 60:
        return f"{days // 7} weeks ago"
    return f"{days // 30} months ago"
