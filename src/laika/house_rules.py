"""House rules: rooms Laika must not drive into by herself (#51).

The rules are what people told her ("non entrare in camera", "non andare in cucina", "stai fuori
dallo studio": facts of kind "rule" in memory) plus `[house] no_go_rooms`. A rule that names a room is
read from the user's own words with a few Italian patterns; Italian room words are mapped to the room
names the world model uses (the simulator's living_room, kitchen, office, bedroom), and any other
room name is kept as it is spoken, so a room the vision model calls "cucina" or "garage" matches too.
Rules that are not about a room ("non toccare i giocattoli") are ignored here. `forget` on the fact
lifts the rule: nothing is cached.

Honest limit: Laika has no map. She knows which room she is in only from what she sees (or, in the
simulator, from ground truth), so she finds out she has entered a no-go room after the fact, and she
cannot plan a route around one. The safety controller refuses her autonomous forward moves while the
world model says she is in a no-go room, and the explorer reverses out (docs/behaviours.md).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from laika.observations import fold

if TYPE_CHECKING:
    from laika.config import HouseConfig
    from laika.memory import Memory
    from laika.world_model import WorldModel

log = logging.getLogger(__name__)

MAX_ROOM_WORDS = 3
MAX_FOLLOWERS = 4

# Italian room words (folded) and the room names the world model and the simulator use.
ROOM_WORDS = {
    "camera da letto": "bedroom", "stanza da letto": "bedroom", "camera": "bedroom",
    "cucina": "kitchen",
    "soggiorno": "living_room", "salotto": "living_room", "sala": "living_room",
    "studio": "office", "ufficio": "office",
    "bagno": "bathroom",
    "ingresso": "entrance",
}
_ARTICLES = frozenset("il lo la l i gli le un uno una".split())
_PREPOSITIONS = frozenset("in nel nella nello nell a al alla allo all da dal dalla dallo dall di dentro sul sulla".split())

# "non entrare in X", "non devi mai andare nella X", "non puoi piu venire in X", "stai fuori dallo X".
_ENTRY = re.compile(
    r"\b(?:non|mai)\s+(?:(?:devi|dovresti|puoi|bisogna|piu|mai)\s+)*"
    r"(?:entrare|andare|andarci|venire|passare|salire|girare)\s+(?:(?:piu|mai)\s+)?"
    r"(?:in|nel|nella|nello|nell|a|al|alla|dentro|sul|sulla)\s+(?P<place>.+)$"
)
_OUTSIDE = re.compile(
    r"\b(?:stai|resta|restare|stare|rimani)\s+fuori\s+(?:da|dal|dallo|dalla|dall|di)\s+(?P<place>.+)$"
)
_SPLIT = re.compile(r"\s+(?:e|o)\s+|\s*[,;]\s*")
_STOP_AT = re.compile(r"\s+(?:per favore|per piacere|mai|ok|grazie|laika|quando|se|perche|finche)\b.*$")


def canonical_room(name: str) -> str | None:
    """One id for a room spoken or stored in any form: the world model's name when it is a known Italian
    word, else the words as spoken with underscores. None when it is not a plausible room name."""
    words = re.sub(r"['’`]", " ", fold(name)).split()
    while words and words[0] in _PREPOSITIONS | _ARTICLES:
        words.pop(0)
    if not words:
        return None
    phrase = " ".join(words)
    for key in sorted(ROOM_WORDS, key=len, reverse=True):
        if phrase == key or phrase.startswith(key + " "):
            return ROOM_WORDS[key]
    if len(words) > MAX_ROOM_WORDS or not all(re.fullmatch(r"[a-z0-9']+", w) for w in words):
        return None
    return "_".join(words)


def rooms_in_rule(text: str) -> set[str]:
    """The rooms an entry-forbidding rule names (empty for any other rule)."""
    folded = " ".join(re.sub(r"[.!?\"()'’`]", " ", fold(text)).split())
    for pattern in (_ENTRY, _OUTSIDE):
        match = pattern.search(folded)
        if match is None:
            continue
        place = _STOP_AT.sub("", match.group("place"))
        rooms = {canonical_room(piece) for piece in _SPLIT.split(place) if piece.strip()}
        return {room for room in rooms if room}
    return set()


class HouseRules:
    """The no-go rooms, read from memory and the settings every time they are asked (nothing is cached)."""

    def __init__(self, memory: Memory, world: WorldModel, config: HouseConfig | None = None) -> None:
        self.memory = memory
        self.world = world
        self._configured = {room for room in (canonical_room(r) for r in (config.no_go_rooms if config else ())) if room}

    def no_go_rooms(self) -> set[str]:
        rooms = set(self._configured)
        for fact in self.memory.facts.of_kind("rule"):
            found = rooms_in_rule(fact.text)
            if not found and fact.subject and _forbids_entry(fact.text):
                room = canonical_room(fact.subject)  # "non entrare" with the room as the subject
                found = {room} if room else set()
            rooms |= found
        return rooms

    def is_no_go(self, room: str | None) -> bool:
        key = canonical_room(room) if room else None
        return key is not None and key in self.no_go_rooms()

    def blocked_room(self) -> str | None:
        """The room she is in now when it is a no-go room, else None (also when she does not know where she is)."""
        room = self.world.current_room
        return room if room and self.is_no_go(room) else None

    def count(self) -> int:
        return len(self.no_go_rooms())

    def record_turned_back(self, room: str) -> None:
        """Remember in episodic memory that she backed out of a room she must not enter."""
        name = room.replace("_", " ")
        self.memory.episodes.record("turned_back", f"Turned back from the {name}: a house rule says not to enter", [room], room)


def _forbids_entry(text: str) -> bool:
    """A rule that forbids going somewhere without saying where ("non entrare"): the subject may be the room."""
    folded = " ".join(fold(text).split())
    return bool(re.search(r"\b(?:non|mai)\s+(?:\w+\s+){0,2}(?:entrare|andare|venire)\b|\bstai fuori\b", folded))
