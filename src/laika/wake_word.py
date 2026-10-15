"""Recognise "ehi Laika" in whatever the speech recogniser actually heard.

Vosk's small Italian model has a fixed vocabulary without "Laika", so it
writes the closest words it knows ("laica", "l'aica", "lai ca"...). Instead of
an exact match, any word or pair of adjacent words close enough to a wake name
counts. Words after the name are kept, so "ehi Laika, quanti anni hai?" works
in a single breath.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

FILLER_WORDS = {"ehi", "hey", "ehy", "ei", "hei", "e", "oh", "ok", "okay"}
MIN_SIMILARITY = 0.8
MIN_LENGTH = 4

_WORD = re.compile(r"\w+", re.UNICODE)


def normalise(word: str) -> str:
    decomposed = unicodedata.normalize("NFKD", word.lower())
    return "".join(c for c in decomposed if c.isalnum())


@dataclass(frozen=True)
class WakeMatch:
    heard: str
    remainder: str


class WakeWordMatcher:
    def __init__(self, names: list[str], min_similarity: float = MIN_SIMILARITY) -> None:
        self.names = [normalise(n) for n in names if normalise(n)]
        self.min_similarity = min_similarity

    def _similar(self, candidate: str) -> bool:
        if len(candidate) < MIN_LENGTH:
            return False
        return any(
            SequenceMatcher(None, candidate, name).ratio() >= self.min_similarity
            for name in self.names
        )

    def match(self, text: str | None) -> WakeMatch | None:
        if not text:
            return None
        words = _WORD.findall(text)
        normalised = [normalise(w) for w in words]

        name_at: tuple[int, int] | None = None
        for i, word in enumerate(normalised):
            if self._similar(word):
                name_at = (i, i + 1)
                break
            if i + 1 < len(normalised) and self._similar(word + normalised[i + 1]):
                name_at = (i, i + 2)
                break
        if name_at is None:
            return None

        start, end = name_at
        before = [w for w, n in zip(words[:start], normalised[:start]) if n not in FILLER_WORDS]
        remainder = " ".join(before + words[end:]).strip()
        return WakeMatch(heard=text, remainder=remainder)
