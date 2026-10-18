"""Bedtime stories (the owner's ask): "Laika, raccontami una storia della buonanotte".

Every story is new: the `bedtime_story` tool asks the model for one with the owner's prompt
(`STORY_PROMPT`, used word for word as the system message: Leo and Ludo, two bunny siblings, for
children from 2 to 7) and Laika's own constraints in the user message: Italian plain text, a title, short
paragraphs, punctuation for the pauses and the emphasis (the only emphasis a voice reliably renders),
`STORY_WORDS` words for three to five minutes, and not one of the recent titles.

The story is written by its own client (`client`), with a longer timeout than the conversation's: a long
story takes a while. While it is written she says a short line, so the child does not wait in silence.
The story is saved under `stories/` before she tells it (an interrupted story is kept too), then told
one paragraph per call to the voice with a pause between them. Between paragraphs she checks whether
the turn was cancelled and stops there.

At the end she asks, in code, whether to tell another one, and the conversation waits `ANSWER_WAIT_S`
for the answer; the model's own reply for that turn is not spoken, so the question is the last thing
she said. Silence ends the conversation: the child is probably asleep.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from laika.ai_client import AIUnavailable, system_message, user_message
from laika.storage import FILE_MODE, _chmod, prune_directory

log = logging.getLogger(__name__)

# The owner's prompt, word for word.
STORY_PROMPT = (
    "Sei un esperto scrittore di storie per bambini da 2 a 7 anni.\n"
    "Sei specializzato in storie della buonanotte.\n"
    "\n"
    "I protagonisti sono due coniglietti fratello e sorella: Leo e Ludo."
)
# Three to five minutes told slowly (about 120-130 words a minute). To be tuned once heard on the robot.
STORY_WORDS = (450, 650)
STORY_TIMEOUT_S = 90.0  # writing a long story takes longer than a conversation reply
ANSWER_WAIT_S = 15.0  # how long she waits for "another one?"
PARAGRAPH_PAUSE_S = 0.8
MAX_CHUNK_CHARS = 450  # a paragraph longer than this is told in pieces of whole sentences
RECENT_TITLES = 10
MAX_THEME_CHARS = 80
# "Save every story", within the bounded-storage rule: a story is about 4 KB, so this is years of evenings.
MAX_STORIES = 1000
MAX_STORIES_BYTES = 20 * 1024 * 1024
STORY_PATTERN = "story-*.txt"

THINKING = "Mmm... fammi pensare a una storia nuova per Leo e Ludo..."
ANOTHER = "Ti è piaciuta? Vuoi che ti racconti un'altra storia?"
FALLBACK_TITLE = "Una storia di Leo e Ludo"


@dataclass(frozen=True)
class Story:
    title: str
    paragraphs: tuple[str, ...]

    @property
    def words(self) -> int:
        return sum(len(p.split()) for p in self.paragraphs)

    def to_text(self) -> str:
        return self.title + "\n\n" + "\n\n".join(self.paragraphs) + "\n"


def story_request(recent_titles: list[str], theme: str | None = None) -> str:
    """The user message: Laika's constraints around the owner's prompt."""
    low, high = STORY_WORDS
    lines = [
        "Scrivi una storia della buonanotte nuova, mai raccontata prima, con Leo e Ludo.",
        f"Lunghezza: tra {low} e {high} parole, da leggere ad alta voce in tre-cinque minuti, con calma.",
        "Scrivi in italiano, solo testo semplice: niente markdown, niente elenchi, niente emoji.",
        "La prima riga è \"Titolo: \" seguito dal titolo. Poi la storia, in paragrafi brevi separati da una riga vuota.",
        "Frasi corte e chiare. Usa la punteggiatura per le pause e l'enfasi (\"...\", \"!\", \"?\"), "
        "qualche ripetizione dolce e qualche suono (shhh, toc toc).",
        "Un piccolo problema, risolto con gentilezza; niente paura, niente cattivi. "
        "Il finale è tranquillo: Leo e Ludo si addormentano felici.",
    ]
    if theme:
        lines.append(f"Il bambino ha chiesto una storia con: {theme}.")
    if recent_titles:
        lines.append("Deve essere diversa da queste storie già raccontate: " + "; ".join(recent_titles) + ".")
    return "\n".join(lines)


_MARKUP = re.compile(r"[*_#`>]+")
_TITLE = re.compile(r"^\s*titolo\s*:\s*", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def parse_story(text: str) -> Story:
    """Title and paragraphs from the model's text; markdown stripped, long paragraphs cut by sentences."""
    lines = [_MARKUP.sub("", line).strip() for line in text.replace("\r\n", "\n").split("\n")]
    title = ""
    body: list[str] = []
    for line in lines:
        if not title and not body and line:
            if _TITLE.match(line):
                title = _TITLE.sub("", line).strip().strip("\"'«»“”")
                continue
            if len(line) <= 60 and not line.endswith((".", "!", "?", "…")):
                title = line  # a bare title line
                continue
        body.append(line)
    paragraphs: list[str] = []
    current: list[str] = []
    for line in body + [""]:
        if line:
            current.append(line)
        elif current:
            paragraphs.extend(_chunks(" ".join(current)))
            current = []
    return Story(title or FALLBACK_TITLE, tuple(paragraphs))


def _chunks(paragraph: str) -> list[str]:
    if len(paragraph) <= MAX_CHUNK_CHARS:
        return [paragraph]
    out: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(paragraph):
        if current and len(current) + 1 + len(sentence) > MAX_CHUNK_CHARS:
            out.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        out.append(current)
    return out


class StoryArchive:
    """The stories told, one text file each under the data directory's `stories/`."""

    def __init__(self, directory: Path, clock: Callable[[], float] = time.time) -> None:
        self.directory = directory
        self._clock = clock
        self._lock = threading.Lock()

    def save(self, story: Story) -> Path | None:
        """Write the story and prune. None when it could not be written (she tells it anyway)."""
        with self._lock:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
            except OSError:
                log.warning("could not create %s", self.directory, exc_info=True)
                return None
            moment = datetime.fromtimestamp(self._clock())
            target = self.directory / f"story-{moment:%Y%m%d-%H%M%S}.txt"
            n = 1
            while target.exists():
                n += 1
                target = self.directory / f"story-{moment:%Y%m%d-%H%M%S}-{n}.txt"
            temp = self.directory / f".{target.name}.tmp"
            try:
                temp.write_text(story.to_text(), encoding="utf-8")
                _chmod(temp, FILE_MODE)
                os.replace(temp, target)
            except OSError:
                log.warning("could not save story %s", target, exc_info=True)
                temp.unlink(missing_ok=True)
                return None
            prune_directory(self.directory, MAX_STORIES, MAX_STORIES_BYTES, STORY_PATTERN)
            return target

    def recent_titles(self, limit: int = RECENT_TITLES) -> list[str]:
        """The titles of the newest stories, newest first."""
        files = sorted(self.directory.glob(STORY_PATTERN), key=lambda f: f.name, reverse=True)[:limit]
        titles = []
        for file in files:
            try:
                with file.open(encoding="utf-8") as handle:
                    title = handle.readline().strip()
            except OSError:
                continue
            if title:
                titles.append(title)
        return titles


class BedtimeStories:
    """Writes, saves and tells a new story; remembers that she just asked for another. Safe from any thread."""

    def __init__(
        self,
        client: Any,
        archive: StoryArchive,
        speak: Callable[[str], None],
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client  # a ChatClient (only `complete` is used)
        self.archive = archive
        self.speak = speak
        self._sleep = sleep
        self._lock = threading.Lock()
        self._asked = False  # she asked "another one?" and nobody answered yet
        self._quiet_reply = False  # the model's reply to this turn must not follow the question

    def tell(self, *, theme: str | None = None, should_stop: Callable[[], bool] | None = None) -> dict[str, Any]:
        self.speak(THINKING)
        theme = (theme or "").strip()[:MAX_THEME_CHARS] or None
        messages = [system_message(STORY_PROMPT), user_message(story_request(self.archive.recent_titles(), theme))]
        try:
            answer = self.client.complete(messages)
        except AIUnavailable as error:
            print(f"[laika] no bedtime story: {error}")
            return {"ok": False, "error": "could not write the story: say sorry and offer a song instead"}
        if answer.finish_reason == "length":
            print("[laika] the bedtime story was cut short by the model's output limit")
        story = parse_story(answer.content or "")
        if not story.paragraphs:
            return {"ok": False, "error": "the story came back empty: say sorry and offer a song instead"}
        path = self.archive.save(story)  # before telling it: an interrupted story is kept too
        print(f"[laika] bedtime story: {story.title} ({story.words} words)")
        result: dict[str, Any] = {"ok": True, "title": story.title, "words": story.words, "saved": path is not None}
        if should_stop is not None and should_stop():  # cancelled while it was being written
            result.update(ok=False, error="interrupted", paragraphs_told=0)
            return result
        self.speak(f"La storia di stasera si intitola: {story.title}.")
        told = 0
        for paragraph in story.paragraphs:
            if should_stop is not None and should_stop():
                result.update(ok=False, error="interrupted", paragraphs_told=told)
                return result
            self._sleep(PARAGRAPH_PAUSE_S)
            self.speak(paragraph)
            told += 1
        self._sleep(PARAGRAPH_PAUSE_S)
        self.speak(ANOTHER)
        with self._lock:
            self._asked = True
            self._quiet_reply = True
        result["paragraphs_told"] = told
        result["asked"] = ANOTHER
        result["note"] = (
            "story told and the question already asked: don't repeat or summarise it, your reply is not spoken. "
            "If they want another one, call bedtime_story again; if not, wish them buonanotte"
        )
        return result

    def take_quiet_reply(self) -> bool:
        """True once after a story: the conversation does not speak the model's reply to that turn."""
        with self._lock:
            quiet, self._quiet_reply = self._quiet_reply, False
            return quiet

    def follow_up_s(self, default: float) -> float:
        """Exactly `ANSWER_WAIT_S` for the answer to "another one?", once; `default` otherwise."""
        with self._lock:
            asked, self._asked = self._asked, False
        return ANSWER_WAIT_S if asked else default

    def end(self) -> None:
        with self._lock:
            self._asked = self._quiet_reply = False
