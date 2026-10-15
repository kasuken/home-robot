"""What a conversation leaves in long-term memory (#35, docs/prd.md "Conversation memory").

Not every message becomes memory. The model decides first: its prompt says to call `remember`
(or `answer_question`) when someone tells her what an object is, where it stays, who owns it, a
preference or a house rule. `MemorySelector` is the safety net that runs after each completed
turn, in tool mode and in the JSON fallback mode alike: when the model stored nothing, a few
deterministic Italian patterns look at the user's words and keep what is worth keeping.

Each pattern gives a candidate with a confidence; only candidates at or above
`[memory] min_confidence` are stored. Nothing is stored for questions, greetings, one- or
two-word remarks (unless they answer a question Laika is waiting to have answered), requests to
forget, or when `[memory] auto_select` is false. Matching is case- and accent-insensitive
(`observations.fold`); the stored text keeps the user's own words.

`store_user_fact` and `record_answer` are the one code path for putting a told fact (and an
answer to a pending question) into memory, shared with the `remember` and `answer_question`
tools: an object someone told her about becomes a known object in the world model, with a note.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

from laika.config import MemoryConfig
from laika.memory import Fact, Memory, _clip
from laika.observations import fold, normalise
from laika.world_model import WorldModel

if TYPE_CHECKING:
    from laika.investigate import PendingQuestion, PendingQuestions

# Tools that already stored (or deliberately removed) something this turn.
MEMORY_TOOLS = frozenset({"remember", "answer_question", "forget"})
MAX_SHOWN_QUESTIONS = 3
MIN_WORDS = 3
MAX_NOTE_CHARS = 120
# Confidences: how sure a pattern is that the words are worth keeping.
ANSWER_CONFIDENCE = 0.85
NOUN_ANSWER_CONFIDENCE = 0.7
RULE_CONFIDENCE = 0.9
STRONG_LOCATION_CONFIDENCE = 0.85
LOCATION_CONFIDENCE = 0.65
OWNERSHIP_CONFIDENCE = 0.8
NAME_CONFIDENCE = 0.8
PREFERENCE_CONFIDENCE = 0.75
IDENTITY_CONFIDENCE = 0.7  # "this is a X" about the one thing in view
GUESS_CONFIDENCE = 0.5  # too uncertain to keep unless the owner lowers the threshold

DEMO = r"(?:questo|questa|quest|quello|quella|quel|quell)"
ARTICLE = r"(?:il|lo|la|l|i|gli|le|un|uno|una|dei|delle|degli)"
PLACE_PREP = (
    r"(?:in|nel|nella|nello|nell|vicino a|vicino al|vicino alla|vicino allo|vicino all|accanto a|"
    r"accanto al|accanto alla|accanto all|sotto|sopra|dietro|davanti a|davanti al|davanti alla|"
    r"davanti all|su|sul|sulla)"
)
USUALLY = r"(?:di solito|sempre|normalmente)"

IT_IS = re.compile(rf"^(?:(?P<subj>.+?)\s+)?(?:e|sarebbe)\s+(?P<what>{ARTICLE}\s+\S.*)$")
DEMONSTRATIVE = re.compile(rf"^{DEMO}$")
NAMED = re.compile(r"^(?:(?P<subj>.+?)\s+)?si chiama\s+(?P<name>\S+(?: \S+){0,2})$")
OWNED_BY = re.compile(r"^(?:(?P<subj>.+?)\s+)?(?:e|sono)\s+di\s+(?P<owner>\S+(?: \S+){0,2})$")
BELONGS_TO = re.compile(
    r"^(?:(?P<subj>.+?)\s+)?(?:appartiene|appartengono)\s+a\s+(?P<owner>\S+(?: \S+){0,2})$"
)
LOCATION = re.compile(
    rf"^(?:(?P<subj>.+?)\s+)?(?:(?P<usually>{USUALLY})\s+)?(?P<verb>sta|stanno|resta|restano|rimane|"
    rf"rimangono|e|sono)\s+(?:(?P<usually2>{USUALLY})\s+)?(?P<prep>{PLACE_PREP})\s+(?P<place>.+)$"
)
RULE = re.compile(
    r"^(?:ti prego di\s+)?non\s+(?:(?:devi|dovresti|puoi|bisogna)\s+)?(?:piu\s+)?"
    r"(?:entrare|andare|andarci|venire|passare|salire)\s+(?:in|nel|nella|nello|nell|a|al|alla|sul|sulla)\s+"
    r"(?P<place>.+)$"
)
RULE_OUTSIDE = re.compile(
    r"^(?:ti prego di\s+)?(?:stai|resta|restare|stare|rimani)\s+fuori\s+(?:da|dal|dallo|dalla|dall)\s+(?P<place>.+)$"
)
PREFERENCE = re.compile(
    r"^(?:a me\s+)?(?:mi piace|mi piacciono|non mi piace|non mi piacciono|preferisco|adoro|amo|odio|"
    r"detesto)\s+(?!come\b|quando\b|che\b|se\b)\S.*$"
)

# Italian names of rooms (folded) and the room names the world model uses.
ROOMS = {
    "soggiorno": "living_room", "salotto": "living_room", "cucina": "kitchen",
    "studio": "office", "ufficio": "office", "camera": "bedroom",
    "camera da letto": "bedroom", "stanza da letto": "bedroom",
}
GREETINGS = frozenset(
    "ciao salve buongiorno buonasera buonanotte grazie prego ok va bene si no ehi hey arrivederci".split()
)
QUESTION_WORDS = frozenset(
    "cosa cos che chi dove come quando perche quanto quanta quanti quale quali sai puoi potresti "
    "riesci vuoi hai".split()
)
FORGET_WORDS = ("dimentica", "dimenticati", "dimenticare", "scordati", "scordare", "cancella")
DISMISSALS = frozenset("no non boh niente nulla nessuno".split())
# Words that make an owner phrase a material or a place, not a person ("e di legno").
NOT_OWNERS = frozenset(
    "legno plastica metallo ferro vetro carta stoffa pelle cuoio lana cotone gomma marmo qui li la "
    "tutti nessuno niente".split()
)
# A reply that starts with an article but holds one of these is a sentence, not a noun phrase.
SENTENCE_WORDS = frozenset("sta sono ho hai ha vai va fai fa puoi devi vedi guarda mi ti ci lo".split())
NOUN_REPLY_START = frozenset("un uno una dei delle degli il lo la".split())


# --- shared writes ------------------------------------------------------------------------------


def _pending_for(questions: PendingQuestions | None, subject: str) -> PendingQuestion | None:
    if questions is None:
        return None
    key = normalise(subject)
    return next((q for q in questions.pending() if q.object == key), None)


def record_answer(
    world: WorldModel,
    memory: Memory,
    question: PendingQuestion,
    answer: str,
    now: float,
    kind: str = "general",
) -> Fact:
    """Store the answer to a pending question: the object becomes known (with the answer as its
    note), a fact and an "answer" episode are kept. The caller removes the question."""
    answer = " ".join(answer.split())
    world.learn(question.object, room=question.room, note=_clip(answer, MAX_NOTE_CHARS), timestamp=now)
    fact = memory.facts.remember(f"{question.object}: {answer}", question.object, kind, source="user")
    memory.episodes.record(
        "answer", f"The user explained what the {question.object} is: {answer}",
        [question.object], question.room, now,
    )
    return fact


def store_user_fact(
    world: WorldModel,
    memory: Memory,
    *,
    text: str,
    subject: str | None,
    kind: str,
    now: float,
    room: str | None = None,
    confidence: float = 1.0,
    note: str | None = None,
    questions: PendingQuestions | None = None,
    known_object: bool | None = None,
) -> Fact:
    """Keep something a person told Laika: a fact, an episode and, when it is about an object, the
    world model.

    A general or ownership fact about an object (one the world model knows, one a pending question
    is about, or `known_object=True`) makes it a known object with a note; the pending question,
    if any, is answered. A location with a room moves the subject there.
    """
    fact = memory.facts.remember(text, subject, kind, source="user", confidence=confidence)
    episode_kind = "rule" if kind == "rule" else "conversation"
    summary = f"User said: {text}"
    if subject:
        question = _pending_for(questions, subject)
        existing = world.where_is(subject)
        is_object = known_object if known_object is not None else (existing is not None or question is not None)
        told_what = is_object and kind in ("general", "ownership")
        learn_note = None
        if told_what and (kind == "general" or not (existing and existing.note)):
            learn_note = _clip(note or text, MAX_NOTE_CHARS)
        if room or learn_note:
            world.learn(subject, room=room, note=learn_note, timestamp=now)
        if question is not None and told_what and questions is not None:
            questions.remove(question.id)
            episode_kind = "answer"
            summary = f"The user explained what the {question.object} is: {text}"
    memory.episodes.record(episode_kind, summary, [subject] if subject else [], room, now)
    return fact


# --- reading the user's words -------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    kind: str
    subject: str | None
    text: str
    confidence: float
    room: str | None = None
    note: str | None = None
    # The subject is the thing in view or a pending question's object: it becomes a known object.
    about_object: bool = False
    answers: PendingQuestion | None = None


class _Words:
    """The user's text as parallel lists of original and folded words, so a match on the folded
    text can be read back in the user's own spelling."""

    def __init__(self, text: str) -> None:
        spaced = re.sub(r"(['’`])(?=\w)", r"\1 ", text)
        self.original: list[str] = []
        self.folded: list[str] = []
        for word in spaced.split():
            plain = re.sub(r"[.,;:!\"()“”']", "", fold(word)).replace(" ", "_")
            if plain:
                self.original.append(word.strip(".,;:!\"()“”"))
                self.folded.append(plain)
        self.text = " ".join(self.folded)

    def __len__(self) -> int:
        return len(self.folded)

    def of(self, match: re.Match[str], group: str) -> str | None:
        """The user's own words for a group of a match on `text`, or None when it did not take part."""
        if match.group(group) is None:
            return None
        start, end = match.span(group)
        first = self.text.count(" ", 0, start)
        last = first + self.text.count(" ", start, end)
        return re.sub(r"' ", "'", " ".join(self.original[first : last + 1]))


def _strip_article(phrase: str) -> str:
    words = phrase.replace("'", "' ").split()
    while words and fold(words[0]).rstrip("'") in {"il", "lo", "la", "l", "i", "gli", "le", "un", "uno", "una"}:
        words.pop(0)
    return " ".join(words)


def _subject(raw: str | None) -> str | None:
    """A subject phrase from the user's words: without articles, demonstratives or "di solito"."""
    if not raw:
        return None
    words = raw.replace("'", "' ").split()
    while words and fold(words[0]).rstrip("'") in {
        "questo", "questa", "quest", "quello", "quella", "quel", "quell", "di", "solito",
        "normalmente", "sempre", "il", "lo", "la", "l", "i", "gli", "le", "un", "uno", "una",
    }:
        words.pop(0)
    if not words or len(words) > 4:
        return None
    return normalise(" ".join(words))


class MemorySelector:
    """Keeps what a finished turn taught Laika when the model did not store it itself."""

    def __init__(
        self,
        world: WorldModel,
        memory: Memory,
        questions: PendingQuestions | None = None,
        *,
        in_view: Callable[[], Sequence[str]] | None = None,
        config: MemoryConfig | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.world = world
        self.memory = memory
        self.questions = questions
        self.in_view = in_view
        self.config = config if config is not None else MemoryConfig()
        self._clock = clock
        self._shown: tuple[PendingQuestion, ...] = ()
        self._last_shown: tuple[PendingQuestion, ...] = ()

    def begin_turn(self) -> None:
        """Note the questions the agent is shown this turn; the previous turn's still count."""
        self._last_shown = self._shown
        pending = self.questions.pending()[:MAX_SHOWN_QUESTIONS] if self.questions is not None else []
        self._shown = tuple(pending)

    def select(self, user_text: str, tool_calls_made: Sequence[str] = ()) -> list[Fact]:
        """Store what is worth keeping from `user_text`. Returns the facts stored (usually none)."""
        if not self.config.auto_select or MEMORY_TOOLS.intersection(tool_calls_made):
            return []
        words = _Words(user_text)
        if not len(words) or user_text.rstrip().endswith("?") or self._not_for_memory(words):
            return []
        pending = self._waiting()
        candidate = self._answer(words, pending) if pending else None
        if candidate is None:
            if len(words) < MIN_WORDS:
                return []
            candidate = self._best(words, user_text)
        if candidate is None or candidate.confidence < self.config.min_confidence:
            return []
        return [self._store(candidate)]

    # --- what never becomes memory ----------------------------------------------------------

    @staticmethod
    def _not_for_memory(words: _Words) -> bool:
        first = words.folded[0]
        if first in QUESTION_WORDS:
            return True
        if first in GREETINGS and len(words) <= 4:
            return True
        return any(w.startswith(FORGET_WORDS) for w in words.folded) or "non ricordare" in words.text

    # --- answers to a question Laika asked ---------------------------------------------------

    def _waiting(self) -> list[PendingQuestion]:
        """Questions shown this turn or last that are still waiting for an answer."""
        if self.questions is None:
            return []
        seen: dict[str, PendingQuestion] = {}
        for question in (*self._shown, *self._last_shown):
            if self.questions.get(question.id) is not None:
                seen.setdefault(question.id, question)
        return list(seen.values())

    def _answer(self, words: _Words, pending: list[PendingQuestion]) -> Candidate | None:
        if words.folded[0] in DISMISSALS or "non so" in words.text or "non lo so" in words.text:
            return None
        question = next(
            (q for q in pending if f" {fold(q.object)} " in f" {words.text} "), pending[0]
        )
        user_text = " ".join(words.original)
        m = IT_IS.match(words.text)
        if m and len((m.group("subj") or "").split()) <= 4:
            return Candidate("general", question.object, words.of(m, "what") or user_text,
                             ANSWER_CONFIDENCE, about_object=True, answers=question)
        if m := OWNED_BY.match(words.text) or BELONGS_TO.match(words.text):
            owner = words.of(m, "owner") or ""
            if m.group("owner").split()[0] in NOT_OWNERS:
                return None
            return Candidate("ownership", question.object, f"di {owner}", ANSWER_CONFIDENCE,
                             about_object=True, answers=question)
        if m := NAMED.match(words.text):
            return Candidate("general", question.object, f"si chiama {words.of(m, 'name')}",
                             ANSWER_CONFIDENCE, about_object=True, answers=question)
        if (
            words.folded[0] in NOUN_REPLY_START
            and len(words) <= 5
            and not SENTENCE_WORDS.intersection(words.folded)
        ):
            return Candidate("general", question.object, user_text, NOUN_ANSWER_CONFIDENCE,
                             about_object=True, answers=question)
        return None

    # --- the patterns --------------------------------------------------------------------------

    def _in_view(self) -> str | None:
        """The one object in view, when there is exactly one: what "this" and "it" mean."""
        if self.in_view is None:
            return None
        names = list(dict.fromkeys(normalise(n) for n in self.in_view() if n and n.strip()))
        return names[0] if len(names) == 1 else None

    def _best(self, words: _Words, user_text: str) -> Candidate | None:
        text = " ".join(user_text.split())
        context = self._in_view()
        for rule in (self._rule, self._preference, self._named, self._ownership, self._location, self._identity):
            candidate = rule(words, text, context)
            if candidate is not None:
                return candidate
        return None

    def _resolve(self, words: _Words, m: re.Match[str], context: str | None) -> tuple[str | None, bool]:
        """The subject of a statement: named in it, or the thing in view. (subject, is that thing)."""
        named = _subject(words.of(m, "subj")) if "subj" in m.groupdict() else None
        if named is not None:
            return named, named == context
        return context, context is not None

    @staticmethod
    def _rule(words: _Words, text: str, _context: str | None) -> Candidate | None:
        if not (m := RULE.match(words.text) or RULE_OUTSIDE.match(words.text)):
            return None
        place = _strip_article(words.of(m, "place") or "")
        if not place or len(place.split()) > 4:
            return None
        return Candidate("rule", normalise(place), text, RULE_CONFIDENCE)

    @staticmethod
    def _preference(words: _Words, text: str, _context: str | None) -> Candidate | None:
        if not PREFERENCE.match(words.text):
            return None
        return Candidate("preference", None, text, PREFERENCE_CONFIDENCE)

    def _named(self, words: _Words, text: str, context: str | None) -> Candidate | None:
        if not (m := NAMED.match(words.text)):
            return None
        subject, about = self._resolve(words, m, context)
        if subject is None:
            return None
        return Candidate("general", subject, text, NAME_CONFIDENCE,
                         note=f"si chiama {words.of(m, 'name')}", about_object=about)

    def _ownership(self, words: _Words, text: str, context: str | None) -> Candidate | None:
        m = OWNED_BY.match(words.text) or BELONGS_TO.match(words.text)
        if m is None or m.group("owner").split()[0] in NOT_OWNERS:
            return None
        subject, about = self._resolve(words, m, context)
        if subject is None:
            return None
        return Candidate("ownership", subject, text, OWNERSHIP_CONFIDENCE, about_object=about)

    def _location(self, words: _Words, text: str, context: str | None) -> Candidate | None:
        m = LOCATION.match(words.text)
        if m is None:
            return None
        strong = bool(m.group("usually") or m.group("usually2")) or bool(
            m.group("subj") and m.group("subj").split()[0] in ("di", "normalmente", "sempre")
        )
        if m.group("verb") in ("e", "sono") and not strong:
            return None  # "è in cucina" is where something is now, not where it stays
        subject, about = self._resolve(words, m, context)
        if subject is None:
            return None
        place = _strip_article(words.of(m, "place") or "")
        room = ROOMS.get(fold(place)) if m.group("prep") in ("in", "nel", "nella", "nello", "nell") else None
        return Candidate("location", subject, text,
                         STRONG_LOCATION_CONFIDENCE if strong else LOCATION_CONFIDENCE,
                         room=room, about_object=about)

    def _identity(self, words: _Words, text: str, context: str | None) -> Candidate | None:
        m = IT_IS.match(words.text)
        if m is None or context is None:
            return None
        subj = m.group("subj")
        if subj is not None and not DEMONSTRATIVE.match(subj):
            return None  # "il cane è un bravo ragazzo" is about something else
        # Only "this/that is a X" says what the thing in view is; a bare "è un X" could be anything.
        confidence = IDENTITY_CONFIDENCE if subj else GUESS_CONFIDENCE
        return Candidate("general", context, words.of(m, "what") or text, confidence, about_object=True)

    # --- storing ---------------------------------------------------------------------------------

    def _store(self, candidate: Candidate) -> Fact:
        now = self._clock()
        subject, text = candidate.subject, candidate.text
        if candidate.answers is not None:
            fact = record_answer(self.world, self.memory, candidate.answers, text, now, candidate.kind)
            if self.questions is not None:
                self.questions.remove(candidate.answers.id)
        else:
            if candidate.kind == "general" and candidate.about_object and subject and candidate.note is None:
                note, text = text, f"{subject}: {text}"  # "this is a X": the fact reads like an answer
            else:
                note = candidate.note
            fact = store_user_fact(
                self.world, self.memory, text=text, subject=subject, kind=candidate.kind, now=now,
                room=candidate.room, confidence=candidate.confidence, note=note,
                questions=self.questions, known_object=True if candidate.about_object else None,
            )
        print(f"[laika] remembered: {fact.kind} {fact.subject or '-'}: {fact.text}")
        return fact
