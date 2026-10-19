"""The tool-calling loop: the model proposes, the registry validates, Laika acts.

`Agent.respond` asks the model, runs the tool calls it proposes (in order, one
by one, through `ToolRegistry.run`), feeds the results back and repeats until
the model answers in plain text. Only the user's text and the final answer are
kept as history; tool messages live inside one turn, so memory stays small.

Every turn has a generation. `cancel()` (AI outage, sleep word, shutdown, a new
wake) bumps it, and from then on any call of the older turn is rejected as
stale, including the rest of a batch while an earlier call was still running.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from laika.ai_client import (
    AIUnavailable,
    ChatClient,
    assistant_message,
    system_message,
    tool_message,
    user_message,
)
from laika.prompts import autonomy_prompt, battery_prompt, memory_prompt, outside_prompt, pending_questions_prompt
from laika.memory import iso_minutes, relative_time
from laika.tools import Turn, ToolRegistry
from laika.world_model import WorldModel

if TYPE_CHECKING:
    from laika.battery import BatteryMonitor
    from laika.behaviour import Lease
    from laika.investigate import PendingQuestions
    from laika.memory import Memory
    from laika.outside import WindowWatcher
    from laika.floorplan import FloorPlan
    from laika.places import PlaceNotes

MAX_ROUNDS = 4
MAX_TOOL_CALLS = 8
MAX_QUESTIONS_SHOWN = 3
MAX_FACTS_SHOWN = 5
BOUNDS_FALLBACK = "Mi sono un po' persa. Riprova, per favore."


@dataclass
class AgentReply:
    text: str
    tool_calls_made: list[str] = field(default_factory=list)
    # True when the turn was cancelled before it finished: there is nothing to say.
    cancelled: bool = False


class Agent:
    def __init__(
        self,
        client: ChatClient,
        registry: ToolRegistry,
        system_prompt: str,
        *,
        world: WorldModel | None = None,
        questions: PendingQuestions | None = None,
        memory: Memory | None = None,
        battery: BatteryMonitor | None = None,
        outside: WindowWatcher | None = None,
        places: PlaceNotes | None = None,
        plan: FloorPlan | None = None,
        max_history_turns: int = 8,
        max_rounds: int = MAX_ROUNDS,
        max_tool_calls: int = MAX_TOOL_CALLS,
    ) -> None:
        self.client = client
        self.registry = registry
        self.system_prompt = system_prompt
        self.world = world
        # Questions Laika wants to ask (#29): offered to the model, which asks only if the moment fits.
        self.questions = questions
        # Facts for the user's message are put in front of the model; recall finds the rest.
        self.memory = memory
        # A low battery is mentioned once per conversation (#50): the note goes in until a turn answered.
        self.battery = battery
        self._battery_told = False
        # The last look out of the window (#48): put in front of the model once, then only when there is a newer one.
        self.outside = outside
        # What explorations taught her about each room (places.py): how it looks, where it leads.
        self.places = places
        # The plan of the house (floorplan.py), told to the model in every conversation.
        self.plan = plan
        # Requests to the model in flight: the status LEDs show it as "thinking" (leds.py).
        self.thinking = 0
        self._thinking_lock = threading.Lock()
        self._outside_told = 0.0
        self._outside_told_id: str | None = None
        self.max_rounds = max(1, max_rounds)
        self.max_tool_calls = max(0, max_tool_calls)
        # One turn is the user's text plus the final answer.
        self.history: deque[dict[str, Any]] = deque(maxlen=max(1, max_history_turns) * 2)
        self._generation = 0
        self._lock = threading.Lock()

    def cancel(self) -> None:
        """Make every call of the turn in progress stale."""
        with self._lock:
            self._generation += 1

    def replace_last_answer(self, user_text: str, text: str) -> None:
        """Make `text` the remembered answer to `user_text`: what she actually said last (a bedtime story's
        question is spoken in code), so a bare "sì" on the next turn still has its question."""
        with self._lock:
            if len(self.history) >= 2 and self.history[-2] == user_message(user_text):
                self.history[-1] = {"role": "assistant", "content": text}
            else:
                self.history.append(user_message(user_text))
                self.history.append({"role": "assistant", "content": text})

    def reset(self) -> None:
        """End of a conversation: cancel what is running and forget the history."""
        self.cancel()
        self.history.clear()
        self._battery_told = False

    def _current_generation(self) -> int:
        return self._generation

    def respond(
        self, user_text: str, lease: Lease | None = None, registry: ToolRegistry | None = None
    ) -> AgentReply:
        """Answer `user_text`, acting through tools. Raises `AIUnavailable` if the model is down.

        With a `lease` the tools act under it: once it is pre-empted the turn goes stale. With a
        `registry` the turn uses that (narrower) set of tools instead of the agent's own.
        """
        registry = registry or self.registry
        turn = Turn(self._generation, self._current_generation, lease)
        note = self._battery_note()
        outside = self._outside_briefing()
        messages = self._start_messages(user_text, note, outside)
        specs = registry.specs()
        made: list[str] = []
        text = ""
        for round_number in range(self.max_rounds):
            last_round = round_number == self.max_rounds - 1
            with self._thinking_lock:
                self.thinking += 1
            try:
                answer = self.client.complete(
                    messages, tools=specs, tool_choice="none" if last_round else "auto"
                )
            except AIUnavailable:
                self.cancel()  # nothing this turn proposed may run later
                raise
            finally:
                with self._thinking_lock:
                    self.thinking -= 1
            if turn.cancelled:
                return AgentReply("", made, cancelled=True)
            text = answer.content.strip()
            if not answer.tool_calls:
                return self._finish(user_text, text, made, bool(note), outside)
            if last_round:
                print("[laika] tool calls refused: no rounds left this turn")
                break
            messages.append(assistant_message(answer))
            for call in answer.tool_calls:
                if len(made) >= self.max_tool_calls:
                    result = registry.reject(call.name, "too many tool calls in this turn")
                else:
                    made.append(call.name)
                    result = registry.run(call, turn)
                messages.append(tool_message(call.id, json.dumps(result, ensure_ascii=False)))
            if turn.cancelled:
                return AgentReply("", made, cancelled=True)
        print("[laika] agent stopped: round limit reached")
        return self._finish(user_text, f"{text} {BOUNDS_FALLBACK}".strip(), made, bool(note), outside)

    def _battery_note(self) -> str:
        """The Italian note about the battery, until a turn of this conversation carried it."""
        if self.battery is None or self._battery_told:
            return ""
        return battery_prompt(self.battery.state.value, self.battery.percent)

    def _outside_briefing(self) -> tuple[Any, list[tuple[float, str]]] | None:
        """What she saw out of the window that the user has not been offered yet, or None."""
        if self.outside is None:
            return None
        try:
            return self.outside.briefing(self._outside_told, self._outside_told_id)
        except Exception:  # noqa: BLE001 - a note is a bonus, never a reason to fail the turn
            return None

    def _outside_note(self, briefing: tuple[Any, list[tuple[float, str]]] | None) -> str:
        if briefing is None or self.outside is None:
            return ""
        observation, changes = briefing
        now = self.outside.now()
        return outside_prompt(
            iso_minutes(observation.timestamp)[11:16],
            relative_time(observation.timestamp, now),
            observation.summary,
            [f"{text} ({relative_time(when, now)})" for when, text in changes],
        )

    def _autonomy_note(self) -> str:
        """The pause and the no-go rooms (#51), so she can tell the user and does not offer to wander there."""
        safety = getattr(self.registry, "safety", None)
        if safety is None:
            return ""
        rules = getattr(safety, "house_rules", None)
        try:
            return autonomy_prompt(bool(safety.paused), sorted(rules.no_go_rooms()) if rules is not None else [])
        except Exception:  # noqa: BLE001 - a note is a bonus, never a reason to fail the turn
            return ""

    def _start_messages(
        self, user_text: str, battery_note: str = "", outside: tuple[Any, list[tuple[float, str]]] | None = None
    ) -> list[dict[str, Any]]:
        prompt = self.system_prompt
        if battery_note:
            prompt += "\n\n" + battery_note
        if outside_note := self._outside_note(outside):
            prompt += "\n\n" + outside_note
        if autonomy := self._autonomy_note():
            prompt += "\n\n" + autonomy
        if self.world is not None:
            prompt += f"\n\nCosa sai della casa (in inglese): {self.world.summary()}"
        if self.plan is not None:
            prompt += f"\n\nLa pianta della casa (in inglese): {self.plan.summary()}"
        if self.places is not None and (notes := self.places.summary()):
            prompt += f"\n\nCosa hai imparato esplorando le stanze (in inglese): {notes}"
        if self.memory is not None:
            facts = self.memory.facts.relevant(user_text, MAX_FACTS_SHOWN)
            section = memory_prompt([f.text for f in facts])
            if section:
                prompt += "\n\n" + section
        if self.questions is not None:
            waiting = [(q.id, q.text) for q in self.questions.pending()[:MAX_QUESTIONS_SHOWN]]
            if waiting:
                prompt += "\n\n" + pending_questions_prompt(waiting)
        return [system_message(prompt), *self.history, user_message(user_text)]

    def _finish(
        self,
        user_text: str,
        text: str,
        made: list[str],
        battery_noted: bool = False,
        outside: tuple[Any, list[tuple[float, str]]] | None = None,
    ) -> AgentReply:
        if text and battery_noted:
            self._battery_told = True
        if text and outside is not None:
            self._outside_told = outside[0].timestamp
            self._outside_told_id = outside[0].id
        if text:
            self.history.append(user_message(user_text))
            self.history.append({"role": "assistant", "content": text})
        return AgentReply(text, made)
