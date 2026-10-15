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
from laika.prompts import memory_prompt, pending_questions_prompt
from laika.tools import Turn, ToolRegistry
from laika.world_model import WorldModel

if TYPE_CHECKING:
    from laika.behaviour import Lease
    from laika.investigate import PendingQuestions
    from laika.memory import Memory

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

    def reset(self) -> None:
        """End of a conversation: cancel what is running and forget the history."""
        self.cancel()
        self.history.clear()

    def _current_generation(self) -> int:
        return self._generation

    def respond(self, user_text: str, lease: Lease | None = None) -> AgentReply:
        """Answer `user_text`, acting through tools. Raises `AIUnavailable` if the model is down.

        With a `lease` the tools act under it: once it is pre-empted the turn goes stale.
        """
        turn = Turn(self._generation, self._current_generation, lease)
        messages = self._start_messages(user_text)
        specs = self.registry.specs()
        made: list[str] = []
        text = ""
        for round_number in range(self.max_rounds):
            last_round = round_number == self.max_rounds - 1
            try:
                answer = self.client.complete(
                    messages, tools=specs, tool_choice="none" if last_round else "auto"
                )
            except AIUnavailable:
                self.cancel()  # nothing this turn proposed may run later
                raise
            if turn.cancelled:
                return AgentReply("", made, cancelled=True)
            text = answer.content.strip()
            if not answer.tool_calls:
                return self._finish(user_text, text, made)
            if last_round:
                print("[laika] tool calls refused: no rounds left this turn")
                break
            messages.append(assistant_message(answer))
            for call in answer.tool_calls:
                if len(made) >= self.max_tool_calls:
                    result = self.registry.reject(call.name, "too many tool calls in this turn")
                else:
                    made.append(call.name)
                    result = self.registry.run(call, turn)
                messages.append(tool_message(call.id, json.dumps(result, ensure_ascii=False)))
            if turn.cancelled:
                return AgentReply("", made, cancelled=True)
        print("[laika] agent stopped: round limit reached")
        return self._finish(user_text, f"{text} {BOUNDS_FALLBACK}".strip(), made)

    def _start_messages(self, user_text: str) -> list[dict[str, Any]]:
        prompt = self.system_prompt
        if self.world is not None:
            prompt += f"\n\nCosa sai della casa (in inglese): {self.world.summary()}"
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

    def _finish(self, user_text: str, text: str, made: list[str]) -> AgentReply:
        if text:
            self.history.append(user_message(user_text))
            self.history.append({"role": "assistant", "content": text})
        return AgentReply(text, made)
