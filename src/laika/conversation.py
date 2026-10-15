"""The prototype loop: sleep until "ehi Laika", then chat until it goes quiet."""

from __future__ import annotations

from laika.actions import ActionPlayer
from laika.agent import Agent
from laika.ai_client import AIReply, AIUnavailable, ChatClient
from laika.audio import Ear, wait_for_wake
from laika.behaviour import Arbiter, Lease, Priority
from laika.config import VoiceConfig
from laika.conversation_memory import MemorySelector
from laika.events import AIEndpointMonitor, EventBus, EventType, emit
from laika.robot import Robot
from laika.safety import SafetyController
from laika.storage import TranscriptLog
from laika.wake_word import WakeWordMatcher, normalise

NO_CONNECTION = "Non riesco a raggiungere il mio cervello. Riprova tra poco."


class Conversation:
    def __init__(
        self,
        *,
        ear: Ear,
        robot: Robot,
        ai: ChatClient,
        actions: ActionPlayer,
        voice: VoiceConfig,
        events: EventBus | None = None,
        safety: SafetyController | None = None,
        agent: Agent | None = None,
        transcript: TranscriptLog | None = None,
        arbiter: Arbiter | None = None,
        selector: MemorySelector | None = None,
    ) -> None:
        self.ear = ear
        self.robot = robot
        self.ai = ai
        self.actions = actions
        self.voice = voice
        self.events = events
        self.safety = safety
        # With an agent the model acts through tools; without one it replies with JSON.
        self.agent = agent
        self.transcript = transcript
        # Keeps what a turn taught her when the model did not (#35); works with and without an agent.
        self.selector = selector
        self._turn_tools: list[str] = []
        self._turn_cancelled = False
        # With an arbiter, a chat holds a CONVERSATION lease from the wake word to the end.
        self.arbiter = arbiter
        self._lease: Lease | None = None
        self.endpoint = AIEndpointMonitor(
            events, on_unavailable=self._ai_unavailable, on_recovered=self._ai_recovered
        )
        self.matcher = WakeWordMatcher(voice.wake_names)
        self.sleep_words = {normalise(w) for w in voice.sleep_words}

    def _ai_unavailable(self) -> None:
        """Stop moving and forget queued gestures; nothing is replayed on recovery."""
        if self.agent is not None:
            self.agent.cancel()
        if self.safety is not None:
            self.safety.set_ai_available(False)
        self.actions.cancel_pending()

    def _ai_recovered(self) -> None:
        if self.safety is not None:
            self.safety.set_ai_available(True)

    def run(self) -> None:
        self.speak(self.voice.welcome)
        while True:
            print("[laika] sleeping, say 'ehi Laika'")
            wake = wait_for_wake(self.ear, self.matcher)
            self.converse(wake.remainder)

    def converse(self, remainder: str | None) -> None:
        """One awake period: from the wake word until the chat ends (sleep word or silence)."""
        if self.agent is not None:
            self.agent.cancel()  # a new wake drops anything left from before
        self._open_lease()
        try:
            self.actions.play("look_up")
            question = remainder
            if not question:
                self.speak(self.voice.wake_greeting)
                question = self.ear.listen(timeout_s=self.voice.follow_up_s)
            emit(self.events, EventType.INTERACTION_STARTED)
            try:
                self.chat(question)
            finally:
                emit(self.events, EventType.INTERACTION_COMPLETED)
            self.actions.play("look_forward")
            self.actions.wait_done()
        finally:
            self._close_lease()

    def _open_lease(self) -> None:
        if self.arbiter is None:
            return
        # Resumable: if a direct user command pre-empts the chat, it continues afterwards.
        self._lease = self.arbiter.request(
            "conversation",
            Priority.CONVERSATION,
            on_preempt=self._preempted,
            resumable=True,
        )
        if self._lease is None:
            # Something more important owns the body: talk, but every move is refused.
            print("[laika] conversation cannot move: a more important behaviour holds the body")
            self._lease = Lease.denied("conversation", Priority.CONVERSATION)

    def _close_lease(self) -> None:
        lease, self._lease = self._lease, None
        if lease is not None and self.arbiter is not None:
            self.arbiter.release(lease)

    def _preempted(self, _lease: Lease) -> None:
        """The turn in progress must not act any more."""
        if self.agent is not None:
            self.agent.cancel()

    def speak(self, text: str) -> None:
        """Say `text` while the head moves a little, like someone talking."""
        self.actions.set_talking(True)
        try:
            self.robot.speak(text)
        finally:
            self.actions.set_talking(False)

    def chat(self, question: str | None) -> None:
        """Answer, then keep listening for follow-ups until silence or a sleep word."""
        while question:
            if self._is_sleep_word(question):
                self.speak(self.voice.goodbye)
                break
            self.reply(question)
            question = self.ear.listen(timeout_s=self.voice.follow_up_s)
        self.ai.reset()
        if self.agent is not None:
            self.agent.reset()

    def reply(self, question: str) -> AIReply:
        print(f"[laika] question: {question}")
        self.actions.play("think")
        answered = False
        self._turn_tools, self._turn_cancelled = [], False
        if self.selector is not None:
            self.selector.begin_turn()
        try:
            reply = self._ask(question)
            self.endpoint.succeeded()
            answered = not self._turn_cancelled
        except AIUnavailable as error:
            self.endpoint.failed(error)
            print(f"[laika] AI unavailable: {error}")
            # Before "sad": drop what is queued (also on repeated failures).
            self.actions.cancel_pending()
            if self.agent is not None:
                self.agent.cancel()
            reply = AIReply(say=NO_CONNECTION, actions=["sad"])

        # Gestures the agent queued are still waiting: let them play instead of dropping them.
        self.actions.wait_done(timeout=15 if self.agent is not None else 2)
        self.actions.play("look_forward", *reply.actions)
        if reply.actions:
            print(f"[laika] actions: {', '.join(reply.actions)}")
        if reply.say:
            self.speak(reply.say)
        self.actions.wait_done()
        if answered:
            self._select_memory(question)
        if self.transcript is not None:
            self.transcript.record("user", question)
            self.transcript.record("assistant", reply.say)
        return reply

    def _select_memory(self, question: str) -> None:
        """After a completed turn: keep what is worth keeping. Never breaks the conversation."""
        if self.selector is None:
            return
        try:
            self.selector.select(question, self._turn_tools)
        except Exception as error:  # noqa: BLE001 - memory is a bonus, the chat goes on
            print(f"[laika] could not select memories: {error}")

    def _ask(self, question: str) -> AIReply:
        if self.agent is None:
            return self.ai.ask(question)
        result = self.agent.respond(question, lease=self._lease)
        self._turn_tools, self._turn_cancelled = list(result.tool_calls_made), result.cancelled
        if result.tool_calls_made:
            print(f"[laika] tools: {', '.join(result.tool_calls_made)}")
        return AIReply(say=result.text)

    def _is_sleep_word(self, text: str) -> bool:
        return normalise(text) in self.sleep_words
