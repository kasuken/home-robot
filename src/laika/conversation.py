"""The prototype loop: sleep until "ehi Laika", then chat until it goes quiet."""

from __future__ import annotations

import threading
import time
from contextlib import nullcontext
from typing import TYPE_CHECKING

from laika.actions import ActionPlayer
from laika.agent import Agent
from laika.ai_client import AIReply, AIUnavailable, ChatClient
from laika.audio import Ear, wait_for_wake
from laika.autonomy import autonomy_command
from laika.bedtime import ANOTHER
from laika.behaviour import Arbiter, Lease, Priority
from laika.commands import InvalidCommand, LookCommand
from laika.config import VoiceConfig
from laika.conversation_memory import MemorySelector
from laika.events import AIEndpointMonitor, EventBus, EventType, emit
from laika.investigate import _heartbeat as keep_alive
from laika.personality import Personality
from laika.prompts import proactive_instruction
from laika.proactive import ProactiveRequest
from laika.robot import Robot
from laika.safety import SafetyController
from laika.storage import TranscriptLog
from laika.tools import PROACTIVE_TOOLS
from laika.wake_word import WakeWordMatcher, normalise

if TYPE_CHECKING:
    from laika.investigate import PendingQuestions
    from laika.bedtime import BedtimeStories
    from laika.games import GameSession

# A question handed over but not started within this long is stale: the moment has passed.
PROACTIVE_MAX_AGE_S = 30.0
NO_CONNECTION = "Non riesco a raggiungere il mio cervello. Riprova tra poco."
PAUSED_SAY = "Va bene, resto ferma. Quando vuoi, dimmi che posso muovermi di nuovo."
RESUMED_SAY = "Va bene, posso muovermi di nuovo."


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
        personality: Personality | None = None,
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
        # Humour and sarcasm reach the question she starts herself (#41).
        self.personality = personality
        self._turn_tools: list[str] = []
        self._turn_cancelled = False
        # With an arbiter, a chat holds a CONVERSATION lease from the wake word to the end.
        self.arbiter = arbiter
        self._lease: Lease | None = None
        # The pending questions, to tell whether a proactive one was answered (#40).
        self.questions: PendingQuestions | None = None
        # The game being played (games.py): set by the app with the agent.
        self.games: GameSession | None = None
        # Bedtime stories (bedtime.py): after one she waits for "another one?". Set by the app with the agent.
        self.stories: BedtimeStories | None = None
        # The hand-off from the proactive thread: only the main thread ever speaks (#40).
        self._proactive_lock = threading.Lock()
        self._proactive: ProactiveRequest | None = None
        self._awake = False
        self._waiting = False
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
            self._waiting = True
            try:
                woke = wait_for_wake(self.ear, self.matcher, self._take_proactive)
            finally:
                self._waiting = False
            if isinstance(woke, ProactiveRequest):
                self.converse_proactive(woke)
            else:
                self.converse(woke.remainder)

    # --- proactive hand-off (called from other threads) ---------------------------

    @property
    def awake(self) -> bool:
        """True from a wake word (or a proactive start) until the conversation ends."""
        return self._awake

    def request_proactive(self, request: ProactiveRequest) -> bool:
        """Ask the main thread to start talking. False when she is already in a conversation.

        Thread-safe and quick: it stores the request and wakes the sleeping ear; it never speaks.
        """
        with self._proactive_lock:
            if self._awake or self._proactive is not None:
                return False
            self._proactive = request
            wake_ear = self._waiting
        if wake_ear:
            self.ear.interrupt()
        return True

    def _take_proactive(self) -> ProactiveRequest | None:
        with self._proactive_lock:
            request, self._proactive = self._proactive, None
            if request is None:
                return None
            if time.monotonic() - request.created > PROACTIVE_MAX_AGE_S:
                print("[laika] dropped a stale proactive question")
                if request.lease is not None:
                    request.lease.release()  # an encounter's body goes back to what it interrupted
                request.taken.set()
                request.finished.set()
                return None
            self._awake = True
            return request

    def _wake_up(self) -> None:
        """A user wake word wins over a question that was waiting."""
        with self._proactive_lock:
            self._awake = True
            self._proactive = None

    def converse(self, remainder: str | None) -> None:
        """One awake period: from the wake word until the chat ends (sleep word or silence)."""
        self._wake_up()
        if self.agent is not None:
            self.agent.cancel()  # a new wake drops anything left from before
        self._open_lease()
        try:
            self.actions.play("look_up")
            question = remainder
            if not question:
                self.speak(self.voice.wake_greeting)
                question = self.ear.listen(timeout_s=self._follow_up_s())
            emit(self.events, EventType.INTERACTION_STARTED, initiator="user")
            try:
                self.chat(question)
            finally:
                emit(self.events, EventType.INTERACTION_COMPLETED, initiator="user")
            self.actions.play("look_forward")
            self.actions.wait_done()
        finally:
            self._close_lease()
            self._awake = False

    def converse_proactive(self, request: ProactiveRequest) -> bool:
        """Laika speaks first: turn to the person, ask the pending question, then chat like normal.

        Runs on the main thread. If the body is held by something more important she gives up quietly.
        Nobody answering is not an error: the question stays pending and she does not insist.
        Returns True when the question got an answer.
        """
        social: Lease | None = None
        started = answered = False
        try:
            if self.agent is not None:
                self.agent.cancel()
            if self.arbiter is not None:
                held = request.lease
                if held is not None and held.active:
                    # An encounter (#49) already holds the body at this priority: keep it, so whatever she
                    # interrupted does not get the body back for a moment.
                    social = held
                    social.on_preempt = self._preempted
                    social.kick()
                else:
                    social = self.arbiter.request(
                        "proactive", Priority.SOCIAL, on_preempt=self._preempted, resumable=False
                    )
                request.taken.set()
                if social is None:
                    print("[laika] proactive question dropped: a more important behaviour holds the body")
                    return False
            else:
                request.taken.set()
            self._lease = social
            if social is not None and self.arbiter is not None:
                beat = keep_alive(social, self.arbiter.lease_timeout_s)
            else:
                beat = nullcontext()
            with beat:
                self._face(request, social)
                emit(self.events, EventType.INTERACTION_STARTED, **self._interaction_data(request))
                started = True
                request.spoke.set()
                heard = self._ask_proactively(request, social)
            if heard:
                # The person replied: from here it is a conversation, which pre-empts the social lease.
                self._open_lease()
                if social is not None:
                    social.release()
                self.chat(heard)
                answered = (
                    request.kind == "question"
                    and self.questions is not None
                    and self.questions.get(request.question_id) is None
                )
            else:
                self._forget_chat()  # nobody answered: say nothing more, the question stays pending
            return answered
        finally:
            if started:
                done = self._interaction_data(request)
                done.pop("reason", None)
                emit(self.events, EventType.INTERACTION_COMPLETED, **done, answered=answered)
            try:
                self.actions.play("look_forward")
                self.actions.wait_done()
            finally:
                if social is not None:
                    social.release()
                self._close_lease()
                self._awake = False
                request.taken.set()
                request.finished.set()

    @staticmethod
    def _interaction_data(request: ProactiveRequest) -> dict[str, object]:
        """The event data of a conversation Laika starts: a question carries its id, an encounter says so."""
        data: dict[str, object] = {"initiator": "laika", "reason": request.reason}
        if request.kind == "question":
            data["questionId"] = request.question_id
        if request.encounter:
            data["encounter"] = True
        return data

    def _face(self, request: ProactiveRequest, lease: Lease | None) -> None:
        """Turn the head to where the person was seen."""
        if self._lease_gone(lease):
            return
        try:
            command = LookCommand(pan=request.pan, tilt=request.tilt)
        except InvalidCommand:
            return
        self.robot.look(command.pan, command.tilt)

    def _ask_proactively(self, request: ProactiveRequest, lease: Lease | None) -> str | None:
        """Say the opening line and listen for the answer; None when she stayed quiet or nobody replied."""
        opening = self._opening(request, lease)
        if opening is None or self._lease_gone(lease):
            return None
        self.speak(opening)
        self.actions.wait_done()
        if self.transcript is not None:
            self.transcript.record("assistant", opening)
        return self.ear.listen(timeout_s=self._follow_up_s())

    @staticmethod
    def _lease_gone(lease: Lease | None) -> bool:
        """Taken away by someone more important (or a stop). An AI outage does not stop her asking
        with the plain question: speaking needs no body."""
        return lease is not None and not lease.active and lease.end_reason != "ai_unavailable"

    def _opening(self, request: ProactiveRequest, lease: Lease | None) -> str | None:
        """The model words the question in her own style, with the head and gestures only.

        Without an agent, or when the AI fails, the question's own (Italian) text is used.
        """
        if self.agent is None:
            return request.text
        try:
            result = self.agent.respond(
                proactive_instruction(request.detail or request.text, self.personality, request.kind),
                lease=lease,
                registry=self.agent.registry.restricted(PROACTIVE_TOOLS),
            )
            self.endpoint.succeeded()
        except AIUnavailable as error:
            self.endpoint.failed(error)
            print(f"[laika] AI unavailable, asking with the plain question: {error}")
            self.actions.cancel_pending()
            self.agent.cancel()
            return request.text
        if result.cancelled:
            return None
        self.actions.wait_done(timeout=15)
        return result.text or request.text

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
            question = self.ear.listen(timeout_s=self._follow_up_s())
        self._forget_chat()

    def _follow_up_s(self) -> float:
        """How long to wait for an answer: longer while a game is on (children think), exactly the
        story's wait after "another story?"."""
        default = self.voice.follow_up_s
        if self.games is not None:
            default = self.games.follow_up_s(default)
        return self.stories.follow_up_s(default) if self.stories is not None else default

    def _forget_chat(self) -> None:
        if self.games is not None:
            self.games.end()  # a game does not outlive the conversation
        if self.stories is not None:
            self.stories.end()
        self.ai.reset()
        if self.agent is not None:
            self.agent.reset()

    def reply(self, question: str) -> AIReply:
        print(f"[laika] question: {question}")
        if (local := self._autonomy_reply(question)) is not None:
            return local
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
        if self.stories is not None and self.stories.take_quiet_reply():
            # A story ended with her question "another one?": that stays the last thing she said, and the
            # model's history remembers it as her answer, so a bare "sì" next still makes sense.
            reply = AIReply(say="", actions=reply.actions)
            if self.agent is not None:
                self.agent.replace_last_answer(question, ANOTHER)
        if reply.say:
            self.speak(reply.say)
        self.actions.wait_done()
        if answered:
            self._select_memory(question)
        if self.transcript is not None:
            self.transcript.record("user", question)
            self.transcript.record("assistant", reply.say)
        return reply

    def _autonomy_reply(self, question: str) -> AIReply | None:
        """"Laika, resta ferma" and "puoi muoverti di nuovo" (#51) are handled here, not by the model: they
        must work in JSON mode and with the model down. Only the user's own words get here."""
        command = autonomy_command(question) if self.safety is not None else None
        if command is None or self.safety is None:
            return None
        if command == "pause":
            self.safety.pause_autonomy()
            say = PAUSED_SAY
        else:
            self.safety.resume_autonomy()
            say = RESUMED_SAY
        print(f"[laika] autonomy {'paused' if command == 'pause' else 'resumed'} by the user")
        self.speak(say)
        self.actions.wait_done()
        if self.transcript is not None:
            self.transcript.record("user", question)
            self.transcript.record("assistant", say)
        return AIReply(say=say)

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
