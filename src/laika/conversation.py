"""The prototype loop: sleep until "ehi Laika", then chat until it goes quiet."""

from __future__ import annotations

from laika.actions import ActionPlayer
from laika.ai_client import AIReply, AIUnavailable, ChatClient
from laika.audio import Ear, Mouth, wait_for_wake
from laika.config import VoiceConfig
from laika.wake_word import WakeWordMatcher, normalise

NO_CONNECTION = "Non riesco a raggiungere il mio cervello. Riprova tra poco."


class Conversation:
    def __init__(
        self,
        *,
        ear: Ear,
        mouth: Mouth,
        ai: ChatClient,
        actions: ActionPlayer,
        voice: VoiceConfig,
    ) -> None:
        self.ear = ear
        self.mouth = mouth
        self.ai = ai
        self.actions = actions
        self.voice = voice
        self.matcher = WakeWordMatcher(voice.wake_names)
        self.sleep_words = {normalise(w) for w in voice.sleep_words}

    def run(self) -> None:
        self.mouth.say(self.voice.welcome)
        while True:
            print("[laika] sleeping, say 'ehi Laika'")
            wake = wait_for_wake(self.ear, self.matcher)
            self.actions.play("look_up")
            question = wake.remainder
            if not question:
                self.mouth.say(self.voice.wake_greeting)
                question = self.ear.listen(timeout_s=self.voice.follow_up_s)
            self.chat(question)
            self.actions.play("look_forward")
            self.actions.wait_done()

    def chat(self, question: str | None) -> None:
        """Answer, then keep listening for follow-ups until silence or a sleep word."""
        while question:
            if self._is_sleep_word(question):
                self.mouth.say(self.voice.goodbye)
                return
            self.reply(question)
            question = self.ear.listen(timeout_s=self.voice.follow_up_s)
        self.ai.reset()

    def reply(self, question: str) -> AIReply:
        print(f"[laika] question: {question}")
        self.actions.play("think")
        try:
            reply = self.ai.ask(question)
        except AIUnavailable as error:
            print(f"[laika] AI unavailable: {error}")
            reply = AIReply(say=NO_CONNECTION, actions=["sad"])

        self.actions.wait_done(timeout=2)
        self.actions.play("look_forward", *reply.actions)
        if reply.actions:
            print(f"[laika] actions: {', '.join(reply.actions)}")
        if reply.say:
            self.mouth.say(reply.say)
        self.actions.wait_done()
        return reply

    def _is_sleep_word(self, text: str) -> bool:
        return normalise(text) in self.sleep_words
