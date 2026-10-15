"""Listening and speaking: SunFounder's Vosk and Piper on the robot, the console elsewhere."""

from __future__ import annotations

import threading
from typing import Protocol

from laika.wake_word import WakeMatch, WakeWordMatcher


class Ear(Protocol):
    def listen(self, timeout_s: float | None = None) -> str | None:
        """Return the next thing heard, or None after `timeout_s` of silence."""

    def close(self) -> None: ...


class Mouth(Protocol):
    def say(self, text: str) -> None: ...


def wait_for_wake(ear: Ear, matcher: WakeWordMatcher) -> WakeMatch:
    while True:
        heard = ear.listen()
        if heard:
            print(f"[laika] heard: {heard}")
        match = matcher.match(heard)
        if match:
            return match


class VoskEar:
    """Offline speech recognition through `picarx.stt.Vosk`."""

    def __init__(self, language: str) -> None:
        from picarx.stt import Vosk

        # The first run downloads the model for `language` (about 50 MB for Italian).
        self._stt = Vosk(language=language)

    def listen(self, timeout_s: float | None = None) -> str | None:
        timer = None
        if timeout_s:
            # Vosk only returns once it hears speech; stopping it ends the wait.
            timer = threading.Timer(timeout_s, self._stt.stop_listening)
            timer.daemon = True
            timer.start()
        try:
            text = self._stt.listen(stream=False)
        finally:
            if timer:
                timer.cancel()
        return text.strip() if text else None

    def close(self) -> None:
        self._stt.close()


class PiperMouth:
    """Offline neural voice through `picarx.tts.Piper`."""

    def __init__(self, model: str) -> None:
        from picarx.tts import Piper

        # The first run downloads the voice model.
        self._tts = Piper(model=model)

    def say(self, text: str) -> None:
        print(f"Laika: {text}")
        self._tts.say(text)


class ConsoleEar:
    """Typed input, for the simulator or a robot without a working microphone."""

    def listen(self, timeout_s: float | None = None) -> str | None:
        prompt = "Tu (invio vuoto = fine): " if timeout_s else "Tu (scrivi 'ehi Laika ...'): "
        try:
            text = input(prompt).strip()
        except EOFError:
            raise KeyboardInterrupt from None
        return text or None

    def close(self) -> None:
        pass


class ConsoleMouth:
    def say(self, text: str) -> None:
        print(f"Laika: {text}")
