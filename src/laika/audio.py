"""Listening and speaking: SunFounder's Vosk and Edge or Piper voices on the robot, the console elsewhere."""

from __future__ import annotations

import threading
from typing import Any, Callable, Protocol

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


TTS_ENGINES = ("edge", "piper")


def piper_engine(model: str) -> Any:
    """Offline neural voice. The first use downloads the voice model."""
    from picarx.tts import Piper

    return Piper(model=model)


def edge_engine(voice: str) -> Any:
    """Microsoft Edge online voice. Needs internet, `edge-tts`, and sox with MP3 support."""
    from picarx.tts import EdgeTTS

    return EdgeTTS(voice=voice)


class SpeakerMouth:
    """Speaks through a SunFounder TTS engine, switching to a backup engine when it fails."""

    def __init__(self, engine: Any, backup: Callable[[], Any] | None = None) -> None:
        self._engine = engine
        self._make_backup = backup
        self._backup: Any = None

    def say(self, text: str) -> None:
        print(f"Laika: {text}")
        try:
            self._engine.say(text)
            return
        except Exception as error:  # Network, sox or engine errors must not end the conversation.
            print(f"[laika] voice failed: {error}")
        if self._make_backup is None:
            return
        try:
            if self._backup is None:
                print("[laika] switching to the offline voice for this sentence")
                self._backup = self._make_backup()
            self._backup.say(text)
        except Exception as error:
            print(f"[laika] offline voice failed too: {error}")


def make_mouth(engine: str, *, edge_voice: str, piper_model: str) -> SpeakerMouth:
    if engine not in TTS_ENGINES:
        raise ValueError(f"Unknown tts_engine {engine!r}, use one of {', '.join(TTS_ENGINES)}")
    if engine == "piper":
        return SpeakerMouth(piper_engine(piper_model))
    try:
        voice = edge_engine(edge_voice)
    except ImportError as error:
        print(
            f"[laika] Edge voice unavailable ({error}); using Piper. To enable it:\n"
            "  sudo pip3 install edge-tts --break-system-packages\n"
            "  sudo apt install sox libsox-fmt-mp3"
        )
        return SpeakerMouth(piper_engine(piper_model))
    return SpeakerMouth(voice, backup=lambda: piper_engine(piper_model))


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
