"""Listening and speaking: SunFounder's Vosk and Edge or Piper voices on the robot, the console elsewhere."""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, Protocol

from laika.wake_word import WakeMatch, WakeWordMatcher


class Ear(Protocol):
    def listen(self, timeout_s: float | None = None) -> str | None:
        """Return the next thing heard, or None after `timeout_s` of silence or after `interrupt()`."""

    def interrupt(self) -> None:
        """Make the `listen` in progress (or the next one) return None now. Safe from any thread."""

    def close(self) -> None: ...


class Mouth(Protocol):
    def say(self, text: str) -> None: ...


def wait_for_wake(
    ear: Ear, matcher: WakeWordMatcher, interrupted: Callable[[], Any | None] | None = None
) -> Any:
    """Listen until the wake word. Returns the `WakeMatch`.

    With `interrupted`, it is asked before each listen and after each wake-up: whatever it returns
    (not None) ends the wait and is returned instead, so another thread can `ear.interrupt()` the
    sleeping robot to hand it something to do (a proactive question, #40). A wake word heard
    meanwhile still comes first: the user is never kept waiting behind her own initiative.
    """
    while True:
        if interrupted is not None and (signal := interrupted()) is not None:
            return signal
        heard = ear.listen()
        if heard:
            print(f"[laika] heard: {heard}")
        match = matcher.match(heard)
        if match:
            return match
        if interrupted is not None and (signal := interrupted()) is not None:
            return signal


INTERRUPT_POLL_S = 0.1
# The longest single sentence Laika waits for once someone has started talking.
MAX_UTTERANCE_S = 30.0


class VoskEar:
    """Offline speech recognition through `picarx.stt.Vosk`."""

    def __init__(self, language: str) -> None:
        from picarx.stt import Vosk

        # The first run downloads the model for `language` (about 50 MB for Italian).
        self._stt = Vosk(language=language)
        self._interrupted = threading.Event()

    def interrupt(self) -> None:
        self._interrupted.set()
        self._stt.stop_listening()

    def listen(self, timeout_s: float | None = None) -> str | None:
        if self._interrupted.is_set():
            self._interrupted.clear()
            return None
        # `timeout_s` is the silence allowed before someone starts talking. Vosk only returns once it
        # hears speech, so a timer stops it; the first partial result cancels that timer, so a person
        # who starts answering is never cut off mid-sentence. One sentence is capped at
        # MAX_UTTERANCE_S, so background noise can't keep her listening forever.
        timers: list[threading.Timer] = []
        if timeout_s:
            timers.append(self._timer(timeout_s))
        # SunFounder's listen() clears its stop flag when it starts, so an interrupt that lands
        # just before would be lost. While an interrupt is pending, keep stopping it until it returns.
        done = threading.Event()
        watcher = threading.Thread(target=self._stop_while_interrupted, args=(done,), daemon=True)
        watcher.start()
        text = None
        speaking = False
        try:
            for result in self._stt.listen(stream=True) or ():
                if not isinstance(result, dict):
                    continue
                if result.get("done"):
                    text = result.get("final")
                    break
                if result.get("partial") and not speaking:
                    speaking = True
                    for timer in timers:
                        timer.cancel()
                    timers = [self._timer(MAX_UTTERANCE_S)]
        finally:
            done.set()
            for timer in timers:
                timer.cancel()
            self._interrupted.clear()
        return text.strip() if text else None

    def _timer(self, seconds: float) -> threading.Timer:
        timer = threading.Timer(seconds, self._stt.stop_listening)
        timer.daemon = True
        timer.start()
        return timer

    def _stop_while_interrupted(self, done: threading.Event) -> None:
        while not done.is_set():
            if self._interrupted.wait(INTERRUPT_POLL_S) and not done.is_set():
                self._stt.stop_listening()
                done.wait(INTERRUPT_POLL_S)

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
    """Typed input, for the simulator or a robot without a working microphone.

    A daemon thread reads stdin into a queue, so `interrupt` can wake a `listen`. End of input
    still ends the app, as before. With `wait_for_typing` (the default) a follow-up waits until
    the person presses Enter instead of timing out: typing is slower than speaking.
    """

    _EOF = object()
    _INTERRUPT = object()
    POLL_S = 0.5  # a blocking wait would not let Ctrl+C through on Windows

    def __init__(self, wait_for_typing: bool = True) -> None:
        self._lines: queue.Queue[Any] = queue.Queue()
        self._reader: threading.Thread | None = None
        self.wait_for_typing = wait_for_typing

    def _read(self) -> None:
        while True:
            try:
                self._lines.put(input())
            except EOFError:
                self._lines.put(self._EOF)
                return

    def listen(self, timeout_s: float | None = None) -> str | None:
        if self._reader is None:
            self._reader = threading.Thread(target=self._read, name="console-ear", daemon=True)
            self._reader.start()
        print("Tu (invio vuoto = fine): " if timeout_s else "Tu (scrivi 'ehi Laika ...'): ", end="", flush=True)
        deadline = None if not timeout_s or self.wait_for_typing else time.monotonic() + timeout_s
        while True:
            wait = self.POLL_S if deadline is None else min(self.POLL_S, deadline - time.monotonic())
            if wait <= 0:
                return None
            try:
                item = self._lines.get(timeout=wait)
            except queue.Empty:
                continue
            if item is self._EOF:
                self._lines.put(self._EOF)  # every later listen ends too
                raise KeyboardInterrupt
            if item is self._INTERRUPT:
                return None
            return item.strip() or None

    def interrupt(self) -> None:
        self._lines.put(self._INTERRUPT)

    def close(self) -> None:
        pass


class ConsoleMouth:
    def say(self, text: str) -> None:
        print(f"Laika: {text}")
