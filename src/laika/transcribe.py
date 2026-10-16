"""Cloud speech to text: OpenAI's transcription endpoint (`/audio/transcriptions`), Italian.

Vosk's small Italian model is fast and offline but often mishears whole sentences. `CloudEar` (audio.py)
keeps Vosk for the wake word and sends what is said once she is awake to this endpoint, which
understands Italian (and children) far better. Only utterances that start with the wake word, or that
come during a conversation, leave the robot; the wait for the wake word stays offline.

`Transcriber.transcribe` never raises: on any error it returns None and the ear uses Vosk's text instead.
"""

from __future__ import annotations

import io
import logging
import wave
from typing import Any

log = logging.getLogger(__name__)

TIMEOUT_S = 15.0
# Words the model should expect (it spells names better with a hint).
PROMPT = "Ehi Laika! Conversazione in italiano con Laika, una robottina, e un bambino."


def wav_bytes(pcm: bytes, samplerate: int, channels: int = 1) -> bytes:
    """16-bit PCM wrapped in a WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(samplerate)
        out.writeframes(pcm)
    return buffer.getvalue()


class Transcriber:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        language: str = "it",
        timeout_s: float = TIMEOUT_S,
        session: Any = None,
    ) -> None:
        self.url = base_url.rstrip("/") + "/audio/transcriptions"
        self.api_key = api_key
        self.model = model
        self.language = language
        self.timeout_s = timeout_s
        self._session = session
        self.failures = 0

    def transcribe(self, pcm: bytes, samplerate: int) -> str | None:
        if not pcm:
            return None
        try:
            session = self._session
            if session is None:
                import requests

                session = self._session = requests.Session()
            response = session.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                files={"file": ("speech.wav", wav_bytes(pcm, samplerate), "audio/wav")},
                data={"model": self.model, "language": self.language, "prompt": PROMPT, "response_format": "json"},
                timeout=self.timeout_s,
            )
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
            text = response.json().get("text")
        except Exception as error:  # noqa: BLE001 - the ear falls back to Vosk
            self.failures += 1
            log.warning("cloud transcription failed: %s", error)
            print(f"[laika] cloud speech recognition failed ({error}); using Vosk's text")
            return None
        self.failures = 0
        return text.strip() if isinstance(text, str) else None
