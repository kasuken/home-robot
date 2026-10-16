from __future__ import annotations

import contextlib
import io
import json
import threading
import unittest
import wave

from laika.audio import CloudEar
from laika.transcribe import Transcriber, wav_bytes
from laika.wake_word import WakeWordMatcher


class ScriptMic:
    """Blocks of fake audio; each block is its own label so the test can see what was sent."""

    samplerate = 16000

    def __init__(self, blocks):
        self.blocks = list(blocks)
        self.opened = 0
        self.closed = 0

    def open(self):
        self.opened += 1

    def read(self, timeout_s):
        return self.blocks.pop(0) if self.blocks else None

    def close(self):
        self.closed += 1


class ScriptRecognizer:
    """Plays Vosk: a block b"w:<text>" is a word (partial), b"end" ends the utterance, anything else is silence."""

    def __init__(self):
        self.words = []
        self.final = None

    def AcceptWaveform(self, block):
        if block == b"end":
            self.final = " ".join(self.words)
            return True
        if block.startswith(b"w:"):
            self.words.append(block[2:].decode())
        return False

    def PartialResult(self):
        return json.dumps({"partial": " ".join(self.words)})

    def Result(self):
        return json.dumps({"text": self.final or ""})

    def FinalResult(self):
        return json.dumps({"text": " ".join(self.words)})


class FakeTranscriber:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def transcribe(self, pcm, samplerate):
        self.calls.append(pcm)
        return self.text


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 0.2  # one block per look at the clock
        return self.now


def ear(blocks, cloud_text):
    transcriber = FakeTranscriber(cloud_text)
    mic = ScriptMic(blocks)
    out = CloudEar(
        "it", transcriber, wants_cloud=WakeWordMatcher(["laika", "laica"]).match,
        mic=mic, recognizer=ScriptRecognizer, clock=Clock(),
    )
    return out, transcriber, mic


class CloudEarTests(unittest.TestCase):
    def setUp(self):
        redirect = contextlib.redirect_stdout(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def test_the_wake_utterance_is_transcribed_in_the_cloud(self):
        listener, cloud, mic = ear([b"q", b"w:ehi", b"w:laica", b"w:che", b"end"], "Ehi Laika, che ore sono?")
        self.assertEqual(listener.listen(), "Ehi Laika, che ore sono?")
        # The audio from just before the first word is sent too, so nothing is cut.
        self.assertEqual(cloud.calls, [b"qw:ehiw:laicaw:cheend"])
        self.assertEqual((mic.opened, mic.closed), (1, 1))

    def test_talk_without_the_wake_word_stays_on_the_robot(self):
        listener, cloud, _ = ear([b"w:passami", b"w:il", b"w:sale", b"end"], "never")
        self.assertEqual(listener.listen(), "passami il sale")
        self.assertEqual(cloud.calls, [])

    def test_during_a_conversation_everything_heard_goes_to_the_cloud(self):
        listener, cloud, _ = ear([b"w:si", b"end"], "Sì, il quaderno rosso!")
        self.assertEqual(listener.listen(timeout_s=15), "Sì, il quaderno rosso!")
        self.assertEqual(len(cloud.calls), 1)

    def test_a_cloud_failure_falls_back_to_vosk(self):
        listener, _, _ = ear([b"w:ciao", b"end"], None)
        self.assertEqual(listener.listen(timeout_s=15), "ciao")

    def test_a_cloud_text_without_the_wake_word_does_not_hide_it(self):
        listener, _, _ = ear([b"w:ehi", b"w:laica", b"end"], "Ehi, ciao a tutti")
        self.assertEqual(listener.listen(), "ehi laica")  # Vosk's text still wakes her

    def test_noise_without_words_is_never_sent(self):
        listener, cloud, _ = ear([b"x", b"end", b"x"], "made up")
        self.assertIsNone(listener.listen(timeout_s=1))
        self.assertEqual(cloud.calls, [])

    def test_silence_times_out(self):
        listener, cloud, mic = ear([b"x"] * 50, "never")
        self.assertIsNone(listener.listen(timeout_s=2))
        self.assertEqual((cloud.calls, mic.closed), ([], 1))

    def test_an_interrupt_ends_the_listen(self):
        listener, _, _ = ear([b"x"] * 50, "never")
        listener.interrupt()
        self.assertIsNone(listener.listen())
        timer = threading.Timer(0.05, listener.interrupt)
        timer.start()
        listener.mic.blocks = [b"x"] * 10**6
        self.assertIsNone(listener.listen())


class TranscriberTests(unittest.TestCase):
    def test_it_posts_a_wav_and_returns_the_text(self):
        seen = {}

        class Response:
            status_code = 200

            def json(self):
                return {"text": " Ciao Laika! "}

        class Session:
            def post(self, url, **kwargs):
                seen.update(url=url, **kwargs)
                return Response()

        transcriber = Transcriber(base_url="https://api.example/v1/", api_key="k", model="m", session=Session())
        self.assertEqual(transcriber.transcribe(b"\0\0" * 100, 16000), "Ciao Laika!")
        self.assertEqual(seen["url"], "https://api.example/v1/audio/transcriptions")
        self.assertEqual(seen["data"]["language"], "it")
        self.assertEqual(seen["headers"], {"Authorization": "Bearer k"})
        with wave.open(io.BytesIO(seen["files"]["file"][1])) as sound:
            self.assertEqual((sound.getframerate(), sound.getnframes()), (16000, 100))

    def test_an_error_returns_none(self):
        class Session:
            def post(self, url, **kwargs):
                raise OSError("offline")

        transcriber = Transcriber(base_url="u", api_key="", model="m", session=Session())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(transcriber.transcribe(b"\0\0", 16000))
        self.assertEqual(transcriber.failures, 1)
        self.assertIsNone(transcriber.transcribe(b"", 16000))

    def test_wav_bytes(self):
        with wave.open(io.BytesIO(wav_bytes(b"\1\0" * 8, 8000))) as sound:
            self.assertEqual((sound.getnchannels(), sound.getsampwidth(), sound.getnframes()), (1, 2, 8))


if __name__ == "__main__":
    unittest.main()
