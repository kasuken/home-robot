import contextlib
import io
import threading
import time
import unittest
from unittest import mock

from laika import audio
from laika.audio import SpeakerMouth, VoskEar, make_mouth


class FakeEngine:
    def __init__(self, fails=False):
        self.fails = fails
        self.said = []

    def say(self, text):
        if self.fails:
            raise RuntimeError("no network")
        self.said.append(text)


def quietly(action):
    with contextlib.redirect_stdout(io.StringIO()):
        return action()


class SpeakerMouthTests(unittest.TestCase):
    def test_uses_the_main_engine_when_it_works(self):
        engine, backups = FakeEngine(), []
        mouth = SpeakerMouth(engine, backup=lambda: backups.append(FakeEngine()) or backups[-1])
        quietly(lambda: mouth.say("Ciao"))
        self.assertEqual(engine.said, ["Ciao"])
        self.assertEqual(backups, [])

    def test_falls_back_and_reuses_the_backup(self):
        backups = []

        def make_backup():
            backups.append(FakeEngine())
            return backups[-1]

        mouth = SpeakerMouth(FakeEngine(fails=True), backup=make_backup)
        quietly(lambda: mouth.say("Uno"))
        quietly(lambda: mouth.say("Due"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].said, ["Uno", "Due"])

    def test_never_raises_when_every_voice_fails(self):
        mouth = SpeakerMouth(FakeEngine(fails=True), backup=lambda: FakeEngine(fails=True))
        quietly(lambda: mouth.say("Ciao"))

    def test_rejects_unknown_engines(self):
        with self.assertRaises(ValueError):
            make_mouth("robotic", edge_voice="x", piper_model="y")



class ScriptedVosk:
    """SunFounder-like streaming recogniser: (seconds, partial text) steps, then a final sentence."""

    def __init__(self, partials=(), final=None, final_at=None):
        self.partials = list(partials)
        self.final = final
        self.final_at = final_at
        self.stop = threading.Event()

    def listen(self, stream=False):
        assert stream, "the ear streams so it can tell when someone starts talking"
        self.stop.clear()
        return self._stream(time.monotonic())

    def _stream(self, start):
        for at, text in self.partials:
            if self.stop.wait(max(0.0, start + at - time.monotonic())):
                return
            yield {"done": False, "partial": text, "final": ""}
        if self.final is None:
            self.stop.wait(10)
            return
        if self.stop.wait(max(0.0, start + self.final_at - time.monotonic())):
            return
        yield {"done": True, "partial": "", "final": self.final}

    def stop_listening(self):
        self.stop.set()


def ear(vosk):
    instance = VoskEar.__new__(VoskEar)  # skip loading the real model
    instance._stt = vosk
    instance._interrupted = threading.Event()
    return instance


class VoskEarTimeoutTests(unittest.TestCase):
    def test_silence_ends_the_wait_after_the_timeout(self):
        started = time.monotonic()
        self.assertIsNone(ear(ScriptedVosk()).listen(timeout_s=0.3))
        self.assertLess(time.monotonic() - started, 2)

    def test_someone_who_started_talking_is_not_cut_off(self):
        # Starts answering just before the 0.4 s timeout and finishes well after it.
        vosk = ScriptedVosk(partials=[(0.2, "è uno"), (0.6, "è uno stabilizzatore")],
                            final="è uno stabilizzatore per la fotocamera", final_at=1.0)
        self.assertEqual(ear(vosk).listen(timeout_s=0.4), "è uno stabilizzatore per la fotocamera")

    def test_one_sentence_is_capped(self):
        vosk = ScriptedVosk(partials=[(0.1, "ehm")])  # noise that never becomes a sentence
        with mock.patch.object(audio, "MAX_UTTERANCE_S", 0.5):
            started = time.monotonic()
            self.assertIsNone(ear(vosk).listen(timeout_s=0.3))
        self.assertLess(time.monotonic() - started, 3)

    def test_without_a_timeout_it_waits_for_speech(self):
        vosk = ScriptedVosk(final="ehi laika", final_at=0.5)
        self.assertEqual(ear(vosk).listen(), "ehi laika")

if __name__ == "__main__":
    unittest.main()
