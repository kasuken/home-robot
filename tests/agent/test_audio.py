import contextlib
import io
import unittest

from laika.audio import SpeakerMouth, make_mouth


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


if __name__ == "__main__":
    unittest.main()
