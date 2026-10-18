from __future__ import annotations

import contextlib
import io
import unittest

from laika.actions import ACTIONS
from laika.ai_client import ToolCall
from laika.prompts import build_tool_system_prompt
from laika.safety import SafetyController
from laika.songs import BY_ID, SONGS, Singer
from laika.tools import ToolRegistry
from tests.fakes import LookRecordingRobot


class RecordingActions:
    """Only what the singer uses of the ActionPlayer."""

    def __init__(self):
        self.played = []
        self.cancelled = 0

    def play(self, *names):
        self.played.extend(names)

    def cancel_pending(self):
        self.cancelled += 1


class SingerTests(unittest.TestCase):
    def setUp(self):
        out = io.StringIO()
        redirect = contextlib.redirect_stdout(out)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)
        self.said = []
        self.actions = RecordingActions()
        self.singer = Singer(self.said.append, actions=self.actions)

    def test_she_sings_every_verse_in_order_with_its_gesture(self):
        song = BY_ID["bella_lavanderina"]
        result = self.singer.sing(song)
        self.assertEqual(result, {"ok": True, "song": song.id, "title": song.title, "verses_sung": len(song.verses)})
        self.assertEqual(self.said, [verse.text for verse in song.verses])
        self.assertEqual(self.actions.played, [verse.gesture or song.gesture for verse in song.verses])
        self.assertEqual(self.actions.cancelled, 1)

    def test_birthday_song_names_whoever_it_is_for(self):
        self.singer.sing(BY_ID["tanti_auguri"], name="Leonardo")
        self.assertIn("tanti auguri a Leonardo", self.said[-1])
        self.said.clear()
        self.singer.sing(BY_ID["tanti_auguri"], name="  ")
        self.assertIn("tanti auguri a te", self.said[-1])
        self.assertNotIn("{name}", " ".join(self.said))

    def test_a_cancelled_turn_stops_between_verses(self):
        song = BY_ID["canzone_di_laika"]
        result = self.singer.sing(song, should_stop=lambda: len(self.said) >= 2)
        self.assertEqual((result["ok"], result["error"], result["verses_sung"]), (False, "interrupted", 2))
        self.assertEqual(len(self.said), 2)
        self.assertEqual(self.actions.cancelled, 1)  # the waiting gestures go too

    def test_gestures_are_cancelled_when_the_voice_fails(self):
        def broken(_text):
            raise RuntimeError("no voice")

        with self.assertRaises(RuntimeError):
            Singer(broken, actions=self.actions).sing(BY_ID["fra_martino"])
        self.assertEqual(self.actions.cancelled, 1)

    def test_she_sings_without_gestures(self):
        result = Singer(self.said.append).sing(BY_ID["fra_martino"])
        self.assertTrue(result["ok"])


class CatalogueTests(unittest.TestCase):
    def test_the_catalogue_is_varied_and_well_formed(self):
        self.assertGreaterEqual(len(SONGS), 6)
        self.assertEqual(len(BY_ID), len(SONGS))
        self.assertEqual({s.origin for s in SONGS}, {"traditional", "laika"})
        for song in SONGS:
            self.assertTrue(song.verses, song.id)
            for verse in song.verses:
                self.assertTrue(verse.text.strip(), song.id)
                gesture = verse.gesture or song.gesture
                self.assertIn(gesture, ACTIONS, song.id)
                self.assertFalse(ACTIONS[gesture].requires & {"drive"}, f"{song.id}: {gesture} drives")

    def test_only_the_birthday_song_takes_a_name(self):
        with_name = {s.id for s in SONGS if any("{name}" in v.text for v in s.verses)}
        self.assertEqual(with_name, {"tanti_auguri"})


class ToolTests(unittest.TestCase):
    def setUp(self):
        out = io.StringIO()
        redirect = contextlib.redirect_stdout(out)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)
        self.robot = LookRecordingRobot(distance_cm=100)
        self.safety = SafetyController(self.robot)
        self.addCleanup(self.safety.close)
        self.said = []
        self.registry = ToolRegistry(self.robot, self.safety, speak=self.said.append)

    def call(self, name, **arguments):
        return self.registry.run(ToolCall("c1", name, arguments))

    def test_sing_sings_the_catalogue_words(self):
        self.assertIn("sing", self.registry.names())
        result = self.call("sing", song="giro_giro_tondo")
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.said, [v.text for v in BY_ID["giro_giro_tondo"].verses])

    def test_a_song_not_in_the_catalogue_is_rejected(self):
        self.assertFalse(self.call("sing", song="il_coccodrillo_come_fa")["ok"])
        self.assertEqual(self.said, [])

    def test_the_prompt_mentions_singing_only_when_offered(self):
        self.assertIn("usa sing", build_tool_system_prompt(ACTIONS, 3, True, singing=True))
        self.assertNotIn("usa sing", build_tool_system_prompt(ACTIONS, 3, True))


if __name__ == "__main__":
    unittest.main()
