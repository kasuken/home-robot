from __future__ import annotations

import contextlib
import io
import random
import time
import unittest

from laika.ai_client import ToolCall
from laika.games import BY_ID, FOLLOW_UP_S, GAMES, GameRounds, GameSession, catalogue
from laika.limits import PAN_MAX, TILT_MIN
from laika.prompts import build_tool_system_prompt
from laika.robot import RobotSensorState
from laika.safety import SafetyController
from laika.tools import ToolRegistry
from tests.fakes import FakeClock, LookRecordingRobot


class FakeSafety:
    """Only what the rounds read: the monitor's latest reading, from a script of distances."""

    def __init__(self, distances):
        self.distances = list(distances)

    @property
    def latest_sensors(self):
        value = self.distances.pop(0) if len(self.distances) > 1 else self.distances[0]
        return RobotSensorState(distance_cm=value, cliff_detected=False, timestamp=time.time())


class StubSearcher:
    def __init__(self, found=True):
        self.found = found
        self.calls = 0

    def find_person(self, **kwargs):
        self.calls += 1
        return {"ok": True, "found": self.found, "how": "scan", "where": {"direction": "a sinistra"}}


class RoundCase(unittest.TestCase):
    def rounds(self, distances, searcher=None):
        self.robot = LookRecordingRobot(distance_cm=100)
        self.said = []
        clock = FakeClock(0.0)
        return GameRounds(
            self.robot,
            FakeSafety(distances),
            self.said.append,
            searcher=searcher,
            sleep=clock.advance,
            clock=clock,
            rng=random.Random(1),
        )


class RedLightTests(RoundCase):
    def test_standing_still_is_safe(self):
        result = self.rounds([150, 151, 149, 150]).red_light()
        self.assertEqual((result["seen"], result["moved"]), (True, False))
        self.assertNotIn("won", result)
        self.assertEqual(self.said[-1], "Stella!")
        self.assertIn((PAN_MAX, 0), self.robot.poses)  # she turned away to count

    def test_moving_is_caught(self):
        result = self.rounds([150, 140, 130, 120, 110]).red_light()
        self.assertTrue(result["moved"])
        self.assertGreater(result["movement_cm"], 12)

    def test_reaching_her_wins(self):
        result = self.rounds([20, 20, 21]).red_light()
        self.assertTrue(result["won"])

    def test_nobody_in_front_is_not_a_result(self):
        result = self.rounds([None]).red_light()
        self.assertFalse(result["seen"])
        self.assertIn("davanti", result["say"])

    def test_a_stale_reading_counts_as_none(self):
        rounds = self.rounds([150])
        rounds.safety = type("Old", (), {"latest_sensors": RobotSensorState(150, False, timestamp=time.time() - 5)})()
        self.assertFalse(rounds.red_light()["seen"])

    def test_an_interrupted_round_says_so(self):
        result = self.rounds([150]).red_light(should_stop=lambda: True)
        self.assertEqual(result, {"ok": False, "error": "interrupted"})


class StatueTests(RoundCase):
    def test_statues_watch_without_a_win(self):
        result = self.rounds([20, 20]).statues()
        self.assertEqual(self.said, ["Statua!"])
        self.assertFalse(result["moved"])
        self.assertNotIn("won", result)


class HideAndSeekTests(RoundCase):
    def test_she_counts_looking_down_then_searches(self):
        searcher = StubSearcher(found=True)
        result = self.rounds([100], searcher=searcher).hide_and_seek(lease=None)
        self.assertTrue(result["found"])
        self.assertEqual(searcher.calls, 1)
        self.assertIn("10! Pronto o no, arrivo!", self.said[0])
        self.assertIn((0, TILT_MIN), self.robot.poses)

    def test_without_a_searcher_it_is_refused(self):
        self.assertFalse(self.rounds([100]).hide_and_seek(lease=None)["ok"])


class SessionTests(unittest.TestCase):
    def test_a_game_waits_longer_for_answers_and_ends(self):
        session = GameSession()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(session.follow_up_s(15), 15)
            session.start("quiz")
            self.assertEqual(session.follow_up_s(15), FOLLOW_UP_S)
            self.assertEqual(session.end().id, "quiz")
        self.assertFalse(session.active)
        self.assertIsNone(session.end())

    def test_the_catalogue_is_varied_and_filtered(self):
        self.assertGreaterEqual(len(GAMES), 10)
        self.assertEqual(len(BY_ID), len(GAMES))
        self.assertEqual({g.kind for g in GAMES}, {"voice", "body"})
        ids = {g.id for g in catalogue(camera=False, search=False)}
        self.assertNotIn("caccia_al_colore", ids)
        self.assertNotIn("nascondino", ids)
        self.assertIn("un_due_tre_stella", ids)


class ToolTests(unittest.TestCase):
    def setUp(self):
        out = io.StringIO()
        redirect = contextlib.redirect_stdout(out)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)
        self.robot = LookRecordingRobot(distance_cm=100)
        self.safety = SafetyController(self.robot)
        self.addCleanup(self.safety.close)
        self.games = GameSession()
        self.registry = ToolRegistry(self.robot, self.safety, games=self.games)

    def call(self, name, **arguments):
        return self.registry.run(ToolCall("c1", name, arguments))

    def test_start_game_hands_the_rules(self):
        self.assertIn("start_game", self.registry.names())
        result = self.call("start_game", game="indovinelli")
        self.assertTrue(result["ok"])
        self.assertIn("indovinello", result["rules"])
        self.assertEqual(self.games.game.id, "indovinelli")
        self.assertFalse(self.call("play_round")["ok"])  # a word game has no rounds
        self.assertEqual(self.call("stop_game"), {"ok": True, "was_playing": "indovinelli"})

    def test_an_unknown_game_is_rejected(self):
        self.assertFalse(self.call("start_game", game="scacchi")["ok"])
        self.assertIsNone(self.games.game)

    def test_a_body_round_counts(self):
        self.call("start_game", game="statue")
        self.registry.game_rounds._sleep = lambda s: None
        self.registry.game_rounds._clock = iter(range(1000)).__next__
        result = self.call("play_round")
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["round"], 1)

    def test_the_prompt_mentions_games_only_when_offered(self):
        from laika.actions import ACTIONS

        self.assertIn("start_game", build_tool_system_prompt(ACTIONS, 3, True, games=True))
        self.assertNotIn("start_game", build_tool_system_prompt(ACTIONS, 3, True))


if __name__ == "__main__":
    unittest.main()
