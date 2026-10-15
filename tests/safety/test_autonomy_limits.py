"""The driving budget and the pause by voice (#51)."""

import contextlib
import io
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path

from laika.actions import ACTIONS, ActionContext
from laika.ai_client import AIReply
from laika.autonomy import AutonomyPause, DriveBudget, autonomy_command, local_day
from laika.behaviour import Arbiter, Priority
from laika.commands import Direction, MoveCommand, ResultStatus
from laika.config import VoiceConfig
from laika.conversation import PAUSED_SAY, RESUMED_SAY, Conversation
from laika.safety import SafetyController
from laika.store import Store
from laika.tools import PROACTIVE_TOOLS, TAINT_BLOCKED, Turn
from tests.agent.test_tools import ToolTestCase, call
from tests.behaviour.test_explore import ExploreCase, settings, wait_for
from tests.fakes import FakeClock, RecordingRobot
from tests.safety.test_ai_outage import FakeAI, OrderedPlayer


def move(ms=200, direction=Direction.FORWARD):
    return MoveCommand(direction=direction, speed=25, duration_ms=ms)


class DriveBudgetTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock(datetime(2023, 11, 14, 9, 0).timestamp())  # local 09:00

    def budget(self, hour=100.0, day=500.0, store=None):
        return DriveBudget(hour, day, store, wall_clock=self.clock)

    def test_no_limit_is_never_used_up(self):
        budget = DriveBudget(wall_clock=self.clock)
        budget.add(100000)
        self.assertEqual(budget.exhausted(), "")
        self.assertIsNone(budget.snapshot()["left_s"])

    def test_the_hourly_limit_slides(self):
        budget = self.budget(hour=100, day=10_000)
        budget.add(60)
        self.assertEqual(budget.exhausted(), "")
        self.clock.advance(1800)
        budget.add(45)
        self.assertEqual(budget.exhausted(), "hour")
        self.clock.advance(1801)  # the first 60 s left the hour
        self.assertEqual(budget.exhausted(), "")
        self.assertEqual(budget.snapshot()["hour_used_s"], 45.0)
        self.clock.advance(1800)
        self.assertEqual(budget.snapshot()["hour_used_s"], 0.0)

    def test_the_daily_limit_holds_until_the_next_local_day(self):
        budget = self.budget(hour=1000, day=200)
        for _ in range(3):
            budget.add(80)
            self.clock.advance(3700)  # always outside the hour window
        self.assertEqual(budget.exhausted(), "day")
        self.clock.advance(2 * 86400)
        self.assertEqual(budget.exhausted(), "")
        self.assertEqual(budget.snapshot()["day_used_s"], 0.0)

    def test_the_day_survives_a_restart_and_the_hour_does_not(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "laika.db"
            store = Store(path)
            budget = self.budget(hour=50, day=100, store=store)
            budget.add(60)
            self.assertEqual(budget.exhausted(), "hour")
            self.assertTrue(store.flush())
            store.close()

            again = Store(path)
            restarted = self.budget(hour=50, day=100, store=again)
            self.assertEqual(restarted.snapshot()["day_used_s"], 60.0)
            self.assertEqual(restarted.exhausted(), "")  # a new hour window, but the day is counted
            counting = self.budget(hour=500, day=100, store=again)
            counting.add(50)
            self.assertEqual(counting.exhausted(), "day")
            self.assertTrue(again.flush())
            again.close()

            third = Store(path)
            self.assertEqual(self.budget(hour=50, day=100, store=third).exhausted(), "day")
            third.close()

    def test_yesterdays_stored_total_is_not_loaded(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "laika.db"
            store = Store(path)
            store.kv_put("autonomy", "drive_day", {"date": local_day(self.clock() - 2 * 86400), "seconds": 9999})
            self.assertTrue(store.flush())
            self.assertEqual(self.budget(store=store).snapshot()["day_used_s"], 0.0)
            store.close()

    def test_nonsense_amounts_are_ignored(self):
        budget = self.budget()
        for bad in (0, -5, float("nan"), float("inf")):
            budget.add(bad)
        self.assertEqual(budget.snapshot()["day_used_s"], 0.0)


class ControllerBudgetTests(unittest.TestCase):
    def build(self, budget):
        self.robot = RecordingRobot(distance_cm=100)
        self.safety = SafetyController(
            self.robot, speed=25, allow_autonomous=True, drive_budget=budget, watchdog_timeout_s=0
        )
        self.addCleanup(self.safety.close)
        self.safety.start()
        self.arbiter = Arbiter(self.safety)
        self.addCleanup(self.arbiter.close)

    def lease(self, priority=Priority.EXPLORATION):
        lease = self.arbiter.request("test", priority)
        self.assertIsNotNone(lease)
        self.addCleanup(lease.release)
        return lease

    def test_autonomous_moves_stop_when_the_budget_is_used_up(self):
        budget = DriveBudget(per_hour_s=0.5)
        self.build(budget)
        lease = self.lease()
        results = []
        for _ in range(10):
            lease.kick()
            result = self.safety.execute(move(300), lease=lease)
            results.append((result.result, result.reason))
            if result.result is ResultStatus.REJECTED:
                break
        self.assertEqual(results[-1], (ResultStatus.REJECTED, "drive_budget"))
        self.assertLessEqual(len(results), 4)  # 0.3 s of real run time per move against 0.5 s
        self.assertEqual(budget.exhausted(), "hour")
        # What was counted is the real run time (a move lasts about as long as it was asked to).
        self.assertGreater(budget.snapshot()["hour_used_s"], 0.5)
        self.assertLess(budget.snapshot()["hour_used_s"], 2.0)

    def test_a_cancelled_move_counts_for_the_time_it_ran(self):
        budget = DriveBudget(per_hour_s=100)
        self.build(budget)
        lease = self.lease()
        command = move(1000)
        self.assertIsNone(self.safety.execute(command, wait=False, lease=lease))
        time.sleep(0.3)
        self.safety.stop()
        self.assertEqual(self.safety.wait(command.id, 1).result, ResultStatus.CANCELLED)
        used = budget.snapshot()["hour_used_s"]
        self.assertGreater(used, 0.2)
        self.assertLess(used, 0.9)

    def test_the_users_moves_are_not_limited_or_counted(self):
        budget = DriveBudget(per_hour_s=0.1, per_day_s=0.1)
        self.build(budget)
        budget.add(5)
        self.assertEqual(budget.exhausted(), "hour")
        self.assertEqual(self.safety.execute(move(100)).result, ResultStatus.SUCCESS)
        conversation = self.lease(Priority.CONVERSATION)
        self.assertEqual(self.safety.execute(move(100), lease=conversation).result, ResultStatus.SUCCESS)
        self.assertEqual(budget.snapshot()["hour_used_s"], 5.0)

    def test_a_self_directed_move_under_a_conversation_lease_is_limited(self):
        budget = DriveBudget(per_hour_s=1)
        self.build(budget)
        budget.add(5)
        conversation = self.lease(Priority.CONVERSATION)
        result = self.safety.execute(move(100), lease=conversation, self_directed=True)
        self.assertEqual((result.result, result.reason), (ResultStatus.REJECTED, "drive_budget"))

    def test_the_status_is_visible(self):
        budget = DriveBudget(per_hour_s=100, per_day_s=400)
        self.build(budget)
        budget.add(30)
        status = self.safety.autonomy_status()
        self.assertEqual(status["drive_budget"]["hour_used_s"], 30.0)
        self.assertEqual(status["drive_budget"]["left_s"], 70.0)
        self.assertFalse(status["paused"])


class ExplorationBudgetTests(ExploreCase):
    def test_the_budget_stops_idle_driving_once_it_is_used_up(self):
        budget = DriveBudget(per_hour_s=0.6)
        explorer = self.build(
            config=settings(step_ms=250, scan_every_steps=2), safety_kwargs={"drive_budget": budget}
        )
        self.assertTrue(explorer.start())
        self.assertTrue(wait_for(lambda: explorer.mode == "head_only"))
        self.assertGreaterEqual(len(self.robot.forward_drives), 1)
        self.assertLessEqual(len(self.robot.forward_drives), 5)
        self.assertEqual(budget.exhausted(), "hour")
        count = len(self.robot.trace)
        self.assertTrue(wait_for(lambda: explorer.scans >= 1))  # she keeps looking around
        time.sleep(0.4)
        self.assertEqual(len(self.robot.trace), count)
        explorer.stop()

    def test_a_new_session_with_the_budget_used_up_only_looks_around(self):
        budget = DriveBudget(per_day_s=10)
        budget.add(10)
        explorer = self.build(safety_kwargs={"drive_budget": budget})
        self.assertFalse(explorer.driving)
        self.assertTrue(explorer.start())
        self.assertEqual(explorer.mode, "head_only")
        self.assertTrue(wait_for(lambda: explorer.scans >= 1))
        explorer.stop()
        self.assertEqual(self.robot.forward_drives, [])


class PauseStateTests(unittest.TestCase):
    def test_pause_and_resume_survive_a_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "laika.db"
            store = Store(path)
            pause = AutonomyPause(store)
            self.assertFalse(pause.paused)
            self.assertTrue(pause.pause())
            self.assertFalse(pause.pause())  # already paused
            self.assertTrue(store.flush())
            store.close()

            again = Store(path)
            restarted = AutonomyPause(again)
            self.assertTrue(restarted.paused)
            self.assertIsNotNone(restarted.since)
            robot = RecordingRobot(distance_cm=100)
            controller = SafetyController(robot, pause=restarted, watchdog_timeout_s=0)
            self.addCleanup(controller.close)
            self.assertTrue(controller.paused)
            self.assertTrue(restarted.resume())
            self.assertTrue(again.flush())
            again.close()

            third = Store(path)
            self.assertFalse(AutonomyPause(third).paused)
            third.close()

    def test_phrases(self):
        for text in ("Laika, resta ferma", "resta ferma!", "stai ferma un attimo", "Non ti muovere", "rimani ferma"):
            self.assertEqual(autonomy_command(text), "pause", text)
        for text in ("puoi muoverti di nuovo", "Laika puoi muoverti di nuovo?", "muoviti pure", "riprendi a muoverti", "puoi esplorare"):
            self.assertEqual(autonomy_command(text), "resume", text)
        for text in (
            "dove sei?", "muoviti", "stai fuori dallo studio", "non entrare in camera",
            "il mio cane sta sempre fermo quando arriva il postino e poi resta ferma a guardare la porta",
        ):
            self.assertIsNone(autonomy_command(text), text)


class PauseBehaviourTests(ExploreCase):
    def test_resta_ferma_stops_every_autonomous_activity_until_it_is_lifted(self):
        explorer = self.build()
        self.assertTrue(explorer.start())
        self.assertTrue(wait_for(lambda: explorer.steps >= 2))
        self.assertIsNotNone(self.arbiter.holder)
        self.safety.pause_autonomy()
        # The exploration lease is revoked and the session ends, with the reason.
        self.assertTrue(wait_for(lambda: not explorer.active))
        self.assertEqual(explorer.end_reason, "paused")
        self.assertIsNone(self.arbiter.holder)
        self.assert_stays_still()
        # No new autonomous lease, whatever the priority of the autonomous behaviour.
        for priority in (Priority.CURIOSITY, Priority.SOCIAL, Priority.EXPLORATION):
            self.assertIsNone(self.arbiter.request("x", priority))
        self.assertFalse(explorer.start())
        self.assertIn("paused", explorer.refusal)
        # The conversation and the user are not touched.
        conversation = self.arbiter.request("conversation", Priority.CONVERSATION)
        self.assertIsNotNone(conversation)
        self.assertEqual(self.safety.execute(move(100), lease=conversation).result, ResultStatus.SUCCESS)
        conversation.release()
        # Lifted by the user: exploration works again.
        self.safety.resume_autonomy()
        self.assertTrue(explorer.start())
        self.assertTrue(wait_for(lambda: explorer.steps >= 1))
        explorer.stop()

    def test_autonomous_moves_are_refused_and_a_move_in_flight_is_cancelled(self):
        self.build()
        lease = self.arbiter.request("explore", Priority.EXPLORATION)
        command = move(1000)
        self.assertIsNone(self.safety.execute(command, wait=False, lease=lease))
        self.assertTrue(wait_for(lambda: self.robot.speed != 0))
        self.safety.pause_autonomy()
        result = self.safety.wait(command.id, 1)
        self.assertEqual((result.result, result.reason), (ResultStatus.CANCELLED, "paused"))
        self.assertEqual(self.robot.speed, 0)
        self.assertFalse(lease.active)
        # A move that is not under an autonomous lease but is her own choice (a search) is refused too.
        conversation = self.arbiter.request("conversation", Priority.CONVERSATION)
        refused = self.safety.execute(move(100), lease=conversation, self_directed=True)
        self.assertEqual((refused.result, refused.reason), (ResultStatus.REJECTED, "paused"))
        self.assertEqual(self.safety.execute(move(100), lease=conversation).result, ResultStatus.SUCCESS)
        self.assertEqual(self.safety.execute(move(100)).result, ResultStatus.SUCCESS)

    def test_a_pause_does_not_end_a_follow_up(self):
        self.build()
        follow_up = self.arbiter.request("follow-up", Priority.FOLLOW_UP)
        self.assertIsNotNone(follow_up)
        self.safety.pause_autonomy()
        self.assertTrue(follow_up.active)
        self.assertIsNotNone(self.arbiter.holder)


class PauseToolTests(ToolTestCase):
    def test_the_tools_pause_and_resume(self):
        result, _ = self.run_tool("pause_autonomy")
        self.assertTrue(result["ok"] and result["paused"])
        self.assertTrue(self.safety.paused)
        state, _ = self.run_tool("get_robot_state")
        self.assertTrue(state["autonomy"]["paused"])
        result, _ = self.run_tool("resume_autonomy")
        self.assertTrue(result["ok"] and result["was_paused"])
        self.assertFalse(self.safety.paused)

    def test_the_state_shows_the_limits(self):
        state, _ = self.run_tool("get_robot_state")
        self.assertEqual(set(state["autonomy"]), {"paused", "drive_budget", "no_go_rooms", "in_no_go_room"})

    def test_a_proactive_turn_cannot_resume_or_pause(self):
        self.assertNotIn("resume_autonomy", PROACTIVE_TOOLS)
        self.assertNotIn("pause_autonomy", PROACTIVE_TOOLS)
        restricted = self.registry.restricted(PROACTIVE_TOOLS)
        self.assertNotIn("resume_autonomy", restricted.names())
        self.safety.pause_autonomy()
        result = restricted.run(call("resume_autonomy"))
        self.assertFalse(result["ok"])
        self.assertTrue(self.safety.paused)

    def test_a_turn_that_read_the_web_cannot_resume(self):
        self.assertIn("resume_autonomy", TAINT_BLOCKED)
        self.safety.pause_autonomy()
        turn = Turn()
        turn.tainted = True
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.registry.run(call("resume_autonomy"), turn)
        self.assertFalse(result["ok"])
        self.assertTrue(self.safety.paused)


class SpokenPauseTests(unittest.TestCase):
    """"Laika, resta ferma" works from the conversation itself, so also without the model."""

    def setUp(self):
        self.robot = RecordingRobot(distance_cm=100)
        self.robot.verbose = False
        self.safety = SafetyController(self.robot, speed=25, watchdog_timeout_s=0)
        self.addCleanup(self.safety.close)
        self.player = OrderedPlayer(
            ActionContext(self.robot, self.safety),
            {k: ACTIONS[k] for k in ("think", "sad", "look_forward", "nod")},
            talk_motion=False,
        )
        self.addCleanup(self.player.close)
        self.ai = FakeAI([AIReply("non dovrei essere interpellata")])
        self.conversation = Conversation(
            ear=None, robot=self.robot, ai=self.ai, actions=self.player, voice=VoiceConfig(), safety=self.safety
        )

    def say(self, text):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.conversation.reply(text)

    def test_resta_ferma_pauses_and_resume_lifts_it_without_asking_the_model(self):
        self.say("Laika, resta ferma")
        self.assertTrue(self.safety.paused)
        self.assertEqual(self.robot.spoken[-1], PAUSED_SAY)
        self.say("Laika, puoi muoverti di nuovo")
        self.assertFalse(self.safety.paused)
        self.assertEqual(self.robot.spoken[-1], RESUMED_SAY)
        self.assertEqual(len(self.ai.replies), 1)  # the model was never asked

    def test_other_sentences_go_to_the_model(self):
        reply = self.say("che tempo fa oggi")
        self.assertEqual(reply.say, "non dovrei essere interpellata")
        self.assertFalse(self.safety.paused)


class PromptTests(ToolTestCase):
    def test_the_prompt_tells_her_about_the_pause_and_the_house_rules_only_when_asked(self):
        from laika.prompts import autonomy_prompt, build_tool_system_prompt

        gestures = self.registry.actions.allowed
        plain = build_tool_system_prompt(gestures, 3)
        with_rules = build_tool_system_prompt(gestures, 3, autonomy=True)
        self.assertNotIn("pause_autonomy", plain)
        self.assertIn("pause_autonomy", with_rules)
        self.assertIn("resume_autonomy", with_rules)
        self.assertIn("non entrare in camera", with_rules)  # the user's own requests are mentioned, not blocked
        self.assertEqual(autonomy_prompt(False, []), "")
        self.assertIn("stare ferma", autonomy_prompt(True, []))
        self.assertIn("living room, office", autonomy_prompt(False, ["living_room", "office"]))

    def test_the_agent_shows_the_pause_and_the_no_go_rooms(self):
        from laika.agent import Agent
        from laika.config import HouseConfig
        from laika.house_rules import HouseRules
        from laika.memory import Memory

        agent = Agent(None, self.registry, "sistema", world=self.world)
        self.assertEqual(agent._autonomy_note(), "")
        self.safety.house_rules = HouseRules(Memory(), self.world, HouseConfig(no_go_rooms=["bagno"]))
        self.safety.pause_autonomy()
        note = agent._autonomy_note()
        self.assertIn("stare ferma", note)
        self.assertIn("bathroom", note)
        self.assertIn(note, agent._start_messages("ciao")[0]["content"])


if __name__ == "__main__":
    unittest.main()
