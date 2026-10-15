import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from laika.actions import available_actions
from laika.config import InteractionConfig, PresenceConfig, load_config, parse_quiet_hours
from laika.events import EventBus, EventType
from laika.interaction_policy import Availability, Context, Decision, InteractionPolicy
from laika.personality import Personality
from laika.presence import PresenceTracker
from laika.prompts import build_tool_system_prompt
from laika.store import Store
from laika.tools import ToolRegistry
from tests.agent.test_tools import ToolTestCase
from tests.fakes import FakeClock

A = Availability
# Neutral initiative and tolerance (50): the factors are 1.0, so these tests check the configured numbers.
NEUTRAL = Personality(initiative=50, interruption_tolerance=50)


def at(hour, minute=0):
    return datetime(2026, 10, 15, hour, minute).timestamp()


class FakePresence:
    def __init__(self, clock):
        self.clock = clock
        self.present = False
        self.since = None
        self.last_seen = None

    def arrive(self):
        self.present, self.since, self.last_seen = True, self.clock(), self.clock()

    def leave(self):
        self.present, self.since = False, None


class PolicyTestCase(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock(1000.0)
        self.wall = FakeClock(at(14))
        self.bus = EventBus()
        self.bus.start()
        self.addCleanup(self.bus.close)
        self.presence = FakePresence(self.clock)
        self.questions = 0
        self.busy = False

    def advance(self, seconds):
        self.clock.advance(seconds)
        self.wall.advance(seconds)

    def policy(self, store=None, personality=NEUTRAL, **overrides):
        policy = InteractionPolicy(
            InteractionConfig(**overrides),
            self.presence,
            events=self.bus,
            store=store,
            threshold=0.35,
            pending_questions=lambda: self.questions,
            body_busy=lambda: self.busy,
            clock=self.clock,
            wall_clock=self.wall,
            personality=personality,
        )
        policy.start()
        self.addCleanup(policy.close)
        return policy

    def ready(self, **overrides):
        """A policy in the state where the first rule allows: someone settled, something to say."""
        policy = self.policy(**overrides)
        self.presence.arrive()
        self.advance(10)
        policy.note_interest(0.8)
        return policy

    def emit(self, type, **data):
        self.bus.emit(type, **data)
        self.bus.drain()


class DecisionTableTests(PolicyTestCase):
    def check(self, policy, allowed, reason, state):
        self.assertEqual(policy.decide(), Decision(allowed, reason, state))

    def test_the_first_rule_allows(self):
        self.check(self.ready(), True, "allowed", A.AVAILABLE)

    def test_disabled(self):
        self.check(self.ready(enabled=False), False, "disabled", A.AVAILABLE)

    def test_sleeping_in_quiet_hours(self):
        policy = self.ready()
        self.wall.now = at(22)
        self.check(policy, False, "sleeping", A.SLEEPING)

    def test_do_not_disturb(self):
        policy = self.ready()
        policy.set_do_not_disturb(30)
        self.check(policy, False, "do_not_disturb", A.DO_NOT_DISTURB)

    def test_busy_because_the_user_said_so(self):
        policy = self.ready()
        policy.set_busy(30)
        self.check(policy, False, "busy", A.BUSY)

    def test_busy_while_the_body_is_held_by_a_conversation(self):
        policy = self.ready()
        self.busy = True
        self.check(policy, False, "busy", A.BUSY)

    def test_busy_during_an_interaction_seen_by_event(self):
        policy = self.ready()
        self.emit(EventType.INTERACTION_STARTED)
        self.check(policy, False, "busy", A.BUSY)
        self.emit(EventType.INTERACTION_COMPLETED)
        self.check(policy, False, "cooldown", A.AVAILABLE)

    def test_away_when_nobody_was_seen_for_a_while(self):
        policy = self.ready()
        self.presence.leave()
        self.advance(880)
        self.check(policy, False, "unknown", A.UNKNOWN)
        self.advance(25)
        self.check(policy, False, "away", A.AWAY)

    def test_never_seeing_anybody_becomes_away_not_unknown_forever(self):
        policy = self.policy()
        self.check(policy, False, "unknown", A.UNKNOWN)
        self.advance(901)
        self.check(policy, False, "away", A.AWAY)

    def test_unknown_without_a_presence_tracker(self):
        policy = InteractionPolicy(InteractionConfig(), None, clock=self.clock, wall_clock=self.wall)
        self.check(policy, False, "unknown", A.UNKNOWN)

    def test_cooldown(self):
        policy = self.ready()
        policy.record_attempt()
        self.check(policy, False, "cooldown", A.AVAILABLE)

    def test_nothing_to_say(self):
        policy = self.policy()
        self.presence.arrive()
        self.advance(10)
        self.check(policy, False, "nothing_to_say", A.AVAILABLE)

    def test_nothing_to_say_is_something_to_say_for_an_encounter(self):
        policy = self.policy()
        self.presence.arrive()
        self.advance(10)
        self.check(policy, False, "nothing_to_say", A.AVAILABLE)
        self.assertEqual(policy.decide(Context(encounter=True)), Decision(True, "allowed", A.AVAILABLE))
        policy.record_attempt()  # the cooldown still applies to an encounter
        self.assertEqual(policy.decide(Context(encounter=True)).reason, "cooldown")

    def test_rate_limit(self):
        policy = self.ready(cooldown_s=0, max_per_hour=2)
        policy.record_attempt()
        policy.record_attempt()
        self.check(policy, False, "rate_limit", A.AVAILABLE)

    def test_state_precedence(self):
        policy = self.ready()
        policy.set_do_not_disturb(30)
        policy.set_busy(30)
        self.busy = True
        self.assertEqual(policy.decide().state, A.DO_NOT_DISTURB)
        self.wall.now = at(23)
        self.assertEqual(policy.decide().state, A.SLEEPING)


class SettleTests(PolicyTestCase):
    def test_someone_walking_past_is_not_ambushed(self):
        policy = self.policy(settle_s=5)
        policy.note_interest(0.9)
        self.presence.arrive()
        self.advance(4.9)
        self.assertEqual(policy.decide(), Decision(False, "unknown", A.UNKNOWN))
        self.advance(0.2)
        self.assertTrue(policy.decide().allowed)

    def test_a_real_tracker_drives_the_state(self):
        tracker = PresenceTracker(self.bus, PresenceConfig(), clock=self.clock)
        policy = InteractionPolicy(
            InteractionConfig(), tracker, clock=self.clock, wall_clock=self.wall, personality=NEUTRAL
        )
        policy.note_interest(0.9)
        tracker.observe({"people": 1, "confidence": 0.9})
        self.assertEqual(policy.state(), A.UNKNOWN)
        self.advance(6)
        tracker.observe({"people": 1, "confidence": 0.9})
        self.assertEqual(policy.state(), A.AVAILABLE)
        self.assertTrue(policy.decide().allowed)


class CooldownTests(PolicyTestCase):
    def test_a_normal_conversation_starts_the_cooldown(self):
        policy = self.ready()
        self.emit(EventType.INTERACTION_STARTED)
        self.emit(EventType.INTERACTION_COMPLETED)
        self.advance(1799)
        self.assertEqual(policy.decide().reason, "cooldown")
        self.advance(2)
        policy.note_interest(0.8)
        self.assertEqual(policy.decide(), Decision(True, "allowed", A.AVAILABLE))

    def test_the_cooldown_runs_from_the_end_of_a_long_conversation(self):
        policy = self.ready()
        self.emit(EventType.INTERACTION_STARTED)
        self.advance(1700)
        self.emit(EventType.INTERACTION_COMPLETED)
        self.advance(1700)
        policy.note_interest(0.8)
        self.assertEqual(policy.decide().reason, "cooldown")

    def test_a_proactive_attempt_starts_the_cooldown(self):
        policy = self.ready()
        policy.record_attempt()
        self.advance(1799)
        self.assertEqual(policy.decide().reason, "cooldown")
        self.advance(2)
        policy.note_interest(0.8)
        self.assertTrue(policy.decide().allowed)

    def test_a_lost_completed_event_does_not_block_for_ever(self):
        policy = self.ready()
        self.emit(EventType.INTERACTION_STARTED)
        self.assertEqual(policy.decide().reason, "busy")
        self.advance(3601)
        policy.note_interest(0.8)
        self.assertNotEqual(policy.decide().reason, "busy")

    def test_recent_interactions_are_tracked_and_bounded(self):
        policy = self.ready()
        for _ in range(100):
            policy.record_attempt()
        self.emit(EventType.INTERACTION_STARTED)
        recent = policy.recent_interactions()
        self.assertLessEqual(len(recent), 64)
        self.assertEqual(recent[-1][0], "started")


class RateLimitTests(PolicyTestCase):
    def test_at_most_max_per_hour(self):
        policy = self.ready(cooldown_s=60, max_per_hour=2)
        policy.record_attempt()
        self.advance(100)
        policy.note_interest(0.8)
        self.assertTrue(policy.decide().allowed)
        policy.record_attempt()
        self.advance(100)
        policy.note_interest(0.8)
        self.assertEqual(policy.decide().reason, "rate_limit")
        self.advance(3500)  # the first attempt is now more than an hour old
        policy.note_interest(0.8)
        self.assertTrue(policy.decide().allowed)


class QuietHoursTests(PolicyTestCase):
    def sleeping(self, policy, hour, minute=0):
        self.wall.now = at(hour, minute)
        return policy.state() is A.SLEEPING

    def test_across_midnight(self):
        policy = self.policy(quiet_hours="21:30-07:30")
        self.assertFalse(self.sleeping(policy, 21, 29))
        self.assertTrue(self.sleeping(policy, 21, 30))
        self.assertTrue(self.sleeping(policy, 23, 59))
        self.assertTrue(self.sleeping(policy, 0, 0))
        self.assertTrue(self.sleeping(policy, 7, 29))
        self.assertFalse(self.sleeping(policy, 7, 30))
        self.assertFalse(self.sleeping(policy, 12))

    def test_inside_one_day(self):
        policy = self.policy(quiet_hours="13:00-15:00")
        self.assertTrue(self.sleeping(policy, 13))
        self.assertTrue(self.sleeping(policy, 14, 59))
        self.assertFalse(self.sleeping(policy, 15))
        self.assertFalse(self.sleeping(policy, 12, 59))

    def test_empty_means_never(self):
        policy = self.policy(quiet_hours="")
        self.assertFalse(self.sleeping(policy, 3))

    def test_parsing(self):
        self.assertEqual(parse_quiet_hours("21:30-07:30"), (1290, 450))
        self.assertIsNone(parse_quiet_hours(" "))
        for bad in ("night", "25:00-07:00", "21:30-07:61", "10:00-10:00", "21:30", 5):
            with self.assertRaises(ValueError, msg=repr(bad)):
                parse_quiet_hours(bad)


class UserSettingTests(PolicyTestCase):
    def test_do_not_disturb_expires(self):
        policy = self.ready()
        until = policy.set_do_not_disturb(10)
        self.assertEqual(until, self.wall.now + 600)
        self.advance(599)
        self.assertEqual(policy.decide().reason, "do_not_disturb")
        self.advance(2)
        policy.note_interest(0.8)
        self.assertEqual(policy.decide().reason, "allowed")
        self.assertIsNone(policy.do_not_disturb_until)

    def test_clear(self):
        policy = self.ready()
        policy.set_do_not_disturb(60)
        self.assertTrue(policy.clear_do_not_disturb())
        self.assertFalse(policy.clear_do_not_disturb())
        self.assertTrue(policy.decide().allowed)

    def test_busy_expires_and_clears(self):
        policy = self.ready()
        policy.set_busy(5)
        self.assertEqual(policy.decide().reason, "busy")
        self.advance(301)
        self.assertTrue(policy.decide().allowed)
        policy.set_busy(5)
        self.assertTrue(policy.clear_busy())
        self.assertTrue(policy.decide().allowed)

    def test_a_shorter_request_never_shortens_do_not_disturb(self):
        policy = self.policy()
        first = policy.set_do_not_disturb(120)
        self.assertEqual(policy.set_do_not_disturb(5), first)

    def test_minutes_are_validated(self):
        policy = self.policy()
        for bad in (0, 4, 721, -1, float("nan"), True, "10"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                policy.set_do_not_disturb(bad)
        self.assertIsNone(policy.do_not_disturb_until)

    def test_the_settings_survive_a_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "laika.db"
            store = Store(path)
            policy = self.policy(store=store)
            dnd = policy.set_do_not_disturb(60)
            busy = policy.set_busy(30)
            self.assertTrue(store.flush())
            store.close()

            again = Store(path)
            restarted = self.policy(store=again)
            self.assertEqual(restarted.do_not_disturb_until, dnd)
            self.assertEqual(restarted.busy_until, busy)
            self.presence.arrive()
            self.advance(10)
            restarted.note_interest(0.9)
            self.assertEqual(restarted.decide().reason, "do_not_disturb")
            restarted.clear_do_not_disturb()
            self.assertTrue(again.flush())
            self.assertIsNone(again.kv_get("interaction", "do_not_disturb_until"))
            again.close()

    def test_an_expired_setting_is_gone_after_a_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / "laika.db")
            policy = self.policy(store=store)
            policy.set_do_not_disturb(10)
            self.assertTrue(store.flush())
            self.advance(601)
            self.assertIsNone(self.policy(store=store).do_not_disturb_until)
            store.close()

    def test_garbage_in_the_store_is_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / "laika.db")
            store.kv_put("interaction", "do_not_disturb_until", "tomorrow")
            store.kv_put("interaction", "busy_until", True)
            self.assertTrue(store.flush())
            policy = self.policy(store=store)
            self.assertIsNone(policy.do_not_disturb_until)
            self.assertIsNone(policy.busy_until)
            store.close()


class InterestTests(PolicyTestCase):
    def settled(self, **overrides):
        policy = self.policy(**overrides)
        self.presence.arrive()
        self.advance(10)
        return policy

    def test_below_the_threshold_is_not_worth_saying(self):
        policy = self.settled()
        policy.note_interest(0.34)
        self.assertEqual(policy.decide().reason, "nothing_to_say")
        policy.note_interest(0.35)
        self.assertTrue(policy.decide().allowed)

    def test_the_interest_expires(self):
        policy = self.settled(interest_window_s=600)
        policy.note_interest(0.9)
        self.advance(599)
        self.assertTrue(policy.decide().allowed)
        self.advance(2)
        self.assertEqual(policy.decide().reason, "nothing_to_say")

    def test_the_event_counts(self):
        policy = self.settled()
        self.emit(EventType.INTERESTING_CHANGE_DETECTED, total=0.7)
        self.assertTrue(policy.decide().allowed)

    def test_a_pending_question_is_something_to_say(self):
        policy = self.settled()
        self.assertEqual(policy.decide().reason, "nothing_to_say")
        self.questions = 1
        self.assertTrue(policy.decide().allowed)
        self.assertEqual(policy.decide(Context(pending_questions=0)).reason, "nothing_to_say")

    def test_a_failing_provider_is_safe(self):
        policy = self.settled()
        policy._pending_questions = lambda: 1 / 0
        self.assertEqual(policy.decide().reason, "nothing_to_say")
        policy._body_busy = lambda: 1 / 0
        self.assertEqual(policy.decide().reason, "busy")


class DeterminismTests(PolicyTestCase):
    def test_same_inputs_same_decision_and_decide_changes_nothing(self):
        policy = self.ready()
        first = [policy.decide() for _ in range(5)]
        self.assertEqual(set(first), {Decision(True, "allowed", A.AVAILABLE)})
        self.assertEqual(policy.recent_interactions(), [])

    def test_two_policies_with_the_same_history_agree(self):
        left, right = self.policy(), self.policy()
        for policy in (left, right):
            policy.note_interest(0.5)
        self.presence.arrive()
        self.advance(7)
        self.assertEqual(left.decide(), right.decide())


class ConfigTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "laika.toml"
            path.write_text(text, encoding="utf-8")
            return load_config(path).interaction

    def test_defaults(self):
        config = InteractionConfig()
        self.assertEqual((config.enabled, config.cooldown_s, config.max_per_hour), (True, 1800.0, 2))
        self.assertEqual(config.quiet_hours, "21:30-07:30")

    def test_loaded(self):
        config = self.load('[interaction]\ncooldown_s = 60\nquiet_hours = ""\nmax_per_hour = 5\n')
        self.assertEqual((config.cooldown_s, config.quiet_hours, config.max_per_hour), (60, "", 5))

    def test_invalid_values_name_the_setting(self):
        for text in (
            "enabled = 1", "cooldown_s = -1", "settle_s = 9999", "away_after_s = 1",
            "interest_window_s = 0", "max_per_hour = 0", "max_per_hour = 1.5", 'quiet_hours = "late"',
            "unknown = 1",
        ):
            with self.assertRaises(ValueError, msg=text) as raised:
                self.load(f"[interaction]\n{text}\n")
            self.assertIn("interaction", str(raised.exception))

    def test_the_example_file_loads(self):
        path = Path(__file__).resolve().parents[2] / "config" / "laika.example.toml"
        self.assertEqual(load_config(path).interaction.max_per_hour, 2)


class ToolTests(ToolTestCase):
    FORBIDDEN_WORDS = ("interact", "proactive", "cooldown", "conversation", "initiate", "talk")
    POLICY_TOOLS = {"set_do_not_disturb", "clear_do_not_disturb", "set_busy", "clear_busy"}

    def setUp(self):
        super().setUp()
        self.clock = FakeClock(1000.0)
        self.wall = FakeClock(at(14))
        self.presence = FakePresence(self.clock)
        self.policy = InteractionPolicy(
            InteractionConfig(), self.presence, clock=self.clock, wall_clock=self.wall
        )
        self.registry = ToolRegistry(
            self.robot,
            self.safety,
            world=self.world,
            actions=self.actions,
            speak=self.spoken.append,
            interaction=self.policy,
        )
        self.presence.arrive()
        self.clock.advance(10)
        self.policy.note_interest(0.9)

    def test_the_tools_exist_only_with_a_policy(self):
        self.assertTrue(self.POLICY_TOOLS <= set(self.registry.names()))
        bare = ToolRegistry(self.robot, self.safety)
        self.assertFalse(self.POLICY_TOOLS & set(bare.names()))

    def test_no_tool_can_start_an_interaction_or_change_the_cooldown(self):
        for spec in self.registry.specs():
            for word in self.FORBIDDEN_WORDS:
                self.assertNotIn(word, spec.name)
            if spec.name in self.POLICY_TOOLS:
                self.assertLessEqual(set(spec.parameters["properties"]), {"minutes"})

    def test_do_not_disturb_through_the_agent_makes_the_policy_stricter(self):
        self.assertTrue(self.policy.decide().allowed)
        result, _ = self.run_tool("set_do_not_disturb", minutes=60)
        self.assertTrue(result["ok"], result)
        self.assertIn("until", result)
        self.assertEqual(self.policy.decide(), Decision(False, "do_not_disturb", A.DO_NOT_DISTURB))

    def test_set_busy_and_clears(self):
        self.assertTrue(self.run_tool("set_busy", minutes=30)[0]["ok"])
        self.assertEqual(self.policy.decide().reason, "busy")
        self.assertEqual(self.run_tool("clear_busy")[0], {"ok": True, "was_set": True})
        self.assertTrue(self.policy.decide().allowed)
        self.run_tool("set_do_not_disturb", minutes=30)
        self.assertEqual(self.run_tool("clear_do_not_disturb")[0], {"ok": True, "was_set": True})

    def test_no_tool_call_turns_a_refusal_into_an_allowance(self):
        # Whatever the model calls, a refusal that is not the user's own setting stays a refusal.
        self.wall.now = at(23)
        self.assertEqual(self.policy.decide().reason, "sleeping")
        for name in sorted(self.POLICY_TOOLS):
            args = {"minutes": 10} if name.startswith("set_") else {}
            self.run_tool(name, **args)
            self.assertFalse(self.policy.decide().allowed, name)

    def test_out_of_range_minutes_are_rejected(self):
        for minutes in (0, 4, 721, 1.5):
            result, _ = self.run_tool("set_do_not_disturb", minutes=minutes)
            self.assertFalse(result["ok"])
        result, _ = self.run_tool("set_do_not_disturb", raw='{"minutes": "ten"}')
        self.assertFalse(result["ok"])
        self.assertIsNone(self.policy.do_not_disturb_until)
        self.assertTrue(self.policy.decide().allowed)

    def test_the_prompt_mentions_the_tools(self):
        gestures = available_actions(self.robot.identity().capabilities, True)
        self.assertIn("set_do_not_disturb", build_tool_system_prompt(gestures, 3, interaction=True))
        self.assertNotIn("set_do_not_disturb", build_tool_system_prompt(gestures, 3))


if __name__ == "__main__":
    unittest.main()
