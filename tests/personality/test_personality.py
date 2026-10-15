"""The personality (#41): config, prompt bands, policy and curiosity bounds, and the done-when check."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from laika.config import InteractionConfig, load_config
from laika.interaction_policy import Availability, InteractionPolicy, effective_tuning
from laika.personality import (
    CURIOSITY_LINES,
    HUMOR_LINES,
    SARCASM_LINES,
    VERBOSITY_LINES,
    Personality,
    band,
    curiosity_threshold,
    investigate_max_per_hour,
    persona,
    spoken_style,
)
from laika.prompts import build_system_prompt, build_tool_system_prompt, proactive_instruction
from tests.fakes import FakeClock

PARAMETERS = ("humor", "sarcasm", "verbosity", "initiative", "curiosity", "interruption_tolerance")
NEUTRAL = Personality(initiative=50, interruption_tolerance=50)


def load(text: str):
    """Load a settings file with `load_config`, as the app does at startup."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "laika.toml"
        path.write_text(text, encoding="utf-8")
        return load_config(path)


def at(hour: int, minute: int = 0) -> float:
    return datetime(2026, 10, 15, hour, minute).timestamp()


class FakePresence:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.present = False
        self.since: float | None = None
        self.last_seen: float | None = None

    def arrive(self) -> None:
        self.present, self.since, self.last_seen = True, self.clock(), self.clock()


class ConfigTests(unittest.TestCase):
    def test_no_section_gives_the_prd_example(self):
        personality = load("").personality
        self.assertEqual(personality, Personality())
        self.assertEqual(
            tuple(getattr(personality, name) for name in PARAMETERS), (70, 45, 20, 60, 75, 25)
        )

    def test_every_parameter_loads(self):
        personality = load(
            "[personality]\nhumor = 10\nsarcasm = 0\nverbosity = 100\ninitiative = 33\n"
            "curiosity = 1\ninterruption_tolerance = 99\n"
        ).personality
        self.assertEqual(
            tuple(getattr(personality, name) for name in PARAMETERS), (10, 0, 100, 33, 1, 99)
        )

    def test_a_partial_table_keeps_the_other_defaults(self):
        personality = load("[personality]\nhumor = 0\n").personality
        self.assertEqual((personality.humor, personality.sarcasm), (0, 45))

    def test_values_outside_0_to_100_are_refused_and_named(self):
        for value in ("101", "-1", '"high"', "50.5", "true"):
            with self.subTest(value=value), self.assertRaises(ValueError) as caught:
                load(f"[personality]\nhumor = {value}\n")
            self.assertIn("[personality] humor", str(caught.exception))

    def test_unknown_parameters_are_refused(self):
        with self.assertRaisesRegex(ValueError, "Unknown setting .personality. humour"):
            load("[personality]\nhumour = 50\n")

    def test_the_section_must_be_a_table(self):
        with self.assertRaisesRegex(ValueError, "must be a table"):
            load("personality = 5\n")

    def test_the_startup_line_names_every_parameter(self):
        self.assertEqual(
            Personality().describe(),
            "humor 70, sarcasm 45, verbosity 20, initiative 60, curiosity 75, interruption tolerance 25",
        )


class BandTests(unittest.TestCase):
    def test_band_edges(self):
        self.assertEqual([band(v) for v in (0, 30, 31, 69, 70, 100)], [0, 0, 1, 1, 2, 2])

    def test_each_prompt_parameter_changes_the_text_per_band(self):
        for name in ("humor", "sarcasm", "verbosity", "curiosity"):
            with self.subTest(parameter=name):
                texts = {persona(Personality(**{name: value})) for value in (0, 50, 90)}
                self.assertEqual(len(texts), 3)

    def test_initiative_and_tolerance_do_not_change_the_prompt(self):
        self.assertEqual(
            persona(Personality(initiative=0, interruption_tolerance=0)),
            persona(Personality(initiative=100, interruption_tolerance=100)),
        )

    def test_the_core_character_stays_fixed(self):
        for personality in (Personality(), Personality(humor=0), Personality(curiosity=100)):
            with self.subTest(personality=personality):
                text = persona(personality)
                self.assertIn("TARS", text)
                self.assertIn("Se non sai una cosa, dillo.", text)


class PromptTests(unittest.TestCase):
    EXTREMES = (
        Personality(),
        Personality(*(0,) * 6),
        Personality(*(100,) * 6),
        Personality(humor=0, sarcasm=100, verbosity=0, curiosity=100),
    )

    def test_every_prompt_keeps_the_child_safety_and_italian_rules(self):
        for personality in self.EXTREMES:
            prompts = {
                "json": build_system_prompt({}, 3, personality),
                "tools": build_tool_system_prompt({}, 3, personality=personality),
                "proactive": proactive_instruction("Vuoi giocare?", personality),
            }
            for kind, text in prompts.items():
                with self.subTest(kind=kind, personality=personality):
                    self.assertIn("mai cattiva", text)
                    self.assertIn("soprattutto con i bambini", text)
                    self.assertIn("italiano", text.lower())

    def test_json_prompt_follows_the_bands(self):
        self.assertIn(VERBOSITY_LINES[0], build_system_prompt({}, 3, Personality(verbosity=0)))
        self.assertIn(VERBOSITY_LINES[2], build_system_prompt({}, 3, Personality(verbosity=100)))
        self.assertIn(HUMOR_LINES[0], build_system_prompt({}, 3, Personality(humor=0)))
        self.assertNotIn(HUMOR_LINES[0], build_system_prompt({}, 3, Personality(humor=100)))

    def test_tool_prompt_follows_the_bands(self):
        self.assertIn(SARCASM_LINES[0], build_tool_system_prompt({}, 3, personality=Personality(sarcasm=0)))
        self.assertIn(SARCASM_LINES[2], build_tool_system_prompt({}, 3, personality=Personality(sarcasm=100)))
        self.assertIn(CURIOSITY_LINES[0], build_tool_system_prompt({}, 3, personality=Personality(curiosity=0)))
        self.assertIn(CURIOSITY_LINES[2], build_tool_system_prompt({}, 3, personality=Personality(curiosity=100)))

    def test_proactive_prompt_follows_the_bands_and_keeps_the_question(self):
        text = proactive_instruction("Che cos'è quella cosa?", Personality(humor=100, verbosity=0))
        self.assertIn(HUMOR_LINES[2], text)
        self.assertIn(VERBOSITY_LINES[0], text)
        self.assertIn("Che cos'è quella cosa?", text)
        self.assertNotIn(HUMOR_LINES[2], spoken_style(Personality(humor=0)))

    def test_without_a_personality_the_defaults_are_used(self):
        self.assertEqual(build_system_prompt({}, 3), build_system_prompt({}, 3, Personality()))
        self.assertEqual(
            build_tool_system_prompt({}, 3), build_tool_system_prompt({}, 3, personality=Personality())
        )


class TuningTests(unittest.TestCase):
    def test_the_defaults_scale_the_configured_numbers(self):
        tuning = effective_tuning(InteractionConfig(), Personality())
        self.assertAlmostEqual(tuning.cooldown_s, 1800 * 0.9)  # initiative 60
        self.assertEqual(tuning.max_per_hour, 2)
        self.assertAlmostEqual(tuning.settle_s, 5 * 1.5)  # tolerance 25
        self.assertEqual(tuning.min_present_s, 0)

    def test_initiative_scales_the_cooldown_both_ways(self):
        def cooldown(initiative):
            return effective_tuning(InteractionConfig(cooldown_s=1800), Personality(initiative=initiative)).cooldown_s

        self.assertAlmostEqual(cooldown(100), 900)
        self.assertAlmostEqual(cooldown(50), 1800)
        self.assertAlmostEqual(cooldown(0), 3600)

    def test_the_cooldown_never_drops_below_ten_minutes(self):
        tuning = effective_tuning(InteractionConfig(cooldown_s=1000), Personality(initiative=100))
        self.assertEqual(tuning.cooldown_s, 600)
        self.assertEqual(effective_tuning(InteractionConfig(cooldown_s=0), Personality(initiative=100)).cooldown_s, 0)

    def test_initiative_picks_one_to_four_attempts_within_the_configured_ceiling(self):
        def per_hour(initiative, ceiling=60):
            config = InteractionConfig(max_per_hour=ceiling)
            return effective_tuning(config, Personality(initiative=initiative)).max_per_hour

        self.assertEqual([per_hour(i) for i in (0, 33, 67, 100)], [1, 2, 3, 4])
        self.assertEqual(per_hour(100, ceiling=2), 2)  # the configured value is a ceiling

    def test_tolerance_scales_the_settle_time_both_ways(self):
        def settle(tolerance):
            return effective_tuning(InteractionConfig(settle_s=5), Personality(interruption_tolerance=tolerance)).settle_s

        self.assertAlmostEqual(settle(0), 10)
        self.assertAlmostEqual(settle(50), 5)
        self.assertAlmostEqual(settle(100), 2.5)

    def test_low_tolerance_needs_thirty_seconds_of_presence(self):
        low = effective_tuning(InteractionConfig(settle_s=5), Personality(interruption_tolerance=19))
        high = effective_tuning(InteractionConfig(settle_s=5), Personality(interruption_tolerance=20))
        self.assertEqual(low.min_present_s, 30)
        self.assertEqual(low.present_s, 30)
        self.assertEqual(high.min_present_s, 0)

    def test_the_bounds_hold_at_zero_and_one_hundred(self):
        config = InteractionConfig(cooldown_s=1800, max_per_hour=60, settle_s=5)
        for initiative in (0, 100):
            for tolerance in (0, 100):
                tuning = effective_tuning(config, Personality(initiative=initiative, interruption_tolerance=tolerance))
                self.assertTrue(600 <= tuning.cooldown_s <= 3600, tuning)
                self.assertTrue(1 <= tuning.max_per_hour <= 4, tuning)
                self.assertTrue(2.5 <= tuning.settle_s <= 10, tuning)


class PolicyCase(unittest.TestCase):
    """A policy with a fake clock, a fake presence and a fake body, so decisions are exact."""

    def setUp(self) -> None:
        self.clock = FakeClock(1000.0)
        self.wall = FakeClock(at(14))
        self.presence = FakePresence(self.clock)
        self.busy = False

    def policy(self, personality: Personality, **config) -> InteractionPolicy:
        return InteractionPolicy(
            InteractionConfig(**config),
            self.presence,
            threshold=0.35,
            pending_questions=lambda: 0,
            body_busy=lambda: self.busy,
            clock=self.clock,
            wall_clock=self.wall,
            personality=personality,
        )

    def advance(self, seconds: float) -> None:
        self.clock.advance(seconds)
        self.wall.advance(seconds)


class PolicyDecisionTests(PolicyCase):
    def test_full_initiative_still_refuses_during_do_not_disturb(self):
        policy = self.policy(Personality(initiative=100, interruption_tolerance=100))
        self.presence.arrive()
        self.advance(40)
        policy.note_interest(0.9)
        self.assertTrue(policy.decide().allowed)
        policy.set_do_not_disturb(30)
        self.assertEqual(policy.decide().reason, "do_not_disturb")

    def test_full_initiative_still_refuses_in_quiet_hours_and_when_busy(self):
        policy = self.policy(Personality(initiative=100, interruption_tolerance=100))
        self.presence.arrive()
        self.advance(40)
        policy.note_interest(0.9)
        self.wall.now = at(23)
        self.assertEqual(policy.decide().reason, "sleeping")
        self.wall.now = at(14)
        self.busy = True
        self.assertEqual(policy.decide().reason, "busy")

    def test_low_tolerance_waits_thirty_seconds_before_speaking(self):
        policy = self.policy(Personality(interruption_tolerance=10), settle_s=5)
        policy.note_interest(0.9)
        self.presence.arrive()
        self.advance(20)
        self.assertEqual(policy.state(), Availability.UNKNOWN)
        self.advance(11)
        self.assertEqual(policy.state(), Availability.AVAILABLE)

    def test_high_tolerance_is_quick_to_address_someone(self):
        policy = self.policy(Personality(interruption_tolerance=100), settle_s=5)
        policy.note_interest(0.9)
        self.presence.arrive()
        self.advance(3)
        self.assertTrue(policy.decide().allowed)

    def test_the_hourly_limit_follows_initiative(self):
        policy = self.policy(Personality(initiative=100), cooldown_s=0, max_per_hour=4)
        self.presence.arrive()
        self.advance(40)
        policy.note_interest(0.9)
        for _ in range(3):
            policy.record_attempt()
        self.assertTrue(policy.decide().allowed)
        policy.record_attempt()
        self.assertEqual(policy.decide().reason, "rate_limit")


class CuriosityShiftTests(unittest.TestCase):
    def test_the_threshold_moves_with_curiosity(self):
        self.assertAlmostEqual(curiosity_threshold(0.35, 0), 0.455)
        self.assertAlmostEqual(curiosity_threshold(0.35, 75), 0.2975)
        self.assertAlmostEqual(curiosity_threshold(0.35, 100), 0.245)

    def test_the_threshold_stays_between_0_2_and_0_6(self):
        for base in (0.0, 0.1, 0.35, 0.8, 1.0):
            for curiosity in (0, 50, 100):
                with self.subTest(base=base, curiosity=curiosity):
                    self.assertTrue(0.2 <= curiosity_threshold(base, curiosity) <= 0.6)

    def test_the_investigator_limit_scales_between_half_and_one_and_a_half(self):
        self.assertEqual([investigate_max_per_hour(12, c) for c in (0, 50, 100)], [6, 12, 18])
        self.assertEqual(investigate_max_per_hour(1, 0), 1)  # never below one


class DoneWhenTests(PolicyCase):
    """Two settings files, the same code: different prompts and different decisions."""

    CALM = (
        "[personality]\nhumor = 0\nsarcasm = 0\nverbosity = 0\ninitiative = 0\n"
        "curiosity = 0\ninterruption_tolerance = 0\n"
        "[interaction]\ncooldown_s = 1800\nsettle_s = 5\n"
    )
    BOLD = (
        "[personality]\nhumor = 100\nsarcasm = 100\nverbosity = 100\ninitiative = 100\n"
        "curiosity = 100\ninterruption_tolerance = 100\n"
        "[interaction]\ncooldown_s = 1800\nsettle_s = 5\n"
    )

    def test_two_files_give_two_different_prompts(self):
        calm, bold = load(self.CALM), load(self.BOLD)
        self.assertNotEqual(
            build_tool_system_prompt({}, 3, personality=calm.personality),
            build_tool_system_prompt({}, 3, personality=bold.personality),
        )
        self.assertNotEqual(
            build_system_prompt({}, 3, calm.personality), build_system_prompt({}, 3, bold.personality)
        )
        self.assertNotEqual(
            proactive_instruction("Q", calm.personality), proactive_instruction("Q", bold.personality)
        )

    def test_two_files_give_two_different_decisions_in_the_same_situation(self):
        # The same situation: someone has been near for a minute, something interesting was seen, and
        # Laika started a conversation 1000 seconds ago. Calm waits an hour; bold is allowed again.
        decisions = {}
        for name, text in (("calm", self.CALM), ("bold", self.BOLD)):
            settings = load(text)
            policy = self.policy(settings.personality, **vars(settings.interaction))
            self.presence.arrive()
            self.advance(60)
            policy.note_interest(0.9)
            policy.record_attempt()
            self.advance(1000)
            policy.note_interest(0.9)
            decisions[name] = policy.decide()
        self.assertEqual(decisions["calm"].reason, "cooldown")
        self.assertTrue(decisions["bold"].allowed)


if __name__ == "__main__":
    unittest.main()
