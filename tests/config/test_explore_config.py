"""[explore] and [robot] allow_autonomous_driving: safe defaults, validated values."""

import tempfile
import unittest
from pathlib import Path

from laika.config import ExploreConfig, RobotConfig, load_config

EXAMPLE = Path(__file__).resolve().parents[2] / "config" / "laika.example.toml"


class ExploreConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def load(self, text):
        path = Path(self.tmp.name) / "laika.toml"
        path.write_text(text, encoding="utf-8")
        return load_config(path)

    def rejected(self, text):
        with self.assertRaises(ValueError) as caught:
            self.load(text)
        return str(caught.exception)

    def test_autonomous_driving_is_off_by_default_and_in_the_example_file(self):
        self.assertFalse(RobotConfig().allow_autonomous_driving)
        self.assertFalse(load_config(EXAMPLE).robot.allow_autonomous_driving)
        self.assertEqual(self.load("").robot.allow_autonomous_driving, False)

    def test_it_can_be_turned_on_explicitly(self):
        self.assertTrue(self.load("[robot]\nallow_autonomous_driving = true").robot.allow_autonomous_driving)

    def test_it_must_be_a_boolean(self):
        self.assertIn("allow_autonomous_driving", self.rejected('[robot]\nallow_autonomous_driving = "yes"'))

    def test_the_example_file_matches_the_defaults(self):
        self.assertEqual(load_config(EXAMPLE).explore, ExploreConfig())

    def test_values_are_loaded(self):
        config = self.load("[explore]\nspeed = 20\nstep_ms = 500\nidle_after_s = 120\nenabled = false")
        self.assertEqual(
            (config.explore.speed, config.explore.step_ms, config.explore.idle_after_s, config.explore.enabled),
            (20, 500, 120, False),
        )

    def test_unsafe_or_meaningless_values_are_rejected_naming_the_setting(self):
        for text, name in (
            ("speed = 0", "speed"),
            ("speed = 41", "speed"),
            ("speed = 10.5", "speed"),
            ("step_ms = 50", "step_ms"),
            ("step_ms = 5000", "step_ms"),
            ("scan_every_steps = 0", "scan_every_steps"),
            ("max_duration_s = 1", "max_duration_s"),
            ("max_steps = 0", "max_steps"),
            ("idle_after_s = 5", "idle_after_s"),
            ("idle_after_s = -1", "idle_after_s"),
            ('enabled = "yes"', "enabled"),
        ):
            with self.subTest(text):
                self.assertIn(name, self.rejected(f"[explore]\n{text}"))

    def test_unknown_settings_are_rejected(self):
        self.assertIn("Unknown setting", self.rejected("[explore]\nturbo = true"))


if __name__ == "__main__":
    unittest.main()
