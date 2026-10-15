"""[investigate]: safe defaults, validated values, and the example file matches the defaults."""

import tempfile
import unittest
from pathlib import Path

from laika.config import InvestigateConfig, load_config, validate_investigate

EXAMPLE = Path(__file__).resolve().parents[2] / "config" / "laika.example.toml"


class InvestigateConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def load(self, text):
        path = Path(self.tmp.name) / "laika.toml"
        path.write_text(text, encoding="utf-8")
        return load_config(path).investigate

    def rejected(self, text):
        with self.assertRaises(ValueError) as caught:
            self.load(text)
        return str(caught.exception)

    def test_defaults_are_valid_and_cautious(self):
        defaults = InvestigateConfig()
        validate_investigate(defaults)
        self.assertEqual(defaults.max_approach_moves, 1)
        self.assertEqual(defaults.approach_stop_cm, 50)

    def test_the_example_file_matches_the_defaults(self):
        self.assertEqual(load_config(EXAMPLE).investigate, InvestigateConfig())

    def test_a_section_overrides_values(self):
        config = self.load("[investigate]\nenabled = false\nmax_approach_moves = 0\nmax_per_hour = 3\n")
        self.assertEqual((config.enabled, config.max_approach_moves, config.max_per_hour), (False, 0, 3))

    def test_unknown_keys_and_bad_values_are_rejected(self):
        self.assertIn("[investigate] mood", self.rejected("[investigate]\nmood = 1\n"))
        self.assertIn("enabled", self.rejected('[investigate]\nenabled = "yes"\n'))
        self.assertIn("max_per_hour", self.rejected("[investigate]\nmax_per_hour = 0\n"))
        self.assertIn("max_per_hour", self.rejected("[investigate]\nmax_per_hour = 2.5\n"))
        self.assertIn("object_cooldown_s", self.rejected("[investigate]\nobject_cooldown_s = -1\n"))
        self.assertIn("approach_stop_cm", self.rejected("[investigate]\napproach_stop_cm = 5\n"))
        self.assertIn("max_approach_moves", self.rejected("[investigate]\nmax_approach_moves = 3\n"))
        self.assertIn("question_ttl_h", self.rejected("[investigate]\nquestion_ttl_h = 0\n"))
