"""[curiosity]: safe defaults, validated values, and the example file matches the defaults."""

import tempfile
import unittest
from pathlib import Path

from laika.config import CURIOSITY_INPUTS, CuriosityConfig, load_config, validate_curiosity

EXAMPLE = Path(__file__).resolve().parents[2] / "config" / "laika.example.toml"


class CuriosityConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def load(self, text):
        path = Path(self.tmp.name) / "laika.toml"
        path.write_text(text, encoding="utf-8")
        return load_config(path).curiosity

    def rejected(self, text):
        with self.assertRaises(ValueError) as caught:
            self.load(text)
        return str(caught.exception)

    def test_defaults_are_valid_and_weights_sum_to_one(self):
        validate_curiosity(CuriosityConfig())
        defaults = CuriosityConfig()
        self.assertAlmostEqual(sum(getattr(defaults, name) for name in CURIOSITY_INPUTS), 1.0)
        self.assertEqual(defaults.threshold, 0.35)
        self.assertEqual(defaults.sound, 0.0)

    def test_the_example_file_matches_the_defaults(self):
        self.assertEqual(load_config(EXAMPLE).curiosity, CuriosityConfig())

    def test_a_section_overrides_values(self):
        config = self.load("[curiosity]\nnovelty = 0.5\nthreshold = 0.6\ncooldown_s = 0\n")
        self.assertEqual((config.novelty, config.threshold, config.cooldown_s), (0.5, 0.6, 0))

    def test_unknown_keys_are_rejected(self):
        self.assertIn("[curiosity] boredom", self.rejected("[curiosity]\nboredom = 1\n"))

    def test_weights_must_be_numbers_in_range(self):
        self.assertIn("[curiosity] novelty", self.rejected('[curiosity]\nnovelty = "high"\n'))
        self.assertIn("[curiosity] novelty", self.rejected("[curiosity]\nnovelty = -0.1\n"))
        self.assertIn("[curiosity] person", self.rejected("[curiosity]\nperson = 11\n"))
        self.assertIn("[curiosity] person", self.rejected("[curiosity]\nperson = true\n"))

    def test_at_least_one_weight_must_be_positive(self):
        text = "[curiosity]\n" + "".join(f"{name} = 0\n" for name in CURIOSITY_INPUTS)
        self.assertIn("at least one input weight", self.rejected(text))

    def test_threshold_and_unknown_confidence_are_fractions(self):
        self.assertIn("threshold", self.rejected("[curiosity]\nthreshold = 1.5\n"))
        self.assertIn(
            "unknown_object_confidence",
            self.rejected("[curiosity]\nunknown_object_confidence = -1\n"),
        )

    def test_cooldown_is_a_non_negative_number_of_seconds(self):
        self.assertIn("cooldown_s", self.rejected("[curiosity]\ncooldown_s = -5\n"))
        self.assertIn("cooldown_s", self.rejected("[curiosity]\ncooldown_s = 100000\n"))
