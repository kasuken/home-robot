"""[outside] (#48): the example file matches the defaults, unusable values fail with the setting named."""

import tempfile
import unittest
from pathlib import Path

from laika.config import Config, OutsideConfig, load_config

EXAMPLE = Path(__file__).resolve().parents[2] / "config" / "laika.example.toml"


class OutsideConfigTests(unittest.TestCase):
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

    def test_the_example_file_matches_the_defaults(self):
        self.assertEqual(load_config(EXAMPLE).outside, OutsideConfig())
        self.assertEqual(Config().outside, OutsideConfig(True, 1800.0, 24, 0.6))

    def test_values_load(self):
        config = self.load("[outside]\nenabled = false\nmin_interval_s = 600\nmax_per_day = 6\nchange_interest = 0\n")
        self.assertEqual(config.outside, OutsideConfig(False, 600, 6, 0))

    def test_unusable_values_are_rejected(self):
        for line, name in (
            ('enabled = "yes"', "enabled"),
            ("min_interval_s = 5", "min_interval_s"),
            ("min_interval_s = 100000", "min_interval_s"),
            ("max_per_day = 0", "max_per_day"),
            ("max_per_day = 2.5", "max_per_day"),
            ("max_per_day = 500", "max_per_day"),
            ("change_interest = 1.5", "change_interest"),
            ("change_interest = -0.1", "change_interest"),
            ('change_interest = "high"', "change_interest"),
        ):
            self.assertIn(f"[outside] {name}", self.rejected(f"[outside]\n{line}\n"), line)

    def test_unknown_settings_are_rejected(self):
        self.assertIn("[outside] colour", self.rejected("[outside]\ncolour = 1\n"))


if __name__ == "__main__":
    unittest.main()
