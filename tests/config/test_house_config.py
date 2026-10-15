"""[house] and the [robot] driving budgets (#51): safe values load, unusable ones fail with the setting named."""

import tempfile
import unittest
from pathlib import Path

from laika.config import Config, HouseConfig, RobotConfig, load_config, validate_robot

EXAMPLE = Path(__file__).resolve().parents[2] / "config" / "laika.example.toml"


class HouseConfigTests(unittest.TestCase):
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

    def test_defaults_match_the_example_file(self):
        example = load_config(EXAMPLE)
        self.assertEqual(example.house, HouseConfig())
        self.assertEqual(example.robot, RobotConfig())
        self.assertEqual(Config().house.no_go_rooms, [])

    def test_no_go_rooms_load(self):
        config = self.load('[house]\nno_go_rooms = ["camera da letto", "bagno"]\n')
        self.assertEqual(config.house.no_go_rooms, ["camera da letto", "bagno"])

    def test_bad_no_go_rooms_are_rejected(self):
        for text in ('no_go_rooms = "bagno"', "no_go_rooms = [1]", 'no_go_rooms = [""]', 'no_go_rooms = ["' + "a" * 61 + '"]'):
            self.assertIn("[house] no_go_rooms", self.rejected(f"[house]\n{text}\n"), text)
        too_many = ", ".join(f'"stanza {n}"' for n in range(21))
        self.assertIn("[house] no_go_rooms", self.rejected(f"[house]\nno_go_rooms = [{too_many}]\n"))

    def test_unknown_house_settings_are_rejected(self):
        self.assertIn("[house] colour", self.rejected("[house]\ncolour = 1\n"))

    def test_budgets_default_to_a_documented_limit(self):
        robot = RobotConfig()
        self.assertEqual(robot.max_autonomous_drive_s_per_hour, 120)
        self.assertEqual(robot.max_autonomous_drive_s_per_day, 900)

    def test_budgets_load_and_zero_means_no_limit(self):
        robot = self.load("[robot]\nmax_autonomous_drive_s_per_hour = 60\nmax_autonomous_drive_s_per_day = 300\n").robot
        self.assertEqual((robot.max_autonomous_drive_s_per_hour, robot.max_autonomous_drive_s_per_day), (60, 300))
        robot = self.load("[robot]\nmax_autonomous_drive_s_per_hour = 0\nmax_autonomous_drive_s_per_day = 0\n").robot
        validate_robot(robot)

    def test_bad_budgets_are_rejected(self):
        for text in (
            "max_autonomous_drive_s_per_hour = 5",
            "max_autonomous_drive_s_per_hour = 5000",
            "max_autonomous_drive_s_per_day = 100000",
            "max_autonomous_drive_s_per_day = true",
            'max_autonomous_drive_s_per_hour = "120"',
        ):
            self.assertIn("max_autonomous_drive_s_per", self.rejected(f"[robot]\n{text}\n"), text)

    def test_an_hour_above_the_day_is_rejected(self):
        message = self.rejected("[robot]\nmax_autonomous_drive_s_per_hour = 600\nmax_autonomous_drive_s_per_day = 300\n")
        self.assertIn("must not be more than", message)


if __name__ == "__main__":
    unittest.main()
