"""[robot] safety settings: safe values load, unsafe ones fail at startup with the setting named."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from laika import app
from laika.config import RobotConfig, load_config, validate_robot

EXAMPLE = Path(__file__).resolve().parents[2] / "config" / "laika.example.toml"


class RobotConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def load(self, text):
        path = Path(self.tmp.name) / "laika.toml"
        path.write_text(f"[robot]\n{text}\n", encoding="utf-8")
        return load_config(path).robot

    def rejected(self, text):
        with self.assertRaises(ValueError) as caught:
            self.load(text)
        return str(caught.exception)

    def test_defaults_are_valid_and_match_the_example_file(self):
        validate_robot(RobotConfig())
        example = load_config(EXAMPLE).robot
        self.assertEqual(example, RobotConfig())

    def test_defaults_match_the_documented_values(self):
        robot = RobotConfig()
        self.assertEqual(robot.drive_speed, 25)
        self.assertEqual(robot.max_drive_s, 1.5)
        self.assertEqual(robot.backward_max_s, 0.6)
        self.assertEqual(robot.min_front_distance_cm, 25.0)
        self.assertEqual(robot.command_max_age_s, 1.0)
        self.assertEqual(robot.safety_cadence_hz, 20.0)
        self.assertEqual(robot.ultrasonic_invalid_limit, 5)
        self.assertEqual(robot.watchdog_timeout_s, 0.5)
        self.assertIsNone(robot.cliff_reference)

    def test_good_values_at_the_edges_are_accepted(self):
        robot = self.load(
            "drive_speed = 40\nmax_drive_s = 0.1\nbackward_max_s = 1\n"
            "min_front_distance_cm = 100\ncommand_max_age_s = 5\nsafety_cadence_hz = 50\n"
            "ultrasonic_invalid_limit = 50\nwatchdog_timeout_s = 2\n"
            "cliff_reference = [0, 4095, 400]\n"
        )
        self.assertEqual(robot.drive_speed, 40)
        self.assertEqual(robot.cliff_reference, [0, 4095, 400])

    def test_zero_turns_the_watchdog_and_the_invalid_reading_rule_off(self):
        robot = self.load("watchdog_timeout_s = 0\nultrasonic_invalid_limit = 0\n")
        self.assertEqual((robot.watchdog_timeout_s, robot.ultrasonic_invalid_limit), (0, 0))

    def test_values_outside_the_safe_range_are_rejected_with_the_setting_named(self):
        cases = {
            "drive_speed = 41": "drive_speed must be 1 to 40, got 41",
            "drive_speed = 0": "drive_speed must be 1 to 40, got 0",
            "max_drive_s = 5": "max_drive_s must be 0.1 to 3, got 5",
            "backward_max_s = 2": "backward_max_s must be 0.1 to 1, got 2",
            "min_front_distance_cm = 5": "min_front_distance_cm must be 10 to 100, got 5",
            "command_max_age_s = 0.1": "command_max_age_s must be 0.2 to 5, got 0.1",
            "safety_cadence_hz = 5": "safety_cadence_hz must be 10 to 50, got 5",
            "ultrasonic_invalid_limit = 51": "ultrasonic_invalid_limit must be 1 to 50 (or 0 to turn it off), got 51",
            "watchdog_timeout_s = 0.1": "watchdog_timeout_s must be 0.2 to 2 (or 0 to turn it off), got 0.1",
            "watchdog_timeout_s = 3": "watchdog_timeout_s must be 0.2 to 2",
        }
        for line, message in cases.items():
            with self.subTest(line=line):
                self.assertIn(message, self.rejected(line))

    def test_wrong_types_are_rejected(self):
        self.assertIn("drive_speed must be a whole number", self.rejected("drive_speed = 25.5"))
        self.assertIn("must be a number", self.rejected('max_drive_s = "fast"'))
        self.assertIn("must be a number", self.rejected("max_drive_s = true"))
        self.assertIn("safety_cadence_hz must be 10 to 50", self.rejected("safety_cadence_hz = nan"))

    def test_cliff_reference_needs_three_grayscale_values(self):
        self.assertIn("three numbers", self.rejected("cliff_reference = [400, 400]"))
        self.assertIn("three numbers", self.rejected("cliff_reference = 400"))
        self.assertIn("cliff_reference must be 0 to 4095, got 5000", self.rejected("cliff_reference = [5000, 1, 1]"))
        self.assertIn("cliff_reference must be 0 to 4095, got -1", self.rejected("cliff_reference = [400, -1, 400]"))

    def test_unknown_robot_key_is_still_rejected(self):
        self.assertIn("Unknown setting [robot] bogus", self.rejected("bogus = 1"))

    def test_app_reports_a_settings_error_and_exits_without_starting(self):
        path = Path(self.tmp.name) / "bad.toml"
        path.write_text("[robot]\ndrive_speed = 90\n", encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = app.main(["--config", str(path), "--sim"])
        self.assertEqual(code, 2)
        self.assertIn("settings error", output.getvalue())
        self.assertIn("drive_speed must be 1 to 40, got 90", output.getvalue())


if __name__ == "__main__":
    unittest.main()
