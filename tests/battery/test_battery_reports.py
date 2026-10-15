"""The battery in the hardware check (multimeter comparison) and in the field-trial report (#50)."""

import tempfile
import unittest

from laika import trial_report
from laika.events import EventType
from laika.house import default_house
from laika.robot import ALL_CAPABILITIES, BATTERY
from laika.simulator import BATTERY_START_V, SimulatedRobot
from tests.hardware.test_hardware_check import FAIL, PASS, SKIP, Scripted, SimStaging, StepRecord, make_sim_check
from tests.trial.synthetic import at, line, write_logs
from tests.trial.test_trial_report import Case

MULTIMETER = "multimeter"


class StepCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def step(self, robot=None, answers=None, **kwargs):
        robot = robot or SimulatedRobot(house=default_house(), verbose=False)
        check, safety, self.output = make_sim_check(robot, self.tmp.name, answers=answers, **kwargs)
        self.addCleanup(safety.close)
        record = StepRecord("battery", "Battery voltage")
        check.step_battery(record)
        record.ran = True
        return record


class HardwareCheckBatteryTests(StepCase):
    def test_it_shows_the_voltage_and_records_it_with_the_multimeter_value(self):
        record = self.step(answers=Scripted({MULTIMETER: "8.05"}))
        self.assertEqual(record.status, PASS)
        self.assertEqual(record.data["reading_v"], BATTERY_START_V)
        self.assertEqual(record.data["multimeter_v"], 8.05)
        self.assertAlmostEqual(record.data["difference_v"], -0.15)
        self.assertAlmostEqual(record.data["suggested_voltage_offset"], -0.15)
        self.assertTrue(any("8.20 V" in line for line in self.output))
        self.assertTrue(any("voltage_offset = -0.15" in line for line in self.output))

    def test_a_comma_is_a_fine_decimal_mark(self):
        record = self.step(answers=Scripted({MULTIMETER: "8,35"}))
        self.assertAlmostEqual(record.data["suggested_voltage_offset"], 0.15)

    def test_without_a_multimeter_value_the_reading_is_only_shown(self):
        record = self.step(answers=Scripted({MULTIMETER: "s"}))
        self.assertEqual(record.status, PASS)
        self.assertNotIn("suggested_voltage_offset", record.data)
        self.assertTrue(any("not compared" in note for note in record.notes))

    def test_the_configured_offset_is_shown_too(self):
        self.step(answers=Scripted({MULTIMETER: "s"}), battery_offset=0.1)
        self.assertTrue(any("8.30 V with the configured voltage_offset" in line for line in self.output))

    def test_a_reading_far_from_the_multimeter_fails(self):
        record = self.step(answers=Scripted({MULTIMETER: "7.2"}))
        self.assertEqual(record.status, FAIL)
        self.assertIn("away from the multimeter", record.detail)

    def test_a_reading_that_cannot_be_taken_fails(self):
        robot = SimulatedRobot(house=default_house(), verbose=False)
        robot.faults.battery_error = True
        record = self.step(robot)
        self.assertEqual(record.status, FAIL)
        self.assertIn("could not be read", record.detail)

    def test_a_robot_without_the_capability_skips_it(self):
        robot = SimulatedRobot(verbose=False, capabilities=ALL_CAPABILITIES - {BATTERY})
        from laika.hardware_check import Skip

        with self.assertRaises(Skip):
            self.step(robot)

    def test_yes_mode_never_asks_for_a_number(self):
        from laika.hardware_check import Operator

        asked = []
        operator = Operator(lambda prompt: asked.append(prompt) or "y", lambda text: None, assume_yes=True)
        self.assertIsNone(operator.ask_number("volts?"))
        self.assertEqual(asked, [])

    def test_the_report_suggests_the_offset_under_battery(self):
        robot = SimulatedRobot(house=default_house(), verbose=False)
        check, safety, _ = make_sim_check(
            robot, self.tmp.name, staging=SimStaging(robot), answers=Scripted({MULTIMETER: "8.05"})
        )
        self.addCleanup(safety.close)
        report = check.run()
        steps = {s.name: s for s in report.steps}
        self.assertEqual(steps["battery"].status, PASS)
        self.assertEqual(report.to_dict()["suggestedSettings"]["voltage_offset"], -0.15)
        self.assertEqual(report.battery_suggestion_lines(), ["voltage_offset = -0.15"])
        markdown = report.to_markdown()
        self.assertIn("Paste under `[battery]`", markdown)
        self.assertIn("voltage_offset = -0.15", markdown)
        self.assertNotIn("voltage_offset", "\n".join(report.suggestion_lines()))


class TrialReportBatteryTests(Case):
    def setUp(self):
        super().setUp()
        write_logs(self.dir)
        with (self.dir / "logs" / "events.jsonl").open("a", encoding="utf-8") as file:
            file.write("\n")
            for kind, hour, voltage, lowest in (
                (EventType.BATTERY_LOW, 15, 6.95, 6.95),
                (EventType.BATTERY_CRITICAL, 16, 6.55, 6.55),
                (EventType.BATTERY_OK, 20, 7.4, 6.5),
            ):
                file.write(line(kind, at(1, hour), voltage=voltage, percent=20, previous="ok", lowestV=lowest) + "\n")
        self.r = self.report()

    def test_the_transitions_and_the_lowest_voltage(self):
        battery = self.r["battery"]
        self.assertEqual((battery["low"], battery["critical"], battery["ok"]), (1, 1, 1))
        self.assertEqual([t["state"] for t in battery["transitions"]], ["low", "critical", "ok"])
        self.assertEqual(battery["transitions"][0]["voltage"], 6.95)
        self.assertEqual(battery["lowest_v"], 6.5)

    def test_the_markdown_and_the_summary_say_it(self):
        markdown = trial_report.render_markdown(self.r)
        self.assertIn("## Battery", markdown)
        self.assertIn("6.50 V", markdown)
        self.assertIn("1 low, 1 critical, 1 back to OK", markdown)
        self.assertIn("battery: 1 low", trial_report.summary_text(self.r))

    def test_battery_events_are_known_event_types(self):
        self.assertNotIn("BatteryLow", self.r["sources"]["unknown_event_types"])

    def test_a_period_without_battery_events(self):
        empty = trial_report.build_battery([])
        self.assertEqual(empty["transitions"], [])
        self.assertIsNone(empty["lowest_v"])
        self.assertIn("not known", trial_report._battery_summary(empty))

    def test_a_refused_move_is_counted_by_its_reason(self):
        with (self.dir / "logs" / "events.jsonl").open("a", encoding="utf-8") as file:
            file.write("\n" + line(EventType.MOVEMENT_FAILED, at(1, 15, 5), result="rejected", reason="battery_low") + "\n")
        acts = self.report()["core_loop"]["acts"]
        self.assertEqual(acts["failed_by_reason"]["battery_low"], 1)


if __name__ == "__main__":
    unittest.main()
