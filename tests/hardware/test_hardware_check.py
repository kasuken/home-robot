"""The guided hardware check, run against the simulator and against PiCarXRobot with faked SunFounder modules."""

import argparse
import json
import math
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from laika import app
from laika.camera import Camera
from laika.config import Config
from laika.hardware_check import (
    FAIL,
    PASS,
    SKIP,
    SPEECH_TEST,
    HardwareCheck,
    StepRecord,
    Operator,
    SimStaging,
    Staging,
    run_from_args,
    suggest_cliff_reference,
)
from laika.house import Thing, default_house
from laika.safety import SafetyController
from laika.simulator import ROBOT_RADIUS_CM, SimulatedRobot
from laika.storage import ImageStore
from tests.simulator.test_robot_contract import SpyMouth, fake_sunfounder_modules

STEP_NAMES = [
    "steering",
    "head",
    "ultrasonic",
    "grayscale",
    "drive",
    "obstacle",
    "emergency_stop",
    "camera",
    "speaker",
    "battery",
    "microphone",
]


class Scripted:
    """An input function: answers "y" unless a substring of the prompt has an override."""

    def __init__(self, overrides=None, default="y", fail_with=None):
        self.overrides = overrides or {}
        self.default = default
        self.fail_with = fail_with or {}
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        for text, error in self.fail_with.items():
            if text in prompt:
                raise error
        for text, answer in self.overrides.items():
            if text in prompt:
                return answer
        return self.default


def namespace(**overrides):
    values = dict(sim=True, no_driving=False, yes=False, no_mic=True, check_hardware=True)
    values.update(overrides)
    return argparse.Namespace(**values)


class RunFromArgsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.dict(os.environ, {"LAIKA_DATA_DIR": self.tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.output = []

    def run_check(self, input_fn=None, args=None, config=None):
        code = run_from_args(
            args or namespace(),
            config or Config(),
            input_fn=input_fn or Scripted(),
            output=self.output.append,
            sleep=lambda seconds: None,
        )
        reports = sorted((Path(self.tmp.name) / "logs").glob("hardware-check-*.json"))
        report = json.loads(reports[-1].read_text(encoding="utf-8")) if reports else None
        return code, report

    def steps(self, report):
        return {step["name"]: step for step in report["steps"]}

    def test_all_yes_passes_and_writes_both_reports(self):
        code, report = self.run_check()
        self.assertEqual(code, 0)
        self.assertEqual(report["result"], PASS)
        self.assertEqual([s["name"] for s in report["steps"]], STEP_NAMES)
        steps = self.steps(report)
        for name in STEP_NAMES[:-1]:
            self.assertEqual(steps[name]["status"], PASS, name)
        self.assertEqual(steps["microphone"]["status"], SKIP)
        logs = Path(self.tmp.name) / "logs"
        stem = re.compile(r"hardware-check-\d{8}-\d{6}")
        self.assertTrue(any(stem.fullmatch(p.stem) for p in logs.glob("*.json")))
        markdown = next(logs.glob("hardware-check-*.md")).read_text(encoding="utf-8")
        self.assertIn("issue #15", markdown)
        self.assertIn("Steering", markdown)
        for key in ("identity", "limits", "summary", "issues", "durationS", "mode"):
            self.assertIn(key, report)
        self.assertEqual(report["limits"]["drive_speed"], 25)
        self.assertEqual(report["identity"]["robotId"], "laika-sim")
        self.assertTrue(all("durationS" in step for step in report["steps"]))

    def test_yes_flag_never_reads_input(self):
        script = Scripted(fail_with={"": AssertionError("input was read")})
        code, report = self.run_check(script, namespace(yes=True))
        self.assertEqual(code, 0)
        self.assertEqual(script.prompts, [])
        self.assertEqual(report["summary"], {PASS: 10, FAIL: 0, SKIP: 1})

    def test_a_no_answer_fails_the_step_and_the_run(self):
        code, report = self.run_check(Scripted({"Did the front wheels turn LEFT": "n wrong way"}))
        self.assertEqual(code, 1)
        steps = self.steps(report)
        self.assertEqual(steps["steering"]["status"], FAIL)
        self.assertTrue(any("wrong way" in note for note in steps["steering"]["operatorNotes"]))
        self.assertEqual(steps["head"]["status"], PASS)  # later steps still ran
        self.assertEqual(report["result"], FAIL)

    def test_a_wrong_pan_direction_is_reported_as_a_sign_issue(self):
        code, report = self.run_check(Scripted({"Did the head turn LEFT": "n"}))
        self.assertEqual(code, 1)
        self.assertTrue(any("sign issue" in issue and "LEFT" in issue for issue in report["issues"]))
        markdown = next((Path(self.tmp.name) / "logs").glob("*.md")).read_text(encoding="utf-8")
        self.assertIn("sign issue", markdown)

    def test_skip_answers_skip_the_step_and_exit_zero(self):
        code, report = self.run_check(Scripted({"Did the head": "s"}))
        self.assertEqual(code, 0)
        self.assertEqual(self.steps(report)["head"]["status"], SKIP)

    def test_driving_steps_are_skipped_when_driving_is_off(self):
        code, report = self.run_check(args=namespace(no_driving=True))
        steps = self.steps(report)
        self.assertEqual(code, 0)
        for name in ("drive", "obstacle", "emergency_stop"):
            self.assertEqual(steps[name]["status"], SKIP, name)
        self.assertEqual(steps["steering"]["status"], PASS)

    def test_closed_input_aborts_but_still_writes_the_report(self):
        code, report = self.run_check(Scripted(fail_with={"Did the head": EOFError()}))
        self.assertEqual(code, 1)
        self.assertTrue(report["interrupted"])
        steps = self.steps(report)
        self.assertEqual(steps["head"]["status"], FAIL)
        self.assertEqual(steps["camera"]["status"], SKIP)
        self.assertEqual([s["name"] for s in report["steps"]], STEP_NAMES)

    def test_ctrl_c_stops_the_robot_and_writes_the_report(self):
        code, report = self.run_check(Scripted(fail_with={"Did the robot roll": KeyboardInterrupt()}))
        self.assertEqual(code, 1)
        self.assertEqual(report["interrupted"], "Ctrl+C")
        self.assertEqual(self.steps(report)["drive"]["status"], FAIL)
        self.assertEqual(self.steps(report)["speaker"]["status"], SKIP)

    def test_main_runs_the_check_without_an_ai_key(self):
        with mock.patch("laika.hardware_check.time.sleep"), mock.patch.dict(
            os.environ, {"OPENAI_API_KEY": ""}
        ):
            code = app.main(["--sim", "--check-hardware", "--yes", "--no-mic"])
        self.assertEqual(code, 0)
        self.assertTrue(list((Path(self.tmp.name) / "logs").glob("hardware-check-*.md")))

    def test_report_suggests_a_cliff_reference_with_the_toml_line(self):
        code, report = self.run_check()
        self.assertEqual(code, 0)
        # The simulator's floor reads 1000 and its edge 50 on every channel: halfway is 525.
        self.assertEqual(report["suggestedSettings"], {"cliff_reference": [525, 525, 525]})
        markdown = next((Path(self.tmp.name) / "logs").glob("*.md")).read_text(encoding="utf-8")
        self.assertIn("## Suggested settings", markdown)
        self.assertIn("cliff_reference = [525, 525, 525]", markdown)
        self.assertTrue(any("cliff_reference = [525, 525, 525]" in line for line in self.output))
        steps = self.steps(report)
        self.assertEqual(steps["grayscale"]["data"]["floor"]["grayscale"], [1000, 1000, 1000])
        self.assertEqual(steps["grayscale"]["data"]["lifted"]["grayscale"], [50, 50, 50])

    def test_report_records_the_refused_distances(self):
        code, report = self.run_check()
        steps = self.steps(report)
        obstacle = steps["obstacle"]["data"]
        self.assertIsNotNone(obstacle["refused_at_cm"])
        self.assertLess(obstacle["refused_at_cm"], obstacle["limit_cm"])
        self.assertEqual(steps["grayscale"]["data"]["move_over_edge"], {"result": "rejected", "reason": "cliff"})
        self.assertIn("ultrasonic_invalid_limit", report["limits"])

    def test_the_voice_check_reaches_the_mouth(self):
        mouth = SpyMouth()
        with mock.patch("laika.audio.ConsoleMouth", lambda: mouth):
            self.run_check()
        self.assertEqual(mouth.said, [SPEECH_TEST])


def make_sim_check(robot, tmp, *, staging=None, answers=None, **kwargs):
    safety = SafetyController(robot, speed=25)
    camera = Camera(robot, ImageStore(Path(tmp)), save_images=True)
    output = []
    check = HardwareCheck(
        robot,
        safety,
        camera,
        operator=Operator(answers or Scripted(), output.append),
        staging=staging,
        sleep=lambda seconds: None,
        mode="test",
        **kwargs,
    )
    return check, safety, output


class SuggestionTests(unittest.TestCase):
    def test_halfway_between_floor_and_edge_on_each_channel(self):
        self.assertEqual(suggest_cliff_reference([1000, 1200, 900], [50, 60, 40]), [525, 630, 470])

    def test_the_result_is_rounded_to_whole_counts(self):
        self.assertEqual(suggest_cliff_reference([1001, 1000, 1000], [50, 50, 51]), [526, 525, 526])

    def test_the_minimum_contrast_is_enough(self):
        self.assertEqual(suggest_cliff_reference([200, 200, 200], [100, 100, 100]), [150, 150, 150])

    def test_no_suggestion_when_a_floor_is_not_clearly_above_its_edge(self):
        self.assertIsNone(suggest_cliff_reference([1000, 120, 1000], [50, 90, 50]))  # 30 counts apart
        self.assertIsNone(suggest_cliff_reference([50, 1000, 1000], [1000, 50, 50]))  # inverted


class StepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def run_check(self, robot, **kwargs):
        check, safety, _ = make_sim_check(robot, self.tmp.name, **kwargs)
        self.addCleanup(safety.close)
        return check.run(), safety

    def test_an_exception_in_one_step_does_not_stop_the_others(self):
        class BrokenSensor(SimulatedRobot):
            def distance_cm(self):
                raise RuntimeError("sensor on fire")

        robot = BrokenSensor(house=default_house(), verbose=False)
        report, _ = self.run_check(robot, staging=SimStaging(robot))
        steps = {s.name: s for s in report.steps}
        self.assertEqual(steps["ultrasonic"].status, FAIL)
        self.assertIn("sensor on fire", steps["ultrasonic"].error)
        for name in ("steering", "head", "grayscale", "drive", "obstacle", "camera", "speaker"):
            self.assertEqual(steps[name].status, PASS, name)
        self.assertEqual(report.exit_code, 1)

    def test_the_obstacle_step_is_really_rejected_by_the_safety_controller(self):
        robot = SimulatedRobot(house=default_house(), verbose=False)
        report, _ = self.run_check(robot, staging=SimStaging(robot))
        step = {s.name: s for s in report.steps}["obstacle"]
        self.assertEqual(step.status, PASS)
        self.assertEqual(step.data["result"], {"result": "rejected", "reason": "obstacle"})
        self.assertLess(step.data["distance_cm"], 25)
        self.assertEqual(step.data["refused_at_cm"], step.data["distance_cm"])
        self.assertEqual(robot.speed, 0)

    def test_the_ultrasonic_step_reports_the_invalid_reading_rate(self):
        class SometimesBlind(SimulatedRobot):
            def __init__(self):
                super().__init__(verbose=False, house=default_house())
                self.calls = 0

            def distance_cm(self):
                self.calls += 1
                if self.calls <= 10:  # the clear reading: 3 of 10 invalid
                    return None if self.calls <= 3 else 100.0
                return 15.0  # a hand in front

        report, _ = self.run_check(SometimesBlind())
        step = {s.name: s for s in report.steps}["ultrasonic"]
        self.assertEqual(step.status, PASS)
        self.assertEqual(step.data["clear"]["invalid_rate"], 0.3)
        self.assertEqual(step.data["invalid_rate"], {"clear": 0.3, "hand": 0.0})

    def test_the_cliff_step_finds_a_cliff_reference_and_checks_the_refusal(self):
        robot = SimulatedRobot(house=default_house(), verbose=False)
        report, _ = self.run_check(robot, staging=SimStaging(robot))
        step = {s.name: s for s in report.steps}["grayscale"]
        self.assertEqual(step.status, PASS)
        self.assertEqual(step.data["suggested_cliff_reference"], [525, 525, 525])
        self.assertEqual(step.data["move_over_edge"]["reason"], "cliff")
        self.assertEqual(robot.speed, 0)

    def test_a_forward_move_that_is_not_refused_over_the_edge_fails_the_step(self):
        class EdgeOnlyInTheCheck(SimulatedRobot):
            def cliff_detected(self):
                return True  # the lift check sees the edge, the safety controller's read does not

        robot = EdgeOnlyInTheCheck(house=default_house(), verbose=False)
        report, _ = self.run_check(robot)
        step = {s.name: s for s in report.steps}["grayscale"]
        self.assertEqual(step.status, FAIL)
        self.assertEqual(step.data["move_over_edge"]["result"], "success")
        self.assertTrue(any("expected REJECTED (cliff)" in failure for failure in step.failures))

    def test_a_box_placed_through_the_house_api_is_refused_and_no_box_is_not(self):
        house = default_house()
        robot = SimulatedRobot(house=house, verbose=False)
        pose = robot.pose()
        reach = ROBOT_RADIUS_CM + 25.0
        house.add_thing(
            Thing(
                "box",
                pose.x + reach * math.cos(math.radians(pose.heading)),
                pose.y + reach * math.sin(math.radians(pose.heading)),
                5.0,
            )
        )
        check, safety, _ = make_sim_check(robot, self.tmp.name)
        self.addCleanup(safety.close)
        refused = StepRecord("obstacle", "Obstacle")
        check._run_step(refused, check.step_obstacle)
        self.assertEqual(refused.status, PASS)
        self.assertEqual(refused.data["result"]["reason"], "obstacle")

        house.remove_thing("box")
        moved = StepRecord("obstacle", "Obstacle")
        check._run_step(moved, check.step_obstacle)
        self.assertEqual(moved.status, FAIL)  # nothing in the way: the move went ahead

    def test_the_emergency_stop_step_stops_and_resets_the_latch(self):
        robot = SimulatedRobot(house=default_house(), verbose=False)
        report, safety = self.run_check(robot)
        step = {s.name: s for s in report.steps}["emergency_stop"]
        self.assertEqual(step.status, PASS)
        self.assertEqual(step.data["move"], {"result": "cancelled", "reason": "emergency_stop"})
        self.assertEqual(step.data["speed_after"], 0)
        self.assertEqual(step.data["move_while_latched"], "rejected")
        self.assertFalse(safety.emergency_stopped)

    def test_the_robot_ends_stopped_and_centred(self):
        robot = SimulatedRobot(house=default_house(), verbose=False)
        self.run_check(robot, staging=SimStaging(robot))
        state = robot.get_state()
        self.assertEqual((state.speed, state.steering, state.head_pan, state.head_tilt), (0, 0, 0, 0))

    def test_an_invalid_answer_is_asked_again(self):
        answers = iter(["maybe", "", "YES"])
        output = []
        operator = Operator(lambda prompt: next(answers), output.append)
        self.assertEqual(operator.ask("ok?").choice, "y")
        self.assertTrue(any("Please answer" in line for line in output))

    def test_answers_carry_notes(self):
        operator = Operator(lambda prompt: "no it went right", lambda text: None)
        answer = operator.ask("left?")
        self.assertEqual((answer.choice, answer.note), ("n", "it went right"))

    def test_the_camera_step_fails_when_the_camera_fails(self):
        robot = SimulatedRobot(house=default_house(), verbose=False)
        robot.faults.camera_error = True
        report, _ = self.run_check(robot, staging=SimStaging(robot))
        step = {s.name: s for s in report.steps}["camera"]
        self.assertEqual(step.status, FAIL)

    def test_the_microphone_step_listens_through_the_ear(self):
        class Ear:
            closed = False

            def listen(self, timeout_s=None):
                self.timeout = timeout_s
                return "ehi laika"

            def close(self):
                self.closed = True

        ear = Ear()
        robot = SimulatedRobot(house=default_house(), verbose=False)
        report, _ = self.run_check(robot, staging=SimStaging(robot), ear_factory=lambda: ear)
        step = {s.name: s for s in report.steps}["microphone"]
        self.assertEqual(step.status, PASS)
        self.assertEqual(step.data["heard"], "ehi laika")
        self.assertEqual(ear.timeout, 6.0)
        self.assertTrue(ear.closed)


class FakeCarStaging(Staging):
    """What the operator does by hand, applied to the fake SunFounder car."""

    def __init__(self, car):
        self.car = car
        self.clear = car.distance

    def hand_near(self):
        self.car.distance = 15.0

    def hand_away(self):
        self.car.distance = self.clear

    def lift(self):
        self.car.gray = [50, 50, 50]

    def lower(self):
        self.car.gray = [1000, 1000, 1000]

    def box_ahead(self):
        self.car.distance = 20.0

    def box_away(self):
        self.hand_away()


class PiCarXCheckTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(sys.modules, fake_sunfounder_modules())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_the_whole_check_passes_through_laika_interfaces(self):
        from laika.hardware import PiCarXRobot

        mouth = SpyMouth()
        robot = PiCarXRobot(mouth=mouth)
        car = robot._car
        check, safety, _ = make_sim_check(robot, self.tmp.name, staging=FakeCarStaging(car))
        self.addCleanup(safety.close)
        report = check.run()
        for step in report.steps:
            self.assertIn(step.status, (PASS, SKIP), f"{step.name}: {step.detail}")
        self.assertEqual(report.exit_code, 0)
        calls = [name for name, _ in car.calls]
        for expected in ("steer", "pan", "tilt", "forward", "backward", "stop"):
            self.assertIn(expected, calls)
        self.assertEqual(mouth.said, [SPEECH_TEST])
        # The obstacle step must not have driven: only the two drive-step moves and the
        # emergency step's move reach the motors.
        self.assertEqual(calls.count("forward"), 2)
        self.assertEqual(calls.count("backward"), 1)
        self.assertEqual(car.calls[-1][0] in ("stop", "pan", "tilt", "steer", "reset"), True)
        self.assertEqual(robot.get_state().speed, 0)
        saved = list(Path(self.tmp.name).glob("img-*.jpg"))
        self.assertEqual(len(saved), 1)

    def test_a_vendor_failure_in_a_step_is_recorded_not_raised(self):
        from laika.hardware import PiCarXRobot

        robot = PiCarXRobot(mouth=SpyMouth())

        def boom(angle):
            raise OSError("servo bus down")

        robot._car.set_dir_servo_angle = boom
        check, safety, _ = make_sim_check(robot, self.tmp.name, staging=FakeCarStaging(robot._car))
        self.addCleanup(safety.close)
        report = check.run()
        steps = {s.name: s for s in report.steps}
        self.assertEqual(steps["steering"].status, FAIL)
        self.assertIn("servo bus down", steps["steering"].error)
        self.assertEqual(steps["head"].status, PASS)


class SourceBoundaryTests(unittest.TestCase):
    def test_only_the_adapters_import_sunfounder_libraries(self):
        source = Path(__file__).resolve().parents[2] / "src" / "laika"
        pattern = re.compile(r"^\s*(?:from|import)\s+(?:picarx|robot_hat|picamera2)\b", re.M)
        offenders = [
            path.name
            for path in source.glob("*.py")
            if path.name not in ("hardware.py", "audio.py") and pattern.search(path.read_text("utf-8"))
        ]
        self.assertEqual(offenders, [])
        text = (source / "hardware_check.py").read_text("utf-8")
        for name in ("picarx", "robot_hat", "picamera2"):
            self.assertNotIn(name, text)


if __name__ == "__main__":
    unittest.main()
