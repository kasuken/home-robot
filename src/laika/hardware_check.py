"""A guided hardware check: `python3 run-laika.py --check-hardware`.

It walks the operator through every capability of the V0.2 list (docs/prd.md)
and exercises each one through Laika's own interfaces: the `Robot` protocol,
the `SafetyController`, the `Camera` service and the mouth. It never calls the
SunFounder libraries itself, so a pass means Laika's integration works, not
just the vendor code. It runs against the simulator too (`--sim`), which makes
a dry run on a laptop possible.

Each step prints what is about to happen, does it, then asks the operator to
confirm (y / n / s to skip). An exception in one step is recorded as a failure
and the next step still runs. Ctrl+C stops the robot and the report is still
written, as JSON and Markdown under `<data_dir>/logs/`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from laika import __version__
from laika.audio import Ear
from laika.camera import Camera
from laika.commands import CommandResult, Direction, MoveCommand, ResultStatus
from laika.config import Config
from laika.events import EventBus
from laika.robot import PAN_MAX, STEER_MAX, TILT_MAX, TILT_MIN, Robot
from laika.safety import SafetyController
from laika.storage import FILE_MODE, DataPaths

log = logging.getLogger(__name__)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

STEER_TEST_ANGLE = min(25.0, STEER_MAX)
PAN_TEST_ANGLE = min(40.0, PAN_MAX)
TILT_UP_ANGLE = min(30.0, TILT_MAX)
TILT_DOWN_ANGLE = max(-25.0, TILT_MIN)
MOVE_MS = 500
MIC_TIMEOUT_S = 6.0
SPEECH_TEST = "Ciao, sono Laika. Prova audio."
HAND_MAX_CM = 30.0  # the operator holds a hand about 15 cm away; allow for a loose estimate
CLIFF_WAIT_S = 5.0
MOTOR_SETTLE_S = 0.3
# Counts a floor reading must sit above its edge reading for a cliff reference to tell them apart.
MIN_CLIFF_CONTRAST = 100


class Skip(Exception):
    """Raised inside a step to end it as skipped."""


class Aborted(Exception):
    """The operator's input ended (EOF): stop the whole check."""


@dataclass(frozen=True)
class Answer:
    choice: str  # "y", "n" or "s"
    note: str = ""


class Operator:
    """Asks the operator y / n / s questions. Input and output are injectable for tests."""

    def __init__(
        self,
        input_fn: Callable[[str], str] = input,
        output: Callable[[str], None] = print,
        assume_yes: bool = False,
    ) -> None:
        self._input = input_fn
        self.say = output
        self.assume_yes = assume_yes

    def ask(self, question: str) -> Answer:
        """y/yes, n/no or s/skip, optionally followed by a note: `n the head did not move`."""
        if self.assume_yes:
            self.say(f"  ? {question} -> y (--yes)")
            return Answer("y")
        while True:
            try:
                raw = self._input(f"  ? {question} [y/n/s] ").strip()
            except EOFError:
                raise Aborted("input closed") from None
            word, _, note = raw.partition(" ")
            choice = {"y": "y", "yes": "y", "n": "n", "no": "n", "s": "s", "skip": "s"}.get(
                word.lower()
            )
            if choice:
                return Answer(choice, note.strip())
            self.say("  Please answer y (yes), n (no) or s (skip). Add a note after it if useful.")


class Staging:
    """What the operator physically does between prompts. The real robot needs hands, not code."""

    def hand_near(self) -> None: ...

    def hand_away(self) -> None: ...

    def lift(self) -> None: ...

    def lower(self) -> None: ...

    def box_ahead(self) -> None: ...

    def box_away(self) -> None: ...


class SimStaging(Staging):
    """Does for the simulator what the operator would do by hand, so a dry run can pass."""

    BOX = "check-box"

    def __init__(self, robot: Any) -> None:
        self.robot = robot

    def hand_near(self) -> None:
        self.robot.distance = 15.0

    def hand_away(self) -> None:
        from laika.simulator import AUTO

        self.robot.distance = AUTO

    def lift(self) -> None:
        self.robot.cliff = True

    def lower(self) -> None:
        self.robot.cliff = None

    def box_ahead(self) -> None:
        house = self.robot.house
        if house is None:
            self.robot.distance = 20.0
            return
        import math

        from laika.house import Thing
        from laika.simulator import ROBOT_RADIUS_CM

        pose = self.robot.pose()
        radius = 5.0
        reach = ROBOT_RADIUS_CM + 20.0 + radius
        angle = math.radians(pose.heading)
        house.add_thing(
            Thing(self.BOX, pose.x + reach * math.cos(angle), pose.y + reach * math.sin(angle), radius)
        )

    def box_away(self) -> None:
        house = self.robot.house
        if house is None:
            self.hand_away()
            return
        try:
            house.remove_thing(self.BOX)
        except (KeyError, ValueError):
            pass


@dataclass
class StepRecord:
    name: str
    title: str
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    skip_reason: str = ""
    error: str = ""
    duration_s: float = 0.0
    ran: bool = False

    def fail(self, message: str) -> None:
        self.failures.append(message)

    @property
    def status(self) -> str:
        if self.failures or self.error:
            return FAIL
        return SKIP if self.skip_reason or not self.ran else PASS

    @property
    def detail(self) -> str:
        parts = [*self.failures]
        if self.error:
            parts.append(f"error: {self.error}")
        if not parts and self.skip_reason:
            parts.append(self.skip_reason)
        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "status": self.status,
            "detail": self.detail,
            "failures": self.failures,
            "error": self.error,
            "skipReason": self.skip_reason,
            "operatorNotes": self.notes,
            "data": self.data,
            "durationS": round(self.duration_s, 2),
        }


@dataclass
class CheckReport:
    mode: str
    started_at: datetime
    finished_at: datetime
    identity: dict[str, Any]
    limits: dict[str, Any]
    steps: list[StepRecord]
    issues: list[str]
    interrupted: str = ""

    @property
    def suggestions(self) -> dict[str, list[int]]:
        """Settings the steps worked out from this run, to paste into config/laika.toml."""
        found: dict[str, list[int]] = {}
        for step in self.steps:
            value = step.data.get("suggested_cliff_reference")
            if value:
                found["cliff_reference"] = value
        return found

    def suggestion_lines(self) -> list[str]:
        """The TOML lines to paste under [robot], or nothing when there is no suggestion."""
        return [f"{key} = {_toml_list(value)}" for key, value in self.suggestions.items()]

    @property
    def counts(self) -> dict[str, int]:
        return {s: sum(1 for step in self.steps if step.status == s) for s in (PASS, FAIL, SKIP)}

    @property
    def passed(self) -> bool:
        return not self.interrupted and self.counts[FAIL] == 0

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "report": "laika hardware check",
            "laikaVersion": __version__,
            "mode": self.mode,
            "result": PASS if self.passed else FAIL,
            "interrupted": self.interrupted,
            "startedAt": self.started_at.isoformat(timespec="seconds"),
            "finishedAt": self.finished_at.isoformat(timespec="seconds"),
            "durationS": round((self.finished_at - self.started_at).total_seconds(), 1),
            "identity": self.identity,
            "limits": self.limits,
            "summary": self.counts,
            "issues": self.issues,
            "suggestedSettings": self.suggestions,
            "steps": [step.to_dict() for step in self.steps],
        }

    def summary_table(self) -> str:
        width = max([len(step.title) for step in self.steps] + [4])
        lines = [f"{'Step'.ljust(width)}  Result  Time    Detail"]
        for step in self.steps:
            lines.append(
                f"{step.title.ljust(width)}  {step.status.ljust(6)}  "
                f"{step.duration_s:5.1f}s  {step.detail}"
            )
        return "\n".join(lines)

    def to_markdown(self) -> str:
        data = self.to_dict()
        lines = [
            f"# Laika hardware check: {data['result']}",
            "",
            f"- Mode: {self.mode}",
            f"- Robot: {self.identity.get('name')} ({self.identity.get('robotId')}), "
            f"Laika {__version__}",
            f"- Started: {data['startedAt']}, duration {data['durationS']} s",
            f"- Summary: {self.counts[PASS]} passed, {self.counts[FAIL]} failed, "
            f"{self.counts[SKIP]} skipped",
        ]
        if self.interrupted:
            lines.append(f"- Interrupted: {self.interrupted}")
        lines += ["", "Attach this report to GitHub issue #15.", ""]
        if self.issues:
            lines += ["## Issues to fix", ""] + [f"- {issue}" for issue in self.issues] + [""]
        if self.suggestions:
            lines += [
                "## Suggested settings", "",
                "Paste under `[robot]` in `config/laika.toml`, then run the check again:", "",
                "```toml", *self.suggestion_lines(), "```", "",
            ]
        lines += ["## Steps", "", "| Step | Result | Time | Detail |", "| --- | --- | --- | --- |"]
        for step in self.steps:
            detail = step.detail.replace("|", "/")
            lines.append(f"| {step.title} | {step.status} | {step.duration_s:.1f} s | {detail} |")
        lines += ["", "## Limits used", ""]
        lines += [f"- {key}: {value}" for key, value in self.limits.items()]
        lines += ["", "## Details", ""]
        for step in self.steps:
            lines += [f"### {step.title}: {step.status}", ""]
            if step.detail:
                lines.append(f"- Detail: {step.detail}")
            lines += [f"- Operator note: {note}" for note in step.notes]
            lines += [
                f"- {key}: {json.dumps(value, ensure_ascii=False)}" for key, value in step.data.items()
            ]
            lines.append("")
        return "\n".join(lines)


def write_report(report: CheckReport, directory: Path) -> tuple[Path, Path]:
    """Write `hardware-check-<timestamp>.json` and `.md`; never overwrites an earlier report."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = report.started_at.strftime("%Y%m%d-%H%M%S")
    suffix = 0
    while True:
        base = f"hardware-check-{stamp}" + (f"-{suffix}" if suffix else "")
        json_path, md_path = directory / f"{base}.json", directory / f"{base}.md"
        if not json_path.exists() and not md_path.exists():
            break
        suffix += 1
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    for path in (json_path, md_path):
        try:
            os.chmod(path, FILE_MODE)
        except OSError:
            pass
    return json_path, md_path


def suggest_cliff_reference(floor: Sequence[int], edge: Sequence[int]) -> list[int] | None:
    """The cliff reference halfway between each channel's floor and edge readings, or None.

    A reading at or below the reference is a cliff, so halfway leaves the same
    margin on both sides. None when a floor does not read clearly above its
    edge: no reference can then tell them apart.
    """
    if any(f - e < MIN_CLIFF_CONTRAST for f, e in zip(floor, edge)):
        return None
    return [round((f + e) / 2) for f, e in zip(floor, edge)]


def _toml_list(values: Sequence[int]) -> str:
    return "[" + ", ".join(str(value) for value in values) + "]"


def _percent(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate:.0%}"


def _stats(values: list[float | None]) -> dict[str, Any]:
    valid = [v for v in values if v is not None]
    return {
        "readings": len(values),
        "invalid": len(values) - len(valid),
        "invalid_rate": round((len(values) - len(valid)) / len(values), 2) if values else None,
        "min": round(min(valid), 1) if valid else None,
        "median": round(statistics.median(valid), 1) if valid else None,
        "max": round(max(valid), 1) if valid else None,
    }


class HardwareCheck:
    def __init__(
        self,
        robot: Robot,
        safety: SafetyController,
        camera: Camera,
        *,
        operator: Operator | None = None,
        ear_factory: Callable[[], Ear] | None = None,
        mic_skip_reason: str = "microphone check not requested",
        staging: Staging | None = None,
        limits: Mapping[str, Any] | None = None,
        honk_volume: int = 80,
        mode: str = "robot",
        sleep: Callable[[float], None] | None = None,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.robot = robot
        self.safety = safety
        self.camera = camera
        self.op = operator or Operator()
        self.ear_factory = ear_factory
        self.mic_skip_reason = mic_skip_reason
        self.staging = staging or Staging()
        self.honk_volume = honk_volume
        self.mode = mode
        self._sleep = sleep or time.sleep
        self._now = now
        self.issues: list[str] = []
        self.limits: dict[str, Any] = dict(limits) if limits is not None else {
            "drive_speed": safety.speed,
            "max_drive_s": safety.max_duration_s,
            "min_front_distance_cm": safety.min_front_distance_cm,
            "backward_max_s": safety.backward_max_s,
            "ultrasonic_invalid_limit": safety.ultrasonic_invalid_limit,
            "driving_enabled": safety.enabled,
        }
        self.steps: list[tuple[str, str, Callable[[StepRecord], None]]] = [
            ("steering", "Steering", self.step_steering),
            ("head", "Head pan and tilt", self.step_head),
            ("ultrasonic", "Ultrasonic sensor", self.step_ultrasonic),
            ("grayscale", "Grayscale and cliff", self.step_grayscale),
            ("drive", "Drive forward and backward", self.step_drive),
            ("obstacle", "Obstacle refusal", self.step_obstacle),
            ("emergency_stop", "Stop and emergency stop", self.step_emergency_stop),
            ("camera", "Camera capture", self.step_camera),
            ("speaker", "Speaker", self.step_speaker),
            ("microphone", "Microphone", self.step_microphone),
        ]

    # --- running ---------------------------------------------------------

    def run(self) -> CheckReport:
        started = self._now()
        identity = self.robot.identity()
        say = self.op.say
        say(f"Laika hardware check ({self.mode}): {identity.name} ({identity.robot_id}), Laika {__version__}.")
        say("Answer y (yes), n (no) or s (skip) to each question. Ctrl+C stops the robot.")
        records: list[StepRecord] = []
        interrupted = ""
        try:
            for index, (name, title, step) in enumerate(self.steps, 1):
                record = StepRecord(name, title)
                records.append(record)
                say(f"\n== Step {index}/{len(self.steps)}: {title} ==")
                try:
                    self._run_step(record, step)
                except (KeyboardInterrupt, Aborted) as stop:
                    interrupted = "Ctrl+C" if isinstance(stop, KeyboardInterrupt) else str(stop)
                    record.fail(f"interrupted ({interrupted})")
                    break
            for name, title, _ in self.steps[len(records) :]:
                skipped = StepRecord(name, title, skip_reason="not run: check interrupted")
                records.append(skipped)
        finally:
            self._make_safe()
        report = CheckReport(
            self.mode, started, self._now(), identity.to_dict(), self.limits, records, self.issues,
            interrupted,
        )
        say("\n" + report.summary_table())
        for line in report.suggestion_lines():
            say(f"Suggested [robot] setting: {line}")
        say(f"\nResult: {PASS if report.passed else FAIL}")
        return report

    def _run_step(self, record: StepRecord, step: Callable[[StepRecord], None]) -> None:
        begin = time.monotonic()
        try:
            step(record)
            record.ran = True
        except Skip as skip:
            record.skip_reason = str(skip) or "skipped by the operator"
            record.ran = True
        except (KeyboardInterrupt, Aborted):
            raise
        except Exception as error:  # noqa: BLE001 - one broken step must not stop the others
            log.exception("hardware check step %s failed", record.name)
            record.error = f"{type(error).__name__}: {error}"
            record.ran = True
        finally:
            record.duration_s = time.monotonic() - begin
            self._make_safe()
        self.op.say(f"  -> {record.status}" + (f": {record.detail}" if record.detail else ""))

    def _make_safe(self) -> None:
        """Between steps and at the end: motors stopped, latch cleared, head and wheels centred."""
        for action in (
            self.safety.stop,
            self.safety.reset_emergency_stop,
            self.staging.hand_away,
            self.staging.box_away,
            self.staging.lower,
            self.robot.reset,
        ):
            try:
                action()
            except Exception as error:  # noqa: BLE001
                log.error("hardware check cleanup failed: %s", error)

    # --- helpers ---------------------------------------------------------

    def say(self, message: str) -> None:
        self.op.say(f"  {message}")

    def confirm(self, record: StepRecord, question: str, issue: str = "") -> bool:
        """Ask a yes/no question. `n` fails the step (and records `issue`); `s` ends it as skipped."""
        answer = self.op.ask(question)
        if answer.note:
            record.notes.append(f"{question} -> {answer.note}")
        if answer.choice == "s":
            raise Skip("skipped by the operator")
        if answer.choice == "n":
            record.fail(f"operator said no: {question}")
            if issue:
                self.issues.append(issue)
                record.fail(issue)
            return False
        return True

    def ready(self, record: StepRecord, instruction: str) -> None:
        """Tell the operator what to prepare; anything but y skips the step."""
        answer = self.op.ask(f"{instruction} Ready?")
        if answer.note:
            record.notes.append(f"{instruction} -> {answer.note}")
        if answer.choice != "y":
            raise Skip("operator was not ready")

    def _read_distances(self, count: int, interval_s: float) -> list[float | None]:
        values: list[float | None] = []
        for _ in range(count):
            values.append(self.robot.distance_cm())
            self._sleep(interval_s)
        return values

    def _move(self, direction: Direction) -> CommandResult:
        command = MoveCommand(
            direction=direction, speed=max(1, self.safety.speed), duration_ms=MOVE_MS
        )
        result = self.safety.execute(command)
        assert result is not None  # wait=True always returns a result
        self.say(f"{direction.value} {MOVE_MS} ms at speed {command.speed}: "
                 f"{result.result.value}{' (' + result.reason + ')' if result.reason else ''}")
        return result

    def _require_driving(self) -> None:
        if not self.safety.enabled or self.safety.speed <= 0:
            raise Skip("driving is disabled (allow_driving = false, --no-driving or speed 0)")

    # --- steps -----------------------------------------------------------

    def step_steering(self, record: StepRecord) -> None:
        self.say("The front wheels will turn centre, left, right, centre. The wheels do not spin,")
        self.say("so this is safe even with the robot on a table.")
        self.robot.steer(0)
        self._sleep(0.5)
        for label, angle in (("LEFT", -STEER_TEST_ANGLE), ("RIGHT", STEER_TEST_ANGLE), ("straight", 0)):
            self.robot.steer(angle)
            self._sleep(0.6)
            wording = "Are the front wheels straight again" if angle == 0 else (
                f"Did the front wheels turn {label}"
            )
            self.confirm(
                record,
                wording + "?",
                "" if angle == 0 else f"steering: angle {angle:+.0f} did not turn the wheels {label}",
            )
        record.data["angles"] = [0, -STEER_TEST_ANGLE, STEER_TEST_ANGLE, 0]

    def step_head(self, record: StepRecord) -> None:
        self.say("The head will pan left and right, then tilt up and down, then centre.")
        self.say("This also checks the sign convention: negative pan = left, positive tilt = up.")
        moves = (
            ("pan LEFT", (-PAN_TEST_ANGLE, 0), "turn LEFT", "sign issue: negative pan (-) did not turn the head LEFT"),
            ("pan RIGHT", (PAN_TEST_ANGLE, 0), "turn RIGHT", "sign issue: positive pan (+) did not turn the head RIGHT"),
            ("tilt UP", (0, TILT_UP_ANGLE), "tilt UP", "sign issue: positive tilt (+) did not tilt the head UP"),
            ("tilt DOWN", (0, TILT_DOWN_ANGLE), "tilt DOWN", "sign issue: negative tilt (-) did not tilt the head DOWN"),
        )
        answers: dict[str, str] = {}
        for label, (pan, tilt), verb, issue in moves:
            self.robot.look(pan, tilt)
            self._sleep(0.6)
            ok = self.confirm(record, f"Did the head {verb} (from the robot's point of view)?", issue)
            answers[label] = "ok" if ok else "wrong"
        self.robot.look(0, 0)
        self._sleep(0.5)
        record.data["answers"] = answers
        record.data["pose_after_centre"] = list(self.robot.head_pose())

    def step_ultrasonic(self, record: StepRecord) -> None:
        self.say("Reading the ultrasonic sensor 10 times over about 2 seconds. Keep the front clear.")
        baseline = _stats(self._read_distances(10, 0.2))
        record.data["clear"] = baseline
        self.say(f"clear: {baseline}")
        if baseline["median"] is None:
            record.fail("no valid ultrasonic readings with the front clear")
        self.ready(record, "Hold a hand about 15 cm in front of the sensor.")
        self.staging.hand_near()
        self._sleep(0.3)
        near = _stats(self._read_distances(5, 0.2))
        self.staging.hand_away()
        record.data["hand"] = near
        record.data["invalid_rate"] = {"clear": baseline["invalid_rate"], "hand": near["invalid_rate"]}
        self.say(f"hand: {near}")
        self.say(
            f"invalid readings: {_percent(baseline['invalid_rate'])} with the front clear, "
            f"{_percent(near['invalid_rate'])} with a hand in front"
        )
        limit = self.safety.ultrasonic_invalid_limit
        self.say(
            f"a forward move stops after {limit} invalid readings in a row"
            if limit else "invalid readings never stop a forward move (ultrasonic_invalid_limit = 0)"
        )
        median = near["median"]
        if median is None:
            record.fail("no valid readings with a hand in front")
        elif median > HAND_MAX_CM or (
            baseline["median"] is not None and median >= baseline["median"]
        ):
            record.fail(
                f"the reading did not drop with a hand in front (median {median} cm, "
                f"clear {baseline['median']} cm)"
            )
        self.confirm(record, "Did the readings look plausible?")

    def step_grayscale(self, record: StepRecord) -> None:
        self.say("Reading the 3 grayscale sensors with the robot on the floor.")
        floor = self.robot.get_sensors()
        record.data["floor"] = {"grayscale": list(floor.grayscale), "cliff": floor.cliff_detected}
        self.say(f"floor: grayscale {list(floor.grayscale)}, cliff_detected {floor.cliff_detected}")
        if floor.cliff_detected:
            record.fail("the floor reads as a cliff: recalibrate the cliff reference")
        # Lifted into the air, never held over a real edge: this step also asks for a forward
        # move that must be refused, and if the sensor missed the edge the wheels would turn.
        self.ready(
            record,
            "Lift the robot into the air with your hands so the sensors see nothing below. "
            "Do not hold it over a table edge: the next check asks it to drive.",
        )
        self.staging.lift()
        lifted = False
        for _ in range(int(CLIFF_WAIT_S / 0.2)):
            if self.robot.cliff_detected():
                lifted = True
                break
            self._sleep(0.2)
        sensors = self.robot.get_sensors()
        record.data["lifted"] = {"grayscale": list(sensors.grayscale), "cliff": lifted}
        self.say(f"lifted: grayscale {list(sensors.grayscale)}, cliff_detected {lifted}")
        if not lifted:
            record.fail(f"cliff_detected never became true within {CLIFF_WAIT_S:.0f} s")
        else:
            self._suggest_cliff(record, list(floor.grayscale), list(sensors.grayscale))
            self._refuse_over_edge(record)
        self.ready(record, "Put the robot back on the floor.")
        self.staging.lower()
        self._sleep(0.3)
        if self.robot.cliff_detected():
            record.fail("the floor still reads as a cliff after putting the robot back")

    def _suggest_cliff(self, record: StepRecord, floor: list[int], edge: list[int]) -> None:
        suggestion = suggest_cliff_reference(floor, edge)
        record.data["suggested_cliff_reference"] = suggestion
        if suggestion is None:
            record.fail(
                f"no cliff reference: each floor reading must be at least {MIN_CLIFF_CONTRAST} "
                f"above its edge reading (floor {floor}, edge {edge})"
            )
            return
        self.say(f"suggested cliff_reference {_toml_list(suggestion)} (halfway between floor and edge)")

    def _refuse_over_edge(self, record: StepRecord) -> None:
        """With the edge in view a forward move must be REJECTED for a cliff.

        A refused move never reaches the motors, so this is safe with the wheels in the air.
        """
        if not self.safety.enabled or self.safety.speed <= 0:
            record.notes.append("forward move over the edge not checked: driving is off")
            return
        command = MoveCommand(
            direction=Direction.FORWARD, speed=max(1, self.safety.speed), duration_ms=MOVE_MS
        )
        result = self.safety.execute(command)
        assert result is not None  # wait=True always returns a result
        record.data["move_over_edge"] = {"result": result.result.value, "reason": result.reason}
        self.say(f"forward move with the edge in view: {result.result.value} ({result.reason})")
        if result.result is not ResultStatus.REJECTED or result.reason != "cliff":
            record.fail(
                f"expected REJECTED (cliff) with the edge in view, got {result.result.value} "
                f"({result.reason})"
            )

    def step_drive(self, record: StepRecord) -> None:
        self._require_driving()
        self.say("The robot will drive forward 0.5 s, then backward 0.5 s, through the safety controller.")
        self.ready(record, "Put the robot on the floor with about 1 m of free space. The wheels will spin.")
        results = {}
        for direction in (Direction.FORWARD, Direction.BACKWARD):
            result = self._move(direction)
            results[direction.value] = {"result": result.result.value, "reason": result.reason}
            if result.result is not ResultStatus.SUCCESS:
                record.fail(f"{direction.value} move was {result.result.value}: {result.reason}")
        record.data["moves"] = results
        record.data["speed"] = self.safety.speed
        self.confirm(record, "Did the robot roll forward and then back?")

    def step_obstacle(self, record: StepRecord) -> None:
        self._require_driving()
        self.say("With a box in front, a forward move must be REJECTED for an obstacle.")
        self.ready(record, f"Place a box about 20 cm in front of the robot (limit is {self.safety.min_front_distance_cm:g} cm).")
        self.staging.box_ahead()
        try:
            self._sleep(0.3)
            distance = self.robot.get_sensors().distance_cm
            record.data["distance_cm"] = distance
            self.say(f"ultrasonic reads {distance} cm")
            result = self._move(Direction.FORWARD)
            record.data["result"] = {"result": result.result.value, "reason": result.reason}
            refused = result.result is ResultStatus.REJECTED and result.reason == "obstacle"
            record.data["refused_at_cm"] = distance if refused else None
            record.data["limit_cm"] = self.safety.min_front_distance_cm
            if refused:
                self.say(f"refused at {distance} cm (limit {self.safety.min_front_distance_cm:g} cm)")
            else:
                record.fail(
                    f"expected REJECTED (obstacle), got {result.result.value} ({result.reason})"
                )
        finally:
            self.safety.stop()
            self.staging.box_away()

    def step_emergency_stop(self, record: StepRecord) -> None:
        self._require_driving()
        self.say("A 1.5 s forward move starts; after 0.3 s Laika calls the emergency stop.")
        self.ready(record, "Keep the robot on the floor with free space ahead and the box removed.")
        command = MoveCommand(
            direction=Direction.FORWARD, speed=max(1, self.safety.speed), duration_ms=1500
        )
        try:
            early = self.safety.execute(command, wait=False)
            if early is not None:
                record.fail(f"the move did not start: {early.result.value} ({early.reason})")
                return
            self._sleep(0.3)
            self.safety.emergency_stop("hardware check")
            final = self.safety.wait(command.id, 2.0)
            speed = self.robot.get_state().speed
            record.data["move"] = (
                {"result": final.result.value, "reason": final.reason} if final else None
            )
            record.data["speed_after"] = speed
            self.say(f"move ended: {record.data['move']}, motor speed now {speed}")
            if final is None or final.result is not ResultStatus.CANCELLED or final.reason != "emergency_stop":
                record.fail("the move was not cancelled by the emergency stop")
            if speed != 0:
                record.fail(f"the motors were still running after the emergency stop (speed {speed})")
            blocked = self.safety.execute(
                MoveCommand(direction=Direction.FORWARD, speed=max(1, self.safety.speed), duration_ms=200)
            )
            record.data["move_while_latched"] = blocked.result.value if blocked else None
            if blocked is None or blocked.reason != "emergency_stop":
                record.fail("a move was not refused while the emergency stop was latched")
        finally:
            self.safety.reset_emergency_stop()
        if self.safety.emergency_stopped:
            record.fail("the emergency stop latch did not reset")
        self.confirm(record, "Did the robot stop right away?")

    def step_camera(self, record: StepRecord) -> None:
        self.say("Capturing one image through the camera service.")
        observation = self.camera.capture("hardware-check")
        if observation is None:
            record.fail("the camera returned no image (see the log for the hardware error)")
            return
        path = self.camera.last_path
        size = len(observation.jpeg) if observation.jpeg else 0
        record.data.update(
            width=observation.width,
            height=observation.height,
            jpeg_bytes=size,
            path=str(path) if path else None,
        )
        self.say(f"{observation.width}x{observation.height}, {size} bytes, saved to {path}")
        if observation.jpeg is None:
            record.notes.append("no JPEG: this robot has no real camera (simulator)")
            return
        if not observation.jpeg.startswith(b"\xff\xd8"):
            record.fail("the image is not a JPEG")
        if self.camera.save_images and (path is None or not path.exists()):
            record.fail("the image was not saved")
        self.confirm(record, f"Open {path} and check the picture. Does it look right?")

    def step_speaker(self, record: StepRecord) -> None:
        self.say("The robot will honk, then say an Italian test sentence.")
        self.ready(record, "Check the speaker volume.")
        self.robot.honk(self.honk_volume)
        self._sleep(1.5)
        self.confirm(record, "Did you hear the horn?")
        self.robot.speak(SPEECH_TEST)
        self.confirm(record, f"Did you hear Laika say '{SPEECH_TEST}'?")

    def step_microphone(self, record: StepRecord) -> None:
        if self.ear_factory is None:
            raise Skip(self.mic_skip_reason)
        self.say(f"Laika will listen once for {MIC_TIMEOUT_S:.0f} seconds.")
        self.ready(record, "Get ready to say a short sentence in Italian after this.")
        ear = self.ear_factory()
        try:
            self.say("Speak now...")
            heard = ear.listen(timeout_s=MIC_TIMEOUT_S)
        finally:
            try:
                ear.close()
            except Exception as error:  # noqa: BLE001
                log.error("could not close the ear: %s", error)
        record.data["heard"] = heard
        self.say(f"heard: {heard!r}")
        if not heard:
            record.fail(f"nothing was heard within {MIC_TIMEOUT_S:.0f} s")
            return
        self.confirm(record, f"Laika heard '{heard}'. Is that what you said?")


def run_from_args(
    args: argparse.Namespace,
    config: Config,
    *,
    input_fn: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
    sleep: Callable[[float], None] | None = None,
) -> int:
    """Build the same objects as the app and run the check. Returns the exit code (0 pass, 1 fail)."""
    from laika.audio import TTS_ENGINES, ConsoleMouth, VoskEar, make_mouth
    from laika.events import EventType
    from laika.house import default_house
    from laika.simulator import SimulatedRobot
    from laika.storage import ImageStore, attach_storage, resolve_data_dir

    if config.voice.tts_engine not in TTS_ENGINES:
        output(f"[voice] tts_engine must be one of: {', '.join(TTS_ENGINES)}")
        return 2
    try:
        paths = DataPaths(resolve_data_dir(config.storage.data_dir)).ensure()
        event_file, _ = attach_storage(config.storage, paths)
    except OSError as error:
        output(f"Cannot use the data directory: {error}. Set [storage] data_dir or LAIKA_DATA_DIR.")
        return 2
    output(f"[laika] data directory: {paths.root}")
    events = EventBus()
    events.subscribe(event_file)
    events.start()

    sim = bool(args.sim)
    mouth: Any = None
    if sim:
        mouth = ConsoleMouth()
    else:
        try:
            mouth = make_mouth(
                config.voice.tts_engine,
                edge_voice=config.voice.edge_voice,
                piper_model=config.voice.tts_model,
            )
        except Exception as error:  # noqa: BLE001 - the speaker step will report it
            output(f"[laika] warning: could not set up the voice ({error}); the speaker step will fail")
    robot: Robot
    try:
        if sim:
            robot = SimulatedRobot(mouth=mouth, house=default_house(), events=events, verbose=False)
        else:
            from laika.hardware import PiCarXRobot

            robot = PiCarXRobot(
                mouth=mouth,
                image_size=config.camera.image_size(),
                jpeg_quality=config.camera.quality(),
                cliff_reference=config.robot.cliff_reference,
            )
    except Exception as error:  # noqa: BLE001
        output(f"Cannot start the robot: {error}")
        events.close()
        event_file.close()
        return 1

    driving = config.robot.allow_driving and not args.no_driving
    camera = Camera(
        robot,
        ImageStore(paths.images, config.storage.max_images, config.storage.max_images_mb),
        events,
        min_interval_s=config.camera.min_interval_s,
        max_per_minute=config.camera.max_per_minute,
        save_images=config.camera.save_images,
    )
    safety = SafetyController(
        robot,
        enabled=driving,
        speed=config.robot.drive_speed,
        max_duration_s=config.robot.max_drive_s,
        backward_max_s=config.robot.backward_max_s,
        min_front_distance_cm=config.robot.min_front_distance_cm,
        cadence_hz=config.robot.safety_cadence_hz,
        command_max_age_s=config.robot.command_max_age_s,
        ultrasonic_invalid_limit=config.robot.ultrasonic_invalid_limit,
        events=events,
        watchdog_timeout_s=config.robot.watchdog_timeout_s,
    )
    safety.start()
    ear_factory: Callable[[], Ear] | None = None
    mic_reason = "skipped with --no-mic"
    if sim:
        mic_reason = "no microphone in the simulator"
    elif not args.no_mic:
        ear_factory = lambda: VoskEar(config.voice.stt_language)  # noqa: E731
    if sim:
        cliff_reference: object = "simulator"
    else:
        cliff_reference = config.robot.cliff_reference or "vendor value from picar-x.conf"
    limits = {
        "drive_speed": safety.speed,
        "max_drive_s": safety.max_duration_s,
        "backward_max_s": min(safety.max_duration_s, safety.backward_max_s),
        "min_front_distance_cm": safety.min_front_distance_cm,
        "command_max_age_s": safety.command_max_age_s,
        "safety_cadence_hz": config.robot.safety_cadence_hz,
        "ultrasonic_invalid_limit": safety.ultrasonic_invalid_limit,
        "cliff_reference": cliff_reference,
        "driving_enabled": driving,
        "watchdog_timeout_s": config.robot.watchdog_timeout_s,
        "honk_volume": config.robot.honk_volume,
        "camera_image_size": list(config.camera.image_size()),
        "camera_min_interval_s": config.camera.min_interval_s,
    }
    check = HardwareCheck(
        robot,
        safety,
        camera,
        operator=Operator(input_fn, output, assume_yes=args.yes),
        ear_factory=ear_factory,
        mic_skip_reason=mic_reason,
        staging=SimStaging(robot) if sim else Staging(),
        limits=limits,
        honk_volume=config.robot.honk_volume,
        mode="simulator" if sim else "robot",
        **({"sleep": sleep} if sleep is not None else {}),
    )
    events.emit(EventType.HARDWARE_READY, robotId=robot.identity().robot_id, name="hardware-check")
    try:
        report = check.run()
        json_path, md_path = write_report(report, paths.logs)
        output(f"[laika] report: {json_path}")
        output(f"[laika] report: {md_path}")
        output("[laika] attach the report to GitHub issue #15")
        return report.exit_code
    finally:
        for close in (safety.close, robot.reset, robot.close, events.close, event_file.close):
            try:
                close()
            except Exception as error:  # noqa: BLE001 - shutting down: nothing more to do
                log.error("hardware check shutdown: %s", error)
