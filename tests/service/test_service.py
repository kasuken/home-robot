import configparser
import contextlib
import io
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from laika import app
from laika.safety import SafetyController
from laika.simulator import SimulatedRobot

ROOT = Path(__file__).resolve().parents[2]
UNIT = ROOT / "deploy" / "laika.service"


def run_main(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = app.main(list(argv))
    return code, out.getvalue()


class StopMotorsTests(unittest.TestCase):
    def test_sim_stops_resets_and_closes(self):
        made: list[SimulatedRobot] = []

        class Recording(SimulatedRobot):
            def __init__(self) -> None:
                super().__init__(verbose=False)
                self.calls: list[str] = []
                self.speed = 30
                made.append(self)

            def stop(self) -> None:
                self.calls.append("stop")
                super().stop()

            def reset(self) -> None:
                self.calls.append("reset")
                super().reset()

            def close(self) -> None:
                self.calls.append("close")
                super().close()

        with mock.patch.object(app, "SimulatedRobot", Recording):
            code, out = run_main("--stop-motors", "--sim")
        self.assertEqual(code, 0)
        self.assertEqual(made[0].speed, 0)
        self.assertEqual(made[0].calls[0], "stop")
        self.assertIn("reset", made[0].calls)
        self.assertIn("close", made[0].calls)  # close resets again, so stop can follow it
        self.assertIn("robot stopped", out)

    def test_hardware_errors_are_printed_and_exit_zero(self):
        import laika.hardware as hardware

        with mock.patch.object(hardware, "PiCarXRobot", side_effect=RuntimeError("no HAT")):
            code, out = run_main("--stop-motors")
        self.assertEqual(code, 0)
        self.assertIn("no HAT", out)

    def test_failing_stop_still_exits_zero(self):
        class Broken(SimulatedRobot):
            def stop(self) -> None:
                raise RuntimeError("bus error")

        with mock.patch.object(app, "SimulatedRobot", Broken):
            code, out = run_main("--stop-motors", "--sim")
        self.assertEqual(code, 0)
        self.assertIn("bus error", out)
        self.assertIn("with errors", out)

    def test_needs_no_config(self):
        code, _ = run_main("--stop-motors", "--sim", "--config", "/nonexistent/laika.toml")
        self.assertEqual(code, 0)


class HeartbeatTests(unittest.TestCase):
    def wait_for(self, condition, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return True
            time.sleep(0.01)
        return False

    def test_monitor_beats_while_healthy_and_stops_when_it_hangs(self):
        beats: list[int] = []
        release = threading.Event()
        controller = SafetyController(
            SimulatedRobot(verbose=False), heartbeat=lambda: beats.append(1), watchdog_timeout_s=0
        )
        real_check = controller.check
        hang = threading.Event()

        def check() -> None:
            if hang.is_set():
                release.wait(5)
            real_check()

        controller.check = check
        controller.start()
        try:
            self.assertTrue(self.wait_for(lambda: len(beats) >= 3))
            hang.set()
            time.sleep(0.2)
            frozen = len(beats)
            time.sleep(0.3)
            self.assertLessEqual(len(beats) - frozen, 0)
        finally:
            release.set()
            controller.close()

    def test_a_failing_heartbeat_does_not_disturb_the_monitor(self):
        def boom() -> None:
            raise RuntimeError("socket gone")

        controller = SafetyController(SimulatedRobot(verbose=False), heartbeat=boom)
        with self.assertLogs("laika.safety", level="ERROR"):
            controller.start()
            try:
                time.sleep(0.2)
                self.assertFalse(controller.watchdog_expired)
            finally:
                controller.close()


@unittest.skipUnless(hasattr(signal, "SIGTERM") and os.name == "posix", "needs POSIX signals")
class SignalTests(unittest.TestCase):
    def start(self, tmp: str) -> subprocess.Popen:
        env = dict(
            os.environ,
            LAIKA_DATA_DIR=tmp,
            LAIKA_AI_BASE_URL="http://127.0.0.1:9/v1",
            OPENAI_API_KEY="x",
            PYTHONIOENCODING="utf-8",
        )
        env.pop("NOTIFY_SOCKET", None)
        return subprocess.Popen(
            [sys.executable, str(ROOT / "run-laika.py"), "--sim"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

    def wait_for(self, proc: subprocess.Popen, text: str, timeout: float = 20) -> str:
        seen = ""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            seen += line
            if text in line:
                return seen
        self.fail(f"never saw {text!r}; output so far:\n{seen}")

    def test_signals_run_the_shutdown_path(self):
        for sig in (signal.SIGTERM, signal.SIGHUP):
            with self.subTest(signal=sig.name), tempfile.TemporaryDirectory() as tmp:
                proc = self.start(tmp)
                try:
                    seen = self.wait_for(proc, "sleeping")
                    proc.send_signal(sig)
                    rest, _ = proc.communicate(timeout=10)
                finally:
                    if proc.poll() is None:
                        proc.kill()
                    proc.stdout.close()
                self.assertEqual(proc.returncode, 0, seen + rest)
                self.assertIn(f"{sig.name} received", rest)
                self.assertIn("closed", rest)


class UnitFileTests(unittest.TestCase):
    def setUp(self):
        # strict=False tolerates the repeated keys systemd allows (Environment, ExecStartPre).
        parser = configparser.ConfigParser(strict=False, interpolation=None)
        parser.optionxform = str
        self.raw = UNIT.read_text(encoding="utf-8")
        parser.read_string(self.raw)
        self.service = parser["Service"]
        self.unit = parser["Unit"]

    def lines(self, key: str) -> list[str]:
        return [
            line.split("=", 1)[1].strip()
            for line in self.raw.splitlines()
            if line.startswith(key + "=")
        ]

    def test_type_and_watchdog(self):
        self.assertEqual(self.service["Type"], "notify")
        self.assertGreater(int(self.service["WatchdogSec"]), 0)

    def test_robot_is_stopped_before_and_after(self):
        self.assertTrue(any("--stop-motors" in v for v in self.lines("ExecStopPost")))
        pre = self.lines("ExecStartPre")
        self.assertTrue(pre and all("--stop-motors" in v for v in pre))
        start = self.lines("ExecStart")
        self.assertEqual(len(start), 1)
        self.assertNotIn("--stop-motors", start[0])
        self.assertIn("run-laika.py", start[0])

    def test_restart_and_stop_policy(self):
        self.assertEqual(self.service["Restart"], "on-failure")
        self.assertEqual(self.service["KillSignal"], "SIGTERM")
        self.assertIn("TimeoutStopSec", self.service)
        self.assertIn("StartLimitBurst", self.unit)

    def test_logs_and_data(self):
        self.assertEqual(self.service["StandardOutput"], "journal")
        self.assertEqual(self.service["StandardError"], "journal")
        self.assertIn("LAIKA_DATA_DIR=/var/lib/laika", self.lines("Environment"))
        self.assertEqual(self.service["StateDirectoryMode"], "0700")
        self.assertEqual(self.service["EnvironmentFile"], "-/etc/laika/laika.env")

    def test_repo_path_is_a_placeholder_the_installer_fills(self):
        self.assertIn("@REPO@", self.service["WorkingDirectory"])
        installer = (ROOT / "deploy" / "install-service.sh").read_text(encoding="utf-8")
        self.assertIn("@REPO@", installer)


if __name__ == "__main__":
    unittest.main()
