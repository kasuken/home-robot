"""The unattended checklist (#51): it passes and fails on synthetic data directories."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from laika import app, unattended
from laika.config import Config
from laika.store import DB_NAME, Store
from laika.unattended import FAIL, INFO, OK, SKIP

SERVICE_OK = (OK, "installed and enabled")
SERVICE_MISSING = (FAIL, "missing")


def report(mode="robot", result="pass", offset=None, used=0.0):
    data = {"report": "laika hardware check", "mode": mode, "result": result, "limits": {"battery_voltage_offset": used}}
    if offset is not None:
        data["suggestedSettings"] = {"voltage_offset": offset}
    return data


class ChecklistCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        (self.dir / "logs").mkdir()
        self.config = Config()
        self.config.robot.allow_autonomous_driving = True
        self.config.battery.voltage_offset = 0.25

    def write(self, name, data):
        (self.dir / "logs" / name).write_text(json.dumps(data), encoding="utf-8")

    def evaluate(self, probe=lambda: SERVICE_OK, **kwargs):
        items = unattended.evaluate(self.config, self.dir, service_probe=probe, **kwargs)
        return {item.name.split(" ")[0]: item for item in items}

    def statuses(self, **kwargs):
        return {name: item.status for name, item in self.evaluate(**kwargs).items()}


class EvaluateTests(ChecklistCase):
    def test_everything_met(self):
        self.write("hardware-check-20261001-100000.json", report())
        items = unattended.evaluate(self.config, self.dir, service_probe=lambda: SERVICE_OK)
        self.assertEqual(unattended.unmet(items), [])
        self.assertEqual(items[-1].status, INFO)  # the house rules are information, not a gate
        text = unattended.render(items)
        self.assertIn("All required items are met", text)
        self.assertIn("still yours", text)

    def test_no_report_fails(self):
        item = self.evaluate()["hardware"]
        self.assertEqual(item.status, FAIL)
        self.assertIn("--check-hardware", item.detail)

    def test_the_latest_report_counts(self):
        self.write("hardware-check-20261001-100000.json", report())
        self.write("hardware-check-20261002-100000.json", report(result="fail"))
        self.assertEqual(self.statuses()["hardware"], FAIL)
        self.write("hardware-check-20261003-100000.json", report())
        self.assertEqual(self.statuses()["hardware"], OK)

    def test_a_simulator_report_does_not_count(self):
        self.write("hardware-check-20261001-100000.json", report(mode="simulator"))
        item = self.evaluate()["hardware"]
        self.assertEqual(item.status, FAIL)
        self.assertIn("simulator", item.detail)

    def test_an_unreadable_report_fails(self):
        (self.dir / "logs" / "hardware-check-20261001-100000.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(self.statuses()["hardware"], FAIL)

    def test_battery_calibration(self):
        self.write("hardware-check-20261001-100000.json", report())
        self.assertEqual(self.statuses()["battery"], OK)  # the offset is set
        self.config.battery.voltage_offset = 0.0
        self.assertEqual(self.statuses()["battery"], FAIL)  # nothing recorded
        self.write("hardware-check-20261001-100000.json", report(offset=-0.12))
        self.assertEqual(self.statuses()["battery"], OK)  # recorded in a passed report
        self.write("hardware-check-20261001-100000.json", report(result="fail", offset=-0.12))
        self.assertEqual(self.statuses()["battery"], FAIL)
        self.write("hardware-check-20261001-100000.json", report(used=0.3))
        self.assertEqual(self.statuses()["battery"], OK)
        self.config.battery.enabled = False
        self.assertEqual(self.statuses()["battery"], FAIL)

    def test_the_service(self):
        self.assertEqual(self.statuses(probe=lambda: SERVICE_MISSING)["service"], FAIL)
        self.assertEqual(self.statuses()["service"], OK)

    def test_the_default_probe_is_skipped_away_from_linux_and_looks_for_the_unit_on_linux(self):
        with mock.patch.object(sys, "platform", "win32"):
            self.assertEqual(unattended.default_service_probe()[0], SKIP)
        with mock.patch.object(sys, "platform", "linux"):
            unit = self.dir / "nope.service"
            link = self.dir / "link"
            with mock.patch.object(unattended, "UNIT_PATH", unit), mock.patch.object(unattended, "ENABLED_LINK", link):
                self.assertEqual(unattended.default_service_probe()[0], FAIL)
                unit.write_text("[Unit]")
                status, detail = unattended.default_service_probe()
                self.assertEqual(status, FAIL)
                self.assertIn("not enabled", detail)
                link.write_text("")
                self.assertEqual(unattended.default_service_probe()[0], OK)

    def test_a_skipped_service_is_not_a_failure(self):
        self.write("hardware-check-20261001-100000.json", report())
        items = unattended.evaluate(self.config, self.dir, service_probe=lambda: (SKIP, "not checked"))
        self.assertEqual(unattended.unmet(items), [])

    def test_autonomous_driving_watchdog_and_budgets_must_be_on(self):
        self.config.robot.allow_autonomous_driving = False
        self.assertEqual(self.statuses()["allow_autonomous_driving"], FAIL)
        self.config.robot.watchdog_timeout_s = 0
        self.assertEqual(self.statuses()["watchdog"], FAIL)
        self.config.robot.max_autonomous_drive_s_per_hour = 0
        self.assertEqual(self.statuses()["driving"], FAIL)
        self.config.robot.max_autonomous_drive_s_per_hour = 100
        self.config.robot.max_autonomous_drive_s_per_day = 0
        self.assertEqual(self.statuses()["driving"], FAIL)

    def test_house_rules_are_counted(self):
        self.assertIn("nothing is off limits", self.evaluate()["house"].detail)
        self.config.house.no_go_rooms = ["bagno"]
        item = self.evaluate(taught_rooms={"bedroom", "bathroom"})["house"]
        self.assertEqual(item.status, INFO)
        self.assertIn("2 no-go room(s)", item.detail)


class TaughtRoomsTests(ChecklistCase):
    def test_rooms_come_from_the_rule_facts_of_the_store(self):
        path = self.dir / DB_NAME
        store = Store(path)
        for fact_id, text, subject, kind in (
            ("f1", "Non entrare in camera da letto", "camera da letto", "rule"),
            ("f2", "Non toccare i giocattoli", "giocattoli", "rule"),
            ("f3", "Non andare in cucina", "cucina", "general"),
        ):
            store.put_fact({"id": fact_id, "text": text, "subject": subject, "kind": kind, "source": "user",
                            "created": 1.0, "updated": 1.0, "confidence": 1.0})
        self.assertTrue(store.flush())
        try:
            self.assertEqual(unattended.taught_no_go_rooms(path), {"bedroom"})
        finally:
            store.close()

    def test_a_missing_store_gives_none(self):
        self.assertEqual(unattended.taught_no_go_rooms(self.dir / "nope.db"), set())


class CommandTests(ChecklistCase):
    def run_check(self, config_text=""):
        config_path = self.dir / "laika.toml"
        config_path.write_text(config_text, encoding="utf-8")
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"LAIKA_DATA_DIR": str(self.dir)}), contextlib.redirect_stdout(out):
            code = app.main(["--unattended-check", "--config", str(config_path)])
        return code, out.getvalue()

    def test_the_exit_code_follows_the_checklist(self):
        with mock.patch.object(unattended, "default_service_probe", return_value=SERVICE_OK):
            code, text = self.run_check()
            self.assertEqual(code, 1)
            self.assertIn("[FAIL] hardware check passed on the robot", text)
            self.write("hardware-check-20261001-100000.json", report(offset=0.2))
            code, text = self.run_check("[robot]\nallow_autonomous_driving = true\n")
            self.assertEqual(code, 0, text)
            self.assertIn("All required items are met", text)

    def test_it_never_starts_the_robot(self):
        with mock.patch.object(unattended, "default_service_probe", return_value=SERVICE_OK):
            with mock.patch("laika.app.SimulatedRobot") as sim, mock.patch("laika.app.SafetyController") as safety:
                self.run_check()
        sim.assert_not_called()
        safety.assert_not_called()

    def test_startup_warns_only_when_autonomous_driving_is_on_and_the_checklist_is_not_met(self):
        with mock.patch.object(unattended, "default_service_probe", return_value=SERVICE_OK):
            warning = unattended.startup_warning(self.config, self.dir)
            self.assertIn("--unattended-check", warning)
            self.assertIn("hardware check", warning)
            self.config.robot.allow_autonomous_driving = False
            self.assertIsNone(unattended.startup_warning(self.config, self.dir))
            self.config.robot.allow_autonomous_driving = True
            self.write("hardware-check-20261001-100000.json", report())
            self.assertIsNone(unattended.startup_warning(self.config, self.dir))


if __name__ == "__main__":
    unittest.main()
