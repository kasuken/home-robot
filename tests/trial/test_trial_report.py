"""The field-trial report (#44): counts, loops, supervision, robustness and read-only access."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

from laika import trial_report
from laika.app import main
from laika.events import Event, EventType
from laika.store import DB_NAME
from tests.trial.synthetic import END, START, at, line, write_logs, write_store

ROOT = Path(__file__).resolve().parents[2]


class Case(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def report(self, start=START, end=END, **kwargs):
        return trial_report.build_report(self.dir, start, end, now=end, **kwargs)


class ScenarioCase(Case):
    def setUp(self) -> None:
        super().setUp()
        write_logs(self.dir)
        write_store(self.dir)
        self.r = self.report(quiet_hours="21:30-07:30")


class CoreLoopTests(ScenarioCase):
    def test_sees_and_understands(self):
        loop = self.r["core_loop"]
        self.assertEqual(loop["sees"]["images"], 4)
        self.assertEqual(loop["sees"]["images_by_reason"], {"investigate": 2, "presence": 2})
        self.assertEqual(loop["sees"]["observations_stored"], 4)
        self.assertEqual(loop["understands"]["observations_created"], 1)
        self.assertEqual(loop["understands"]["distinct_objects"], 2)
        self.assertIsNone(loop["understands"]["rejections_logged"])
        self.assertEqual(loop["understands"]["frames_with_people"], 1)

    def test_notices(self):
        notices = self.r["core_loop"]["notices"]
        self.assertEqual(notices["interesting_changes"], 1)
        self.assertEqual(notices["reasons"], {"new": 1})
        self.assertEqual(notices["change_episodes"], {"moved": 1, "new": 1, "missing": 0})
        self.assertEqual(notices["changes_in_observations"], {"new": 1})

    def test_acts(self):
        acts = self.r["core_loop"]["acts"]
        self.assertEqual(acts["moves"], {"started": 2, "completed": 1, "failed": 3})
        self.assertEqual(acts["failed_by_reason"], {"obstacle": 1, "preempted": 1, "watchdog": 1})
        self.assertEqual(acts["investigation_pictures"], 2)
        self.assertEqual(acts["obstacles"], 1)

    def test_asks_answers_and_remembers(self):
        core = self.r["core_loop"]
        self.assertEqual(core["asks"]["questions_queued"], 4)
        self.assertEqual(core["asks"]["proactive_asks"], 3)
        self.assertEqual(core["human_answers"]["answers_in_proactive_asks"], 1)
        self.assertEqual(core["human_answers"]["answer_episodes"], 2)
        self.assertEqual(core["remembers"]["facts_created"], 3)
        known = {o["name"]: o["explained"] for o in core["remembers"]["objects_became_known"]}
        self.assertEqual(known, {"backpack": True, "cup": False})  # "sofa" was known before the period


class LoopTests(ScenarioCase):
    def test_completed_loops_are_chained(self):
        done = {c["object"]: c for c in self.r["loops"]["completed"]}
        self.assertEqual(set(done), {"backpack", "lamp"})
        backpack = done["backpack"]
        self.assertEqual(backpack["question_id"], "q-1")
        self.assertEqual(backpack["via"], "laika asked")
        self.assertTrue(backpack["fact"] and backpack["remembered"])
        self.assertTrue(backpack["queued"] < backpack["asked"] < backpack["answered"])
        self.assertEqual(backpack["queued_to_answer_s"], 84 * 60 + 30)
        # question ids restart with every run: the second run's q-1 is the lamp, not the backpack
        self.assertEqual(done["lamp"]["question_id"], "q-1")
        self.assertEqual(done["lamp"]["via"], "asked in a conversation (inferred)")

    def test_open_loops_have_an_age(self):
        asked = self.r["loops"]["open_asked_unanswered"]
        never = self.r["loops"]["open_queued_never_asked"]
        self.assertEqual([o["object"] for o in asked], ["toy"])
        self.assertEqual([o["object"] for o in never], ["vase"])
        self.assertEqual(asked[0]["ask_count"], 1)
        self.assertAlmostEqual(never[0]["age_s"], END - at(0, 8, 20), places=0)
        self.assertTrue(asked[0]["likely_lost"] and never[0]["likely_lost"])  # restarts since

    def test_an_ask_with_no_queued_question_is_listed(self):
        orphans = self.r["loops"]["asked_without_queue_record"]
        self.assertEqual([o["question_id"] for o in orphans], ["q-9"])

    def test_a_loop_answered_before_the_period_is_not_counted_again(self):
        later = self.report(start=at(1, 0))
        self.assertEqual([c["object"] for c in later["loops"]["completed"]], ["lamp"])

    def test_answers_without_a_question_are_counted(self):
        self.assertEqual(self.r["loops"]["answers_without_queue"], 0)


class SupervisionTests(ScenarioCase):
    def test_safety_stops(self):
        sup = self.r["supervision"]
        self.assertEqual(sup["safety_stops_by_reason"], {"obstacle": 1, "preempted": 1, "watchdog": 1})
        self.assertEqual(sup["attention_stops"], 2)  # preempted is normal traffic
        self.assertEqual(len(sup["local_watchdog_expired"]), 1)
        self.assertEqual(sup["local_watchdog_expired"][0]["reason"], "monitor_stalled")
        self.assertEqual(sup["hardware_unavailable"][0]["device"], "camera")

    def test_ai_outage_durations(self):
        outages = self.r["supervision"]["ai_outages"]
        self.assertEqual([o["ended"] for o in outages],
                         ["recovered", "ended by a restart", "still down at the end of the period"])
        self.assertEqual(outages[0]["duration_s"], 25 * 60)
        self.assertEqual(outages[1]["duration_s"], 8 * 3600)
        self.assertEqual(outages[2]["duration_s"], 8 * 3600)
        self.assertEqual(self.r["supervision"]["ai_outage_total_s"], 25 * 60 + 16 * 3600)

    def test_restarts_and_interventions(self):
        sup = self.r["supervision"]
        self.assertEqual(sup["hardware_ready"], 3)
        self.assertEqual(len(sup["restarts"]), 2)  # the first start is not a restart
        self.assertTrue(sup["restarts"][1]["ai_down"])
        self.assertGreater(sup["restarts"][0]["silent_before_s"], 3600)
        text = "\n".join(sup["possible_interventions"])
        self.assertIn("watchdog", text)
        self.assertIn("restart", text)
        self.assertIn("hardware unavailable (camera)", text)

    def test_silent_gaps_are_listed(self):
        self.assertTrue(any(g["duration_s"] > 3600 for g in self.r["supervision"]["silent_gaps"]))


class ConversationTests(ScenarioCase):
    def test_who_started_and_the_answered_rate(self):
        conv = self.r["conversations"]
        self.assertEqual(conv["user_initiated"], 2)
        self.assertEqual(conv["laika_initiated"], 3)
        self.assertEqual(conv["laika_answered"], 1)
        self.assertAlmostEqual(conv["answered_rate"], 1 / 3, places=3)

    def test_quiet_hours_and_do_not_disturb(self):
        dnd = self.r["conversations"]["do_not_disturb"]
        self.assertEqual(dnd["configured"], "21:30-07:30")
        self.assertIn("2026-10-12T22:00", dnd["do_not_disturb_until"])

    def test_asks_inside_quiet_hours_are_counted(self):
        conv = self.r["conversations"]
        self.assertEqual([a["question_id"] for a in conv["asks"]], ["q-1", "q-3", "q-9"])
        self.assertEqual(conv["asks_in_quiet_hours"], 1)  # q-9 at 07:00, quiet hours end at 07:30
        self.assertIn("inside quiet hours", trial_report.render_markdown(self.r))
        self.assertIsNone(self.report()["conversations"]["asks_in_quiet_hours"])  # no quiet hours given
        self.assertEqual(self.report(quiet_hours="")["conversations"]["asks_in_quiet_hours"], None)
        self.assertEqual(self.report(quiet_hours="21:30-07:30")["conversations"]["asks_in_quiet_hours"], 1)
        self.assertEqual(self.report(quiet_hours="12:00-13:00")["conversations"]["asks_in_quiet_hours"], 0)
        self.assertIsNone(self.report(quiet_hours="garbage")["conversations"]["asks_in_quiet_hours"])

    def test_an_interaction_that_never_completed_is_noticed(self):
        extra = line(EventType.INTERACTION_STARTED, at(2, 12), initiator="user")
        with (self.dir / "logs" / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("\n" + extra + "\n")
        self.assertEqual(self.report()["conversations"]["started_not_completed"], {"user": 1})


class PerDayTests(ScenarioCase):
    def test_days_are_split(self):
        days = {d["date"]: d for d in self.r["per_day"]}
        self.assertEqual(list(days), ["2026-10-10", "2026-10-11", "2026-10-12"])
        self.assertEqual(days["2026-10-10"]["images"], 3)
        self.assertEqual(days["2026-10-12"]["images"], 1)
        self.assertEqual(days["2026-10-10"]["moves"], 2)
        self.assertEqual(days["2026-10-10"].get("restarts", 0), 0)
        self.assertEqual(days["2026-10-11"]["restarts"], 1)
        self.assertEqual(days["2026-10-12"]["restarts"], 1)
        self.assertEqual(days["2026-10-10"]["loops_completed"], 1)
        self.assertEqual(days["2026-10-11"]["loops_completed"], 1)
        self.assertEqual(days["2026-10-10"]["asks"], 2)
        self.assertEqual(days["2026-10-11"]["watchdog"], 1)
        self.assertEqual(days["2026-10-10"]["outage_s"], 25 * 60)

    def test_the_markdown_has_a_per_day_section_only_for_several_days(self):
        self.assertIn("## Per day", trial_report.render_markdown(self.r))
        one = self.report(start=at(1, 0), end=at(2, 0) - 1)
        self.assertNotIn("## Per day", trial_report.render_markdown(one))


class RobustnessTests(Case):
    def test_bad_lines_are_skipped_and_counted(self):
        write_logs(self.dir)
        sources = self.report()["sources"]
        self.assertEqual(sources["bad_lines"], 3)  # not JSON, no timestamp, cut off
        self.assertEqual(sources["unknown_event_types"], {"FutureThing": 1})
        self.assertEqual(sources["event_files"], ["events.jsonl.1", "events.jsonl"])
        self.assertFalse(sources["log_rotated_past_start"])

    def test_rotated_files_are_read_in_order(self):
        logs = self.dir / "logs"
        logs.mkdir()
        for name, hour in (("events.jsonl.2", 8), ("events.jsonl.1", 9), ("events.jsonl", 10)):
            (logs / name).write_text(line(EventType.IMAGE_CAPTURED, at(0, hour), reason="x") + "\n", encoding="utf-8")
        loaded = trial_report.load_events(logs)
        self.assertEqual([e.ts for e in loaded.events], [at(0, 8), at(0, 9), at(0, 10)])
        self.assertEqual(loaded.files, ["events.jsonl.2", "events.jsonl.1", "events.jsonl"])

    def test_a_missing_database_only_removes_the_store_numbers(self):
        write_logs(self.dir)
        report = self.report()
        self.assertFalse(report["sources"]["store_available"])
        self.assertIn("no database", report["sources"]["store_note"])
        self.assertEqual(report["core_loop"]["sees"]["images"], 4)
        self.assertIsNone(report["core_loop"]["asks"]["questions_queued"])
        self.assertEqual(report["loops"]["completed"], [])
        self.assertIn("Warning: no database", trial_report.render_markdown(report))
        self.assertFalse((self.dir / "memory").exists(), "reading must not create the store")

    def test_a_broken_database_is_reported_not_raised(self):
        write_logs(self.dir)
        (self.dir / "memory").mkdir()
        (self.dir / "memory" / DB_NAME).write_bytes(b"this is not a database" * 50)
        report = self.report()
        self.assertEqual(report["core_loop"]["sees"]["images"], 4)
        self.assertFalse(report["sources"]["store_available"] and report["loops"]["completed"])

    def test_an_old_store_without_tables_is_tolerated(self):
        write_logs(self.dir)
        (self.dir / "memory").mkdir()
        sqlite3.connect(self.dir / "memory" / DB_NAME).close()
        report = self.report()
        self.assertTrue(report["sources"]["store_errors"])
        self.assertEqual(report["loops"]["completed"], [])

    def test_an_empty_data_dir_gives_an_empty_report(self):
        (self.dir / "logs").mkdir()
        report = self.report()
        self.assertEqual(report["core_loop"]["sees"]["images"], 0)
        self.assertIn("# Laika field-trial report", trial_report.render_markdown(report))

    def test_a_log_that_rotated_past_the_start_warns(self):
        write_logs(self.dir)
        (self.dir / "logs" / "events.jsonl.1").unlink()
        (self.dir / "logs" / "events.jsonl.2").write_text("", encoding="utf-8")
        report = self.report(start=at(0, 0))
        self.assertTrue(report["sources"]["log_rotated_past_start"])
        self.assertIn("rotated away", trial_report.render_markdown(report))

    def test_non_event_shapes_are_bad_lines(self):
        logs = self.dir / "logs"
        logs.mkdir()
        (logs / "events.jsonl").write_text(
            "\n".join(["[]", "42", '{"type": 5, "timestamp": 1}', '{"type": "ImageCaptured", "timestamp": "x"}',
                       '{"type": "ImageCaptured", "timestamp": NaN}', "", line(EventType.IMAGE_CAPTURED, at(0, 8))]),
            encoding="utf-8",
        )
        loaded = trial_report.load_events(logs)
        self.assertEqual((len(loaded.events), loaded.bad_lines), (1, 5))


class PrivacyTests(ScenarioCase):
    def test_nothing_that_was_said_reaches_the_report(self):
        text = json.dumps(self.r) + trial_report.render_markdown(self.r)
        self.assertNotIn("SECRET", text)
        self.assertIn("no transcripts", trial_report.render_markdown(self.r).split("\n\n")[2])


class ReadOnlyTests(Case):
    def fingerprint(self, path: Path):
        return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns

    def test_the_store_file_is_not_touched(self):
        write_logs(self.dir)
        write_store(self.dir)
        db = self.dir / "memory" / DB_NAME
        before = self.fingerprint(db)
        time.sleep(0.05)
        report = self.report()
        self.assertTrue(report["sources"]["store_available"])
        self.assertEqual(self.fingerprint(db), before)
        names = {p.name for p in (self.dir / "memory").iterdir()}
        self.assertLessEqual(names, {DB_NAME, f"{DB_NAME}-shm", f"{DB_NAME}-wal"})

    def test_a_store_that_is_open_in_wal_mode_is_read_and_not_written(self):
        write_logs(self.dir)
        store = write_store(self.dir, keep_open=True)
        self.addCleanup(store.close)
        db = self.dir / "memory" / DB_NAME
        before = self.fingerprint(db)
        report = self.report()
        self.assertEqual(report["core_loop"]["asks"]["questions_queued"], 4)
        self.assertEqual(self.fingerprint(db), before)

    def test_the_connection_cannot_write(self):
        write_store(self.dir)
        conn, _ = trial_report._open_readonly(self.dir / "memory" / DB_NAME)
        self.addCleanup(conn.close)
        with self.assertRaises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE intruder (a)")
        with self.assertRaises(sqlite3.OperationalError):
            conn.execute("DELETE FROM episodes")


class PeriodTests(unittest.TestCase):
    NOW = at(2, 12)

    def test_default_is_the_last_day(self):
        self.assertEqual(trial_report.resolve_period(None, None, self.NOW), (self.NOW - 86400, self.NOW))

    def test_days(self):
        self.assertEqual(trial_report.resolve_period(5, None, self.NOW), (self.NOW - 5 * 86400, self.NOW))

    def test_since_runs_to_now_or_for_n_days(self):
        self.assertEqual(trial_report.resolve_period(None, date(2026, 10, 10), self.NOW), (at(0, 0), self.NOW))
        self.assertEqual(trial_report.resolve_period(1, date(2026, 10, 10), self.NOW), (at(0, 0), at(1, 0)))
        self.assertEqual(trial_report.resolve_period(9, date(2026, 10, 10), self.NOW), (at(0, 0), self.NOW))

    def test_bad_options(self):
        with self.assertRaises(ValueError):
            trial_report.resolve_period(0, None, self.NOW)
        with self.assertRaises(ValueError):
            trial_report.resolve_period(None, date(2026, 11, 1), self.NOW)
        with self.assertRaises(ValueError):
            trial_report.parse_since("yesterday")


def recent_data_dir(root: Path) -> None:
    """A small data dir ending a few minutes ago, for the command-line tests."""
    now = time.time()
    logs = root / "logs"
    logs.mkdir(parents=True)
    events = [
        Event(EventType.HARDWARE_READY, {"robotId": "r", "name": "Laika"}, now - 3600),
        Event(EventType.IMAGE_CAPTURED, {"reason": "presence"}, now - 3000),
        Event(EventType.MOVEMENT_FAILED, {"reason": "obstacle", "result": "cancelled"}, now - 2000),
    ]
    (logs / "events.jsonl").write_text("\n".join(e.to_json() for e in events) + "\ngarbage\n", encoding="utf-8")


class CommandLineTests(Case):
    def run_main(self, *args, data_dir=None):
        env = {"LAIKA_DATA_DIR": str(data_dir or self.dir), "OPENAI_API_KEY": "", "LAIKA_AI_API_KEY": ""}
        out = io.StringIO()
        with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(out):
            code = main(["--config", str(self.dir / "none.toml"), *args])
        return code, out.getvalue()

    def test_report_writes_both_files_and_exits_zero_without_an_ai_key(self):
        recent_data_dir(self.dir)
        code, output = self.run_main("--report")
        self.assertEqual(code, 0)
        today = date.today().isoformat()
        md = self.dir / "logs" / f"trial-report-{today}.md"
        data = self.dir / "logs" / f"trial-report-{today}.json"
        self.assertTrue(md.is_file() and data.is_file())
        self.assertIn("# Laika field-trial report", md.read_text(encoding="utf-8"))
        parsed = json.loads(data.read_text(encoding="utf-8"))
        self.assertEqual(parsed["core_loop"]["sees"]["images"], 1)
        self.assertEqual(parsed["sources"]["bad_lines"], 1)
        self.assertIn("Laika trial report", output)
        self.assertIn(str(md), output)
        self.assertFalse((self.dir / "memory").exists())

    def test_days_and_since(self):
        write_logs(self.dir)
        write_store(self.dir)
        code, output = self.run_main("--report", "--since", "2026-10-10", "--days", "3")
        self.assertEqual(code, 0)
        self.assertIn("completed loops 2", output)
        self.assertIn("3 days", output)

    def test_a_bad_date_or_count_exits_two(self):
        for args in (("--since", "tomorrow"), ("--days", "0"), ("--since", "2999-01-01")):
            code, output = self.run_main("--report", *args)
            self.assertEqual(code, 2, args)
            self.assertIn("[report]", output)

    def test_a_missing_data_dir_exits_two_and_creates_nothing(self):
        missing = self.dir / "nowhere"
        code, output = self.run_main("--report", data_dir=missing)
        self.assertEqual(code, 2)
        self.assertIn("does not exist", output)
        self.assertFalse(missing.exists())

    def test_run_laika_py_is_the_entry_point(self):
        recent_data_dir(self.dir)
        env = {**os.environ, "LAIKA_DATA_DIR": str(self.dir), "OPENAI_API_KEY": "", "PYTHONIOENCODING": "utf-8"}
        env.pop("LAIKA_AI_API_KEY", None)
        done = subprocess.run(
            [sys.executable, str(ROOT / "run-laika.py"), "--config", str(self.dir / "none.toml"), "--report"],
            capture_output=True, text=True, env=env, timeout=60, cwd=self.dir,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("Laika trial report", done.stdout)
        self.assertEqual(len(list((self.dir / "logs").glob("trial-report-*"))), 2)


if __name__ == "__main__":
    unittest.main()
