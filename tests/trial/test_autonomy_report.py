"""The field-trial report shows the unattended limits (#51): pause, driving of the day, refusals, turn-backs."""

import tempfile
import time
import unittest
from collections import Counter
from datetime import date
from pathlib import Path

from laika import trial_report
from laika.autonomy import AutonomyPause, DriveBudget, local_day
from laika.store import DB_NAME, Store


class AutonomyReportTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        (self.dir / "memory").mkdir()
        self.now = time.time()
        store = Store(self.dir / "memory" / DB_NAME)
        try:
            AutonomyPause(store).pause()
            DriveBudget(100, 500, store).add(42.5)
            store.put_episode(
                {"id": "ep-1", "timestamp": self.now - 60, "kind": "turned_back", "summary": "Turned back",
                 "subjects": ["bedroom"], "location": "bedroom"}
            )
            self.assertTrue(store.flush())
        finally:
            store.close()

    def test_the_report_holds_the_limits(self):
        report = trial_report.build_report(self.dir, self.now - 3600, self.now + 60, now=self.now)
        autonomy = report["autonomy"]
        self.assertTrue(autonomy["paused"])
        self.assertIsNotNone(autonomy["paused_since"])
        self.assertEqual(autonomy["drive_day"], {"date": local_day(time.time()), "seconds": 42.5})
        self.assertEqual(autonomy["turned_back_from_no_go_rooms"], 1)
        self.assertEqual(autonomy["refused_by_reason"], {"paused": 0, "drive_budget": 0, "no_go_room": 0})
        text = trial_report.render_markdown(report)
        self.assertIn("## Autonomy limits", text)
        self.assertIn("Pause: ON since", text)
        self.assertIn("Turned back from a no-go room: 1", text)

    def test_refusals_are_counted_by_reason(self):
        data = trial_report.StoreData()
        autonomy = trial_report.build_autonomy(Counter({"paused": 3, "drive_budget": 2, "obstacle": 9}), data, 0, 1)
        self.assertEqual(autonomy["refused_by_reason"], {"paused": 3, "drive_budget": 2, "no_go_room": 0})
        self.assertFalse(autonomy["paused"])
        self.assertIsNone(autonomy["turned_back_from_no_go_rooms"])  # no store to read
        self.assertIsNone(autonomy["drive_day"]["seconds"])


if __name__ == "__main__":
    unittest.main()
