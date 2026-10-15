"""The real app pieces run in the simulator for a moment, then the report reads what they wrote."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from laika import trial_report
from laika.camera import Camera
from laika.commands import Direction, MoveCommand
from laika.events import EventBus, EventType
from laika.house import default_house
from laika.memory import EpisodeRecorder, Memory
from laika.observations import ObservationLog, Observer
from laika.safety import SafetyController
from laika.simulator import SimulatedRobot
from laika.storage import DataPaths, EventFileLogger
from laika.store import DB_NAME, ObservationArchiver, Store


class SimulatorToReportTests(unittest.TestCase):
    def test_what_the_app_writes_is_what_the_report_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = DataPaths(Path(tmp)).ensure()
            started = time.time() - 1
            bus = EventBus()
            event_file = EventFileLogger(paths.logs)
            bus.subscribe(event_file)
            bus.start()
            store = Store(paths.memory / DB_NAME)
            archiver = ObservationArchiver(store, bus)
            memory = Memory(store)
            recorder = EpisodeRecorder(memory.episodes, bus, interval_s=0)
            robot = SimulatedRobot(verbose=False, house=default_house(), events=bus)
            safety = SafetyController(robot, speed=25, allow_autonomous=True, events=bus)
            safety.start()
            observer = Observer(bus, ObservationLog())
            camera = Camera(robot, None, bus, min_interval_s=0, max_per_minute=0)
            observer.attach(camera)
            try:
                bus.emit(EventType.HARDWARE_READY, robotId="laika-sim", name="Laika")
                for pan in (-30.0, 0.0, 30.0):
                    robot.look(pan, 0)
                    image = camera.capture("scan")
                    observer.wait_for(image, 3.0)
                safety.execute(MoveCommand(direction=Direction.FORWARD, speed=25, duration_ms=200))
                bus.emit(EventType.INTERACTION_STARTED, initiator="user")
                bus.emit(EventType.INTERACTION_COMPLETED, initiator="user")
            finally:
                safety.close()
                observer.close()
                self.assertTrue(bus.drain(5.0))
                store.flush(5.0)
                recorder.close()
                archiver.close()
                bus.close()
                store.close()
                event_file.close()

            report = trial_report.build_report(Path(tmp), started, time.time() + 1)
            core = report["core_loop"]
            self.assertEqual(core["sees"]["images"], 3)
            self.assertEqual(core["sees"]["images_by_reason"], {"scan": 3})
            self.assertEqual(core["understands"]["observations_created"], 3)
            self.assertGreaterEqual(core["acts"]["moves"]["started"], 1)
            self.assertTrue(report["sources"]["store_available"])
            self.assertEqual(core["sees"]["observations_stored"], 3)
            self.assertGreaterEqual(report["sources"]["store_totals"]["episodes"], 1)
            self.assertEqual(report["conversations"]["user_initiated"], 1)
            self.assertEqual(report["sources"]["bad_lines"], 0)
            self.assertEqual(report["supervision"]["hardware_ready"], 1)
            self.assertEqual(report["supervision"]["restarts"], [])
            self.assertIn("## Core loop", trial_report.render_markdown(report))


if __name__ == "__main__":
    unittest.main()
