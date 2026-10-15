"""Where the windows are (#48): learned from vision and taught by voice, kept apart by direction."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from laika.camera import Camera
from laika.config import MemoryConfig
from laika.conversation_memory import MemorySelector
from laika.events import EventBus
from laika.house import Pose, default_house
from laika.memory import Memory
from laika.observations import Observer
from laika.store import DB_NAME, Store
from laika.tools import ToolRegistry
from laika.windows import (
    MAX_PER_ROOM,
    WindowRegistry,
    canonical_name,
    is_window_name,
    parse_window_statement,
)
from laika.world_model import WorldModel
from laika.safety import SafetyController
from tests.agent.test_tools import call
from tests.fakes import LookRecordingRobot

NOW = 1_800_000_000.0


class NamesTests(unittest.TestCase):
    def test_what_counts_as_a_window(self):
        for name in ("window", "Windows", "finestra", "finestra", "bay window", "FINESTRE"):
            self.assertTrue(is_window_name(name), name)
        for name in ("windowsill", "sofa", "finestrino", "door"):
            self.assertFalse(is_window_name(name), name)

    def test_names_are_normalised_to_window(self):
        self.assertEqual(canonical_name("finestra"), "window")
        self.assertEqual(canonical_name("Sofa  Bed"), "sofa bed")


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.windows = WindowRegistry(clock=lambda: NOW)

    def test_two_windows_in_a_room_are_told_apart_by_direction(self):
        first = self.windows.note_window("living_room", 60, 5)
        again = self.windows.note_window("living_room", 70, 5)
        other = self.windows.note_window("living_room", -60, 5)
        self.assertEqual(again.id, first.id)
        self.assertNotEqual(other.id, first.id)
        self.assertEqual(len(self.windows.in_room("living_room")), 2)
        self.assertEqual(self.windows.get(first.id).pan, 70)  # the latest sighting corrects the direction

    def test_a_window_with_no_direction_takes_the_first_sighting(self):
        told = self.windows.note_window("kitchen", None, source="user")
        seen = self.windows.note_window("kitchen", 40, 8, source="vision")
        self.assertEqual(seen.id, told.id)
        self.assertEqual((seen.pan, seen.source), (40, "vision"))
        self.assertEqual(len(self.windows), 1)

    def test_a_hint_does_not_overrule_what_the_camera_saw(self):
        seen = self.windows.note_window("kitchen", 40, source="vision")
        hinted = self.windows.note_window("kitchen", 20, source="user")
        self.assertEqual((hinted.id, hinted.pan), (seen.id, 40))

    def test_no_room_no_window(self):
        self.assertIsNone(self.windows.note_window(None, 10))
        self.assertIsNone(self.windows.note_window("  ", 10))

    def test_the_registry_is_bounded_per_room(self):
        for pan in range(-90, 91, 40):
            self.windows.note_window("office", pan)
        self.assertEqual(len(self.windows.in_room("office")), MAX_PER_ROOM)

    def test_windows_survive_a_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / DB_NAME)
            WindowRegistry(store).note_window("living_room", 60, 5, source="vision", when=NOW)
            self.assertTrue(store.flush())
            store.close()
            again = Store(Path(folder) / DB_NAME)
            restored = WindowRegistry(again)
            self.assertEqual(restored.load(), 1)
            (window,) = restored.in_room("living_room")
            self.assertEqual((window.pan, window.tilt), (60, 5))
            again.close()


class VisionTests(unittest.TestCase):
    """Learned from vision: the simulated scan sees the living-room window on the right."""

    def test_a_scan_of_the_living_room_learns_its_window(self):
        bus = EventBus()
        bus.start()
        self.addCleanup(bus.close)
        world = WorldModel()
        robot = LookRecordingRobot(house=default_house(), start=Pose(300, 200, 0))
        world.enter_room(robot.current_room())
        camera = Camera(robot, None, bus, min_interval_s=0, max_per_minute=0, save_images=False)
        observer = Observer(bus, default_location=lambda: world.current_room)
        observer.attach(camera)
        self.addCleanup(observer.close)
        windows = WindowRegistry(bus=bus)
        self.addCleanup(windows.close)
        self.assertEqual(windows.in_room("living_room"), [])
        robot.look(0, 5)
        observer.wait_for(camera.capture("scan"))  # straight ahead: no window
        bus.drain(2.0)
        self.assertEqual(len(windows), 0)
        robot.look(60, 5)
        image = camera.capture("scan")
        self.assertIn("window", image.scene)
        observer.wait_for(image)
        bus.drain(2.0)
        (window,) = windows.in_room("living_room")
        self.assertEqual((window.pan, window.source), (60, "vision"))


class StatementTests(unittest.TestCase):
    def test_where_and_not_how(self):
        self.assertEqual(parse_window_statement("la finestra e dietro il divano"), ("somewhere", None))
        self.assertEqual(parse_window_statement("in soggiorno c e una finestra a sinistra"), ("left", -45.0))
        self.assertEqual(parse_window_statement("c e una finestra a destra"), ("right", 45.0))
        self.assertIsNone(parse_window_statement("la finestra e aperta"))
        self.assertIsNone(parse_window_statement("il divano e dietro il tavolo"))


class VoiceTests(unittest.TestCase):
    """Taught by voice: the memory selector and the remember tool."""

    def setUp(self):
        self.world = WorldModel()
        self.memory = Memory()
        self.windows = WindowRegistry(clock=lambda: NOW)
        self.printed = io.StringIO()

    def select(self, text):
        selector = MemorySelector(
            self.world, self.memory, config=MemoryConfig(), clock=lambda: NOW, windows=self.windows
        )
        selector.begin_turn()
        with contextlib.redirect_stdout(self.printed):
            return selector.select(text)

    def test_the_window_is_behind_the_sofa(self):
        self.world.enter_room("living_room")
        (fact,) = self.select("La finestra è dietro il divano")
        self.assertEqual((fact.kind, fact.subject), ("location", "window living room"))
        (window,) = self.windows.in_room("living_room")
        self.assertEqual((window.source, window.pan), ("user", None))
        self.assertEqual(self.world.where_is("window").room, "living_room")
        self.assertTrue(any(e.kind == "conversation" and "window" in e.subjects for e in self.memory.episodes.recent()))

    def test_a_room_and_a_side_in_the_words(self):
        self.world.enter_room("kitchen")  # she is elsewhere: the words name the room
        self.select("In soggiorno c'è una finestra a sinistra")
        (window,) = self.windows.in_room("living_room")
        self.assertEqual(window.pan, -45.0)
        self.assertEqual(self.windows.in_room("kitchen"), [])

    def test_a_state_of_the_window_is_not_a_place(self):
        self.world.enter_room("living_room")
        self.assertEqual(self.select("La finestra è aperta da stamattina"), [])
        self.assertEqual(len(self.windows), 0)

    def test_without_a_known_room_nothing_is_stored(self):
        self.assertEqual(self.select("La finestra è dietro il divano"), [])

    def test_a_second_window_in_another_room_keeps_the_first_fact(self):
        self.world.enter_room("living_room")
        self.select("La finestra è dietro il divano")
        self.select("In cucina c'è una finestra a destra")
        self.assertEqual({f.subject for f in self.memory.facts.all()}, {"window living room", "window kitchen"})

    def test_the_remember_tool_with_subject_window(self):
        robot = LookRecordingRobot(house=default_house(), start=Pose(300, 200, 0))
        safety = SafetyController(robot, speed=25)
        self.addCleanup(safety.close)
        self.world.enter_room("living_room")
        registry = ToolRegistry(
            robot, safety, world=self.world, memory=self.memory, windows=self.windows, clock=lambda: NOW
        )
        result = registry.run(call("remember", text="La finestra è dietro il divano", subject="window"))
        self.assertTrue(result["ok"], result)
        self.assertIn("living room", result["window"])
        result = registry.run(
            call("remember", text="In cucina c'è una finestra", subject="finestra", kind="location", room="kitchen")
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual(len(self.windows.in_room("kitchen")), 1)
        self.assertEqual(len(self.windows), 2)

    def test_the_remember_tool_needs_a_room_for_a_window(self):
        robot = LookRecordingRobot(house=default_house(), start=Pose(300, 200, 0))
        safety = SafetyController(robot, speed=25)
        self.addCleanup(safety.close)
        registry = ToolRegistry(robot, safety, world=self.world, memory=self.memory, windows=self.windows)
        result = registry.run(call("remember", text="C'è una finestra", subject="window"))
        self.assertFalse(result["ok"])
        self.assertIn("room", result["error"])


if __name__ == "__main__":
    unittest.main()
