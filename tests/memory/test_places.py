from __future__ import annotations

import contextlib
import io
import tempfile
import time
import unittest
from pathlib import Path

from laika.events import EventBus, EventType
from laika.memory import Memory
from laika.places import MAX_LOOKS, PlaceNotes
from laika.store import Store
from laika.world_model import WorldModel, WorldModelUpdater
from tests.fakes import FakeClock


class PlaceCase(unittest.TestCase):
    def setUp(self):
        out = io.StringIO()
        redirect = contextlib.redirect_stdout(out)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)
        self.stdout = out
        self.dir = Path(tempfile.mkdtemp())
        self.clock = FakeClock(1_800_000_000.0)
        self.bus = EventBus()
        self.bus.start()
        self.addCleanup(self.bus.close)
        self.open()

    def open(self):
        self.store = Store(self.dir / "laika.db", clock=self.clock)
        self.addCleanup(self.store.close)
        self.world = WorldModel()
        self.addCleanup(WorldModelUpdater(self.world, self.bus).close)
        self.memory = Memory(self.store, clock=self.clock)
        self.notes = PlaceNotes(self.bus, self.store, world=self.world, memory=self.memory, clock=self.clock)
        self.addCleanup(self.notes.close)

    def enter(self, room, previous=None):
        self.world.enter_room(room)
        self.bus.emit(EventType.ROOM_ENTERED, room=room, previous=previous, source="simulator")
        self.bus.drain()

    def see(self, room, *objects, description="", people=0):
        self.bus.emit(
            EventType.OBSERVATION_CREATED,
            location=room,
            description=description,
            objects=[{"name": name} for name in objects],
            people=people,
        )
        self.bus.drain()


class NotesTests(PlaceCase):
    def test_a_room_keeps_how_it_looks_and_where_it_leads(self):
        self.enter("living_room")
        self.enter("kitchen", previous="living_room")
        self.see("kitchen", "fridge", description="A small kitchen with a white fridge and a window.")
        kitchen = self.notes.place("kitchen")
        self.assertEqual(kitchen.looks, ["A small kitchen with a white fridge and a window."])
        self.assertEqual(kitchen.neighbours, {"living_room": 1})
        self.assertEqual(self.notes.place("living_room").neighbours, {"kitchen": 1})
        summary = self.notes.summary()
        self.assertIn("Kitchen: looks like: A small kitchen", summary)
        self.assertIn("leads to living room", summary)

    def test_looks_are_deduplicated_and_bounded(self):
        for i in range(MAX_LOOKS + 3):
            self.see("hall", description=f"View {i}")
            self.see("hall", description=f"View {i}")
        self.assertEqual(self.notes.place("hall").looks, [f"View {i}" for i in range(3, MAX_LOOKS + 3)])

    def test_nothing_known_gives_an_empty_summary(self):
        self.assertEqual(self.notes.summary(), "")

    def test_the_notes_survive_a_restart(self):
        self.enter("kitchen", previous="living_room")
        self.see("kitchen", description="Kitchen with a table.")
        self.store.flush()
        self.notes.close()
        self.store.close()
        self.open()
        self.assertEqual(self.notes.place("kitchen").looks, ["Kitchen with a table."])
        self.assertEqual(self.notes.place("kitchen").neighbours, {"living_room": 1})


class SessionTests(PlaceCase):
    def test_an_exploration_becomes_an_explored_episode(self):
        self.world.observe(["sofa"], "living_room")
        self.enter("living_room")
        self.notes.begin()
        self.see("living_room", "sofa", "plant", description="Living room with a sofa.")
        self.enter("kitchen", previous="living_room")
        self.see("kitchen", "lamp", people=1)
        self.clock.advance(540)
        summary = self.notes.end(steps=40, scans=6)
        self.assertEqual(
            summary,
            "Explored the living room and the kitchen for 9 minutes (40 steps): new: plant, lamp; met 1 person",
        )
        [episode] = self.memory.episodes.activities_since(0)
        self.assertEqual(episode.kind, "explored")
        self.assertEqual(episode.location, "kitchen")
        self.assertIn("plant", episode.subjects)
        kitchen = self.notes.place("kitchen")
        self.assertEqual((kitchen.explored, kitchen.last_explored, kitchen.people_met), (1, self.clock(), 1))
        self.assertIn("explored 1x", self.notes.summary())

    def test_an_exploration_that_did_nothing_is_not_recorded(self):
        self.notes.begin()
        self.assertIsNone(self.notes.end(steps=0, scans=0))
        self.assertEqual(self.memory.episodes.activities_since(0), [])

    def test_end_without_begin_is_harmless(self):
        self.assertIsNone(self.notes.end(steps=3, scans=1))


class ExplorerHookTests(unittest.TestCase):
    def test_the_explorer_opens_and_closes_a_session(self):
        from tests.behaviour.test_explore import ExploreCase

        class Case(ExploreCase):
            def runTest(self):
                pass

        case = Case()
        calls = []

        class Spy:
            def begin(self):
                calls.append("begin")

            def end(self, **kwargs):
                calls.append(("end", kwargs["driving"]))

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            explorer = case.build()
            explorer.notes = Spy()
            explorer.start()
            deadline = time.monotonic() + 5
            while explorer.steps < 2 and time.monotonic() < deadline:
                time.sleep(0.02)
            explorer.stop()
            case.doCleanups()
        self.assertEqual(calls, ["begin", ("end", True)])


if __name__ == "__main__":
    unittest.main()
