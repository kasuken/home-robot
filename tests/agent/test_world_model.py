import json
import logging
import tempfile
import threading
import unittest
from pathlib import Path

from laika.camera import Camera
from laika.events import Event, EventBus, EventType
from laika.observations import Observer
from laika.house import Pose, default_house
from laika.simulator import SimulatedRobot
from laika.store import DB_NAME, Store
from laika.world_model import (
    MAX_PREVIOUS_ROOMS,
    KnownObject,
    SUMMARY_MAX_CHARS,
    WorldModel,
    WorldModelPersister,
    WorldModelUpdater,
)
from tests.fakes import FakeClock


class WorldModelTests(unittest.TestCase):
    def test_starts_empty(self):
        world = WorldModel()
        self.assertEqual(world.rooms(), [])
        self.assertIsNone(world.current_room)
        self.assertIsNone(world.where_is("sofa"))
        self.assertEqual(world.objects_in("living_room"), [])
        self.assertIn("not been explored", world.summary())

    def test_observe_records_objects_in_a_room(self):
        world = WorldModel()
        world.observe([" Sofa ", "television"], "living_room", timestamp=10.0)
        sofa = world.where_is("SOFA")
        self.assertEqual((sofa.name, sofa.room, sofa.times_seen), ("sofa", "living_room", 1))
        self.assertEqual((sofa.first_seen, sofa.last_seen, sofa.source), (10.0, 10.0, "vision"))
        self.assertEqual(world.rooms(), ["living_room"])
        self.assertEqual({o.name for o in world.objects_in("Living_Room")}, {"sofa", "television"})

    def test_repeated_sightings_count_and_move_last_seen(self):
        world = WorldModel()
        world.observe(["sofa"], "living_room", timestamp=1.0)
        world.observe(["sofa"], "living_room", timestamp=5.0)
        sofa = world.where_is("sofa")
        self.assertEqual((sofa.times_seen, sofa.first_seen, sofa.last_seen), (2, 1.0, 5.0))

    def test_learn_from_conversation_keeps_the_note(self):
        world = WorldModel()
        world.learn("Guitars", room="living_room", note="belongs to Leonardo", timestamp=3.0)
        guitars = world.where_is("guitars")
        self.assertEqual((guitars.source, guitars.note, guitars.times_seen), ("conversation", "belongs to Leonardo", 0))
        world.observe(["guitars"], "living_room", timestamp=4.0)
        self.assertEqual(world.where_is("guitars").note, "belongs to Leonardo")
        self.assertEqual(world.where_is("guitars").source, "vision")

    def test_moving_an_object_updates_its_room_and_history(self):
        world = WorldModel()
        world.observe(["backpack"], "living_room", timestamp=1.0)
        world.observe(["backpack"], "kitchen", timestamp=2.0)
        backpack = world.where_is("backpack")
        self.assertEqual((backpack.room, backpack.previous_rooms), ("kitchen", ["living_room"]))
        self.assertEqual((backpack.first_seen, backpack.last_seen), (1.0, 2.0))
        self.assertEqual(world.objects_in("living_room"), [])

    def test_location_history_is_bounded(self):
        world = WorldModel()
        rooms = ["a", "b", "c", "d", "e", "f", "g", "h"]
        for n, room in enumerate(rooms):
            world.observe(["cup"], room, timestamp=float(n))
        cup = world.where_is("cup")
        self.assertEqual(cup.room, "h")
        self.assertEqual(len(cup.previous_rooms), MAX_PREVIOUS_ROOMS)
        self.assertEqual(cup.previous_rooms, rooms[-1 - MAX_PREVIOUS_ROOMS : -1])

    def test_an_unknown_room_does_not_erase_the_last_known_one(self):
        world = WorldModel()
        world.observe(["sofa"], "living_room")
        world.observe(["sofa"], None)
        self.assertEqual(world.where_is("sofa").room, "living_room")

    def test_enter_room_tracks_the_current_room(self):
        world = WorldModel()
        world.enter_room("Kitchen")
        world.enter_room("living_room")
        self.assertEqual(world.current_room, "living_room")
        self.assertEqual(world.rooms(), ["kitchen", "living_room"])

    def test_least_recently_seen_objects_are_evicted(self):
        world = WorldModel(max_objects=3)
        world.observe(["a", "b", "c"], "room")
        world.observe(["a"], "room")
        world.observe(["d"], "room")
        self.assertEqual({o.name for o in world.objects_in("room")}, {"a", "c", "d"})

    def test_round_trip_through_json(self):
        world = WorldModel()
        world.enter_room("living_room")
        world.observe(["sofa"], "living_room", timestamp=1.0)
        world.observe(["backpack"], "kitchen", timestamp=2.0)
        world.observe(["backpack"], "living_room", timestamp=3.0)
        world.learn("guitars", room="living_room", note="belongs to Leonardo", timestamp=4.0)

        raw = json.loads(json.dumps(world.to_dict()))
        restored = WorldModel.from_dict(raw)

        self.assertEqual(restored.to_dict(), world.to_dict())
        self.assertEqual(restored.current_room, "living_room")
        self.assertEqual(restored.where_is("backpack").previous_rooms, ["kitchen"])
        self.assertEqual(restored.where_is("guitars").note, "belongs to Leonardo")

    def test_summary_describes_rooms_and_stays_bounded(self):
        world = WorldModel()
        world.enter_room("living_room")
        world.observe(["sofa", "television"], "living_room")
        world.learn("guitars", room="living_room", note="belongs to Leonardo")
        text = world.summary()
        self.assertIn("Laika is in the living room.", text)
        self.assertIn("Living room: ", text)
        self.assertIn("guitars (belongs to Leonardo)", text)

        big = WorldModel(max_objects=1000)
        big.observe([f"object number {n}" for n in range(300)], "office")
        self.assertLessEqual(len(big.summary()), SUMMARY_MAX_CHARS)
        self.assertLessEqual(len(big.summary(max_chars=120)), 120)

    def test_concurrent_updates_keep_the_model_consistent(self):
        world = WorldModel(max_objects=50)

        def work(worker: int) -> None:
            for n in range(200):
                world.observe([f"item {worker}-{n}"], f"room {worker}")
                world.where_is("item 0-0")
                world.summary()

        threads = [threading.Thread(target=work, args=(w,)) for w in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertLessEqual(len(world.to_dict()["objects"]), 50)


class UsualRoomTests(unittest.TestCase):
    def test_an_object_that_never_moved_usually_stays_where_it_is_seen(self):
        world = WorldModel()
        for t in (1.0, 2.0, 3.0):
            world.observe(["sofa"], "living_room", timestamp=t)
        self.assertEqual(world.usual_room("sofa"), "living_room")

    def test_one_sighting_is_not_yet_a_habit(self):
        world = WorldModel()
        world.observe(["sofa"], "living_room", timestamp=1.0)
        self.assertIsNone(world.usual_room("sofa"))

    def test_the_usual_room_is_where_it_was_seen_most_not_where_it_is_now(self):
        world = WorldModel()
        for t in range(5):
            world.observe(["keys"], "kitchen", timestamp=float(t))
        world.observe(["keys"], "office", timestamp=10.0)
        self.assertEqual(world.where_is("keys").room, "office")
        self.assertEqual(world.usual_room("keys"), "kitchen")

    def test_sighting_counts_survive_a_round_trip_and_stay_bounded(self):
        world = WorldModel()
        for i in range(12):
            world.observe(["ball"], f"room_{i}", timestamp=float(i))
        world.observe(["ball"], "room_11", timestamp=20.0)
        counts = world.where_is("ball").room_sightings
        self.assertLessEqual(len(counts), 8)
        self.assertEqual(counts["room_11"], 2)
        restored = KnownObject.from_dict(json.loads(json.dumps(world.where_is("ball").to_dict())))
        self.assertEqual(restored.room_sightings, counts)

    def test_objects_stored_before_the_counts_fall_back_to_their_room_history(self):
        old = {"name": "mug", "room": "kitchen", "first_seen": 1.0, "last_seen": 5.0, "times_seen": 3,
               "previous_rooms": ["kitchen", "office"]}
        world = WorldModel()
        world.load([old], [])
        self.assertEqual(world.usual_room("mug"), "kitchen")


class WorldModelUpdaterTests(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus()
        self.bus.start()
        self.addCleanup(self.bus.close)
        self.world = WorldModel()
        self.updater = WorldModelUpdater(self.world, self.bus)
        self.addCleanup(self.updater.close)
        self.observations: list[Event] = []
        self.bus.subscribe(self.observations.append, EventType.OBSERVATION_CREATED)

    def test_room_and_image_events_update_the_model(self):
        self.bus.emit(EventType.ROOM_ENTERED, room="office", previous="living_room")
        self.bus.emit(
            EventType.OBSERVATION_CREATED,
            location="office", objects=["desk", {"name": "chair", "position": "left"}],
        )
        self.assertTrue(self.bus.drain())
        self.assertEqual(self.world.current_room, "office")
        self.assertEqual({o.name for o in self.world.objects_in("office")}, {"desk", "chair"})

    def test_the_updater_no_longer_publishes_observations(self):
        self.bus.emit(EventType.IMAGE_CAPTURED, scene=["desk"], room="office")
        self.assertTrue(self.bus.drain())
        self.assertEqual(self.observations, [])
        self.assertIsNone(self.world.where_is("desk"))

    def test_observation_without_location_uses_the_tracked_room(self):
        self.world.enter_room("kitchen")
        self.bus.emit(EventType.OBSERVATION_CREATED, location=None, objects=["table"])
        self.assertTrue(self.bus.drain())
        self.assertEqual(self.world.where_is("table").room, "kitchen")

    def test_closed_updater_stops_listening(self):
        self.updater.close()
        self.bus.emit(EventType.OBSERVATION_CREATED, location="kitchen", objects=["table"])
        self.assertTrue(self.bus.drain())
        self.assertIsNone(self.world.where_is("table"))


class SimulatedWorldTests(unittest.TestCase):
    def test_exploring_the_house_fills_the_world_model(self):
        clock = FakeClock()
        bus = EventBus()
        bus.start()
        self.addCleanup(bus.close)
        world = WorldModel()
        updater = WorldModelUpdater(world, bus)
        self.addCleanup(updater.close)
        observed = []
        bus.subscribe(observed.append, EventType.OBSERVATION_CREATED)

        # Facing north in the living room; pan -90 turns the camera to the west.
        robot = SimulatedRobot(
            verbose=False, house=default_house(), start=Pose(250, 200, 90), events=bus, clock=clock
        )
        camera = Camera(robot, events=bus, min_interval_s=0, clock=clock)
        observer = Observer(bus)
        observer.attach(camera)
        self.addCleanup(observer.close)
        world.enter_room(robot.current_room())
        robot.look(-90, 0)
        first = camera.capture()
        observer.wait_for(first)
        robot.look(0, 0)

        robot.drive(50)
        for _ in range(60):
            clock.advance(0.5)
            if robot.current_room() == "office":
                break
        robot.stop()
        robot.look(-40, 0)
        observer.wait_for(camera.capture())

        self.assertTrue(bus.drain())
        self.assertEqual(set(world.rooms()), {"living_room", "office"})
        self.assertEqual(world.current_room, "office")
        self.assertEqual(world.where_is("sofa").room, "living_room")
        self.assertEqual(world.where_is("television").room, "living_room")
        self.assertEqual(world.where_is("desk").room, "office")
        self.assertEqual(world.where_is("chair").room, "office")
        self.assertEqual(len(observed), 2)


class WorldChangeTests(unittest.TestCase):
    def setUp(self):
        self.world = WorldModel(max_objects=2)
        self.changes: list[tuple] = []
        self.world.on_change(lambda *change: self.changes.append(change))

    def test_each_mutation_reports_its_changes_in_order(self):
        self.world.enter_room("kitchen", timestamp=1.0)
        self.world.observe(["cup"], "kitchen", timestamp=2.0)
        self.assertEqual(
            [c[0] for c in self.changes], ["room", "room", "object"]
        )
        self.assertEqual(self.changes[0], ("room", "kitchen", 1.0, 1.0))
        self.assertEqual(self.changes[1], ("room", "kitchen", 1.0, 2.0))
        self.assertEqual(self.changes[2][1].name, "cup")

        self.changes.clear()
        self.world.observe(["pan", "lid"], "kitchen", timestamp=3.0)  # the limit is 2: cup is evicted
        self.assertEqual(self.changes[0], ("room", "kitchen", 1.0, 3.0))
        self.assertEqual([c[0] for c in self.changes[1:]], ["object", "object", "deleted"])
        self.assertEqual(self.changes[-1], ("deleted", "cup"))

    def test_a_reported_object_is_a_copy(self):
        self.world.observe(["sofa"], "living_room", timestamp=1.0)
        self.changes[-1][1].times_seen = 99
        self.assertEqual(self.world.where_is("sofa").times_seen, 1)

    def test_a_seen_room_with_an_older_time_is_not_reported_again(self):
        self.world.enter_room("office", timestamp=5.0)
        self.changes.clear()
        self.world.enter_room("office", timestamp=4.0)
        self.assertEqual(self.changes, [])
        self.assertEqual(self.world.to_dict()["rooms"][0]["last_visit"], 5.0)

    def test_a_failing_listener_does_not_break_the_model_or_the_others(self):
        def broken(*_change):
            raise RuntimeError("listener bug")

        self.world.on_change(broken)
        with self.assertLogs("laika.world_model", logging.ERROR):
            self.world.observe(["sofa"], "living_room", timestamp=1.0)
        self.assertEqual(self.world.where_is("sofa").times_seen, 1)
        self.assertEqual([c[0] for c in self.changes], ["room", "object"])

    def test_an_unsubscribed_listener_is_not_called(self):
        unsubscribe = self.world.on_change(lambda *change: self.fail(change))
        unsubscribe()
        unsubscribe()
        self.world.observe(["sofa"], "living_room", timestamp=1.0)

    def test_load_restores_silently_and_keeps_the_times(self):
        source = WorldModel()
        source.observe(["sofa"], "living_room", timestamp=10.0)
        source.observe(["backpack"], "kitchen", timestamp=11.0)
        source.observe(["backpack"], "office", timestamp=12.0)
        stored = json.loads(json.dumps(source.to_dict()))

        self.changes.clear()
        self.world.load(stored["objects"], stored["rooms"])
        self.assertEqual(self.changes, [])
        self.assertEqual(self.world.where_is("backpack").previous_rooms, ["kitchen"])
        self.assertEqual(self.world.where_is("sofa").last_seen, 10.0)
        self.assertIsNone(self.world.current_room)
        self.assertEqual(self.world.to_dict()["rooms"], stored["rooms"])

    def test_load_skips_unreadable_rows_and_logs_them(self):
        good = {"name": "sofa", "room": "living_room", "first_seen": 1.0, "last_seen": 1.0}
        with self.assertLogs("laika.world_model", logging.WARNING):
            self.world.load([good, {"room": "kitchen"}], [{"name": "kitchen"}, {"name": "office", "first_seen": "x"}])
        self.assertEqual([o.name for o in self.world.objects_in("living_room")], ["sofa"])
        self.assertEqual(self.world.rooms(), [])  # both rooms lack a time: skipped


NOW = 1_800_000_000.0


class WorldPersistenceTests(unittest.TestCase):
    """The model written through the real store, closed and reopened, as after a restart (#33)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "memory" / DB_NAME

    def open_store(self, **kwargs) -> Store:
        store = Store(self.path, clock=lambda: NOW, **kwargs)
        self.addCleanup(store.close)
        return store

    @staticmethod
    def restore(store: Store) -> WorldModel:
        world = WorldModel()
        world.load(*store.load_world())
        return world

    def test_the_model_after_a_restart_matches_the_one_before(self):
        store = self.open_store()
        world = WorldModel()
        WorldModelPersister(world, store)
        world.enter_room("living_room", timestamp=1.0)
        world.observe(["sofa", "television"], "living_room", timestamp=2.0)
        world.observe(["backpack"], "kitchen", timestamp=3.0)
        world.observe(["backpack"], "living_room", timestamp=4.0)  # moved: history is kitchen
        world.learn("guitars", room="living_room", note="belongs to Leonardo", timestamp=5.0)
        world.observe(["sofa"], "living_room", timestamp=6.0)  # second sighting
        world.enter_room("kitchen", timestamp=7.0)
        world.observe(["cup"], "kitchen", timestamp=8.0)
        self.assertTrue(store.flush())
        store.close()

        restored = self.restore(self.open_store())

        before, after = world.to_dict(), restored.to_dict()
        del before["current_room"], after["current_room"]
        self.assertEqual(after, before)
        self.assertIsNone(restored.current_room)
        self.assertEqual(restored.where_is("sofa").times_seen, 2)
        self.assertEqual(restored.where_is("backpack").previous_rooms, ["kitchen"])
        self.assertEqual(restored.where_is("guitars").note, "belongs to Leonardo")
        self.assertEqual(restored.rooms(), ["living_room", "kitchen"])

    def test_sightings_after_a_restart_merge_into_the_restored_objects(self):
        store = self.open_store()
        world = WorldModel()
        WorldModelPersister(world, store)
        world.learn("guitars", room="living_room", note="belongs to Leonardo", timestamp=1.0)
        world.observe(["backpack"], "kitchen", timestamp=2.0)
        self.assertTrue(store.flush())
        store.close()

        second = self.open_store()
        restored = self.restore(second)
        WorldModelPersister(restored, second)
        restored.observe(["guitars"], "living_room", timestamp=10.0)
        restored.observe(["backpack"], "office", timestamp=11.0)
        self.assertTrue(second.flush())
        second.close()

        third = self.restore(self.open_store())
        guitars = third.where_is("guitars")
        self.assertEqual((guitars.times_seen, guitars.note, guitars.source), (1, "belongs to Leonardo", "vision"))
        backpack = third.where_is("backpack")
        self.assertEqual((backpack.room, backpack.previous_rooms, backpack.times_seen), ("office", ["kitchen"], 2))

    def test_eviction_deletes_the_object_from_the_store(self):
        store = self.open_store()
        world = WorldModel(max_objects=3)
        WorldModelPersister(world, store)
        world.observe(["a", "b", "c"], "room", timestamp=1.0)
        world.observe(["d"], "room", timestamp=2.0)  # least recently seen, "a", is evicted
        self.assertTrue(store.flush())
        store.close()

        objects, _ = self.open_store().load_world()
        self.assertEqual({o["name"] for o in objects}, {"b", "c", "d"})

    def test_a_closed_persister_stops_writing(self):
        store = self.open_store()
        world = WorldModel()
        persister = WorldModelPersister(world, store)
        persister.close()
        world.observe(["sofa"], "living_room", timestamp=1.0)
        self.assertTrue(store.flush())
        self.assertEqual(store.load_world(), ([], []))

    def test_restoring_from_a_store_that_is_off_gives_an_empty_model(self):
        blocker = Path(self.path.parent.parent) / "file"
        blocker.write_text("not a directory")
        with self.assertLogs("laika.store", logging.ERROR):
            store = Store(blocker / DB_NAME)
        self.addCleanup(store.close)
        world = self.restore(store)
        WorldModelPersister(world, store)
        world.observe(["sofa"], "living_room", timestamp=1.0)
        self.assertEqual(world.where_is("sofa").times_seen, 1)
        self.assertEqual(world.rooms(), ["living_room"])
