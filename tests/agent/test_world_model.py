import json
import threading
import unittest

from laika.events import Event, EventBus, EventType
from laika.house import Pose, default_house
from laika.simulator import SimulatedRobot
from laika.world_model import MAX_PREVIOUS_ROOMS, SUMMARY_MAX_CHARS, WorldModel, WorldModelUpdater
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
        self.bus.emit(EventType.IMAGE_CAPTURED, scene=["desk", "chair"], room="office", pan=0, tilt=0)
        self.assertTrue(self.bus.drain())
        self.assertEqual(self.world.current_room, "office")
        self.assertEqual({o.name for o in self.world.objects_in("office")}, {"desk", "chair"})
        self.assertEqual(self.observations[0].data, {"room": "office", "objects": ["desk", "chair"]})

    def test_image_without_room_uses_the_tracked_room(self):
        self.world.enter_room("kitchen")
        self.bus.emit(EventType.IMAGE_CAPTURED, scene=["table"])
        self.assertTrue(self.bus.drain())
        self.assertEqual(self.world.where_is("table").room, "kitchen")

    def test_closed_updater_stops_listening(self):
        self.updater.close()
        self.bus.emit(EventType.IMAGE_CAPTURED, scene=["table"], room="kitchen")
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
        world.enter_room(robot.current_room())
        robot.look(-90, 0)
        robot.capture_image()
        robot.look(0, 0)

        robot.drive(50)
        for _ in range(60):
            clock.advance(0.5)
            if robot.current_room() == "office":
                break
        robot.stop()
        robot.look(-40, 0)
        robot.capture_image()

        self.assertTrue(bus.drain())
        self.assertEqual(set(world.rooms()), {"living_room", "office"})
        self.assertEqual(world.current_room, "office")
        self.assertEqual(world.where_is("sofa").room, "living_room")
        self.assertEqual(world.where_is("television").room, "living_room")
        self.assertEqual(world.where_is("desk").room, "office")
        self.assertEqual(world.where_is("chair").room, "office")
        self.assertEqual(len(observed), 2)
