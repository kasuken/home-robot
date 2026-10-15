import contextlib
import io
import json
import math
import threading
import unittest

from laika.camera import Camera
from laika.events import EventBus, EventType
from laika.house import Pose, default_house
from laika.observations import (
    InvalidObservation,
    Observation,
    ObservationLog,
    Observer,
    SeenObject,
    observation_from_scene,
    parse_model_observation,
)
from laika.robot import ImageObservation
from laika.safety import SafetyController
from laika.simulator import SimulatedRobot
from laika.tools import ToolRegistry
from laika.world_model import WorldModel, WorldModelUpdater
from tests.agent.test_tools import call
from tests.fakes import FakeClock

PRD_EXAMPLE = {
    "timestamp": "2026-10-08T12:00:00",
    "location": "living_room",
    "description": "A backpack is next to the sofa.",
    "objects": ["backpack", "sofa"],
    "confidence": 0.91,
}

GOOD = {"description": "A backpack next to the sofa.", "objects": ["Backpack", " SOFA "], "confidence": 0.9}


def image(**kwargs):
    kwargs.setdefault("jpeg", b"\xff\xd8\xff\xd9")
    return ImageObservation(width=640, height=480, camera_pan=10.0, camera_tilt=-5.0, **kwargs)


def parse(output, **kwargs):
    return parse_model_observation(output, image=image(), location=kwargs.get("location", "kitchen"))


class SchemaTests(unittest.TestCase):
    def test_prd_example_round_trips(self):
        observation = Observation.from_dict(PRD_EXAMPLE)
        self.assertEqual(observation.object_names, ["backpack", "sofa"])
        self.assertEqual(observation.location, "living_room")
        data = observation.to_dict()
        for key, value in PRD_EXAMPLE.items():
            self.assertEqual(data[key], value)
        self.assertTrue(data["id"].startswith("obs-"))
        self.assertEqual(Observation.from_dict(data), observation.__class__.from_dict(data))
        self.assertEqual(Observation.from_dict(data).to_dict(), data)

    def test_objects_with_details_round_trip_and_json_is_serialisable(self):
        observation = Observation(
            description="A cup.",
            objects=(SeenObject("cup", 0.7, "left", "blue"), SeenObject("table")),
            confidence=0.8,
            location="kitchen",
            people=1,
            timestamp=1_700_000_000.0,
            image_path="/tmp/x.jpg",
            source="simulator",
        )
        data = json.loads(json.dumps(observation.to_dict()))
        self.assertEqual(data["objects"][1], "table")
        self.assertEqual(data["objects"][0]["position"], "left")
        self.assertEqual(Observation.from_dict(data), observation)

    def test_from_dict_accepts_strings_and_objects(self):
        observation = Observation.from_dict({**PRD_EXAMPLE, "objects": ["cup", {"name": "Lamp", "position": "right"}]})
        self.assertEqual(observation.objects[1], SeenObject("lamp", 1.0, "right"))

    def test_from_dict_rejects_bad_data(self):
        with self.assertRaises(InvalidObservation):
            Observation.from_dict({**PRD_EXAMPLE, "timestamp": "yesterday"})
        with self.assertRaises(InvalidObservation):
            Observation.from_dict({**PRD_EXAMPLE, "id": "../x"})
        with self.assertRaises(InvalidObservation):
            Observation.from_dict(["not", "a", "dict"])


class ParseTests(unittest.TestCase):
    def test_accepts_json_text_and_a_dict(self):
        for output in (json.dumps(GOOD), GOOD):
            observation = parse(output)
            self.assertEqual(observation.object_names, ["backpack", "sofa"])
            self.assertEqual(observation.confidence, 0.9)
            self.assertEqual(observation.location, "kitchen")
            self.assertEqual((observation.camera_pan, observation.camera_tilt), (10.0, -5.0))
            self.assertEqual(observation.source, "vision")

    def test_objects_without_a_confidence_inherit_the_observation_s(self):
        observation = parse({**GOOD, "objects": ["sofa", {"name": "lamp", "confidence": 0.3}]})
        confidences = {o.name: o.confidence for o in observation.objects}
        self.assertEqual(confidences, {"sofa": 0.9, "lamp": 0.3})

    def test_accepts_fenced_json(self):
        text = "```json\n" + json.dumps(GOOD) + "\n```"
        self.assertEqual(parse(text).description, GOOD["description"])
        self.assertEqual(parse("```\n" + json.dumps(GOOD) + "\n```").confidence, 0.9)

    def test_names_are_normalised_and_duplicates_dropped(self):
        observation = parse({**GOOD, "objects": ["Sofa", " sofa ", "SOFA", "Coffee   Table", {"name": "sofa", "position": "left"}]})
        self.assertEqual(observation.object_names, ["sofa", "coffee table"])

    def test_object_details_are_kept(self):
        observation = parse({**GOOD, "objects": [{"name": "Cup", "confidence": 0.4, "position": "centre", "note": " on  the table "}]})
        self.assertEqual(observation.objects[0], SeenObject("cup", 0.4, "centre", "on the table"))

    def test_unknown_keys_are_ignored_but_logged_at_debug(self):
        with self.assertLogs("laika.observations", level="DEBUG") as logs:
            observation = parse({**GOOD, "mood": "happy"})
        self.assertEqual(observation.description, GOOD["description"])
        self.assertTrue(any("mood" in line for line in logs.output))

    def test_model_cannot_set_time_or_place(self):
        observation = parse({**GOOD, "timestamp": "2001-01-01T00:00:00", "location": "attic", "id": "obs-1"})
        self.assertEqual(observation.location, "kitchen")
        self.assertNotEqual(observation.id, "obs-1")

    def test_people_count(self):
        self.assertEqual(parse({**GOOD, "people": 2}).people, 2)

    def test_limits_are_inclusive(self):
        parse({**GOOD, "description": "x" * 400, "objects": [f"o{i}" for i in range(20)], "people": 20})
        parse({**GOOD, "confidence": 0})
        parse({**GOOD, "confidence": 1})
        parse({**GOOD, "objects": ["x" * 40]})


class RejectionTests(unittest.TestCase):
    CASES = {
        "not json": "this is a sofa",
        "empty text": "",
        "json array": "[1, 2]",
        "json string": '"hello"',
        "not an object": 42,
        "missing description": {"objects": [], "confidence": 0.5},
        "missing objects": {"description": "x", "confidence": 0.5},
        "missing confidence": {"description": "x", "objects": []},
        "description not a string": {**GOOD, "description": 5},
        "empty description": {**GOOD, "description": "   "},
        "objects not a list": {**GOOD, "objects": "sofa"},
        "object of wrong type": {**GOOD, "objects": [3]},
        "object name not a string": {**GOOD, "objects": [{"name": 3}]},
        "object without a name": {**GOOD, "objects": [{"position": "left"}]},
        "empty object name": {**GOOD, "objects": ["  "]},
        "object name too long": {**GOOD, "objects": ["x" * 41]},
        "confidence as text": {**GOOD, "confidence": "0.9"},
        "confidence too high": {**GOOD, "confidence": 1.01},
        "confidence negative": {**GOOD, "confidence": -0.1},
        "confidence NaN": {**GOOD, "confidence": math.nan},
        "confidence infinite": {**GOOD, "confidence": math.inf},
        "confidence bool": {**GOOD, "confidence": True},
        "object confidence out of range": {**GOOD, "objects": [{"name": "a", "confidence": 2}]},
        "too many objects": {**GOOD, "objects": [f"o{i}" for i in range(21)]},
        "description too long": {**GOOD, "description": "x" * 401},
        "bad position": {**GOOD, "objects": [{"name": "a", "position": "above"}]},
        "people too many": {**GOOD, "people": 21},
        "people negative": {**GOOD, "people": -1},
        "people bool": {**GOOD, "people": True},
        "people float": {**GOOD, "people": 1.5},
        "output too long": "x" * 30_000,
    }

    def test_every_malformed_output_is_rejected(self):
        for label, output in self.CASES.items():
            with self.subTest(label), self.assertRaises(InvalidObservation) as caught:
                parse(output)
            self.assertTrue(str(caught.exception), label)

    def test_invalid_observation_is_a_value_error(self):
        self.assertTrue(issubclass(InvalidObservation, ValueError))

    def test_json_with_nan_literal_is_rejected(self):
        with self.assertRaises(InvalidObservation):
            parse('{"description": "x", "objects": [], "confidence": NaN}')

    def test_deeply_nested_json_is_rejected_not_crashed(self):
        with self.assertRaises(InvalidObservation):
            parse("[" * 5000 + "]" * 5000)


class SceneTests(unittest.TestCase):
    def test_scene_observation(self):
        frame = image(jpeg=None, scene=("Sofa", "television", "anna"), people=("anna",), simulated=True)
        observation = observation_from_scene(frame, "Living_room")
        self.assertEqual(observation.object_names, ["sofa", "television"])
        self.assertEqual(observation.people, 1)
        self.assertEqual(observation.confidence, 1.0)
        self.assertEqual(observation.source, "simulator")
        self.assertEqual(observation.location, "living_room")
        self.assertEqual(observation.description, "I can see sofa, television, anna.")

    def test_empty_scene(self):
        observation = observation_from_scene(image(jpeg=None, simulated=True), None)
        self.assertEqual(observation.objects, ())
        self.assertIn("anything", observation.description)


class ObservationLogTests(unittest.TestCase):
    def test_bounded_and_newest_first(self):
        log = ObservationLog(max_items=3)
        self.assertIsNone(log.last())
        for i in range(5):
            log.add(Observation(description=f"d{i}"))
        self.assertEqual(len(log), 3)
        self.assertEqual([o.description for o in log.recent(2)], ["d4", "d3"])
        self.assertEqual(log.last().description, "d4")

    def test_thread_safe(self):
        log = ObservationLog(max_items=20)

        def work():
            for i in range(200):
                log.add(Observation(description="x"))
                log.recent(3)

        threads = [threading.Thread(target=work) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(log), 20)


def quiet(func, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = func(*args)
    return result, out.getvalue()


class ObserverTestCase(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus()
        self.bus.start()
        self.addCleanup(self.bus.close)
        self.created = []
        self.bus.subscribe(self.created.append, EventType.OBSERVATION_CREATED)

    def observer(self, **kwargs):
        observer = Observer(self.bus, **kwargs)
        self.addCleanup(observer.close)
        return observer


class ObserverTests(ObserverTestCase):
    def test_simulator_capture_publishes_and_logs_an_observation(self):
        robot = SimulatedRobot(verbose=False, house=default_house(), start=Pose(300, 200, 180), events=self.bus)
        camera = Camera(robot, events=self.bus)
        observer = self.observer()
        observer.attach(camera)
        frame = camera.capture()
        observation = observer.wait_for(frame)
        self.assertIsNotNone(observation)
        self.assertIn("sofa", observation.object_names)
        self.assertEqual(observation.location, "living_room")
        self.assertEqual(observer.log.last(), observation)
        self.assertTrue(self.bus.drain())
        (event,) = self.created
        self.assertEqual(event.data["id"], observation.id)
        self.assertEqual(event.data["location"], "living_room")
        self.assertIn("sofa", event.data["objects"])

    def test_real_robot_frames_produce_nothing_without_a_describer(self):
        observer = self.observer()
        observer.submit(image(), "kitchen")
        self.assertIsNone(observer.wait_for(image()))
        self.assertEqual(len(observer.log), 0)
        self.assertTrue(self.bus.drain())
        self.assertEqual(self.created, [])

    def test_describer_output_is_used_and_the_saved_path_attached(self):
        frame = image()
        seen = []

        def describe(img, location):
            seen.append((img, location))
            return parse(GOOD, location=location)

        observer = self.observer(describe=describe)
        observer.submit(frame, "kitchen", "/data/images/a.jpg")
        observation = observer.wait_for(frame)
        self.assertEqual(seen, [(frame, "kitchen")])
        self.assertEqual(observation.image_path, "/data/images/a.jpg")
        self.assertEqual(observer.log.last(), observation)

    def test_describer_that_returns_none_makes_no_observation(self):
        observer = self.observer(describe=lambda img, loc: None)
        frame = image()
        observer.submit(frame, "kitchen")
        self.assertIsNone(observer.wait_for(frame))
        self.assertEqual(len(observer.log), 0)

    def test_malformed_model_output_is_rejected_and_logged(self):
        frame = image()
        observer = self.observer(describe=lambda img, loc: parse({**GOOD, "confidence": 7}, location=loc))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            observer.submit(frame, "kitchen")
            self.assertIsNone(observer.wait_for(frame))
        self.assertIn("[laika] observation rejected: confidence must be between 0 and 1", out.getvalue())
        self.assertEqual(len(observer.rejections), 1)
        self.assertEqual(len(observer.log), 0)
        self.assertTrue(self.bus.drain())
        self.assertEqual(self.created, [])

    def test_every_rejection_case_is_logged_by_the_observer(self):
        outputs = list(RejectionTests.CASES.values())
        observer = self.observer(describe=lambda img, loc: parse(outputs[len(observer.rejections)]))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            for _ in outputs:
                frame = image()
                observer.submit(frame, "kitchen")
                observer.wait_for(frame)
        self.assertEqual(out.getvalue().count("[laika] observation rejected:"), len(outputs))
        self.assertEqual(len(observer.log), 0)
        self.assertEqual(self.created, [])

    def test_a_crashing_describer_does_not_kill_the_worker(self):
        calls_made = []

        def describe(img, loc):
            calls_made.append(1)
            if len(calls_made) == 1:
                raise RuntimeError("boom")
            return parse(GOOD)

        observer = self.observer(describe=describe)
        first, second = image(), image()
        with self.assertLogs("laika.observations", level="ERROR"):
            observer.submit(first, None)
            self.assertIsNone(observer.wait_for(first))
        observer.submit(second, None)
        self.assertIsNotNone(observer.wait_for(second))

    def test_location_falls_back_to_the_default(self):
        observer = self.observer(default_location=lambda: "hall")
        frame = image(jpeg=None, scene=("shoes",), simulated=True)
        observer.submit(frame, None)
        self.assertEqual(observer.wait_for(frame).location, "hall")

    def test_submit_after_close_is_ignored(self):
        observer = self.observer()
        observer.close()
        frame = image(jpeg=None, simulated=True)
        observer.submit(frame, None)
        self.assertIsNone(observer.wait_for(frame, 0.1))


class QueueTests(ObserverTestCase):
    def test_queue_is_bounded_and_drops_the_oldest(self):
        gate = threading.Event()
        started = threading.Event()
        described = []

        def describe(img, loc):
            started.set()
            gate.wait(5)
            described.append(img)
            return parse(GOOD)

        observer = self.observer(describe=describe, queue_size=2)
        self.addCleanup(gate.set)
        frames = [image() for _ in range(5)]
        observer.submit(frames[0], None)
        self.assertTrue(started.wait(5))  # frame 0 is being described, the queue is empty
        for frame in frames[1:]:
            observer.submit(frame, None)
        self.assertEqual(observer.dropped, 2)
        # The dropped frames have no observation, and waiting for them does not hang.
        self.assertIsNone(observer.wait_for(frames[1], 0.1))
        self.assertIsNone(observer.wait_for(frames[2], 0.1))
        gate.set()
        for frame in (frames[0], frames[3], frames[4]):
            self.assertIsNotNone(observer.wait_for(frame, 5))
        self.assertEqual(described, [frames[0], frames[3], frames[4]])

    def test_submit_does_not_wait_for_a_slow_describer(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        observer = self.observer(describe=lambda img, loc: gate.wait(5) and parse(GOOD))
        for _ in range(4):
            observer.submit(image(), None)  # returns immediately
        self.assertIsNone(observer.wait_for(image(), 0.05))

    def test_wait_for_times_out_while_the_frame_is_still_in_progress(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        observer = self.observer(describe=lambda img, loc: gate.wait(5) and parse(GOOD))
        frame = image()
        observer.submit(frame, None)
        self.assertIsNone(observer.wait_for(frame, 0.1))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus()
        self.bus.start()
        self.addCleanup(self.bus.close)
        self.world = WorldModel()
        updater = WorldModelUpdater(self.world, self.bus)
        self.addCleanup(updater.close)
        self.clock = FakeClock()
        self.robot = SimulatedRobot(
            verbose=False, house=default_house(), start=Pose(300, 200, 180), events=self.bus, clock=self.clock
        )
        self.camera = Camera(self.robot, events=self.bus, min_interval_s=0, clock=self.clock)
        self.observer = Observer(self.bus, default_location=lambda: self.world.current_room)
        self.observer.attach(self.camera)
        self.addCleanup(self.observer.close)

    def test_world_model_is_filled_through_observation_created(self):
        self.observer.wait_for(self.camera.capture())
        self.assertTrue(self.bus.drain())
        self.assertEqual(self.world.where_is("sofa").room, "living_room")

    def test_image_events_alone_do_not_update_the_world_model(self):
        self.bus.emit(EventType.IMAGE_CAPTURED, scene=["ghost"], room="attic")
        self.assertTrue(self.bus.drain())
        self.assertIsNone(self.world.where_is("ghost"))

    def test_capture_image_tool_returns_the_description(self):
        safety = SafetyController(self.robot, enabled=True, speed=25)
        self.addCleanup(safety.close)
        safety.start()
        self.world.enter_room("living_room")
        registry = ToolRegistry(
            self.robot, safety, world=self.world, camera=self.camera, observer=self.observer
        )
        result = registry.run(call("capture_image"))
        self.assertTrue(result["ok"])
        self.assertIn("sofa", result["scene"])
        self.assertTrue(result["description"].startswith("I can see"))
        self.assertIn("sofa", result["objects"])
        self.assertEqual(result["confidence"], 1.0)

    def test_capture_image_tool_still_works_without_an_observer(self):
        safety = SafetyController(self.robot, enabled=True, speed=25)
        self.addCleanup(safety.close)
        safety.start()
        registry = ToolRegistry(self.robot, safety, world=self.world, camera=self.camera)
        result = registry.run(call("capture_image"))
        self.assertTrue(result["ok"])
        self.assertNotIn("description", result)


if __name__ == "__main__":
    unittest.main()
