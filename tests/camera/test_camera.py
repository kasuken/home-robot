import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from laika.camera import Camera
from laika.config import CameraConfig, load_config
from laika.events import EventBus, EventType
from laika.house import Pose, default_house
from laika.observations import Observer
from laika.robot import HardwareError, ImageObservation
from laika.safety import SafetyController
from laika.simulator import SimulatedRobot
from laika.storage import ImageStore
from laika.tools import ToolRegistry
from laika.world_model import WorldModel, WorldModelUpdater
from tests.agent.test_tools import call
from tests.fakes import FakeClock
from tests.simulator.test_robot_contract import SpyMouth, fake_sunfounder_modules


class FakeRobot:
    """Returns small fake JPEGs; `fail` makes captures raise."""

    def __init__(self):
        self.captures = 0
        self.fail = False
        self.pose = (12.0, -5.0)

    def capture_image(self):
        if self.fail:
            raise HardwareError("camera unplugged")
        self.captures += 1
        return ImageObservation(
            jpeg=b"\xff\xd8" + bytes([self.captures % 256]) * 20 + b"\xff\xd9",
            width=640,
            height=480,
            camera_pan=self.pose[0],
            camera_tilt=self.pose[1],
            timestamp=1_700_000_000.0 + self.captures,
        )


class CameraTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.clock = FakeClock()
        self.robot = FakeRobot()
        self.bus = EventBus()
        self.bus.start()
        self.addCleanup(self.bus.close)
        self.seen = []
        self.bus.subscribe(self.seen.append, EventType.IMAGE_CAPTURED, EventType.HARDWARE_UNAVAILABLE)

    def camera(self, max_images=200, **kwargs):
        kwargs.setdefault("min_interval_s", 0.5)
        kwargs.setdefault("max_per_minute", 30)
        return Camera(
            self.robot, ImageStore(self.dir, max_images=max_images), self.bus, clock=self.clock, **kwargs
        )

    def events(self, type):
        self.bus.drain()
        return [e for e in self.seen if e.type is type]

    def files(self):
        return sorted(self.dir.glob("img-*.jpg"))


class CaptureTests(CameraTestCase):
    def test_returns_an_observation_with_pose_and_timestamp(self):
        image = self.camera().capture()
        self.assertEqual((image.camera_pan, image.camera_tilt), (12.0, -5.0))
        self.assertGreater(image.timestamp, 0)

    def test_saves_the_jpeg_and_publishes_one_event(self):
        camera = self.camera()
        camera.capture("look_around")
        (path,) = self.files()
        self.assertEqual(camera.last_path, path)
        (event,) = self.events(EventType.IMAGE_CAPTURED)
        self.assertEqual(event.data["path"], str(path))
        self.assertEqual(
            (event.data["width"], event.data["height"], event.data["pan"], event.data["tilt"]),
            (640, 480, 12.0, -5.0),
        )
        self.assertEqual(event.data["reason"], "look_around")
        self.assertNotIn("room", event.data)  # this robot has no current_room()

    def test_save_images_off_keeps_nothing_on_disk(self):
        camera = self.camera(save_images=False)
        camera.capture()
        self.assertEqual(self.files(), [])
        self.assertIsNone(camera.last_path)
        self.assertIsNone(self.events(EventType.IMAGE_CAPTURED)[0].data["path"])

    def test_only_the_latest_observation_is_kept(self):
        camera = self.camera()
        camera.capture()
        self.clock.advance(1)
        second = camera.capture()
        self.assertIs(camera.last, second)

    def test_old_images_are_pruned(self):
        camera = self.camera(max_images=3, max_per_minute=0)
        for _ in range(10):
            self.clock.advance(1)
            camera.capture()
        self.assertEqual(self.robot.captures, 10)
        self.assertEqual(len(self.files()), 3)
        self.assertEqual(max(self.files()).read_bytes()[2], 10)


class LimitTests(CameraTestCase):
    def test_a_request_inside_the_interval_returns_the_previous_frame(self):
        camera = self.camera()
        first = camera.capture()
        self.clock.advance(0.2)
        again = camera.capture()
        self.assertIs(again, first)
        self.assertTrue(camera.reused)
        self.assertEqual(self.robot.captures, 1)
        self.assertEqual(len(self.events(EventType.IMAGE_CAPTURED)), 1)
        self.clock.advance(0.4)
        self.assertIsNot(camera.capture(), first)
        self.assertFalse(camera.reused)
        self.assertEqual(self.robot.captures, 2)

    def test_per_minute_cap_then_it_recovers(self):
        camera = self.camera(max_per_minute=5)
        for _ in range(8):
            self.clock.advance(1)
            camera.capture()
        self.assertEqual(self.robot.captures, 5)
        self.clock.advance(60)
        camera.capture()
        self.assertEqual(self.robot.captures, 6)


class FailureTests(CameraTestCase):
    def test_hardware_error_returns_none_and_reports_once(self):
        camera = self.camera(min_interval_s=0)
        self.robot.fail = True
        for _ in range(3):
            self.assertIsNone(camera.capture())
        (event,) = self.events(EventType.HARDWARE_UNAVAILABLE)
        self.assertIn("unplugged", event.data["error"])
        self.assertEqual(self.events(EventType.IMAGE_CAPTURED), [])

    def test_recovery_re_arms_the_report(self):
        camera = self.camera(min_interval_s=0)
        self.robot.fail = True
        camera.capture()
        self.robot.fail = False
        self.assertIsNotNone(camera.capture())
        self.robot.fail = True
        camera.capture()
        self.assertEqual(len(self.events(EventType.HARDWARE_UNAVAILABLE)), 2)


class OnePublisherTests(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus()
        self.bus.start()
        self.addCleanup(self.bus.close)
        self.seen = []
        self.bus.subscribe(self.seen.append, EventType.IMAGE_CAPTURED)

    def test_simulated_robot_publishes_one_event_with_scene_and_room(self):
        robot = SimulatedRobot(
            verbose=False, house=default_house(), start=Pose(300, 200, 180), events=self.bus
        )
        Camera(robot, events=self.bus).capture()
        self.bus.drain()
        (event,) = self.seen
        self.assertIn("sofa", event.data["scene"])
        self.assertEqual(event.data["room"], "living_room")
        self.assertIsNone(event.data["path"])

    def test_picarx_robot_publishes_one_event_and_saves_a_jpeg(self):
        patcher = mock.patch.dict(sys.modules, fake_sunfounder_modules())
        patcher.start()
        self.addCleanup(patcher.stop)
        from laika.hardware import PiCarXRobot

        robot = PiCarXRobot(mouth=SpyMouth(), image_size=(320, 240), jpeg_quality=70)
        self.addCleanup(robot.close)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        camera = Camera(robot, ImageStore(Path(tmp.name)), self.bus)
        image = camera.capture()
        self.bus.drain()
        self.assertEqual((image.width, image.height), (320, 240))
        (event,) = self.seen
        self.assertEqual((event.data["width"], event.data["height"]), (320, 240))
        self.assertTrue(Path(event.data["path"]).read_bytes().startswith(b"\xff\xd8"))
        self.assertEqual(robot._camera.options["quality"], 70)
        self.assertEqual(robot._camera.configured_size, (320, 240))


class ConfigTests(unittest.TestCase):
    def test_defaults_and_clamping(self):
        config = CameraConfig()
        self.assertEqual((config.image_size(), config.quality()), ((640, 480), 80))
        config = CameraConfig(width=4000, height=3000, jpeg_quality=100)
        self.assertEqual((config.image_size(), config.quality()), ((1280, 960), 95))

    def test_the_example_file_has_a_camera_section(self):
        config = load_config(Path("config/laika.example.toml"))
        self.assertEqual(config.camera.max_per_minute, 30)
        self.assertTrue(config.camera.save_images)


class IntegrationTests(unittest.TestCase):
    def test_world_model_is_populated_through_the_camera_and_tool_has_no_bytes(self):
        bus = EventBus()
        bus.start()
        self.addCleanup(bus.close)
        world = WorldModel()
        updater = WorldModelUpdater(world, bus)
        self.addCleanup(updater.close)
        robot = SimulatedRobot(
            verbose=False, house=default_house(), start=Pose(300, 200, 180), events=bus
        )
        world.enter_room(robot.current_room())
        safety = SafetyController(robot, enabled=False)
        self.addCleanup(safety.close)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        camera = Camera(robot, ImageStore(Path(tmp.name)), bus)
        observer = Observer(bus)
        observer.attach(camera)
        self.addCleanup(observer.close)
        registry = ToolRegistry(robot, safety, world=world, camera=camera, observer=observer)
        result = registry.run(call("capture_image"))
        bus.drain()
        self.assertTrue(result["ok"])
        self.assertIn("sofa", observer.log.last().object_names)
        self.assertIn("saved", result)
        self.assertIsNotNone(world.where_is("sofa"))
        json.dumps(result)  # plain data only
        self.assertFalse(any(isinstance(v, (bytes, bytearray)) for v in result.values()))

    def test_tool_reports_an_unavailable_camera(self):
        robot = SimulatedRobot(verbose=False, house=default_house(), start=Pose(300, 200, 180))
        robot.faults.camera_error = True
        safety = SafetyController(robot, enabled=False)
        self.addCleanup(safety.close)
        result = ToolRegistry(robot, safety, camera=Camera(robot)).run(call("capture_image"))
        self.assertFalse(result["ok"])
        self.assertIn("camera", result["error"])


if __name__ == "__main__":
    unittest.main()
