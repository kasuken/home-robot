import threading
import time
import tempfile
import unittest
from pathlib import Path

from laika.camera import Camera
from laika.capture_policy import CapturePolicy
from laika.commands import Direction, MoveCommand
from laika.config import CaptureConfig, load_config
from laika.events import EventBus, EventType
from laika.house import Pose, default_house
from laika.observations import Observer
from laika.safety import SafetyController
from laika.simulator import SimulatedRobot
from laika.tools import ToolRegistry
from laika.world_model import WorldModel, WorldModelUpdater
from tests.agent.test_tools import call
from tests.camera.test_camera import FakeRobot
from tests.fakes import FakeClock, settled_events

TERMINAL = {EventType.MOVEMENT_COMPLETED, EventType.MOVEMENT_FAILED}


class GatedRobot(FakeRobot):
    """A camera that waits for `gate` before returning, so a test can hold the worker busy."""

    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.gate = threading.Event()
        self.gate.set()

    def capture_image(self):
        self.entered.set()
        self.gate.wait(5)
        return super().capture_image()


def wait_until(predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class PolicyTestCase(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.robot = FakeRobot()
        self.bus = EventBus()
        self.bus.start()
        self.addCleanup(self.bus.close)
        self.seen = []
        self.bus.subscribe(self.seen.append, EventType.IMAGE_CAPTURED)
        # The camera's own limits are off here, so the policy's limits are what the tests measure.
        self.camera = Camera(
            self.robot, None, self.bus, min_interval_s=0, max_per_minute=0, save_images=False,
            clock=self.clock,
        )

    def policy(self, clock=None, camera=None, **overrides):
        config = CaptureConfig(**overrides)
        policy = CapturePolicy(camera or self.camera, self.bus, config, clock=clock or self.clock)
        self.addCleanup(policy.close)
        policy.start()
        self.policy_ = policy
        return policy

    def settle(self):
        """Deliver events, wait for the policy's worker, then deliver the ImageCaptured events it caused."""
        self.bus.drain()
        self.policy_.flush()
        self.bus.drain()

    def reasons(self):
        return [e.data["reason"] for e in self.seen if e.type is EventType.IMAGE_CAPTURED]

    def fire(self, event_type, **data):
        self.bus.emit(event_type, **data)
        self.settle()


class TriggerTests(PolicyTestCase):
    def test_each_trigger_captures_with_its_reason(self):
        policy = self.policy(after_movement_cooldown_s=0, person_cooldown_s=0, interesting_cooldown_s=0)
        self.fire(EventType.MOVEMENT_COMPLETED, commandId="c1", result="success")
        self.fire(EventType.PERSON_DETECTED, name="anna", room="kitchen")
        self.fire(EventType.INTERESTING_CHANGE_DETECTED, what="door")
        policy.investigate()
        self.settle()
        self.assertEqual(self.reasons(), ["after_movement", "person", "interesting", "investigate"])
        self.assertEqual(policy.stats["after_movement"].captured, 1)

    def test_disabled_triggers_do_not_capture(self):
        policy = self.policy(after_movement=False, on_person=False, on_interesting=False)
        self.fire(EventType.MOVEMENT_COMPLETED, commandId="c1", result="success")
        self.fire(EventType.PERSON_DETECTED, name="anna", room="kitchen")
        self.fire(EventType.INTERESTING_CHANGE_DETECTED, what="door")
        self.assertEqual(self.reasons(), [])
        self.assertEqual(
            [policy.stats[name].disabled for name in ("after_movement", "person", "interesting")], [1, 1, 1]
        )
        # Investigation has no switch of its own: it still captures.
        policy.investigate()
        self.settle()
        self.assertEqual(self.reasons(), ["investigate"])

    def test_failed_moves_do_not_trigger_a_capture(self):
        self.policy()
        self.fire(EventType.MOVEMENT_FAILED, commandId="c1", result="obstacle", reason="obstacle")
        self.assertEqual(self.reasons(), [])


class CooldownTests(PolicyTestCase):
    def test_each_trigger_has_its_own_cooldown(self):
        policy = self.policy(after_movement_cooldown_s=10, person_cooldown_s=30)
        self.fire(EventType.MOVEMENT_COMPLETED, commandId="c1", result="success")
        self.clock.advance(5)
        self.fire(EventType.MOVEMENT_COMPLETED, commandId="c2", result="success")  # inside its cooldown
        self.fire(EventType.PERSON_DETECTED, name="anna", room="kitchen")  # other trigger, not blocked
        self.assertEqual(self.reasons(), ["after_movement", "person"])
        self.assertEqual(policy.stats["after_movement"].skipped_cooldown, 1)
        self.clock.advance(6)  # 11 s after the first capture
        self.fire(EventType.MOVEMENT_COMPLETED, commandId="c3", result="success")
        self.assertEqual(self.reasons(), ["after_movement", "person", "after_movement"])

    def test_person_cooldown_is_separate_from_movement(self):
        self.policy(after_movement_cooldown_s=0, person_cooldown_s=30)
        self.fire(EventType.PERSON_DETECTED, name="anna", room="kitchen")
        self.clock.advance(29)
        self.fire(EventType.PERSON_DETECTED, name="anna", room="kitchen")
        self.assertEqual(self.reasons(), ["person"])
        self.clock.advance(1)
        self.fire(EventType.PERSON_DETECTED, name="anna", room="kitchen")
        self.assertEqual(self.reasons(), ["person", "person"])


class RateTests(PolicyTestCase):
    def test_one_per_minute_cap_is_shared_by_all_automatic_triggers(self):
        policy = self.policy(max_auto_per_minute=3, after_movement_cooldown_s=0, person_cooldown_s=0,
                             interesting_cooldown_s=0)
        self.fire(EventType.MOVEMENT_COMPLETED, commandId="c1", result="success")
        self.fire(EventType.PERSON_DETECTED, name="anna", room="kitchen")
        self.fire(EventType.INTERESTING_CHANGE_DETECTED, what="door")
        policy.investigate()  # fourth automatic capture inside the minute
        self.settle()
        self.assertEqual(self.reasons(), ["after_movement", "person", "interesting"])
        self.assertEqual(policy.stats["investigate"].skipped_rate, 1)
        self.clock.advance(60)
        policy.investigate()
        self.settle()
        self.assertEqual(self.reasons()[-1], "investigate")

    def test_request_captures_are_not_counted_against_the_cap(self):
        policy = self.policy(max_auto_per_minute=1)
        for _ in range(3):
            self.camera.capture("request")
        self.settle()
        self.assertEqual(self.reasons(), ["request"] * 3)
        self.fire(EventType.MOVEMENT_COMPLETED, commandId="c1", result="success")
        self.assertEqual(policy.stats["after_movement"].captured, 1)

    def test_a_camera_that_hands_back_the_previous_frame_is_not_a_capture(self):
        camera = Camera(
            self.robot, None, self.bus, min_interval_s=0.5, max_per_minute=0, save_images=False,
            clock=self.clock,
        )
        policy = self.policy(camera=camera)
        camera.capture("request")  # the camera's own interval now applies
        self.fire(EventType.MOVEMENT_COMPLETED, commandId="c1", result="success")
        self.assertEqual(self.robot.captures, 1)
        self.assertEqual(policy.stats["after_movement"].reused, 1)
        self.assertEqual(policy.stats["after_movement"].captured, 0)

    def test_a_failed_camera_uses_no_budget_and_no_cooldown(self):
        self.robot.fail = True
        policy = self.policy(max_auto_per_minute=1, after_movement_cooldown_s=30)
        self.fire(EventType.MOVEMENT_COMPLETED, commandId="c1", result="success")
        self.assertEqual(policy.stats["after_movement"].failed, 1)
        self.robot.fail = False
        self.fire(EventType.MOVEMENT_COMPLETED, commandId="c2", result="success")
        self.assertEqual(self.reasons(), ["after_movement"])


class BurstTests(PolicyTestCase):
    def test_a_burst_leaves_at_most_one_pending_trigger(self):
        robot = GatedRobot()
        self.robot = robot
        self.camera = Camera(robot, None, self.bus, min_interval_s=0, max_per_minute=0,
                             save_images=False, clock=self.clock)
        policy = self.policy(after_movement_cooldown_s=0, max_auto_per_minute=60)
        robot.gate.clear()
        self.bus.emit(EventType.MOVEMENT_COMPLETED, commandId="first", result="success")
        self.assertTrue(robot.entered.wait(5))  # the worker is now inside the camera
        for n in range(50):
            self.bus.emit(EventType.MOVEMENT_COMPLETED, commandId=f"c{n}", result="success")
        self.bus.drain()
        self.assertEqual(policy.stats["after_movement"].replaced, 49)
        robot.gate.set()
        self.settle()
        self.assertEqual(policy.stats["after_movement"].captured, 2)  # the running one and the last pending
        self.assertEqual(robot.captures, 2)


class ExplorationTests(PolicyTestCase):
    def test_timer_captures_while_exploring_and_stops_when_not(self):
        policy = self.policy(clock=time.monotonic, exploration_interval_s=0.05)
        policy.set_exploring(True)
        self.assertTrue(wait_until(lambda: policy.stats["exploration"].captured >= 2))
        policy.set_exploring(False)
        self.assertTrue(policy.flush())
        stopped_at = policy.stats["exploration"].captured
        time.sleep(0.3)
        self.assertEqual(policy.stats["exploration"].captured, stopped_at)

    def test_no_exploration_captures_unless_asked(self):
        policy = self.policy(clock=time.monotonic, exploration_interval_s=0.05)
        time.sleep(0.2)
        self.assertEqual(policy.stats["exploration"].captured, 0)

    def test_exploration_respects_the_per_minute_cap(self):
        policy = self.policy(clock=time.monotonic, exploration_interval_s=0.05, max_auto_per_minute=2)
        policy.set_exploring(True)
        self.assertTrue(wait_until(lambda: policy.stats["exploration"].skipped_rate >= 1))
        policy.set_exploring(False)
        self.assertTrue(policy.flush())
        self.assertEqual(policy.stats["exploration"].captured, 2)


class LifecycleTests(PolicyTestCase):
    def test_close_stops_the_worker_and_ignores_later_events(self):
        policy = self.policy()
        policy.close()
        self.fire(EventType.MOVEMENT_COMPLETED, commandId="c1", result="success")
        self.assertEqual(self.reasons(), [])
        policy.close()  # a second close is harmless

    def test_start_twice_is_harmless(self):
        policy = self.policy()
        policy.start()
        self.fire(EventType.PERSON_DETECTED, name="anna", room="kitchen")
        self.assertEqual(self.reasons(), ["person"])


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        config = CaptureConfig()
        self.assertEqual(
            (config.after_movement, config.after_movement_cooldown_s, config.on_person,
             config.person_cooldown_s, config.on_interesting, config.interesting_cooldown_s,
             config.exploration_interval_s, config.max_auto_per_minute),
            (True, 10, True, 30, True, 20, 30, 6),
        )

    def test_the_example_file_has_a_capture_section(self):
        config = load_config(Path("config/laika.example.toml"))
        self.assertEqual(config.capture, CaptureConfig())

    def write(self, text):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "laika.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_values_are_read_from_the_capture_section(self):
        path = self.write(
            "[capture]\nafter_movement = false\nperson_cooldown_s = 45\n"
            "exploration_interval_s = 60\nmax_auto_per_minute = 4\n"
        )
        config = load_config(path).capture
        self.assertFalse(config.after_movement)
        self.assertEqual(config.person_cooldown_s, 45)
        self.assertEqual(config.exploration_interval_s, 60)
        self.assertEqual(config.max_auto_per_minute, 4)
        self.assertTrue(config.on_person)

    def test_out_of_range_or_wrong_values_name_the_setting(self):
        cases = {
            "exploration_interval_s = 2": "exploration_interval_s must be 5 to 300",
            "exploration_interval_s = 301": "exploration_interval_s must be 5 to 300",
            "max_auto_per_minute = 0": "max_auto_per_minute must be 1 to 60",
            "max_auto_per_minute = 2.5": "max_auto_per_minute must be a whole number",
            "after_movement_cooldown_s = -1": "after_movement_cooldown_s must be 0 to 600",
            "on_person = 1": "on_person must be true or false",
            "exploration_interval_s = \"30\"": "exploration_interval_s must be a number",
        }
        for line, message in cases.items():
            with self.subTest(line=line):
                with self.assertRaisesRegex(ValueError, message):
                    load_config(self.write(f"[capture]\n{line}\n"))

    def test_unknown_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown setting \\[capture\\] burst"):
            load_config(self.write("[capture]\nburst = 3\n"))


class IntegrationTests(unittest.TestCase):
    def test_a_completed_move_triggers_exactly_one_capture_and_updates_the_world_model(self):
        bus = EventBus()
        bus.start()
        self.addCleanup(bus.close)
        moves = []
        bus.subscribe(moves.append, EventType.MOVEMENT_COMPLETED, EventType.MOVEMENT_FAILED)
        images = []
        bus.subscribe(images.append, EventType.IMAGE_CAPTURED)
        world = WorldModel()
        updater = WorldModelUpdater(world, bus)
        self.addCleanup(updater.close)
        robot = SimulatedRobot(verbose=False, house=default_house(), start=Pose(300, 200, 180), events=bus)
        world.enter_room(robot.current_room())
        safety = SafetyController(robot, speed=25, events=bus)
        self.addCleanup(safety.close)
        camera = Camera(robot, events=bus, min_interval_s=0, max_per_minute=0, save_images=False)
        observer = Observer(bus)
        observer.attach(camera)
        self.addCleanup(observer.close)
        policy = CapturePolicy(camera, bus, CaptureConfig(on_person=False))
        policy.start()
        self.addCleanup(policy.close)
        registry = ToolRegistry(robot, safety, world=world, camera=camera, observer=observer)

        safety.execute(MoveCommand(direction=Direction.FORWARD, speed=30, duration_ms=50))
        settled_events(bus, moves, TERMINAL)
        self.assertTrue(policy.flush())
        bus.drain()
        self.assertEqual([e.type for e in moves], [EventType.MOVEMENT_COMPLETED])
        self.assertEqual([e.data["reason"] for e in images], ["after_movement"])
        self.assertEqual(policy.stats["after_movement"].captured, 1)

        observation = observer.wait_for(camera.last)
        bus.drain()
        self.assertIsNotNone(observation)
        self.assertIsNotNone(world.where_is("sofa"))

        # The agent's tool still captures straight away and does not count against the policy.
        result = registry.run(call("capture_image"))
        bus.drain()
        self.assertTrue(result["ok"])
        self.assertEqual([e.data["reason"] for e in images], ["after_movement", "request"])
        self.assertEqual(policy.stats["after_movement"].captured, 1)


if __name__ == "__main__":
    unittest.main()
