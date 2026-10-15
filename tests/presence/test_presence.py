import tempfile
import time
import unittest
from pathlib import Path

from laika.camera import Camera
from laika.capture_policy import CapturePolicy
from laika.config import CaptureConfig, PresenceConfig, load_config
from laika.events import EventBus, EventType
from laika.house import Person, default_house
from laika.observations import Observation, Observer
from laika.presence import PresenceTracker
from laika.simulator import SimulatedRobot
from tests.camera.test_camera import FakeRobot
from tests.camera.test_capture_policy import wait_until
from tests.fakes import FakeClock

PERSON_EVENTS = (EventType.PERSON_DETECTED, EventType.PERSON_LOST)


class TrackerTestCase(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.bus = EventBus()
        self.bus.start()
        self.addCleanup(self.bus.close)
        self.seen = []
        self.bus.subscribe(self.seen.append, *PERSON_EVENTS)

    def tracker(self, clock=None, **overrides):
        tracker = PresenceTracker(self.bus, PresenceConfig(**overrides), clock=clock or self.clock)
        tracker.start()
        self.addCleanup(tracker.close)
        return tracker

    def see(self, people=1, confidence=0.9, room="kitchen", pan=10.0, tilt=0.0):
        observation = Observation(
            "a frame", confidence=confidence, people=people, location=room, camera_pan=pan, camera_tilt=tilt
        )
        self.bus.emit(EventType.OBSERVATION_CREATED, **observation.to_dict())
        self.bus.drain()

    def types(self):
        self.bus.drain()
        return [e.type for e in self.seen]


class TrackerTests(TrackerTestCase):
    def test_a_person_in_a_frame_raises_person_detected_with_details(self):
        tracker = self.tracker()
        self.see(people=2, confidence=0.8, room="office", pan=-20.0)
        self.assertEqual(self.types(), [EventType.PERSON_DETECTED])
        data = self.seen[0].data
        self.assertEqual((data["count"], data["confidence"], data["room"]), (2, 0.8, "office"))
        self.assertEqual((data["pan"], data["tilt"], data["duration_s"]), (-20.0, 0.0, 0.0))
        self.assertIsNotNone(data["since"])
        self.assertTrue(tracker.present)
        self.assertEqual(tracker.count, 2)
        self.assertEqual(tracker.last_seen, self.clock.now)

    def test_one_arrival_publishes_one_event_however_many_frames_show_them(self):
        self.tracker()
        for _ in range(4):
            self.see()
            self.clock.advance(5)
        self.assertEqual(self.types(), [EventType.PERSON_DETECTED])

    def test_a_changed_count_updates_the_state_without_an_event(self):
        tracker = self.tracker()
        self.see(people=1)
        self.see(people=3)
        self.assertEqual(tracker.count, 3)
        self.assertEqual(self.types(), [EventType.PERSON_DETECTED])

    def test_person_lost_after_lost_after_s_without_a_person(self):
        tracker = self.tracker(lost_after_s=30)
        self.see()
        self.clock.advance(29)
        tracker.check()
        self.assertTrue(tracker.present)
        self.clock.advance(2)
        tracker.check()
        self.assertEqual(self.types(), [EventType.PERSON_DETECTED, EventType.PERSON_LOST])
        lost = self.seen[1].data
        self.assertEqual((lost["count"], lost["room"]), (1, "kitchen"))
        self.assertEqual(lost["duration_s"], 31)
        self.assertFalse(tracker.present)
        self.assertEqual(tracker.count, 0)
        self.assertIsNotNone(tracker.last_seen)
        tracker.check()
        self.assertEqual(len(self.seen), 2)

    def test_the_timer_publishes_person_lost_without_any_new_frame(self):
        tracker = self.tracker(clock=time.monotonic, lost_after_s=0.1)
        self.see()
        self.assertTrue(wait_until(lambda: not tracker.present))
        self.assertEqual(self.types(), [EventType.PERSON_DETECTED, EventType.PERSON_LOST])

    def test_one_blurry_frame_does_not_flap(self):
        tracker = self.tracker(lost_after_s=30)
        self.see()
        self.clock.advance(20)
        self.see(people=0, confidence=0.2)
        self.see(people=0, confidence=1.0)
        self.assertTrue(tracker.present)
        self.clock.advance(5)
        self.see()  # the person is back before the time ran out
        self.clock.advance(25)
        self.see(people=0)
        self.assertTrue(tracker.present)
        self.assertEqual(self.types(), [EventType.PERSON_DETECTED])

    def test_empty_frames_end_the_visit_once_the_time_is_up(self):
        tracker = self.tracker(lost_after_s=30)
        self.see()
        self.clock.advance(31)
        self.see(people=0)
        self.assertFalse(tracker.present)
        self.assertEqual(self.types(), [EventType.PERSON_DETECTED, EventType.PERSON_LOST])

    def test_low_confidence_frames_are_ignored(self):
        tracker = self.tracker(min_confidence=0.5)
        self.see(confidence=0.4)
        self.assertFalse(tracker.present)
        self.assertEqual(self.types(), [])
        self.see(confidence=0.5)
        self.assertTrue(tracker.present)

    def test_low_confidence_people_do_not_keep_a_visit_alive(self):
        tracker = self.tracker(lost_after_s=30)
        self.see()
        self.clock.advance(31)
        self.see(confidence=0.1)
        self.assertFalse(tracker.present)

    def test_coming_back_after_a_long_gap_is_a_new_arrival(self):
        self.tracker(lost_after_s=30)
        self.see()
        self.clock.advance(100)
        self.see()
        self.assertEqual(
            self.types(), [EventType.PERSON_DETECTED, EventType.PERSON_LOST, EventType.PERSON_DETECTED]
        )

    def test_malformed_data_is_ignored(self):
        tracker = self.tracker()
        tracker.observe({"people": "many", "confidence": 1})
        tracker.observe(None)
        self.assertFalse(tracker.present)


class SimulatorTests(unittest.TestCase):
    """The done-when: a person entering the view raises PersonDetected, through the real pipeline."""

    def setUp(self):
        self.clock = FakeClock()
        self.bus = EventBus()
        self.bus.start()
        self.addCleanup(self.bus.close)
        self.house = default_house()
        self.robot = SimulatedRobot(verbose=False, house=self.house, events=self.bus)
        self.camera = Camera(
            self.robot, None, self.bus, min_interval_s=0, max_per_minute=0, save_images=False
        )
        self.observer = Observer(self.bus)
        self.observer.attach(self.camera)
        self.addCleanup(self.observer.close)
        self.tracker = PresenceTracker(self.bus, PresenceConfig(lost_after_s=30), clock=self.clock)
        self.tracker.start()
        self.addCleanup(self.tracker.close)
        self.seen = []
        self.bus.subscribe(self.seen.append, *PERSON_EVENTS)

    def look(self):
        image = self.camera.capture("presence")
        self.observer.wait_for(image)
        self.bus.drain()

    def test_a_person_entering_the_view_raises_one_person_detected(self):
        self.look()
        self.assertEqual(self.seen, [])
        self.house.add_person(Person("anna", 420, 200))  # in front of the robot (it starts facing +x)
        self.look()
        self.look()
        self.assertEqual([e.type for e in self.seen], [EventType.PERSON_DETECTED])
        data = self.seen[0].data
        self.assertEqual((data["count"], data["confidence"], data["room"]), (1, 1.0, "living_room"))
        self.assertNotIn("name", data)

    def test_leaving_the_view_raises_person_lost_after_the_timeout(self):
        self.house.add_person(Person("anna", 420, 200))
        self.look()
        self.house.move_person("anna", 100, 100)  # behind the robot
        self.clock.advance(20)
        self.look()
        self.assertTrue(self.tracker.present)
        self.clock.advance(15)
        self.look()
        self.assertEqual([e.type for e in self.seen], [EventType.PERSON_DETECTED, EventType.PERSON_LOST])

    def test_idle_presence_captures_notice_someone_walking_in(self):
        policy = CapturePolicy(
            self.camera,
            self.bus,
            CaptureConfig(person_cooldown_s=0),
            presence=PresenceConfig(presence_interval_s=0.05, presence_check_s=3600),
            person_present=lambda: self.tracker.present,
        )
        self.addCleanup(policy.close)
        policy.start()
        self.house.add_person(Person("anna", 420, 200))
        self.assertTrue(wait_until(lambda: any(e.type is EventType.PERSON_DETECTED for e in self.seen)))
        policy.close()
        self.bus.drain()
        self.assertEqual([e.type for e in self.seen], [EventType.PERSON_DETECTED])


class PresenceCaptureTests(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus()
        self.bus.start()
        self.addCleanup(self.bus.close)
        self.camera = Camera(
            FakeRobot(), None, self.bus, min_interval_s=0, max_per_minute=0, save_images=False
        )
        self.present = False
        self.talking = False

    def policy(self, capture=None, **presence):
        policy = CapturePolicy(
            self.camera,
            self.bus,
            capture or CaptureConfig(),
            presence=PresenceConfig(**presence),
            person_present=lambda: self.present,
            conversation_active=lambda: self.talking,
        )
        self.addCleanup(policy.close)
        policy.start()
        return policy

    def count(self, policy):
        return policy.stats["presence"].captured

    def test_idle_pictures_are_taken_at_the_presence_interval(self):
        policy = self.policy(presence_interval_s=0.05, presence_check_s=3600)
        self.assertTrue(wait_until(lambda: self.count(policy) >= 3))

    def test_no_pictures_during_a_conversation(self):
        self.talking = True
        policy = self.policy(presence_interval_s=0.05, presence_check_s=0.05)
        time.sleep(0.4)
        self.assertEqual(self.count(policy), 0)
        self.talking = False
        self.assertTrue(wait_until(lambda: self.count(policy) >= 1))

    def test_pictures_slow_down_while_someone_is_present(self):
        self.present = True
        policy = self.policy(presence_interval_s=0.05, presence_check_s=3600)
        time.sleep(0.4)
        self.assertEqual(self.count(policy), 0)
        self.present = False
        self.assertTrue(wait_until(lambda: self.count(policy) >= 1))

    def test_the_check_interval_applies_while_present(self):
        self.present = True
        policy = self.policy(presence_interval_s=3600, presence_check_s=0.05)
        self.assertTrue(wait_until(lambda: self.count(policy) >= 2))

    def test_zero_turns_each_state_off(self):
        policy = self.policy(presence_interval_s=0, presence_check_s=0)
        time.sleep(0.3)
        self.assertEqual(self.count(policy), 0)

    def test_without_a_presence_config_there_are_no_idle_pictures(self):
        policy = CapturePolicy(self.camera, self.bus, CaptureConfig())
        self.addCleanup(policy.close)
        policy.start()
        time.sleep(0.2)
        self.assertEqual(self.count(policy), 0)

    def test_idle_pictures_share_the_per_minute_cap(self):
        policy = self.policy(CaptureConfig(max_auto_per_minute=2), presence_interval_s=0.05)
        self.assertTrue(wait_until(lambda: policy.stats["presence"].skipped_rate >= 1))
        self.assertEqual(self.count(policy), 2)

    def test_the_person_trigger_still_works_off_the_new_event(self):
        policy = self.policy(CaptureConfig(person_cooldown_s=0), presence_interval_s=0)
        self.bus.emit(EventType.PERSON_DETECTED, count=1, confidence=0.9, room="kitchen")
        self.bus.drain()
        self.assertTrue(policy.flush())
        self.assertEqual(policy.stats["person"].captured, 1)


class ConfigTests(unittest.TestCase):
    def write(self, text):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "laika.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_defaults_and_example_file(self):
        config = PresenceConfig()
        self.assertEqual(
            (config.min_confidence, config.lost_after_s, config.presence_interval_s, config.presence_check_s),
            (0.5, 600, 300, 300),  # one idle photo every 5 minutes at most
        )
        self.assertEqual(load_config(Path("config/laika.example.toml")).presence, config)

    def test_values_are_read_from_the_presence_section(self):
        text = "[presence]\nlost_after_s = 45\npresence_interval_s = 0\npresence_check_s = 30\n"
        config = load_config(self.write(text)).presence
        self.assertEqual((config.lost_after_s, config.presence_interval_s), (45, 0))

    def test_lost_after_s_must_outlast_the_presence_check(self):
        with self.assertRaisesRegex(ValueError, "lost_after_s must be longer than presence_check_s"):
            load_config(self.write("[presence]\nlost_after_s = 300\npresence_check_s = 300\n"))
        # With no presence checks, any lost_after_s is fine.
        config = load_config(self.write("[presence]\nlost_after_s = 30\npresence_check_s = 0\n")).presence
        self.assertEqual(config.lost_after_s, 30)

    def test_bad_values_name_the_setting(self):
        cases = {
            "min_confidence = 1.5": "min_confidence must be 0 to 1",
            "lost_after_s = 1\npresence_check_s = 0": "lost_after_s must be 5 to 3600",
            "presence_interval_s = 2": "presence_interval_s must be 5 to 3600",
            "presence_check_s = -1": "presence_check_s must be 5 to 3600",
            "presence_interval_s = \"60\"": "presence_interval_s must be a number",
        }
        for line, message in cases.items():
            with self.subTest(line=line):
                with self.assertRaisesRegex(ValueError, message):
                    load_config(self.write(f"[presence]\n{line}\n"))

    def test_unknown_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown setting \\[presence\\] names"):
            load_config(self.write("[presence]\nnames = true\n"))


if __name__ == "__main__":
    unittest.main()
