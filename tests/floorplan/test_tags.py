"""AprilTag position fixes: the maths, the plan's tag table, the reader, and the tracker's fix."""

import math
import time
import unittest

from laika.config import REPO_ROOT, RoomsConfig, TagsConfig, load_config, validate_tags
from laika.floorplan import load_plan, parse_plan
from laika.robot import ImageObservation
from laika.rooms import RoomTracker
from laika.tags import RawTag, TagReader, sighting
from laika.world_model import WorldModel

EXAMPLE = REPO_ROOT / "config" / "house-plan.example.toml"
CONFIG = TagsConfig()


def square(cx, width, cy=240):
    half = width / 2
    return [(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half), (cx - half, cy + half)]


def frame(t=0.0):
    return ImageObservation(jpeg=b"fake", width=640, height=480, camera_pan=0, camera_tilt=0, timestamp=1000.0 + t)


class FakeDetector:
    def __init__(self, *tags):
        self.tags = list(tags)
        self.calls = 0

    def detect(self, jpeg):
        self.calls += 1
        return list(self.tags)


class SightingTests(unittest.TestCase):
    def test_distance_follows_the_size(self):
        focal = 320 / math.tan(math.radians(31))
        near = sighting(RawTag(1, square(320, 100)), 640, 62, 16)
        far = sighting(RawTag(1, square(320, 50)), 640, 62, 16)
        self.assertAlmostEqual(near.distance_cm, 16 * focal / 100, places=3)
        self.assertAlmostEqual(far.distance_cm, 2 * near.distance_cm, places=3)

    def test_bearing_is_zero_ahead_and_signed(self):
        self.assertAlmostEqual(sighting(RawTag(1, square(320, 80)), 640, 62, 16).bearing_deg, 0.0, places=6)
        self.assertGreater(sighting(RawTag(1, square(500, 80)), 640, 62, 16).bearing_deg, 10)
        self.assertLess(sighting(RawTag(1, square(100, 80)), 640, 62, 16).bearing_deg, -10)

    def test_nonsense_is_dropped(self):
        self.assertIsNone(sighting(RawTag(1, square(320, 0.1)), 640, 62, 16))
        self.assertIsNone(sighting(RawTag(1, [(0, 0), (1, 1)]), 640, 62, 16))
        self.assertIsNone(sighting(RawTag(1, square(320, 50)), 0, 62, 16))


class PlanTagTests(unittest.TestCase):
    def test_the_example_has_tags(self):
        plan = load_plan(EXAMPLE)
        self.assertEqual(plan.tags[1].marks, "kids_room")
        self.assertEqual(plan.tags[1].seen_from, "hall")
        self.assertEqual(plan.tags[0].seen_from, "living_room")  # defaults to the marked room

    def test_bad_tags_are_named(self):
        rooms = [{"id": "a"}, {"id": "b"}]
        bad = {
            "id": [{"id": 700, "marks": "a"}],
            "negative": [{"id": -1, "marks": "a"}],
            "not a number": [{"id": "1", "marks": "a"}],
            "twice": [{"id": 1, "marks": "a"}, {"id": 1, "marks": "b"}],
            "unknown room": [{"id": 1, "marks": "zzz"}],
            "unknown seen_from": [{"id": 1, "marks": "a", "seen_from": "zzz"}],
        }
        for label, tags in bad.items():
            with self.subTest(label), self.assertRaises(ValueError):
                parse_plan({"room": rooms, "tag": tags})


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.plan = load_plan(EXAMPLE)
        self.world = WorldModel()
        self.tracker = RoomTracker(self.world, None, RoomsConfig(), plan=self.plan)

    def reader(self, *tags, **config):
        return TagReader(self.plan, self.tracker, FakeDetector(*tags), TagsConfig(**config))

    def test_a_near_tag_fixes_the_room(self):
        reader = self.reader(RawTag(1, square(320, 120)))  # about 80 cm
        reader.read(frame())
        self.assertEqual(self.tracker.current_room, "hall")
        self.assertEqual(reader.fixes, 1)

    def test_a_tag_fix_ignores_plausibility(self):
        self.tracker.correct("bathroom")
        self.reader(RawTag(2, square(320, 120))).read(frame())  # the kitchen tag, seen from the living room
        self.assertEqual(self.tracker.current_room, "living_room")

    def test_a_far_tag_is_ignored(self):
        reader = self.reader(RawTag(1, square(320, 5)))  # about 17 m by the size
        reader.read(frame())
        self.assertIsNone(self.tracker.current_room)
        self.assertEqual((reader.fixes, reader.ignored), (0, 1))

    def test_a_tag_the_plan_does_not_know_is_ignored(self):
        reader = self.reader(RawTag(99, square(320, 120)))
        reader.read(frame())
        self.assertIsNone(self.tracker.current_room)
        self.assertEqual(reader.ignored, 1)

    def test_the_nearest_known_tag_decides(self):
        reader = self.reader(RawTag(1, square(200, 60)), RawTag(2, square(440, 140)))
        reader.read(frame())
        self.assertEqual(self.tracker.current_room, "living_room")  # tag 2 is nearer

    def test_the_fix_is_published_with_the_tag_as_its_source(self):
        from laika.events import EventBus, EventType

        bus = EventBus()
        bus.start()
        self.addCleanup(bus.close)
        seen = []
        bus.subscribe(seen.append, EventType.ROOM_ENTERED)
        tracker = RoomTracker(self.world, bus, RoomsConfig(), plan=self.plan)
        TagReader(self.plan, tracker, FakeDetector(RawTag(1, square(320, 120))), CONFIG).read(frame())
        bus.drain()
        self.assertEqual([e.data["source"] for e in seen], ["tag"])
        self.assertEqual(self.world.current_room, "hall")

    def test_the_worker_reads_a_submitted_frame_and_skips_frames_without_a_picture(self):
        reader = self.reader(RawTag(1, square(320, 120)))
        self.addCleanup(reader.close)
        reader.submit(ImageObservation(jpeg=None, width=640, height=480, camera_pan=0, camera_tilt=0))
        reader.submit(frame())
        deadline = time.time() + 5
        while reader.frames < 1 and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(reader.frames, 1)
        self.assertEqual(self.tracker.current_room, "hall")

    def test_a_failing_detector_does_not_stop_the_worker(self):
        class Broken:
            calls = 0

            def detect(self, jpeg):
                Broken.calls += 1
                if Broken.calls == 1:
                    raise RuntimeError("bad frame")
                return [RawTag(1, square(320, 120))]

        reader = TagReader(self.plan, self.tracker, Broken(), CONFIG)
        self.addCleanup(reader.close)
        reader.submit(frame())
        deadline = time.time() + 5
        while Broken.calls < 1 and time.time() < deadline:
            time.sleep(0.01)
        reader.submit(frame(1))
        while reader.fixes < 1 and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(reader.fixes, 1)


try:
    import cv2
    import numpy
except ImportError:  # CI installs only requests; the robot has OpenCV
    cv2 = None


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class OpenCVDetectorTests(unittest.TestCase):
    """The real detector on the real tag36h11 id 0 (docs/images), as a JPEG like the camera's."""

    def setUp(self):
        from laika.tags import OpenCVTagDetector

        self.detector = OpenCVTagDetector()
        self.tag = cv2.imread(str(REPO_ROOT / "docs" / "images" / "apriltag-36h11-id0.png"), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(self.tag)

    def jpeg(self, image):
        ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        self.assertTrue(ok)
        return data.tobytes()

    def scene(self, side, x=300, y=200):
        canvas = numpy.full((480, 640), 190, numpy.uint8)
        canvas[y:y + side, x:x + side] = cv2.resize(self.tag, (side, side), interpolation=cv2.INTER_AREA)
        return canvas

    def test_finds_the_tag_and_its_width(self):
        found = self.detector.detect(self.jpeg(self.scene(100)))
        self.assertEqual([t.tag_id for t in found], [0])
        seen = sighting(found[0], 640, 62, 16)
        self.assertAlmostEqual(seen.width_px, 100 * 8 / 14, delta=4)  # the black square is 8 of the image's 14 modules

    def test_finds_a_small_far_tag(self):
        self.assertEqual([t.tag_id for t in self.detector.detect(self.jpeg(self.scene(50)))], [0])

    def test_nothing_in_an_empty_or_broken_frame(self):
        self.assertEqual(self.detector.detect(self.jpeg(numpy.full((480, 640), 120, numpy.uint8))), [])
        self.assertEqual(self.detector.detect(b"not a jpeg"), [])


class ConfigTests(unittest.TestCase):
    def test_defaults_validate_and_bad_values_are_named(self):
        validate_tags(TagsConfig())
        for name, value in (("tag_size_cm", 0), ("fov_deg", 5), ("max_fix_cm", 5000), ("enabled", "yes")):
            with self.subTest(name), self.assertRaisesRegex(ValueError, name):
                validate_tags(TagsConfig(**{name: value}))

    def test_the_committed_settings_load(self):
        self.assertTrue(load_config(REPO_ROOT / "config" / "laika.toml").tags.enabled)


if __name__ == "__main__":
    unittest.main()
