"""Environment changes (#36): detection, episodes, the what_changed tool, recall's habits, curiosity."""

from __future__ import annotations

import time
import unittest

from laika.camera import Camera
from laika.changes import Change, ChangeDetector, ChangeTracker
from laika.config import CuriosityConfig
from laika.curiosity import InterestGate, InterestScorer
from laika.events import EventBus, EventType
from laika.house import Pose
from laika.memory import EpisodicMemory, Memory, iso_minutes
from laika.observations import Observation, ObservationLog, Observer, SeenObject
from laika.simulator import SimulatedRobot
from laika.tools import ToolRegistry
from laika.world_model import WorldModel, WorldModelUpdater
from tests.agent.test_tools import ToolTestCase, call
from tests.memory.test_memory import HOUR, RegistryCase, local


def obs(*names, location="living_room", confidence=1.0, pan=0.0, timestamp=1000.0):
    return Observation(
        description="a view",
        objects=tuple(SeenObject(n) for n in names),
        location=location,
        confidence=confidence,
        camera_pan=pan,
        timestamp=timestamp,
    )


def kinds(changes):
    return [(c.kind, c.object) for c in changes]


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.world = WorldModel()
        self.log = ObservationLog()
        self.detector = ChangeDetector(self.world, self.log, CuriosityConfig())

    def test_an_object_last_known_in_another_room_has_moved(self):
        self.world.observe(["keys"], "kitchen", timestamp=1.0)
        (change,) = self.detector.detect(obs("keys", location="office"))
        self.assertEqual((change.kind, change.object, change.location, change.previous_location),
                         ("moved", "keys", "office", "kitchen"))
        self.assertEqual(change.text, "keys moved from kitchen to office")
        self.assertEqual(change.timestamp, 1000.0)

    def test_a_never_seen_object_in_a_familiar_room_is_new(self):
        self.world.observe(["sofa"], "living_room", timestamp=1.0)
        changes = self.detector.detect(obs("sofa", "backpack", timestamp=1000.0))
        self.assertEqual(kinds(changes), [("new", "backpack")])
        self.assertEqual(changes[0].text, "backpack appeared in living room")

    def test_discovering_a_room_is_not_a_change(self):
        self.world.observe(["sofa"], "living_room", timestamp=1000.0)
        self.assertEqual(self.detector.detect(obs("sofa", "backpack", timestamp=1010.0)), [])
        self.assertEqual(self.detector.detect(obs("lamp", location="office")), [])

    def test_unusual_needs_an_established_habit(self):
        self.world.observe(["mug"], "living_room", timestamp=1.0)
        self.world.observe(["mug"], "office", timestamp=2.0)
        self.assertEqual(kinds(self.detector.detect(obs("mug", location="kitchen"))),
                         [("moved", "mug")])  # one move is not a habit
        self.world.observe(["mug"], "living_room", timestamp=3.0)
        changes = self.detector.detect(obs("mug", location="kitchen"))
        self.assertEqual(kinds(changes), [("moved", "mug"), ("unusual", "mug")])
        self.assertEqual(changes[1].previous_location, "living_room")
        self.assertEqual(self.detector.detect(obs("mug", location="living_room")), [])

    def test_missing_from_the_same_view_when_the_frame_is_confident(self):
        self.log.add(obs("sofa", "backpack", timestamp=900.0))
        changes = self.detector.detect(obs("sofa", timestamp=1000.0))
        self.assertEqual(kinds(changes), [("missing", "backpack")])
        self.assertEqual(changes[0].text, "backpack is no longer in living room")

    def test_a_blurry_frame_does_not_make_things_vanish(self):
        self.log.add(obs("sofa", "backpack", timestamp=900.0))
        self.assertEqual(self.detector.detect(obs("sofa", confidence=0.5)), [])
        self.assertEqual(kinds(self.detector.detect(obs("sofa", confidence=0.6))), [("missing", "backpack")])

    def test_missing_needs_the_same_view(self):
        self.log.add(obs("sofa", "backpack", timestamp=900.0, pan=40.0))
        self.log.add(obs("sofa", "backpack", timestamp=900.0, location="office"))
        self.assertEqual(self.detector.detect(obs("sofa", timestamp=1000.0)), [])

    def test_detecting_changes_nothing(self):
        self.world.observe(["keys"], "kitchen", timestamp=1.0)
        before = self.world.to_dict()
        self.detector.detect(obs("keys", location="office"))
        self.assertEqual(self.world.to_dict(), before)
        self.assertEqual(len(self.log), 0)

    def test_without_a_location_there_is_nothing_to_compare(self):
        self.world.observe(["keys"], "kitchen", timestamp=1.0)
        self.assertEqual(self.detector.detect(obs("keys", location=None)), [])


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.episodes = EpisodicMemory()
        self.tracker = ChangeTracker(self.episodes)

    def moved(self, to, before, at=1.0):
        return Change("moved", "backpack", to, before, at, "obs-1")

    def test_moved_new_and_missing_become_episodes_with_the_change_text(self):
        changes = [
            self.moved("kitchen", "living_room"),
            Change("new", "lamp", "office", None, 2.0, "obs-2"),
            Change("missing", "mug", "office", None, 3.0, "obs-3"),
            Change("unusual", "keys", "office", "kitchen", 4.0, "obs-4"),
        ]
        self.assertEqual(len(self.tracker.record(changes)), 3)
        recorded = {e.kind: e.summary for e in self.episodes.recent()}
        self.assertEqual(recorded, {
            "moved": "backpack moved from living room to kitchen",
            "new": "lamp appeared in office",
            "missing": "mug is no longer in office",
        })
        self.assertEqual([c.kind for c in self.tracker.recent()], ["missing", "new", "moved"])

    def test_the_same_change_is_recorded_once_until_the_state_changes(self):
        self.assertEqual(len(self.tracker.record([self.moved("kitchen", "living_room")])), 1)
        self.assertEqual(self.tracker.record([self.moved("kitchen", "living_room", at=5.0)]), [])
        self.assertEqual(len(self.episodes), 1)
        # Back to the living room, then to the kitchen again: each is a new state.
        self.assertEqual(len(self.tracker.record([self.moved("living_room", "kitchen")])), 1)
        self.assertEqual(len(self.tracker.record([self.moved("kitchen", "living_room")])), 1)
        self.assertEqual(len(self.episodes), 3)

    def test_missing_again_counts_after_the_object_was_back(self):
        gone = Change("missing", "mug", "office", None, 1.0, "obs-1")
        self.assertEqual(len(self.tracker.record([gone])), 1)
        self.assertEqual(self.tracker.record([gone]), [])
        self.tracker.record([], present=["mug"])
        self.assertEqual(len(self.tracker.record([gone])), 1)

    def test_the_recent_list_is_bounded(self):
        tracker = ChangeTracker(max_items=3)
        for index in range(10):
            tracker.record([Change("new", f"thing {index}", "office", None, float(index), "obs")])
        self.assertEqual([c.object for c in tracker.recent(10)], ["thing 9", "thing 8", "thing 7"])


class CuriosityTests(unittest.TestCase):
    def test_the_scorer_consumes_the_detectors_changes(self):
        world = WorldModel()
        world.observe(["keys"], "kitchen", timestamp=1.0)
        scorer = InterestScorer(world, config=CuriosityConfig())
        observation = obs("keys", location="office")
        given = scorer.score(observation, changes=[Change("moved", "keys", "office", "kitchen", 1000.0, "x")])
        self.assertEqual(given.components["environmental_change"], 1.0)
        self.assertEqual(given.reasons, ["moved: keys from kitchen to office"])
        self.assertEqual(scorer.score(observation).to_dict(), given.to_dict())
        # What the detector did not report is not scored, whatever the world model says.
        self.assertEqual(scorer.score(observation, changes=[]).components["environmental_change"], 0.0)

    def test_missing_objects_do_not_change_the_scores(self):
        scorer = InterestScorer(WorldModel(), config=CuriosityConfig())
        gone = Change("missing", "mug", "office", None, 1.0, "x")
        score = scorer.score(obs("sofa", location="office"), changes=[gone])
        self.assertEqual(score.components["environmental_change"], 0.0)
        self.assertEqual(score.components["unusual_location"], 0.0)


class SimulatedHouseTests(ToolTestCase):
    """Real Camera, Observer, scorer, detector, world model and memory, in the default house."""

    start = Pose(250, 270, 180)  # the living room, facing west: the backpack is in view

    def setUp(self):
        super().setUp()
        self.bus = EventBus()
        self.bus.start()
        self.addCleanup(self.bus.close)
        self.world = WorldModel()
        updater = WorldModelUpdater(self.world, self.bus)
        self.addCleanup(updater.close)
        self.memory = Memory()
        config = CuriosityConfig()
        log = ObservationLog()
        self.observer = Observer(
            self.bus,
            log,
            default_location=lambda: self.world.current_room,
            scorer=InterestScorer(self.world, memory=self.memory, config=config),
            gate=InterestGate(config, self.bus),
            detector=ChangeDetector(self.world, log, config),
            tracker=ChangeTracker(self.memory.episodes),
        )
        self.addCleanup(self.observer.close)
        self.created = []
        self.bus.subscribe(self.created.append, EventType.OBSERVATION_CREATED)
        self.registry = ToolRegistry(
            self.robot, self.safety, world=self.world, memory=self.memory, clock=time.time
        )
        self.house = self.robot.house

    def look(self, robot, room):
        camera = Camera(robot, events=self.bus, min_interval_s=0, clock=time.monotonic)
        self.observer.attach(camera)
        self.world.enter_room(room)
        observation = self.observer.wait_for(camera.capture())
        self.assertIsNotNone(observation)
        self.assertTrue(self.bus.drain())
        return observation

    def test_moving_an_object_in_the_simulator_is_reported_as_a_change(self):
        first = self.look(self.robot, "living_room")
        self.assertIn("backpack", first.object_names)
        self.assertEqual(self.world.where_is("backpack").room, "living_room")
        self.assertEqual(self.created[-1].data["changes"], [])

        self.house.move_thing("backpack", 700, 120)  # into the kitchen
        kitchen_robot = SimulatedRobot(verbose=False, house=self.house, start=Pose(600, 150, 0))
        second = self.look(kitchen_robot, "kitchen")
        self.assertIn("backpack", second.object_names)

        text = "backpack moved from living room to kitchen"
        changes = self.created[-1].data["changes"]
        self.assertEqual([(c["kind"], c["text"]) for c in changes], [("moved", text)])
        self.assertEqual(changes[0]["observation_id"], second.id)
        self.assertEqual([(e.kind, e.summary) for e in self.memory.episodes.recent() if e.kind == "moved"],
                         [("moved", text)])
        score = self.observer.log.score_of(second.id)
        self.assertEqual(score.components["environmental_change"], 1.0)
        self.assertIn("moved: backpack from living room to kitchen", score.reasons)

        result = self.registry.run(call("what_changed"))
        self.assertTrue(result["ok"])
        self.assertEqual([c["summary"] for c in result["changes"]], [text])
        self.assertEqual(result["changes"][0]["object"], "backpack")
        self.assertEqual(result["changes"][0]["location"], "kitchen")

        # Looking again does not report (or record) the same move a second time.
        self.look(kitchen_robot, "kitchen")
        self.assertEqual(self.created[-1].data["changes"], [])
        self.assertEqual(len([e for e in self.memory.episodes.recent() if e.kind == "moved"]), 1)

    def test_an_observer_without_a_detector_publishes_no_changes(self):
        observer = Observer(self.bus)
        self.addCleanup(observer.close)
        camera = Camera(self.robot, events=self.bus, min_interval_s=0)
        observer.attach(camera)
        observer.wait_for(camera.capture())
        self.assertTrue(self.bus.drain())
        self.assertNotIn("changes", self.created[-1].data)


class WhatChangedTests(RegistryCase):
    def setUp(self):
        super().setUp()  # now is 2026-10-15 14:30 local
        episodes = self.memory.episodes
        episodes.record_moved("backpack", "living_room", "office", local(2026, 10, 13, 20, 0))
        episodes.record_moved("keys", "kitchen", "office", local(2026, 10, 14, 9, 0), "keys moved from kitchen to office")
        episodes.record_new("lamp", "office", local(2026, 10, 15, 8, 0))
        episodes.record_missing("mug", "kitchen", local(2026, 10, 15, 13, 20))
        episodes.record("seen", "Saw sofa in the living room", ["sofa"], "living_room", local(2026, 10, 15, 14, 0))

    def summaries(self, **arguments):
        result = self.tool("what_changed", **arguments)
        self.assertTrue(result["ok"], result)
        return [c["kind"] + ":" + c["object"] for c in result["changes"]]

    def test_yesterday_starts_at_the_previous_local_midnight(self):
        self.assertEqual(self.summaries(since="yesterday"), ["missing:mug", "new:lamp", "moved:keys"])

    def test_today_starts_at_local_midnight(self):
        self.assertEqual(self.summaries(since="today"), ["missing:mug", "new:lamp"])

    def test_the_default_is_the_last_twenty_four_hours(self):
        self.assertEqual(self.summaries(), ["missing:mug", "new:lamp"])
        self.assertEqual(self.summaries(since_hours=30), ["missing:mug", "new:lamp", "moved:keys"])

    def test_hours_back_and_the_times_it_returns(self):
        self.assertEqual(self.summaries(since_hours=2), ["missing:mug"])
        result = self.tool("what_changed", since_hours=2)
        (change,) = result["changes"]
        self.assertEqual(change["when"], iso_minutes(local(2026, 10, 15, 13, 20)))
        self.assertEqual(change["ago"], "1 hour ago")
        self.assertEqual(change["location"], "kitchen")
        self.assertEqual(result["since"], iso_minutes(self.clock() - 2 * HOUR))

    def test_nothing_changed_says_so(self):
        result = self.tool("what_changed", since_hours=1)
        self.assertEqual((result["ok"], result["changes"]), (True, []))
        self.assertIn("note", result)

    def test_the_period_is_validated(self):
        self.assertFalse(self.tool("what_changed", since_hours=0)["ok"])
        self.assertFalse(self.tool("what_changed", since_hours=169)["ok"])
        self.assertFalse(self.tool("what_changed", since="last week")["ok"])

    def test_it_works_after_a_restart_from_the_store(self):
        self.end_session()
        self.start_session()
        self.memory.episodes._recent.clear()  # only the store has them now
        self.assertEqual(len(self.memory.episodes), 0)
        self.assertEqual(self.summaries(since="yesterday"), ["missing:mug", "new:lamp", "moved:keys"])

    def test_the_result_is_bounded(self):
        for index in range(40):
            self.memory.episodes.record_new(f"thing {index}", "office", self.clock() - 60 - index)
        self.assertEqual(len(self.tool("what_changed", since_hours=1)["changes"]), 20)

    def test_recall_lists_the_move_among_the_episodes(self):
        found = self.tool("recall", query="keys")
        self.assertEqual([e["kind"] for e in found["episodes"]], ["moved"])


class RecallHabitTests(RegistryCase):
    def test_have_i_seen_it_and_is_it_normally_here(self):
        first = local(2026, 10, 14, 9, 0)
        self.world.observe(["mug"], "living_room", timestamp=first)
        self.world.observe(["mug"], "office", timestamp=first + 60)
        self.world.observe(["mug"], "living_room", timestamp=first + 120)
        (mug,) = self.tool("recall", query="mug")["objects"]
        self.assertEqual((mug["usual_room"], mug["is_usual_here"]), ("living_room", True))
        self.assertEqual(mug["first_seen"], iso_minutes(first))
        self.assertEqual(mug["first_seen_ago"], "yesterday")
        self.assertEqual(mug["times_seen"], 3)

        self.world.observe(["mug"], "kitchen", timestamp=first + 180)
        (mug,) = self.tool("recall", query="mug")["objects"]
        self.assertEqual((mug["usual_room"], mug["is_usual_here"]), ("living_room", False))

    def test_without_a_habit_it_is_unknown(self):
        self.world.observe(["lamp"], "office", timestamp=self.clock())
        (lamp,) = self.tool("recall", query="lamp")["objects"]
        self.assertEqual((lamp["usual_room"], lamp["is_usual_here"]), (None, None))
        self.assertEqual(lamp["times_seen"], 1)

    def test_an_object_never_seen_is_not_found(self):
        self.assertFalse(self.tool("recall", query="unicorn")["found"])


class WorldHelperTests(unittest.TestCase):
    def test_usual_room_and_familiar_room(self):
        world = WorldModel()
        self.assertIsNone(world.usual_room("mug"))
        world.observe(["mug"], "kitchen", timestamp=10.0)
        self.assertIsNone(world.usual_room("Mug"))
        world.observe(["mug"], "office", timestamp=20.0)
        world.observe(["mug"], "kitchen", timestamp=30.0)
        self.assertEqual(world.usual_room("MUG"), "kitchen")
        self.assertTrue(world.is_familiar("kitchen", 10.0))
        self.assertFalse(world.is_familiar("kitchen", 9.0))
        self.assertFalse(world.is_familiar("bedroom", 1000.0))
        self.assertFalse(world.is_familiar(None, 1000.0))


if __name__ == "__main__":
    unittest.main()
