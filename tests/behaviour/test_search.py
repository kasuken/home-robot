"""find_object, find_person and approach (#43): the house simulator with the real scanner, explorer, safety, arbiter, camera, observer, world model and memory."""

import contextlib
import io
import json
import math
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from laika.agent import Agent
from laika.behaviour import Arbiter, Priority
from laika.camera import Camera
from laika.config import ExploreConfig, PresenceConfig, SearchConfig, load_config
from laika.events import EventBus
from laika.explore import Explorer
from laika.house import Person, Pose, Thing, default_house
from laika.memory import Memory
from laika.observations import Observer
from laika.presence import PresenceTracker
from laika.prompts import build_tool_system_prompt
from laika.robot import ALL_CAPABILITIES, CAMERA_PAN, DRIVE
from laika.safety import SafetyController
from laika.scan import Scanner
from laika.search import Searcher, name_matches
from laika.tools import ToolRegistry, Turn
from laika.world_model import WorldModel, WorldModelUpdater
from tests.agent.test_agent import ScriptedClient, calls, text
from tests.agent.test_tools import call
from tests.fakes import LookRecordingRobot

P = Priority
FACING_WEST = Pose(300, 200, 180)  # living room: sofa, television and backpack ahead, the guitars to the left
# Facing the doorway to the kitchen. The coffee machine is hidden by the wall until she has driven about
# 40 cm east, and three exploration steps (about 70 cm) take her into the kitchen.
AT_THE_DOOR = Pose(370, 200, 0)
NEAR_THE_DOOR = Pose(460, 200, 0)
HOUR = 3600.0


def wait_for(condition, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


class TraceRobot(LookRecordingRobot):
    """Records every drive as (speed, steering)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trace = []

    def drive(self, speed):
        super().drive(speed)
        self.trace.append((self.speed, self.steering))

    @property
    def forward_drives(self):
        return [t for t in self.trace if t[0] > 0]


class StubActions:
    def cancel_pending(self):
        pass


class PolicySpy:
    def set_exploring(self, exploring):
        pass


class SearchCase(unittest.TestCase):
    def build(
        self,
        *,
        start=FACING_WEST,
        allow_autonomous=True,
        allow_driving=True,
        capabilities=ALL_CAPABILITIES,
        search=None,
        explore=None,
        clock=time.time,
        with_explorer=True,
    ):
        out = io.StringIO()
        redirect = contextlib.redirect_stdout(out)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)
        self.house = default_house()
        self.bus = EventBus()
        self.bus.start()
        self.addCleanup(self.bus.close)
        self.world = WorldModel()
        self.addCleanup(WorldModelUpdater(self.world, self.bus).close)
        self.memory = Memory()
        self.robot = TraceRobot(house=self.house, start=start, events=self.bus, capabilities=capabilities)
        self.world.enter_room(self.robot.current_room())
        self.camera = Camera(self.robot, None, self.bus, min_interval_s=0, max_per_minute=0, save_images=False)
        self.observer = Observer(self.bus)
        self.observer.attach(self.camera)
        self.addCleanup(self.observer.close)
        self.presence = PresenceTracker(self.bus, PresenceConfig())
        self.presence.start()
        self.addCleanup(self.presence.close)
        self.safety = SafetyController(
            self.robot, speed=40, enabled=allow_driving, allow_autonomous=allow_autonomous
        )
        self.addCleanup(self.safety.close)
        self.safety.start()
        self.arbiter = Arbiter(self.safety, StubActions(), lease_timeout_s=2.0)
        self.addCleanup(self.arbiter.close)
        self.explorer = None
        if with_explorer:
            self.explorer = Explorer(
                self.arbiter,
                self.safety,
                Scanner(self.robot, self.camera, self.observer, settle_s=0.01),
                PolicySpy(),
                config=explore
                or ExploreConfig(speed=40, step_ms=1000, scan_every_steps=3, max_duration_s=60, max_steps=10000),
                body_wait_s=1.0,
            )
            self.addCleanup(self.explorer.stop)
        self.searcher = Searcher(
            self.robot,
            self.safety,
            self.arbiter,
            Scanner(self.robot, self.camera, self.observer, settle_s=0.01),
            self.explorer,
            self.world,
            self.memory,
            self.presence,
            self.observer,
            search or SearchConfig(),
            clock,
            settle_s=0.01,
        )
        self.lease = self.arbiter.request("conversation", P.CONVERSATION)
        return self.searcher

    # --- helpers ---------------------------------------------------------------

    def know(self, name, room, ago=2 * HOUR):
        """Memory of a sighting `ago` seconds ago, as the world model and the episodes keep it."""
        when = time.time() - ago
        self.world.observe([name], room, timestamp=when)
        self.memory.episodes.record("seen", f"Saw {name} in the {room}", [name, room], room, when)

    def see(self, pan=None):
        """Look (optionally), take a picture, wait for its observation and for the events."""
        if pan is not None:
            self.robot.look(pan, 5)
        image = self.camera.capture("request")
        observation = self.observer.wait_for(image, 2.0)
        self.bus.drain()
        return observation

    def place(self, name, pan=0.0, distance=150.0):
        """A new thing straight along the camera direction for head pan `pan`."""
        pose = self.robot.pose()
        angle = math.radians(pose.heading - pan)
        self.house.add_thing(
            Thing(name, pose.x + distance * math.cos(angle), pose.y + distance * math.sin(angle), 10)
        )

    def find(self, name, **kwargs):
        return self.searcher.find_object(name, lease=self.lease, **kwargs)

    def drives(self):
        return [s for s in self.robot.speeds if s != 0]

    def assert_stopped(self):
        self.assertEqual(self.robot.speed, 0)
        self.assertFalse(self.explorer is not None and self.explorer.active)

    def in_thread(self, work):
        box = {}

        def run():
            box["result"] = work()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5.0)
        return thread, box


class NameMatchTests(unittest.TestCase):
    def test_names_match_folded_and_by_whole_words(self):
        self.assertTrue(name_matches("Backpack", "backpack"))
        self.assertTrue(name_matches("coffee machine", "coffee_machine"))
        self.assertTrue(name_matches("backpack", "black backpack"))
        self.assertTrue(name_matches("black backpack", "backpack"))
        self.assertTrue(name_matches("Caffè", "caffe"))

    def test_other_things_and_partial_words_do_not_match(self):
        self.assertFalse(name_matches("back", "backpack"))
        self.assertFalse(name_matches("sofa", "table"))
        self.assertFalse(name_matches("", "table"))


class FindObjectInViewTests(SearchCase):
    def test_a_known_object_in_view_is_found_by_the_head_scan(self):
        self.build()
        self.know("guitars", "living_room")
        result = self.find("guitars")
        self.assertTrue(result["ok"])
        self.assertTrue(result["found"])
        self.assertEqual(result["how"], "scan")
        self.assertEqual(result["where"]["room"], "living_room")
        self.assertEqual(result["where"]["direction"], "left")
        self.assertEqual(result["where"]["pan"], -60)
        self.assertEqual(self.robot.head_pose()[0], -60)  # the head points at it
        self.assertEqual(result["searched"], ["head_scan"])
        self.assertEqual(self.drives(), [])  # she was in the right room: no need to drive
        self.assertFalse(result["drove"])
        self.assertEqual(result["last_known"]["room"], "living_room")
        self.assertEqual(result["last_known"]["ago"], "2 hours ago")

    def test_a_name_in_any_case_or_spelling_is_found(self):
        self.build()
        self.know("coffee_machine", "kitchen")
        self.house.add_thing(Thing("black backpack", 170, 150, 10))
        self.assertTrue(self.find("BACKPACK")["found"])
        self.assertTrue(self.find("Black  Backpack")["found"])

    def test_an_object_never_seen_is_still_looked_for_here(self):
        self.build()
        result = self.find("television")
        self.assertTrue(result["found"])
        self.assertEqual(result["how"], "scan")
        self.assertIn("never seen", " ".join(self.find("unicorn", max_s=5)["notes"]))

    def test_the_result_is_compact_json(self):
        self.build()
        self.know("guitars", "living_room")
        text_form = json.dumps(self.find("guitars"))
        self.assertLess(len(text_form), 900)

    def test_a_missing_name_or_a_person_word_is_refused(self):
        self.build()
        self.assertFalse(self.find("   ")["ok"])
        self.assertIn("find_person", self.find("person")["error"])


class FindObjectElsewhereTests(SearchCase):
    def test_a_known_object_in_another_part_of_the_house_is_found_by_exploring(self):
        self.build(start=NEAR_THE_DOOR)
        self.know("coffee_machine", "kitchen")
        before = self.house.thing("coffee_machine")
        started = time.monotonic()
        with mock.patch.object(self.arbiter, "request", wraps=self.arbiter.request) as request:
            result = self.find("coffee machine", max_s=30)
        self.assertLess(time.monotonic() - started, 25)
        self.assertTrue(result["found"], result)
        self.assertEqual(result["how"], "exploration")
        self.assertEqual(result["where"]["room"], "kitchen")
        self.assertIn(result["where"]["direction"], ("left", "centre", "right"))
        self.assertTrue(result["drove"])
        self.assertTrue(result["autonomous_driving"])
        self.assertEqual(result["searched"], ["exploration"])  # she knew it was elsewhere: no scan first
        self.assertEqual(result["explored"]["ended"], "decision")
        self.assertGreaterEqual(result["explored"]["steps"], 1)
        # The head points at it: the pan is one of the scan's.
        self.assertEqual(self.robot.head_pose()[0], result["where"]["pan"])
        self.assertIs(before, self.house.thing("coffee_machine"))
        # Her own lease did it: nothing asked the arbiter for an exploration lease.
        request.assert_not_called()
        self.assertIs(self.arbiter.holder, self.lease)
        self.assert_stopped()

    def test_a_search_cannot_run_longer_than_its_time(self):
        self.build(start=AT_THE_DOOR)
        self.know("monitor", "office")  # not reachable in the time: there is no map
        started = time.monotonic()
        result = self.find("monitor", max_s=4)
        self.assertLess(time.monotonic() - started, 9)
        self.assertTrue(result["ok"])
        self.assertFalse(result["found"])
        self.assertEqual(result["explored"]["ended"], "duration")
        self.assertIn("time ran out", " ".join(result["notes"]))
        self.assert_stopped()

    def test_the_time_asked_is_capped_by_the_configuration(self):
        self.build(start=AT_THE_DOOR, search=SearchConfig(default_max_s=5, max_max_s=5))
        started = time.monotonic()
        result = self.find("monitor", max_s=500)
        self.assertLess(time.monotonic() - started, 10)
        self.assertEqual(result["explored"]["ended"], "duration")

    def test_not_found_says_where_it_was_last_seen_and_that_she_cannot_navigate(self):
        self.build(allow_autonomous=False)
        self.know("monitor", "office")
        result = self.find("monitor")
        self.assertTrue(result["ok"])
        self.assertFalse(result["found"])
        self.assertEqual(result["how"], "memory")
        self.assertFalse(result["confirmed"])
        self.assertEqual(result["last_known"]["room"], "office")
        self.assertEqual(result["last_known"]["ago"], "2 hours ago")
        self.assertIn("T", result["last_known"]["when"])
        self.assertEqual(result["last_known"]["latest"][0]["room"], "office")
        notes = " ".join(result["notes"])
        self.assertIn("cannot navigate to a room", notes)
        self.assertIn("office", notes)
        self.assertNotIn("where", result)

    def test_the_usual_room_and_what_the_user_said_are_part_of_the_answer(self):
        self.build(allow_autonomous=False)
        self.world.observe(["monitor"], "office", timestamp=time.time() - 3 * HOUR)
        self.world.observe(["monitor"], "office", timestamp=time.time() - 2 * HOUR)
        self.world.learn("monitor", "office", note="work screen")
        self.memory.facts.remember("Il monitor sta in studio", "monitor", "location")
        last = self.find("monitor")["last_known"]
        self.assertEqual(last["usual_room"], "office")
        self.assertEqual(last["note"], "work screen")
        self.assertEqual(last["told"], "Il monitor sta in studio")

    def test_an_object_that_is_not_where_it_was_is_reported_as_not_in_view(self):
        self.build(allow_autonomous=False)
        self.know("sofa", "living_room")
        self.house.remove_thing("sofa")
        result = self.find("sofa")
        self.assertFalse(result["found"])
        self.assertEqual(result["searched"], ["head_scan"])
        self.assertIn("not in view", " ".join(result["notes"]))
        self.assertNotIn("cannot navigate", " ".join(result["notes"]))  # same room: nothing to go to


class AutonomousSwitchOffTests(SearchCase):
    def test_with_the_switch_off_the_search_is_head_only_and_no_drive_reaches_the_robot(self):
        self.build(start=AT_THE_DOOR, allow_autonomous=False)
        self.know("coffee_machine", "kitchen")
        result = self.find("coffee machine", max_s=30)
        self.assertFalse(result["found"])
        self.assertFalse(result["autonomous_driving"])
        self.assertFalse(result["drove"])
        self.assertEqual(result["searched"], ["head_scan"])  # she looks where she is, then memory answers
        self.assertIn("autonomous_disabled", " ".join(result["notes"]))
        self.assertEqual(self.robot.speeds, [])
        self.assertEqual(self.explorer.steps, 0)

    def test_with_the_switch_off_a_known_object_in_view_is_still_found(self):
        self.build(allow_autonomous=False)
        self.know("guitars", "living_room")
        result = self.find("guitars")
        self.assertTrue(result["found"])
        self.assertEqual(result["how"], "scan")
        self.assertEqual(self.robot.speeds, [])

    def test_the_explorer_never_drives_a_search_without_the_switch(self):
        self.build(allow_autonomous=False)
        reason = self.explorer.run_goal(self.lease, 5, lambda scan: None)
        self.assertEqual(reason, "autonomous_disabled")
        self.assertEqual(self.robot.speeds, [])

    def test_without_wheels_or_an_explorer_she_only_looks_around(self):
        self.build(allow_driving=False)
        result = self.find("monitor", max_s=10)
        self.assertIn("driving_disabled", " ".join(result["notes"]))
        self.assertEqual(self.robot.speeds, [])
        self.build(with_explorer=False)
        result = self.find("monitor", max_s=10)
        self.assertIn("exploration_disabled", " ".join(result["notes"]))
        self.assertEqual(self.robot.speeds, [])

    def test_approach_does_not_drive(self):
        self.build(allow_autonomous=False)
        self.see(pan=0)
        result = self.searcher.approach("sofa", lease=self.lease)
        self.assertFalse(result["ok"])
        self.assertFalse(result["moved"])
        self.assertEqual(result["reason"], "autonomous_disabled")
        self.assertEqual(self.robot.speeds, [])


class FindPersonTests(SearchCase):
    def find_person(self, **kwargs):
        return self.searcher.find_person(lease=self.lease, **kwargs)

    def test_a_person_who_is_near_is_pointed_at_and_confirmed(self):
        self.build(start=Pose(300, 200, 0))
        self.house.add_person(Person("anna", 420, 200))
        self.see(pan=0)
        self.assertTrue(wait_for(lambda: self.presence.present))
        self.robot.look(50, 0)  # the head is somewhere else now
        result = self.find_person()
        self.assertTrue(result["found"])
        self.assertEqual(result["how"], "presence")
        self.assertEqual(result["people"], 1)
        self.assertEqual(result["where"]["direction"], "centre")
        self.assertEqual(self.robot.head_pose()[0], 0)
        self.assertEqual(result["searched"], ["presence"])
        self.assertEqual(self.drives(), [])

    def test_a_person_who_left_is_looked_for_with_the_head_scan(self):
        self.build(start=Pose(300, 200, 0))
        self.house.add_person(Person("anna", 420, 200))
        self.see(pan=0)
        self.assertTrue(wait_for(lambda: self.presence.present))
        self.house.move_person("anna", 375, 70)  # now to her right
        result = self.find_person()
        self.assertTrue(result["found"], result)
        self.assertEqual(result["how"], "scan")
        self.assertEqual(result["where"]["direction"], "right")
        self.assertEqual(result["searched"], ["presence", "head_scan"])

    def test_a_person_in_view_is_found_by_the_scan(self):
        self.build(start=Pose(300, 200, 0))
        self.house.add_person(Person("anna", 375, 70))
        result = self.find_person()
        self.assertTrue(result["found"])
        self.assertEqual(result["how"], "scan")
        self.assertEqual(result["where"]["direction"], "right")
        self.assertEqual(result["where"]["pan"], 60)
        self.assertEqual(self.robot.head_pose()[0], 60)
        self.assertEqual(result["people"], 1)
        notes = " ".join(result["notes"])
        self.assertIn("cannot tell people apart", notes)  # never a name
        self.assertNotIn("anna", json.dumps(result))

    def test_a_person_in_another_room_is_found_by_exploring(self):
        self.build(start=NEAR_THE_DOOR)
        self.house.add_person(Person("anna", 600, 10))  # behind the wall from here, to her right in the kitchen
        result = self.find_person(max_s=30)
        self.assertTrue(result["found"], result)
        self.assertEqual(result["how"], "exploration")
        self.assertEqual(result["where"]["room"], "kitchen")
        self.assertEqual(result["searched"], ["head_scan", "exploration"])
        self.assertTrue(result["drove"])
        self.assert_stopped()

    def test_nobody_is_found_and_she_says_so(self):
        self.build(allow_autonomous=False)
        result = self.find_person()
        self.assertTrue(result["ok"])
        self.assertFalse(result["found"])
        self.assertIsNone(result["how"])
        self.assertNotIn("last_known", result)
        notes = " ".join(result["notes"])
        self.assertIn("not seen anyone", notes)
        self.assertIn("autonomous_disabled", notes)
        self.assertEqual(self.robot.speeds, [])

    def test_the_last_sighting_of_a_person_is_reported(self):
        self.build(start=Pose(300, 200, 0), allow_autonomous=False)
        self.house.add_person(Person("anna", 420, 200))
        self.see(pan=0)
        self.house.remove_person("anna")
        self.robot.look(0, 0)
        result = self.find_person()
        self.assertFalse(result["found"])
        self.assertEqual(result["last_known"]["room"], "living_room")
        self.assertEqual(result["last_known"]["people"], 1)
        self.assertEqual(result["last_known"]["ago"], "just now")


class ApproachTests(SearchCase):
    def approach(self, target, **kwargs):
        return self.searcher.approach(target, lease=self.lease, **kwargs)

    def test_it_stops_at_the_stop_distance(self):
        self.build(search=SearchConfig(approach_moves=5, approach_stop_cm=105))
        self.see(pan=0)  # the sofa, 120 cm ahead
        self.assertAlmostEqual(self.robot.distance_cm(), 120, delta=2)
        result = self.approach("sofa")
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["reason"], "close_enough")
        self.assertTrue(result["reached"])
        self.assertTrue(result["moved"])
        self.assertGreaterEqual(result["moves"], 1)
        self.assertLess(result["moves"], 5)
        self.assertLessEqual(result["distance_cm"], 105)
        self.assertGreater(result["distance_cm"], 80)
        self.assert_stopped()

    def test_it_makes_at_most_approach_moves_short_moves(self):
        self.build(search=SearchConfig(approach_moves=2, approach_stop_cm=40))
        self.see(pan=0)
        self.robot.distance = 200.0  # the reading never gets smaller
        result = self.approach("sofa")
        self.assertEqual(result["moves"], 2)
        self.assertEqual(result["reason"], "max_moves")
        self.assertFalse(result["reached"])
        self.assertEqual(len(self.robot.forward_drives), 2)
        self.assertTrue(all(steering == 0 for _, steering in self.robot.forward_drives))

    def test_it_does_not_move_when_already_close(self):
        self.build(search=SearchConfig(approach_stop_cm=40))
        self.see(pan=0)
        self.robot.distance = 30.0
        result = self.approach("sofa")
        self.assertEqual(result["reason"], "close_enough")
        self.assertTrue(result["reached"])
        self.assertFalse(result["moved"])
        self.assertEqual(self.drives(), [])

    def test_an_obstacle_stops_it(self):
        self.build(search=SearchConfig(approach_stop_cm=20))
        self.see(pan=0)
        self.robot.distance = 22.0  # above her stop distance, below the safety layer's minimum
        result = self.approach("sofa")
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "obstacle")
        self.assertFalse(result["moved"])
        self.assertEqual(self.drives(), [])

    def test_a_cliff_stops_it_after_the_moves_made(self):
        self.build(search=SearchConfig(approach_moves=3, approach_stop_cm=20))
        self.see(pan=0)
        self.robot.distance = 200.0
        outcomes = iter([None, "cliff"])
        real = self.searcher._drive

        def drive(run, steering):
            reason = next(outcomes)
            return reason if reason else real(run, steering)

        with mock.patch.object(self.searcher, "_drive", drive):
            result = self.approach("sofa")
        self.assertEqual(result["moves"], 1)
        self.assertEqual(result["reason"], "cliff")
        self.assertTrue(result["ok"])  # it did move closer first

    def test_a_target_that_is_not_ahead_gets_one_turn_then_forward_moves(self):
        self.build(search=SearchConfig(approach_moves=2, approach_stop_cm=40))
        self.place("box", pan=-50, distance=200)
        observation = self.see(pan=-50)
        self.assertIn("box", observation.object_names)
        heading = self.robot.pose().heading
        result = self.approach("box")
        self.assertTrue(result["turned"], result)
        self.assertGreaterEqual(result["moves"], 1)
        self.assertTrue(result["ok"])
        first = self.robot.forward_drives[0]
        self.assertLess(first[1], 0)  # steering left, towards a target on the left
        self.assertTrue(all(steering == 0 for _, steering in self.robot.forward_drives[1:]))
        turned = (self.robot.pose().heading - heading) % 360
        self.assertGreater(turned, 20)  # the nose came round to the left (a larger heading)
        self.assertLess(turned, 60)

    def test_a_person_can_be_approached(self):
        self.build(start=Pose(300, 200, 0), search=SearchConfig(approach_moves=2, approach_stop_cm=40))
        self.house.add_person(Person("anna", 450, 200))
        self.see(pan=0)
        result = self.approach("person")
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["target"], "person")
        self.assertGreaterEqual(result["moves"], 1)
        self.assertNotIn("anna", json.dumps(result))

    def test_a_target_never_seen_is_refused(self):
        self.build()
        result = self.approach("unicorn")
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "not_seen_recently")
        self.assertIn("find_object", result["error"])
        self.assertEqual(self.drives(), [])

    def test_an_old_sighting_is_not_used(self):
        self.build(clock=lambda: time.time() + 1000)
        self.see(pan=0)
        result = self.approach("sofa")
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "not_seen_recently")
        self.assertEqual(self.drives(), [])

    def test_a_sighting_in_another_room_is_not_used(self):
        self.build()
        self.see(pan=0)
        self.world.enter_room("kitchen")
        result = self.approach("sofa")
        self.assertFalse(result["ok"])
        self.assertIn("another room", result["error"])
        self.assertEqual(self.drives(), [])

    def test_losing_sight_ends_the_approach(self):
        self.build(search=SearchConfig(approach_moves=3, approach_stop_cm=40))
        self.place("box", pan=0, distance=200)
        self.see(pan=0)
        self.robot.distance = 200.0
        self.house.remove_thing("box")  # gone after the first move
        result = self.approach("box")
        self.assertEqual(result["reason"], "lost_sight")
        self.assertEqual(result["moves"], 1)


class CancellationTests(SearchCase):
    def test_a_cancelled_turn_stops_the_search_and_the_robot(self):
        self.build(start=AT_THE_DOOR)
        cancelled = threading.Event()
        thread, box = self.in_thread(lambda: self.find("unicorn", max_s=60, should_stop=cancelled.is_set))
        self.assertTrue(wait_for(lambda: self.robot.forward_drives))
        cancelled.set()
        stopped_at = time.monotonic()
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertLess(time.monotonic() - stopped_at, 3)
        self.assertFalse(box["result"]["ok"])
        self.assertEqual(box["result"]["reason"], "cancelled")
        self.assert_stopped()

    def test_a_more_important_behaviour_taking_the_body_stops_the_search(self):
        self.build(start=AT_THE_DOOR)
        thread, box = self.in_thread(lambda: self.find("unicorn", max_s=60))
        self.assertTrue(wait_for(lambda: self.robot.forward_drives))
        self.assertIsNotNone(self.arbiter.request("user", P.USER_COMMAND, resumable=False))
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(box["result"]["ok"])
        self.assertEqual(box["result"]["reason"], "preempted")
        self.assertEqual(self.robot.speed, 0)
        self.assertFalse(self.explorer.active)

    def test_a_search_for_a_lease_already_gone_does_nothing(self):
        self.build()
        self.lease.release()
        result = self.find("guitars")
        self.assertEqual(result["reason"], "preempted")
        self.assertEqual(self.robot.poses, [])

    def test_a_cancelled_turn_stops_an_approach(self):
        self.build()
        self.see(pan=0)
        self.robot.distance = 200.0
        result = self.searcher.approach("sofa", lease=self.lease, should_stop=lambda: True)
        self.assertEqual(result["reason"], "cancelled")
        self.assertEqual(self.drives(), [])

    def test_only_one_search_runs_at_a_time(self):
        self.build(start=AT_THE_DOOR)
        cancelled = threading.Event()
        thread, box = self.in_thread(lambda: self.find("unicorn", max_s=60, should_stop=cancelled.is_set))
        self.assertTrue(wait_for(lambda: self.robot.forward_drives))
        second = self.find("guitars")
        self.assertFalse(second["ok"])
        self.assertIn("already", second["error"])
        cancelled.set()
        thread.join(5)

    def test_without_a_conversation_the_arbiter_gives_a_lease_and_takes_it_back(self):
        self.build()
        self.lease.release()
        self.know("guitars", "living_room")
        result = self.searcher.find_object("guitars")
        self.assertTrue(result["found"])
        self.assertIsNone(self.arbiter.holder)

    def test_without_a_conversation_a_busy_body_is_reported(self):
        self.build()
        result = self.searcher.find_object("guitars", lease=None)
        self.assertEqual(result["reason"], "busy")  # the conversation holds the body


class ToolTests(SearchCase):
    def registry(self, **kwargs):
        return ToolRegistry(
            self.robot,
            self.safety,
            world=self.world,
            memory=self.memory,
            camera=self.camera,
            observer=self.observer,
            allow_driving=kwargs.pop("allow_driving", True),
            searcher=self.searcher,
            **kwargs,
        )

    def test_the_tools_are_offered_with_the_needed_capabilities(self):
        self.build()
        names = self.registry().names()
        for name in ("find_object", "find_person", "approach"):
            self.assertIn(name, names)

    def test_without_a_searcher_none_is_offered(self):
        self.build()
        plain = ToolRegistry(self.robot, self.safety)
        for name in ("find_object", "find_person", "approach"):
            self.assertNotIn(name, plain.names())

    def test_without_a_head_there_is_no_search(self):
        self.build(capabilities=ALL_CAPABILITIES - {CAMERA_PAN})
        names = self.registry().names()
        self.assertNotIn("find_object", names)
        self.assertNotIn("find_person", names)
        self.assertIn("approach", names)

    def test_without_wheels_there_is_no_approach(self):
        self.build(capabilities=ALL_CAPABILITIES - {DRIVE})
        names = self.registry().names()
        self.assertNotIn("approach", names)
        self.assertIn("find_object", names)
        self.build()
        names = self.registry(allow_driving=False).names()
        self.assertNotIn("approach", names)
        self.assertIn("find_person", names)

    def test_the_prompt_mentions_the_tools_only_when_offered(self):
        gestures = {}
        plain = build_tool_system_prompt(gestures, 3)
        self.assertNotIn("find_object", plain)
        self.assertNotIn("approach", plain)
        only_find = build_tool_system_prompt(gestures, 3, searching=True)
        self.assertIn("find_object", only_find)
        self.assertIn("find_person", only_find)
        self.assertIn("da dove sei", only_find)
        self.assertNotIn("approach", only_find)
        both = build_tool_system_prompt(gestures, 3, searching=True, approaching=True)
        self.assertIn("approach", both)
        self.assertIn("vieni qui", both)

    def test_find_object_runs_under_the_turns_lease(self):
        self.build()
        self.know("guitars", "living_room")
        result = self.registry().run(call("find_object", name="guitars"), Turn(lease=self.lease))
        self.assertTrue(result["ok"])
        self.assertTrue(result["found"])
        json.dumps(result)

    def test_the_arguments_are_validated(self):
        self.build()
        registry = self.registry()
        turn = Turn(lease=self.lease)
        self.assertFalse(registry.run(call("find_object"), turn)["ok"])
        self.assertFalse(registry.run(call("find_object", name="guitars", max_seconds=500), turn)["ok"])
        self.assertFalse(registry.run(call("find_object", name="guitars", max_seconds=1), turn)["ok"])
        self.assertFalse(registry.run(call("find_person", name="Leonardo"), turn)["ok"])  # no names
        self.assertFalse(registry.run(call("approach"), turn)["ok"])
        self.assertEqual(self.robot.poses, [])

    def test_a_cancelled_turn_stops_a_search_started_by_a_tool(self):
        self.build(start=AT_THE_DOOR)
        generation = {"now": 0}
        turn = Turn(0, lambda: generation["now"], self.lease)
        registry = self.registry()
        thread, box = self.in_thread(lambda: registry.run(call("find_object", name="unicorn", max_seconds=60), turn))
        self.assertTrue(wait_for(lambda: self.robot.forward_drives))
        generation["now"] = 1  # the agent cancelled the turn
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(box["result"]["ok"])
        self.assert_stopped()

    def test_a_preempted_search_aborts_the_rest_of_the_turn(self):
        self.build(start=AT_THE_DOOR)
        turn = Turn(lease=self.lease)
        registry = self.registry()
        thread, box = self.in_thread(lambda: registry.run(call("find_object", name="unicorn", max_seconds=60), turn))
        self.assertTrue(wait_for(lambda: self.robot.forward_drives))
        self.arbiter.request("user", P.USER_COMMAND, resumable=False)
        thread.join(5)
        self.assertEqual(box["result"]["reason"], "preempted")
        self.assertIn("preempted", turn.stale_reason or "")  # nothing else this turn may run

    def test_the_agent_finds_an_object_end_to_end(self):
        self.build()
        self.know("guitars", "living_room")
        client = ScriptedClient(
            calls(("find_object", {"name": "guitars"})),
            text("Le chitarre sono a sinistra."),
        )
        agent = Agent(client, self.registry(), "Sei Laika.", world=self.world, memory=self.memory)
        reply = agent.respond("Dove sono le chitarre?", lease=self.lease)
        self.assertEqual(reply.text, "Le chitarre sono a sinistra.")
        self.assertEqual(reply.tool_calls_made, ["find_object"])
        tool_result = json.loads(client.requests[1]["messages"][-1]["content"])
        self.assertTrue(tool_result["found"])
        self.assertEqual(tool_result["where"]["direction"], "left")
        self.assertEqual(self.robot.head_pose()[0], -60)

    def test_the_agent_can_approach_what_it_just_found(self):
        self.build(search=SearchConfig(approach_moves=2, approach_stop_cm=40))
        self.robot.distance = 200.0
        client = ScriptedClient(
            calls(("find_object", {"name": "sofa"}), ("approach", {"target": "sofa"})),
            text("Eccomi."),
        )
        agent = Agent(client, self.registry(), "Sei Laika.", world=self.world, memory=self.memory)
        agent.respond("Vieni dal divano", lease=self.lease)
        approach = json.loads(client.requests[1]["messages"][-1]["content"])
        self.assertTrue(approach["ok"], approach)
        self.assertEqual(approach["moves"], 2)


class ConfigTests(unittest.TestCase):
    def load(self, body):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "laika.toml"
            path.write_text(body, encoding="utf-8")
            return load_config(path)

    def test_the_defaults(self):
        config = SearchConfig()
        self.assertEqual((config.default_max_s, config.max_max_s), (60, 180))
        self.assertEqual((config.approach_moves, config.approach_stop_cm), (3, 40))

    def test_a_valid_section_is_read(self):
        config = self.load("[search]\ndefault_max_s = 30\nmax_max_s = 120\napproach_moves = 2\napproach_stop_cm = 60\n")
        self.assertEqual(config.search.default_max_s, 30)
        self.assertEqual(config.search.approach_moves, 2)

    def test_bad_values_name_the_setting(self):
        for body, name in (
            ("[search]\ndefault_max_s = 1\n", "default_max_s"),
            ("[search]\nmax_max_s = 9999\n", "max_max_s"),
            ("[search]\napproach_moves = 0\n", "approach_moves"),
            ("[search]\napproach_moves = 1.5\n", "approach_moves"),
            ("[search]\napproach_stop_cm = 5\n", "approach_stop_cm"),
            ("[search]\ndefault_max_s = 200\nmax_max_s = 100\n", "max_max_s"),
            ("[search]\ncolour = 1\n", "colour"),
        ):
            with self.subTest(body=body):
                with self.assertRaises(ValueError) as error:
                    self.load(body)
                self.assertIn(name, str(error.exception))

    def test_the_example_file_loads(self):
        root = Path(__file__).resolve().parents[2]
        config = load_config(root / "config" / "laika.example.toml")
        self.assertEqual(config.search.approach_moves, 3)


if __name__ == "__main__":
    unittest.main()
