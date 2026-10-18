"""The plan of the house: loading, routes, names, plausibility, and where it plugs in."""

import tempfile
import unittest
from pathlib import Path

from laika.agent import Agent
from laika.config import REPO_ROOT, HouseConfig, RoomsConfig, load_config, validate_house_plan
from laika.floorplan import FloorPlan, load_plan, parse_plan, seed_world
from laika.observations import Observation
from laika.rooms import RoomTracker
from laika.world_model import WorldModel
from tests.agent.test_tools import ToolTestCase

EXAMPLE = REPO_ROOT / "config" / "house-plan.example.toml"
REAL = REPO_ROOT / "config" / "house-plan.toml"
NOW = 1_800_000_000.0


def small(**changes):
    data = {
        "plan": {"home_room": "a"},
        "room": [
            {"id": "a", "names": ["stanza a"], "links": ["b"]},
            {"id": "b", "names": ["stanza b"], "links": ["c"]},
            {"id": "c", "names": ["stanza c"]},
            {"id": "d", "names": ["stanza d"], "no_go": True, "links": ["b"]},
        ],
    }
    data.update(changes)
    return parse_plan(data)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = load_plan(EXAMPLE)

    def test_the_example_loads_with_two_way_links(self):
        self.assertIsNotNone(self.plan)
        for room in self.plan.rooms.values():
            for other in room.links:
                self.assertIn(room.id, self.plan.rooms[other].links)

    @unittest.skipUnless(REAL.exists(), "the real plan is private and not in the repository")
    def test_the_real_plan_loads(self):
        plan = load_plan(REAL)
        self.assertIsNotNone(plan)
        self.assertIn(plan.home_room, plan.rooms)

    def test_a_missing_file_means_no_plan(self):
        self.assertIsNone(load_plan(Path(tempfile.mkdtemp()) / "nothing.toml"))

    def test_routes_cross_the_rooms_between(self):
        self.assertEqual(self.plan.route("bedroom", "kitchen"), ["bedroom", "hall", "living_room", "kitchen"])
        self.assertEqual(self.plan.route("kitchen", "kitchen"), ["kitchen"])
        self.assertIsNone(self.plan.route("kitchen", "nowhere"))

    def test_routes_never_pass_through_a_no_go_room(self):
        plan = small()
        self.assertEqual(plan.route("a", "c"), ["a", "b", "c"])
        plan = parse_plan({"room": [
            {"id": "a", "links": ["x", "b"]},
            {"id": "x", "no_go": True, "links": ["b"]},
            {"id": "b"},
        ]})
        self.assertEqual(plan.route("a", "b"), ["a", "b"])
        plan = parse_plan({"room": [{"id": "a", "links": ["x"]}, {"id": "x", "no_go": True, "links": ["b"]}, {"id": "b"}]})
        self.assertIsNone(plan.route("a", "b"))
        self.assertEqual(plan.route("a", "x"), ["a", "x"])  # as a destination it is only looked at

    def test_names_resolve_without_articles(self):
        self.assertEqual(self.plan.resolve("la cucina"), "kitchen")
        self.assertEqual(self.plan.resolve("il Bagno"), "bathroom")
        self.assertEqual(self.plan.resolve("kids_room"), "kids_room")
        self.assertIsNone(self.plan.resolve("la cantina"))

    def test_plausible_moves(self):
        plan = small()
        self.assertTrue(plan.plausible_move("a", "b"))
        self.assertTrue(plan.plausible_move("b", "a"))
        self.assertFalse(plan.plausible_move("a", "c"))
        self.assertTrue(plan.plausible_move(None, "c"))
        self.assertTrue(plan.plausible_move("a", "elsewhere"))

    def test_no_go_rooms_and_the_summary(self):
        plan = small()
        self.assertEqual(plan.no_go_rooms(), ["d"])
        text = plan.summary()
        self.assertIn("stanza b", text)
        self.assertIn("off limits", text)
        self.assertIn("home spot", text)
        self.assertLessEqual(len(text), 1800)

    def test_bad_plans_are_named(self):
        bad = {
            "no rooms": {"room": []},
            "bad id": {"room": [{"id": "Bad Id"}]},
            "twice": {"room": [{"id": "a"}, {"id": "a"}]},
            "unknown link": {"room": [{"id": "a", "links": ["zzz"]}]},
            "unknown home": {"plan": {"home_room": "zzz"}, "room": [{"id": "a"}]},
            "too many names": {"room": [{"id": "a", "names": list("abcdef")}]},
            "area": {"room": [{"id": "a", "area_m2": 0}]},
            "no_go": {"room": [{"id": "a", "no_go": "yes"}]},
        }
        for label, data in bad.items():
            with self.subTest(label), self.assertRaises(ValueError):
                parse_plan(data)

    def test_the_names_reach_the_world_model(self):
        world = WorldModel()
        seed_world(world, self.plan)
        self.assertEqual(world.room_for_alias("cucina"), "kitchen")
        self.assertIn("camera dei bambini", world.aliases_of("kids_room"))
        self.assertIn("kitchen", world.rooms())


def frames(room, conf=0.9, n=1, start=0.0):
    return [
        Observation(
            description="a frame", objects=(), confidence=0.9, timestamp=NOW + start + i,
            room_guess=room, room_confidence=conf, source="vision",
        )
        for i in range(n)
    ]


class TrackerPlanTests(unittest.TestCase):
    def make(self, plan):
        self.world = WorldModel()
        return RoomTracker(self.world, None, RoomsConfig(), plan=plan)

    def settle(self, tracker, room):
        for observation in frames(room, 0.9, 3):
            tracker.observe(observation)
        self.assertEqual(tracker.current_room, room)

    def test_a_neighbour_is_reached_as_usual(self):
        tracker = self.make(small())
        self.settle(tracker, "a")
        for observation in frames("b", 0.9, 2, start=10):
            tracker.observe(observation)
        self.assertEqual(tracker.current_room, "b")

    def test_a_room_that_is_not_next_door_needs_more_evidence(self):
        plain = self.make(None)
        self.settle(plain, "a")
        for observation in frames("c", 0.9, 2, start=10):
            plain.observe(observation)
        self.assertEqual(plain.current_room, "c")  # without a plan two frames are enough

        planned = self.make(small())
        self.settle(planned, "a")
        for observation in frames("c", 0.9, 2, start=10):
            planned.observe(observation)
        self.assertEqual(planned.current_room, "a")  # a jump over a room: not on two frames
        for observation in frames("c", 0.9, 3, start=20):
            planned.observe(observation)
        self.assertEqual(planned.current_room, "c")  # but real evidence wins


class WiringTests(ToolTestCase):
    def test_house_route_is_offered_only_with_a_plan(self):
        self.assertNotIn("house_route", self.registry.names())

    def test_house_route_answers_from_the_plan(self):
        plan = load_plan(EXAMPLE)
        self.registry.plan = plan
        self.registry._tools = {t.name: t for t in self.registry._build() if t.name == "house_route"}
        self.world.enter_room("bedroom")
        result, _ = self.run_tool("house_route", room="la cucina")
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["from"], "camera da letto")
        self.assertIn("corridoio", result["route"])
        self.assertIn("cucina", result["route"])
        bad, _ = self.run_tool("house_route", room="la cantina")
        self.assertFalse(bad["ok"])
        balcony, _ = self.run_tool("house_route", room="balcone")
        self.assertIn("off limits", balcony["note"])

    def test_the_plan_is_in_the_prompt(self):
        plan = load_plan(EXAMPLE)
        agent = Agent(None, self.registry, "system", world=self.world, plan=plan)
        text = agent._start_messages("dov'è il bagno?")[0]["content"]
        self.assertIn("kids_room", text)


class ConfigTests(unittest.TestCase):
    def test_the_setting_is_a_file_name(self):
        validate_house_plan(HouseConfig())
        for bad in ("../x.toml", "a/b.toml", 5):
            with self.assertRaises(ValueError):
                validate_house_plan(HouseConfig(plan=bad))

    def test_the_default_is_the_private_file(self):
        self.assertEqual(load_config(REPO_ROOT / "config" / "laika.toml").house.plan, "house-plan.toml")


if __name__ == "__main__":
    unittest.main()
