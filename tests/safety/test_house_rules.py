"""House rules (#51): rule facts become no-go rooms, the safety gate refuses autonomous moves in them,
and in the simulator a rule taught by voice keeps idle exploration out of the room."""

import time
import unittest

from laika.behaviour import Arbiter, Priority
from laika.camera import Camera
from laika.commands import Direction, MoveCommand, ResultStatus
from laika.config import ExploreConfig, HouseConfig, MemoryConfig
from laika.conversation_memory import MemorySelector
from laika.events import EventBus
from laika.explore import Explorer
from laika.house import Pose, default_house
from laika.house_rules import HouseRules, canonical_room, rooms_in_rule
from laika.memory import Memory
from laika.safety import NO_GO_ESCAPE_MOVES, SafetyController
from laika.scan import Scanner
from laika.simulator import SimulatedRobot
from laika.world_model import WorldModel, WorldModelUpdater
from tests.fakes import RecordingRobot


def wait_for(condition, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


class RuleReadingTests(unittest.TestCase):
    def test_italian_rules_name_their_rooms(self):
        for text, rooms in (
            ("Non entrare in camera da letto", {"bedroom"}),
            ("non entrare in camera", {"bedroom"}),
            ("Non andare in cucina", {"kitchen"}),
            ("Stai fuori dallo studio", {"office"}),
            ("non devi mai andare nel salotto, per favore", {"living_room"}),
            ("non andare più in bagno", {"bathroom"}),
            ("non entrare nell'ingresso", {"entrance"}),
            ("Non entrare in cucina e in camera da letto", {"kitchen", "bedroom"}),
            ("non entrare in garage", {"garage"}),
        ):
            self.assertEqual(rooms_in_rule(text), rooms, text)

    def test_other_rules_are_not_no_go_rooms(self):
        for text in ("non toccare i giocattoli", "mi piace il caffè", "la sera si parla piano", "entra pure in cucina"):
            self.assertEqual(rooms_in_rule(text), set(), text)

    def test_room_words_map_to_the_world_models_names(self):
        for word, name in (
            ("camera", "bedroom"), ("camera da letto", "bedroom"), ("cucina", "kitchen"), ("soggiorno", "living_room"),
            ("salotto", "living_room"), ("studio", "office"), ("ufficio", "office"), ("bagno", "bathroom"),
            ("ingresso", "entrance"), ("living_room", "living_room"), ("Kitchen", "kitchen"), ("la Cucina", "kitchen"),
        ):
            self.assertEqual(canonical_room(word), name, word)


class HouseRulesTests(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        self.world = WorldModel()

    def rules(self, *rooms):
        return HouseRules(self.memory, self.world, HouseConfig(no_go_rooms=list(rooms)))

    def test_a_rule_fact_makes_a_no_go_room_and_forget_lifts_it(self):
        rules = self.rules()
        self.assertEqual(rules.no_go_rooms(), set())
        fact = self.memory.facts.remember("Non entrare in camera da letto", "camera da letto", "rule")
        self.assertEqual(rules.no_go_rooms(), {"bedroom"})
        self.assertTrue(rules.is_no_go("bedroom"))
        self.assertTrue(rules.is_no_go("camera da letto"))  # the vision model may use Italian names
        self.assertFalse(rules.is_no_go("kitchen"))
        self.assertEqual(rules.count(), 1)
        self.memory.facts.forget(fact.id)
        self.assertEqual(rules.no_go_rooms(), set())
        self.assertFalse(rules.is_no_go("bedroom"))

    def test_only_rule_facts_count(self):
        self.memory.facts.remember("Non entrare in cucina", "cucina", "general")
        self.assertEqual(self.rules().no_go_rooms(), set())

    def test_a_rule_that_names_no_room_uses_its_subject(self):
        self.memory.facts.remember("non entrare", "studio", "rule")
        self.memory.facts.remember("non toccare", "studio", "rule")
        self.assertEqual(self.rules().no_go_rooms(), {"office"})

    def test_rooms_come_from_the_settings_too(self):
        rules = self.rules("bagno", "living_room")
        self.assertEqual(rules.no_go_rooms(), {"bathroom", "living_room"})
        self.memory.facts.remember("Stai fuori dallo studio", None, "rule")
        self.assertEqual(rules.no_go_rooms(), {"bathroom", "living_room", "office"})

    def test_blocked_room_follows_the_world_model(self):
        rules = self.rules("camera")
        self.assertIsNone(rules.blocked_room())  # she does not know where she is
        self.world.enter_room("kitchen")
        self.assertIsNone(rules.blocked_room())
        self.world.enter_room("bedroom")
        self.assertEqual(rules.blocked_room(), "bedroom")

    def test_a_rule_taught_by_voice_reaches_the_rules(self):
        selector = MemorySelector(self.world, self.memory, config=MemoryConfig())
        stored = selector.select("Non entrare in camera da letto")
        self.assertEqual([f.kind for f in stored], ["rule"])
        self.assertEqual(self.rules().no_go_rooms(), {"bedroom"})
        stored = selector.select("Stai fuori dallo studio")
        self.assertEqual([f.kind for f in stored], ["rule"])
        self.assertEqual(self.rules().no_go_rooms(), {"bedroom", "office"})


class GateTests(unittest.TestCase):
    """The safety controller refuses autonomous moves in a no-go room (apart from reversing out)."""

    def setUp(self):
        self.robot = RecordingRobot(distance_cm=100)
        self.memory = Memory()
        self.world = WorldModel()
        self.rules = HouseRules(self.memory, self.world, HouseConfig(no_go_rooms=["camera da letto"]))
        self.safety = SafetyController(
            self.robot, speed=25, allow_autonomous=True, house_rules=self.rules, watchdog_timeout_s=0
        )
        self.addCleanup(self.safety.close)
        self.safety.start()
        self.arbiter = Arbiter(self.safety)
        self.addCleanup(self.arbiter.close)

    def move(self, direction=Direction.FORWARD, ms=100):
        return MoveCommand(direction=direction, speed=25, duration_ms=ms)

    def lease(self, priority=Priority.EXPLORATION):
        lease = self.arbiter.request("test", priority)
        self.assertIsNotNone(lease)
        self.addCleanup(lease.release)
        return lease

    def test_autonomous_moves_run_outside_the_room(self):
        self.world.enter_room("kitchen")
        result = self.safety.execute(self.move(), lease=self.lease())
        self.assertEqual(result.result, ResultStatus.SUCCESS)

    def test_autonomous_forward_is_refused_inside_the_room(self):
        self.world.enter_room("bedroom")
        result = self.safety.execute(self.move(), lease=self.lease())
        self.assertEqual((result.result, result.reason), (ResultStatus.REJECTED, "no_go_room"))
        self.assertEqual(self.robot.speeds, [])

    def test_self_directed_moves_under_a_conversation_lease_are_refused_too(self):
        self.world.enter_room("bedroom")
        lease = self.lease(Priority.CONVERSATION)
        result = self.safety.execute(self.move(), lease=lease, self_directed=True)
        self.assertEqual((result.result, result.reason), (ResultStatus.REJECTED, "no_go_room"))

    def test_the_users_own_moves_are_not_blocked(self):
        self.world.enter_room("bedroom")
        self.assertEqual(self.safety.execute(self.move()).result, ResultStatus.SUCCESS)
        lease = self.lease(Priority.USER_COMMAND)
        self.assertEqual(self.safety.execute(self.move(), lease=lease).result, ResultStatus.SUCCESS)
        lease.release()
        conversation = self.lease(Priority.CONVERSATION)
        self.assertEqual(self.safety.execute(self.move(), lease=conversation).result, ResultStatus.SUCCESS)

    def test_a_few_moves_backwards_are_allowed_to_get_out(self):
        self.world.enter_room("bedroom")
        lease = self.lease()
        for _ in range(NO_GO_ESCAPE_MOVES):
            self.assertEqual(self.safety.execute(self.move(Direction.BACKWARD), lease=lease).result, ResultStatus.SUCCESS)
        refused = self.safety.execute(self.move(Direction.BACKWARD), lease=lease)
        self.assertEqual((refused.result, refused.reason), (ResultStatus.REJECTED, "no_go_room"))
        # Out of the room the allowance starts again.
        self.world.enter_room("kitchen")
        self.assertEqual(self.safety.execute(self.move(), lease=lease).result, ResultStatus.SUCCESS)
        self.world.enter_room("bedroom")
        self.assertEqual(self.safety.execute(self.move(Direction.BACKWARD), lease=lease).result, ResultStatus.SUCCESS)

    def test_forgetting_the_rule_lifts_the_refusal(self):
        rules = HouseRules(self.memory, self.world)
        self.safety.house_rules = rules
        fact = self.memory.facts.remember("Non andare in cucina", "cucina", "rule")
        self.world.enter_room("kitchen")
        lease = self.lease()
        self.assertEqual(self.safety.execute(self.move(), lease=lease).reason, "no_go_room")
        self.memory.facts.forget(fact.id)
        self.assertEqual(self.safety.execute(self.move(), lease=lease).result, ResultStatus.SUCCESS)


class SimulatorTests(unittest.TestCase):
    """The done-when of #51: a rule taught by voice keeps idle exploration out of the bedroom."""

    def build(self, rule):
        self.bus = EventBus()
        self.bus.start()
        self.addCleanup(self.bus.close)
        house = default_house()
        self.drives = []  # (speed, room it started in)
        drives = self.drives

        class Robot(SimulatedRobot):
            def drive(self, speed):
                super().drive(speed)
                drives.append((self.speed, self._room_name()))

        self.robot = Robot(verbose=False, house=house, start=Pose(650, 330, 90), events=self.bus)
        self.world = WorldModel()
        self.world.enter_room(self.robot.current_room())
        updater = WorldModelUpdater(self.world, self.bus)
        self.addCleanup(updater.close)
        self.memory = Memory()
        if rule:
            selector = MemorySelector(self.world, self.memory, config=MemoryConfig())
            self.assertTrue(selector.select(rule))
        rules = HouseRules(self.memory, self.world)
        self.safety = SafetyController(
            self.robot, speed=40, allow_autonomous=True, events=self.bus, house_rules=rules
        )
        self.addCleanup(self.safety.close)
        self.safety.start()
        self.arbiter = Arbiter(self.safety)
        self.addCleanup(self.arbiter.close)
        camera = Camera(self.robot, min_interval_s=0, max_per_minute=0)
        scanner = Scanner(self.robot, camera, None, settle_s=0.01, observation_wait_s=0)

        class Policy:
            def set_exploring(self, exploring):
                pass

        config = ExploreConfig(speed=40, step_ms=800, scan_every_steps=3, max_duration_s=60, max_steps=10000)
        self.explorer = Explorer(
            self.arbiter, self.safety, scanner, Policy(), config=config, head_only_interval_s=0.05
        )
        self.addCleanup(self.explorer.stop)
        return self.explorer

    def in_bedroom_forward(self):
        return [d for d in self.drives if d[0] > 0 and d[1] == "bedroom"]

    def turn_backs(self):
        return [e for e in self.memory.episodes.recent(50) if e.kind == "turned_back"]

    def test_without_a_rule_she_drives_into_the_bedroom(self):
        explorer = self.build(None)
        self.assertTrue(explorer.start())
        self.assertTrue(wait_for(lambda: len(self.in_bedroom_forward()) >= 2))
        explorer.stop()
        self.assertEqual(self.turn_backs(), [])

    def test_a_rule_taught_by_voice_keeps_exploration_out_of_the_bedroom(self):
        explorer = self.build("Non entrare in camera da letto")
        self.assertTrue(explorer.start())
        self.assertTrue(wait_for(lambda: self.turn_backs()))
        # Let her carry on for a while: she may come back to the door, but never drives on inside.
        steps = explorer.steps
        self.assertTrue(wait_for(lambda: explorer.steps >= steps + 6, timeout=40))
        explorer.stop()
        self.assertEqual(self.in_bedroom_forward(), [])
        self.assertGreaterEqual(explorer.avoided, 1)
        episode = self.turn_backs()[0]
        self.assertEqual(episode.location, "bedroom")
        self.assertIn("bedroom", episode.summary)


if __name__ == "__main__":
    unittest.main()
