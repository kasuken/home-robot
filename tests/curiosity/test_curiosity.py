import unittest

from laika.camera import Camera
from laika.config import CuriosityConfig
from laika.curiosity import InterestGate, InterestScore, InterestScorer
from laika.events import EventBus, EventType
from laika.house import Pose, default_house
from laika.observations import Observation, ObservationLog, Observer, SeenObject, find_previous
from laika.simulator import SimulatedRobot
from laika.tools import Memory
from laika.world_model import WorldModel, WorldModelUpdater
from tests.fakes import FakeClock

COMPONENTS = (
    "novelty",
    "environmental_change",
    "uncertainty",
    "person",
    "unknown_object",
    "unusual_location",
    "movement",
    "sound",
    "user_relevance",
)


def obs(*names, location="living_room", people=0, confidence=1.0, pan=0.0):
    objects = tuple(n if isinstance(n, SeenObject) else SeenObject(n) for n in names)
    return Observation(
        description="a view",
        objects=objects,
        location=location,
        people=people,
        confidence=confidence,
        camera_pan=pan,
    )


def only(name, **config):
    """A config where only `name` has weight 1 (and the rest 0)."""
    weights = {n: 0.0 for n in COMPONENTS}
    weights[name] = 1.0
    weights.update(config)
    return CuriosityConfig(**weights)


def scorer(world=None, memory=None, config=None, **weights):
    return InterestScorer(world or WorldModel(), memory=memory, config=config or CuriosityConfig(**weights))


def score_of(total, *reasons):
    return InterestScore(total=total, components={}, reasons=list(reasons))


class ComponentTests(unittest.TestCase):
    def test_novelty_is_the_share_of_objects_never_seen(self):
        world = WorldModel()
        world.observe(["sofa"], "living_room", timestamp=1.0)
        score = scorer(world).score(obs("sofa", "backpack"))
        self.assertEqual(score.components["novelty"], 0.5)
        self.assertEqual(score.reasons, ["new object: backpack"])

    def test_novelty_is_zero_without_objects(self):
        score = scorer().score(obs())
        self.assertEqual(score.components["novelty"], 0.0)
        self.assertEqual(score.reasons, [])

    def test_environmental_change_counts_objects_last_known_in_another_room(self):
        world = WorldModel()
        world.observe(["keys"], "kitchen", timestamp=1.0)
        world.observe(["lamp"], "office", timestamp=1.0)
        score = scorer(world).score(obs("keys", "lamp", location="office"))
        self.assertEqual(score.components["environmental_change"], 0.5)
        self.assertEqual(score.reasons, ["moved: keys from kitchen to office"])

    def test_environmental_change_needs_a_location(self):
        world = WorldModel()
        world.observe(["keys"], "kitchen", timestamp=1.0)
        score = scorer(world).score(obs("keys", location=None))
        self.assertEqual(score.components["environmental_change"], 0.0)

    def test_uncertainty_is_one_minus_confidence(self):
        self.assertAlmostEqual(scorer().score(obs(confidence=0.6)).components["uncertainty"], 0.4)
        self.assertEqual(scorer().score(obs(confidence=1.0)).components["uncertainty"], 0.0)

    def test_person_is_one_when_someone_is_in_view(self):
        score = scorer().score(obs(people=2))
        self.assertEqual(score.components["person"], 1.0)
        self.assertEqual(score.reasons, ["person nearby"])
        self.assertEqual(scorer().score(obs()).components["person"], 0.0)

    def test_unknown_object_uses_confidence_and_vague_names(self):
        objects = [SeenObject("mug", confidence=0.3), SeenObject("thing"), SeenObject("lamp", confidence=0.9)]
        score = scorer().score(obs(*objects))
        self.assertAlmostEqual(score.components["unknown_object"], 2 / 3)
        self.assertIn("unknown object: mug", score.reasons)
        self.assertIn("unknown object: thing", score.reasons)
        self.assertNotIn("unknown object: lamp", score.reasons)

    def test_unknown_object_confidence_is_configurable(self):
        objects = [SeenObject("mug", confidence=0.3)]
        lenient = scorer(unknown_object_confidence=0.2).score(obs(*objects))
        self.assertEqual(lenient.components["unknown_object"], 0.0)

    def test_unusual_location_when_an_object_is_not_where_it_usually_is(self):
        world = WorldModel()
        # The mug goes living room -> office -> living room, so living room is its usual place.
        world.observe(["mug"], "living_room", timestamp=1.0)
        world.observe(["mug"], "office", timestamp=2.0)
        world.observe(["mug"], "living_room", timestamp=3.0)
        score = scorer(world).score(obs("mug", location="kitchen"))
        self.assertEqual(score.components["unusual_location"], 1.0)
        self.assertIn("unusual location: mug in kitchen (usually living room)", score.reasons)
        at_home = scorer(world).score(obs("mug", location="living_room"))
        self.assertEqual(at_home.components["unusual_location"], 0.0)

    def test_one_move_is_not_a_habit(self):
        world = WorldModel()
        world.observe(["backpack"], "living_room", timestamp=1.0)
        world.observe(["backpack"], "kitchen", timestamp=2.0)
        score = scorer(world).score(obs("backpack", location="office"))
        self.assertEqual(score.components["unusual_location"], 0.0)

    def test_movement_compares_with_the_previous_look_at_the_same_view(self):
        previous = obs("sofa", people=0, pan=-40.0)
        score = scorer().score(obs("sofa", "lamp", people=1, pan=-35.0), previous=previous)
        # Lamp appeared and the people count changed: 2 changes over 3 slots.
        self.assertAlmostEqual(score.components["movement"], 2 / 3)
        self.assertIn("appeared: lamp", score.reasons)
        self.assertIn("people count changed: 0 to 1", score.reasons)

    def test_movement_needs_a_comparable_previous_look(self):
        self.assertEqual(scorer().score(obs("sofa", "lamp")).components["movement"], 0.0)
        far = obs("sofa", pan=-60.0)
        self.assertEqual(scorer().score(obs("lamp"), previous=far).components["movement"], 0.0)
        other_room = obs("sofa", location="kitchen")
        self.assertEqual(scorer().score(obs("lamp"), previous=other_room).components["movement"], 0.0)

    def test_movement_reports_disappeared_objects(self):
        score = scorer().score(obs("sofa"), previous=obs("sofa", "keys"))
        self.assertEqual(score.components["movement"], 0.5)
        self.assertIn("disappeared: keys", score.reasons)

    def test_sound_is_always_zero_even_with_a_weight(self):
        score = scorer(config=only("sound")).score(obs("sofa", people=3))
        self.assertEqual(score.components["sound"], 0.0)
        self.assertEqual(score.total, 0.0)

    def test_user_relevance_from_memory_or_a_note(self):
        world = WorldModel()
        world.learn("guitars", room="living_room", note="belongs to Leonardo", timestamp=1.0)
        memory = Memory()
        memory.add("the backpack is for school")
        score = scorer(world, memory).score(obs("guitars", "backpack", "lamp"))
        self.assertAlmostEqual(score.components["user_relevance"], 2 / 4)
        self.assertEqual(
            score.reasons[-2:],
            ["relevant to the user: backpack", "relevant to the user: guitars"],
        )

    def test_user_relevance_counts_a_remembered_room(self):
        memory = Memory()
        memory.add("the spoons are in the kitchen drawer")
        score = scorer(memory=memory).score(obs("lamp", location="kitchen"))
        self.assertAlmostEqual(score.components["user_relevance"], 0.5)
        self.assertIn("relevant to the user: kitchen", score.reasons)

    def test_user_relevance_is_zero_without_memory_or_notes(self):
        self.assertEqual(scorer().score(obs("lamp")).components["user_relevance"], 0.0)


class TotalTests(unittest.TestCase):
    def test_total_is_the_weighted_average_of_the_components(self):
        # Novelty 1 and person 1, both weight 1: (1 + 1) / (1 + 1).
        both = only("novelty", person=1.0)
        self.assertEqual(scorer(config=both).score(obs("backpack", people=1)).total, 1.0)
        # Novelty 1 with weight 1 and person 0 with weight 3: 1 / 4.
        heavy_person = only("novelty", person=3.0)
        self.assertEqual(scorer(config=heavy_person).score(obs("backpack")).total, 0.25)

    def test_scaling_all_weights_does_not_change_the_total(self):
        world = WorldModel()
        world.observe(["keys"], "kitchen", timestamp=1.0)
        target = obs("keys", "backpack", location="office", people=1, confidence=0.7)
        default = scorer(world).score(target)
        base = CuriosityConfig()
        scaled_config = CuriosityConfig(
            **{n: 10 * getattr(base, n) for n in COMPONENTS}
        )
        scaled = scorer(world, config=scaled_config).score(target)
        self.assertGreater(default.total, 0.0)
        self.assertLessEqual(default.total, 1.0)
        self.assertAlmostEqual(scaled.total, default.total)

    def test_the_default_weights_give_a_total_between_zero_and_one(self):
        world = WorldModel()
        world.observe(["keys"], "kitchen", timestamp=1.0)
        score = scorer(world).score(obs("keys", "thing", location="office", people=2, confidence=0.1))
        self.assertTrue(0.0 <= score.total <= 1.0)

    def test_zero_weights_give_zero(self):
        config = CuriosityConfig(**{n: 0.0 for n in COMPONENTS})
        self.assertEqual(scorer(config=config).score(obs("keys", people=2)).total, 0.0)

    def test_components_are_each_between_zero_and_one_in_a_fixed_order(self):
        world = WorldModel()
        world.observe(["keys"], "kitchen", timestamp=1.0)
        score = scorer(world).score(
            obs("keys", "thing", location="office", people=2, confidence=0.1), previous=obs("sofa")
        )
        for name, value in score.components.items():
            self.assertTrue(0.0 <= value <= 1.0, name)
        self.assertEqual(list(score.components), list(COMPONENTS))

    def test_to_dict_is_json_friendly_and_rounded(self):
        data = scorer().score(obs("backpack", people=1, confidence=0.9)).to_dict()
        self.assertEqual(set(data), {"total", "components", "reasons"})
        self.assertEqual(data["reasons"], ["new object: backpack", "person nearby", "uncertain view"])
        self.assertEqual(data["components"]["uncertainty"], 0.1)


class DeterminismTests(unittest.TestCase):
    def scenario(self):
        world = WorldModel()
        world.observe(["keys", "sofa"], "kitchen", timestamp=1.0)
        world.observe(["keys"], "office", timestamp=2.0)
        world.learn("guitars", room="office", note="belongs to Leonardo", timestamp=3.0)
        memory = Memory()
        memory.add("keys are usually by the door")
        target = obs("keys", "backpack", "thing", "guitars", location="living_room", people=1, confidence=0.4)
        previous = obs("sofa", location="living_room", pan=5.0)
        return world, memory, target, previous

    def test_same_inputs_give_identical_scores_and_reason_order(self):
        world, memory, target, previous = self.scenario()
        first = scorer(world, memory).score(target, previous=previous)
        self.assertEqual(first.reasons, sorted(first.reasons))
        for _ in range(5):
            again = scorer(world, memory).score(target, previous=previous)
            self.assertEqual(again, first)
            self.assertEqual(again.reasons, first.reasons)

    def test_scoring_never_changes_the_world_model_or_memory(self):
        world, memory, target, previous = self.scenario()
        before, facts = world.to_dict(), len(memory)
        instance = scorer(world, memory)
        results = [instance.score(target, previous=previous) for _ in range(3)]
        self.assertEqual(results[0], results[2])
        self.assertEqual(world.to_dict(), before)
        self.assertEqual(len(memory), facts)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus()
        self.bus.start()
        self.addCleanup(self.bus.close)
        self.events = []
        self.bus.subscribe(self.events.append, EventType.INTERESTING_CHANGE_DETECTED)
        self.clock = FakeClock()

    def gate(self, **kwargs):
        return InterestGate(CuriosityConfig(**kwargs), self.bus, clock=self.clock)

    def published(self):
        self.assertTrue(self.bus.drain())
        return [event.data for event in self.events]

    def test_below_threshold_publishes_nothing(self):
        announced = self.gate(threshold=0.35).consider(obs("backpack"), score_of(0.34, "new object: backpack"))
        self.assertFalse(announced)
        self.assertEqual(self.published(), [])

    def test_at_threshold_publishes_the_event(self):
        observation = obs("backpack", "sofa", location="living_room")
        score = score_of(0.35, "new object: backpack", "new object: sofa")
        self.assertTrue(self.gate(threshold=0.35).consider(observation, score))
        (data,) = self.published()
        self.assertEqual(data["observation_id"], observation.id)
        self.assertEqual(data["location"], "living_room")
        self.assertEqual(data["total"], 0.35)
        self.assertEqual(data["reasons"], ["new object: backpack", "new object: sofa"])
        self.assertEqual(data["objects"], ["backpack", "sofa"])

    def test_the_same_reason_is_suppressed_during_its_cooldown(self):
        gate = self.gate(cooldown_s=120)
        score = score_of(0.5, "person nearby")
        self.assertTrue(gate.consider(obs(people=1), score))
        self.clock.advance(119)
        self.assertFalse(gate.consider(obs(people=1), score))
        self.assertEqual(len(self.published()), 1)

    def test_the_cooldown_expires_and_the_reason_fires_again(self):
        gate = self.gate(cooldown_s=120)
        score = score_of(0.5, "person nearby")
        gate.consider(obs(people=1), score)
        self.clock.advance(120)
        self.assertTrue(gate.consider(obs(people=1), score))
        self.assertEqual(len(self.published()), 2)

    def test_a_new_reason_fires_while_an_old_one_is_cooling_down(self):
        gate = self.gate(cooldown_s=120)
        gate.consider(obs(people=1), score_of(0.5, "person nearby"))
        self.clock.advance(60)
        mixed = score_of(0.5, "person nearby", "new object: lamp")
        self.assertTrue(gate.consider(obs(people=1), mixed))
        # The old reason did not restart its cooldown at t=60, so it is free again at t=120.
        self.clock.advance(60)
        self.assertTrue(gate.consider(obs(people=1), score_of(0.5, "person nearby")))
        self.assertEqual(len(self.published()), 3)

    def test_zero_cooldown_never_suppresses(self):
        gate = self.gate(cooldown_s=0)
        score = score_of(0.5, "person nearby")
        self.assertTrue(gate.consider(obs(people=1), score))
        self.assertTrue(gate.consider(obs(people=1), score))
        self.assertEqual(len(self.published()), 2)

    def test_the_gate_works_without_a_bus(self):
        gate = InterestGate(CuriosityConfig(), None, clock=self.clock)
        self.assertTrue(gate.consider(obs(people=1), score_of(0.5, "person nearby")))


class PreviousLookupTests(unittest.TestCase):
    def test_finds_the_newest_same_view_within_twenty_degrees(self):
        older = obs("sofa", pan=0.0)
        newer = obs("lamp", pan=-5.0)
        self.assertIs(find_previous([newer, older], obs("tv", pan=10.0)), newer)

    def test_ignores_other_locations_and_wider_pans(self):
        self.assertIsNone(find_previous([obs("sofa", location="kitchen")], obs("tv")))
        self.assertIsNone(find_previous([obs("sofa", pan=-25.0)], obs("tv", pan=0.0)))

    def test_the_observation_log_keeps_the_score_with_each_entry(self):
        log = ObservationLog(max_items=2)
        first, second, third = obs("a"), obs("b"), obs("c")
        score = score_of(0.4, "new object: a")
        log.add(first, score)
        log.add(second)
        self.assertEqual(log.score_of(first.id), score)
        log.add(third)
        self.assertIsNone(log.score_of(first.id))
        self.assertEqual(log.recent(2), [third, second])


class ObserverOrderingTests(unittest.TestCase):
    """The real Observer and WorldModelUpdater, with the simulator's living room."""

    def setUp(self):
        self.clock = FakeClock()
        self.bus = EventBus()
        self.bus.start()
        self.addCleanup(self.bus.close)
        self.world = WorldModel()
        self.updater = WorldModelUpdater(self.world, self.bus)
        self.addCleanup(self.updater.close)
        self.created = []
        self.bus.subscribe(self.created.append, EventType.OBSERVATION_CREATED)
        self.changes = []
        self.bus.subscribe(self.changes.append, EventType.INTERESTING_CHANGE_DETECTED)

    def build(self, threshold=0.35):
        config = CuriosityConfig(threshold=threshold)
        self.observer = Observer(
            self.bus,
            default_location=lambda: self.world.current_room,
            scorer=InterestScorer(self.world, memory=Memory(), config=config),
            gate=InterestGate(config, self.bus, clock=self.clock),
        )
        self.addCleanup(self.observer.close)
        robot = SimulatedRobot(
            verbose=False, house=default_house(), start=Pose(250, 200, 90), events=self.bus, clock=self.clock
        )
        self.camera = Camera(robot, events=self.bus, min_interval_s=0, clock=self.clock)
        self.observer.attach(self.camera)
        self.world.enter_room(robot.current_room())
        robot.look(-90, 0)

    def look_once(self):
        image = self.camera.capture()
        observation = self.observer.wait_for(image)
        self.assertIsNotNone(observation)
        self.assertTrue(self.bus.drain())
        return observation

    def test_a_new_object_is_novel_the_first_time_and_not_the_second(self):
        self.build()
        first = self.look_once()
        second = self.look_once()

        first_score = self.observer.log.score_of(first.id)
        second_score = self.observer.log.score_of(second.id)
        self.assertEqual(first_score.components["novelty"], 1.0)
        self.assertIn("new object: sofa", first_score.reasons)
        self.assertEqual(second_score.components["novelty"], 0.0)
        self.assertNotIn("new object: sofa", second_score.reasons)

    def test_the_second_look_is_compared_with_the_first(self):
        self.build()
        self.look_once()
        second = self.look_once()
        self.assertEqual(self.observer.log.score_of(second.id).components["movement"], 0.0)

    def test_the_event_carries_the_score_and_the_gate_announces_once(self):
        self.build(threshold=0.2)
        first = self.look_once()
        second = self.look_once()
        created = {event.data["id"]: event.data for event in self.created}
        self.assertEqual(created[first.id]["interest"], self.observer.log.score_of(first.id).to_dict())
        self.assertIn("interest", created[second.id])
        self.assertEqual(len(self.changes), 1)
        announced = self.changes[0].data
        self.assertEqual(announced["observation_id"], first.id)
        self.assertEqual(announced["location"], "living_room")
        self.assertIn("new object: sofa", announced["reasons"])

    def test_without_a_scorer_the_event_has_no_interest_data(self):
        observer = Observer(self.bus)
        self.addCleanup(observer.close)
        robot = SimulatedRobot(
            verbose=False, house=default_house(), start=Pose(250, 200, 90), events=self.bus, clock=self.clock
        )
        camera = Camera(robot, events=self.bus, min_interval_s=0, clock=self.clock)
        observer.attach(camera)
        observer.wait_for(camera.capture())
        self.assertTrue(self.bus.drain())
        self.assertNotIn("interest", self.created[-1].data)
