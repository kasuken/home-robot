"""The investigate behaviour (#29): the house simulator with the real arbiter, safety, camera, observer and scorer."""

import contextlib
import io
import math
import time
import unittest

from laika.agent import Agent
from laika.behaviour import Arbiter, Priority
from laika.camera import Camera
from laika.config import CuriosityConfig, ExploreConfig, InvestigateConfig
from laika.curiosity import InterestGate, InterestScorer
from laika.events import EventBus, EventType
from laika.explore import Explorer
from laika.house import Pose, Thing, default_house
from laika.investigate import (
    MAX_QUESTIONS,
    Investigator,
    PendingQuestions,
    Target,
    focus_object,
    question_text,
)
from laika.observations import Observer
from laika.prompts import build_tool_system_prompt
from laika.safety import SafetyController
from laika.scan import Scanner
from laika.memory import Memory
from laika.tools import ToolRegistry, Turn
from laika.world_model import WorldModel, WorldModelUpdater
from tests.agent.test_agent import ScriptedClient, calls, text
from tests.agent.test_tools import call
from tests.fakes import FakeClock, LookRecordingRobot

P = Priority
FACING_WEST = Pose(300, 200, 180)  # from the living room: sofa, television and backpack in view
CURIOUS = dict(threshold=0.01, cooldown_s=0.0)


def wait_for(condition, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


class StubActions:
    def cancel_pending(self):
        pass


class PolicySpy:
    def set_exploring(self, exploring):
        pass


class InvestigateCase(unittest.TestCase):
    def build(
        self,
        *,
        start=FACING_WEST,
        allow_autonomous=True,
        config=None,
        clock=None,
        curiosity=None,
        **options,
    ):
        out = io.StringIO()
        redirect = contextlib.redirect_stdout(out)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)
        self.stdout = out
        self.house = default_house()
        self.bus = EventBus()
        self.bus.start()
        self.addCleanup(self.bus.close)
        self.events = []
        self.bus.subscribe(self.events.append)
        self.world = WorldModel()
        self.addCleanup(WorldModelUpdater(self.world, self.bus).close)
        self.memory = Memory()
        self.robot = LookRecordingRobot(house=self.house, start=start)
        self.world.enter_room(self.robot.current_room())
        self.camera = Camera(self.robot, None, self.bus, min_interval_s=0, max_per_minute=0, save_images=False)
        curiosity = curiosity or CuriosityConfig(**CURIOUS)
        self.observer = Observer(
            self.bus,
            scorer=InterestScorer(self.world, self.memory, curiosity),
            gate=InterestGate(curiosity, self.bus),
        )
        self.observer.attach(self.camera)
        self.addCleanup(self.observer.close)
        self.safety = SafetyController(self.robot, speed=25, allow_autonomous=allow_autonomous)
        self.addCleanup(self.safety.close)
        self.safety.start()
        self.arbiter = Arbiter(self.safety, StubActions(), lease_timeout_s=2.0)
        self.addCleanup(self.arbiter.close)
        options.setdefault("settle_s", 0.01)
        options.setdefault("body_wait_s", 0.3)
        self.investigator = Investigator(
            self.arbiter,
            self.safety,
            self.robot,
            self.camera,
            self.observer,
            self.world,
            self.memory,
            self.bus,
            config or InvestigateConfig(),
            threshold=curiosity.threshold,
            clock=clock or time.monotonic,
            **options,
        )
        self.addCleanup(self.investigator.close)
        return self.investigator

    # --- helpers ---------------------------------------------------------------

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

    def investigate_captures(self):
        self.bus.drain()
        return [
            e.data for e in self.events
            if e.type is EventType.IMAGE_CAPTURED and e.data.get("reason") == "investigate"
        ]

    def positive_drives(self):
        return [s for s in self.robot.speeds if s > 0]

    def prime(self, pan=0.0, tilt=0.0):
        """Let the world model learn what is in view at this head pose, then put the head back."""
        before = self.robot.head_pose()
        self.see(pan)  # a tilt of 5 is what `see` uses; the scene does not depend on tilt
        self.robot.look(*before)
        self.robot.poses.clear()
        self.robot.speeds.clear()
        del self.events[:]

    def run_one(self, target, prime=True):
        if prime:
            self.prime(target.pan or 0.0)
        self.investigator.start()
        self.assertTrue(self.investigator.enqueue(target))
        self.assertTrue(self.investigator.wait_idle())
        return self.investigator.results[-1]

    def box_target(self, pan=0.0, tilt=0.0, **kwargs):
        return Target("box", "living_room", pan=pan, tilt=tilt, new=True, **kwargs)


class NewObjectTests(InvestigateCase):
    def test_with_the_default_settings_a_new_object_in_a_familiar_room_is_investigated(self):
        # The issue's done-when with the shipped thresholds: the living room is familiar
        # (known for longer than familiar_room_after_s), then something new appears in it.
        investigator = self.build(curiosity=CuriosityConfig())
        long_ago = time.time() - 2 * CuriosityConfig().familiar_room_after_s
        self.world.observe(["sofa", "television", "backpack"], "living_room", timestamp=long_ago)
        investigator.start()
        self.robot.look(10, 5)
        self.see()  # nothing new: not interesting
        self.assertEqual(self.investigate_captures(), [])
        self.place("camera_stabilizer", pan=10.0, distance=100)
        self.see()
        self.assertTrue(investigator.wait_idle())
        self.assertIn(EventType.INTERESTING_CHANGE_DETECTED, [e.type for e in self.events])
        self.assertEqual(investigator.results[-1].object, "camera_stabilizer")
        (question,) = investigator.questions.pending()
        self.assertEqual(question.object, "camera_stabilizer")

    def test_with_the_default_settings_discovering_a_room_is_not_investigated(self):
        investigator = self.build(curiosity=CuriosityConfig())
        investigator.start()
        self.robot.look(10, 5)
        self.see()  # every object is new, but the room is new too: that is just discovery
        self.assertTrue(investigator.wait_idle())
        self.assertNotIn(EventType.INTERESTING_CHANGE_DETECTED, [e.type for e in self.events])
        self.assertEqual(self.investigate_captures(), [])

    def test_a_new_object_triggers_an_investigation(self):
        investigator = self.build()
        self.robot.look(10, 5)
        self.see()  # the world model learns the sofa, the television and the backpack
        investigator.start()
        self.assertEqual(self.investigate_captures(), [])
        self.place("camera_stabilizer", pan=10.0, distance=100)
        self.see()  # novel: InterestingChangeDetected -> the investigator
        self.assertTrue(investigator.wait_idle())
        types = [e.type for e in self.events]
        self.assertIn(EventType.INTERESTING_CHANGE_DETECTED, types)
        result = investigator.results[-1]
        self.assertEqual((result.object, result.source, result.status), ("camera_stabilizer", "event", "done"))
        self.assertTrue(result.confirmed)
        shots = self.investigate_captures()
        self.assertGreaterEqual(len(shots), 1)
        self.assertEqual(shots[0]["pan"], 10.0)  # looking where it was seen
        self.assertIn("camera_stabilizer", shots[0]["scene"])
        # A question is queued, in Italian, about it; nothing was said out loud.
        (question,) = investigator.questions.pending()
        self.assertEqual(question.object, "camera_stabilizer")
        self.assertEqual(question.text, "Ho visto una cosa nuova in soggiorno: camera_stabilizer. Che cos'è?")
        self.assertEqual(self.robot.spoken, [])
        self.assertIn("[laika] investigating camera_stabilizer in living room", self.stdout.getvalue())
        self.assertIsNone(self.arbiter.holder)

    def test_the_head_points_where_the_object_was_seen(self):
        investigator = self.build()
        self.place("box", pan=25.0)
        self.robot.look(0, 0)
        result = self.run_one(self.box_target(pan=25.0, tilt=5.0))
        self.assertTrue(result.confirmed)
        self.assertIn((25.0, 5.0), self.robot.poses)
        self.assertEqual(self.investigate_captures()[0]["pan"], 25.0)
        self.assertEqual(self.robot.head_pose(), (0.0, 0.0))  # and back again afterwards
        self.assertEqual(len(investigator.questions), 1)

    def test_an_object_that_is_gone_is_not_asked_about(self):
        investigator = self.build()
        result = self.run_one(self.box_target())  # nothing there
        self.assertEqual(result.status, "not_seen")
        self.assertFalse(result.confirmed)
        self.assertEqual(len(investigator.questions), 0)
        self.assertEqual(len(self.memory), 0)

    def test_an_object_the_user_named_is_remembered_not_asked_about(self):
        investigator = self.build()
        self.place("box", distance=100)
        self.world.learn("box", room="living_room", note="la scatola dei giochi")
        result = self.run_one(self.box_target())
        self.assertIsNone(result.question)
        self.assertEqual(len(investigator.questions), 0)
        (stored,) = self.memory.facts.search("box")
        fact = stored.text
        self.assertEqual((stored.subject, stored.kind, stored.source), ("box", "location", "investigation"))
        self.assertTrue(fact.startswith("Saw box next to "))
        self.assertTrue(fact.endswith(" in the living room"))
        self.assertIsNotNone(result.remembered)

    def test_a_doubtful_object_gets_a_question_even_when_known(self):
        investigator = self.build(unknown_confidence=1.5)  # everything the simulator sees is "doubtful"
        self.place("box", distance=100)
        result = self.run_one(Target("box", "living_room", pan=0.0, new=False))
        self.assertIsNotNone(result.question)
        self.assertEqual(len(investigator.questions), 1)


class MovedObjectTests(InvestigateCase):
    def test_a_known_object_that_moved_is_remembered_without_a_question(self):
        investigator = self.build(start=Pose(300, 550, 180))
        self.world.observe(["backpack"], "living_room", timestamp=1.0)
        self.see()  # the office: desk, monitor and chair become known
        investigator.start()
        self.house.move_thing("backpack", 200, 550)
        self.see()
        self.assertTrue(investigator.wait_idle())
        reasons = [e.data["reasons"] for e in self.events if e.type is EventType.INTERESTING_CHANGE_DETECTED]
        self.assertTrue(any("moved: backpack from living room to office" in r for r in reasons))
        result = investigator.results[-1]
        self.assertEqual((result.object, result.status), ("backpack", "done"))
        self.assertIsNone(result.question)
        self.assertEqual(len(investigator.questions), 0)
        (stored,) = self.memory.facts.search("backpack")
        fact = stored.text
        self.assertTrue(fact.startswith("Saw backpack"))
        self.assertTrue(fact.endswith("in the office"))
        self.assertIn(" next to ", fact)


class ExplorationHandOverTests(InvestigateCase):
    def test_a_new_object_during_exploration_pre_empts_it_and_it_resumes(self):
        # No moving closer: a new vantage point would show new things and muddy the count.
        investigator = self.build(config=InvestigateConfig(max_approach_moves=0))
        scanner = Scanner(self.robot, self.camera, self.observer, settle_s=0.01, observation_wait_s=2.0)
        scanner.scan()
        self.bus.drain()
        investigator.start()
        explorer = Explorer(
            self.arbiter,
            self.safety,
            scanner,
            PolicySpy(),
            config=ExploreConfig(scan_every_steps=1, max_duration_s=30.0),
            on_scan=investigator.consider_scan,
            drive=False,
            head_only_interval_s=0.3,
            body_wait_s=5.0,
        )
        self.addCleanup(explorer.stop)
        self.place("box", pan=0.0, distance=100)
        self.assertTrue(explorer.start())
        self.assertTrue(wait_for(lambda: any(r.object == "box" for r in list(investigator.results))))
        self.assertTrue(wait_for(lambda: "exploration resumed" in self.stdout.getvalue()))
        scans = explorer.scans
        self.assertTrue(wait_for(lambda: explorer.scans > scans))  # exploring again
        explorer.stop()
        output = self.stdout.getvalue()
        self.assertIn("behaviour exploration preempted by investigation", output)
        self.assertLess(output.index("preempted by investigation"), output.index("exploration resumed"))
        self.assertEqual(len(investigator.questions), 1)
        self.assertEqual([r.object for r in investigator.results].count("box"), 1)


class ApproachTests(InvestigateCase):
    def test_it_moves_closer_when_the_object_is_ahead_and_far(self):
        self.build()
        self.place("box", pan=0.0, distance=200)
        start = self.robot.pose()
        result = self.run_one(self.box_target())
        self.assertEqual(result.approached, 1)
        self.assertTrue(self.positive_drives())
        self.assertLess(self.robot.pose().x, start.x)  # facing west
        self.assertEqual(len(self.investigate_captures()), 2)  # before and after the move
        self.assertTrue(result.confirmed)

    def moves_with_limit(self, limit):
        self.build(config=InvestigateConfig(max_approach_moves=limit))
        self.place("box", pan=0.0, distance=300)
        result = self.run_one(self.box_target())
        self.assertEqual(result.approached, limit)
        self.assertEqual(len(self.positive_drives()), limit)
        self.assertEqual(len(self.investigate_captures()), 1 + limit)

    def test_it_never_moves_more_than_the_configured_number_of_times(self):
        self.moves_with_limit(1)

    def test_two_moves_when_two_are_allowed(self):
        self.moves_with_limit(2)

    def test_no_moves_when_none_are_allowed(self):
        self.moves_with_limit(0)

    def test_it_does_not_move_when_the_object_is_not_ahead(self):
        self.build()
        self.place("box", pan=40.0, distance=200)
        result = self.run_one(self.box_target(pan=40.0))
        self.assertTrue(result.confirmed)
        self.assertEqual(result.approached, 0)
        self.assertEqual(result.drive_note, "not ahead")
        self.assertEqual(self.positive_drives(), [])

    def test_it_does_not_move_when_it_is_close_enough(self):
        self.build()
        self.place("box", pan=0.0, distance=60)
        result = self.run_one(self.box_target())
        self.assertTrue(result.confirmed)
        self.assertEqual(result.approached, 0)
        self.assertEqual(self.positive_drives(), [])

    def test_with_autonomous_driving_off_no_drive_reaches_the_robot(self):
        self.build(allow_autonomous=False)
        self.place("box", pan=0.0, distance=200)
        result = self.run_one(self.box_target())
        self.assertEqual(result.approached, 0)
        self.assertEqual(result.drive_note, "autonomous_disabled")
        self.assertEqual([s for s in self.robot.speeds if s != 0], [])
        # Looking and taking the picture still happen, and the question is still queued.
        self.assertTrue(result.confirmed)
        self.assertEqual(len(self.investigate_captures()), 1)
        self.assertIn((0.0, 0.0), self.robot.poses)
        self.assertEqual(len(self.investigator.questions), 1)


class PreemptionTests(InvestigateCase):
    def test_a_conversation_aborts_the_investigation_cleanly(self):
        investigator = self.build()
        self.place("box", pan=0.0, distance=200)
        real = self.camera.capture
        taken = []

        def capture(reason="request"):
            if reason == "investigate":
                taken.append(reason)
            if len(taken) == 2:  # the picture after moving closer: someone says "ehi Laika"
                self.conversation = self.arbiter.request("conversation", P.CONVERSATION)
            return real(reason)

        self.camera.capture = capture
        result = self.run_one(self.box_target())
        self.assertEqual(result.status, "aborted")
        self.assertEqual(result.reason, "preempted")
        self.assertIsNone(result.question)
        self.assertEqual(len(investigator.questions), 0)
        self.assertEqual(len(self.memory), 0)
        self.assertIs(self.arbiter.holder, self.conversation)
        self.assertEqual(self.robot.speed, 0)
        self.assertEqual(len(self.positive_drives()), 1)  # it moved once, before the interruption

    def test_a_busy_body_skips_the_investigation(self):
        self.build(body_wait_s=0.2)
        chat = self.arbiter.request("conversation", P.CONVERSATION)
        self.place("box", pan=0.0, distance=200)
        self.investigator.start()
        self.assertTrue(self.investigator.enqueue(self.box_target()))
        self.assertTrue(self.investigator.wait_idle())
        self.assertEqual(len(self.investigator.results), 0)
        self.assertEqual(self.investigate_captures(), [])
        self.assertIs(self.arbiter.holder, chat)
        self.assertIn("skipped", self.stdout.getvalue())


class CapTests(InvestigateCase):
    def test_per_hour_cap(self):
        clock = FakeClock(1000.0)
        investigator = self.build(config=InvestigateConfig(max_per_hour=2, object_cooldown_s=0), clock=clock)
        self.prime()
        investigator.start()
        for name in ("a", "b", "c"):
            investigator.enqueue(Target(name, "living_room"))
            self.assertTrue(investigator.wait_idle())
        self.assertEqual([r.object for r in investigator.results], ["a", "b"])
        self.assertIn("hourly limit", self.stdout.getvalue())
        clock.advance(3601)
        investigator.enqueue(Target("d", "living_room"))
        self.assertTrue(investigator.wait_idle())
        self.assertEqual([r.object for r in investigator.results], ["a", "b", "d"])

    def test_per_object_cooldown(self):
        clock = FakeClock(1000.0)
        investigator = self.build(config=InvestigateConfig(object_cooldown_s=300), clock=clock)
        self.prime()
        investigator.start()
        self.assertTrue(investigator.enqueue(Target("box", "living_room")))
        self.assertTrue(investigator.wait_idle())
        self.assertFalse(investigator.enqueue(Target("box", "living_room")))
        clock.advance(299)
        self.assertFalse(investigator.enqueue(Target("box", "living_room")))
        self.assertTrue(investigator.enqueue(Target("lamp", "living_room")))  # another object is fine
        clock.advance(2)
        self.assertTrue(investigator.wait_idle())
        self.assertTrue(investigator.enqueue(Target("box", "living_room")))
        self.assertTrue(investigator.wait_idle())
        self.assertEqual([r.object for r in investigator.results], ["box", "lamp", "box"])

    def test_the_queue_is_bounded_and_the_newest_replaces_the_oldest(self):
        investigator = self.build()  # the worker is not started: nothing leaves the queue
        for name in "abcde":
            self.assertTrue(investigator.enqueue(Target(name, None)))
        self.assertEqual([t.object for t in investigator._queue], ["c", "d", "e"])
        self.assertEqual(investigator.dropped, 2)

    def test_the_same_object_or_observation_is_not_queued_twice(self):
        investigator = self.build()
        self.assertTrue(investigator.enqueue(Target("a", None, observation_id="obs-1")))
        self.assertFalse(investigator.enqueue(Target("a", None, observation_id="obs-2")))
        self.assertFalse(investigator.enqueue(Target("b", None, observation_id="obs-1")))
        self.assertEqual(len(investigator._queue), 1)

    def test_a_disabled_investigator_queues_nothing(self):
        investigator = self.build(config=InvestigateConfig(enabled=False))
        self.assertFalse(investigator.enqueue(Target("a", None)))

    def test_the_time_cap_ends_a_long_investigation(self):
        clock = FakeClock(1000.0)
        investigator = self.build(clock=clock, max_duration_s=30.0)
        self.place("box", pan=0.0, distance=200)
        real = self.observer.wait_for

        def slow(image, timeout_s=2.0):
            observation = real(image, timeout_s)
            clock.advance(100)  # a very slow vision request
            return observation

        self.observer.wait_for = slow
        result = self.run_one(self.box_target())
        self.assertEqual(result.status, "timeout")
        self.assertEqual(len(investigator.questions), 0)
        self.assertEqual(self.positive_drives(), [])

    def test_only_objects_the_score_is_about_become_targets(self):
        self.assertEqual(focus_object(["person nearby"], ["sofa"]), None)
        self.assertEqual(
            focus_object(["moved: coffee machine from kitchen to office", "new object: lamp"],
                         ["coffee machine", "lamp"]),
            "lamp",
        )
        self.assertEqual(focus_object(["moved: coffee machine from kitchen to office"], ["coffee", "coffee machine"]),
                         "coffee machine")


class PendingQuestionTests(unittest.TestCase):
    def test_it_is_bounded_and_the_oldest_goes_first(self):
        questions = PendingQuestions()
        for index in range(MAX_QUESTIONS + 2):
            self.assertIsNotNone(questions.add(f"thing {index}", "kitchen"))
        names = [q.object for q in questions.pending()]
        self.assertEqual(len(names), MAX_QUESTIONS)
        self.assertEqual(names[0], "thing 2")
        self.assertEqual(names[-1], f"thing {MAX_QUESTIONS + 1}")

    def test_one_question_per_object(self):
        questions = PendingQuestions()
        first = questions.add("Backpack", "living_room")
        self.assertIsNone(questions.add("  backpack ", "office"))
        self.assertEqual(questions.pending(), [first])
        self.assertTrue(questions.remove(first.id))
        self.assertFalse(questions.remove(first.id))
        self.assertIsNotNone(questions.add("backpack", "office"))

    def test_questions_expire(self):
        clock = FakeClock(0.0)
        questions = PendingQuestions(ttl_s=3600, clock=clock)
        question = questions.add("lamp", "office")
        clock.advance(3599)
        self.assertEqual(len(questions), 1)
        clock.advance(2)
        self.assertEqual(questions.pending(), [])
        self.assertIsNone(questions.get(question.id))
        self.assertIsNotNone(questions.add("lamp", "office"))  # an expired one no longer blocks

    def test_the_text_is_italian_with_the_room_name(self):
        expected = {
            "living_room": "in soggiorno",
            "kitchen": "in cucina",
            "office": "in studio",
            "bedroom": "in camera da letto",
            "garage": "in casa",
            None: "in casa",
        }
        for room, phrase in expected.items():
            self.assertEqual(
                question_text("zaino", room), f"Ho visto una cosa nuova {phrase}: zaino. Che cos'è?"
            )

    def test_ids_are_unique_and_looked_up(self):
        questions = PendingQuestions()
        a, b = questions.add("a", None), questions.add("b", None)
        self.assertNotEqual(a.id, b.id)
        self.assertIs(questions.get(b.id), b)


class AgentIntegrationTests(InvestigateCase):
    def setUp(self):
        super().setUp()
        self.investigator = self.build()
        self.registry = ToolRegistry(
            self.robot,
            self.safety,
            world=self.world,
            memory=self.memory,
            camera=self.camera,
            observer=self.observer,
            investigator=self.investigator,
        )

    def agent(self, *script):
        self.client = ScriptedClient(*script)
        return Agent(self.client, self.registry, "Sei Laika.", world=self.world,
                     questions=self.investigator.questions)

    def test_the_agent_sees_pending_questions(self):
        self.investigator.questions.add("backpack", "living_room")
        agent = self.agent(text("Ciao"))
        agent.respond("ciao")
        prompt = self.client.requests[0]["messages"][0]["content"]
        self.assertIn("Domande in sospeso", prompt)
        self.assertIn("al massimo una", prompt)
        self.assertIn("q-1: Ho visto una cosa nuova in soggiorno: backpack. Che cos'è?", prompt)

    def test_without_questions_the_prompt_is_unchanged(self):
        agent = self.agent(text("Ciao"))
        agent.respond("ciao")
        self.assertNotIn("Domande", self.client.requests[0]["messages"][0]["content"])

    def test_answer_question_stores_the_answer_and_removes_the_question(self):
        question = self.investigator.questions.add("stabilizer", "office")
        agent = self.agent(
            calls(("answer_question", {"question_id": question.id, "answer": "uno stabilizzatore"})),
            text("Utile."),
        )
        reply = agent.respond("È uno stabilizzatore per la fotocamera")
        self.assertEqual(reply.tool_calls_made, ["answer_question"])
        self.assertEqual(self.world.where_is("stabilizer").note, "uno stabilizzatore")
        self.assertEqual(self.world.where_is("stabilizer").room, "office")
        self.assertEqual(
            [f.text for f in self.memory.facts.search("stabilizer")], ["stabilizer: uno stabilizzatore"]
        )
        (episode,) = self.memory.episodes.about("stabilizer")
        self.assertEqual((episode.kind, episode.location), ("answer", "office"))
        self.assertEqual(self.investigator.questions.pending(), [])

    def test_answering_an_unknown_question_is_an_error(self):
        result = self.registry.run(call("answer_question", question_id="q-9", answer="boh"))
        self.assertFalse(result["ok"])

    def test_the_prompt_mentions_the_tools_only_when_offered(self):
        gestures = {}
        self.assertIn("answer_question", build_tool_system_prompt(gestures, 3, investigating=True))
        self.assertNotIn("answer_question", build_tool_system_prompt(gestures, 3))

    def test_the_tools_need_an_investigator(self):
        plain = ToolRegistry(self.robot, self.safety)
        self.assertNotIn("investigate", plain.names())
        self.assertNotIn("answer_question", plain.names())
        self.assertIn("investigate", self.registry.names())
        self.assertIn("answer_question", self.registry.names())


class InvestigateToolTests(InvestigateCase):
    def setUp(self):
        super().setUp()
        self.investigator = self.build(allow_autonomous=False)
        self.registry = ToolRegistry(
            self.robot,
            self.safety,
            world=self.world,
            memory=self.memory,
            camera=self.camera,
            observer=self.observer,
            investigator=self.investigator,
        )
        self.lease = self.arbiter.request("conversation", P.CONVERSATION)

    def run_tool(self, **arguments):
        return self.registry.run(call("investigate", **arguments), Turn(lease=self.lease))

    def test_with_a_name_it_looks_where_the_object_was_last_seen(self):
        self.see(pan=0.0)  # the sofa, television and backpack
        self.robot.look(-30, 0)
        self.robot.poses.clear()
        result = self.run_tool(object="Backpack")
        self.assertTrue(result["ok"])
        self.assertTrue(result["found"])
        self.assertEqual(result["object"], "backpack")
        self.assertEqual(result["room"], "living_room")
        self.assertIn((0.0, 5.0), self.robot.poses)
        self.assertEqual(len(self.investigate_captures()), 1 + result["moved_closer"])
        self.assertTrue(result["remembered"])
        self.assertFalse(result["question_queued"])

    def test_an_object_it_has_never_seen_is_reported_not_found(self):
        result = self.run_tool(object="unicorn")
        self.assertTrue(result["ok"])
        self.assertFalse(result["found"])
        self.assertFalse(result["question_queued"])

    def test_without_a_name_it_investigates_the_latest_interesting_sighting(self):
        self.see()  # everything is new: interesting
        result = self.run_tool()
        self.assertTrue(result["ok"])
        self.assertTrue(result["found"])
        self.assertIn(result["object"], ("sofa", "television", "backpack"))
        self.assertEqual(len(self.investigate_captures()), 1 + result["moved_closer"])

    def test_with_nothing_seen_there_is_nothing_to_investigate(self):
        result = self.run_tool()
        self.assertFalse(result["ok"])
        self.assertIn("nothing to investigate", result["error"])

    def test_with_autonomous_driving_off_a_requested_investigation_never_drives(self):
        # How far to move is Laika's own decision, so the switch applies even when asked.
        self.place("box", pan=0.0, distance=200)
        self.see()
        result = self.run_tool(object="box")
        self.assertEqual(result["moved_closer"], 0)
        self.assertFalse(self.positive_drives())
        self.assertIs(self.arbiter.holder, self.lease)  # the conversation never lost the body

    def test_with_autonomous_driving_on_it_moves_closer_under_the_conversation_lease(self):
        self.safety.allow_autonomous = True
        self.place("box", pan=0.0, distance=200)
        self.see()
        result = self.run_tool(object="box")
        self.assertEqual(result["moved_closer"], 1)
        self.assertTrue(self.positive_drives())
        self.assertIs(self.arbiter.holder, self.lease)

    def test_a_pre_empted_turn_is_stale_and_the_tool_does_not_run(self):
        self.arbiter.request("user", P.USER_COMMAND)  # takes the body from the conversation
        result = self.run_tool(object="backpack")
        self.assertFalse(result["ok"])
        self.assertIn("stale", result["error"])
        self.assertEqual(self.investigate_captures(), [])

    def test_a_tool_investigation_does_not_count_against_the_hourly_cap(self):
        self.investigator.config = InvestigateConfig(max_per_hour=1, object_cooldown_s=0)
        for _ in range(3):
            self.assertTrue(self.run_tool(object="sofa")["ok"])
        self.assertEqual(len(self.investigator._starts), 0)


if __name__ == "__main__":
    unittest.main()
