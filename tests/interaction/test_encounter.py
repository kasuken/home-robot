"""Meeting people while she is out and about (#49).

Real house simulator, safety controller, arbiter, explorer, presence tracker, interaction policy, agent and
conversation; only the model (scripted) and the ear (scripted) are fake.
"""

import json
import math
import tempfile
import time
import unittest
from pathlib import Path

from laika import trial_report
from laika.behaviour import Priority
from laika.ai_client import AIUnavailable
from laika.autonomy import DriveBudget
from laika.commands import ResultStatus
from laika.config import (
    EncounterConfig,
    ExploreConfig,
    InteractionConfig,
    load_config,
    validate_encounter,
)
from laika.encounter import Encounters, greeting_line
from laika.events import EventType
from laika.explore import Explorer
from laika.interaction_policy import Availability, Context, InteractionPolicy
from laika.personality import Personality
from laika.prompts import proactive_instruction
from laika.proactive import ProactiveRequest
from laika.scan import Scanner
from laika.store import DB_NAME, Store
from laika.tools import PROACTIVE_TOOLS
from tests.agent.test_agent import text
from tests.behaviour.test_investigate import wait_for
from tests.interaction.test_proactive import (
    ANSWER,
    LOOP_WAIT_S,
    OFFICE,
    QUESTION,
    ProactiveCase,
    at,
)

HELLO = "Oh, ciao. Stavo facendo un giro."
HEAD = (12.0, 3.0)  # where the person is seen: not a pose the exploration scan uses
NOD_POSE = (0, -25)
SETTLED = InteractionConfig(settle_s=2.0, recheck_s=60.0)  # a person must be near for 3 s with the default personality


class SpyCapture:
    def set_exploring(self, exploring):
        pass


class EncounterCase(ProactiveCase):
    """She is exploring (head only unless a test says otherwise) and somebody walks in."""

    def prepare(self, *script, interaction=SETTLED, encounter=None, battery_low=None, **kwargs):
        # The office is not "familiar" unless a test needs a new object to be interesting: exploring it
        # must not make her notice things by itself.
        kwargs.setdefault("familiar", False)
        self.assemble(*script, interaction=interaction, **kwargs)
        self.run_loop()
        self.person_cm = None
        self.encounters = Encounters(
            encounter or EncounterConfig(),
            self.interaction_config,
            self.policy,
            self.presence,
            self.arbiter,
            self.safety,
            self.conversation,
            self.investigator.questions,
            self.bus,
            self.robot,
            actions=self.actions,
            memory=self.memory,
            battery_low=battery_low,
            clock=self.clock,
        )
        self.encounters.start()
        self.addCleanup(self.encounters.close)
        self.proactive.encounters = self.encounters
        return self.encounters

    def explore(self, drive=False):
        self.explorer = Explorer(
            self.arbiter,
            self.safety,
            Scanner(self.robot, self.camera, None, settle_s=0.01, observation_wait_s=0),
            SpyCapture(),
            config=ExploreConfig(speed=15, step_ms=100, scan_every_steps=3, max_duration_s=120.0, max_steps=100000),
            drive=drive,
            head_only_interval_s=0.05,
            body_wait_s=5.0,
        )
        self.addCleanup(self.explorer.stop)
        self.assertTrue(self.explorer.start())
        self.assertTrue(wait_for(self.exploring), self.stdout.getvalue()[-2000:])

    def exploring(self):
        holder = self.arbiter.holder
        return holder is not None and holder.name == "exploration"

    def encountering(self):
        holder = self.arbiter.holder
        return holder is not None and holder.name == "encounter"

    def someone_walks_in(self, **kwargs):
        kwargs.setdefault("pan", HEAD[0])
        kwargs.setdefault("tilt", HEAD[1])
        self.a_person_appears(**kwargs)

    def stays(self, seconds=5, **kwargs):
        """Time passes and they are still there."""
        kwargs.setdefault("pan", HEAD[0])
        kwargs.setdefault("tilt", HEAD[1])
        self.advance(seconds)
        self.a_person_appears(**kwargs)

    def encounter_over(self):
        return wait_for(lambda: not self.encounters.handling, LOOP_WAIT_S)

    def episodes(self):
        return [e for e in self.memory.episodes.recent(50) if e.kind == "encounter"]

    def person_at(self, cm):
        """A person `cm` ahead: the ultrasonic reading (the safety controller reads it too) shrinks as she moves."""
        self.person_cm = cm
        start = self.robot.pose()
        self.robot.distance_cm = lambda: self.person_cm - math.hypot(
            self.robot.pose().x - start.x, self.robot.pose().y - start.y
        )

    def person_leaves(self):
        """Nobody seen for ten minutes. Done while the presence lock is held, so nothing sees the later
        clock with the person still there."""
        with self.presence._cond:
            self.advance(700)
            self.presence.check()

    def moves(self):
        """How many times the wheels were told to go forward."""
        return len(self.positive_drives())

    def instructions(self):
        return [r["messages"][-1]["content"] for r in self.client.requests]

    def assertResumed(self):
        """The interrupted exploration carries on, and ends cleanly when asked."""
        self.assertTrue(wait_for(self.exploring, LOOP_WAIT_S), self.stdout.getvalue()[-2000:])
        self.assertTrue(self.explorer.active)
        self.explorer.stop()
        self.assertFalse(self.explorer.active)
        self.assertEqual(self.explorer.end_reason, "stopped")
        self.assertIsNone(self.arbiter.holder)


class ConversationTests(EncounterCase):
    """Done when: a person appearing during exploration leads to a conversation when the policy allows."""

    def test_a_person_during_exploration_gets_a_hello_and_the_exploration_resumes(self):
        self.prepare(text(HELLO))
        self.explore(drive=True)
        self.someone_walks_in()
        # She stops: exploration waits (resumable), the wheels are still, and her head is on the person.
        self.assertTrue(wait_for(self.encountering))
        (waiting,) = self.arbiter.waiting
        self.assertEqual(waiting.name, "exploration")
        self.assertTrue(wait_for(lambda: self.robot.speed == 0))
        self.assertTrue(wait_for(lambda: HEAD in self.robot.poses))
        self.assertEqual(self.spoken(), [])  # not settled yet: she does not speak at once
        self.stays()
        self.assertTrue(wait_for(lambda: self.spoken() == [HELLO], LOOP_WAIT_S), self.stdout.getvalue()[-2000:])
        self.assertTrue(self.encounter_over())
        # The opening was asked of the model, in her personality, with the head and gestures only.
        (request,) = self.client.requests
        self.assertIn("Hai incontrato qualcuno", request["messages"][-1]["content"])
        self.assertEqual({spec.name for spec in request["tools"]}, PROACTIVE_TOOLS & set(self.registry.names()))
        # InteractionStarted and Completed, from Laika, marked as an encounter, with no question in them.
        events = self.interactions()
        self.assertEqual([t for t, _ in events], [EventType.INTERACTION_STARTED, EventType.INTERACTION_COMPLETED])
        self.assertEqual(
            events[0][1], {"initiator": "laika", "reason": "encounter_greeting", "encounter": True}
        )
        self.assertEqual(events[1][1], {"initiator": "laika", "encounter": True, "answered": False})
        # The attempt counts for the cooldown, like any other proactive one.
        self.assertEqual(self.policy.decide().reason, "cooldown")
        (episode,) = self.episodes()
        self.assertEqual(episode.summary, "Met someone while exploring: talked")
        self.assertEqual(episode.subjects, ("encounter", "talked"))
        self.assertResumed()

    def test_the_person_answers_and_the_chat_goes_on(self):
        self.prepare(text(HELLO), text("Bene, grazie."))
        self.explore()
        self.someone_walks_in()
        self.assertTrue(wait_for(self.encountering))
        self.ear.reply("Ciao Laika, tutto bene?")
        self.stays()
        self.assertTrue(wait_for(lambda: len(self.spoken()) >= 2, LOOP_WAIT_S), self.spoken())
        self.assertEqual(self.spoken(), [HELLO, "Bene, grazie."])
        self.assertTrue(self.encounter_over())
        self.assertTrue(wait_for(lambda: not self.conversation.awake, LOOP_WAIT_S))
        self.assertResumed()

    def test_the_body_is_never_given_back_in_between(self):
        self.prepare(text(HELLO))
        self.explore()
        holders = []
        self.someone_walks_in()
        self.assertTrue(wait_for(self.encountering))
        self.stays()
        deadline = time.monotonic() + LOOP_WAIT_S
        while time.monotonic() < deadline and not self.spoken():
            holders.append(self.arbiter.holder.name if self.arbiter.holder else None)
            time.sleep(0.005)
        self.assertTrue(self.encounter_over())
        # From the pre-emption to the end of the hello the holder was the encounter, then the same lease
        # was kept by the conversation; exploration only came back afterwards.
        self.assertNotIn("exploration", holders)
        self.assertNotIn(None, holders)

    def test_a_pending_question_comes_first(self):
        self.prepare(text("Che cos'è quella cosa vicino alla scrivania?"), text("Grazie!"), familiar=True)
        self.notice_the_stabilizer()
        self.explore()
        self.someone_walks_in()
        self.assertTrue(wait_for(self.encountering))
        self.ear.reply(ANSWER)
        self.stays()
        self.assertTrue(wait_for(lambda: self.client.requests, LOOP_WAIT_S))
        self.assertIn(QUESTION, self.instructions()[0])
        self.assertTrue(self.encounter_over())
        (question,) = [e for e in self.interactions() if e[0] is EventType.INTERACTION_STARTED]
        self.assertEqual(question[1]["reason"], "pending_question")
        self.assertTrue(question[1]["encounter"])
        self.assertTrue(question[1]["questionId"])

    def test_something_she_noticed_is_the_next_thing_to_say(self):
        self.prepare(text("Ho visto un aquilone."))
        self.explore()
        self.bus.emit(EventType.INTERESTING_CHANGE_DETECTED, total=0.9, reasons=["appeared: kite"], location="office")
        self.bus.drain()
        self.someone_walks_in()
        self.assertTrue(wait_for(self.encountering))
        self.stays()
        self.assertTrue(wait_for(lambda: self.client.requests, LOOP_WAIT_S))
        self.assertIn("appeared: kite", self.instructions()[0])
        self.assertIn("una cosa che hai notato", self.instructions()[0])
        self.assertTrue(self.encounter_over())
        started = [d for t, d in self.interactions() if t is EventType.INTERACTION_STARTED]
        self.assertEqual(started[0]["reason"], "encounter_noticed")

    def test_a_low_battery_is_something_she_noticed(self):
        self.prepare(text("Sono stanca."), battery_low=lambda: True)
        self.explore()
        self.someone_walks_in()
        self.assertTrue(wait_for(self.encountering))
        self.stays()
        self.assertTrue(wait_for(lambda: self.client.requests, LOOP_WAIT_S))
        self.assertIn("batteria", self.instructions()[0])
        self.assertTrue(self.encounter_over())

    def test_without_a_model_the_hello_is_a_template(self):
        self.prepare()  # no scripted turn: the agent fails and the plain line is spoken
        self.explore()
        self.client.script.append(AIUnavailable("down"))
        self.someone_walks_in()
        self.assertTrue(wait_for(self.encountering))
        self.stays()
        self.assertTrue(wait_for(lambda: self.spoken() == [HELLO], LOOP_WAIT_S), self.stdout.getvalue()[-2000:])
        self.assertTrue(self.encounter_over())

    def test_a_curiosity_activity_is_never_preempted_and_is_named(self):
        self.prepare(text("Oh, ciao."))
        # A short curiosity activity (investigation) is not pre-empted: the encounter waits, then goes on.
        lease = self.arbiter.request("investigation", Priority.CURIOSITY, resumable=False)
        self.someone_walks_in()
        time.sleep(0.3)
        lease.kick()
        self.assertIs(self.arbiter.holder, lease)
        self.stays()
        time.sleep(0.2)
        self.assertIs(self.arbiter.holder, lease)  # still the investigation's: never pre-empted
        self.assertEqual(self.spoken(), [])
        lease.release()
        self.assertTrue(wait_for(lambda: self.spoken() == ["Oh, ciao."], LOOP_WAIT_S), self.stdout.getvalue()[-2000:])
        self.assertTrue(self.encounter_over())
        (episode,) = self.episodes()
        self.assertEqual(episode.summary, "Met someone while investigating something: talked")


class QuietTests(EncounterCase):
    """Done when: and to silence (at most a gesture) when it does not."""

    def gestures(self):
        return [pose for pose in self.robot.poses if pose == NOD_POSE]

    def assertSilentButLooked(self, reason):
        self.assertTrue(self.encounter_over())
        self.assertEqual(self.spoken(), [])
        self.assertEqual(self.client.requests, [])
        self.assertEqual(self.interactions(), [])
        (episode,) = self.episodes()
        self.assertEqual(episode.subjects, ("encounter", "quiet", reason))
        self.assertEqual(episode.summary, f"Met someone while exploring: stayed quiet ({reason.replace('_', ' ')})")
        self.assertIn(HEAD, self.robot.poses)  # she did turn to them
        self.assertResumed()

    def test_the_cooldown_means_a_nod_and_no_words(self):
        self.prepare(text(HELLO))
        self.bus.emit(EventType.INTERACTION_STARTED, initiator="user")
        self.bus.emit(EventType.INTERACTION_COMPLETED, initiator="user")
        self.bus.drain()
        self.advance(60)
        self.explore()
        self.someone_walks_in()
        self.assertTrue(wait_for(self.encountering))
        self.stays()
        self.assertTrue(wait_for(lambda: self.gestures(), LOOP_WAIT_S))
        self.assertSilentButLooked("cooldown")

    def test_a_person_who_has_not_settled_is_not_ambushed(self):
        self.prepare(text(HELLO), interaction=InteractionConfig(settle_s=60.0, recheck_s=60.0))
        self.explore()
        self.someone_walks_in()  # settle time 90 s: she asks at once, and the policy says "unknown"
        self.assertTrue(wait_for(lambda: self.episodes(), LOOP_WAIT_S))
        self.assertSilentButLooked("unknown")
        self.assertTrue(self.gestures())

    def test_the_hourly_limit(self):
        self.prepare(text(HELLO), interaction=InteractionConfig(settle_s=2.0, recheck_s=60.0, cooldown_s=0.0, max_per_hour=1))
        self.policy.record_attempt()
        self.explore()
        self.someone_walks_in()
        self.assertTrue(wait_for(self.encountering))
        self.stays()
        self.assertTrue(wait_for(lambda: self.episodes(), LOOP_WAIT_S))
        self.assertSilentButLooked("rate_limit")

    def test_greeting_can_be_turned_off(self):
        self.prepare(text(HELLO), encounter=EncounterConfig(greet_when_nothing_to_say=False))
        self.explore()
        self.someone_walks_in()
        self.assertTrue(wait_for(self.encountering))
        self.stays()
        self.assertTrue(wait_for(lambda: self.episodes(), LOOP_WAIT_S))
        self.assertSilentButLooked("nothing_to_say")

    def test_the_gesture_can_be_turned_off(self):
        self.prepare(text(HELLO), encounter=EncounterConfig(greet_when_nothing_to_say=False, gesture_when_quiet=False))
        self.explore()
        self.someone_walks_in()
        self.assertTrue(wait_for(self.encountering))
        self.stays()
        self.assertTrue(wait_for(lambda: self.episodes(), LOOP_WAIT_S))
        self.assertSilentButLooked("nothing_to_say")
        self.assertEqual(self.gestures(), [])

    def test_do_not_disturb_means_nothing_at_all(self):
        self.prepare(text(HELLO))
        self.policy.set_do_not_disturb(30)
        self.explore()
        lease = self.arbiter.holder
        self.someone_walks_in()
        self.stays()
        time.sleep(0.5)
        # She does not even stop: it is not an encounter.
        self.assertIs(self.arbiter.holder, lease)
        self.assertTrue(lease.active)
        self.assertEqual(self.encounters.count, 0)
        self.assertNotIn(HEAD, self.robot.poses)
        self.assertEqual(self.spoken(), [])
        self.assertEqual(self.episodes(), [])
        self.assertResumed()

    def test_do_not_disturb_asked_during_the_encounter_cancels_the_nod_too(self):
        self.prepare(text(HELLO))
        self.explore()
        self.someone_walks_in()
        self.assertTrue(wait_for(self.encountering))
        self.policy.set_do_not_disturb(30)
        self.stays()
        self.assertTrue(wait_for(lambda: self.episodes(), LOOP_WAIT_S))
        self.assertTrue(self.encounter_over())
        self.assertEqual(self.gestures(), [])
        self.assertEqual(self.spoken(), [])
        (episode,) = self.episodes()
        self.assertEqual(episode.subjects, ("encounter", "quiet", "do_not_disturb"))
        self.assertResumed()

    def test_quiet_hours_mean_nothing_at_all(self):
        self.prepare(text(HELLO))
        self.wall.now = at(22)
        self.explore()
        lease = self.arbiter.holder
        self.someone_walks_in()
        self.stays()
        time.sleep(0.5)
        self.assertIs(self.arbiter.holder, lease)
        self.assertEqual(self.encounters.count, 0)
        self.assertEqual(self.gestures(), [])
        self.assertEqual(self.spoken(), [])
        self.assertResumed()

    def test_busy_means_nothing_at_all(self):
        self.prepare(text(HELLO))
        self.policy.set_busy(30)
        self.explore()
        lease = self.arbiter.holder
        self.someone_walks_in()
        time.sleep(0.4)
        self.assertIs(self.arbiter.holder, lease)
        self.assertEqual(self.encounters.count, 0)

    def test_the_pause_ends_the_encounter_quietly(self):
        self.prepare(text(HELLO))
        self.explore()
        self.someone_walks_in()
        self.assertTrue(wait_for(self.encountering))
        self.safety.pause_autonomy()  # "Laika, resta ferma"
        self.addCleanup(self.safety.resume_autonomy)
        self.stays()
        self.assertTrue(self.encounter_over())
        self.assertEqual(self.spoken(), [])
        self.assertEqual(self.client.requests, [])
        (episode,) = self.episodes()
        self.assertEqual(episode.subjects, ("encounter", "quiet", "paused"))
        self.assertEqual(self.gestures(), [])

    def test_a_person_who_arrives_while_she_is_paused_is_no_encounter(self):
        self.prepare(text(HELLO))
        self.explore()
        self.safety.pause_autonomy()
        self.addCleanup(self.safety.resume_autonomy)
        self.assertTrue(wait_for(lambda: not self.explorer.active))
        self.someone_walks_in()
        self.stays()
        time.sleep(0.3)
        self.assertEqual(self.encounters.count, 0)
        self.assertEqual(self.spoken(), [])

    def test_the_switch_turns_encounters_off(self):
        self.prepare(text(HELLO), encounter=EncounterConfig(enabled=False))
        self.explore()
        lease = self.arbiter.holder
        self.someone_walks_in()
        time.sleep(0.4)
        self.assertIs(self.arbiter.holder, lease)
        self.assertEqual(self.encounters.count, 0)

    def test_a_person_who_leaves_before_the_decision(self):
        self.prepare(text(HELLO))
        self.explore()
        self.someone_walks_in()
        self.assertTrue(wait_for(self.encountering))
        self.person_leaves()
        self.assertTrue(self.encounter_over())
        self.assertEqual(self.spoken(), [])
        (episode,) = self.episodes()
        self.assertEqual(episode.subjects, ("encounter", "quiet", "left"))
        self.assertResumed()


class OnlyEncountersGreetTests(EncounterCase):
    def test_a_person_who_arrives_while_she_is_idle_gets_no_hello(self):
        self.prepare(text(HELLO), familiar=False)
        self.advance(2000)
        self.someone_walks_in()
        self.stays(10)
        self.assertFalse(self.proactive.check("recheck"))
        self.assertEqual(self.policy.decide().reason, "nothing_to_say")
        self.assertEqual(self.encounters.count, 0)
        self.assertQuiet()
        self.assertEqual(self.client.requests, [])
        # The same moment, as an encounter, would have been allowed.
        self.assertTrue(self.policy.decide(Context(encounter=True)).allowed)

    def test_a_person_during_a_user_conversation_or_command_is_no_encounter(self):
        self.prepare(text(HELLO))
        self.explore()
        chat = self.arbiter.request("conversation", Priority.CONVERSATION)
        self.addCleanup(chat.release)
        self.assertTrue(wait_for(lambda: self.arbiter.holder is chat))
        self.someone_walks_in()
        self.stays()
        time.sleep(0.4)
        self.assertIs(self.arbiter.holder, chat)
        self.assertTrue(chat.active)
        self.assertEqual(self.encounters.count, 0)
        chat.release()
        command = self.arbiter.request("user-command", Priority.USER_COMMAND)
        self.addCleanup(command.release)
        self.assertIs(self.arbiter.holder, command)
        self.assertFalse(self.encounters.on_person_detected({}))
        self.assertIs(self.arbiter.holder, command)
        self.assertEqual(self.spoken(), [])

    def test_a_user_who_wakes_her_during_the_encounter_wins(self):
        self.prepare(text("Sono le tre."))
        self.explore()
        self.someone_walks_in()
        self.assertTrue(wait_for(self.encountering))
        self.ear.wake("ehi Laika che ore sono")
        self.assertTrue(wait_for(lambda: self.spoken() == ["Sono le tre."], LOOP_WAIT_S), self.stdout.getvalue()[-2000:])
        self.assertTrue(self.encounter_over())
        self.assertNotIn(
            "encounter_greeting", [d.get("reason") for t, d in self.interactions() if t is EventType.INTERACTION_STARTED]
        )
        (episode,) = self.episodes()
        self.assertEqual(episode.subjects[:2], ("encounter", "quiet"))
        self.assertTrue(wait_for(lambda: not self.conversation.awake, LOOP_WAIT_S))
        self.assertTrue(wait_for(self.exploring, LOOP_WAIT_S))


class ApproachTests(EncounterCase):
    """Optionally a little closer before talking: only with autonomous driving, never closer than stop_cm."""

    CLOSE = EncounterConfig(approach_distance_cm=125.0, stop_cm=110.0)

    def hello_after_settling(self, **kwargs):
        self.someone_walks_in(**kwargs)
        self.assertTrue(wait_for(self.encountering))
        self.stays(**kwargs)
        self.assertTrue(wait_for(lambda: self.spoken() == [HELLO], LOOP_WAIT_S), self.stdout.getvalue()[-2000:])
        self.assertTrue(self.encounter_over())

    def test_a_far_person_ahead_gets_one_short_move_and_then_the_hello(self):
        self.prepare(text(HELLO), encounter=self.CLOSE)
        self.explore()
        self.person_at(200)
        self.hello_after_settling(pan=5)
        self.assertEqual(len(self.positive_drives()), 1)
        self.assertEqual(self.moves(), 1)
        self.assertEqual(self.robot.speed, 0)
        self.assertEqual(self.spoken(), [HELLO])
        # She came closer, and stayed outside the distance she keeps.
        self.assertLess(self.robot.distance_cm(), 200)
        self.assertGreaterEqual(self.robot.distance_cm(), self.CLOSE.stop_cm)

    def test_a_person_who_is_close_enough_is_not_approached(self):
        self.prepare(text(HELLO), encounter=self.CLOSE)
        self.explore()
        self.person_at(124)
        self.hello_after_settling(pan=5)
        self.assertEqual(self.positive_drives(), [])

    def test_a_person_just_far_enough_is_approached_once_and_never_too_close(self):
        self.prepare(text(HELLO), encounter=self.CLOSE)
        self.explore()
        self.person_at(125)  # exactly approach_distance_cm: one move (about 10 cm) leaves 115, above stop_cm
        self.hello_after_settling(pan=5)
        self.assertEqual(self.moves(), 1)
        self.assertGreaterEqual(self.robot.distance_cm(), self.CLOSE.stop_cm)

    def test_a_person_to_the_side_is_not_approached(self):
        self.prepare(text(HELLO), encounter=self.CLOSE)
        self.explore()
        self.person_at(250)
        self.hello_after_settling(pan=35)
        self.assertEqual(self.positive_drives(), [])

    def test_without_autonomous_driving_no_drive_reaches_the_robot(self):
        self.prepare(text(HELLO), encounter=self.CLOSE)
        self.safety.allow_autonomous = False  # the robot's default
        self.explore()
        self.person_at(250)
        self.hello_after_settling(pan=5)
        self.assertEqual(self.positive_drives(), [])
        self.assertEqual(self.robot.speed, 0)

    def test_the_approach_can_be_turned_off(self):
        self.prepare(text(HELLO), encounter=EncounterConfig(approach=False))
        self.explore()
        self.person_at(250)
        self.hello_after_settling(pan=5)
        self.assertEqual(self.positive_drives(), [])

    def test_the_safety_layer_can_still_refuse_the_move(self):
        # The move is her own driving, so the pause, budget, battery and no-go rooms apply: with the
        # budget used up nothing moves, and she talks from where she is.
        self.prepare(text(HELLO), encounter=self.CLOSE)
        self.explore()
        self.person_at(250)
        self.safety.drive_budget = DriveBudget(per_hour_s=1.0)
        self.safety.drive_budget.add(5.0)  # the hour's driving is used up
        self.hello_after_settling(pan=5)
        self.assertEqual(self.positive_drives(), [])

    def test_a_person_who_walks_away_is_not_followed(self):
        self.prepare(text(HELLO), encounter=self.CLOSE)
        self.explore()
        self.person_at(250)
        original = type(self.robot).drive
        left = []

        def drive(speed):
            original(self.robot, speed)
            if speed > 0 and not left:
                left.append(True)
                self.person_leaves()  # gone, while the move is under way

        self.robot.drive = drive
        results = []
        execute = self.safety.execute

        def record(*args, **kwargs):
            result = execute(*args, **kwargs)
            results.append(result)
            return result

        self.safety.execute = record
        self.someone_walks_in(pan=5)
        self.assertTrue(wait_for(self.encountering))
        self.stays(pan=5)
        self.assertTrue(self.encounter_over())
        self.assertTrue(left)
        # The move was stopped as soon as they were gone, nothing was said, and there was no second move.
        (result,) = results
        self.assertIs(result.result, ResultStatus.CANCELLED)
        self.assertEqual(result.reason, "stopped")
        self.assertEqual(self.robot.speed, 0)
        self.assertEqual(len(self.positive_drives()), 1)
        self.assertEqual(self.moves(), 1)
        self.assertEqual(self.spoken(), [])
        (episode,) = self.episodes()
        self.assertEqual(episode.subjects, ("encounter", "quiet", "left"))
        self.assertResumed()
        self.assertEqual(self.moves(), 1)  # exploring head only: no search for them, no second move


class PolicyTests(unittest.TestCase):
    """The decision table with `encounter` (the rest lives in test_interaction_policy)."""

    def setUp(self):
        from tests.fakes import FakeClock
        from tests.interaction.test_interaction_policy import FakePresence

        self.clock = FakeClock(1000.0)
        self.wall = FakeClock(at(14))
        self.presence = FakePresence(self.clock)

    def policy(self, **overrides):
        return InteractionPolicy(
            InteractionConfig(**overrides),
            self.presence,
            clock=self.clock,
            wall_clock=self.wall,
            personality=Personality(initiative=50, interruption_tolerance=50),
        )

    def settled(self, **overrides):
        policy = self.policy(**overrides)
        self.presence.arrive()
        self.clock.advance(10)
        self.wall.advance(10)
        return policy

    def test_a_hello_is_something_to_say_only_for_an_encounter(self):
        policy = self.settled()
        self.assertEqual(policy.decide().reason, "nothing_to_say")
        self.assertEqual(policy.decide(Context(encounter=False)).reason, "nothing_to_say")
        self.assertTrue(policy.decide(Context(encounter=True)).allowed)

    def test_every_other_rule_still_applies(self):
        encounter = Context(encounter=True)
        policy = self.settled()
        policy.set_do_not_disturb(30)
        self.assertEqual(policy.decide(encounter).reason, "do_not_disturb")
        policy = self.settled()
        self.wall.now = at(22)
        self.assertEqual(policy.decide(encounter).reason, "sleeping")
        self.wall.now = at(14)
        self.assertEqual(self.settled(enabled=False).decide(encounter).reason, "disabled")
        self.assertEqual(
            self.settled().decide(Context(encounter=True, conversation_active=True)).reason, "busy"
        )
        policy = self.settled()
        policy.record_attempt()
        self.assertEqual(policy.decide(encounter).reason, "cooldown")
        policy = self.settled(cooldown_s=0, max_per_hour=1)
        policy.record_attempt()
        self.assertEqual(policy.decide(encounter).reason, "rate_limit")

    def test_the_settle_time_still_applies(self):
        policy = self.policy()
        self.presence.arrive()
        decision = policy.decide(Context(encounter=True))
        self.assertEqual((decision.allowed, decision.reason, decision.state), (False, "unknown", Availability.UNKNOWN))

    def test_nobody_near_is_away_or_unknown_even_for_an_encounter(self):
        policy = self.policy()
        self.assertEqual(policy.decide(Context(encounter=True)).reason, "unknown")
        self.clock.advance(1000)
        self.assertEqual(policy.decide(Context(encounter=True)).reason, "away")


class RequestAndPromptTests(unittest.TestCase):
    def test_the_instruction_adapts_to_what_she_opens_with(self):
        question = proactive_instruction("Che cos'è?")
        self.assertIn("Domanda: Che cos'è?", question)
        noticed = proactive_instruction("appeared: kite", kind="noticed")
        self.assertIn("appeared: kite", noticed)
        self.assertIn("Hai incontrato qualcuno", noticed)
        greeting = proactive_instruction("stavo facendo un giro", kind="greeting")
        self.assertIn("Oh, ciao", greeting)
        self.assertIn("non lo identifichi", greeting)
        for instruction in (noticed, greeting):
            self.assertIn("senza altri strumenti oltre ai gesti", instruction)

    def test_a_question_request_is_what_it_was(self):
        request = ProactiveRequest("q-1", "Che cos'è?")
        self.assertEqual((request.kind, request.encounter, request.lease), ("question", False, None))

    def test_the_hello(self):
        self.assertEqual(greeting_line("stavo facendo un giro"), HELLO)


class ConfigTests(unittest.TestCase):
    def load(self, body):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "laika.toml"
            path.write_text(body, encoding="utf-8")
            return load_config(path)

    def test_defaults_and_the_example_file(self):
        defaults = EncounterConfig()
        validate_encounter(defaults)
        self.assertEqual(
            (defaults.enabled, defaults.greet_when_nothing_to_say, defaults.approach, defaults.gesture_when_quiet),
            (True, True, True, True),
        )
        self.assertEqual((defaults.approach_distance_cm, defaults.stop_cm), (120.0, 80.0))
        self.assertEqual(load_config(Path("config/laika.example.toml")).encounter, defaults)

    def test_loaded(self):
        config = self.load("[encounter]\nenabled = false\napproach_distance_cm = 150\nstop_cm = 90\n")
        self.assertFalse(config.encounter.enabled)
        self.assertEqual((config.encounter.approach_distance_cm, config.encounter.stop_cm), (150, 90))

    def test_bad_values_name_the_setting(self):
        for body, name in (
            ("enabled = 1", "enabled"),
            ("greet_when_nothing_to_say = \"yes\"", "greet_when_nothing_to_say"),
            ("approach = 2", "approach"),
            ("gesture_when_quiet = 0", "gesture_when_quiet"),
            ("stop_cm = 10", "stop_cm"),
            ("stop_cm = 1000", "stop_cm"),
            ("approach_distance_cm = 20", "approach_distance_cm"),
            ("stop_cm = 100\napproach_distance_cm = 105", "approach_distance_cm"),
            ("colour = 1", "colour"),
        ):
            with self.subTest(body=body):
                with self.assertRaises(ValueError) as caught:
                    self.load(f"[encounter]\n{body}\n")
                self.assertIn(name, str(caught.exception))


class ReportTests(unittest.TestCase):
    def test_the_trial_report_counts_encounters_and_their_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "memory").mkdir()
            now = time.time()
            store = Store(root / "memory" / DB_NAME)
            try:
                for index, subjects in enumerate(
                    (["encounter", "talked"], ["encounter", "quiet", "cooldown"], ["encounter", "quiet", "cooldown"],
                     ["encounter", "quiet", "left"])
                ):
                    store.put_episode(
                        {"id": f"ep-{index}", "timestamp": now - 60 - index, "kind": "encounter",
                         "summary": "Met someone while exploring", "subjects": subjects, "location": "office"}
                    )
                self.assertTrue(store.flush())
            finally:
                store.close()
            report = trial_report.build_report(root, now - 3600, now + 60, now=now)
            self.assertEqual(
                report["encounters"],
                {"count": 4, "talked": 1, "quiet": 3, "quiet_by_reason": {"cooldown": 2, "left": 1}},
            )
            markdown = trial_report.render_markdown(report)
            self.assertIn("## Meeting people", markdown)
            self.assertIn("Met while out and about: 4 (talked 1, stayed quiet 3: cooldown 2, left 1)", markdown)
            self.assertIn("encounters: 4 met while out and about, talked 1, quiet 3", trial_report.summary_text(report))

    def test_no_store_means_not_known(self):
        data = trial_report.StoreData()
        self.assertEqual(trial_report.build_encounters(data, 0, 1)["count"], None)

    def test_an_encounter_hello_is_not_an_orphaned_ask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            (root / "memory").mkdir()
            now = time.time()
            lines = []
            for kind, ts, data in (
                ("HardwareReady", now - 100, {}),
                ("InteractionStarted", now - 50, {"initiator": "laika", "reason": "encounter_greeting", "encounter": True}),
                ("InteractionCompleted", now - 40, {"initiator": "laika", "encounter": True, "answered": False}),
            ):
                lines.append(json.dumps({"id": f"evt-{ts}", "type": kind, "data": data, "timestamp": ts}))
            (root / "logs" / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
            report = trial_report.build_report(root, now - 3600, now + 60, now=now)
            self.assertEqual(report["conversations"]["laika_initiated"], 1)
            self.assertEqual(report["conversations"]["asks"], [])
            self.assertEqual(report["loops"]["asked_without_queue_record"], [])


if __name__ == "__main__":
    unittest.main()
