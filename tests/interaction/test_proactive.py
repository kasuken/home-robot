"""Questions Laika asks on her own (#40): the PRD's camera stabiliser, end to end.

Real house simulator, camera, observer, scorer, investigator, presence tracker, interaction policy,
memory, world model, agent and conversation; only the model (scripted) and the ear (scripted) are fake.
"""

import contextlib
import io
import queue
import threading
import time
import unittest
import unittest.mock
from datetime import datetime

from laika.actions import ActionContext, ActionPlayer, available_actions
from laika.agent import Agent
from laika.ai_client import AIReply, AIUnavailable
from laika.audio import ConsoleEar, wait_for_wake
from laika.behaviour import Priority
from laika.config import CuriosityConfig, InteractionConfig, PresenceConfig, VoiceConfig
from laika.conversation import Conversation
from laika.conversation_memory import MemorySelector
from laika.events import EventType
from laika.house import Pose, Thing
from laika.interaction_policy import InteractionPolicy
from laika.investigate import question_text
from laika.observations import Observation
from laika.presence import PresenceTracker
from laika.prompts import build_tool_system_prompt, proactive_instruction
from laika.proactive import ProactiveInteraction, ProactiveRequest
from laika.robot import ALL_CAPABILITIES
from laika.tools import MAX_GESTURES_PER_TURN, PROACTIVE_TOOLS, ToolRegistry
from laika.wake_word import WakeWordMatcher
from tests.agent.test_agent import JsonAI, ScriptedClient, calls, text
from tests.behaviour.test_investigate import InvestigateCase, wait_for
from tests.fakes import FakeClock

OFFICE = Pose(100, 500, 90)  # facing north: the desk and the monitor are in view
ANSWER = "È uno stabilizzatore per la fotocamera"
OPENING = "Emanuele, domanda veloce: che cos'è quella cosa vicino alla scrivania?"
REPLY = "Informazione utile. Pensavo fosse un oggetto sospetto di plastica."
QUESTION = question_text("camera_stabilizer", "office")
HEAD_AT_PERSON = (20.0, 5.0)
LOOP_WAIT_S = 8.0


def at(hour, minute=0):
    return datetime(2026, 10, 15, hour, minute).timestamp()


class ScriptedEar:
    """A fake Ear. `replies` answer a listen with a timeout; the wake listen (no timeout) only ends
    when `wake()` says something, `interrupt()` is called, or the test stops."""

    def __init__(self):
        self.replies = queue.Queue()
        self.wakes = queue.Queue()
        self.interrupts = 0
        self.listens = []
        self._interrupted = threading.Event()
        self.stopped = threading.Event()

    def reply(self, *utterances):
        for utterance in utterances:
            self.replies.put(utterance)

    def wake(self, utterance):
        self.wakes.put(utterance)

    def interrupt(self):
        self.interrupts += 1
        self._interrupted.set()

    def listen(self, timeout_s=None):
        self.listens.append(timeout_s)
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        source = self.wakes if timeout_s is None else self.replies
        while True:
            if self.stopped.is_set():
                raise KeyboardInterrupt
            if self._interrupted.is_set():
                self._interrupted.clear()
                return None
            try:
                return source.get(timeout=0.02)
            except queue.Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    return None

    def close(self):
        pass


class ProactiveCase(InvestigateCase):
    """The house, plus everything between a person arriving and the answer being remembered."""

    def setUp(self):
        self.clock = FakeClock(1000.0)
        self.wall = FakeClock(at(14))

    def advance(self, seconds):
        self.clock.advance(seconds)
        self.wall.advance(seconds)

    def assemble(self, *script, interaction=None, voice=None, proactive=True, familiar=True):
        self.build(start=OFFICE, curiosity=CuriosityConfig())
        if familiar:
            # She has known the office for a long time, so something new in it is interesting.
            long_ago = time.time() - 2 * CuriosityConfig().familiar_room_after_s
            self.world.observe(["desk", "monitor", "chair"], "office", timestamp=long_ago)
        self.interaction_config = interaction or InteractionConfig(recheck_s=60.0)
        self.presence = PresenceTracker(self.bus, PresenceConfig(), clock=self.clock)
        self.presence.start()
        self.addCleanup(self.presence.close)
        self.policy = InteractionPolicy(
            self.interaction_config,
            self.presence,
            events=self.bus,
            threshold=CuriosityConfig().threshold,
            clock=self.clock,
            wall_clock=self.wall,
        )
        self.policy.start()
        self.addCleanup(self.policy.close)
        capabilities = self.robot.identity().capabilities
        allowed = available_actions(capabilities, True)
        self.actions = ActionPlayer(ActionContext(self.robot, self.safety), allowed, talk_motion=False)
        self.addCleanup(self.actions.close)
        self.client = ScriptedClient(*script)
        self.registry = ToolRegistry(
            self.robot, self.safety, world=self.world, actions=self.actions, memory=self.memory,
            speak=self.robot.speak, investigator=self.investigator, interaction=self.policy,
        )
        questions = self.investigator.questions
        self.agent = Agent(
            self.client, self.registry, build_tool_system_prompt(allowed, MAX_GESTURES_PER_TURN),
            world=self.world, questions=questions, memory=self.memory,
        )
        self.ear = ScriptedEar()
        self.conversation = Conversation(
            ear=self.ear, robot=self.robot, ai=JsonAI(AIReply("non usato")), actions=self.actions,
            voice=voice or VoiceConfig(follow_up_s=0.4), events=self.bus, safety=self.safety,
            agent=self.agent, arbiter=self.arbiter,
            selector=MemorySelector(self.world, self.memory, questions, config=None),
        )
        self.conversation.questions = questions
        self.proactive = ProactiveInteraction(
            self.policy, self.presence, questions, self.conversation, self.arbiter, self.bus,
            self.interaction_config, clock=self.clock,
        )
        if proactive:
            self.proactive.start()
        self.addCleanup(self.proactive.close)
        self.runner = None
        return self.conversation

    def run_loop(self):
        """The main thread of the app: `Conversation.run`, until the test ends."""

        def target():
            with contextlib.suppress(KeyboardInterrupt):
                self.conversation.run()

        self.runner = threading.Thread(target=target, name="main-loop", daemon=True)
        self.runner.start()

        def stop():
            self.ear.stopped.set()
            self.runner.join(5)

        self.addCleanup(stop)
        self.assertTrue(wait_for(lambda: self.robot.spoken))  # the welcome: she is now asleep
        self.welcome = len(self.robot.spoken)

    def spoken(self):
        return self.robot.spoken[self.welcome:]

    def notice_the_stabilizer(self):
        """The new object appears; she looks into it and queues the question. Nothing is said."""
        self.investigator.start()
        self.see()  # the desk and the monitor: nothing new
        self.house.add_thing(Thing("camera_stabilizer", 130, 560, 10))
        self.see()
        self.assertTrue(self.investigator.wait_idle())
        self.robot.look(0, 0)
        self.robot.poses.clear()

    def a_person_appears(self, pan=HEAD_AT_PERSON[0], tilt=HEAD_AT_PERSON[1]):
        observation = Observation(
            "a person at the desk", people=1, confidence=0.9, location="office", camera_pan=pan, camera_tilt=tilt
        )
        self.bus.emit(EventType.OBSERVATION_CREATED, **observation.to_dict())
        self.bus.drain()

    def settle(self, seconds=10):
        """Time passes; the person keeps being seen, so they stay."""
        self.advance(seconds)
        self.a_person_appears()

    def stay(self, seconds, step=20):
        """The person stays in view for `seconds`, seen every `step`."""
        for _ in range(int(seconds / step)):
            self.settle(step)

    def interactions(self):
        self.bus.drain()
        return [
            (e.type, e.data) for e in self.events
            if e.type in (EventType.INTERACTION_STARTED, EventType.INTERACTION_COMPLETED)
            and e.data.get("initiator") == "laika"
        ]

    def assertQuiet(self, wait_s=0.6):
        time.sleep(wait_s)
        self.assertEqual(self.interactions(), [])
        self.assertEqual(self.spoken() if self.runner else self.robot.spoken, [])


class CameraStabilizerTests(ProactiveCase):
    """Done when: the PRD's example works end to end."""

    def test_the_prd_example(self):
        self.assemble(text(OPENING), text(REPLY))
        self.run_loop()
        # The office is familiar; the stabiliser is new: she looks into it and queues a question, silently.
        self.notice_the_stabilizer()
        (question,) = self.investigator.questions.pending()
        self.assertEqual(question.text, QUESTION)
        self.assertEqual(self.spoken(), [])
        # Earlier talk is long past (cooldown over).
        self.bus.emit(EventType.INTERACTION_STARTED, initiator="user")
        self.bus.emit(EventType.INTERACTION_COMPLETED, initiator="user")
        self.bus.drain()
        self.advance(2000)
        # A person enters the view; too early to start (settle_s), so she waits.
        self.a_person_appears()
        time.sleep(0.5)
        self.assertEqual(self.spoken(), [])
        # They stay; the policy now agrees and the user is going to answer.
        self.ear.reply(ANSWER)
        self.settle(10)
        self.assertTrue(wait_for(lambda: len(self.spoken()) >= 2, LOOP_WAIT_S), self.spoken() or self.stdout.getvalue()[-3000:])
        self.assertTrue(wait_for(lambda: self.conversation.awake is False, LOOP_WAIT_S))

        # She asked with the model's words, which got the question text in their instruction.
        self.assertEqual(self.spoken(), [OPENING, REPLY])
        first = self.client.requests[0]
        self.assertIn(QUESTION, first["messages"][-1]["content"])
        self.assertEqual(
            {spec.name for spec in first["tools"]}, PROACTIVE_TOOLS & set(self.registry.names())
        )
        # She turned her head to the person before speaking.
        self.assertIn(HEAD_AT_PERSON, self.robot.poses)
        # The answer is memory: the object is known (with the answer as its note), the question is gone.
        known = self.world.where_is("camera_stabilizer")
        self.assertEqual(known.note, "uno stabilizzatore per la fotocamera")
        self.assertEqual(self.investigator.questions.pending(), [])
        self.assertIn("answer", [e.kind for e in self.memory.episodes.about("camera_stabilizer")])
        # InteractionStarted then InteractionCompleted, both from Laika, the second one answered.
        events = self.interactions()
        self.assertEqual([t for t, _ in events], [EventType.INTERACTION_STARTED, EventType.INTERACTION_COMPLETED])
        self.assertEqual(
            events[0][1], {"initiator": "laika", "reason": "pending_question", "questionId": question.id}
        )
        self.assertEqual(events[1][1], {"initiator": "laika", "answered": True, "questionId": question.id})
        self.assertIsNone(self.arbiter.holder)
        # She does not ask again at once: the attempt counts for the cooldown.
        self.assertEqual(self.policy.decide().reason, "cooldown")

    def test_the_user_can_still_wake_her_afterwards(self):
        self.assemble(text(OPENING), text(REPLY), text("Dimmi pure."))
        self.run_loop()
        self.notice_the_stabilizer()
        self.advance(2000)
        self.a_person_appears()
        self.ear.reply(ANSWER)
        self.settle(10)
        self.assertTrue(wait_for(lambda: len(self.spoken()) >= 2, LOOP_WAIT_S))
        self.assertTrue(wait_for(lambda: not self.conversation.awake, LOOP_WAIT_S))
        before = len(self.spoken())
        self.ear.reply("che ore sono?")
        self.ear.wake("ehi Laika")
        self.assertTrue(wait_for(lambda: len(self.spoken()) > before, LOOP_WAIT_S), self.spoken())
        self.assertEqual(self.client.requests[-1]["messages"][-1]["content"], "che ore sono?")


class QuietTests(ProactiveCase):
    """She stays quiet when the policy says no, and nothing is spoken."""

    def ready(self, **kwargs):
        kwargs.setdefault("proactive", False)  # the checks below are made by hand, not by the worker
        self.assemble(text(OPENING), text(REPLY), **kwargs)
        self.run_loop()
        self.notice_the_stabilizer()
        self.advance(2000)

    def arrive_and_settle(self):
        self.a_person_appears()
        self.settle(10)

    def test_during_the_cooldown(self):
        self.ready()
        self.bus.emit(EventType.INTERACTION_STARTED, initiator="user")
        self.bus.emit(EventType.INTERACTION_COMPLETED, initiator="user")
        self.bus.drain()
        self.advance(60)  # a minute after a chat
        self.arrive_and_settle()
        self.proactive.check("recheck")
        self.assertEqual(self.policy.decide().reason, "cooldown")
        self.assertQuiet()
        self.assertEqual(len(self.investigator.questions), 1)

    def test_do_not_disturb(self):
        self.ready()
        self.policy.set_do_not_disturb(30)
        self.arrive_and_settle()
        self.assertFalse(self.proactive.check("recheck"))
        self.assertQuiet()

    def test_quiet_hours(self):
        self.ready()
        self.wall.now = at(22)
        self.arrive_and_settle()
        self.assertFalse(self.proactive.check("recheck"))
        self.assertQuiet()

    def test_nothing_to_ask(self):
        self.assemble(text(OPENING), interaction=InteractionConfig(recheck_s=60.0), familiar=False)
        self.run_loop()
        self.advance(2000)
        self.arrive_and_settle()
        self.assertFalse(self.proactive.check("recheck"))
        self.assertEqual(self.policy.decide().reason, "nothing_to_say")
        self.assertQuiet()
        self.assertEqual(self.client.requests, [])

    def test_a_person_passing_by_is_not_ambushed(self):
        self.ready()
        self.a_person_appears()  # just arrived: not settled yet
        self.advance(2)
        self.assertFalse(self.proactive.check("recheck"))
        self.assertEqual(self.policy.decide().reason, "unknown")
        self.assertQuiet()

    def test_during_a_conversation_with_the_user(self):
        self.ready()
        self.arrive_and_settle()
        self.bus.emit(EventType.INTERACTION_STARTED, initiator="user")  # the user is talking to her
        self.bus.drain()
        self.assertFalse(self.proactive.check("recheck"))
        self.assertEqual(self.policy.decide().reason, "busy")
        self.assertQuiet()

    def test_while_a_more_important_behaviour_holds_the_body(self):
        self.ready()
        self.arrive_and_settle()
        lease = self.arbiter.request("user-command", Priority.USER_COMMAND)
        self.addCleanup(lease.release)
        self.assertFalse(self.proactive.check("recheck"))
        self.assertQuiet()

    def test_the_master_switch_turns_it_all_off(self):
        self.ready(interaction=InteractionConfig(recheck_s=60.0, proactive=False), proactive=True)
        self.arrive_and_settle()
        time.sleep(0.5)
        self.assertEqual(self.policy.recent_interactions(), [])
        self.assertQuiet()

    def test_a_request_that_finds_the_body_taken_is_dropped_quietly(self):
        self.ready()
        lease = self.arbiter.request("user-command", Priority.USER_COMMAND)
        self.addCleanup(lease.release)
        request = ProactiveRequest("q-1", QUESTION)
        self.assertFalse(self.conversation.converse_proactive(request))
        self.assertEqual(self.spoken(), [])
        self.assertEqual(self.interactions(), [])
        self.assertFalse(self.conversation.awake)


class NoNaggingTests(ProactiveCase):
    def test_nobody_answers_the_question_stays_and_the_cooldown_applies(self):
        self.assemble(text(OPENING))
        self.run_loop()
        self.notice_the_stabilizer()
        (question,) = self.investigator.questions.pending()
        self.advance(2000)
        self.a_person_appears()
        self.settle(10)  # nobody replies to her
        self.assertTrue(wait_for(lambda: len(self.interactions()) == 2, LOOP_WAIT_S))
        self.assertEqual(self.spoken(), [OPENING])  # one question, then silence
        (_, started), (_, completed) = self.interactions()
        self.assertEqual((started["initiator"], completed["answered"]), ("laika", False))
        self.assertEqual(self.investigator.questions.pending(), [question])
        self.assertIsNone(self.world.where_is("camera_stabilizer").note)
        self.assertEqual(len(self.agent.history), 0)
        # She does not come back for it while the cooldown lasts, however long the person stays.
        self.stay(600)
        self.assertFalse(self.proactive.check("recheck"))
        self.assertEqual(self.policy.decide().reason, "cooldown")
        time.sleep(0.5)
        self.assertEqual(self.spoken(), [OPENING])
        # After the cooldown the question is asked again.
        self.stay(1300)
        self.client.script.append(text("Ancora io: che cos'è?"))
        self.assertTrue(self.proactive.check("recheck"))
        self.assertTrue(wait_for(lambda: len(self.spoken()) == 2, LOOP_WAIT_S))


class FallbackTests(ProactiveCase):
    def test_the_template_question_is_spoken_when_the_ai_is_unavailable(self):
        self.assemble(AIUnavailable("rete assente"))
        self.run_loop()
        self.notice_the_stabilizer()
        self.advance(2000)
        self.a_person_appears()
        self.settle(10)
        self.assertTrue(wait_for(lambda: len(self.interactions()) == 2, LOOP_WAIT_S))
        self.assertEqual(self.spoken(), [QUESTION])
        self.assertEqual(self.interactions()[1][1]["answered"], False)

    def test_the_template_question_is_spoken_when_the_model_says_nothing(self):
        self.assemble(text(""))
        self.run_loop()
        self.notice_the_stabilizer()
        self.advance(2000)
        self.a_person_appears()
        self.settle(10)
        self.assertTrue(wait_for(lambda: self.spoken() == [QUESTION], LOOP_WAIT_S), self.spoken())


class RestrictedTurnTests(ProactiveCase):
    def test_the_proactive_registry_has_no_move_or_policy_tools(self):
        self.assemble()
        full = set(self.registry.names())
        for tool in ("move", "investigate", "set_do_not_disturb", "clear_do_not_disturb",
                     "set_busy", "clear_busy", "remember", "answer_question", "speak", "capture_image"):
            self.assertIn(tool, full)
        restricted = self.registry.restricted(PROACTIVE_TOOLS)
        self.assertEqual(set(restricted.names()), PROACTIVE_TOOLS & full)
        self.assertIn("gesture", restricted.names())
        self.assertIn("look_left", restricted.names())

    def test_a_forbidden_tool_in_the_opening_is_refused_and_changes_nothing(self):
        self.assemble(
            calls(("move", {"direction": "forward", "duration_ms": 500}),
                  ("clear_do_not_disturb", {}), ("set_busy", {"minutes": 5}), ("gesture", {"name": "nod"})),
            text(OPENING),
        )
        self.policy.set_do_not_disturb(30)
        position = self.robot.pose()
        reply = self.agent.respond(
            proactive_instruction(QUESTION), registry=self.registry.restricted(PROACTIVE_TOOLS)
        )
        self.assertEqual(reply.text, OPENING)
        self.assertEqual(self.robot.pose(), position)
        self.assertIsNotNone(self.policy.do_not_disturb_until)
        self.assertIsNone(self.policy.busy_until)
        refused = [name for name, why in self.registry.rejections if "unknown tool" in why]
        self.assertEqual(refused, ["move", "clear_do_not_disturb", "set_busy"])

    def test_the_instruction_carries_the_question(self):
        self.assertIn(QUESTION, proactive_instruction(QUESTION))


class FakeVosk:
    """Behaves like SunFounder's Vosk: listen() clears the stop flag when it starts."""

    def __init__(self):
        self.stop = threading.Event()
        self.started = threading.Event()

    def listen(self, stream=False):
        # Like SunFounder: the stop flag is cleared on the call, the waiting happens in the generator.
        self.stop.clear()
        self.started.set()
        return self._stream()

    def _stream(self):
        self.stop.wait(5)
        return
        yield  # a generator that ends when stopped, like SunFounder's streaming listen

    def stop_listening(self):
        self.stop.set()


class VoskEarInterruptTests(unittest.TestCase):
    def ear(self):
        from laika.audio import VoskEar

        ear = VoskEar.__new__(VoskEar)  # skip the real Vosk model
        ear._stt = FakeVosk()
        ear._interrupted = threading.Event()
        return ear

    def test_an_interrupt_during_a_listen_ends_it(self):
        ear = self.ear()
        threading.Timer(0.2, ear.interrupt).start()
        started = time.monotonic()
        self.assertIsNone(ear.listen())
        self.assertLess(time.monotonic() - started, 2)

    def test_an_interrupt_landing_as_listen_starts_is_not_lost(self):
        # The race: interrupt() stops the recogniser, then its listen() clears the flag again.
        ear = self.ear()
        original = ear._stt.listen

        def racing_listen(stream=False):
            ear.interrupt()  # lands after VoskEar's own check, before the vendor's clear
            return original(stream)

        ear._stt.listen = racing_listen
        started = time.monotonic()
        self.assertIsNone(ear.listen())
        self.assertLess(time.monotonic() - started, 2)


class InterruptTests(unittest.TestCase):
    def console_ear(self, *lines):
        ear = ConsoleEar()
        feed = iter(lines)

        def fake_input(*_):
            try:
                return next(feed)
            except StopIteration:
                raise EOFError from None

        patcher = unittest.mock.patch("builtins.input", fake_input)
        patcher.start()
        self.addCleanup(patcher.stop)
        return ear

    def listen(self, ear, timeout_s=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return ear.listen(timeout_s)

    def test_console_ear_interrupt_wakes_a_listen(self):
        blocker = threading.Event()
        ear = ConsoleEar()
        ear._reader = threading.Thread()  # the reader never delivers: only interrupt() can wake it
        result = []

        def listening():
            result.append(self.listen(ear))
            blocker.set()

        threading.Thread(target=listening, daemon=True).start()
        time.sleep(0.2)
        self.assertFalse(blocker.is_set())
        ear.interrupt()
        self.assertTrue(blocker.wait(3))
        self.assertEqual(result, [None])

    def test_console_ear_waits_for_typing_by_default(self):
        ear = ConsoleEar()
        ear._reader = threading.Thread()
        threading.Timer(0.3, ear._lines.put, args=("ciao",)).start()
        self.assertEqual(self.listen(ear, timeout_s=0.05), "ciao")  # the short timeout is ignored

    def test_console_ear_times_out_and_reads_lines(self):
        ear = ConsoleEar(wait_for_typing=False)
        ear._reader = threading.Thread()
        self.assertIsNone(self.listen(ear, timeout_s=0.1))
        ear._lines.put("  ciao  ")
        self.assertEqual(self.listen(ear, timeout_s=1), "ciao")
        ear._lines.put("")
        self.assertIsNone(self.listen(ear, timeout_s=1))

    def test_console_ear_end_of_input_still_ends_the_app(self):
        ear = self.console_ear("ehi laika")
        self.assertEqual(self.listen(ear, timeout_s=2), "ehi laika")
        with self.assertRaises(KeyboardInterrupt):
            self.listen(ear, timeout_s=2)
        with self.assertRaises(KeyboardInterrupt):
            self.listen(ear)

    def test_wait_for_wake_returns_the_signal_and_a_wake_word_still_wins(self):
        ear = ScriptedEar()
        matcher = WakeWordMatcher(["laika"])
        signals = queue.Queue()

        def poll():
            return signals.get_nowait() if not signals.empty() else None

        outcome = []
        thread = threading.Thread(
            target=lambda: outcome.append(wait_for_wake(ear, matcher, poll)), daemon=True
        )
        thread.start()
        time.sleep(0.1)
        self.assertEqual(outcome, [])
        signals.put("something to do")
        ear.interrupt()
        thread.join(3)
        self.assertEqual(outcome, ["something to do"])
        # A plain timeout or an unrelated sentence keeps it waiting.
        outcome.clear()
        thread = threading.Thread(
            target=lambda: outcome.append(wait_for_wake(ear, matcher, poll)), daemon=True
        )
        thread.start()
        ear.interrupt()
        ear.wake("che bel tempo")
        time.sleep(0.2)
        self.assertEqual(outcome, [])
        ear.wake("ehi Laika")
        thread.join(3)
        self.assertFalse(outcome[0].remainder)

    def test_request_proactive_wakes_a_sleeping_conversation_and_refuses_while_awake(self):
        ear = ScriptedEar()
        conversation = Conversation(
            ear=ear, robot=_Mute(), ai=JsonAI(AIReply("x")), actions=_NoActions(), voice=VoiceConfig()
        )
        request = ProactiveRequest("q-1", QUESTION)
        conversation._waiting = True
        self.assertTrue(conversation.request_proactive(request))
        self.assertEqual(ear.interrupts, 1)
        self.assertFalse(conversation.request_proactive(request))  # one at a time
        self.assertIs(conversation._take_proactive(), request)
        self.assertTrue(conversation.awake)
        self.assertFalse(conversation.request_proactive(request))  # busy talking
        self.assertEqual(ear.interrupts, 1)

    def test_a_stale_request_is_dropped(self):
        conversation = Conversation(
            ear=ScriptedEar(), robot=_Mute(), ai=JsonAI(AIReply("x")), actions=_NoActions(), voice=VoiceConfig()
        )
        request = ProactiveRequest("q-1", QUESTION, created=time.monotonic() - 3600)
        conversation.request_proactive(request)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(conversation._take_proactive())
        self.assertFalse(conversation.awake)


class _Mute:
    def speak(self, text):
        pass


class _NoActions:
    def cancel_pending(self):
        pass


if __name__ == "__main__":
    unittest.main()
