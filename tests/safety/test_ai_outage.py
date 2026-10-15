import threading
import time
import unittest

from laika.actions import ACTIONS, Action, ActionContext, ActionPlayer
from laika.ai_client import AIReply, AIUnavailable
from laika.commands import Direction, MoveCommand, ResultStatus
from laika.config import VoiceConfig
from laika.conversation import NO_CONNECTION, Conversation
from laika.events import AIEndpointMonitor, EventBus, EventType
from laika.safety import SafetyController
from tests.fakes import RecordingRobot

MARGIN_S = 0.5


def move(duration_ms=1000):
    return MoveCommand(direction=Direction.FORWARD, speed=25, duration_ms=duration_ms)


def wait_for(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


class OutageTestCase(unittest.TestCase):
    def setUp(self):
        self.robot = RecordingRobot(distance_cm=100)
        self.bus = EventBus()
        self.bus.start()
        self.addCleanup(self.bus.close)
        self.seen = []
        self.bus.subscribe(self.seen.append)
        self.controller = SafetyController(self.robot, speed=25, events=self.bus)
        self.addCleanup(self.controller.close)
        self.controller.start()


class ControllerOutageTests(OutageTestCase):
    def test_an_outage_during_a_move_stops_the_robot(self):
        command = move()
        self.assertIsNone(self.controller.execute(command, wait=False))
        self.assertTrue(wait_for(lambda: self.robot.speed == 25))
        self.controller.set_ai_available(False)
        self.assertEqual(self.robot.speed, 0)  # synchronously, not at the next tick
        result = self.controller.wait(command.id, timeout=1)
        self.assertEqual((result.result, result.reason), (ResultStatus.CANCELLED, "ai_unavailable"))

    def test_moves_are_rejected_while_unavailable_and_allowed_after_recovery(self):
        self.controller.set_ai_available(False)
        result = self.controller.execute(move())
        self.assertEqual((result.result, result.reason), (ResultStatus.REJECTED, "ai_unavailable"))
        self.assertEqual(self.robot.speeds, [])
        self.controller.set_ai_available(True)
        self.assertEqual(self.controller.execute(move(100)).result, ResultStatus.SUCCESS)

    def test_the_stop_command_still_works_during_an_outage(self):
        self.controller.set_ai_available(False)
        self.assertEqual(self.controller.execute_raw({"type": "command", "command": "stop"}).result, ResultStatus.SUCCESS)

    def test_a_move_that_slips_in_is_cancelled_by_the_monitor(self):
        command = move()
        self.assertIsNone(self.controller.execute(command, wait=False))
        self.controller._ai_available = False  # as if the flag flipped without the reaction
        result = self.controller.wait(command.id, timeout=1)
        self.assertEqual(result.reason, "ai_unavailable")
        self.assertEqual(self.robot.speed, 0)

    def test_a_stalled_monitor_does_not_block_the_outage_reaction(self):
        # The monitor holds the lock forever; the reaction still stops the robot.
        command = move()
        self.assertIsNone(self.controller.execute(command, wait=False))
        self.assertTrue(wait_for(lambda: self.robot.speed == 25))
        self.controller._lock.acquire()
        self.addCleanup(self.controller._lock.release)
        started = time.monotonic()
        self.controller.set_ai_available(False)
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertEqual(self.robot.speed, 0)
        self.assertEqual(self.controller.wait(command.id, 0) is None, False)


class MonitorCallbackTests(unittest.TestCase):
    def test_callbacks_run_on_transitions_only_and_before_the_event(self):
        order = []
        events = []
        bus = EventBus()
        bus.start()
        self.addCleanup(bus.close)

        def deliver(event):
            events.append(event.type.value)
            order.append(("event", event.type.value, list(order)))

        bus.subscribe(deliver)
        monitor = AIEndpointMonitor(
            bus, lambda: order.append("down-cb"), lambda: order.append("up-cb")
        )
        monitor.succeeded()
        monitor.failed("x")
        monitor.failed("y")
        monitor.succeeded()
        monitor.succeeded()
        self.assertTrue(bus.drain())
        self.assertEqual(events, ["AIEndpointUnavailable", "AIEndpointRecovered"])
        # Each event arrived after its callback had already run.
        first_event = next(i for i, item in enumerate(order) if not isinstance(item, str))
        self.assertIn("down-cb", order[:first_event])
        callbacks = [item for item in order if isinstance(item, str)]
        self.assertEqual(callbacks, ["down-cb", "up-cb"])

    def test_a_slow_subscriber_cannot_delay_the_reaction(self):
        bus = EventBus()
        bus.start()
        release = threading.Event()
        self.addCleanup(bus.close)
        self.addCleanup(release.set)
        bus.subscribe(lambda e: release.wait(5))
        reacted = []
        monitor = AIEndpointMonitor(bus, lambda: reacted.append(1))
        monitor.failed("x")
        self.assertEqual(reacted, [1])

    def test_a_failing_callback_does_not_break_the_caller_or_the_event(self):
        bus = EventBus()
        bus.start()
        self.addCleanup(bus.close)
        seen = []
        bus.subscribe(seen.append)

        def boom():
            raise RuntimeError("boom")

        AIEndpointMonitor(bus, boom).failed("x")
        self.assertTrue(bus.drain())
        self.assertEqual([e.type for e in seen], [EventType.AI_ENDPOINT_UNAVAILABLE])


class OutageWithGesturesTests(OutageTestCase):
    def setUp(self):
        super().setUp()
        self.hold = threading.Event()
        self.addCleanup(self.hold.set)
        self.started = threading.Event()
        self.drive_results = []

        def hold(ctx):
            self.started.set()
            self.hold.wait(5)
            # A running gesture that tries to drive after the outage began.
            self.drive_results.append(ctx.safety.execute(move(100)))

        allowed = {
            "hold": Action("hold", "", hold, frozenset()),
            "forward": ACTIONS["forward"],
            "nod": ACTIONS["nod"],
        }
        ctx = ActionContext(self.robot, self.controller)
        self.player = ActionPlayer(ctx, allowed, talk_motion=False)
        self.addCleanup(self.player.close)
        self.monitor = AIEndpointMonitor(
            self.bus,
            lambda: (self.controller.set_ai_available(False), self.player.cancel_pending()),
            lambda: self.controller.set_ai_available(True),
        )

    def test_queued_gestures_are_dropped_and_not_replayed_after_recovery(self):
        self.player.play("hold", "forward", "forward", "nod")
        self.assertTrue(self.started.wait(2))
        self.monitor.failed("timeout")  # while "hold" runs; the others are still queued
        self.hold.set()
        self.assertTrue(self.player.wait_done(timeout=3))
        self.monitor.succeeded()
        time.sleep(0.3)
        self.assertTrue(self.player.wait_done(timeout=1))
        self.assertEqual(self.robot.speeds, [])  # no drive, before or after recovery
        # The drive the running gesture attempted during the outage was refused.
        self.assertEqual(
            [(r.result, r.reason) for r in self.drive_results],
            [(ResultStatus.REJECTED, "ai_unavailable")],
        )

    def test_gestures_work_again_after_recovery(self):
        self.monitor.failed("timeout")
        self.monitor.succeeded()
        self.player.play("forward")
        self.assertTrue(wait_for(lambda: 25 in self.robot.speeds))
        self.assertTrue(self.player.wait_done(timeout=5))

    def test_cancel_pending_keeps_the_running_gesture(self):
        self.player.play("hold", "nod")
        self.assertTrue(self.started.wait(2))
        self.player.cancel_pending()
        self.hold.set()
        self.assertTrue(self.player.wait_done(timeout=3))
        self.assertEqual(len(self.drive_results), 1)


class OrderedPlayer(ActionPlayer):
    def __init__(self, *args, **kwargs):
        self.calls = []
        super().__init__(*args, **kwargs)

    def play(self, *names):
        self.calls.append(("play", names))
        super().play(*names)

    def cancel_pending(self):
        self.calls.append(("cancel_pending", ()))
        super().cancel_pending()


class FakeAI:
    def __init__(self, replies):
        self.replies = list(replies)

    def ask(self, question):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def reset(self):
        pass


class ConversationOutageTests(OutageTestCase):
    def conversation(self, replies):
        ctx = ActionContext(self.robot, self.controller)
        self.player = OrderedPlayer(
            ctx,
            {k: ACTIONS[k] for k in ("think", "sad", "look_forward", "nod", "forward")},
            talk_motion=False,
        )
        self.addCleanup(self.player.close)
        self.robot.verbose = False
        return Conversation(
            ear=None,
            robot=self.robot,
            ai=FakeAI(replies),
            actions=self.player,
            voice=VoiceConfig(),
            events=self.bus,
            safety=self.controller,
        )

    def test_an_outage_cancels_queued_actions_before_the_sad_gesture_and_speaks(self):
        conversation = self.conversation([AIUnavailable("down")])
        conversation.reply("ciao")
        calls = self.player.calls
        sad = max(i for i, (kind, names) in enumerate(calls) if kind == "play" and "sad" in names)
        cancels = [i for i, (kind, _) in enumerate(calls) if kind == "cancel_pending"]
        self.assertTrue(cancels and max(cancels) < sad)
        self.assertEqual(self.robot.spoken, [NO_CONNECTION])
        self.assertFalse(self.controller.ai_available)
        self.assertTrue(self.bus.drain())
        self.assertIn(EventType.AI_ENDPOINT_UNAVAILABLE, [e.type for e in self.seen])

    def test_a_later_answer_recovers_and_allows_moves(self):
        conversation = self.conversation([AIUnavailable("down"), AIReply("ok", ["forward"])])
        conversation.reply("uno")
        self.assertEqual(self.controller.execute(move(100)).reason, "ai_unavailable")
        conversation.reply("due")
        self.assertTrue(self.controller.ai_available)
        self.assertIn(25, self.robot.speeds)  # the new reply's forward gesture drove
        self.assertTrue(self.bus.drain())
        self.assertIn(EventType.AI_ENDPOINT_RECOVERED, [e.type for e in self.seen])


if __name__ == "__main__":
    unittest.main()
