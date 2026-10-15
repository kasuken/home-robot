import threading
import time
import unittest

from laika.actions import ACTIONS, ActionContext, ActionPlayer
from laika.ai_client import AIReply, ToolCall
from laika.behaviour import Arbiter, Lease, Preempted, Priority
from laika.commands import Direction, MoveCommand, ResultStatus
from laika.config import VoiceConfig
from laika.conversation import Conversation
from laika.safety import SafetyController
from laika.tools import ToolRegistry, Turn
from tests.fakes import RecordingRobot

P = Priority


def move(duration_ms=1000):
    return MoveCommand(direction=Direction.FORWARD, speed=25, duration_ms=duration_ms)


def wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


class StubActions:
    def __init__(self):
        self.cancels = 0

    def cancel_pending(self):
        self.cancels += 1


class StallRobot(RecordingRobot):
    """A robot whose sensor read can be made to hang, like a stuck I2C call."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stalled = threading.Event()
        self.unstall = threading.Event()

    def get_sensors(self):
        if self.stalled.is_set():
            self.unstall.wait(10)
        return super().get_sensors()


class BehaviourTestCase(unittest.TestCase):
    lease_timeout_s = 2.0
    controller_kwargs: dict = {}

    def setUp(self):
        self.robot = StallRobot(distance_cm=100)
        self.controller = SafetyController(self.robot, speed=25, **self.controller_kwargs)
        self.addCleanup(self.controller.close)
        self.addCleanup(self.robot.unstall.set)  # runs first: a stalled monitor can then exit
        self.controller.start()
        self.actions = StubActions()
        self.arbiter = Arbiter(self.controller, self.actions, lease_timeout_s=self.lease_timeout_s)
        self.addCleanup(self.arbiter.close)

    def start_move(self, lease, duration_ms=1500):
        command = move(duration_ms)
        self.assertIsNone(self.controller.execute(command, wait=False, lease=lease))
        self.assertTrue(wait_for(lambda: self.robot.speed == 25))
        return command


class PriorityTests(unittest.TestCase):
    def test_order_follows_the_prd(self):
        order = [
            P.SAFETY, P.USER_COMMAND, P.CONVERSATION, P.FOLLOW_UP,
            P.IMPORTANT_OBSERVATION, P.CURIOSITY, P.SOCIAL, P.EXPLORATION,
        ]
        self.assertEqual(order, sorted(order))
        self.assertEqual([int(p) for p in order], list(range(8)))


class RequestTests(BehaviourTestCase):
    def test_a_free_body_is_granted(self):
        lease = self.arbiter.request("explore", P.EXPLORATION)
        self.assertTrue(lease.active)
        self.assertIs(self.arbiter.holder, lease)

    def test_safety_cannot_be_requested(self):
        with self.assertRaises(ValueError):
            self.arbiter.request("safety", P.SAFETY)

    def test_equal_and_lower_priority_are_refused(self):
        self.arbiter.request("curious", P.CURIOSITY)
        self.assertIsNone(self.arbiter.request("other", P.CURIOSITY))
        self.assertIsNone(self.arbiter.request("explore", P.EXPLORATION))
        self.assertIsNone(self.arbiter.request("social", P.SOCIAL))

    def test_a_more_important_request_preempts_and_the_holder_resumes(self):
        resumed = []
        explore = self.arbiter.request("explore", P.EXPLORATION, on_resume=resumed.append)
        chat = self.arbiter.request("chat", P.CONVERSATION)
        self.assertIsNotNone(chat)
        self.assertFalse(explore.active)
        self.assertTrue(explore.waiting)
        with self.assertRaises(Preempted):
            explore.check()
        self.assertEqual(self.actions.cancels, 1)
        self.arbiter.release(chat)
        self.assertTrue(explore.active)
        self.assertIs(self.arbiter.holder, explore)
        self.assertEqual(resumed, [explore])
        explore.check()

    def test_wait_until_active_returns_when_the_body_is_back(self):
        explore = self.arbiter.request("explore", P.EXPLORATION)
        chat = self.arbiter.request("chat", P.CONVERSATION)
        self.assertFalse(explore.wait_until_active(0.05))
        timer = threading.Timer(0.1, chat.release)
        timer.start()
        self.addCleanup(timer.cancel)
        self.assertTrue(explore.wait_until_active(5))

    def test_a_non_resumable_holder_ends(self):
        explore = self.arbiter.request("explore", P.EXPLORATION, resumable=False)
        chat = self.arbiter.request("chat", P.CONVERSATION)
        self.assertTrue(explore.ended)
        self.assertFalse(explore.wait_until_active(0.05))
        self.arbiter.release(chat)
        self.assertIsNone(self.arbiter.holder)
        with self.assertRaises(Preempted):
            explore.check()

    def test_a_user_command_preempts_a_conversation_and_both_come_back_in_order(self):
        explore = self.arbiter.request("explore", P.EXPLORATION)
        chat = self.arbiter.request("chat", P.CONVERSATION)
        command = self.arbiter.request("command", P.USER_COMMAND)
        self.assertTrue(command.active)
        self.assertTrue(chat.waiting and explore.waiting)
        self.arbiter.release(command)
        self.assertIs(self.arbiter.holder, chat)
        self.assertTrue(explore.waiting)
        self.arbiter.release(chat)
        self.assertIs(self.arbiter.holder, explore)

    def test_a_resumed_behaviour_holds_the_body_against_equal_priority(self):
        explore = self.arbiter.request("explore", P.EXPLORATION)
        chat = self.arbiter.request("chat", P.CONVERSATION)
        self.arbiter.release(chat)
        self.assertTrue(explore.active)
        self.assertIsNone(self.arbiter.request("explore again", P.EXPLORATION))

    def test_releasing_a_waiting_lease_removes_it(self):
        explore = self.arbiter.request("explore", P.EXPLORATION)
        chat = self.arbiter.request("chat", P.CONVERSATION)
        explore.release()
        explore.release()  # idempotent
        self.assertEqual(self.arbiter.waiting, [])
        self.arbiter.release(chat)
        self.assertIsNone(self.arbiter.holder)

    def test_the_waiting_list_is_bounded(self):
        arbiter = Arbiter(self.controller, max_waiting=1)
        self.addCleanup(arbiter.close)
        a = arbiter.request("a", P.EXPLORATION)
        b = arbiter.request("b", P.CURIOSITY)
        c = arbiter.request("c", P.CONVERSATION)
        self.assertTrue(a.waiting)
        self.assertTrue(b.ended)  # the list was full: the newest pre-empted one is dropped
        arbiter.release(c)
        self.assertTrue(a.active)

    def test_on_preempt_runs_outside_the_arbiter_lock(self):
        outcome = []

        def on_preempt(lease):
            probe = threading.Thread(target=lambda: self.arbiter.waiting)
            probe.start()
            probe.join(2)
            outcome.append(not probe.is_alive())

        self.arbiter.request("explore", P.EXPLORATION, on_preempt=on_preempt)
        self.arbiter.request("chat", P.CONVERSATION)
        self.assertEqual(outcome, [True])

    def test_a_failing_callback_does_not_break_the_preemption(self):
        def boom(_lease):
            raise RuntimeError("boom")

        self.arbiter.request("explore", P.EXPLORATION, on_preempt=boom)
        self.assertIsNotNone(self.arbiter.request("chat", P.CONVERSATION))


class Explorer:
    """A fake exploration behaviour: small moves under its lease, kicking between steps."""

    def __init__(self, arbiter, controller, priority=P.EXPLORATION, duration_ms=500):
        self.arbiter, self.controller = arbiter, controller
        self.priority, self.duration_ms = priority, duration_ms
        self.lease = None
        self.results = []
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.lease = self.arbiter.request("explore", self.priority)
        assert self.lease is not None
        self.thread.start()

    def stop(self):
        self._stop.set()
        self.thread.join(5)

    def run(self):
        lease = self.lease
        while not self._stop.is_set():
            try:
                lease.check()
            except Preempted:
                if lease.ended:
                    return
                lease.wait_until_active(0.05)
                continue
            self.results.append(self.controller.execute(move(self.duration_ms), lease=lease))
            lease.kick()


class PreemptionTests(BehaviourTestCase):
    def test_a_conversation_interrupts_exploration_and_exploration_resumes(self):
        explorer = Explorer(self.arbiter, self.controller, duration_ms=1000)
        explorer.start()
        self.addCleanup(explorer.stop)
        self.assertTrue(wait_for(lambda: self.robot.speed == 25))

        chat = self.arbiter.request("chat", P.CONVERSATION)
        self.assertIsNotNone(chat)
        self.assertEqual(self.robot.speed, 0)
        self.assertTrue(wait_for(lambda: explorer.results))
        first = explorer.results[0]
        self.assertEqual((first.result, first.reason), (ResultStatus.CANCELLED, "preempted"))

        # A late move of the pre-empted behaviour is refused before the wheels turn.
        late = self.controller.execute(move(200), lease=explorer.lease)
        self.assertEqual((late.result, late.reason), (ResultStatus.REJECTED, "preempted"))
        time.sleep(0.3)
        self.assertEqual(self.robot.speed, 0)

        # The conversation moves freely while the explorer waits.
        self.assertEqual(self.controller.execute(move(100), lease=chat).result, ResultStatus.SUCCESS)

        count = len(explorer.results)
        self.arbiter.release(chat)
        self.assertTrue(
            wait_for(
                lambda: any(r.result is ResultStatus.SUCCESS for r in explorer.results[count:]),
                timeout=8,
            )
        )

    def test_a_user_command_interrupts_a_conversation_move(self):
        chat = self.arbiter.request("chat", P.CONVERSATION)
        command = self.start_move(chat)
        user = self.arbiter.request("command", P.USER_COMMAND)
        self.assertIsNotNone(user)
        result = self.controller.wait(command.id, timeout=2)
        self.assertEqual((result.result, result.reason), (ResultStatus.CANCELLED, "preempted"))
        self.assertEqual(self.controller.execute(move(100), lease=chat).reason, "preempted")
        self.assertEqual(self.controller.execute(move(100), lease=user).result, ResultStatus.SUCCESS)

    def test_a_user_command_interrupts_exploration(self):
        explore = self.arbiter.request("explore", P.EXPLORATION)
        command = self.start_move(explore)
        self.assertIsNotNone(self.arbiter.request("command", P.USER_COMMAND))
        result = self.controller.wait(command.id, timeout=2)
        self.assertEqual((result.result, result.reason), (ResultStatus.CANCELLED, "preempted"))

    def test_moves_without_a_lease_work_as_before(self):
        self.assertEqual(self.controller.execute(move(100)).result, ResultStatus.SUCCESS)


class HeartbeatTests(BehaviourTestCase):
    lease_timeout_s = 0.3

    def test_a_stalled_autonomous_lease_is_revoked_and_the_robot_stops(self):
        explore = self.arbiter.request("explore", P.EXPLORATION)
        command = self.start_move(explore)
        self.assertTrue(wait_for(lambda: explore.ended))
        self.assertEqual(explore.end_reason, "lease_expired")
        result = self.controller.wait(command.id, timeout=2)
        self.assertEqual((result.result, result.reason), (ResultStatus.CANCELLED, "lease_expired"))
        self.assertEqual(self.robot.speed, 0)
        self.assertIsNone(self.arbiter.holder)
        self.assertEqual(self.controller.execute(move(100), lease=explore).reason, "preempted")

    def test_kicking_keeps_the_lease(self):
        explore = self.arbiter.request("explore", P.CURIOSITY)
        end = time.monotonic() + 1.0
        while time.monotonic() < end:
            explore.kick()
            time.sleep(0.05)
        self.assertTrue(explore.active)

    def test_a_conversation_lease_needs_no_heartbeat(self):
        chat = self.arbiter.request("chat", P.CONVERSATION)
        self.arbiter.request("explore", P.EXPLORATION)  # refused; nothing to expire
        time.sleep(1.0)
        self.assertTrue(chat.active)

    def test_waiting_leases_do_not_expire_and_resume_with_a_fresh_heartbeat(self):
        explore = self.arbiter.request("explore", P.EXPLORATION)
        chat = self.arbiter.request("chat", P.CONVERSATION)
        time.sleep(0.8)
        self.assertTrue(explore.waiting)
        self.arbiter.release(chat)
        time.sleep(0.1)
        self.assertTrue(explore.active)


class SafetyWinsTests(BehaviourTestCase):
    controller_kwargs = {"watchdog_timeout_s": 0.2, "watchdog_period_s": 0.05}

    def stall_monitor(self):
        self.robot.stalled.set()

    def hardware_error(self):
        self.robot.faults.drive_error = True
        self.controller.execute(move(100))

    def revokes_autonomy(self, trigger):
        explore = self.arbiter.request("explore", P.EXPLORATION)
        trigger()
        self.assertTrue(wait_for(lambda: explore.ended))
        self.assertIsNone(self.arbiter.holder)
        refused = self.controller.execute(move(100), lease=explore)
        self.assertEqual(refused.result, ResultStatus.REJECTED)

    def keeps_the_conversation(self, trigger):
        explore = self.arbiter.request("explore", P.EXPLORATION)
        chat = self.arbiter.request("chat", P.CONVERSATION)
        self.assertTrue(explore.waiting)
        trigger()
        self.assertTrue(wait_for(lambda: explore.ended))
        self.assertTrue(chat.active)
        self.assertIs(self.arbiter.holder, chat)

    def test_emergency_stop(self):
        self.revokes_autonomy(lambda: self.controller.emergency_stop("test"))

    def test_emergency_stop_keeps_the_conversation(self):
        self.keeps_the_conversation(lambda: self.controller.emergency_stop("test"))

    def test_ai_outage(self):
        self.revokes_autonomy(lambda: self.controller.set_ai_available(False))

    def test_ai_outage_keeps_the_conversation(self):
        self.keeps_the_conversation(lambda: self.controller.set_ai_available(False))

    def test_watchdog_expiry(self):
        self.revokes_autonomy(self.stall_monitor)

    def test_watchdog_expiry_keeps_the_conversation(self):
        self.keeps_the_conversation(self.stall_monitor)

    def test_hardware_error(self):
        self.revokes_autonomy(self.hardware_error)

    def test_hardware_error_keeps_the_conversation(self):
        self.keeps_the_conversation(self.hardware_error)

    def test_on_preempt_is_told_why_and_gestures_are_dropped(self):
        told = []
        explore = self.arbiter.request("explore", P.EXPLORATION, on_preempt=told.append)
        self.controller.emergency_stop("test")
        self.assertTrue(wait_for(lambda: told == [explore]))
        self.assertEqual(explore.end_reason, "emergency_stop")
        self.assertGreaterEqual(self.actions.cancels, 1)

    def test_autonomy_is_refused_during_an_emergency_stop_and_returns_after_it(self):
        self.controller.emergency_stop("test")
        self.assertIsNone(self.arbiter.request("explore", P.EXPLORATION))
        chat = self.arbiter.request("chat", P.CONVERSATION)
        self.assertIsNotNone(chat)
        self.arbiter.release(chat)
        self.controller.reset_emergency_stop()
        self.assertIsNotNone(self.arbiter.request("explore", P.EXPLORATION))

    def test_autonomy_returns_after_an_ai_outage_recovers(self):
        self.controller.set_ai_available(False)
        self.assertIsNone(self.arbiter.request("explore", P.EXPLORATION))
        self.controller.set_ai_available(True)
        self.assertIsNotNone(self.arbiter.request("explore", P.EXPLORATION))

    def test_the_conversation_lease_stays_but_its_moves_are_still_refused(self):
        chat = self.arbiter.request("chat", P.CONVERSATION)
        self.controller.set_ai_available(False)
        self.assertTrue(chat.active)
        result = self.controller.execute(move(100), lease=chat)
        self.assertEqual((result.result, result.reason), (ResultStatus.REJECTED, "ai_unavailable"))

    def test_a_stalled_monitor_does_not_block_the_revocation(self):
        explore = self.arbiter.request("explore", P.EXPLORATION)
        self.robot.stalled.set()
        self.assertTrue(wait_for(lambda: explore.ended))
        self.assertTrue(self.controller.watchdog_expired)


class SafetyChecksStillApplyTests(BehaviourTestCase):
    def test_an_obstacle_refuses_a_move_with_a_valid_lease(self):
        user = self.arbiter.request("command", P.USER_COMMAND)
        self.robot.distance = 10
        result = self.controller.execute(move(), lease=user)
        self.assertEqual((result.result, result.reason), (ResultStatus.REJECTED, "obstacle"))
        self.assertEqual(self.robot.speeds, [])

    def test_a_cliff_refuses_a_move_with_a_valid_lease(self):
        user = self.arbiter.request("command", P.USER_COMMAND)
        self.robot.cliff = True
        result = self.controller.execute(move(), lease=user)
        self.assertEqual((result.result, result.reason), (ResultStatus.REJECTED, "cliff"))

    def test_an_obstacle_still_stops_a_leased_move_in_flight(self):
        explore = self.arbiter.request("explore", P.EXPLORATION)
        command = self.start_move(explore)
        self.robot.distance = 10
        result = self.controller.wait(command.id, timeout=2)
        self.assertEqual((result.result, result.reason), (ResultStatus.CANCELLED, "obstacle"))

    def test_an_emergency_stop_refuses_a_leased_move(self):
        user = self.arbiter.request("command", P.USER_COMMAND)
        self.controller.emergency_stop("test")
        self.assertEqual(self.controller.execute(move(), lease=user).reason, "emergency_stop")


class FakeEar:
    def listen(self, timeout_s=None):
        return None


class HoldingAI:
    def __init__(self, arbiter):
        self.arbiter, self.holders = arbiter, []

    def ask(self, question):
        self.holders.append(self.arbiter.holder)
        return AIReply("ok")

    def reset(self):
        pass


class ConversationLeaseTests(BehaviourTestCase):
    def setUp(self):
        super().setUp()
        self.robot.verbose = False
        ctx = ActionContext(self.robot, self.controller)
        names = ("think", "look_up", "look_forward")
        self.player = ActionPlayer(ctx, {k: ACTIONS[k] for k in names}, talk_motion=False)
        self.addCleanup(self.player.close)
        self.ai = HoldingAI(self.arbiter)
        self.conversation = Conversation(
            ear=FakeEar(),
            robot=self.robot,
            ai=self.ai,
            actions=self.player,
            voice=VoiceConfig(),
            safety=self.controller,
            arbiter=self.arbiter,
        )

    def test_waking_acquires_and_the_end_of_the_chat_releases(self):
        self.conversation.converse("ciao")
        (holder,) = self.ai.holders
        self.assertEqual((holder.name, holder.priority), ("conversation", P.CONVERSATION))
        self.assertTrue(holder.ended)
        self.assertIsNone(self.arbiter.holder)

    def test_the_sleep_word_releases_too(self):
        self.conversation.converse("basta")
        self.assertIsNone(self.arbiter.holder)
        self.assertEqual(self.ai.holders, [])

    def test_the_lease_is_released_when_the_chat_fails(self):
        def boom(question):
            raise RuntimeError("boom")

        self.ai.ask = boom
        with self.assertRaises(RuntimeError):
            self.conversation.converse("ciao")
        self.assertIsNone(self.arbiter.holder)

    def test_exploration_cannot_take_the_body_during_a_chat_but_can_afterwards(self):
        seen = []

        def ask(question):
            seen.append(self.arbiter.request("explore", P.EXPLORATION))
            return AIReply("ok")

        self.ai.ask = ask
        self.conversation.converse("ciao")
        self.assertEqual(seen, [None])
        self.assertIsNotNone(self.arbiter.request("explore", P.EXPLORATION))

    def test_a_chat_that_cannot_get_the_body_keeps_going_without_the_right_to_move(self):
        user = self.arbiter.request("command", P.USER_COMMAND)
        self.conversation.converse("ciao")
        self.assertEqual(len(self.ai.holders), 1)
        self.assertIs(self.arbiter.holder, user)

    def test_waking_during_exploration_preempts_it_and_it_resumes_afterwards(self):
        explore = self.arbiter.request("explore", P.EXPLORATION)
        self.conversation.converse("ciao")
        self.assertEqual(self.ai.holders[0].name, "conversation")
        self.assertTrue(explore.active)


class ToolsUnderLeaseTests(BehaviourTestCase):
    def call(self, name, **arguments):
        return ToolCall(id="1", name=name, arguments=arguments)

    def test_tools_run_under_an_active_lease(self):
        chat = self.arbiter.request("chat", P.CONVERSATION)
        registry = ToolRegistry(self.robot, self.controller)
        result = registry.run(self.call("move", direction="forward", duration_ms=200), Turn(lease=chat))
        self.assertTrue(result["ok"], result)

    def test_a_preempted_turn_cannot_move_or_look(self):
        chat = self.arbiter.request("chat", P.CONVERSATION)
        self.arbiter.request("command", P.USER_COMMAND)
        registry = ToolRegistry(self.robot, self.controller)
        for call in (
            self.call("move", direction="forward", duration_ms=200),
            self.call("look", pan=10, tilt=0),
        ):
            result = registry.run(call, Turn(lease=chat))
            self.assertFalse(result["ok"])
        self.assertEqual(self.robot.speeds, [])

    def test_a_handler_refuses_when_the_lease_goes_after_validation(self):
        registry = ToolRegistry(self.robot, self.controller)
        chat = self.arbiter.request("chat", P.CONVERSATION)
        turn = Turn(lease=chat)
        tool = registry._tools["move"]
        self.arbiter.request("command", P.USER_COMMAND)  # preempted after the stale check
        result = registry._dispatch(tool, {"direction": "forward", "duration_ms": 200}, turn)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "preempted")
        self.assertTrue(turn.stale_reason)


if __name__ == "__main__":
    unittest.main()
