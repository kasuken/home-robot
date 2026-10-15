"""The exploration loop (#27), against the house simulator, the real safety controller, arbiter and scanner."""

import sys
import threading
import time
import unittest
from unittest import mock

from laika.ai_client import ToolCall
from laika.behaviour import Arbiter, Priority
from laika.camera import Camera
from laika.commands import Direction, MoveCommand, ResultStatus
from laika.config import Config, ExploreConfig
from laika.explore import ExploreDecision, Explorer, IdleStarter
from laika.house import Pose, default_house
from laika.prompts import build_tool_system_prompt
from laika.safety import SafetyController
from laika.scan import Scanner
from laika.tools import ToolRegistry, Turn
from tests.fakes import FakeClock, LookRecordingRobot
from tests.simulator.test_robot_contract import SpyMouth, fake_sunfounder_modules

P = Priority


def wait_for(condition, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def move(duration_ms=200):
    return MoveCommand(direction=Direction.FORWARD, speed=25, duration_ms=duration_ms)


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
    def __init__(self):
        self.calls = []

    def set_exploring(self, exploring):
        self.calls.append(exploring)


def settings(**overrides):
    values = dict(speed=15, step_ms=100, scan_every_steps=3, max_duration_s=30.0, max_steps=10000)
    values.update(overrides)
    return ExploreConfig(**values)


class ExploreCase(unittest.TestCase):
    def build(
        self,
        robot=None,
        *,
        config=None,
        allow_autonomous=True,
        lease_timeout_s=2.0,
        on_scan=None,
        drive=None,
        safety_kwargs=None,
    ):
        self.robot = robot or TraceRobot(distance_cm=100)
        self.safety = SafetyController(
            self.robot, speed=25, allow_autonomous=allow_autonomous, **(safety_kwargs or {})
        )
        self.addCleanup(self.safety.close)
        self.safety.start()
        self.arbiter = Arbiter(self.safety, StubActions(), lease_timeout_s=lease_timeout_s)
        self.addCleanup(self.arbiter.close)
        camera = Camera(self.robot, min_interval_s=0, max_per_minute=0)
        self.scanner = Scanner(self.robot, camera, None, settle_s=0.01, observation_wait_s=0)
        self.policy = PolicySpy()
        self.explorer = Explorer(
            self.arbiter,
            self.safety,
            self.scanner,
            self.policy,
            config=config or settings(),
            on_scan=on_scan,
            drive=drive,
            head_only_interval_s=0.05,
            body_wait_s=10.0,
        )
        self.addCleanup(self.explorer.stop)
        return self.explorer

    def assert_stays_still(self, seconds=0.4):
        count = len(self.robot.trace)
        time.sleep(seconds)
        self.assertEqual(len(self.robot.trace), count)
        self.assertEqual(self.robot.speed, 0)


class LoopTests(ExploreCase):
    def test_it_moves_the_robot_and_scans_periodically(self):
        explorer = self.build(config=settings(scan_every_steps=2))
        self.assertTrue(explorer.start())
        self.assertTrue(explorer.active)
        self.assertTrue(wait_for(lambda: explorer.scans >= 2))
        explorer.stop()
        self.assertFalse(explorer.active)
        self.assertIn((15, 0), self.robot.trace)
        self.assertGreaterEqual(explorer.steps, 4)
        panned = {pose[0] for pose in self.robot.poses}
        self.assertTrue({-60, 0, 60} <= panned)  # the scan looked left, centre and right
        self.assertEqual(self.policy.calls[0], True)
        self.assertEqual(self.policy.calls[-1], False)
        self.assertEqual(explorer.end_reason, "stopped")
        self.assertIsNone(self.arbiter.holder)

    def test_it_ends_after_its_duration(self):
        explorer = self.build(config=settings(max_duration_s=0.6))
        explorer.start()
        self.assertTrue(wait_for(lambda: not explorer.active))
        self.assertEqual(explorer.end_reason, "duration")
        self.assertGreaterEqual(explorer.steps, 1)
        self.assertIsNone(self.arbiter.holder)

    def test_a_requested_duration_is_capped_by_the_configuration(self):
        explorer = self.build(config=settings(max_duration_s=0.6))
        explorer.start(duration_s=300)
        self.assertTrue(wait_for(lambda: not explorer.active, timeout=5))
        self.assertEqual(explorer.end_reason, "duration")

    def test_it_ends_after_its_step_limit(self):
        explorer = self.build(config=settings(max_steps=3))
        explorer.start()
        self.assertTrue(wait_for(lambda: not explorer.active))
        self.assertEqual((explorer.steps, explorer.end_reason), (3, "max_steps"))
        self.assertEqual(len(self.robot.forward_drives), 3)

    def test_it_refuses_to_start_twice_or_when_off(self):
        explorer = self.build()
        self.assertTrue(explorer.start())
        self.assertFalse(explorer.start())
        self.assertEqual(explorer.refusal, "already exploring")
        explorer.stop()
        off = Explorer(
            self.arbiter, self.safety, self.scanner, self.policy, config=settings(enabled=False)
        )
        self.assertFalse(off.start())
        self.assertIn("turned off", off.refusal)

    def test_it_does_not_start_while_something_stops_autonomy(self):
        explorer = self.build()
        self.safety.emergency_stop("test")
        self.assertFalse(explorer.start())
        self.assertIn("emergency_stop", explorer.refusal)

    def test_no_move_is_queued_and_each_one_is_fresh(self):
        explorer = self.build()
        running, peak, seen = [0], [0], []
        real = self.safety.execute
        lock = threading.Lock()

        def spy(command, wait=True, lease=None):
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
                seen.append((command.id, time.time() - command.created_at))
            try:
                return real(command, wait, lease)
            finally:
                with lock:
                    running[0] -= 1

        self.safety.execute = spy
        explorer.start()
        self.assertTrue(wait_for(lambda: len(seen) >= 6))
        explorer.stop()
        self.assertEqual(peak[0], 1)
        ids = [item[0] for item in seen]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(age < 0.5 for _, age in seen))
        self.assertTrue(all(r.reason in ("", "stopped") for r in explorer.results))

    def test_on_scan_receives_each_scan_and_can_stop_exploration(self):
        scans = []

        def on_scan(scan):
            scans.append(scan)
            return ExploreDecision("stop") if len(scans) == 2 else "continue"

        explorer = self.build(config=settings(scan_every_steps=1), on_scan=on_scan)
        explorer.start()
        self.assertTrue(wait_for(lambda: not explorer.active))
        self.assertEqual(len(scans), 2)
        self.assertEqual(explorer.end_reason, "decision")

    def test_a_failing_on_scan_does_not_end_exploration(self):
        calls = []

        def on_scan(scan):
            calls.append(scan)
            raise RuntimeError("boom")

        explorer = self.build(config=settings(scan_every_steps=1), on_scan=on_scan)
        explorer.start()
        self.assertTrue(wait_for(lambda: len(calls) >= 2))
        self.assertTrue(explorer.active)


class HazardTests(ExploreCase):
    def test_an_obstacle_makes_it_turn_away_and_carry_on(self):
        class Clears(TraceRobot):
            def drive(self, speed):
                super().drive(speed)
                if speed < 0:
                    self.distance = 100.0  # reversing away: the way is clear again

        explorer = self.build(Clears(distance_cm=10))
        explorer.start()
        self.assertTrue(wait_for(lambda: self.robot.forward_drives and explorer.avoided >= 1))
        first = explorer.results[0]
        self.assertEqual((first.result, first.reason), (ResultStatus.REJECTED, "obstacle"))
        reverse = [t for t in self.robot.trace if t[0] < 0]
        self.assertTrue(reverse and all(steering != 0 for _, steering in reverse))
        self.assertTrue(explorer.active)
        self.assertTrue(wait_for(lambda: explorer.scans >= 1))  # a turn is followed by a look

    def test_a_cliff_in_the_house_makes_it_back_off_turn_and_carry_on(self):
        house = default_house()
        robot = TraceRobot(house=house, start=Pose(405, 350, 0))  # facing the stairs
        explorer = self.build(robot, config=settings(step_ms=800, scan_every_steps=50))
        explorer.start()
        self.assertTrue(wait_for(lambda: explorer.avoided >= 1))
        self.assertTrue(wait_for(lambda: explorer.steps >= 3 or not explorer.active))
        self.assertTrue(explorer.active, explorer.end_reason)
        reasons = [r.reason for r in explorer.results]
        self.assertIn("cliff", reasons)
        reverse = [t for t in robot.trace if t[0] < 0]
        self.assertTrue(any(steering == 0 for _, steering in reverse))  # straight back first
        self.assertTrue(any(steering != 0 for _, steering in reverse))  # then turn away
        front = robot._front()
        self.assertFalse(house.is_over_cliff(*front))

    def test_a_cliff_that_never_goes_away_ends_it_and_it_never_drives_forward(self):
        robot = TraceRobot(distance_cm=100)
        robot.cliff = True
        explorer = self.build(robot)
        explorer.start()
        self.assertTrue(wait_for(lambda: not explorer.active))
        self.assertEqual(explorer.end_reason, "stuck")
        self.assertEqual(robot.forward_drives, [])

    def test_the_cliff_sensor_flickering_does_not_end_it(self):
        class CliffWhileRolling(TraceRobot):
            armed = True

            def get_sensors(self):
                sensors = super().get_sensors()
                if self.armed and self.speed > 0:
                    self.armed = False
                    return type(sensors)(sensors.distance_cm, True, sensors.grayscale)
                return sensors

        robot = CliffWhileRolling(distance_cm=100)
        explorer = self.build(robot)
        explorer.start()
        self.assertTrue(wait_for(lambda: explorer.avoided >= 1 and explorer.steps >= 3))
        self.assertTrue(explorer.active)
        self.assertIn("cliff", [r.reason for r in explorer.results])


class EndingTests(ExploreCase):
    def assert_ended_and_still(self, explorer, reason):
        self.assertTrue(wait_for(lambda: not explorer.active), "exploration did not end")
        self.assertIn(reason, explorer.end_reason)
        self.assertEqual(self.robot.speed, 0)
        self.assertIsNone(self.arbiter.holder)
        self.assertEqual(self.policy.calls[-1], False)
        self.assert_stays_still()

    def test_an_emergency_stop_ends_it(self):
        explorer = self.build(config=settings(step_ms=500))
        explorer.start()
        self.assertTrue(wait_for(lambda: self.robot.speed == 15))
        self.safety.emergency_stop("test")
        self.assert_ended_and_still(explorer, "emergency_stop")
        self.safety.reset_emergency_stop()
        self.assert_stays_still()  # it does not resume by itself

    def test_an_ai_outage_ends_it_and_recovery_does_not_restart_it(self):
        explorer = self.build(config=settings(step_ms=500))
        explorer.start()
        self.assertTrue(wait_for(lambda: self.robot.speed == 15))
        self.safety.set_ai_available(False)
        self.assert_ended_and_still(explorer, "ai_unavailable")
        self.safety.set_ai_available(True)
        self.assert_stays_still()
        self.assertFalse(explorer.active)

    def test_a_hardware_error_ends_it(self):
        explorer = self.build()
        self.robot.faults.drive_error = True
        explorer.start()
        self.assertTrue(wait_for(lambda: not explorer.active))
        self.assertEqual(explorer.end_reason, "hardware error")
        self.assertEqual(explorer.results[0].result, ResultStatus.FAILED)
        self.assertEqual(self.robot.trace, [])

    def test_a_stalled_safety_monitor_ends_it(self):
        class Stallable(TraceRobot):
            block = False

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.release = threading.Event()

            def get_sensors(self):
                if self.block and threading.current_thread().name == "laika-safety":
                    self.release.wait(10)
                return super().get_sensors()

        robot = Stallable(distance_cm=100)
        self.addCleanup(robot.release.set)
        explorer = self.build(
            robot,
            config=settings(step_ms=500),
            safety_kwargs=dict(watchdog_timeout_s=0.3, watchdog_period_s=0.05),
        )
        self.addCleanup(robot.release.set)
        explorer.start()
        self.assertTrue(wait_for(lambda: robot.speed == 15))
        robot.block = True
        self.assertTrue(wait_for(lambda: not explorer.active))
        self.assertEqual(explorer.end_reason, "watchdog")
        self.assertEqual(robot.speed, 0)
        count = len(robot.trace)
        time.sleep(0.4)
        self.assertEqual(len(robot.trace), count)

    def test_a_stalled_explorer_loses_its_lease_and_the_robot_stops(self):
        release = threading.Event()
        self.addCleanup(release.set)

        def on_scan(scan):
            release.wait(10)  # the explorer thread hangs here: no heartbeat
            return "continue"

        explorer = self.build(
            config=settings(scan_every_steps=1), on_scan=on_scan, lease_timeout_s=0.3
        )
        explorer.start()
        self.assertTrue(wait_for(lambda: explorer.scans >= 1))
        self.assertTrue(wait_for(lambda: self.arbiter.holder is None))
        self.assertEqual(self.robot.speed, 0)
        count = len(self.robot.trace)
        release.set()
        self.assertTrue(wait_for(lambda: not explorer.active))
        self.assertEqual(explorer.end_reason, "lease_expired")
        self.assert_stays_still()
        self.assertEqual(len(self.robot.trace), count)

    def test_a_long_scan_does_not_cost_it_the_lease(self):
        explorer = self.build(config=settings(scan_every_steps=1), lease_timeout_s=0.3)
        self.scanner.settle_s = 0.4  # a scan takes about 1.2 s, four times the lease timeout
        explorer.start()
        self.assertTrue(wait_for(lambda: explorer.scans >= 2, timeout=15))
        self.assertTrue(explorer.active)

    def test_a_camera_error_during_the_scan_does_not_hurt(self):
        explorer = self.build(config=settings(scan_every_steps=1))
        self.robot.faults.camera_error = True
        explorer.start()
        self.assertTrue(wait_for(lambda: explorer.scans >= 2))
        self.assertTrue(explorer.active)


class PreemptionTests(ExploreCase):
    def test_a_conversation_pauses_it_and_it_resumes_afterwards(self):
        explorer = self.build(config=settings(step_ms=1000))
        explorer.start()
        self.assertTrue(wait_for(lambda: self.robot.speed == 15))
        chat = self.arbiter.request("conversation", P.CONVERSATION)
        self.assertIsNotNone(chat)
        self.assertEqual(self.robot.speed, 0)
        self.assertTrue(wait_for(lambda: explorer.results))
        first = explorer.results[0]
        self.assertEqual((first.result, first.reason), (ResultStatus.CANCELLED, "preempted"))
        self.assertTrue(explorer.active)
        self.assertTrue(wait_for(lambda: self.policy.calls[-1:] == [False]))
        self.assert_stays_still(0.6)  # paused: no move while the conversation talks

        count = len(self.robot.trace)
        self.arbiter.release(chat)
        self.assertTrue(wait_for(lambda: len(self.robot.trace) > count))
        self.assertTrue(explorer.active)
        self.assertEqual(self.policy.calls[-1], True)

    def test_a_pause_does_not_use_up_the_duration(self):
        clock = FakeClock()
        explorer = self.build()
        explorer._clock = clock
        explorer.start(duration_s=10)
        self.assertTrue(wait_for(lambda: explorer.steps >= 1))
        chat = self.arbiter.request("conversation", P.CONVERSATION)
        self.assertTrue(wait_for(lambda: self.policy.calls[-1:] == [False]))  # it noticed
        clock.advance(1000)  # a long conversation
        time.sleep(0.4)
        self.assertTrue(explorer.active)
        self.arbiter.release(chat)
        count = len(self.robot.trace)
        self.assertTrue(wait_for(lambda: len(self.robot.trace) > count))
        self.assertTrue(explorer.active)
        clock.advance(11)
        self.assertTrue(wait_for(lambda: not explorer.active))
        self.assertEqual(explorer.end_reason, "duration")

    def test_started_during_a_conversation_it_waits_for_the_body(self):
        explorer = self.build()
        chat = self.arbiter.request("conversation", P.CONVERSATION)
        self.assertTrue(explorer.start())
        time.sleep(0.5)
        self.assertTrue(explorer.active)
        self.assertEqual(self.robot.trace, [])
        self.assertEqual(self.policy.calls, [])
        self.arbiter.release(chat)
        self.assertTrue(wait_for(lambda: self.robot.forward_drives))

    def test_a_safety_stop_while_waiting_for_the_body_ends_it(self):
        explorer = self.build()
        chat = self.arbiter.request("conversation", P.CONVERSATION)
        explorer.start()
        time.sleep(0.2)
        self.safety.emergency_stop("test")
        self.arbiter.release(chat)
        self.assertTrue(wait_for(lambda: not explorer.active))
        self.assertEqual(explorer.end_reason, "emergency_stop")
        self.assertEqual(self.robot.trace, [])

    def test_stop_while_paused_ends_it(self):
        explorer = self.build()
        explorer.start()
        self.assertTrue(wait_for(lambda: explorer.steps >= 1))
        chat = self.arbiter.request("conversation", P.CONVERSATION)
        time.sleep(0.3)
        explorer.stop()
        self.assertFalse(explorer.active)
        self.assertEqual(explorer.end_reason, "stopped")
        self.assertEqual(self.arbiter.waiting, [])
        self.arbiter.release(chat)

    def test_a_scan_in_progress_gives_the_head_back(self):
        explorer = self.build(config=settings(scan_every_steps=1))
        self.scanner.settle_s = 0.3
        explorer.start()
        self.assertTrue(wait_for(lambda: self.robot.poses))
        chat = self.arbiter.request("conversation", P.CONVERSATION)
        scans = explorer.scans
        time.sleep(1.5)
        self.assertEqual(explorer.scans, scans)  # the partial scan was thrown away
        self.arbiter.release(chat)
        explorer.stop()


class AutonomousGateTests(ExploreCase):
    def test_without_the_gate_it_only_looks_around(self):
        explorer = self.build(allow_autonomous=False)
        self.assertFalse(explorer.driving)
        explorer.start()
        self.assertEqual(explorer.mode, "head_only")
        self.assertTrue(wait_for(lambda: explorer.scans >= 2))
        explorer.stop()
        self.assertEqual(self.robot.speeds, [])  # no drive ever reached the robot
        self.assertEqual(self.robot.trace, [])
        self.assertTrue({-60, 60} <= {pose[0] for pose in self.robot.poses})

    def test_a_refused_autonomous_move_switches_to_head_only_without_retrying(self):
        explorer = self.build(allow_autonomous=False, drive=True)
        explorer.start()
        self.assertTrue(wait_for(lambda: explorer.scans >= 2))
        explorer.stop()
        reasons = [(r.result, r.reason) for r in explorer.results]
        self.assertEqual(reasons, [(ResultStatus.REJECTED, "autonomous_disabled")])
        self.assertEqual(explorer.mode, "head_only")
        self.assertEqual(self.robot.speeds, [])

    def test_no_driving_keeps_it_head_only_even_with_the_gate_open(self):
        explorer = self.build(allow_autonomous=True, safety_kwargs=dict(enabled=False))
        explorer.start()
        self.assertEqual(explorer.mode, "head_only")
        self.assertTrue(wait_for(lambda: explorer.scans >= 1))
        explorer.stop()
        self.assertEqual(self.robot.speeds, [])


class SafetyGateTests(unittest.TestCase):
    def controller(self, robot, **kwargs):
        controller = SafetyController(robot, speed=25, **kwargs)
        self.addCleanup(controller.close)
        return controller

    def test_autonomous_leases_are_rejected_by_default(self):
        robot = TraceRobot(distance_cm=100)
        controller = self.controller(robot)
        arbiter = Arbiter(controller, StubActions())
        self.addCleanup(arbiter.close)
        for priority in (P.CURIOSITY, P.SOCIAL, P.EXPLORATION):
            lease = arbiter.request("x", priority)
            result = controller.execute(move(), lease=lease)
            self.assertEqual((result.result, result.reason), (ResultStatus.REJECTED, "autonomous_disabled"))
            lease.release()
        self.assertEqual(robot.speeds, [])

    def test_conversation_user_and_leaseless_moves_are_not_affected(self):
        robot = TraceRobot(distance_cm=100)
        controller = self.controller(robot)
        arbiter = Arbiter(controller, StubActions())
        self.addCleanup(arbiter.close)
        for priority in (P.USER_COMMAND, P.CONVERSATION, P.FOLLOW_UP, P.IMPORTANT_OBSERVATION):
            lease = arbiter.request("x", priority)
            self.assertEqual(controller.execute(move(100), lease=lease).result, ResultStatus.SUCCESS)
            lease.release()
        self.assertEqual(controller.execute(move(100)).result, ResultStatus.SUCCESS)

    def test_the_gate_can_be_opened(self):
        robot = TraceRobot(distance_cm=100)
        controller = self.controller(robot, allow_autonomous=True)
        arbiter = Arbiter(controller, StubActions())
        self.addCleanup(arbiter.close)
        lease = arbiter.request("x", P.EXPLORATION)
        self.assertEqual(controller.execute(move(100), lease=lease).result, ResultStatus.SUCCESS)

    def test_the_default_configuration_keeps_it_closed(self):
        self.assertFalse(Config().robot.allow_autonomous_driving)
        self.assertFalse(SafetyController(TraceRobot(distance_cm=100)).allow_autonomous)

    def test_picarx_with_the_default_configuration_rejects_an_exploration_move(self):
        with mock.patch.dict(sys.modules, fake_sunfounder_modules()):
            from laika.hardware import PiCarXRobot

            robot = PiCarXRobot(mouth=SpyMouth())
            self.addCleanup(robot.close)
            config = Config()
            controller = self.controller(
                robot,
                enabled=config.robot.allow_driving,
                allow_autonomous=config.robot.allow_autonomous_driving,
            )
            arbiter = Arbiter(controller, StubActions())
            self.addCleanup(arbiter.close)
            lease = arbiter.request("exploration", P.EXPLORATION)
            result = controller.execute(move(), lease=lease)
            self.assertEqual((result.result, result.reason), (ResultStatus.REJECTED, "autonomous_disabled"))
            lease.release()
            self.assertFalse([c for c in robot._car.calls if c[0] in ("forward", "backward")])

            # An exploration that runs on it looks around only and never drives.
            camera = Camera(robot, min_interval_s=0, max_per_minute=0)
            scanner = Scanner(robot, camera, None, settle_s=0.01, observation_wait_s=0)
            explorer = Explorer(
                arbiter, controller, scanner, PolicySpy(), config=settings(), head_only_interval_s=0.05
            )
            self.addCleanup(explorer.stop)
            explorer.start()
            self.assertEqual(explorer.mode, "head_only")
            self.assertTrue(wait_for(lambda: explorer.scans >= 1))
            explorer.stop()
            self.assertFalse([c for c in robot._car.calls if c[0] in ("forward", "backward")])

            # A conversation move on the same controller still works.
            chat = arbiter.request("conversation", P.CONVERSATION)
            self.assertEqual(controller.execute(move(100), lease=chat).result, ResultStatus.SUCCESS)


class ToolTests(ExploreCase):
    def call(self, name, **arguments):
        return self.registry.run(ToolCall("c1", name, arguments), self.turn)

    def setUp(self):
        self.turn = None

    def make_registry(self, **kwargs):
        explorer = self.build(**kwargs)
        self.registry = ToolRegistry(
            self.robot, self.safety, explorer=explorer, scan_settle_s=0.01, observation_wait_s=0
        )
        return explorer

    def test_the_tools_are_offered_only_with_an_explorer(self):
        self.make_registry()
        self.assertIn("start_exploring", self.registry.names())
        self.assertIn("stop_exploring", self.registry.names())
        plain = ToolRegistry(self.robot, self.safety)
        self.assertNotIn("start_exploring", plain.names())
        off = ToolRegistry(
            self.robot,
            self.safety,
            explorer=Explorer(
                self.arbiter, self.safety, self.scanner, self.policy, config=settings(enabled=False)
            ),
        )
        self.assertNotIn("start_exploring", off.names())

    def test_start_and_stop_exploring(self):
        explorer = self.make_registry()
        result = self.call("start_exploring", duration_s=60)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["mode"], "driving")
        self.assertTrue(wait_for(lambda: self.robot.forward_drives))
        again = self.call("start_exploring")
        self.assertFalse(again["ok"])
        self.assertIn("already exploring", again["error"])
        stopped = self.call("stop_exploring")
        self.assertEqual(stopped, {"ok": True, "was_exploring": True})
        self.assertFalse(explorer.active)
        self.assertEqual(self.robot.speed, 0)
        self.assertEqual(self.call("stop_exploring")["was_exploring"], False)

    def test_the_duration_is_validated(self):
        self.make_registry()
        for bad in (5, 301, "soon", 12.5):
            self.assertFalse(self.call("start_exploring", duration_s=bad)["ok"], bad)
        self.assertFalse(self.registry.run(ToolCall("c", "start_exploring", {"speed": 40}))["ok"])
        self.assertFalse(self.explorer.active)

    def test_started_from_a_conversation_it_moves_once_the_conversation_ends(self):
        explorer = self.make_registry()
        chat = self.arbiter.request("conversation", P.CONVERSATION)
        self.turn = Turn(lease=chat)
        result = self.call("start_exploring")
        self.assertTrue(result["ok"], result)
        time.sleep(0.5)
        self.assertTrue(explorer.active)
        self.assertEqual(self.robot.trace, [])  # the agent's turn still has the body
        self.arbiter.release(chat)
        self.assertTrue(wait_for(lambda: self.robot.forward_drives))

    def test_the_result_says_when_she_can_only_look_around(self):
        self.make_registry(allow_autonomous=False)
        result = self.call("start_exploring")
        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "head_only")
        self.assertIn("wheels stay still", result["note"])

    def test_a_refusal_is_reported_to_the_model(self):
        self.make_registry()
        self.safety.emergency_stop("test")
        result = self.call("start_exploring")
        self.assertFalse(result["ok"])
        self.assertIn("emergency_stop", result["error"])

    def test_the_prompt_mentions_the_tools_only_when_offered(self):
        from laika.actions import ACTIONS

        with_tools = build_tool_system_prompt(ACTIONS, 3, True, exploring=True)
        without = build_tool_system_prompt(ACTIONS, 3, True)
        self.assertIn("start_exploring", with_tools)
        self.assertIn("stop_exploring", with_tools)
        self.assertNotIn("start_exploring", without)
        self.assertNotIn("{", with_tools.replace('{"', ""))  # no unfilled placeholder


class IdleTests(ExploreCase):
    def test_it_starts_after_the_idle_time_and_not_before(self):
        clock = FakeClock()
        explorer = self.build(drive=False)
        starter = IdleStarter(explorer, self.arbiter, idle_after_s=60, clock=clock)
        self.assertFalse(starter.poll())
        clock.advance(59)
        self.assertFalse(starter.poll())
        self.assertFalse(explorer.active)
        clock.advance(2)
        self.assertTrue(starter.poll())
        self.assertTrue(wait_for(lambda: explorer.scans >= 1))
        # While exploring it counts as busy: the timer restarts, nothing starts twice.
        clock.advance(500)
        self.assertFalse(starter.poll())
        explorer.stop()
        clock.advance(30)
        self.assertFalse(starter.poll())
        clock.advance(31)
        self.assertTrue(starter.poll())

    def test_a_conversation_keeps_it_from_starting(self):
        clock = FakeClock()
        explorer = self.build(drive=False)
        starter = IdleStarter(explorer, self.arbiter, idle_after_s=60, clock=clock)
        chat = self.arbiter.request("conversation", P.CONVERSATION)
        clock.advance(500)
        self.assertFalse(starter.poll())
        self.arbiter.release(chat)
        clock.advance(59)
        self.assertFalse(starter.poll())
        clock.advance(2)
        self.assertTrue(starter.poll())

    def test_zero_means_never(self):
        clock = FakeClock()
        explorer = self.build(drive=False)
        starter = IdleStarter(explorer, self.arbiter, idle_after_s=0, clock=clock)
        clock.advance(10**6)
        self.assertFalse(starter.poll())
        starter.start()
        self.assertIsNone(starter._thread)

    def test_a_refused_start_is_retried_after_another_idle_period(self):
        clock = FakeClock()
        explorer = self.build(drive=False)
        starter = IdleStarter(explorer, self.arbiter, idle_after_s=60, clock=clock)
        self.safety.emergency_stop("test")
        clock.advance(61)
        self.assertFalse(starter.poll())
        self.safety.reset_emergency_stop()
        clock.advance(30)
        self.assertFalse(starter.poll())
        clock.advance(31)
        self.assertTrue(starter.poll())


if __name__ == "__main__":
    unittest.main()
