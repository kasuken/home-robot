"""Know the battery level (#50): the reading, the simulator model, the monitor, and what a low battery does."""

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from unittest import mock

from laika.agent import Agent
from laika.battery import MAX_SKIPS, BatteryMonitor, BatteryState
from laika.behaviour import Arbiter, Priority
from laika.commands import Direction, MoveCommand, ResultStatus, Telemetry
from laika.config import BatteryConfig, load_config, validate_battery
from laika.events import EventBus, EventType
from laika.explore import Explorer
from laika.house import Pose, default_house
from laika.interaction_policy import InteractionPolicy
from laika.camera import Camera
from laika.config import InteractionConfig
from laika.prompts import battery_prompt
from laika.robot import ALL_CAPABILITIES, BATTERY
from laika.safety import SafetyController
from laika.scan import Scanner
from laika.simulator import (
    BATTERY_DRIVE_V_PER_S,
    BATTERY_IDLE_V_PER_S,
    BATTERY_SAG_V,
    BATTERY_START_V,
    SimulatedRobot,
)
from laika.tools import ToolRegistry
from tests.agent.test_agent import ScriptedClient, text
from tests.agent.test_tools import call
from tests.behaviour.test_explore import PolicySpy, StubActions, TraceRobot, move, settings, wait_for
from tests.fakes import FakeClock
from tests.interaction.test_interaction_policy import NEUTRAL, FakePresence, at
from tests.simulator.test_robot_contract import SpyMouth, fake_sunfounder_modules


def config(**overrides):
    return BatteryConfig(**overrides)


class FakeRobot:
    """Only what the monitor reads: a voltage that a test sets, and whether the wheels turn."""

    def __init__(self, volts=8.0):
        self.volts = volts
        self.speed = 0
        self.reads = 0

    def battery_voltage(self):
        self.reads += 1
        if isinstance(self.volts, Exception):
            raise self.volts
        return self.volts

    def get_state(self):
        return types.SimpleNamespace(speed=self.speed)


class MonitorCase(unittest.TestCase):
    def build(self, volts=8.0, cfg=None, **kwargs):
        self.robot = FakeRobot(volts)
        self.clock = FakeClock(1000.0)
        self.bus = EventBus()
        self.bus.start()
        self.addCleanup(self.bus.close)
        self.seen = []
        self.bus.subscribe(self.seen.append)
        self.monitor = BatteryMonitor(self.robot, self.bus, cfg or config(), self.clock, **kwargs)
        self.changes = []
        self.monitor.add_listener(lambda new, old: self.changes.append((new, old)))
        return self.monitor

    def feed(self, *volts, step_s=30.0):
        """One counted reading per value."""
        for value in volts:
            self.robot.volts = value
            self.clock.advance(step_s)
            self.assertIsNotNone(self.monitor.sample())

    def battery_events(self):
        self.bus.drain()
        names = {EventType.BATTERY_LOW, EventType.BATTERY_CRITICAL, EventType.BATTERY_OK}
        return [e.type for e in self.seen if e.type in names]


class PiCarXBatteryTests(unittest.TestCase):
    def setUp(self):
        self.modules = fake_sunfounder_modules()
        self.volts = 7.9
        self.calls = 0

        def get_battery_voltage():
            self.calls += 1
            if isinstance(self.volts, Exception):
                raise self.volts
            return self.volts

        self.get = get_battery_voltage
        self.modules["robot_hat.device"] = types.SimpleNamespace(get_battery_voltage=get_battery_voltage)
        patcher = mock.patch.dict(sys.modules, self.modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        from laika.hardware import PiCarXRobot

        self.robot = PiCarXRobot(mouth=SpyMouth())
        self.addCleanup(self.robot.close)

    def test_the_voltage_comes_from_the_robot_hat(self):
        self.assertEqual(self.robot.battery_voltage(), 7.9)
        self.assertEqual(self.calls, 1)

    def test_the_state_carries_the_battery_and_so_does_the_telemetry(self):
        state = self.robot.get_state()
        self.assertEqual(state.battery_v, 7.9)
        self.assertEqual(state.to_telemetry().battery_v, 7.9)

    def test_the_sensors_do_not_read_the_adc(self):
        self.robot.get_sensors()
        self.assertEqual(self.calls, 0)

    def test_a_failed_read_is_none_and_logged_once(self):
        self.volts = OSError("i2c busy")
        with self.assertLogs("laika.hardware", level="ERROR") as logged:
            self.assertIsNone(self.robot.battery_voltage())
            self.assertIsNone(self.robot.battery_voltage())
        self.assertEqual(len(logged.records), 1)
        self.assertIn("i2c busy", logged.output[0])
        self.assertIsNone(self.robot.get_state().battery_v)  # still no crash

    def test_it_is_logged_again_after_a_read_that_worked(self):
        self.volts = OSError("i2c busy")
        with self.assertLogs("laika.hardware", level="ERROR"):
            self.robot.battery_voltage()
        self.volts = 7.5
        self.assertEqual(self.robot.battery_voltage(), 7.5)
        self.volts = OSError("again")
        with self.assertLogs("laika.hardware", level="ERROR") as logged:
            self.robot.battery_voltage()
        self.assertEqual(len(logged.records), 1)

    def test_an_older_robot_hat_without_the_device_module_uses_utils(self):
        del sys.modules["robot_hat.device"]
        sys.modules["robot_hat.utils"] = types.SimpleNamespace(get_battery_voltage=lambda: 7.7)
        self.assertEqual(self.robot.battery_voltage(), 7.7)

    def test_the_robot_advertises_the_capability(self):
        self.assertIn(BATTERY, self.robot.identity().capabilities)


class SimulatedBatteryTests(unittest.TestCase):
    def build(self, **kwargs):
        self.clock = FakeClock(100.0)
        return SimulatedRobot(verbose=False, clock=self.clock, **kwargs)

    def test_it_starts_full_enough(self):
        self.assertEqual(self.build().battery_voltage(), BATTERY_START_V)

    def test_it_drains_slowly_with_time(self):
        robot = self.build()
        self.clock.advance(3600)
        expected = BATTERY_START_V - 3600 * BATTERY_IDLE_V_PER_S
        self.assertAlmostEqual(robot.battery_voltage(), expected, places=2)
        self.assertGreater(robot.battery_voltage(), 7.0)

    def test_it_drains_faster_while_driving(self):
        idle, busy = self.build(), self.build()
        busy.drive(25)
        self.clock.advance(600)
        busy.stop()
        self.assertLess(busy.battery_voltage(), idle.battery_voltage())
        lost = BATTERY_START_V - busy.battery_voltage()
        self.assertAlmostEqual(lost, 600 * BATTERY_DRIVE_V_PER_S, places=2)

    def test_the_voltage_sags_while_the_motors_run(self):
        robot = self.build()
        rest = robot.battery_voltage()
        robot.drive(25)
        self.assertAlmostEqual(robot.battery_voltage(), rest - BATTERY_SAG_V, places=2)
        robot.stop()
        self.assertAlmostEqual(robot.battery_voltage(), rest, places=2)

    def test_set_battery_for_tests(self):
        robot = self.build()
        robot.set_battery(6.5)
        self.assertEqual(robot.battery_voltage(), 6.5)
        self.assertEqual(robot.get_state().battery_v, 6.5)

    def test_no_battery_capability_means_no_reading(self):
        robot = self.build(capabilities=ALL_CAPABILITIES - {BATTERY})
        self.assertIsNone(robot.battery_voltage())
        self.assertIsNone(robot.get_state().battery_v)

    def test_a_fault_means_no_reading(self):
        robot = self.build()
        robot.faults.battery_error = True
        self.assertIsNone(robot.battery_voltage())


class TelemetryTests(unittest.TestCase):
    base = dict(distance_cm=40.0, cliff_detected=False, camera_pan=0, camera_tilt=0)

    def test_the_battery_round_trips_as_battery_v(self):
        telemetry = Telemetry(**self.base, battery_v=7.62)
        data = json.loads(telemetry.to_json())
        self.assertEqual(data["batteryV"], 7.62)
        self.assertEqual(Telemetry.from_json(telemetry.to_json()), telemetry)

    def test_without_a_battery_the_field_is_absent(self):
        telemetry = Telemetry(**self.base)
        self.assertNotIn("batteryV", json.loads(telemetry.to_json()))
        self.assertIsNone(Telemetry.from_json(telemetry.to_json()).battery_v)

    def test_old_telemetry_json_still_parses(self):
        old = {
            "type": "telemetry",
            "distanceCm": 12.5,
            "cliffDetected": False,
            "camera": {"pan": 5, "tilt": 0},
            "steering": 0,
            "speed": 0,
            "timestamp": 1.0,
        }
        parsed = Telemetry.from_dict(old)
        self.assertIsNone(parsed.battery_v)
        self.assertEqual(parsed.distance_cm, 12.5)

    def test_a_bad_voltage_is_rejected(self):
        from laika.commands import InvalidCommand

        with self.assertRaises(InvalidCommand):
            Telemetry(**self.base, battery_v="high")


class SmoothingAndHysteresisTests(MonitorCase):
    def test_nothing_happens_before_three_readings(self):
        self.build()
        self.feed(6.0, 6.0)
        self.assertIs(self.monitor.state, BatteryState.UNKNOWN)
        self.feed(6.0)
        self.assertIs(self.monitor.state, BatteryState.CRITICAL)

    def test_the_voltage_is_the_median_of_the_last_five(self):
        self.build()
        self.feed(7.6, 7.6, 7.6, 7.6, 7.6)
        self.feed(5.5)  # a glitch
        self.assertEqual(self.monitor.voltage, 7.6)
        self.assertIs(self.monitor.state, BatteryState.OK)

    def test_it_goes_low_below_the_threshold_and_only_once(self):
        self.build()
        self.feed(7.5, 7.5, 7.5)
        self.feed(6.9, 6.9, 6.9)
        self.assertIs(self.monitor.state, BatteryState.LOW)
        self.assertEqual(self.changes, [(BatteryState.OK, BatteryState.UNKNOWN), (BatteryState.LOW, BatteryState.OK)])

    def test_a_voltage_hovering_around_the_threshold_does_not_flap(self):
        self.build()
        self.feed(7.5, 7.5, 7.5, 6.95, 6.95, 6.95)
        self.assertIs(self.monitor.state, BatteryState.LOW)
        for value in (7.05, 6.95, 7.1, 6.98, 7.2, 7.0, 7.25, 6.99, 7.29, 7.0):
            self.feed(value)
        self.assertIs(self.monitor.state, BatteryState.LOW)
        self.assertEqual(self.changes, [(BatteryState.OK, BatteryState.UNKNOWN), (BatteryState.LOW, BatteryState.OK)])

    def test_it_is_ok_again_only_above_ok_v(self):
        self.build()
        self.feed(6.9, 6.9, 6.9)
        self.assertIs(self.monitor.state, BatteryState.LOW)
        self.feed(7.3, 7.3, 7.3)  # exactly ok_v is not above it
        self.assertIs(self.monitor.state, BatteryState.LOW)
        self.feed(7.4, 7.4, 7.4)
        self.assertIs(self.monitor.state, BatteryState.OK)

    def test_critical_below_critical_v_and_back_to_low(self):
        self.build()
        self.feed(6.5, 6.5, 6.5)
        self.assertIs(self.monitor.state, BatteryState.CRITICAL)
        self.feed(6.8, 6.8, 6.8)  # above critical but still below low_v
        self.assertIs(self.monitor.state, BatteryState.CRITICAL)
        self.feed(7.1, 7.1, 7.1)
        self.assertIs(self.monitor.state, BatteryState.LOW)
        self.feed(7.5, 7.5, 7.5)
        self.assertIs(self.monitor.state, BatteryState.OK)

    def test_the_offset_is_added_to_every_reading(self):
        self.build(cfg=config(voltage_offset=0.2))
        self.feed(6.9, 6.9, 6.9)  # 7.1 V after the offset: not low
        self.assertIs(self.monitor.state, BatteryState.OK)
        self.assertAlmostEqual(self.monitor.voltage, 7.1)

    def test_percent_is_a_rough_line_between_critical_and_full(self):
        self.build()
        self.assertIsNone(self.monitor.percent)
        self.feed(8.4, 8.4, 8.4)
        self.assertEqual(self.monitor.percent, 100)
        self.feed(6.6, 6.6, 6.6, 6.6)
        self.assertEqual(self.monitor.percent, 0)
        self.feed(7.5, 7.5, 7.5, 7.5, 7.5)
        self.assertEqual(self.monitor.percent, 50)

    def test_lowest_voltage_is_kept(self):
        self.build()
        self.feed(7.5, 7.4, 7.3, 7.2, 7.1, 7.0, 7.6, 7.6, 7.6, 7.6, 7.6)
        self.assertAlmostEqual(self.monitor.lowest_v, 7.2)


class SamplingTests(MonitorCase):
    def test_it_samples_on_its_own_timer(self):
        self.build()
        self.assertTrue(self.monitor.poll())  # the first one is due at once
        self.assertFalse(self.monitor.poll())
        self.clock.advance(1)
        self.assertFalse(self.monitor.poll())  # warming up: every two seconds
        self.clock.advance(2)
        self.assertTrue(self.monitor.poll())
        self.clock.advance(2)
        self.assertTrue(self.monitor.poll())
        self.clock.advance(10)
        self.assertFalse(self.monitor.poll())  # steady: every sample_s (30 s)
        self.clock.advance(25)
        self.assertTrue(self.monitor.poll())

    def test_a_reading_taken_while_driving_is_skipped(self):
        self.build(is_driving=lambda: self.robot.speed != 0)
        self.robot.speed = 25
        self.robot.volts = 6.0  # the sag under load
        self.assertIsNone(self.monitor.sample())
        self.assertIsNone(self.monitor.voltage)
        self.robot.speed = 0
        self.robot.volts = 7.6
        self.clock.advance(2)
        self.assertEqual(self.monitor.sample(), 7.6)

    def test_a_loaded_reading_counts_after_a_long_drive(self):
        self.build(is_driving=lambda: True)
        for _ in range(MAX_SKIPS):
            self.assertIsNone(self.monitor.sample())
        self.assertIsNotNone(self.monitor.sample())

    def test_the_default_checks_the_robot_state(self):
        self.build()
        self.robot.speed = 20
        self.assertIsNone(self.monitor.sample())

    def test_a_failed_read_is_ignored_and_logged_once(self):
        self.build(OSError("adc"))
        with self.assertLogs("laika.battery", level="WARNING") as logged:
            self.assertIsNone(self.monitor.sample())
            self.assertIsNone(self.monitor.sample())
        self.assertEqual(len(logged.records), 1)
        self.assertIs(self.monitor.state, BatteryState.UNKNOWN)  # never read: not known to be fine
        self.assertIsNone(self.monitor.voltage)

    def test_no_reading_and_absurd_readings_are_not_a_battery(self):
        self.build(None)
        with self.assertLogs("laika.battery", level="WARNING"):
            self.assertIsNone(self.monitor.sample())
        for absurd in (0.0, 0.4, 25.0):
            self.robot.volts = absurd
            self.assertIsNone(self.monitor.sample())
        self.assertIsNone(self.monitor.voltage)
        self.assertIs(self.monitor.state, BatteryState.UNKNOWN)  # an unplugged pack is not "critical"

    def test_the_thread_starts_and_stops(self):
        self.build()
        self.monitor.start()
        self.assertTrue(wait_for(lambda: self.monitor.voltage is not None, 3.0))
        self.monitor.close()
        self.monitor.close()



class UnknownStateTests(MonitorCase):
    def test_it_starts_unknown_and_becomes_ok_quietly(self):
        self.build()
        self.assertIs(self.monitor.state, BatteryState.UNKNOWN)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.feed(7.6, 7.6, 7.6)
        self.assertIs(self.monitor.state, BatteryState.OK)
        self.assertEqual(out.getvalue(), "")  # the routine first reading tells nobody

    def test_readings_that_stop_working_make_it_unknown(self):
        self.build()
        self.feed(7.6, 7.6, 7.6)
        self.robot.volts = None
        with self.assertLogs("laika.battery", level="WARNING"):
            for _ in range(3):
                self.monitor.sample()
        self.assertIs(self.monitor.state, BatteryState.UNKNOWN)
        self.assertFalse(self.monitor.low)  # nothing to ask anyone to charge

    def test_prime_knows_the_state_at_startup(self):
        self.build()
        self.robot.volts = 7.6
        self.assertIs(self.monitor.prime(), BatteryState.OK)

    def test_unknown_refuses_autonomous_driving_but_not_the_user(self):
        from laika.behaviour import Arbiter, Priority
        from laika.commands import Direction, MoveCommand
        from laika.safety import SafetyController
        from tests.fakes import RecordingRobot

        robot = RecordingRobot(distance_cm=100)
        safety = SafetyController(robot, allow_autonomous=True)
        self.addCleanup(safety.close)
        arbiter = Arbiter(safety, None)
        self.addCleanup(arbiter.close)
        safety.set_battery_state(BatteryState.UNKNOWN)
        lease = arbiter.request("explore", Priority.EXPLORATION)
        move = MoveCommand(direction=Direction.FORWARD, speed=20, duration_ms=100)
        self.assertIsNotNone(lease)
        self.assertEqual(safety.execute(move, lease=lease).reason, "battery_unknown")
        arbiter.release(lease)
        user = arbiter.request("chat", Priority.CONVERSATION)
        self.assertEqual(safety.execute(MoveCommand(direction=Direction.FORWARD, speed=20, duration_ms=100), lease=user).result.value, "success")

class EventTests(MonitorCase):
    def test_transitions_publish_events_with_the_voltage(self):
        self.build()
        self.feed(7.5, 7.5, 7.5)
        self.feed(6.9, 6.9, 6.9)
        self.feed(6.4, 6.4, 6.4)
        self.feed(7.5, 7.5, 7.5, 7.5, 7.5)
        self.assertEqual(
            self.battery_events(),
            [EventType.BATTERY_LOW, EventType.BATTERY_CRITICAL, EventType.BATTERY_OK],
        )
        low = next(e for e in self.seen if e.type is EventType.BATTERY_LOW)
        self.assertEqual(low.data["previous"], "ok")
        self.assertAlmostEqual(low.data["voltage"], 6.9)
        self.assertIsInstance(low.data["percent"], int)
        ok = next(e for e in self.seen if e.type is EventType.BATTERY_OK)
        self.assertAlmostEqual(ok.data["lowestV"], 6.4)

    def test_no_event_without_a_transition(self):
        self.build()
        self.feed(7.6, 7.5, 7.7, 7.6, 7.5)
        self.assertEqual(self.battery_events(), [])

    def test_a_failing_listener_does_not_stop_the_monitor(self):
        self.build()
        self.monitor.add_listener(lambda new, old: 1 / 0)
        with self.assertLogs("laika.battery", level="ERROR"):
            self.feed(6.0, 6.0, 6.0)
        self.assertIs(self.monitor.state, BatteryState.CRITICAL)


class SafetyEffectTests(unittest.TestCase):
    def setUp(self):
        self.robot = SimulatedRobot(verbose=False, house=default_house(), start=Pose(300, 200, 0))
        self.safety = SafetyController(self.robot, speed=25, allow_autonomous=True)
        self.addCleanup(self.safety.close)
        self.safety.start()
        self.arbiter = Arbiter(self.safety, StubActions())
        self.addCleanup(self.arbiter.close)
        self.halts = []
        self.safety.add_halt_listener(self.halts.append)

    def run_move(self, lease=None):
        return self.safety.execute(move(150), lease=lease)

    def test_a_low_battery_refuses_autonomous_moves_only(self):
        self.safety.set_battery_state(BatteryState.LOW)
        lease = self.arbiter.request("exploration", Priority.EXPLORATION)
        # the transition ended any autonomous lease; a new one is still granted (head-only work goes on)
        self.assertIsNotNone(lease)
        result = self.run_move(lease)
        self.assertEqual((result.result, result.reason), (ResultStatus.REJECTED, "battery_low"))
        self.assertEqual(self.run_move().result, ResultStatus.SUCCESS)  # the user's move
        conversation = self.arbiter.request("chat", Priority.CONVERSATION)
        self.assertEqual(self.run_move(conversation).result, ResultStatus.SUCCESS)

    def test_going_low_ends_autonomous_leases_like_other_halts(self):
        lease = self.arbiter.request("exploration", Priority.EXPLORATION)
        self.assertTrue(lease.active)
        self.safety.set_battery_state(BatteryState.LOW)
        self.assertFalse(lease.active)
        self.assertEqual(lease.end_reason, "battery_low")
        self.assertEqual(self.halts, ["battery_low"])

    def test_a_conversation_lease_survives_going_low(self):
        lease = self.arbiter.request("chat", Priority.CONVERSATION)
        self.safety.set_battery_state(BatteryState.LOW)
        self.assertTrue(lease.active)

    def test_critical_refuses_every_move_but_never_stop(self):
        self.safety.set_battery_state(BatteryState.CRITICAL)
        for lease in (None, self.arbiter.request("chat", Priority.CONVERSATION)):
            result = self.run_move(lease)
            self.assertEqual((result.result, result.reason), (ResultStatus.REJECTED, "battery_critical"))
        self.safety.stop()
        self.safety.emergency_stop("test")
        self.safety.reset_emergency_stop()
        self.assertEqual(self.robot.speed, 0)

    def test_critical_cancels_the_move_in_progress(self):
        long_move = MoveCommand(direction=Direction.FORWARD, speed=25, duration_ms=1500)
        self.assertIsNone(self.safety.execute(long_move, wait=False))
        self.safety.set_battery_state(BatteryState.CRITICAL)
        self.assertEqual(self.robot.speed, 0)

    def test_critical_refuses_new_autonomous_requests(self):
        self.safety.set_battery_state(BatteryState.CRITICAL)
        self.assertIsNone(self.arbiter.request("exploration", Priority.EXPLORATION))
        self.assertIsNotNone(self.arbiter.request("chat", Priority.CONVERSATION))

    def test_recovery_lifts_the_refusals(self):
        self.safety.set_battery_state(BatteryState.CRITICAL)
        self.safety.set_battery_state(BatteryState.OK)
        lease = self.arbiter.request("exploration", Priority.EXPLORATION)
        self.assertEqual(self.run_move(lease).result, ResultStatus.SUCCESS)

    def test_critical_after_low_notifies_again(self):
        self.safety.set_battery_state(BatteryState.LOW)
        self.safety.set_battery_state(BatteryState.CRITICAL)
        self.assertEqual(self.halts, ["battery_low", "battery_critical"])
        self.safety.set_battery_state(BatteryState.LOW)  # recovering is not a halt
        self.assertEqual(self.halts, ["battery_low", "battery_critical"])


class DrainingDuringExplorationTests(unittest.TestCase):
    """Done when: in the simulator a draining battery stops idle driving at the low threshold."""

    def test_exploration_stops_driving_at_the_low_threshold(self):
        robot = TraceRobot(distance_cm=100)
        safety = SafetyController(robot, speed=25, allow_autonomous=True)
        self.addCleanup(safety.close)
        safety.start()
        arbiter = Arbiter(safety, StubActions())
        self.addCleanup(arbiter.close)
        clock = FakeClock(1000.0)
        monitor = BatteryMonitor(robot, None, config(), clock, is_driving=lambda: False)
        monitor.add_listener(lambda new, old: safety.set_battery_state(new))
        scanner = Scanner(robot, Camera(robot, min_interval_s=0, max_per_minute=0), None, settle_s=0.01, observation_wait_s=0)
        explorer = Explorer(
            arbiter, safety, scanner, PolicySpy(), config=settings(), head_only_interval_s=0.05, body_wait_s=10.0
        )
        self.addCleanup(explorer.stop)

        robot.set_battery(7.6)
        for _ in range(3):
            clock.advance(30)
            monitor.sample()
        self.assertIs(monitor.state, BatteryState.OK)
        self.assertTrue(explorer.start())
        self.assertTrue(wait_for(lambda: len(robot.forward_drives) >= 2))
        lease = arbiter.holder
        self.assertIsNotNone(lease)

        # The battery drains below the low threshold while she explores.
        robot.set_battery(6.9)
        for _ in range(3):
            clock.advance(30)
            monitor.sample()
        self.assertIs(monitor.state, BatteryState.LOW)

        self.assertFalse(lease.active)  # the exploration lease was revoked
        self.assertEqual(lease.end_reason, "battery_low")
        self.assertTrue(wait_for(lambda: not explorer.active))
        # The next autonomous move is refused for the battery ...
        again = arbiter.request("idle", Priority.EXPLORATION)
        refused = safety.execute(move(), lease=again)
        self.assertEqual((refused.result, refused.reason), (ResultStatus.REJECTED, "battery_low"))
        arbiter.release(again)
        # ... a user's move still runs ...
        self.assertEqual(safety.execute(move()).result, ResultStatus.SUCCESS)
        # ... and an idle exploration from now on only looks around.
        robot.trace.clear()
        self.assertFalse(explorer.driving)
        self.assertEqual(explorer.mode, "head_only")
        self.assertTrue(explorer.start())
        self.assertTrue(wait_for(lambda: explorer.scans >= 1))
        explorer.stop()
        self.assertEqual(robot.forward_drives, [])


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.robot = SimulatedRobot(verbose=False, house=default_house(), start=Pose(300, 200, 0))
        self.safety = SafetyController(self.robot, speed=25)
        self.addCleanup(self.safety.close)
        self.safety.start()
        self.clock = FakeClock(1000.0)
        self.monitor = BatteryMonitor(self.robot, None, config(), self.clock)
        self.monitor.add_listener(lambda new, old: self.safety.set_battery_state(new))
        self.registry = ToolRegistry(
            self.robot, self.safety, speak=self.robot.speak, battery=self.monitor
        )

    def drain_to(self, volts):
        self.robot.set_battery(volts)
        for _ in range(3):
            self.clock.advance(30)
            self.monitor.sample()

    def test_the_state_and_the_sensors_report_the_battery(self):
        self.drain_to(7.6)
        state = self.registry.run(call("get_robot_state"))
        self.assertEqual(state["battery"]["state"], "ok")
        self.assertAlmostEqual(state["battery"]["voltage"], 7.6, places=1)
        self.assertIsInstance(state["battery"]["percent"], int)
        self.assertAlmostEqual(state["state"]["batteryV"], 7.6, places=1)
        sensors = self.registry.run(call("get_sensors"))
        self.assertEqual(sensors["battery"]["state"], "ok")

    def test_without_a_monitor_the_raw_voltage_is_still_reported(self):
        registry = ToolRegistry(self.robot, self.safety, speak=self.robot.speak)
        state = registry.run(call("get_robot_state"))
        self.assertEqual(state["battery"]["state"], "unknown")
        self.assertAlmostEqual(state["battery"]["voltage"], BATTERY_START_V, places=1)

    def test_a_critical_battery_blocks_the_move_tool_but_not_stop(self):
        self.drain_to(6.4)
        moved = self.registry.run(call("move", direction="forward", duration_ms=300))
        self.assertFalse(moved["ok"])
        self.assertEqual(moved["reason"], "battery_critical")
        self.assertEqual(self.robot.pose().x, 300)
        self.assertEqual(self.registry.run(call("stop")), {"ok": True})
        self.assertEqual(self.registry.run(call("get_robot_state"))["battery"]["state"], "critical")

    def test_a_low_battery_still_lets_the_user_move_her(self):
        self.drain_to(6.9)
        moved = self.registry.run(call("move", direction="forward", duration_ms=300))
        self.assertTrue(moved["ok"])


class ConversationNoteTests(unittest.TestCase):
    def setUp(self):
        self.robot = SimulatedRobot(verbose=False)
        self.safety = SafetyController(self.robot, speed=25)
        self.addCleanup(self.safety.close)
        self.clock = FakeClock(1000.0)
        self.monitor = BatteryMonitor(self.robot, None, config(), self.clock)
        self.registry = ToolRegistry(self.robot, self.safety, speak=self.robot.speak, battery=self.monitor)

    def agent(self, *script):
        self.client = ScriptedClient(*script)
        return Agent(self.client, self.registry, "Sei Laika.", battery=self.monitor)

    def drain_to(self, volts):
        self.robot.set_battery(volts)
        for _ in range(3):
            self.clock.advance(30)
            self.monitor.sample()

    def system(self, index):
        return self.client.requests[index]["messages"][0]["content"]

    def test_the_note_appears_once_per_conversation(self):
        agent = self.agent(text("Ciao."), text("Sì."), text("Ancora."))
        self.drain_to(6.9)
        agent.respond("ciao")
        agent.respond("come stai?")
        self.assertIn("ricaricata", self.system(0))
        self.assertIn("IMPORTANTE", self.system(0))
        self.assertNotIn("ricaricata", self.system(1))
        agent.reset()  # the conversation ended
        agent.respond("ehi")
        self.assertIn("ricaricata", self.system(2))

    def test_a_critical_note_says_the_wheels_stay_still(self):
        agent = self.agent(text("Ciao."))
        self.drain_to(6.4)
        agent.respond("ciao")
        self.assertIn("ferme", self.system(0))

    def test_no_note_when_the_battery_is_fine(self):
        agent = self.agent(text("Ciao."))
        self.drain_to(7.8)
        agent.respond("ciao")
        self.assertNotIn("batteria", self.system(0))

    def test_a_turn_that_did_not_answer_does_not_use_up_the_note(self):
        from laika.ai_client import AIUnavailable

        agent = self.agent(AIUnavailable("down"), text("Ciao."))
        self.drain_to(6.9)
        with self.assertRaises(AIUnavailable):
            agent.respond("ciao")
        agent.respond("ciao")
        self.assertIn("ricaricata", self.system(1))

    def test_the_prompt_helper(self):
        self.assertEqual(battery_prompt("ok", 80), "")
        self.assertIn("ricaricata", battery_prompt("low", 20))
        self.assertIn("20%", battery_prompt("low", 20))
        self.assertIn("?", battery_prompt("critical", None))


class InteractionPolicyTests(unittest.TestCase):
    def test_a_low_battery_is_something_worth_saying_within_the_normal_rules(self):
        clock, wall = FakeClock(1000.0), FakeClock(at(14))
        presence = FakePresence(clock)
        low = [False]
        policy = InteractionPolicy(
            InteractionConfig(),
            presence,
            clock=clock,
            wall_clock=wall,
            personality=NEUTRAL,
            battery_low=lambda: low[0],
        )
        presence.arrive()
        clock.advance(10)
        wall.advance(10)
        self.assertEqual(policy.decide().reason, "nothing_to_say")
        low[0] = True
        self.assertTrue(policy.decide().allowed)
        wall.now = at(22)  # quiet hours still win
        self.assertEqual(policy.decide().reason, "sleeping")
        wall.now = at(14)
        policy.record_attempt()
        self.assertEqual(policy.decide().reason, "cooldown")  # and so does the cooldown

    def test_a_broken_provider_means_nothing_to_say(self):
        clock, wall = FakeClock(1000.0), FakeClock(at(14))
        presence = FakePresence(clock)
        policy = InteractionPolicy(
            InteractionConfig(), presence, clock=clock, wall_clock=wall, personality=NEUTRAL, battery_low=lambda: 1 / 0
        )
        presence.arrive()
        clock.advance(10)
        self.assertEqual(policy.decide().reason, "nothing_to_say")


class ConfigTests(unittest.TestCase):
    def load(self, body):
        with tempfile.TemporaryDirectory() as tmp:
            path = __import__("pathlib").Path(tmp) / "laika.toml"
            path.write_text(body, encoding="utf-8")
            return load_config(path)

    def test_defaults_are_for_two_18650_cells(self):
        cfg = load_config(__import__("pathlib").Path("does-not-exist.toml")).battery
        self.assertEqual((cfg.critical_v, cfg.low_v, cfg.ok_v), (6.6, 7.0, 7.3))
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.sample_s, 30)
        self.assertEqual(cfg.voltage_offset, 0)

    def test_a_section_overrides_the_defaults(self):
        cfg = self.load("[battery]\nenabled = false\nsample_s = 60\nlow_v = 7.1\nok_v = 7.5\ncritical_v = 6.8\nvoltage_offset = -0.15\n").battery
        self.assertFalse(cfg.enabled)
        self.assertEqual((cfg.sample_s, cfg.low_v, cfg.ok_v, cfg.critical_v, cfg.voltage_offset), (60, 7.1, 7.5, 6.8, -0.15))

    def test_the_example_file_loads(self):
        root = __import__("pathlib").Path(__file__).resolve().parents[2]
        cfg = load_config(root / "config" / "laika.example.toml").battery
        self.assertEqual((cfg.critical_v, cfg.low_v, cfg.ok_v), (6.6, 7.0, 7.3))

    def test_the_threshold_order_is_enforced(self):
        for values in (
            "critical_v = 7.1\nlow_v = 7.0\nok_v = 7.3",
            "critical_v = 6.6\nlow_v = 7.4\nok_v = 7.3",
            "critical_v = 6.6\nlow_v = 7.0\nok_v = 7.0",
        ):
            with self.assertRaises(ValueError) as caught:
                self.load(f"[battery]\n{values}\n")
            self.assertIn("critical_v < low_v < ok_v", str(caught.exception))

    def test_out_of_range_and_wrong_types_name_the_setting(self):
        for body, name in (
            ("sample_s = 1", "sample_s"),
            ("low_v = 12", "low_v"),
            ("voltage_offset = 5", "voltage_offset"),
            ('low_v = "7"', "low_v"),
            ('enabled = "yes"', "enabled"),
        ):
            with self.assertRaises(ValueError) as caught:
                self.load(f"[battery]\n{body}\n")
            self.assertIn(name, str(caught.exception))
            self.assertIn("[battery]", str(caught.exception))

    def test_an_unknown_setting_is_refused(self):
        with self.assertRaises(ValueError):
            self.load("[battery]\nfull_v = 8.4\n")

    def test_validate_battery_directly(self):
        validate_battery(BatteryConfig())
        with self.assertRaises(ValueError):
            validate_battery(BatteryConfig(low_v=6.5))


if __name__ == "__main__":
    unittest.main()
