import threading
import time
import unittest

from laika.commands import CommandResult, Direction, MoveCommand, ResultStatus
from laika.events import EventBus, EventType
from laika.robot import HardwareError
from laika.safety import SafetyController
from tests.fakes import FakeClock, RecordingRobot, settled_events

TERMINAL = {EventType.MOVEMENT_COMPLETED, EventType.MOVEMENT_FAILED}

TICK_S = 0.05


def move(duration_ms=300, direction=Direction.FORWARD, speed=25, steering=0, **kwargs):
    return MoveCommand(
        direction=direction, speed=speed, duration_ms=duration_ms, steering=steering, **kwargs
    )


class BrokenSensors(RecordingRobot):
    def get_sensors(self):
        raise OSError("i2c")


class FlakySensors(RecordingRobot):
    broken = False

    def get_sensors(self):
        if self.broken:
            raise HardwareError("sensor bus lost")
        return super().get_sensors()


class ControllerTestCase(unittest.TestCase):
    def controller(self, robot=None, started=True, **kwargs):
        self.robot = robot or RecordingRobot(distance_cm=100)
        kwargs.setdefault("speed", 25)
        controller = SafetyController(self.robot, **kwargs)
        self.addCleanup(controller.close)
        if started:
            controller.start()
        return controller

    def bus(self):
        bus = EventBus()
        bus.start()
        self.addCleanup(bus.close)
        self.seen = []
        bus.subscribe(self.seen.append)
        return bus

    def event_types(self, bus):
        settled_events(bus, self.seen, TERMINAL)
        return [e.type for e in self.seen]


class StopConditionTests(ControllerTestCase):
    def test_a_full_move_succeeds_and_stops_and_centres(self):
        controller = self.controller()
        command = move(100, steering=10)
        result = controller.execute(command)
        self.assertEqual(result, CommandResult(command.id, ResultStatus.SUCCESS))
        self.assertEqual(self.robot.speeds[0], 25)
        self.assertEqual((self.robot.speed, self.robot.steering), (0, 0))

    def test_obstacle_while_driving_forward_stops_the_move(self):
        controller = self.controller()
        command = move(1000)
        self.assertIsNone(controller.execute(command, wait=False))
        time.sleep(0.1)
        self.assertEqual(self.robot.speed, 25)
        self.robot.distance = 10
        result = controller.wait(command.id, timeout=1)
        self.assertEqual((result.result, result.reason), (ResultStatus.CANCELLED, "obstacle"))
        self.assertEqual(self.robot.speed, 0)

    def test_obstacle_ahead_rejects_a_forward_move(self):
        controller = self.controller(RecordingRobot(distance_cm=10))
        result = controller.execute(move())
        self.assertEqual((result.result, result.reason), (ResultStatus.REJECTED, "obstacle"))
        self.assertEqual(self.robot.speeds, [])

    def test_backward_moves_ignore_the_front_sensor(self):
        controller = self.controller(RecordingRobot(distance_cm=10))
        result = controller.execute(move(50, Direction.BACKWARD))
        self.assertEqual(result.result, ResultStatus.SUCCESS)
        self.assertEqual(self.robot.speeds[0], -25)

    def test_cliff_during_a_move_stops_it(self):
        controller = self.controller()
        command = move(1000, Direction.BACKWARD)
        controller.execute(command, wait=False)
        time.sleep(0.1)
        self.robot.cliff = True
        result = controller.wait(command.id, timeout=1)
        self.assertEqual((result.result, result.reason), (ResultStatus.CANCELLED, "cliff"))
        self.assertEqual(self.robot.speed, 0)

    def test_cliff_rejects_a_move(self):
        robot = RecordingRobot(distance_cm=100)
        robot.cliff = True
        result = self.controller(robot).execute(move())
        self.assertEqual((result.result, result.reason), (ResultStatus.REJECTED, "cliff"))
        self.assertEqual(self.robot.speeds, [])

    def test_the_move_stops_at_its_deadline_even_if_the_caller_stalls(self):
        controller = self.controller()
        command = move(300)
        controller.execute(command, wait=False)
        time.sleep(0.1)
        self.assertEqual(self.robot.speed, 25)
        time.sleep(0.2 + 2 * TICK_S + 0.15)  # the caller does nothing
        self.assertEqual(self.robot.speed, 0)
        self.assertEqual(controller.wait(command.id, 0).result, ResultStatus.SUCCESS)

    def test_the_deadline_uses_the_injected_clock(self):
        clock = FakeClock()
        controller = self.controller(clock=clock, max_duration_s=5)
        command = move(1000)
        controller.execute(command, wait=False)
        clock.advance(0.9)
        controller.check()
        self.assertEqual(self.robot.speed, 25)
        clock.advance(0.2)
        controller.check()
        self.assertEqual(self.robot.speed, 0)
        self.assertEqual(controller.wait(command.id, 1).result, ResultStatus.SUCCESS)

    def test_the_duration_is_clamped_by_the_limits(self):
        clock = FakeClock()
        controller = self.controller(clock=clock, max_duration_s=0.5)
        command = move(1500)
        controller.execute(command, wait=False)
        clock.advance(0.6)
        controller.check()
        self.assertEqual(controller.wait(command.id, 1).result, ResultStatus.SUCCESS)

    def test_backward_moves_are_shorter(self):
        clock = FakeClock()
        controller = self.controller(clock=clock, max_duration_s=1.5)
        command = move(1500, Direction.BACKWARD)
        controller.execute(command, wait=False)
        clock.advance(0.7)
        controller.check()
        self.assertEqual(controller.wait(command.id, 1).result, ResultStatus.SUCCESS)

    def test_an_expired_command_never_moves(self):
        controller = self.controller()
        command = move(created_at=time.time() - 5)
        result = controller.execute(command)
        self.assertEqual((result.result, result.reason), (ResultStatus.EXPIRED, "expired"))
        self.assertEqual(self.robot.speeds, [])

    def test_a_fresh_command_is_not_expired(self):
        controller = self.controller(command_max_age_s=2)
        result = controller.execute(move(50, created_at=time.time() - 0.5))
        self.assertEqual(result.result, ResultStatus.SUCCESS)

    def test_a_non_move_command_is_rejected_as_invalid(self):
        controller = self.controller()
        for bad in ("forward", None, {"command": "move"}, 42):
            result = controller.execute(bad)
            self.assertEqual((result.result, result.reason), (ResultStatus.REJECTED, "invalid"))
        self.assertEqual(self.robot.speeds, [])

    def test_raw_input_is_validated_before_anything_moves(self):
        controller = self.controller()
        data = move(50).to_dict()
        for bad in (
            {**data, "parameters": {**data["parameters"], "speed": 999}},
            {**data, "command": "fly"},
            {**data, "extra": 1},
            "not json",
            [],
            {"id": "x" * 100},
        ):
            result = controller.execute_raw(bad)
            self.assertEqual(result.result, ResultStatus.REJECTED, bad)
            self.assertTrue(result.reason.startswith("invalid"), result.reason)
        self.assertEqual(self.robot.speeds, [])

    def test_raw_input_keeps_the_command_id_in_the_rejection(self):
        data = move(50).to_dict()
        data["parameters"]["durationMs"] = 999999
        result = self.controller().execute_raw(data)
        self.assertEqual(result.id, data["id"])

    def test_valid_raw_input_runs(self):
        controller = self.controller()
        self.assertEqual(controller.execute_raw(move(50).to_json()).result, ResultStatus.SUCCESS)
        self.assertEqual(self.robot.speeds[0], 25)

    def test_raw_non_movement_commands_do_not_run(self):
        controller = self.controller()
        speak = {"type": "command", "command": "speak", "parameters": {"text": "ciao"}}
        self.assertEqual(controller.execute_raw(speak).result, ResultStatus.REJECTED)

    def test_raw_stop_stops_the_current_move(self):
        controller = self.controller()
        command = move(1000)
        controller.execute(command, wait=False)
        stop = {"type": "command", "command": "stop"}
        self.assertEqual(controller.execute_raw(stop).result, ResultStatus.SUCCESS)
        self.assertEqual(controller.wait(command.id, 1).reason, "stopped")
        self.assertEqual(self.robot.speed, 0)

    def test_a_disabled_controller_never_moves(self):
        controller = self.controller(enabled=False)
        result = controller.execute(move())
        self.assertEqual((result.result, result.reason), (ResultStatus.REJECTED, "disabled"))
        self.assertEqual(self.robot.speeds, [])

    def test_speed_is_capped(self):
        controller = self.controller(speed=90)
        controller.execute(move(50, speed=40))
        self.assertEqual(self.robot.speeds[0], 40)
        controller = self.controller(speed=20)
        controller.execute(move(50, speed=40))
        self.assertEqual(self.robot.speeds[0], 20)


class HardwareErrorTests(ControllerTestCase):
    def test_a_motor_error_at_start_fails_the_move(self):
        bus = self.bus()
        controller = self.controller(events=bus)
        self.robot.faults.drive_error = True
        result = controller.execute(move())
        self.assertEqual(result.result, ResultStatus.FAILED)
        self.assertEqual(self.robot.speed, 0)
        self.assertIn(EventType.HARDWARE_UNAVAILABLE, self.event_types(bus))

    def test_a_sensor_error_during_a_move_stops_it(self):
        bus = self.bus()
        robot = FlakySensors(distance_cm=100)
        controller = self.controller(robot, events=bus)
        command = move(1000)
        controller.execute(command, wait=False)
        time.sleep(0.1)
        self.assertEqual(robot.speed, 25)
        robot.broken = True
        result = controller.wait(command.id, timeout=1)
        self.assertEqual(result.result, ResultStatus.FAILED)
        self.assertEqual(robot.speed, 0)
        types = self.event_types(bus)
        self.assertIn(EventType.HARDWARE_UNAVAILABLE, types)
        self.assertIn(EventType.MOVEMENT_FAILED, types)

    def test_any_exception_from_the_robot_is_a_hardware_error(self):
        result = self.controller(BrokenSensors()).execute(move())
        self.assertEqual(result.result, ResultStatus.FAILED)

    def test_the_monitor_survives_a_sensor_error_without_a_move(self):
        bus = self.bus()
        controller = self.controller(BrokenSensors(), events=bus)
        time.sleep(0.3)
        self.assertGreater(controller.stats().ticks, 3)
        unavailable = [t for t in self.event_types(bus) if t is EventType.HARDWARE_UNAVAILABLE]
        self.assertEqual(len(unavailable), 1)


class EmergencyStopTests(ControllerTestCase):
    def test_emergency_stop_cancels_the_move_and_latches(self):
        controller = self.controller()
        command = move(1000)
        controller.execute(command, wait=False)
        time.sleep(0.1)
        controller.emergency_stop("button")
        self.assertEqual(self.robot.speed, 0)
        self.assertEqual(controller.wait(command.id, 1).reason, "emergency_stop")
        self.assertTrue(controller.emergency_stopped)
        result = controller.execute(move(50))
        self.assertEqual((result.result, result.reason), (ResultStatus.REJECTED, "emergency_stop"))

    def test_reset_allows_moving_again(self):
        controller = self.controller()
        controller.emergency_stop()
        controller.reset_emergency_stop()
        self.assertFalse(controller.emergency_stopped)
        self.assertEqual(controller.execute(move(50)).result, ResultStatus.SUCCESS)

    def test_stop_cancels_without_latching(self):
        controller = self.controller()
        command = move(1000)
        controller.execute(command, wait=False)
        controller.stop()
        self.assertEqual(controller.wait(command.id, 1).reason, "stopped")
        self.assertEqual(self.robot.speed, 0)
        self.assertEqual(controller.execute(move(50)).result, ResultStatus.SUCCESS)

    def test_a_new_command_supersedes_the_running_one(self):
        controller = self.controller()
        first = move(1000)
        controller.execute(first, wait=False)
        time.sleep(0.1)
        second = move(100)
        self.assertEqual(controller.execute(second).result, ResultStatus.SUCCESS)
        result = controller.wait(first.id, 0)
        self.assertEqual((result.result, result.reason), (ResultStatus.CANCELLED, "superseded"))

    def test_a_refused_command_does_not_cancel_the_running_one(self):
        controller = self.controller()
        first = move(400)
        controller.execute(first, wait=False)
        stale = move(created_at=time.time() - 9)
        self.assertEqual(controller.execute(stale).result, ResultStatus.EXPIRED)
        self.assertEqual(controller.wait(first.id, 2).result, ResultStatus.SUCCESS)

    def test_close_stops_the_motors_and_the_monitor(self):
        controller = self.controller()
        command = move(1000)
        controller.execute(command, wait=False)
        time.sleep(0.1)
        controller.close()
        self.assertEqual(self.robot.speed, 0)
        self.assertEqual(controller.wait(command.id, 1).reason, "closed")
        ticks = controller.stats().ticks
        time.sleep(0.2)
        self.assertEqual(controller.stats().ticks, ticks)
        self.assertEqual(controller.execute(move()).reason, "closed")

    def test_a_blocked_waiter_is_released_by_close(self):
        controller = self.controller()
        results = []
        thread = threading.Thread(target=lambda: results.append(controller.execute(move(1000))))
        thread.start()
        time.sleep(0.1)
        controller.close()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results[0].reason, "closed")


class CadenceTests(ControllerTestCase):
    def test_the_monitor_ticks_at_the_configured_cadence(self):
        controller = self.controller(cadence_hz=20)
        time.sleep(0.8)
        stats = controller.stats()
        self.assertGreater(stats.ticks, 8)
        self.assertGreater(len(stats.intervals), 8)
        self.assertTrue(0.035 <= stats.mean_interval_s <= 0.08, stats.mean_interval_s)

    def test_a_faster_cadence_ticks_faster(self):
        controller = self.controller(cadence_hz=50)
        time.sleep(0.6)
        self.assertTrue(0.01 <= controller.stats().mean_interval_s <= 0.04)

    def test_monitoring_continues_with_no_active_move(self):
        controller = self.controller()
        before = controller.stats().ticks
        time.sleep(0.3)
        self.assertGreater(controller.stats().ticks, before + 3)

    def test_ticks_continue_while_another_thread_is_busy(self):
        controller = self.controller()
        busy = threading.Thread(target=lambda: time.sleep(0.6))  # a slow fake AI request
        busy.start()
        before = controller.stats().ticks
        busy.join()
        self.assertGreater(controller.stats().ticks - before, 6)

    def test_a_stalled_caller_does_not_stop_the_monitor_from_stopping_the_move(self):
        controller = self.controller()
        command = move(1000)
        controller.execute(command, wait=False)
        time.sleep(0.05)
        self.robot.distance = 5
        time.sleep(TICK_S * 3)  # the caller is busy elsewhere
        self.assertEqual(self.robot.speed, 0)
        self.assertEqual(controller.wait(command.id, 0).reason, "obstacle")

    def test_a_slow_event_subscriber_does_not_slow_the_ticks_or_the_stop(self):
        bus = EventBus()
        bus.start()
        release = threading.Event()
        self.addCleanup(bus.close)
        self.addCleanup(release.set)
        bus.subscribe(lambda e: release.wait(5))
        controller = self.controller(events=bus)
        for _ in range(3):
            command = move(1000)
            controller.execute(command, wait=False)
            time.sleep(0.05)
            self.robot.distance = 5
            result = controller.wait(command.id, timeout=0.5)
            self.assertEqual(result.reason, "obstacle")
            self.robot.distance = 100
            time.sleep(0.05)
        time.sleep(0.5)
        self.assertLessEqual(controller.stats().mean_interval_s, 0.08)

    def test_the_interval_history_is_bounded(self):
        controller = self.controller(cadence_hz=100)
        time.sleep(1.3)
        self.assertLessEqual(len(controller.stats().intervals), 100)


class EventTests(ControllerTestCase):
    def test_successful_move_emits_started_and_completed(self):
        bus = self.bus()
        controller = self.controller(events=bus)
        command = move(50, steering=5)
        controller.execute(command)
        self.assertEqual(
            self.event_types(bus), [EventType.MOVEMENT_STARTED, EventType.MOVEMENT_COMPLETED]
        )
        self.assertEqual(
            self.seen[0].data,
            {"commandId": command.id, "direction": "forward", "speed": 25, "durationMs": 50},
        )

    def test_blocked_move_emits_obstacle_and_failed(self):
        bus = self.bus()
        controller = self.controller(RecordingRobot(distance_cm=10), events=bus)
        controller.execute(move())
        self.assertEqual(
            self.event_types(bus),
            [EventType.OBSTACLE_DETECTED, EventType.MOVEMENT_FAILED],
        )
        self.assertEqual(self.seen[0].data["distanceCm"], 10)
        self.assertEqual(self.seen[1].data["reason"], "obstacle")

    def test_a_cliff_emits_cliff_and_failed(self):
        bus = self.bus()
        robot = RecordingRobot()
        robot.cliff = True
        self.controller(robot, events=bus).execute(move(direction=Direction.BACKWARD))
        self.assertEqual(
            self.event_types(bus),
            [EventType.CLIFF_DETECTED, EventType.MOVEMENT_FAILED],
        )

    def test_works_without_a_bus(self):
        self.assertEqual(self.controller().execute(move(50)).reason, "")


if __name__ == "__main__":
    unittest.main()
