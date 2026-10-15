import threading
import time
import unittest
from unittest import mock

from laika.commands import Direction, MoveCommand, ResultStatus
from laika.events import EventBus, EventType
from laika.safety import SafetyController
from tests.fakes import RecordingRobot

TICK_S = 0.05
WATCHDOG_S = 0.3
WATCHDOG_PERIOD_S = 0.05
MARGIN_S = 0.5  # generous: CI machines are slow


def move(duration_ms=1000, **kwargs):
    return MoveCommand(direction=Direction.FORWARD, speed=25, duration_ms=duration_ms, **kwargs)


class StallableRobot(RecordingRobot):
    """get_sensors hangs (outside the robot lock) when called by the monitor thread."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.block = False
        self.die = False
        self.release = threading.Event()

    def get_sensors(self):
        if threading.current_thread().name == "laika-safety":
            if self.die:
                raise SystemExit("monitor thread killed")
            if self.block:
                self.release.wait(10)
        return super().get_sensors()


class WatchdogTestCase(unittest.TestCase):
    def setUp(self):
        self.robot = StallableRobot(distance_cm=100)
        self.addCleanup(self.robot.release.set)
        self.bus = EventBus()
        self.bus.start()
        self.addCleanup(self.bus.close)
        self.seen = []
        self.bus.subscribe(self.seen.append)
        self.controller = SafetyController(
            self.robot,
            speed=25,
            events=self.bus,
            watchdog_timeout_s=WATCHDOG_S,
            watchdog_period_s=WATCHDOG_PERIOD_S,
        )
        self.addCleanup(self.controller.close)
        self.addCleanup(self.robot.release.set)  # runs first: free a stalled monitor
        self.controller.start()

    def wait_for(self, condition, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return True
            time.sleep(0.01)
        return condition()

    def types(self):
        self.assertTrue(self.bus.drain())
        return [e.type for e in self.seen]

    def start_moving(self, duration_ms=1000):
        command = move(duration_ms)
        self.assertIsNone(self.controller.execute(command, wait=False))
        self.assertTrue(self.wait_for(lambda: self.robot.speed == 25))
        return command

    def stall_monitor(self):
        self.robot.block = True
        return time.monotonic()


class MonitorStallTests(WatchdogTestCase):
    def test_a_stalled_monitor_is_stopped_by_the_watchdog_within_its_timeout(self):
        command = self.start_moving()
        stalled_at = self.stall_monitor()
        self.assertTrue(self.wait_for(lambda: self.robot.speed == 0))
        # The monitor may tick once more before blocking, hence the extra TICK_S.
        self.assertLessEqual(
            time.monotonic() - stalled_at, WATCHDOG_S + WATCHDOG_PERIOD_S + TICK_S + MARGIN_S
        )
        result = self.controller.wait(command.id, timeout=1)
        self.assertEqual((result.result, result.reason), (ResultStatus.CANCELLED, "watchdog"))
        self.assertTrue(self.controller.watchdog_expired)

    def test_the_event_is_published_once_per_expiry(self):
        self.start_moving()
        self.stall_monitor()
        self.assertTrue(self.wait_for(lambda: self.controller.watchdog_expired))
        time.sleep(WATCHDOG_S)  # many more watchdog periods while still stalled
        types = self.types()
        self.assertEqual(types.count(EventType.LOCAL_WATCHDOG_EXPIRED), 1)
        event = next(e for e in self.seen if e.type is EventType.LOCAL_WATCHDOG_EXPIRED)
        self.assertEqual(event.data["reason"], "monitor_stalled")
        self.assertGreater(event.data["ageS"], WATCHDOG_S)
        # A second stall publishes a second event.
        self.robot.block = False
        self.robot.release.set()
        self.assertTrue(self.wait_for(lambda: not self.controller.watchdog_expired))
        self.robot.release.clear()
        self.stall_monitor()
        self.assertTrue(self.wait_for(lambda: self.controller.watchdog_expired))
        self.assertTrue(
            self.wait_for(lambda: self.types().count(EventType.LOCAL_WATCHDOG_EXPIRED) == 2)
        )

    def test_moves_are_rejected_while_expired_even_though_the_monitor_holds_the_lock(self):
        self.stall_monitor()
        self.assertTrue(self.wait_for(lambda: self.controller.watchdog_expired))
        started = time.monotonic()
        result = self.controller.execute(move(), wait=True)
        self.assertLess(time.monotonic() - started, 1.0)  # did not wait for the lock
        self.assertEqual((result.result, result.reason), (ResultStatus.REJECTED, "watchdog"))
        self.assertEqual(self.robot.speed, 0)

    def test_it_clears_by_itself_and_never_restarts_the_cancelled_move(self):
        command = self.start_moving()
        self.stall_monitor()
        self.assertTrue(self.wait_for(lambda: self.robot.speed == 0))
        self.robot.block = False
        self.robot.release.set()
        self.assertTrue(self.wait_for(lambda: not self.controller.watchdog_expired))
        time.sleep(TICK_S * 4)
        self.assertEqual(self.robot.speed, 0)
        result = self.controller.wait(command.id, timeout=0)
        self.assertEqual((result.result, result.reason), (ResultStatus.CANCELLED, "watchdog"))
        # New moves work again, on a fresh command.
        fresh = self.controller.execute(move(100))
        self.assertEqual(fresh.result, ResultStatus.SUCCESS)

    def test_a_waiting_caller_is_released_when_the_watchdog_cancels_the_move(self):
        results = []
        thread = threading.Thread(target=lambda: results.append(self.controller.execute(move())))
        thread.start()
        self.assertTrue(self.wait_for(lambda: self.robot.speed == 25))
        self.stall_monitor()
        thread.join(WATCHDOG_S + MARGIN_S + 1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results[0].reason, "watchdog")

    def test_a_failing_stop_does_not_stop_the_watchdog(self):
        self.start_moving()
        with mock.patch.object(self.robot, "stop", side_effect=OSError("bus")):
            self.stall_monitor()
            self.assertTrue(self.wait_for(lambda: self.controller.watchdog_expired))
            self.assertTrue(
                self.wait_for(lambda: EventType.LOCAL_WATCHDOG_EXPIRED in self.types())
            )

    def test_an_idle_monitor_that_keeps_ticking_never_expires(self):
        time.sleep(WATCHDOG_S * 3)
        self.assertFalse(self.controller.watchdog_expired)
        self.assertNotIn(EventType.LOCAL_WATCHDOG_EXPIRED, self.types())

    def test_close_stops_the_watchdog_thread(self):
        self.controller.close()
        self.assertFalse(self.controller._watchdog._thread.is_alive())
        self.assertFalse(self.controller._thread.is_alive())


class MonitorDeathTests(WatchdogTestCase):
    def test_a_dead_monitor_thread_is_caught(self):
        command = self.start_moving()
        with mock.patch.object(threading, "excepthook"):  # the thread dies on purpose
            self.robot.die = True
            self.assertTrue(self.wait_for(lambda: self.controller.watchdog_expired))
        self.assertTrue(self.wait_for(lambda: self.robot.speed == 0))
        result = self.controller.wait(command.id, timeout=1)
        self.assertEqual((result.result, result.reason), (ResultStatus.CANCELLED, "watchdog"))
        self.assertEqual(self.controller.execute(move()).reason, "watchdog")
        self.types()
        event = next(e for e in self.seen if e.type is EventType.LOCAL_WATCHDOG_EXPIRED)
        self.assertEqual(event.data["reason"], "monitor_dead")
        # Nothing restarts a dead monitor: it stays expired.
        time.sleep(WATCHDOG_S)
        self.assertTrue(self.controller.watchdog_expired)


class DeadMansSwitchTests(unittest.TestCase):
    """Movement only continues while fresh commands keep arriving."""

    def test_a_hung_agent_thread_cannot_keep_the_robot_moving(self):
        robot = RecordingRobot(distance_cm=100)
        controller = SafetyController(robot, speed=25, max_duration_s=0.4)
        self.addCleanup(controller.close)
        hang = threading.Event()
        self.addCleanup(hang.set)

        def agent():
            # Asks for a very long move, then never comes back (a hung request).
            controller.execute(move(1500), wait=False)
            hang.wait()

        threading.Thread(target=agent, daemon=True).start()
        time.sleep(0.1)
        self.assertEqual(robot.speed, 25)
        time.sleep(0.4 - 0.1 + 2 * TICK_S + MARGIN_S)  # deadline + 2 ticks + margin
        self.assertEqual(robot.speed, 0)

    def test_a_stale_command_never_starts(self):
        robot = RecordingRobot(distance_cm=100)
        controller = SafetyController(robot, speed=25, command_max_age_s=1.0)
        self.addCleanup(controller.close)
        old = move(500, created_at=time.time() - 5)
        self.assertEqual(controller.execute(old).result, ResultStatus.EXPIRED)
        self.assertEqual(robot.speeds, [])


if __name__ == "__main__":
    unittest.main()
