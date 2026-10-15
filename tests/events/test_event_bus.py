import json
import threading
import time
import unittest

from laika.commands import Direction, MoveCommand
from laika.events import AIEndpointMonitor, Event, EventBus, EventType
from laika.safety import SafetyController
from tests.fakes import RecordingRobot, settled_events

TERMINAL = {EventType.MOVEMENT_COMPLETED, EventType.MOVEMENT_FAILED}


def started_bus(maxsize=256):
    bus = EventBus(maxsize)
    bus.start()
    return bus


class EventBusTests(unittest.TestCase):
    def setUp(self):
        self.bus = started_bus()
        self.addCleanup(self.bus.close)

    def test_delivers_in_publish_order(self):
        seen = []
        self.bus.subscribe(lambda e: seen.append(e.data["n"]))
        for n in range(20):
            self.bus.emit(EventType.MOVEMENT_STARTED, n=n)
        self.assertTrue(self.bus.drain())
        self.assertEqual(seen, list(range(20)))

    def test_filters_by_type(self):
        seen = []
        self.bus.subscribe(lambda e: seen.append(e.type), EventType.CLIFF_DETECTED)
        self.bus.emit(EventType.OBSTACLE_DETECTED)
        self.bus.emit(EventType.CLIFF_DETECTED)
        self.assertTrue(self.bus.drain())
        self.assertEqual(seen, [EventType.CLIFF_DETECTED])

    def test_unsubscribe_stops_delivery(self):
        seen = []
        unsubscribe = self.bus.subscribe(seen.append)
        self.bus.emit(EventType.HARDWARE_READY)
        self.assertTrue(self.bus.drain())
        unsubscribe()
        self.bus.emit(EventType.HARDWARE_READY)
        self.assertTrue(self.bus.drain())
        self.assertEqual(len(seen), 1)

    def test_handler_exception_does_not_stop_delivery(self):
        seen = []

        def broken(event):
            raise RuntimeError("boom")

        self.bus.subscribe(broken)
        self.bus.subscribe(seen.append)
        with self.assertLogs("laika.events", level="ERROR"):
            self.bus.emit(EventType.HARDWARE_READY)
            self.bus.emit(EventType.HARDWARE_UNAVAILABLE)
            self.assertTrue(self.bus.drain())
        self.assertEqual(len(seen), 2)

    def test_full_queue_drops_the_oldest_and_counts(self):
        bus = EventBus(maxsize=3)  # not started: nothing consumes
        for n in range(5):
            bus.emit(EventType.MOVEMENT_STARTED, n=n)
        self.assertEqual(bus.dropped, 2)
        seen = []
        bus.subscribe(lambda e: seen.append(e.data["n"]))
        bus.start()
        self.addCleanup(bus.close)
        self.assertTrue(bus.drain())
        self.assertEqual(seen, [2, 3, 4])

    def test_publish_after_close_is_ignored(self):
        self.bus.close()
        self.bus.emit(EventType.HARDWARE_READY)

    def test_event_type_names_match_the_prd(self):
        self.assertEqual(len(EventType), 18)
        self.assertEqual(EventType.AI_ENDPOINT_UNAVAILABLE.value, "AIEndpointUnavailable")
        self.assertEqual(EventType.LOCAL_WATCHDOG_EXPIRED.value, "LocalWatchdogExpired")


class EventSerialisationTests(unittest.TestCase):
    def test_round_trip_through_json(self):
        event = Event(EventType.OBSTACLE_DETECTED, {"distanceCm": 12.5, "commandId": "cmd-1"})
        raw = json.loads(event.to_json())
        self.assertEqual(raw["type"], "ObstacleDetected")
        self.assertEqual(raw["data"], {"distanceCm": 12.5, "commandId": "cmd-1"})
        self.assertEqual(raw, event.to_dict())
        self.assertEqual(Event.from_dict(raw), event)


class MovementEventTests(unittest.TestCase):
    def setUp(self):
        self.bus = started_bus()
        self.addCleanup(self.bus.close)
        self.events = []
        self.bus.subscribe(self.events.append)

    def types(self):
        settled_events(self.bus, self.events, TERMINAL)
        return [e.type for e in self.events]

    def test_successful_move_emits_started_and_completed(self):
        robot = RecordingRobot(distance_cm=100)
        command = MoveCommand(direction=Direction.FORWARD, speed=30, duration_ms=50)
        SafetyController(robot, speed=25, events=self.bus).execute(command)
        self.assertEqual(self.types(), [EventType.MOVEMENT_STARTED, EventType.MOVEMENT_COMPLETED])
        self.assertEqual(
            self.events[0].data,
            {"commandId": command.id, "direction": "forward", "speed": 25, "durationMs": 50},
        )

    def test_blocked_move_emits_obstacle_and_failed(self):
        robot = RecordingRobot(distance_cm=10)
        command = MoveCommand(direction=Direction.FORWARD, speed=20, duration_ms=500)
        SafetyController(robot, events=self.bus).execute(command)
        self.assertEqual(
            self.types(),
            [EventType.OBSTACLE_DETECTED, EventType.MOVEMENT_FAILED],
        )
        self.assertEqual(self.events[0].data["distanceCm"], 10)
        self.assertEqual(self.events[1].data["reason"], "obstacle")

    def test_cliff_emits_cliff_and_failed(self):
        robot = RecordingRobot()
        robot.cliff = True
        command = MoveCommand(direction=Direction.BACKWARD, speed=20, duration_ms=500)
        SafetyController(robot, events=self.bus).execute(command)
        self.assertEqual(
            self.types(),
            [EventType.CLIFF_DETECTED, EventType.MOVEMENT_FAILED],
        )

    def test_guard_without_a_bus_still_works(self):
        robot = RecordingRobot(distance_cm=100)
        command = MoveCommand(direction=Direction.FORWARD, speed=20, duration_ms=50)
        self.assertEqual(SafetyController(robot).execute(command).reason, "")


class AIEndpointMonitorTests(unittest.TestCase):
    def test_emits_only_on_transitions(self):
        bus = started_bus()
        self.addCleanup(bus.close)
        seen = []
        bus.subscribe(seen.append)
        monitor = AIEndpointMonitor(bus)
        monitor.succeeded()
        monitor.failed("timeout")
        monitor.failed("timeout again")
        monitor.succeeded()
        monitor.succeeded()
        self.assertTrue(bus.drain())
        self.assertEqual(
            [e.type for e in seen],
            [EventType.AI_ENDPOINT_UNAVAILABLE, EventType.AI_ENDPOINT_RECOVERED],
        )
        self.assertEqual(seen[0].data["error"], "timeout")

    def test_works_without_a_bus(self):
        monitor = AIEndpointMonitor(None)
        monitor.failed("x")
        monitor.succeeded()


class SlowSubscriberTests(unittest.TestCase):
    def test_slow_subscriber_does_not_delay_safety(self):
        bus = started_bus()
        release = threading.Event()
        self.addCleanup(bus.close)
        self.addCleanup(release.set)
        bus.subscribe(lambda e: release.wait(5))

        robot = RecordingRobot(distance_cm=100)
        guard = SafetyController(robot, speed=25, events=bus)
        bus.emit(EventType.HARDWARE_READY)  # the subscriber is now stuck
        time.sleep(0.05)

        start = time.monotonic()
        for _ in range(500):
            bus.emit(EventType.MOVEMENT_STARTED)
        self.assertLess(time.monotonic() - start, 0.2)
        self.assertGreater(bus.dropped, 0)

        command = MoveCommand(direction=Direction.FORWARD, speed=25, duration_ms=100)
        start = time.monotonic()
        guard.execute(command)
        self.assertLess(time.monotonic() - start, 0.5)
        self.assertEqual((robot.speed, robot.steering), (0, 0))

    def test_obstacle_stop_happens_on_time_with_a_stuck_subscriber(self):
        bus = started_bus()
        release = threading.Event()
        self.addCleanup(bus.close)
        self.addCleanup(release.set)
        bus.subscribe(lambda e: release.wait(5))
        robot = RecordingRobot(distance_cm=10)
        command = MoveCommand(direction=Direction.FORWARD, speed=25, duration_ms=500)
        start = time.monotonic()
        result = SafetyController(robot, events=bus).execute(command)
        self.assertLess(time.monotonic() - start, 0.3)
        self.assertEqual(result.reason, "obstacle")
        self.assertEqual(robot.speeds, [])


if __name__ == "__main__":
    unittest.main()
