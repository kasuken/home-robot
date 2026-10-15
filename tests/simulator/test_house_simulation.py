import unittest

from laika.commands import Direction, MoveCommand, ResultStatus
from laika.events import EventBus, EventType
from laika.house import Person, Pose, Thing
from laika.robot import HardwareError
from laika.safety import SafetyController
from laika.simulator import AUTO, SimulatedRobot
from tests.fakes import FakeClock
from tests.simulator.test_house import two_rooms


class HouseTestCase(unittest.TestCase):
    def build(self, start=Pose(50, 100, 0), **kwargs):
        self.clock = FakeClock()
        self.house = two_rooms()
        self.robot = SimulatedRobot(
            verbose=False, house=self.house, start=start, clock=self.clock, **kwargs
        )
        return self.robot

    def safety(self, robot, **kwargs):
        controller = SafetyController(robot, **kwargs)
        self.addCleanup(controller.close)
        return controller

    def forward(self, duration_ms):
        return MoveCommand(direction=Direction.FORWARD, speed=40, duration_ms=duration_ms)

    def start_bus(self):
        bus = EventBus()
        bus.start()
        self.addCleanup(bus.close)
        return bus

    def events_of(self, bus, *types):
        seen = []
        bus.subscribe(seen.append, *types)
        return seen


class MovementTests(HouseTestCase):
    def test_distance_comes_from_the_ray(self):
        robot = self.build(start=Pose(50, 100, 90))
        self.assertAlmostEqual(robot.distance_cm(), 90)
        robot = self.build()
        self.house.add_thing(Thing("box", 150, 100, 20))
        self.assertAlmostEqual(robot.distance_cm(), 70)

    def test_driving_forward_reduces_the_distance(self):
        robot = self.build()
        self.house.add_thing(Thing("box", 250, 100, 20))
        before = robot.distance_cm()
        robot.drive(50)
        self.clock.advance(1.0)
        self.assertAlmostEqual(robot.pose().x, 80)
        self.assertAlmostEqual(robot.distance_cm(), before - 30)

    def test_reversing_moves_backwards(self):
        robot = self.build()
        robot.drive(-50)
        self.clock.advance(1.0)
        self.assertAlmostEqual(robot.pose().x, 20)

    def test_bumping_into_a_wall_stops_the_robot(self):
        robot = self.build(start=Pose(50, 100, 90))
        robot.drive(100)
        self.clock.advance(4.0)
        pose = robot.pose()
        self.assertEqual(robot.speed, 0)
        self.assertEqual(robot.bumps, 1)
        self.assertAlmostEqual(pose.y, 190, delta=2)
        self.clock.advance(4.0)
        self.assertEqual(robot.pose(), pose)

    def test_bumping_into_a_thing_stops_the_robot(self):
        robot = self.build()
        self.house.add_thing(Thing("box", 150, 100, 20))
        robot.drive(100)
        self.clock.advance(4.0)
        self.assertAlmostEqual(robot.pose().x, 120, delta=2)
        self.assertEqual(robot.speed, 0)
        self.assertAlmostEqual(robot.distance_cm(), 0, delta=2)

    def test_steering_changes_the_heading(self):
        robot = self.build()
        robot.steer(30)
        robot.drive(50)
        self.clock.advance(0.5)
        right = robot.pose()
        self.assertGreater(right.heading, 180)
        self.assertLess(right.y, 100)

        robot = self.build()
        robot.steer(-30)
        robot.drive(50)
        self.clock.advance(0.5)
        left = robot.pose()
        self.assertLess(left.heading, 180)
        self.assertGreater(left.y, 100)

    def test_the_robot_only_moves_while_the_motors_run(self):
        robot = self.build()
        robot.drive(50)
        self.clock.advance(1.0)
        robot.stop()
        self.clock.advance(10.0)
        self.assertAlmostEqual(robot.pose().x, 80)

    def test_the_robot_drives_through_a_doorway(self):
        robot = self.build(start=Pose(250, 100, 0))
        robot.drive(50)
        self.clock.advance(3.0)
        self.assertEqual(robot.bumps, 0)
        self.assertEqual(robot.current_room(), "b")

    def test_without_a_house_nothing_moves(self):
        robot = SimulatedRobot(verbose=False)
        robot.drive(100)
        self.assertEqual(robot.pose(), Pose(0.0, 0.0, 0.0))
        self.assertIsNone(robot.current_room())
        self.assertEqual(robot.distance_cm(), 100.0)


class CliffTests(HouseTestCase):
    def test_the_cliff_sensor_sees_the_stairs(self):
        robot = self.build(start=Pose(480, 100, 0))
        self.assertFalse(robot.cliff_detected())
        robot.drive(50)
        self.clock.advance(0.5)
        self.assertTrue(robot.cliff_detected())
        self.assertTrue(robot.get_sensors().cliff_detected)
        self.assertEqual(robot.get_sensors().grayscale, (50, 50, 50))

    def test_the_guard_stops_the_robot_at_the_stairs(self):
        # Real time: the monitor thread paces itself in real time, so this robot uses the real clock.
        robot = SimulatedRobot(verbose=False, house=two_rooms(), start=Pose(488, 100, 0))
        result = self.safety(robot, speed=25).execute(self.forward(1500))
        self.assertEqual((result.result, result.reason), (ResultStatus.CANCELLED, "cliff"))
        self.assertEqual(robot.speed, 0)

    def test_the_guard_refuses_to_start_over_a_cliff(self):
        robot = SimulatedRobot(verbose=False, house=two_rooms(), start=Pose(495, 100, 0))
        result = self.safety(robot).execute(self.forward(1000))
        self.assertEqual((result.result, result.reason), (ResultStatus.REJECTED, "cliff"))

    def test_the_guard_stops_before_an_obstacle(self):
        house = two_rooms()
        house.add_thing(Thing("box", 120, 100, 20))
        robot = SimulatedRobot(verbose=False, house=house, start=Pose(50, 100, 0))
        result = self.safety(robot, speed=40, min_front_distance_cm=25).execute(self.forward(1500))
        self.assertEqual((result.result, result.reason), (ResultStatus.CANCELLED, "obstacle"))
        self.assertEqual(robot.bumps, 0)


class RoomEventTests(HouseTestCase):
    def test_crossing_a_doorway_publishes_room_entered(self):
        bus = self.start_bus()
        seen = self.events_of(bus, EventType.ROOM_ENTERED)
        robot = self.build(start=Pose(250, 100, 0), events=bus)
        robot.drive(50)
        self.clock.advance(1.0)
        robot.pose()
        self.clock.advance(1.0)
        robot.pose()
        self.assertTrue(bus.drain())
        self.assertEqual([e.data for e in seen], [{"room": "b", "previous": "a"}])

    def test_staying_in_a_room_publishes_nothing(self):
        bus = self.start_bus()
        seen = self.events_of(bus)
        robot = self.build(events=bus)
        robot.drive(50)
        self.clock.advance(1.0)
        robot.pose()
        self.assertTrue(bus.drain())
        self.assertEqual(seen, [])


class CameraTests(HouseTestCase):
    def test_the_scene_lists_what_the_camera_sees(self):
        robot = self.build()
        self.house.add_thing(Thing("box", 150, 100, 10))
        self.house.add_thing(Thing("behind", 5, 150, 5))
        self.assertEqual(robot.capture_image().scene, ("box",))

    def test_pan_turns_the_camera(self):
        robot = self.build()
        self.house.add_person(Person("anna", 100, 170))
        self.assertEqual(robot.people_visible(), ())
        robot.look(-54, 0)
        self.assertEqual(robot.people_visible(), ("anna",))
        self.assertEqual(robot.capture_image().scene, ("anna",))
        robot.look(54, 0)
        self.assertEqual(robot.people_visible(), ())

    def test_person_detected_and_lost(self):
        bus = self.start_bus()
        seen = self.events_of(bus, EventType.PERSON_DETECTED, EventType.PERSON_LOST)
        robot = self.build(events=bus)
        robot.capture_image()
        self.house.add_person(Person("anna", 150, 100))
        robot.capture_image()
        robot.capture_image()
        self.house.move_person("anna", 150, 190)
        robot.capture_image()
        self.assertTrue(bus.drain())
        self.assertEqual(
            [(e.type, e.data["name"]) for e in seen],
            [(EventType.PERSON_DETECTED, "anna"), (EventType.PERSON_LOST, "anna")],
        )


class FaultTests(HouseTestCase):
    def test_ultrasonic_invalid(self):
        robot = self.build()
        robot.faults.ultrasonic_invalid = True
        self.assertIsNone(robot.distance_cm())
        self.assertIsNone(robot.get_sensors().distance_cm)
        robot.faults.ultrasonic_invalid = False
        self.assertIsNotNone(robot.distance_cm())

    def test_ultrasonic_stuck(self):
        robot = self.build()
        robot.faults.ultrasonic_stuck_cm = 150.0
        robot.drive(50)
        self.clock.advance(1.0)
        self.assertEqual(robot.distance_cm(), 150.0)

    def test_a_stuck_cliff_sensor_blocks_the_guard(self):
        robot = self.build()
        robot.faults.cliff_stuck = True
        self.assertTrue(robot.cliff_detected())
        result = self.safety(robot).execute(self.forward(500))
        self.assertEqual(result.result, ResultStatus.REJECTED)

    def test_drive_and_steer_raise_hardware_errors(self):
        robot = self.build()
        robot.faults.drive_error = True
        robot.faults.steer_error = True
        with self.assertRaises(HardwareError):
            robot.drive(30)
        with self.assertRaises(HardwareError):
            robot.steer(10)
        self.assertEqual(robot.speed, 0)

    def test_stop_and_reset_still_work_after_a_motor_fault(self):
        robot = self.build()
        robot.drive(30)
        robot.faults.drive_error = True
        robot.faults.steer_error = True
        robot.stop()
        robot.reset()
        self.assertEqual(robot.speed, 0)

    def test_camera_failure(self):
        robot = self.build()
        robot.faults.camera_error = True
        with self.assertRaises(HardwareError):
            robot.capture_image()


class EnvironmentTests(HouseTestCase):
    def test_moving_a_thing_changes_the_reading(self):
        robot = self.build()
        self.house.add_thing(Thing("box", 150, 100, 20))
        self.assertAlmostEqual(robot.distance_cm(), 70)
        self.house.move_thing("box", 200, 100)
        self.assertAlmostEqual(robot.distance_cm(), 120)
        self.house.remove_thing("box")
        self.assertEqual(robot.distance_cm(), 400)

    def test_a_person_walking_in_front_is_an_obstacle(self):
        robot = self.build()
        self.house.add_person(Person("anna", 150, 100, 20))
        self.assertAlmostEqual(robot.distance_cm(), 70)
        self.house.remove_person("anna")
        self.assertEqual(robot.distance_cm(), 400)

    def test_overrides_still_work_in_a_house(self):
        robot = self.build()
        robot.distance = 12.0
        robot.cliff = True
        self.assertEqual(robot.distance_cm(), 12.0)
        self.assertTrue(robot.cliff_detected())
        robot.distance = None
        self.assertIsNone(robot.distance_cm())
        robot.distance = AUTO
        robot.cliff = None
        self.assertEqual(robot.distance_cm(), 400)
        self.assertFalse(robot.cliff_detected())


if __name__ == "__main__":
    unittest.main()
