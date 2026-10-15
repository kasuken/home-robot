import unittest

from laika.safety import MovementGuard
from tests.fakes import RecordingRobot


class MovementGuardTests(unittest.TestCase):
    def test_refuses_to_drive_towards_an_obstacle(self):
        robot = RecordingRobot(distance_cm=10)
        result = MovementGuard(robot, min_front_distance_cm=25).drive(forward=True, duration_s=1)
        self.assertFalse(result.moved)
        self.assertEqual(result.reason, "obstacle")
        self.assertEqual(robot.speeds, [])

    def test_refuses_to_drive_at_a_cliff(self):
        robot = RecordingRobot()
        robot.cliff = True
        self.assertEqual(MovementGuard(robot).drive(forward=False, duration_s=1).reason, "cliff")

    def test_drives_briefly_then_stops_and_centres(self):
        robot = RecordingRobot(distance_cm=100)
        result = MovementGuard(robot, speed=25, max_duration_s=0.1).drive(True, 5, steer=-30)
        self.assertTrue(result.moved)
        self.assertEqual(robot.speeds, [25])
        self.assertEqual(robot.speed, 0)
        self.assertEqual(robot.steering, 0)

    def test_speed_is_capped(self):
        robot = RecordingRobot(distance_cm=100)
        MovementGuard(robot, speed=90, max_duration_s=0.05).drive(True, 1)
        self.assertEqual(robot.speeds, [40])

    def test_disabled_guard_never_moves(self):
        robot = RecordingRobot()
        self.assertFalse(MovementGuard(robot, enabled=False).drive(True, 1).moved)
        self.assertEqual(robot.speeds, [])


if __name__ == "__main__":
    unittest.main()
