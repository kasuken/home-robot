import unittest

from laika.commands import CommandResult, Direction, MoveCommand, ResultStatus
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

    def test_execute_succeeds_and_reports_the_command_id(self):
        robot = RecordingRobot(distance_cm=100)
        command = MoveCommand(direction=Direction.FORWARD, speed=30, duration_ms=50, steering=10)
        result = MovementGuard(robot, speed=25).execute(command)
        self.assertEqual(result, CommandResult(command.id, ResultStatus.SUCCESS))
        self.assertEqual(robot.speeds, [25])
        self.assertEqual((robot.speed, robot.steering), (0, 0))

    def test_execute_rejects_a_blocked_move_with_the_reason(self):
        robot = RecordingRobot(distance_cm=10)
        command = MoveCommand(direction=Direction.FORWARD, speed=20, duration_ms=500)
        result = MovementGuard(robot).execute(command)
        self.assertEqual((result.result, result.reason), (ResultStatus.REJECTED, "obstacle"))
        self.assertEqual(robot.speeds, [])

    def test_execute_rejects_when_driving_is_disabled(self):
        robot = RecordingRobot()
        command = MoveCommand(direction=Direction.BACKWARD, speed=20, duration_ms=500)
        result = MovementGuard(robot, enabled=False).execute(command)
        self.assertEqual((result.result, result.reason), (ResultStatus.REJECTED, "disabled"))

    def test_execute_cancels_a_move_cut_short_by_a_cliff(self):
        class CliffAfterStart(RecordingRobot):
            def cliff_detected(self):
                return bool(self.speeds)

        robot = CliffAfterStart(distance_cm=100)
        command = MoveCommand(direction=Direction.FORWARD, speed=20, duration_ms=1000)
        result = MovementGuard(robot).execute(command)
        self.assertEqual((result.result, result.reason), (ResultStatus.CANCELLED, "cliff"))
        self.assertEqual(robot.speed, 0)

    def test_execute_drives_backward_with_negative_speed(self):
        robot = RecordingRobot()
        command = MoveCommand(direction=Direction.BACKWARD, speed=20, duration_ms=50)
        MovementGuard(robot).execute(command)
        self.assertEqual(robot.speeds[0], -20)


if __name__ == "__main__":
    unittest.main()
