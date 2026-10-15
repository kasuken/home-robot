import unittest

from laika.actions import ACTIONS, ActionContext, ActionPlayer, available_actions, validate_actions
from laika.safety import MovementGuard
from laika.simulator import SimulatedRobot


class RecordingRobot(SimulatedRobot):
    def __init__(self, **kwargs):
        super().__init__(verbose=False, **kwargs)
        self.speeds = []

    def drive(self, speed):
        super().drive(speed)
        self.speeds.append(self.speed)


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


class ActionTests(unittest.TestCase):
    def test_unknown_actions_are_dropped_and_capped(self):
        names = ["NOD", "fly", "wave", "honk", "celebrate", "sad"]
        self.assertEqual(validate_actions(names, ACTIONS), ["nod", "wave", "honk", "celebrate"])

    def test_driving_actions_can_be_disabled(self):
        allowed = available_actions(allow_driving=False)
        self.assertNotIn("forward", allowed)
        self.assertIn("nod", allowed)

    def test_player_runs_gestures_and_returns_to_rest(self):
        robot = RecordingRobot()
        player = ActionPlayer(ActionContext(robot, MovementGuard(robot)), ACTIONS)
        player.play("look_left", "honk", "look_forward")
        self.assertTrue(player.wait_done(timeout=5))
        self.assertEqual(robot.honks, 1)
        self.assertEqual((robot.pan, robot.tilt), (0, 0))
        player.close()

    def test_wait_done_returns_when_nothing_is_queued(self):
        robot = RecordingRobot()
        player = ActionPlayer(ActionContext(robot, MovementGuard(robot)), ACTIONS)
        player.play("not-a-gesture")
        self.assertTrue(player.wait_done(timeout=1))
        player.close()


if __name__ == "__main__":
    unittest.main()
