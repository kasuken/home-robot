import time
import unittest

from laika.actions import ACTIONS, ActionContext, ActionPlayer, available_actions, validate_actions
from laika.prompts import build_system_prompt
from laika.robot import ALL_CAPABILITIES, CAMERA_PAN, DRIVE, SPEAKER, STEER, ULTRASONIC
from laika.safety import MovementGuard
from laika.simulator import SimulatedRobot
from tests.fakes import LookRecordingRobot, RecordingRobot

DRIVING = {"forward", "backward", "spin_left", "spin_right"}
PAN_GESTURES = {"look_left", "look_right", "shake_head", "look_forward", "look_around", "curious"}
STEERING_GESTURES = {"think", "wave", "celebrate", "dance"}


def robot_without(*missing):
    return SimulatedRobot(verbose=False, capabilities=ALL_CAPABILITIES - frozenset(missing))


def offered_by(robot, allow_driving=True):
    return available_actions(robot.identity().capabilities, allow_driving)


class ActionTests(unittest.TestCase):
    def test_unknown_actions_are_dropped_and_capped(self):
        names = ["NOD", "fly", "wave", "honk", "celebrate", "sad"]
        self.assertEqual(validate_actions(names, ACTIONS), ["nod", "wave", "honk", "celebrate"])

    def test_driving_actions_can_be_disabled(self):
        allowed = available_actions(ALL_CAPABILITIES, allow_driving=False)
        self.assertNotIn("forward", allowed)
        self.assertIn("nod", allowed)

    def test_full_robot_is_offered_every_action(self):
        self.assertEqual(set(offered_by(SimulatedRobot(verbose=False))), set(ACTIONS))

    def test_removing_camera_pan_hides_the_pan_gestures(self):
        offered = offered_by(robot_without(CAMERA_PAN))
        for name in PAN_GESTURES:
            self.assertNotIn(name, offered)
        self.assertIn("nod", offered)
        self.assertIn("look_up", offered)

    def test_removing_drive_hides_the_driving_actions(self):
        offered = offered_by(robot_without(DRIVE))
        for name in DRIVING:
            self.assertNotIn(name, offered)
        self.assertIn("nod", offered)

    def test_removing_speaker_hides_honk(self):
        offered = offered_by(robot_without(SPEAKER))
        self.assertNotIn("honk", offered)
        self.assertIn("nod", offered)

    def test_removing_ultrasonic_hides_only_forward(self):
        offered = offered_by(robot_without(ULTRASONIC))
        self.assertNotIn("forward", offered)
        self.assertIn("backward", offered)

    def test_removing_steer_hides_the_gestures_that_steer(self):
        offered = offered_by(robot_without(STEER))
        for name in STEERING_GESTURES | DRIVING:
            self.assertNotIn(name, offered)
        self.assertIn("look_left", offered)

    def test_hidden_actions_are_not_run_even_if_named(self):
        robot = robot_without(SPEAKER)
        player = ActionPlayer(ActionContext(robot, MovementGuard(robot)), offered_by(robot))
        player.play("honk", "nod")
        self.assertTrue(player.wait_done(timeout=5))
        self.assertEqual(robot.honks, 0)
        player.close()

    def test_system_prompt_mentions_only_offered_actions(self):
        robot = robot_without(CAMERA_PAN, DRIVE, SPEAKER)
        offered = offered_by(robot)
        prompt = build_system_prompt(offered, 4)
        for name in ACTIONS:
            line = f"  - {name}:"
            if name in offered:
                self.assertIn(line, prompt)
            else:
                self.assertNotIn(line, prompt)
        self.assertNotIn("movimento delle ruote", prompt)

    def test_player_runs_gestures_and_returns_to_rest(self):
        robot = RecordingRobot()
        player = ActionPlayer(ActionContext(robot, MovementGuard(robot)), ACTIONS)
        player.play("look_left", "honk", "look_forward")
        self.assertTrue(player.wait_done(timeout=5))
        self.assertEqual(robot.honks, 1)
        self.assertEqual((robot.pan, robot.tilt), (0, 0))
        player.close()

    def test_look_left_and_right_follow_the_pan_convention(self):
        # Negative pan is left, as in SunFounder's keyboard example ("j" turns the head left).
        robot = RecordingRobot()
        player = ActionPlayer(ActionContext(robot, MovementGuard(robot)), ACTIONS)
        player.play("look_left")
        self.assertTrue(player.wait_done(timeout=2))
        self.assertLess(robot.pan, 0)
        player.play("look_right")
        self.assertTrue(player.wait_done(timeout=2))
        self.assertGreater(robot.pan, 0)
        player.close()

    def test_dance_never_drives_and_ends_at_rest(self):
        robot = RecordingRobot()
        player = ActionPlayer(
            ActionContext(robot, MovementGuard(robot)), available_actions(ALL_CAPABILITIES, False)
        )
        player.play("dance")
        self.assertTrue(player.wait_done(timeout=10))
        self.assertEqual(robot.speeds, [])
        self.assertEqual((robot.pan, robot.tilt, robot.steering), (0, 0, 0))
        player.close()

    def test_talking_moves_the_head_a_little_then_settles(self):
        robot = LookRecordingRobot()
        player = ActionPlayer(ActionContext(robot, MovementGuard(robot)), ACTIONS)
        player.play("look_left")
        self.assertTrue(player.wait_done(timeout=2))
        base = robot.head_pose()
        robot.poses.clear()

        player.set_talking(True)
        time.sleep(1.2)
        player.set_talking(False)
        self.assertTrue(player.wait_done(timeout=2))

        self.assertGreater(len(robot.poses), 2)
        for pan, tilt in robot.poses:
            self.assertLessEqual(abs(pan - base[0]), 8)
            self.assertLessEqual(abs(tilt - base[1]), 6)
        self.assertEqual(robot.head_pose(), base)
        player.close()

    def test_a_gesture_cuts_into_the_talking_motion(self):
        robot = LookRecordingRobot()
        player = ActionPlayer(ActionContext(robot, MovementGuard(robot)), ACTIONS)
        player.set_talking(True)
        time.sleep(0.3)
        player.play("look_up")
        time.sleep(0.2)
        player.set_talking(False)
        self.assertTrue(player.wait_done(timeout=2))
        self.assertEqual(robot.head_pose(), (0, 35))
        player.close()

    def test_talking_motion_can_be_disabled(self):
        robot = LookRecordingRobot()
        player = ActionPlayer(ActionContext(robot, MovementGuard(robot)), ACTIONS, talk_motion=False)
        player.set_talking(True)
        time.sleep(0.5)
        player.set_talking(False)
        self.assertTrue(player.wait_done(timeout=1))
        self.assertEqual(robot.poses, [])
        player.close()

    def test_wait_done_returns_when_nothing_is_queued(self):
        robot = RecordingRobot()
        player = ActionPlayer(ActionContext(robot, MovementGuard(robot)), ACTIONS)
        player.play("not-a-gesture")
        self.assertTrue(player.wait_done(timeout=1))
        player.close()


if __name__ == "__main__":
    unittest.main()
