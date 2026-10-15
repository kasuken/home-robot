"""One Robot contract, run against every implementation.

PiCarXRobot runs here with fake `picarx`, `robot_hat` and `picamera2` modules,
so both implementations are checked on a laptop.
"""

import sys
import threading
import types
import unittest
from unittest import mock

from laika.commands import Telemetry
from laika.house import default_house
from laika.robot import (
    ALL_CAPABILITIES,
    CAMERA_PAN,
    DRIVE,
    HardwareError,
    ImageObservation,
    RobotIdentity,
    RobotSensorState,
    RobotState,
)
from laika.simulator import SimulatedRobot


class SpyMouth:
    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)


class FakePicarx:
    def __init__(self):
        self.calls = []
        self.distance = 80.0
        self.gray = [1000, 1000, 1000]

    def set_cam_pan_angle(self, angle):
        self.calls.append(("pan", angle))

    def set_cam_tilt_angle(self, angle):
        self.calls.append(("tilt", angle))

    def set_dir_servo_angle(self, angle):
        self.calls.append(("steer", angle))

    def forward(self, speed):
        self.calls.append(("forward", speed))

    def backward(self, speed):
        self.calls.append(("backward", speed))

    def stop(self):
        self.calls.append(("stop", 0))

    def get_distance(self):
        return self.distance

    def get_grayscale_data(self):
        return list(self.gray)

    def get_cliff_status(self, data):
        return min(data) < 200

    def reset(self):
        self.calls.append(("reset", 0))

    def close(self):
        self.calls.append(("close", 0))


class FakeMusic:
    def sound_play_threading(self, path, volume):
        pass


class FakeCamera:
    started = 0
    closed = 0
    fail_capture = False

    def __init__(self):
        self.options = {}
        self.configured_size = None

    def create_still_configuration(self, main):
        return {"main": main}

    def configure(self, config):
        self.configured_size = config["main"]["size"]

    def start(self):
        FakeCamera.started += 1

    def capture_file(self, buffer, format):
        if FakeCamera.fail_capture:
            raise OSError("camera busy")
        buffer.write(b"\xff\xd8fake\xff\xd9")

    def close(self):
        FakeCamera.closed += 1


def fake_sunfounder_modules():
    return {
        "picarx": types.SimpleNamespace(Picarx=FakePicarx),
        "robot_hat": types.ModuleType("robot_hat"),
        "robot_hat.music": types.SimpleNamespace(Music=FakeMusic),
        "picamera2": types.SimpleNamespace(Picamera2=FakeCamera),
    }


class RobotContract:
    """Mixin: subclasses set `self.robot` and `self.mouth`, and implement the two setters."""

    def set_distance(self, value):
        raise NotImplementedError

    def set_cliff(self, value):
        raise NotImplementedError

    def test_look_is_clamped_and_reported(self):
        self.robot.look(500, -500)
        pan, tilt = self.robot.head_pose()
        self.assertLessEqual(pan, 90)
        self.assertGreaterEqual(tilt, -35)
        state = self.robot.get_state()
        self.assertEqual((state.head_pan, state.head_tilt), (pan, tilt))

    def test_state_follows_the_actuators(self):
        self.robot.steer(100)
        self.robot.drive(30)
        state = self.robot.get_state()
        self.assertIsInstance(state, RobotState)
        self.assertEqual(state.steering, 30)
        self.assertEqual(state.speed, 30)
        self.robot.stop()
        self.assertEqual(self.robot.get_state().speed, 0)

    def test_sensors(self):
        self.set_distance(42.0)
        self.set_cliff(False)
        sensors = self.robot.get_sensors()
        self.assertIsInstance(sensors, RobotSensorState)
        self.assertEqual(sensors.distance_cm, 42.0)
        self.assertFalse(sensors.cliff_detected)
        self.assertEqual(len(sensors.grayscale), 3)
        self.assertEqual(self.robot.distance_cm(), 42.0)
        self.assertFalse(self.robot.cliff_detected())

    def test_cliff_and_invalid_distance(self):
        self.set_distance(None)
        self.set_cliff(True)
        sensors = self.robot.get_sensors()
        self.assertIsNone(sensors.distance_cm)
        self.assertTrue(sensors.cliff_detected)
        self.assertTrue(self.robot.cliff_detected())

    def test_state_converts_to_telemetry(self):
        self.robot.look(10, 5)
        telemetry = self.robot.get_state().to_telemetry()
        self.assertIsInstance(telemetry, Telemetry)
        self.assertEqual((telemetry.camera_pan, telemetry.camera_tilt), (10, 5))
        self.assertEqual(Telemetry.from_json(telemetry.to_json()), telemetry)

    def test_capture_image_reports_the_camera_pose(self):
        self.robot.look(20, 10)
        image = self.robot.capture_image()
        self.assertIsInstance(image, ImageObservation)
        self.assertGreater(image.width, 0)
        self.assertGreater(image.height, 0)
        self.assertEqual((image.camera_pan, image.camera_tilt), (20, 10))
        self.assertTrue(image.jpeg is None or image.jpeg.startswith(b"\xff\xd8"))

    def test_speak_goes_through_the_mouth(self):
        self.robot.speak("ciao")
        self.assertEqual(self.mouth.said, ["ciao"])

    def test_reset_centres_everything(self):
        self.robot.look(40, 20)
        self.robot.steer(20)
        self.robot.drive(20)
        self.robot.reset()
        state = self.robot.get_state()
        self.assertEqual((state.head_pan, state.head_tilt, state.steering, state.speed), (0, 0, 0, 0))

    def test_honk_does_not_raise(self):
        self.robot.honk(50)

    def test_concurrent_use_is_safe(self):
        errors = []

        def work(offset):
            try:
                for i in range(50):
                    self.robot.look(offset + i % 10, 0)
                    self.robot.get_state()
            except Exception as error:
                errors.append(error)

        threads = [threading.Thread(target=work, args=(n * 10,)) for n in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])

    def test_close_leaves_the_robot_stopped(self):
        self.robot.drive(20)
        self.robot.close()
        self.assertEqual(self.robot.get_state().speed, 0)

    def test_identity_advertises_capabilities(self):
        identity = self.robot.identity()
        self.assertIsInstance(identity, RobotIdentity)
        self.assertTrue(identity.robot_id)
        self.assertTrue(identity.name)
        self.assertTrue(identity.version)
        self.assertIsInstance(identity.capabilities, frozenset)
        self.assertTrue(identity.capabilities)

    def test_identity_serialises_to_the_prd_shape(self):
        data = self.robot.identity().to_dict()
        self.assertEqual(set(data), {"robotId", "name", "version", "capabilities"})
        self.assertEqual(data["capabilities"], sorted(data["capabilities"]))


class SimulatedRobotContractTests(RobotContract, unittest.TestCase):
    def setUp(self):
        self.mouth = SpyMouth()
        self.robot = SimulatedRobot(verbose=False, mouth=self.mouth)

    def set_distance(self, value):
        self.robot.distance = value

    def set_cliff(self, value):
        self.robot.cliff = value

    def test_simulator_reports_the_capabilities_it_was_given(self):
        self.assertEqual(self.robot.identity().robot_id, "laika-sim")
        self.assertEqual(self.robot.identity().capabilities, ALL_CAPABILITIES)
        limited = SimulatedRobot(verbose=False, capabilities=ALL_CAPABILITIES - {CAMERA_PAN})
        self.assertNotIn(CAMERA_PAN, limited.identity().capabilities)
        self.assertIn(DRIVE, limited.identity().capabilities)


class HouseSimulatedRobotContractTests(RobotContract, unittest.TestCase):
    def setUp(self):
        self.mouth = SpyMouth()
        self.robot = SimulatedRobot(verbose=False, mouth=self.mouth, house=default_house())

    def set_distance(self, value):
        self.robot.distance = value

    def set_cliff(self, value):
        self.robot.cliff = value


class PiCarXRobotContractTests(RobotContract, unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(sys.modules, fake_sunfounder_modules())
        patcher.start()
        self.addCleanup(patcher.stop)
        from laika.hardware import PiCarXRobot

        FakeCamera.started = FakeCamera.closed = 0
        FakeCamera.fail_capture = False
        self.mouth = SpyMouth()
        self.robot = PiCarXRobot(mouth=self.mouth)
        self.car = self.robot._car

    def set_distance(self, value):
        self.car.distance = -1 if value is None else value

    def set_cliff(self, value):
        self.car.gray = [50, 50, 50] if value else [1000, 1000, 1000]

    def test_advertises_every_capability(self):
        self.assertEqual(self.robot.identity().robot_id, "laika-01")
        self.assertEqual(self.robot.identity().capabilities, ALL_CAPABILITIES)

    def test_camera_starts_once_and_closes_with_the_robot(self):
        self.robot.capture_image()
        self.robot.capture_image()
        self.assertEqual(FakeCamera.started, 1)
        self.robot.close()
        self.assertEqual(FakeCamera.closed, 1)

    def break_car(self, *names):
        def boom(*args):
            raise OSError("i2c failure")

        for name in names:
            setattr(self.car, name, boom)

    def test_invalid_distance_readings_are_none(self):
        for value in (-1, -2, 0, None):
            self.car.distance = value
            self.assertIsNone(self.robot.distance_cm())

    def test_vendor_errors_become_hardware_errors(self):
        self.break_car(
            "set_dir_servo_angle",
            "set_cam_pan_angle",
            "forward",
            "backward",
            "stop",
            "get_distance",
            "get_grayscale_data",
        )
        for action in (
            lambda: self.robot.steer(10),
            lambda: self.robot.look(10, 10),
            lambda: self.robot.drive(20),
            lambda: self.robot.drive(-20),
            self.robot.stop,
            self.robot.get_sensors,
            self.robot.get_state,
            self.robot.distance_cm,
            self.robot.cliff_detected,
        ):
            with self.assertRaises(HardwareError):
                action()

    def test_failed_calls_do_not_change_the_reported_state(self):
        self.robot.steer(10)
        self.break_car("set_dir_servo_angle", "forward")
        with self.assertRaises(HardwareError):
            self.robot.steer(-20)
        with self.assertRaises(HardwareError):
            self.robot.drive(30)
        self.assertEqual(self.robot._steering, 10)
        self.assertEqual(self.robot._speed, 0)

    def test_reset_and_close_never_raise_and_try_every_part(self):
        self.break_car("stop", "set_dir_servo_angle", "set_cam_tilt_angle", "close")
        self.robot.reset()
        self.assertIn(("pan", 0), self.car.calls)  # the part after the failing ones still ran
        self.robot.close()
        self.robot.close()

    def test_close_releases_the_camera_even_if_the_car_fails(self):
        self.robot.capture_image()
        self.break_car("close")
        self.robot.close()
        self.assertEqual(FakeCamera.closed, 1)

    def test_camera_failure_is_a_hardware_error_and_the_camera_restarts(self):
        self.robot.capture_image()
        FakeCamera.fail_capture = True
        with self.assertRaises(HardwareError):
            self.robot.capture_image()
        self.assertEqual(FakeCamera.closed, 1)
        FakeCamera.fail_capture = False
        self.assertTrue(self.robot.capture_image().jpeg)
        self.assertEqual(FakeCamera.started, 2)

    def test_horn_failure_is_a_hardware_error(self):
        def boom(*args):
            raise OSError("no sound card")

        self.robot._music.sound_play_threading = boom
        with self.assertRaises(HardwareError):
            self.robot.honk(50)


if __name__ == "__main__":
    unittest.main()
