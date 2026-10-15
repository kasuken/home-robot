"""The PiCar-X adapter: the only module that calls the SunFounder hardware APIs."""

from __future__ import annotations

import io
import threading
from typing import Any

from laika.audio import Mouth
from laika.config import REPO_ROOT
from laika.robot import (
    PAN_MAX,
    PAN_MIN,
    SPEED_MAX,
    STEER_MAX,
    STEER_MIN,
    TILT_MAX,
    TILT_MIN,
    ImageObservation,
    RobotSensorState,
    RobotState,
    clamp,
)

HORN_SOUND = REPO_ROOT / "references" / "sunfounder" / "picar-x" / "sounds" / "car-double-horn.wav"
IMAGE_SIZE = (640, 480)


class PiCarXRobot:
    def __init__(self, mouth: Mouth | None = None) -> None:
        # Imported here so the rest of Laika runs on machines without the Robot HAT.
        from picarx import Picarx
        from robot_hat.music import Music

        self._car = Picarx()
        self._music = Music()
        self._mouth = mouth
        self._camera: Any = None
        self._pose = (0.0, 0.0)
        self._steering = 0.0
        self._speed = 0
        self._lock = threading.RLock()
        self._camera_lock = threading.Lock()
        self.reset()

    def get_sensors(self) -> RobotSensorState:
        with self._lock:
            distance = self._car.get_distance()
            grayscale = self._car.get_grayscale_data()
            cliff = bool(self._car.get_cliff_status(grayscale))
        # The sensor returns -1 or -2 on timeouts and other failed reads.
        valid = distance if distance is not None and distance > 0 else None
        return RobotSensorState(
            distance_cm=valid,
            cliff_detected=cliff,
            grayscale=tuple(int(v) for v in grayscale[:3]),
        )

    def get_state(self) -> RobotState:
        with self._lock:
            return RobotState(
                head_pan=self._pose[0],
                head_tilt=self._pose[1],
                steering=self._steering,
                speed=self._speed,
                sensors=self.get_sensors(),
            )

    def look(self, pan: float, tilt: float) -> None:
        with self._lock:
            self._pose = (clamp(pan, PAN_MIN, PAN_MAX), clamp(tilt, TILT_MIN, TILT_MAX))
            self._car.set_cam_pan_angle(self._pose[0])
            self._car.set_cam_tilt_angle(self._pose[1])

    def capture_image(self) -> ImageObservation:
        # The camera has its own lock: starting it and capturing take long enough
        # that holding the hardware lock would stall the safety checks.
        with self._camera_lock:
            if self._camera is None:
                from picamera2 import Picamera2

                camera = Picamera2()
                camera.configure(camera.create_still_configuration(main={"size": IMAGE_SIZE}))
                camera.start()
                self._camera = camera
            pan, tilt = self.head_pose()
            buffer = io.BytesIO()
            self._camera.capture_file(buffer, format="jpeg")
        return ImageObservation(
            jpeg=buffer.getvalue(),
            width=IMAGE_SIZE[0],
            height=IMAGE_SIZE[1],
            camera_pan=pan,
            camera_tilt=tilt,
        )

    def speak(self, text: str) -> None:
        # Not under the lock: speech takes seconds and must not block the safety loop.
        if self._mouth is not None:
            self._mouth.say(text)
        else:
            print(f"Laika: {text}")

    def head_pose(self) -> tuple[float, float]:
        with self._lock:
            return self._pose

    def steer(self, angle: float) -> None:
        with self._lock:
            self._steering = clamp(angle, STEER_MIN, STEER_MAX)
            self._car.set_dir_servo_angle(self._steering)

    def drive(self, speed: int) -> None:
        speed = int(clamp(speed, -SPEED_MAX, SPEED_MAX))
        with self._lock:
            if speed > 0:
                self._car.forward(speed)
            elif speed < 0:
                self._car.backward(-speed)
            else:
                self._car.stop()
            self._speed = speed

    def stop(self) -> None:
        with self._lock:
            self._car.stop()
            self._speed = 0

    def distance_cm(self) -> float | None:
        return self.get_sensors().distance_cm

    def cliff_detected(self) -> bool:
        with self._lock:
            return bool(self._car.get_cliff_status(self._car.get_grayscale_data()))

    def honk(self, volume: int) -> None:
        with self._lock:
            self._music.sound_play_threading(str(HORN_SOUND), volume)

    def reset(self) -> None:
        with self._lock:
            self._car.reset()
            self._pose = (0.0, 0.0)
            self._steering = 0.0
            self._speed = 0

    def close(self) -> None:
        with self._lock:
            self._car.close()
            self._pose = (0.0, 0.0)
            self._steering = 0.0
            self._speed = 0
        with self._camera_lock:
            if self._camera is not None:
                try:
                    self._camera.close()
                finally:
                    self._camera = None
