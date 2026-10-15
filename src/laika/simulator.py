"""A stand-in robot that prints what the real one would do."""

from __future__ import annotations

import threading

from laika.audio import Mouth
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

IMAGE_SIZE = (640, 480)
GRAYSCALE_FLOOR = (1000, 1000, 1000)
GRAYSCALE_EDGE = (50, 50, 50)


class SimulatedRobot:
    def __init__(
        self,
        distance_cm: float | None = 100.0,
        verbose: bool = True,
        mouth: Mouth | None = None,
    ) -> None:
        self.pan = 0.0
        self.tilt = 0.0
        self.steering = 0.0
        self.speed = 0
        self.distance = distance_cm
        self.cliff = False
        self.honks = 0
        self.spoken: list[str] = []
        self.verbose = verbose
        self._mouth = mouth
        self._lock = threading.RLock()

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"  [robot] {message}")

    def get_sensors(self) -> RobotSensorState:
        with self._lock:
            return RobotSensorState(
                distance_cm=self.distance,
                cliff_detected=self.cliff,
                grayscale=GRAYSCALE_EDGE if self.cliff else GRAYSCALE_FLOOR,
            )

    def get_state(self) -> RobotState:
        with self._lock:
            return RobotState(
                head_pan=self.pan,
                head_tilt=self.tilt,
                steering=self.steering,
                speed=self.speed,
                sensors=self.get_sensors(),
            )

    def look(self, pan: float, tilt: float) -> None:
        with self._lock:
            self.pan = clamp(pan, PAN_MIN, PAN_MAX)
            self.tilt = clamp(tilt, TILT_MIN, TILT_MAX)
        # Not logged: gestures move the head in many small steps.

    def capture_image(self) -> ImageObservation:
        with self._lock:
            return ImageObservation(
                jpeg=None,
                width=IMAGE_SIZE[0],
                height=IMAGE_SIZE[1],
                camera_pan=self.pan,
                camera_tilt=self.tilt,
            )

    def speak(self, text: str) -> None:
        # Not under the lock: speech takes seconds and must not block the safety loop.
        self.spoken.append(text)
        if self._mouth is not None:
            self._mouth.say(text)
        else:
            print(f"Laika: {text}")

    def head_pose(self) -> tuple[float, float]:
        with self._lock:
            return (self.pan, self.tilt)

    def steer(self, angle: float) -> None:
        with self._lock:
            self.steering = clamp(angle, STEER_MIN, STEER_MAX)

    def drive(self, speed: int) -> None:
        with self._lock:
            self.speed = int(clamp(speed, -SPEED_MAX, SPEED_MAX))
            self._log(f"drive speed={self.speed}")

    def stop(self) -> None:
        with self._lock:
            if self.speed:
                self._log("stop")
            self.speed = 0

    def distance_cm(self) -> float | None:
        with self._lock:
            return self.distance

    def cliff_detected(self) -> bool:
        with self._lock:
            return self.cliff

    def honk(self, volume: int) -> None:
        with self._lock:
            self.honks += 1
            self._log("BEEP BEEP")

    def reset(self) -> None:
        with self._lock:
            self.stop()
            self.steer(0)
            self.look(0, 0)

    def close(self) -> None:
        with self._lock:
            self.reset()
            self._log("closed")
