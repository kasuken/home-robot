"""The PiCar-X adapter: the only module that calls the SunFounder hardware APIs."""

from __future__ import annotations

from laika.config import REPO_ROOT
from laika.robot import SPEED_MAX, clamp

HORN_SOUND = REPO_ROOT / "references" / "sunfounder" / "picar-x" / "sounds" / "car-double-horn.wav"


class PiCarXRobot:
    def __init__(self) -> None:
        # Imported here so the rest of Laika runs on machines without the Robot HAT.
        from picarx import Picarx
        from robot_hat.music import Music

        self._car = Picarx()
        self._music = Music()
        self.reset()

    def look(self, pan: float, tilt: float) -> None:
        self._car.set_cam_pan_angle(pan)
        self._car.set_cam_tilt_angle(tilt)

    def steer(self, angle: float) -> None:
        self._car.set_dir_servo_angle(angle)

    def drive(self, speed: int) -> None:
        speed = int(clamp(speed, -SPEED_MAX, SPEED_MAX))
        if speed > 0:
            self._car.forward(speed)
        elif speed < 0:
            self._car.backward(-speed)
        else:
            self._car.stop()

    def stop(self) -> None:
        self._car.stop()

    def distance_cm(self) -> float | None:
        distance = self._car.get_distance()
        # The sensor returns -1 or -2 on timeouts and other failed reads.
        return distance if distance is not None and distance > 0 else None

    def cliff_detected(self) -> bool:
        return self._car.get_cliff_status(self._car.get_grayscale_data())

    def honk(self, volume: int) -> None:
        self._music.sound_play_threading(str(HORN_SOUND), volume)

    def reset(self) -> None:
        self._car.reset()

    def close(self) -> None:
        self._car.close()
