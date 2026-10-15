"""A stand-in robot that prints what the real one would do."""

from __future__ import annotations

from laika.robot import PAN_MAX, PAN_MIN, SPEED_MAX, STEER_MAX, STEER_MIN, TILT_MAX, TILT_MIN, clamp


class SimulatedRobot:
    def __init__(self, distance_cm: float | None = 100.0, verbose: bool = True) -> None:
        self.pan = 0.0
        self.tilt = 0.0
        self.steering = 0.0
        self.speed = 0
        self.distance = distance_cm
        self.cliff = False
        self.honks = 0
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"  [robot] {message}")

    def look(self, pan: float, tilt: float) -> None:
        self.pan = clamp(pan, PAN_MIN, PAN_MAX)
        self.tilt = clamp(tilt, TILT_MIN, TILT_MAX)
        # Not logged: gestures move the head in many small steps.

    def head_pose(self) -> tuple[float, float]:
        return (self.pan, self.tilt)

    def steer(self, angle: float) -> None:
        self.steering = clamp(angle, STEER_MIN, STEER_MAX)

    def drive(self, speed: int) -> None:
        self.speed = int(clamp(speed, -SPEED_MAX, SPEED_MAX))
        self._log(f"drive speed={self.speed}")

    def stop(self) -> None:
        if self.speed:
            self._log("stop")
        self.speed = 0

    def distance_cm(self) -> float | None:
        return self.distance

    def cliff_detected(self) -> bool:
        return self.cliff

    def honk(self, volume: int) -> None:
        self.honks += 1
        self._log("BEEP BEEP")

    def reset(self) -> None:
        self.stop()
        self.steer(0)
        self.look(0, 0)

    def close(self) -> None:
        self.reset()
        self._log("closed")
