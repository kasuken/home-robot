"""The robot interface the rest of Laika talks to.

Only `laika.hardware` imports the SunFounder libraries. Everything else uses
this protocol, so the same code runs against the simulator on a laptop.
"""

from __future__ import annotations

from typing import Protocol

PAN_MIN, PAN_MAX = -90, 90
TILT_MIN, TILT_MAX = -35, 65
STEER_MIN, STEER_MAX = -30, 30
SPEED_MAX = 100


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class Robot(Protocol):
    def look(self, pan: float, tilt: float) -> None:
        """Point the camera head. Pan is left (-) / right (+), tilt is down (-) / up (+)."""

    def head_pose(self) -> tuple[float, float]:
        """The last (pan, tilt) the head was pointed at."""

    def steer(self, angle: float) -> None:
        """Turn the front wheels. Negative is left."""

    def drive(self, speed: int) -> None:
        """Start the rear motors. Positive is forward, negative is backward, 0 stops."""

    def stop(self) -> None:
        """Stop the motors immediately."""

    def distance_cm(self) -> float | None:
        """Ultrasonic distance ahead, or None when the reading is invalid."""

    def cliff_detected(self) -> bool:
        """True when the grayscale sensors see an edge below the robot."""

    def honk(self, volume: int) -> None:
        """Play the horn sound without blocking."""

    def reset(self) -> None:
        """Stop, centre the wheels and the head."""

    def close(self) -> None:
        """Leave the hardware in a safe, stopped state."""
