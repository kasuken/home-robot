"""Physical limits shared by the robot interface and the command validators."""

from __future__ import annotations

PAN_MIN, PAN_MAX = -90, 90
TILT_MIN, TILT_MAX = -35, 65
STEER_MIN, STEER_MAX = -30, 30
SPEED_MAX = 100


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
