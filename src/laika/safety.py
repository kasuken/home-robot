"""Local movement checks. The AI asks to move; this module decides whether it may.

Prototype scope: short, slow drives only, checked against the ultrasonic
sensor and the cliff sensors before and during the move. There is no rear
sensor, so backward moves are kept shorter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from laika.robot import Robot

POLL_INTERVAL_S = 0.05
BACKWARD_MAX_S = 0.6


@dataclass
class DriveResult:
    moved: bool
    reason: str = ""


class MovementGuard:
    def __init__(
        self,
        robot: Robot,
        *,
        enabled: bool = True,
        speed: int = 25,
        max_duration_s: float = 1.5,
        min_front_distance_cm: float = 25.0,
    ) -> None:
        self.robot = robot
        self.enabled = enabled
        self.speed = max(0, min(speed, 40))
        self.max_duration_s = max_duration_s
        self.min_front_distance_cm = min_front_distance_cm

    def _blocked(self, forward: bool) -> str:
        if self.robot.cliff_detected():
            return "cliff"
        if forward:
            distance = self.robot.distance_cm()
            if distance is not None and distance < self.min_front_distance_cm:
                return "obstacle"
        return ""

    def drive(self, forward: bool, duration_s: float, steer: float = 0) -> DriveResult:
        if not self.enabled:
            return DriveResult(False, "disabled")

        limit = self.max_duration_s if forward else min(self.max_duration_s, BACKWARD_MAX_S)
        duration_s = max(0.0, min(duration_s, limit))
        reason = self._blocked(forward)
        if reason:
            return DriveResult(False, reason)

        self.robot.steer(steer)
        deadline = time.monotonic() + duration_s
        try:
            self.robot.drive(self.speed if forward else -self.speed)
            while time.monotonic() < deadline:
                time.sleep(POLL_INTERVAL_S)
                reason = self._blocked(forward)
                if reason:
                    return DriveResult(True, reason)
            return DriveResult(True)
        finally:
            self.robot.stop()
            self.robot.steer(0)
