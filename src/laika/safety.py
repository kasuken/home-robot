"""Local movement checks. The AI asks to move; this module decides whether it may.

Prototype scope: short, slow drives only, checked against the ultrasonic
sensor and the cliff sensors before and during the move. There is no rear
sensor, so backward moves are kept shorter.

`execute` is the movement entry point: it takes a typed `MoveCommand` and is
the only code that should call the robot's `steer`/`drive` for wheel movement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from laika.commands import CommandResult, Direction, MoveCommand, ResultStatus
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

    def execute(self, command: MoveCommand) -> CommandResult:
        """Run a move if the checks allow it.

        SUCCESS: the whole move ran. REJECTED: refused before the wheels turned.
        CANCELLED: stopped early by an obstacle or a cliff.
        """
        speed = min(command.speed, self.speed)
        result = self._drive(
            command.direction is Direction.FORWARD,
            command.duration_ms / 1000,
            command.steering,
            speed,
        )
        if not result.moved:
            return CommandResult(command.id, ResultStatus.REJECTED, result.reason)
        if result.reason:
            return CommandResult(command.id, ResultStatus.CANCELLED, result.reason)
        return CommandResult(command.id, ResultStatus.SUCCESS)

    def drive(self, forward: bool, duration_s: float, steer: float = 0) -> DriveResult:
        return self._drive(forward, duration_s, steer, self.speed)

    def _drive(self, forward: bool, duration_s: float, steer: float, speed: int) -> DriveResult:
        if not self.enabled or speed <= 0:
            return DriveResult(False, "disabled")

        limit = self.max_duration_s if forward else min(self.max_duration_s, BACKWARD_MAX_S)
        duration_s = max(0.0, min(duration_s, limit))
        reason = self._blocked(forward)
        if reason:
            return DriveResult(False, reason)

        self.robot.steer(steer)
        deadline = time.monotonic() + duration_s
        try:
            self.robot.drive(speed if forward else -speed)
            while time.monotonic() < deadline:
                time.sleep(POLL_INTERVAL_S)
                reason = self._blocked(forward)
                if reason:
                    return DriveResult(True, reason)
            return DriveResult(True)
        finally:
            self.robot.stop()
            self.robot.steer(0)
