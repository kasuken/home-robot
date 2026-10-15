"""The robot interface the rest of Laika talks to.

Only two adapters import SunFounder libraries, both lazily: `laika.hardware`
(motors, servos, sensors, camera) and `laika.audio` (the vendor's speech
recognition and voices). Everything else uses this protocol, so the same
code runs against the simulator on a laptop.

Implementations are thread-safe: the gesture worker, the safety layer and the
conversation call them concurrently (see docs/decisions/0001-threads-not-asyncio.md).

There is deliberately no `move()`. `steer`, `drive` and `stop` are low-level
primitives that only the safety layer (`SafetyController.execute`) may use for
wheel movement; everything else asks the safety controller with a `MovementCommand`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from laika.commands import MoveCommand, Telemetry
from laika.limits import (
    PAN_MAX,
    PAN_MIN,
    SPEED_MAX,
    STEER_MAX,
    STEER_MIN,
    TILT_MAX,
    TILT_MIN,
    clamp,
)

__all__ = [
    "ALL_CAPABILITIES",
    "BATTERY",
    "CAMERA",
    "CAMERA_PAN",
    "CAMERA_TILT",
    "DRIVE",
    "GRAYSCALE",
    "HardwareError",
    "PAN_MAX",
    "PAN_MIN",
    "SPEAKER",
    "SPEED_MAX",
    "STEER",
    "STEER_MAX",
    "STEER_MIN",
    "TILT_MAX",
    "TILT_MIN",
    "ULTRASONIC",
    "clamp",
    "ImageObservation",
    "MovementCommand",
    "Robot",
    "RobotIdentity",
    "RobotSensorState",
    "RobotState",
]

# One movement type for the whole system: the typed command from `laika.commands`.
MovementCommand = MoveCommand

# Capability names from docs/prd.md. A robot advertises the ones it has.
DRIVE = "drive"
STEER = "steer"
CAMERA = "camera"
CAMERA_PAN = "camera-pan"
CAMERA_TILT = "camera-tilt"
ULTRASONIC = "ultrasonic"
GRAYSCALE = "grayscale"
SPEAKER = "speaker"
BATTERY = "battery"
ALL_CAPABILITIES = frozenset(
    {DRIVE, STEER, CAMERA, CAMERA_PAN, CAMERA_TILT, ULTRASONIC, GRAYSCALE, SPEAKER, BATTERY}
)


class HardwareError(RuntimeError):
    """A robot call failed because the hardware (or the simulated hardware) did."""


@dataclass(frozen=True)
class RobotIdentity:
    """Who the robot is and what its hardware can do, so the agent never assumes it."""

    robot_id: str
    name: str
    version: str
    capabilities: frozenset[str]

    def to_dict(self) -> dict[str, object]:
        """The PRD's robot metadata shape."""
        return {
            "robotId": self.robot_id,
            "name": self.name,
            "version": self.version,
            "capabilities": sorted(self.capabilities),
        }


@dataclass(frozen=True)
class RobotSensorState:
    """One reading of the sensors. `distance_cm` is None when the reading is invalid."""

    distance_cm: float | None
    cliff_detected: bool
    grayscale: tuple[int, int, int] = (0, 0, 0)
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class RobotState:
    """The actuators plus a sensor reading, taken together."""

    head_pan: float
    head_tilt: float
    steering: float
    speed: int
    sensors: RobotSensorState
    timestamp: float = field(default_factory=time.time)
    # The raw battery voltage, None when the robot cannot read it. Read here, not with the sensors:
    # the safety loop reads the sensors 20 times a second and must not touch the ADC for it.
    battery_v: float | None = None

    def to_telemetry(self) -> Telemetry:
        return Telemetry(
            distance_cm=self.sensors.distance_cm,
            cliff_detected=self.sensors.cliff_detected,
            camera_pan=self.head_pan,
            camera_tilt=self.head_tilt,
            steering=self.steering,
            speed=self.speed,
            timestamp=self.timestamp,
            battery_v=self.battery_v,
        )


@dataclass(frozen=True)
class ImageObservation:
    """A camera frame. `jpeg` is None when the robot has no real camera (the simulator).

    `scene` is simulator ground truth: the names of the things and people in view.
    `people` is the subset of `scene` that are people and `simulated` marks a frame whose
    `scene` is trustworthy ground truth. The real robot leaves all three at their defaults.
    """

    jpeg: bytes | None
    width: int
    height: int
    camera_pan: float
    camera_tilt: float
    timestamp: float = field(default_factory=time.time)
    scene: tuple[str, ...] = ()
    people: tuple[str, ...] = ()
    simulated: bool = False


class Robot(Protocol):
    def identity(self) -> RobotIdentity:
        """Id, name, version and the capabilities this robot advertises."""

    def get_state(self) -> RobotState:
        """Head, steering and motor state plus a fresh sensor reading."""

    def get_sensors(self) -> RobotSensorState:
        """Read the distance, cliff and grayscale sensors."""

    def look(self, pan: float, tilt: float) -> None:
        """Point the camera head. Pan is left (-) / right (+), tilt is down (-) / up (+)."""

    def capture_image(self) -> ImageObservation:
        """Take one camera frame. Blocks for the capture."""

    def speak(self, text: str) -> None:
        """Say `text` out loud. Blocks until the speech is over."""

    def head_pose(self) -> tuple[float, float]:
        """The last (pan, tilt) the head was pointed at."""

    def steer(self, angle: float) -> None:
        """Turn the front wheels. Negative is left."""

    def drive(self, speed: int) -> None:
        """Start the rear motors. Positive is forward, negative is backward, 0 stops.

        Low level: only the safety layer calls this for wheel movement.
        """

    def stop(self) -> None:
        """Stop the motors immediately."""

    def distance_cm(self) -> float | None:
        """Ultrasonic distance ahead, or None when the reading is invalid."""

    def cliff_detected(self) -> bool:
        """True when the grayscale sensors see an edge below the robot."""

    def battery_voltage(self) -> float | None:
        """The battery pack voltage in volts, or None when it cannot be read. Never raises."""

    def honk(self, volume: int) -> None:
        """Play the horn sound without blocking."""

    def reset(self) -> None:
        """Stop, centre the wheels and the head."""

    def close(self) -> None:
        """Leave the hardware in a safe, stopped state."""
