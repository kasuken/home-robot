"""A stand-in robot that prints what the real one would do.

Without a house it only records what it is told. With a `House` it is a small
2D simulation: the robot drives around (a bicycle model, integrated lazily on
every call), bumps into walls and things, and reads ray-cast distance and
cliff sensors. Tests script faults through `robot.faults` and the world
through the `House`.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable

from laika import __version__
from laika.audio import Mouth
from laika.events import EventBus, EventType, emit
from laika.house import House, Pose
from laika.robot import (
    ALL_CAPABILITIES,
    PAN_MAX,
    PAN_MIN,
    SPEED_MAX,
    STEER_MAX,
    STEER_MIN,
    TILT_MAX,
    TILT_MIN,
    HardwareError,
    ImageObservation,
    RobotIdentity,
    RobotSensorState,
    RobotState,
    clamp,
)

IMAGE_SIZE = (640, 480)
GRAYSCALE_FLOOR = (1000, 1000, 1000)
GRAYSCALE_EDGE = (50, 50, 50)

ROBOT_RADIUS_CM = 10.0
WHEELBASE_CM = 10.0
CM_PER_S_PER_SPEED = 0.6
STEP_CM = 2.0
MAX_ELAPSED_S = 5.0


class _Auto:
    def __repr__(self) -> str:
        return "AUTO"


AUTO = _Auto()


@dataclass
class SimFaults:
    """Failures a test can switch on. `stop()` never fails: it must always work."""

    ultrasonic_invalid: bool = False
    ultrasonic_stuck_cm: float | None = None
    cliff_stuck: bool | None = None
    drive_error: bool = False
    steer_error: bool = False
    camera_error: bool = False


MAX_SPOKEN_KEPT = 100


class SimulatedRobot:
    def __init__(
        self,
        distance_cm: float | None | _Auto = AUTO,
        verbose: bool = True,
        mouth: Mouth | None = None,
        capabilities: frozenset[str] = ALL_CAPABILITIES,
        house: House | None = None,
        start: Pose | None = None,
        events: EventBus | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._capabilities = capabilities
        self.pan = 0.0
        self.tilt = 0.0
        self.steering = 0.0
        self.speed = 0
        self.honks = 0
        self.bumps = 0
        self.spoken: list[str] = []
        self.verbose = verbose
        self.house = house
        self.events = events
        self.faults = SimFaults()
        self._mouth = mouth
        self._clock = clock
        self._lock = threading.RLock()
        self._distance_override: float | None | _Auto = distance_cm
        self._cliff_override: bool | None = None
        start = start or (house.start if house else Pose(0.0, 0.0))
        self._x, self._y, self._heading = start.x, start.y, start.heading
        self._last_tick = clock()
        self._room = self._room_name()
        self._seen_people: frozenset[str] = frozenset()

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"  [robot] {message}")

    # --- overridable readings --------------------------------------------

    @property
    def distance(self) -> float | None:
        """The ultrasonic reading. Assign a number or None to override it, `AUTO` to ray cast again."""
        with self._lock:
            self._advance()
            return self._true_distance()

    @distance.setter
    def distance(self, value: float | None | _Auto) -> None:
        with self._lock:
            self._distance_override = value

    @property
    def cliff(self) -> bool:
        """The cliff reading. Assign a bool to override it, None to compute it from the house."""
        with self._lock:
            self._advance()
            return self._true_cliff()

    @cliff.setter
    def cliff(self, value: bool | None) -> None:
        with self._lock:
            self._cliff_override = value

    def _true_distance(self) -> float | None:
        if not isinstance(self._distance_override, _Auto):
            return self._distance_override
        if self.house is None:
            return 100.0
        fx, fy = self._front()
        return self.house.ray_distance(fx, fy, self._heading)

    def _true_cliff(self) -> bool:
        if self._cliff_override is not None:
            return self._cliff_override
        if self.house is None:
            return False
        return self.house.is_over_cliff(*self._front())

    def _read_distance(self) -> float | None:
        if self.faults.ultrasonic_invalid:
            return None
        if self.faults.ultrasonic_stuck_cm is not None:
            return self.faults.ultrasonic_stuck_cm
        return self._true_distance()

    def _read_cliff(self) -> bool:
        if self.faults.cliff_stuck is not None:
            return self.faults.cliff_stuck
        return self._true_cliff()

    # --- position ----------------------------------------------------------

    def _front(self) -> tuple[float, float]:
        angle = math.radians(self._heading)
        return (
            self._x + ROBOT_RADIUS_CM * math.cos(angle),
            self._y + ROBOT_RADIUS_CM * math.sin(angle),
        )

    def _room_name(self) -> str | None:
        room = self.house.room_at(self._x, self._y) if self.house else None
        return room.name if room else None

    def _advance(self) -> None:
        """Move the robot for the time since the last call. Stops at the first contact."""
        now = self._clock()
        elapsed = min(now - self._last_tick, MAX_ELAPSED_S)
        self._last_tick = now
        house = self.house
        if house is None or self.speed == 0 or elapsed <= 0:
            return
        distance = self.speed * CM_PER_S_PER_SPEED * elapsed
        turn = math.degrees(distance / WHEELBASE_CM * math.tan(math.radians(self.steering)))
        steps = max(1, math.ceil(abs(distance) / STEP_CM))
        for _ in range(steps):
            heading = self._heading - turn / steps
            mid = math.radians((self._heading + heading) / 2)
            x = self._x + distance / steps * math.cos(mid)
            y = self._y + distance / steps * math.sin(mid)
            if house.collides(x, y, ROBOT_RADIUS_CM):
                self.speed = 0
                self.bumps += 1
                self._log("bump")
                break
            self._x, self._y, self._heading = x, y, heading
        self._enter_room()

    def _enter_room(self) -> None:
        room = self._room_name()
        if room is not None and room != self._room:
            previous, self._room = self._room, room
            emit(self.events, EventType.ROOM_ENTERED, room=room, previous=previous)

    def pose(self) -> Pose:
        with self._lock:
            self._advance()
            return Pose(self._x, self._y, self._heading % 360.0)

    def current_room(self) -> str | None:
        with self._lock:
            self._advance()
            return self._room_name()

    # --- Robot interface -----------------------------------------------------

    def identity(self) -> RobotIdentity:
        return RobotIdentity(
            robot_id="laika-sim",
            name="Laika",
            version=__version__,
            capabilities=self._capabilities,
        )

    def get_sensors(self) -> RobotSensorState:
        with self._lock:
            self._advance()
            cliff = self._read_cliff()
            return RobotSensorState(
                distance_cm=self._read_distance(),
                cliff_detected=cliff,
                grayscale=GRAYSCALE_EDGE if cliff else GRAYSCALE_FLOOR,
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

    def _camera_view(self) -> tuple[list[str], list[str]]:
        """Names of the things and the people the camera sees right now."""
        if self.house is None:
            return [], []
        # Pan is positive to the right, which is clockwise.
        things, people = self.house.visible(self._x, self._y, self._heading - self.pan)
        return [t.name for t in things], [p.name for p in people]

    def people_visible(self) -> tuple[str, ...]:
        """Ground truth: the people in the camera's field of view."""
        with self._lock:
            self._advance()
            return tuple(self._camera_view()[1])

    def capture_image(self) -> ImageObservation:
        with self._lock:
            if self.faults.camera_error:
                raise HardwareError("simulated camera failure")
            self._advance()
            things, people = self._camera_view()
            self._publish_people(frozenset(people))
            scene = tuple(things + people)
            return ImageObservation(
                jpeg=None,
                width=IMAGE_SIZE[0],
                height=IMAGE_SIZE[1],
                camera_pan=self.pan,
                camera_tilt=self.tilt,
                scene=scene,
                people=tuple(people),
                simulated=True,
            )

    def _publish_people(self, seen: frozenset[str]) -> None:
        for name in sorted(seen - self._seen_people):
            emit(self.events, EventType.PERSON_DETECTED, name=name, room=self._room)
        for name in sorted(self._seen_people - seen):
            emit(self.events, EventType.PERSON_LOST, name=name)
        self._seen_people = seen

    def speak(self, text: str) -> None:
        # Not under the lock: speech takes seconds and must not block the safety loop.
        self.spoken.append(text)
        del self.spoken[:-MAX_SPOKEN_KEPT]
        if self._mouth is not None:
            self._mouth.say(text)
        else:
            print(f"Laika: {text}")

    def head_pose(self) -> tuple[float, float]:
        with self._lock:
            return (self.pan, self.tilt)

    def steer(self, angle: float) -> None:
        with self._lock:
            if self.faults.steer_error:
                raise HardwareError("simulated steering failure")
            self._advance()
            self.steering = clamp(angle, STEER_MIN, STEER_MAX)

    def drive(self, speed: int) -> None:
        with self._lock:
            if self.faults.drive_error:
                raise HardwareError("simulated motor failure")
            self._advance()
            self.speed = int(clamp(speed, -SPEED_MAX, SPEED_MAX))
            self._log(f"drive speed={self.speed}")

    def stop(self) -> None:
        with self._lock:
            self._advance()
            if self.speed:
                self._log("stop")
            self.speed = 0

    def distance_cm(self) -> float | None:
        with self._lock:
            self._advance()
            return self._read_distance()

    def cliff_detected(self) -> bool:
        with self._lock:
            self._advance()
            return self._read_cliff()

    def honk(self, volume: int) -> None:
        with self._lock:
            self.honks += 1
            self._log("BEEP BEEP")

    def reset(self) -> None:
        with self._lock:
            self.stop()
            self.steering = 0.0
            self.look(0, 0)

    def close(self) -> None:
        with self._lock:
            self.reset()
            self._log("closed")
