"""Typed local commands, results and telemetry, independent of the PiCar-X.

Modules pass these objects directly. The JSON form (camelCase keys, see the
PRD "Local Command Contract") is for logs and tests, and `parse_command` is
the one door for untrusted input such as AI tool calls: it validates
everything and raises `InvalidCommand`, so a bad command never reaches the
robot. Validation limits come from `laika.limits`; the move limits mirror what
`laika.safety` will actually allow.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Mapping

from laika.limits import (
    PAN_MAX,
    PAN_MIN,
    SPEED_MAX,
    STEER_MAX,
    STEER_MIN,
    TILT_MAX,
    TILT_MIN,
)

MAX_MOVE_SPEED = 40  # same ceiling SafetyController applies
MAX_MOVE_DURATION_MS = 1500  # same as SafetyController's default max duration
MAX_SPEAK_CHARS = 500
MAX_ID_CHARS = 64
DEFAULT_HONK_VOLUME = 80


class InvalidCommand(ValueError):
    """Untrusted input is not a valid command, result or telemetry message."""


class Direction(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


class ResultStatus(str, Enum):
    SUCCESS = "success"
    REJECTED = "rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# --- validation helpers -------------------------------------------------


def _number(name: str, value: Any, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidCommand(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise InvalidCommand(f"{name} must be finite, got {value!r}")
    if not low <= value <= high:
        raise InvalidCommand(f"{name} must be between {low} and {high}, got {value}")
    return value


def _integer(name: str, value: Any, low: int, high: int) -> int:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidCommand(f"{name} must be a whole number, got {value!r}")
    if not low <= value <= high:
        raise InvalidCommand(f"{name} must be between {low} and {high}, got {value}")
    return value


def _check_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_ID_CHARS:
        raise InvalidCommand(f"id must be a non-empty string of at most {MAX_ID_CHARS} characters")
    return value


def _check_timestamp(name: str, value: Any) -> float:
    return _number(name, value, 0, 1e13)


def _require_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidCommand(f"{name} must be an object, got {type(value).__name__}")
    return value


def _reject_unknown(name: str, data: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise InvalidCommand(f"unknown {name}: {', '.join(unknown)}")


# --- commands -----------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Command:
    """Base class. `created_at` (epoch seconds) lets later code expire stale commands."""

    name: ClassVar[str]
    id: str = field(default_factory=lambda: _new_id("cmd"))
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        _check_id(self.id)
        _check_timestamp("createdAt", self.created_at)

    def _parameters(self) -> dict[str, Any]:
        return {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "command",
            "id": self.id,
            "command": self.name,
            "parameters": self._parameters(),
            "createdAt": self.created_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass(frozen=True, kw_only=True)
class MoveCommand(Command):
    name: ClassVar[str] = "move"
    direction: Direction
    speed: int
    duration_ms: int
    steering: float = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        try:
            direction = Direction(self.direction)
        except ValueError:
            raise InvalidCommand(
                f"direction must be 'forward' or 'backward', got {self.direction!r}"
            ) from None
        object.__setattr__(self, "direction", direction)
        _integer("speed", self.speed, 1, min(MAX_MOVE_SPEED, SPEED_MAX))
        _integer("durationMs", self.duration_ms, 1, MAX_MOVE_DURATION_MS)
        _number("steering", self.steering, STEER_MIN, STEER_MAX)

    def _parameters(self) -> dict[str, Any]:
        return {
            "direction": self.direction.value,
            "speed": self.speed,
            "durationMs": self.duration_ms,
            "steering": self.steering,
        }


@dataclass(frozen=True, kw_only=True)
class StopCommand(Command):
    name: ClassVar[str] = "stop"


@dataclass(frozen=True, kw_only=True)
class LookCommand(Command):
    name: ClassVar[str] = "look"
    pan: float
    tilt: float

    def __post_init__(self) -> None:
        super().__post_init__()
        _number("pan", self.pan, PAN_MIN, PAN_MAX)
        _number("tilt", self.tilt, TILT_MIN, TILT_MAX)

    def _parameters(self) -> dict[str, Any]:
        return {"pan": self.pan, "tilt": self.tilt}


@dataclass(frozen=True, kw_only=True)
class HonkCommand(Command):
    name: ClassVar[str] = "honk"
    volume: int = DEFAULT_HONK_VOLUME

    def __post_init__(self) -> None:
        super().__post_init__()
        _integer("volume", self.volume, 0, 100)

    def _parameters(self) -> dict[str, Any]:
        return {"volume": self.volume}


@dataclass(frozen=True, kw_only=True)
class SpeakCommand(Command):
    name: ClassVar[str] = "speak"
    text: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.text, str) or not self.text.strip():
            raise InvalidCommand("text must be a non-empty string")
        if len(self.text) > MAX_SPEAK_CHARS:
            raise InvalidCommand(f"text must be at most {MAX_SPEAK_CHARS} characters")

    def _parameters(self) -> dict[str, Any]:
        return {"text": self.text}


COMMANDS: dict[str, type[Command]] = {
    cls.name: cls for cls in (MoveCommand, StopCommand, LookCommand, HonkCommand, SpeakCommand)
}

# JSON parameter name -> Python attribute, per command.
_PARAMETER_FIELDS: dict[str, dict[str, str]] = {
    "move": {
        "direction": "direction",
        "speed": "speed",
        "durationMs": "duration_ms",
        "steering": "steering",
    },
    "stop": {},
    "look": {"pan": "pan", "tilt": "tilt"},
    "honk": {"volume": "volume"},
    "speak": {"text": "text"},
}
_REQUIRED: dict[str, set[str]] = {
    "move": {"direction", "speed", "durationMs"},
    "look": {"pan", "tilt"},
    "speak": {"text"},
}


def parse_command(data: str | bytes | Mapping[str, Any]) -> Command:
    """Validate untrusted input (JSON text or a dict) and build a command.

    Raises `InvalidCommand` with a readable message for anything wrong.
    """
    message = _load(data)
    _reject_unknown("field", message, {"type", "id", "command", "parameters", "createdAt"})
    if message.get("type") != "command":
        raise InvalidCommand(f"type must be 'command', got {message.get('type')!r}")

    name = message.get("command")
    if not isinstance(name, str) or name not in COMMANDS:
        raise InvalidCommand(f"unknown command {name!r}; expected one of: {', '.join(COMMANDS)}")

    params = _require_mapping("parameters", message.get("parameters", {}))
    fields = _PARAMETER_FIELDS[name]
    _reject_unknown(f"parameters for {name}", params, set(fields))
    missing = sorted(_REQUIRED.get(name, set()) - set(params))
    if missing:
        raise InvalidCommand(f"{name} is missing parameters: {', '.join(missing)}")

    kwargs: dict[str, Any] = {fields[key]: value for key, value in params.items()}
    if "id" in message:
        kwargs["id"] = _check_id(message["id"])
    if "createdAt" in message:
        kwargs["created_at"] = _check_timestamp("createdAt", message["createdAt"])
    return COMMANDS[name](**kwargs)


def _load(data: str | bytes | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(data, (str, bytes)):
        try:
            data = json.loads(data)
        except ValueError as exc:
            raise InvalidCommand(f"not valid JSON: {exc}") from None
    return _require_mapping("message", data)


# --- results ------------------------------------------------------------


@dataclass(frozen=True)
class CommandResult:
    id: str
    result: ResultStatus
    reason: str = ""

    def __post_init__(self) -> None:
        _check_id(self.id)
        try:
            object.__setattr__(self, "result", ResultStatus(self.result))
        except ValueError:
            valid = ", ".join(s.value for s in ResultStatus)
            raise InvalidCommand(f"result must be one of: {valid}; got {self.result!r}") from None
        if not isinstance(self.reason, str):
            raise InvalidCommand("reason must be a string")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": "command_completed",
            "id": self.id,
            "result": self.result.value,
        }
        if self.reason:
            data["reason"] = self.reason
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: str | bytes | Mapping[str, Any]) -> CommandResult:
        message = _load(data)
        _reject_unknown("field", message, {"type", "id", "result", "reason"})
        if message.get("type") != "command_completed":
            raise InvalidCommand(f"type must be 'command_completed', got {message.get('type')!r}")
        if "id" not in message or "result" not in message:
            raise InvalidCommand("command_completed needs id and result")
        return cls(id=message["id"], result=message["result"], reason=message.get("reason", ""))

    from_json = from_dict


# --- telemetry ----------------------------------------------------------


@dataclass(frozen=True)
class Telemetry:
    """A snapshot of the sensors and actuators, independent of the hardware."""

    distance_cm: float | None
    cliff_detected: bool
    camera_pan: float
    camera_tilt: float
    steering: float = 0
    speed: int = 0
    timestamp: float = field(default_factory=time.time)
    battery_v: float | None = None

    def __post_init__(self) -> None:
        if self.battery_v is not None:
            _number("batteryV", self.battery_v, 0, 100)
        if self.distance_cm is not None:
            _number("distanceCm", self.distance_cm, 0, 1e6)
        if not isinstance(self.cliff_detected, bool):
            raise InvalidCommand("cliffDetected must be true or false")
        _number("camera.pan", self.camera_pan, PAN_MIN, PAN_MAX)
        _number("camera.tilt", self.camera_tilt, TILT_MIN, TILT_MAX)
        _number("steering", self.steering, STEER_MIN, STEER_MAX)
        _integer("speed", self.speed, -SPEED_MAX, SPEED_MAX)
        _check_timestamp("timestamp", self.timestamp)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": "telemetry",
            "distanceCm": self.distance_cm,
            "cliffDetected": self.cliff_detected,
            "camera": {"pan": self.camera_pan, "tilt": self.camera_tilt},
            "steering": self.steering,
            "speed": self.speed,
            "timestamp": self.timestamp,
        }
        if self.battery_v is not None:
            data["batteryV"] = self.battery_v
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: str | bytes | Mapping[str, Any]) -> Telemetry:
        message = _load(data)
        _reject_unknown(
            "field",
            message,
            {"type", "distanceCm", "cliffDetected", "camera", "steering", "speed", "timestamp", "batteryV"},
        )
        if message.get("type") != "telemetry":
            raise InvalidCommand(f"type must be 'telemetry', got {message.get('type')!r}")
        for key in ("cliffDetected", "camera"):
            if key not in message:
                raise InvalidCommand(f"telemetry is missing {key}")
        camera = _require_mapping("camera", message["camera"])
        _reject_unknown("camera fields", camera, {"pan", "tilt"})
        if "pan" not in camera or "tilt" not in camera:
            raise InvalidCommand("camera needs pan and tilt")
        kwargs: dict[str, Any] = {}
        for key, attr in (
            ("steering", "steering"),
            ("speed", "speed"),
            ("timestamp", "timestamp"),
            ("batteryV", "battery_v"),
        ):
            if key in message:
                kwargs[attr] = message[key]
        return cls(
            distance_cm=message.get("distanceCm"),
            cliff_detected=message["cliffDetected"],
            camera_pan=camera["pan"],
            camera_tilt=camera["tilt"],
            **kwargs,
        )

    from_json = from_dict
