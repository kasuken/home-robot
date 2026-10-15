"""The safety controller: the only code allowed to move the wheels.

The agent (or the AI behind it) proposes a `MoveCommand`; `SafetyController`
decides whether it may run and, once it runs, keeps watching it.

Monitoring cadence: a dedicated daemon thread ticks at `cadence_hz` (default
20 Hz, a 50 ms period). Every tick reads the sensors once and, while a move is
active, stops the motors when an obstacle is too close (driving forward), a
cliff is seen, the move's deadline passed, or a hardware call failed. The
tick is paced against an absolute schedule so the mean period stays at the
configured one even when a single tick runs late. 50 ms at the prototype's
top speed is about 1.5 cm of travel, small next to the 25 cm obstacle limit.

The stopping is done by the monitor thread, never by the caller of `execute`.
A caller that stalls (a slow AI request, audio, camera or storage call) is
therefore irrelevant: the monitor only touches the robot, whose locks are
short (docs/decisions/0001-threads-not-asyncio.md), and the event bus, whose
`publish` never blocks. Events are published outside the controller's lock.

Every move is bounded in time (`max_duration_s`, shorter backwards because
there is no rear sensor) and must be fresh: a command older than
`command_max_age_s` when it arrives is EXPIRED and never starts.

Issue #7 hook: the watchdog and AI-outage checks belong in `check()`, next to
`_hazard()`; they only need to end the active move through `_finish()`.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from laika.commands import (
    CommandResult,
    Direction,
    InvalidCommand,
    MoveCommand,
    ResultStatus,
    StopCommand,
    parse_command,
)
from laika.events import Event, EventBus, EventType
from laika.robot import Robot, RobotSensorState

log = logging.getLogger(__name__)

BACKWARD_MAX_S = 0.6
DEFAULT_CADENCE_HZ = 20.0
DEFAULT_COMMAND_MAX_AGE_S = 1.0
MAX_SPEED_CAP = 40
INTERVAL_HISTORY = 100
RESULT_HISTORY = 32
JOIN_TIMEOUT_S = 2.0


@dataclass(frozen=True)
class MonitorStats:
    """How the monitor loop is doing. Intervals are real seconds between tick starts."""

    ticks: int
    mean_interval_s: float | None
    max_interval_s: float | None
    intervals: tuple[float, ...]


@dataclass
class _Move:
    command: MoveCommand
    speed: int
    forward: bool
    deadline: float
    done: threading.Event
    result: CommandResult | None = None


class SafetyController:
    def __init__(
        self,
        robot: Robot,
        *,
        enabled: bool = True,
        speed: int = 25,
        max_duration_s: float = 1.5,
        min_front_distance_cm: float = 25.0,
        cadence_hz: float = DEFAULT_CADENCE_HZ,
        command_max_age_s: float = DEFAULT_COMMAND_MAX_AGE_S,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        events: EventBus | None = None,
    ) -> None:
        self.robot = robot
        self.enabled = enabled
        self.speed = max(0, min(speed, MAX_SPEED_CAP))
        self.max_duration_s = max_duration_s
        self.min_front_distance_cm = min_front_distance_cm
        self.period_s = 1.0 / max(cadence_hz, 1.0)
        self.command_max_age_s = command_max_age_s
        self.events = events
        # `clock` times move deadlines; `wall_clock` ages commands (created_at is epoch time).
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._active: _Move | None = None
        self._moves: OrderedDict[str, _Move] = OrderedDict()
        self._emergency: str | None = None
        self._hardware_ok = True
        self._closed = False
        self._thread: threading.Thread | None = None
        self._wake = threading.Event()
        self._ticks = 0
        self._intervals: deque[float] = deque(maxlen=INTERVAL_HISTORY)

    # --- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Start the monitor thread. Idempotent; `execute` also starts it if needed."""
        with self._lock:
            if self._thread is not None or self._closed:
                return
            self._thread = threading.Thread(target=self._run, name="laika-safety", daemon=True)
            self._thread.start()

    def close(self) -> None:
        """Stop the monitor and the motors. Moves started later are rejected."""
        out: list[Event] = []
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._active is not None:
                self._finish(self._active, ResultStatus.CANCELLED, "closed", out)
            else:
                self._halt()
            thread = self._thread
        self._wake.set()
        self._publish(out)
        if thread is not None and thread is not threading.current_thread():
            thread.join(JOIN_TIMEOUT_S)

    def stats(self) -> MonitorStats:
        with self._lock:
            intervals = tuple(self._intervals)
            ticks = self._ticks
        mean = sum(intervals) / len(intervals) if intervals else None
        return MonitorStats(ticks, mean, max(intervals) if intervals else None, intervals)

    # --- commands -----------------------------------------------------------

    def execute(self, command: MoveCommand, wait: bool = True) -> CommandResult | None:
        """Run a move if the checks allow it, and watch it until the monitor ends it.

        SUCCESS: the whole move ran. REJECTED: refused before the wheels turned
        (invalid, disabled, closed, emergency_stop, obstacle, cliff). EXPIRED: too
        old. CANCELLED: stopped early (obstacle, cliff, emergency_stop, stopped,
        superseded, closed). FAILED: the hardware raised an error.

        With `wait=False` a started move returns None at once (get its outcome
        with `wait`); a refused one still returns its result.
        """
        out: list[Event] = []
        with self._lock:
            result, move = self._begin(command, out)
        self._publish(out)
        if result is not None:
            return result
        self.start()
        if not wait:
            return None
        move.done.wait()
        return move.result

    def execute_raw(
        self, data: str | bytes | Mapping[str, Any], wait: bool = True
    ) -> CommandResult | None:
        """`execute` for untrusted input (AI tool calls): anything invalid is REJECTED, never run."""
        try:
            command = parse_command(data)
        except InvalidCommand as error:
            return CommandResult(_raw_id(data), ResultStatus.REJECTED, f"invalid: {error}")
        if isinstance(command, StopCommand):
            self.stop()
            return CommandResult(command.id, ResultStatus.SUCCESS)
        if not isinstance(command, MoveCommand):
            return CommandResult(
                command.id, ResultStatus.REJECTED, "invalid: not a movement command"
            )
        return self.execute(command, wait)

    def wait(self, command_id: str, timeout: float | None = None) -> CommandResult | None:
        """The outcome of a recent move, waiting for it; None if unknown or still running."""
        with self._lock:
            move = self._moves.get(command_id)
        if move is None or not move.done.wait(timeout):
            return None
        return move.result

    def stop(self) -> None:
        """Cancel the current move and stop the motors, without latching anything."""
        out: list[Event] = []
        with self._lock:
            if self._active is not None:
                self._finish(self._active, ResultStatus.CANCELLED, "stopped", out)
            else:
                self._halt()
        self._publish(out)

    def emergency_stop(self, reason: str = "") -> None:
        """Stop now and refuse every move until `reset_emergency_stop`."""
        out: list[Event] = []
        with self._lock:
            self._emergency = reason or "emergency stop"
            if self._active is not None:
                self._finish(self._active, ResultStatus.CANCELLED, "emergency_stop", out)
            else:
                self._halt()
        self._publish(out)

    def reset_emergency_stop(self) -> None:
        with self._lock:
            self._emergency = None

    @property
    def emergency_stopped(self) -> bool:
        return self._emergency is not None

    # --- starting a move (lock held) ------------------------------------------

    def _begin(
        self, command: object, out: list[Event]
    ) -> tuple[CommandResult | None, _Move | None]:
        """Returns (result, None) when the move did not start, else (None, the active move)."""
        if not isinstance(command, MoveCommand):
            return CommandResult(_safe_id(command), ResultStatus.REJECTED, "invalid"), None
        speed = min(command.speed, self.speed)
        forward = command.direction is Direction.FORWARD
        if self._closed:
            return self._refuse(command, ResultStatus.REJECTED, "closed", out), None
        if not self.enabled or speed <= 0:
            return self._refuse(command, ResultStatus.REJECTED, "disabled", out), None
        if self._emergency is not None:
            return self._refuse(command, ResultStatus.REJECTED, "emergency_stop", out), None
        if self._wall_clock() - command.created_at > self.command_max_age_s:
            return self._refuse(command, ResultStatus.EXPIRED, "expired", out), None

        try:
            sensors = self.robot.get_sensors()
        except Exception as error:  # noqa: BLE001 - any robot failure must stop and report
            return self._refuse(
                command, ResultStatus.FAILED, "hardware error", out, error=error
            ), None
        reason = self._hazard(sensors, forward)
        if reason:
            return self._refuse(
                command, ResultStatus.REJECTED, reason, out, distance=sensors.distance_cm
            ), None

        if self._active is not None:
            self._finish(self._active, ResultStatus.CANCELLED, "superseded", out)
        limit = self.max_duration_s if forward else min(self.max_duration_s, BACKWARD_MAX_S)
        duration_s = max(0.0, min(command.duration_ms / 1000, limit))
        move = _Move(command, speed, forward, self._clock() + duration_s, threading.Event())
        self._remember(move)
        out.append(
            _event(
                EventType.MOVEMENT_STARTED,
                commandId=command.id,
                direction=command.direction.value,
                speed=speed,
                durationMs=command.duration_ms,
            )
        )
        try:
            self.robot.steer(command.steering)
            self.robot.drive(speed if forward else -speed)
        except Exception as error:  # noqa: BLE001
            self._finish(move, ResultStatus.FAILED, "hardware error", out, error=error)
            return move.result, None
        self._active = move
        return None, move

    def _refuse(
        self,
        command: MoveCommand,
        status: ResultStatus,
        reason: str,
        out: list[Event],
        *,
        distance: float | None = None,
        error: Exception | None = None,
    ) -> CommandResult:
        """A move that never started. The motors are halted anyway: refusing must be safe."""
        move = _Move(command, 0, True, 0.0, threading.Event())
        self._remember(move)
        if self._active is None or error is not None:
            self._finish(move, status, reason, out, distance=distance, error=error)
        else:
            # Another move keeps running; do not stop it for a refused one.
            self._finish(move, status, reason, out, distance=distance, halt=False)
        return move.result

    def _remember(self, move: _Move) -> None:
        self._moves[move.command.id] = move
        while len(self._moves) > RESULT_HISTORY:
            self._moves.popitem(last=False)

    # --- the monitor --------------------------------------------------------------

    def _run(self) -> None:
        next_tick = last = time.monotonic()
        first = True
        while not self._closed:
            now = time.monotonic()
            if not first:
                with self._lock:
                    self._intervals.append(now - last)
            first, last = False, now
            try:
                self.check()
            except Exception:  # noqa: BLE001 - the loop must survive anything
                log.exception("safety tick failed")
                self._fail_safe()
            next_tick += self.period_s
            delay = next_tick - time.monotonic()
            if delay < -self.period_s:  # fell far behind (suspended process): resync
                next_tick = time.monotonic()
                delay = 0.0
            if self._wake.wait(max(0.0, delay)):
                return

    def check(self) -> None:
        """One monitoring tick: read the sensors once and stop what must be stopped."""
        out: list[Event] = []
        with self._lock:
            self._ticks += 1
            move = self._active
            try:
                sensors = self.robot.get_sensors()
            except Exception as error:  # noqa: BLE001
                if move is not None:
                    self._finish(move, ResultStatus.FAILED, "hardware error", out, error=error)
                else:
                    self._note_hardware_error(error, out)
            else:
                self._hardware_ok = True
                # Issue #7: watchdog and AI-outage checks go here, before the move checks.
                if move is not None:
                    reason = self._hazard(sensors, move.forward)
                    if reason:
                        self._finish(
                            move, ResultStatus.CANCELLED, reason, out, distance=sensors.distance_cm
                        )
                    elif self._clock() >= move.deadline:
                        self._finish(move, ResultStatus.SUCCESS, "", out)
        self._publish(out)

    def _hazard(self, sensors: RobotSensorState, forward: bool) -> str:
        """Why the robot must not move (or keep moving) given this reading, else ''."""
        if self._emergency is not None:
            return "emergency_stop"
        if sensors.cliff_detected:
            return "cliff"
        distance = sensors.distance_cm
        if forward and distance is not None and distance < self.min_front_distance_cm:
            return "obstacle"
        return ""

    def _fail_safe(self) -> None:
        out: list[Event] = []
        with self._lock:
            if self._active is not None:
                self._finish(self._active, ResultStatus.FAILED, "monitor error", out)
            else:
                self._halt()
        self._publish(out)

    def _note_hardware_error(self, error: Exception, out: list[Event]) -> None:
        if self._hardware_ok:
            self._hardware_ok = False
            out.append(_event(EventType.HARDWARE_UNAVAILABLE, error=str(error)))

    # --- ending a move (lock held) ----------------------------------------------------

    def _halt(self) -> Exception | None:
        """Stop the motors and centre the wheels; returns the first error, never raises."""
        first: Exception | None = None
        for action in (self.robot.stop, lambda: self.robot.steer(0)):
            try:
                action()
            except Exception as error:  # noqa: BLE001
                log.error("could not halt the robot: %s", error)
                first = first or error
        return first

    def _finish(
        self,
        move: _Move,
        status: ResultStatus,
        reason: str,
        out: list[Event],
        *,
        distance: float | None = None,
        error: Exception | None = None,
        halt: bool = True,
    ) -> None:
        halt_error = self._halt() if halt else None
        if halt_error is not None and error is None and status is not ResultStatus.FAILED:
            status, reason, error = ResultStatus.FAILED, "hardware error", halt_error
        if error is not None:
            self._hardware_ok = False
        move.result = CommandResult(move.command.id, status, reason)
        if self._active is move:
            self._active = None
        move.done.set()

        command_id = move.command.id
        if reason == "obstacle":
            out.append(
                _event(EventType.OBSTACLE_DETECTED, commandId=command_id, distanceCm=distance)
            )
        elif reason == "cliff":
            out.append(_event(EventType.CLIFF_DETECTED, commandId=command_id))
        if error is not None:
            out.append(
                _event(EventType.HARDWARE_UNAVAILABLE, commandId=command_id, error=str(error))
            )
        if status is ResultStatus.SUCCESS:
            out.append(
                _event(EventType.MOVEMENT_COMPLETED, commandId=command_id, result=status.value)
            )
        else:
            out.append(
                _event(
                    EventType.MOVEMENT_FAILED,
                    commandId=command_id,
                    result=status.value,
                    reason=reason,
                )
            )

    def _publish(self, events: list[Event]) -> None:
        """Called without the lock: publishing never blocks, but this keeps it provable."""
        if self.events is not None:
            for event in events:
                self.events.publish(event)


def _event(type: EventType, **data: Any) -> Event:
    return Event(type, data)


def _valid_id(value: object) -> str:
    return value if isinstance(value, str) and value.strip() and len(value) <= 64 else "unknown"


def _safe_id(command: object) -> str:
    return _valid_id(getattr(command, "id", None))


def _raw_id(data: object) -> str:
    return _valid_id(data.get("id")) if isinstance(data, Mapping) else "unknown"
