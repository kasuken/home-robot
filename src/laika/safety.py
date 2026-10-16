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

Two more things stop the robot (docs/decisions/0002-watchdog-strategy.md):

* `SafetyWatchdog`: a second, tiny thread that supervises the monitor itself.
  If the monitor stops ticking (stuck in a robot call, or dead) the watchdog
  stops the motors directly, WITHOUT taking the controller lock the stuck
  monitor may hold, rejects new moves with "watchdog" and publishes
  LocalWatchdogExpired. It clears by itself when ticks resume.
* AI endpoint outage (`set_ai_available(False)`): the active move is cancelled
  and new moves are rejected with "ai_unavailable" until the endpoint recovers.
  Today every move is chosen by the model, so "autonomous movement" means all
  wheel movement; direct user commands could be exempted once they exist.
* Battery (`set_battery_state`, #50): LOW refuses autonomous moves ("battery_low"), CRITICAL refuses
  every move ("battery_critical"). `stop` and the emergency stop are never refused.
* Unattended limits (#51): autonomous moves are refused while she is paused ("paused": the user said
  "resta ferma"), when the driving budget of the hour or the day is used up ("drive_budget") and, apart
  from a few moves backwards to get out, while the world model says she is in a no-go room
  ("no_go_room"). A lease somebody asked for (`Lease.requested`: "esplora la casa") is not held to the
  budget and may drive with an UNKNOWN battery. "Autonomous" is a move under an autonomous lease, or one `execute` was told is
  `self_directed` (her own path while she searches for what the user asked). The user's own moves are
  never refused for these reasons.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping

from laika.commands import (
    CommandResult,
    Direction,
    InvalidCommand,
    MoveCommand,
    ResultStatus,
    StopCommand,
    parse_command,
)
from laika.autonomy import AutonomyPause, DriveBudget
from laika.battery import BatteryState
from laika.events import Event, EventBus, EventType
from laika.robot import Robot, RobotSensorState

if TYPE_CHECKING:
    from laika.behaviour import Lease
    from laika.house_rules import HouseRules

log = logging.getLogger(__name__)

BACKWARD_MAX_S = 0.6
DEFAULT_CADENCE_HZ = 20.0
DEFAULT_COMMAND_MAX_AGE_S = 1.0
DEFAULT_ULTRASONIC_INVALID_LIMIT = 5  # 0.25 s of blind driving at 20 Hz
MAX_SPEED_CAP = 40
INTERVAL_HISTORY = 100
RESULT_HISTORY = 32
JOIN_TIMEOUT_S = 2.0
DEFAULT_WATCHDOG_TIMEOUT_S = 0.5  # ten missed ticks at 20 Hz
DEFAULT_WATCHDOG_PERIOD_S = 0.1
AI_LOCK_TIMEOUT_S = 0.2  # how long an outage reaction waits for the controller lock
WATCHDOG_LOCK_TIMEOUT_S = 0.05
# Autonomous moves backwards allowed while she is in a no-go room (to get out of it, never deeper in).
NO_GO_ESCAPE_MOVES = 8


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
    blind_ticks: int = 0  # consecutive ticks with no valid ultrasonic reading (forward moves)
    # A backward move that started with a cliff reading (backing away from the edge): the
    # reading is ignored until it clears once, after that a cliff stops the move again.
    cliff_exempt: bool = False
    # Counts against the driving budget by its real run time (#51).
    autonomous: bool = False
    started: float | None = None


class SafetyController:
    def __init__(
        self,
        robot: Robot,
        *,
        enabled: bool = True,
        speed: int = 25,
        max_duration_s: float = 1.5,
        backward_max_s: float = BACKWARD_MAX_S,
        min_front_distance_cm: float = 25.0,
        cadence_hz: float = DEFAULT_CADENCE_HZ,
        command_max_age_s: float = DEFAULT_COMMAND_MAX_AGE_S,
        ultrasonic_invalid_limit: int = DEFAULT_ULTRASONIC_INVALID_LIMIT,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        events: EventBus | None = None,
        watchdog_timeout_s: float = DEFAULT_WATCHDOG_TIMEOUT_S,
        watchdog_period_s: float = DEFAULT_WATCHDOG_PERIOD_S,
        heartbeat: Callable[[], None] | None = None,
        allow_autonomous: bool = False,
        drive_budget: DriveBudget | None = None,
        pause: AutonomyPause | None = None,
        house_rules: HouseRules | None = None,
    ) -> None:
        self.robot = robot
        # The unattended limits (#51): without arguments there is no budget, no pause and no no-go room.
        self.drive_budget = drive_budget if drive_budget is not None else DriveBudget()
        self.pause = pause if pause is not None else AutonomyPause()
        self.house_rules = house_rules
        self._escape_moves = 0
        # Off unless the caller turns it on: moves under an autonomous lease (exploration and
        # the other behaviours nobody asked for) are refused until the owner has verified the
        # robot on the hardware ([robot] allow_autonomous_driving).
        self.allow_autonomous = allow_autonomous
        # Called after every healthy monitor tick (the systemd watchdog ping hangs off it).
        self._heartbeat = heartbeat
        self.enabled = enabled
        self.speed = max(0, min(speed, MAX_SPEED_CAP))
        self.max_duration_s = max_duration_s
        self.backward_max_s = backward_max_s
        self.min_front_distance_cm = min_front_distance_cm
        self.period_s = 1.0 / max(cadence_hz, 1.0)
        self.command_max_age_s = command_max_age_s
        # Forward moves stop after this many blind ticks in a row; 0 disables the rule.
        self.ultrasonic_invalid_limit = max(0, int(ultrasonic_invalid_limit))
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
        # Plain attributes read without the lock: the watchdog must work when the monitor holds it.
        self._last_tick = time.monotonic()
        self._watchdog_expired = False
        self._ai_available = True
        # Until a battery monitor reports, the battery is not limiting (robots without a battery
        # reading). With a monitor, it starts UNKNOWN and reports its first state at startup.
        self._battery = BatteryState.OK
        self._latest_sensors: RobotSensorState | None = None
        self._watchdog = (
            SafetyWatchdog(self, watchdog_timeout_s, watchdog_period_s)
            if watchdog_timeout_s > 0
            else None
        )
        self._ticks = 0
        self._intervals: deque[float] = deque(maxlen=INTERVAL_HISTORY)
        # Called, without the controller lock, with the reason whenever something that must
        # stop autonomy happened (the behaviour arbiter listens). They must not block.
        self._halt_listeners: list[Callable[[str], None]] = []

    # --- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Start the monitor thread. Idempotent; `execute` also starts it if needed."""
        with self._lock:
            if self._thread is not None or self._closed:
                return
            self._last_tick = time.monotonic()
            self._thread = threading.Thread(target=self._run, name="laika-safety", daemon=True)
            self._thread.start()
        if self._watchdog is not None:
            self._watchdog.start()

    def close(self) -> None:
        """Stop the monitor, the watchdog and the motors. Moves started later are rejected."""
        if self._watchdog is not None:
            self._watchdog.close()
        if self._closed:
            return
        self._closed = True  # first and without the lock: a stalled monitor may hold it
        self._wake.set()
        out = self._cancel_active("closed", JOIN_TIMEOUT_S)
        self._halt()  # also centres the wheels
        self._publish(out)
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(JOIN_TIMEOUT_S)

    @property
    def watchdog_expired(self) -> bool:
        """True while the monitor is stalled or dead and moves are refused."""
        return self._watchdog_expired

    @property
    def ai_available(self) -> bool:
        return self._ai_available

    @property
    def hardware_ok(self) -> bool:
        return self._hardware_ok

    @property
    def moving(self) -> bool:
        """True while a move is in progress (read without the lock: a hint, not a decision)."""
        return self._active is not None

    @property
    def latest_sensors(self) -> RobotSensorState | None:
        """The monitor's last good reading (games read the distance here, never the sensor itself)."""
        return self._latest_sensors

    @property
    def battery_state(self) -> BatteryState:
        return self._battery

    def set_battery_state(self, state: BatteryState) -> None:
        """Tell the controller how the battery is (the battery monitor calls this on every change).

        LOW: moves under an autonomous lease are REJECTED "battery_low"; the user's and the
        conversation's moves still run. CRITICAL: every move is REJECTED "battery_critical" and the
        one in progress is cancelled. `stop` and `emergency_stop` are never affected. Going LOW or
        CRITICAL ends the autonomous leases, like any other halt.
        """
        previous, self._battery = self._battery, state
        if state is previous or state is BatteryState.OK:
            return
        if state is BatteryState.CRITICAL:
            self._publish(self._cancel_active("battery_critical", AI_LOCK_TIMEOUT_S))
        if previous is BatteryState.OK or state is BatteryState.CRITICAL:
            self._notify_halt(f"battery_{state.value}")

    @property
    def paused(self) -> bool:
        """True while the user's "resta ferma" is in force: nothing autonomous may move or start."""
        return self.pause.paused

    def pause_autonomy(self) -> bool:
        """Pause every autonomous activity (kept across restarts). True when it was not paused before.

        An autonomous move in progress is cancelled ("paused") and the autonomous leases end like on
        any halt; the user's own moves and the conversation are not touched.
        """
        changed = self.pause.pause()
        out: list[Event] = []
        if self._lock.acquire(timeout=AI_LOCK_TIMEOUT_S):
            try:
                move = self._active
                if move is not None and move.autonomous:
                    self._finish(move, ResultStatus.CANCELLED, "paused", out)
            finally:
                self._lock.release()
        self._publish(out)
        self._notify_halt("paused")
        return changed

    def resume_autonomy(self) -> bool:
        """Lift the pause. Only the user's explicit request may call this. True when it was paused."""
        return self.pause.resume()

    def autonomous_block(self) -> str:
        """Why an autonomous move would be refused for the unattended limits now ('paused', 'drive_budget'), else ''."""
        if self.pause.paused:
            return "paused"
        return "drive_budget" if self.drive_budget.exhausted() else ""

    def autonomy_status(self) -> dict[str, Any]:
        """The limits as data for get_robot_state and the logs."""
        rules = self.house_rules
        rooms = sorted(rules.no_go_rooms()) if rules is not None else []
        return {
            "paused": self.pause.paused,
            "drive_budget": self.drive_budget.snapshot(),
            "no_go_rooms": rooms,
            "in_no_go_room": rules.blocked_room() if rules is not None else None,
        }

    def add_halt_listener(self, listener: Callable[[str], None]) -> None:
        """Call `listener(reason)` on emergency stop, watchdog expiry, AI outage, hardware error."""
        self._halt_listeners = [*self._halt_listeners, listener]

    def _notify_halt(self, reason: str) -> None:
        for listener in self._halt_listeners:
            try:
                listener(reason)
            except Exception:  # noqa: BLE001 - a failing listener must not break the safety path
                log.exception("halt listener failed")

    def stats(self) -> MonitorStats:
        with self._lock:
            intervals = tuple(self._intervals)
            ticks = self._ticks
        mean = sum(intervals) / len(intervals) if intervals else None
        return MonitorStats(ticks, mean, max(intervals) if intervals else None, intervals)

    # --- commands -----------------------------------------------------------

    def execute(
        self,
        command: MoveCommand,
        wait: bool = True,
        lease: Lease | None = None,
        *,
        self_directed: bool = False,
    ) -> CommandResult | None:
        """Run a move if the checks allow it, and watch it until the monitor ends it.

        SUCCESS: the whole move ran. REJECTED: refused before the wheels turned
        (invalid, disabled, closed, emergency_stop, watchdog, ai_unavailable,
        obstacle, cliff, preempted, autonomous_disabled, battery_low, battery_critical, paused,
        drive_budget, no_go_room). EXPIRED: too old. CANCELLED: stopped early (obstacle,
        cliff, ultrasonic_invalid, emergency_stop, stopped, superseded, closed,
        watchdog, ai_unavailable, preempted, paused). FAILED: the hardware raised an error.

        `self_directed` marks a move Laika chose herself under a lease that is not autonomous (a search
        the user asked for picks its own path): the pause, the driving budget and the no-go rooms apply
        to it like to any autonomous move.

        With `wait=False` a started move returns None at once (get its outcome
        with `wait`); a refused one still returns its result.

        A move made under a `lease` (docs/behaviours.md) is REJECTED "preempted"
        when the lease is no longer active. The lease adds a check, never removes one.
        """
        if self._watchdog_expired and isinstance(command, MoveCommand):
            # Without the lock: a stalled monitor may be holding it.
            return self._refuse_unlocked(command, "watchdog")
        out: list[Event] = []
        with self._lock:
            result, move = self._begin(command, out, lease, self_directed)
        self._publish(out)
        if result is not None:
            return result
        self.start()
        if not wait:
            return None
        move.done.wait()
        return move.result

    def execute_raw(
        self, data: str | bytes | Mapping[str, Any], wait: bool = True, lease: Lease | None = None
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
        return self.execute(command, wait, lease)

    def wait(self, command_id: str, timeout: float | None = None) -> CommandResult | None:
        """The outcome of a recent move, waiting for it; None if unknown or still running."""
        move = self._moves.get(command_id)  # no lock: callers must be released while it is stuck
        if move is None or not move.done.wait(timeout):
            return None
        return move.result

    def stop(self, reason: str = "stopped") -> None:
        """Cancel the current move and stop the motors, without latching anything."""
        out: list[Event] = []
        with self._lock:
            if self._active is not None:
                self._finish(self._active, ResultStatus.CANCELLED, reason, out)
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
        self._notify_halt("emergency_stop")

    def set_ai_available(self, available: bool) -> None:
        """Tell the controller whether the AI endpoint answers.

        Unavailable: the active move is cancelled ("ai_unavailable") and new
        moves are rejected until it is available again. Nothing is replayed on
        recovery: later moves are new commands.
        """
        self._ai_available = available
        if not available:
            self._publish(self._cancel_active("ai_unavailable", AI_LOCK_TIMEOUT_S))
            self._notify_halt("ai_unavailable")

    def reset_emergency_stop(self) -> None:
        with self._lock:
            self._emergency = None

    @property
    def emergency_stopped(self) -> bool:
        return self._emergency is not None

    # --- starting a move (lock held) ------------------------------------------

    def _begin(
        self, command: object, out: list[Event], lease: Lease | None = None, self_directed: bool = False
    ) -> tuple[CommandResult | None, _Move | None]:
        """Returns (result, None) when the move did not start, else (None, the active move)."""
        if not isinstance(command, MoveCommand):
            return CommandResult(_safe_id(command), ResultStatus.REJECTED, "invalid"), None
        self._drop_settled()
        speed = min(command.speed, self.speed)
        forward = command.direction is Direction.FORWARD
        if self._closed:
            return self._refuse(command, ResultStatus.REJECTED, "closed", out), None
        if lease is not None and not lease.active:
            return self._refuse(command, ResultStatus.REJECTED, "preempted", out), None
        if lease is not None and lease.autonomous and not self.allow_autonomous:
            return self._refuse(command, ResultStatus.REJECTED, "autonomous_disabled", out), None
        if self._battery is BatteryState.CRITICAL:
            return self._refuse(command, ResultStatus.REJECTED, "battery_critical", out), None
        if lease is not None and lease.autonomous and self._battery is BatteryState.LOW:
            return self._refuse(command, ResultStatus.REJECTED, "battery_low", out), None
        requested = lease is not None and lease.requested
        if lease is not None and lease.autonomous and not requested and self._battery is BatteryState.UNKNOWN:
            return self._refuse(command, ResultStatus.REJECTED, "battery_unknown", out), None
        autonomous = self_directed or (lease is not None and lease.autonomous)
        if autonomous and (reason := self._unattended_limit(forward, requested)):
            return self._refuse(command, ResultStatus.REJECTED, reason, out), None
        if not self.enabled or speed <= 0:
            return self._refuse(command, ResultStatus.REJECTED, "disabled", out), None
        if self._emergency is not None:
            return self._refuse(command, ResultStatus.REJECTED, "emergency_stop", out), None
        if self._watchdog_expired:
            return self._refuse(command, ResultStatus.REJECTED, "watchdog", out), None
        if not self._ai_available:
            return self._refuse(command, ResultStatus.REJECTED, "ai_unavailable", out), None
        if self._wall_clock() - command.created_at > self.command_max_age_s:
            return self._refuse(command, ResultStatus.EXPIRED, "expired", out), None

        try:
            sensors = self.robot.get_sensors()
        except Exception as error:  # noqa: BLE001 - any robot failure must stop and report
            return self._refuse(
                command, ResultStatus.FAILED, "hardware error", out, error=error
            ), None
        backing_out = not forward and sensors.cliff_detected
        reason = self._hazard(sensors, forward, backing_out)
        if reason:
            return self._refuse(
                command, ResultStatus.REJECTED, reason, out, distance=sensors.distance_cm
            ), None

        if self._active is not None:
            self._finish(self._active, ResultStatus.CANCELLED, "superseded", out)
        limit = self.max_duration_s if forward else min(self.max_duration_s, self.backward_max_s)
        duration_s = max(0.0, min(command.duration_ms / 1000, limit))
        started = self._clock()
        move = _Move(
            command, speed, forward, started + duration_s, threading.Event(), cliff_exempt=backing_out,
            autonomous=autonomous, started=started,
        )
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

    def _unattended_limit(self, forward: bool, requested: bool = False) -> str:
        """The pause, the driving budget or a no-go room: why an autonomous move may not start, else ''.

        The budget is for what she does on her own: a behaviour somebody asked for (`requested`) is not held to it.
        """
        if self.pause.paused:
            return "paused"
        if not requested and self.drive_budget.exhausted():
            return "drive_budget"
        rules = self.house_rules
        room = rules.blocked_room() if rules is not None else None
        if room is None:
            self._escape_moves = 0
            return ""
        # In a room she must not be in: only a few moves backwards, to get out of it.
        if forward or self._escape_moves >= NO_GO_ESCAPE_MOVES:
            return "no_go_room"
        self._escape_moves += 1
        return ""

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

    def _refuse_unlocked(self, command: MoveCommand, reason: str) -> CommandResult:
        result = CommandResult(command.id, ResultStatus.REJECTED, reason)
        self._publish(
            [
                _event(
                    EventType.MOVEMENT_FAILED,
                    commandId=command.id,
                    result=result.result.value,
                    reason=reason,
                )
            ]
        )
        return result

    def _remember(self, move: _Move) -> None:
        self._moves[move.command.id] = move
        while len(self._moves) > RESULT_HISTORY:
            self._moves.popitem(last=False)

    # --- the monitor --------------------------------------------------------------

    def _run(self) -> None:
        next_tick = last = time.monotonic()
        first = True
        while not self._closed:
            now = self._last_tick = time.monotonic()
            if not first:
                with self._lock:
                    self._intervals.append(now - last)
            first, last = False, now
            try:
                self.check()
            except Exception:  # noqa: BLE001 - the loop must survive anything
                log.exception("safety tick failed")
                self._fail_safe()
            else:
                self._beat()
            self._last_tick = time.monotonic()
            next_tick += self.period_s
            delay = next_tick - time.monotonic()
            if delay < -self.period_s:  # fell far behind (suspended process): resync
                next_tick = time.monotonic()
                delay = 0.0
            if self._wake.wait(max(0.0, delay)):
                return

    def _beat(self) -> None:
        if self._heartbeat is None or self._watchdog_expired:
            return
        try:
            self._heartbeat()
        except Exception:  # noqa: BLE001 - a failing ping must never disturb the monitor
            log.exception("heartbeat failed")

    def check(self) -> None:
        """One monitoring tick: read the sensors once and stop what must be stopped."""
        out: list[Event] = []
        with self._lock:
            self._ticks += 1
            if self._drop_settled():
                self._halt()
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
                self._latest_sensors = sensors
                if move is not None and not self._ai_available:
                    self._finish(move, ResultStatus.CANCELLED, "ai_unavailable", out)
                elif move is not None:
                    reason = self._hazard(sensors, move.forward, move.cliff_exempt) or self._blind(
                        move, sensors
                    )
                    if not sensors.cliff_detected:
                        move.cliff_exempt = False
                    if reason:
                        self._finish(
                            move, ResultStatus.CANCELLED, reason, out, distance=sensors.distance_cm
                        )
                    elif self._clock() >= move.deadline:
                        self._finish(move, ResultStatus.SUCCESS, "", out)
        self._publish(out)

    def _blind(self, move: _Move, sensors: RobotSensorState) -> str:
        """Count consecutive invalid ultrasonic readings during a forward move.

        A forward move with no valid distance for `ultrasonic_invalid_limit` ticks
        in a row is stopped: an invalid reading must not mean "clear". Backward
        moves have no sensor to rely on and are not affected.
        """
        if not move.forward or not self.ultrasonic_invalid_limit:
            return ""
        move.blind_ticks = move.blind_ticks + 1 if sensors.distance_cm is None else 0
        return "ultrasonic_invalid" if move.blind_ticks >= self.ultrasonic_invalid_limit else ""

    def _hazard(self, sensors: RobotSensorState, forward: bool, cliff_exempt: bool = False) -> str:
        """Why the robot must not move (or keep moving) given this reading, else ''.

        The cliff sensors are at the front, so a backward move may start with the edge in view
        (the only way away from it); `cliff_exempt` is that case.
        """
        if self._emergency is not None:
            return "emergency_stop"
        if sensors.cliff_detected and not cliff_exempt:
            return "cliff"
        distance = sensors.distance_cm
        if forward and distance is not None and distance < self.min_front_distance_cm:
            return "obstacle"
        return ""

    # --- outside reactions (watchdog, AI outage): never wait for a stuck monitor ------

    def _stop_motors(self) -> None:
        """Stop the motors directly, without the controller lock. Never raises."""
        try:
            self.robot.stop()
        except Exception as error:  # noqa: BLE001
            log.error("could not stop the robot: %s", error)

    def _cancel_active(self, reason: str, lock_timeout_s: float) -> list[Event]:
        """Stop the motors and cancel the active move, even if the monitor holds the lock.

        The motors are stopped first, with no lock. If the lock cannot be had in
        time (a stalled monitor holds it) the move is settled without it, so a
        caller waiting on the move is released; the monitor drops it on resume.
        """
        out: list[Event] = []
        self._stop_motors()
        if self._lock.acquire(timeout=lock_timeout_s):
            try:
                self._drop_settled()
                if self._active is not None:
                    self._finish(self._active, ResultStatus.CANCELLED, reason, out)
            finally:
                self._lock.release()
        else:
            move = self._active
            if move is not None and move.result is None:
                move.result = CommandResult(move.command.id, ResultStatus.CANCELLED, reason)
                move.done.set()
                out.append(
                    _event(
                        EventType.MOVEMENT_FAILED,
                        commandId=move.command.id,
                        result=ResultStatus.CANCELLED.value,
                        reason=reason,
                    )
                )
        return out

    def _drop_settled(self) -> bool:
        """Forget an active move that was already settled without the lock (lock held)."""
        move = self._active
        if move is not None and move.done.is_set():
            self._active = None
            self._account(move)
            return True
        return False

    def _account(self, move: _Move) -> None:
        """Count an autonomous move against the driving budget, once, by its real run time."""
        if move.autonomous and move.started is not None:
            started, move.started = move.started, None
            self.drive_budget.add(self._clock() - started)

    def _watchdog_poll(self, timeout_s: float) -> None:
        """One watchdog check: expire when the monitor stalled or died, clear when it ticks again."""
        if self._closed:
            return
        thread = self._thread
        if thread is None:
            return
        age = time.monotonic() - self._last_tick
        dead = not thread.is_alive()
        if age > timeout_s or dead:
            first = not self._watchdog_expired
            self._watchdog_expired = True
            out = self._cancel_active("watchdog", WATCHDOG_LOCK_TIMEOUT_S)
            if first:
                log.error("safety monitor %s: motors stopped", "died" if dead else "stalled")
                out.append(
                    _event(
                        EventType.LOCAL_WATCHDOG_EXPIRED,
                        reason="monitor_dead" if dead else "monitor_stalled",
                        ageS=round(age, 3),
                        timeoutS=timeout_s,
                    )
                )
            self._publish(out)
        elif self._watchdog_expired:
            self._watchdog_expired = False
            log.warning("safety monitor ticking again: moves allowed")

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
        if move.result is not None:  # already settled without the lock (watchdog, outage)
            if self._active is move:
                self._active = None
            self._account(move)
            return
        if halt_error is not None and error is None and status is not ResultStatus.FAILED:
            status, reason, error = ResultStatus.FAILED, "hardware error", halt_error
        if error is not None:
            self._hardware_ok = False
        move.result = CommandResult(move.command.id, status, reason)
        if self._active is move:
            self._active = None
        self._account(move)
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
        for event in events:
            if event.type in (EventType.HARDWARE_UNAVAILABLE, EventType.LOCAL_WATCHDOG_EXPIRED):
                self._notify_halt(
                    "watchdog" if event.type is EventType.LOCAL_WATCHDOG_EXPIRED else "hardware"
                )
        if self.events is not None:
            for event in events:
                self.events.publish(event)


class SafetyWatchdog:
    """Supervises the monitor thread from a separate thread, at its own cadence.

    It only reads the monitor's last-tick time and calls the controller's
    lock-free reactions, so it cannot be blocked by whatever blocks the
    monitor. Nothing supervises the watchdog itself: it does no I/O beyond
    `robot.stop()` and catches every exception.
    """

    def __init__(self, controller: SafetyController, timeout_s: float, period_s: float) -> None:
        self.controller = controller
        self.timeout_s = timeout_s
        self.period_s = max(0.01, period_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()

    def start(self) -> None:
        with self._start_lock:
            if self._thread is not None or self._stop.is_set():
                return
            self._thread = threading.Thread(target=self._run, name="laika-watchdog", daemon=True)
            self._thread.start()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(JOIN_TIMEOUT_S)

    def _run(self) -> None:
        while not self._stop.wait(self.period_s):
            try:
                self.controller._watchdog_poll(self.timeout_s)
            except Exception:  # noqa: BLE001 - the watchdog must survive anything
                log.exception("watchdog poll failed")


def _event(type: EventType, **data: Any) -> Event:
    return Event(type, data)


def _valid_id(value: object) -> str:
    return value if isinstance(value, str) and value.strip() and len(value) <= 64 else "unknown"


def _safe_id(command: object) -> str:
    return _valid_id(getattr(command, "id", None))


def _raw_id(data: object) -> str:
    return _valid_id(data.get("id")) if isinstance(data, Mapping) else "unknown"
