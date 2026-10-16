"""Basic exploration: move a little, stop, look around, repeat, with every move gated by safety.

`Explorer` is a low-priority behaviour (Priority.EXPLORATION, resumable). While active
it runs on its own thread and does one step at a time: kick the lease, drive one short
slow forward move under it, wait for the result, and every few steps stop and
`Scanner.scan` (head only). A move is never queued: each one is a fresh command created
right before `SafetyController.execute`, so the expiry and staleness rules apply, and the
next one is only made after the previous one reported.

What ends or pauses it (docs/behaviours.md, "Exploration"):

* a more important behaviour (the wake word starts a conversation) pre-empts the lease:
  exploration pauses, with no move and no capture, and resumes when it is offered the body;
* safety (emergency stop, watchdog, AI outage, hardware error) or a stalled heartbeat
  revokes the lease: exploration ENDS and logs why, it never resumes by itself;
* an obstacle or a cliff is not an end: stop, back away a little, turn away, continue;
* a no-go room (a house rule, #51) is not an end either: when the world model says she is in one she
  reverses out, turns away and carries on; if she cannot get out the session ends ("no_go_room"). The
  pause ("resta ferma") ends it at once ("paused"); a used-up driving budget makes it head-only;
* the duration, the step limit, `stop()`, or an `on_scan` decision to stop.

`run_goal` is the same loop for a search the user asked for (#43): it runs on the caller's thread under
the caller's lease (the conversation's), never pauses, ends when that lease goes, and drives only while
`allow_autonomous` is on, because the controller's own gate does not cover a conversation lease.

Exploration starts when somebody asks for it (the tool) or when the idle selector (`idle.py`, #47) chooses it;
there is no timer of its own. Autonomous driving is gated (`SafetyController.allow_autonomous`): while it is off, or when
the controller answers "autonomous_disabled", exploration runs head-only (scans, no wheels).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterator

from laika.battery import BatteryState
from laika.behaviour import Arbiter, Lease, Preempted, Priority
from laika.capture_policy import CapturePolicy
from laika.commands import CommandResult, Direction, MoveCommand, ResultStatus
from laika.config import ExploreConfig
from laika.safety import SafetyController
from laika.scan import EnvironmentScan, ScanBusy, Scanner

if TYPE_CHECKING:
    from laika.places import PlaceNotes

log = logging.getLogger(__name__)

NAME = "exploration"
BODY_POLL_S = 0.2
MAX_BODY_WAIT_S = 120.0  # how long a started exploration waits for the body (a chat is using it)
HEAD_ONLY_INTERVAL_S = 10.0  # between scans when the wheels may not be used: a scan costs vision calls
BACKOFF_MS = 300  # straight back from a cliff edge
LEAVE_MS = 600  # straight back, retracing the way in, out of a no-go room
MAX_LEAVE_MOVES = 4  # moves backwards to get out of one room ...
TURN_AWAY_MOVES = 4  # ... then reversing arcs that point the nose away from its door (the controller allows both)
TURN_MS = 600  # reversing arc that turns the nose away (the controller caps backward moves)
TURN_SPEED = 25
TURN_STEERING = 30
MAX_BLOCKED = 6  # consecutive steps that got nowhere before exploration gives up
JOIN_TIMEOUT_S = 3.0
RESULT_HISTORY = 64

# The controller will not drive by itself now: carry on with the head only.
HEAD_ONLY_REASONS = frozenset({"autonomous_disabled", "battery_low", "battery_unknown", "drive_budget"})
# A forward step was refused because she is in a room she must not be in.
NO_GO_REASON = "no_go_room"
# A forward step ended for one of these: steer away and carry on.
AVOID_REASONS = frozenset({"obstacle", "cliff"})
# Not an error and not a reason to end by itself: look at the lease again, then try a fresh move.
RETRY_REASONS = frozenset({"preempted", "stopped", "superseded", "expired"})


@dataclass(frozen=True)
class ExploreDecision:
    """What `on_scan` may answer. "continue" keeps exploring, "stop" ends it (the seam for #28/#29)."""

    action: str = "continue"
    note: str = ""


OnScan = Callable[[EnvironmentScan], "str | ExploreDecision | None"]


class Explorer:
    def __init__(
        self,
        arbiter: Arbiter,
        safety: SafetyController,
        scanner: Scanner,
        capture_policy: CapturePolicy,
        *,
        config: ExploreConfig,
        clock: Callable[[], float] = time.monotonic,
        on_scan: OnScan | None = None,
        drive: bool | None = None,
        head_only_interval_s: float = HEAD_ONLY_INTERVAL_S,
        body_wait_s: float = MAX_BODY_WAIT_S,
        notes: PlaceNotes | None = None,
    ) -> None:
        self.arbiter = arbiter
        # What a session learns about the rooms is kept for later (places.py): begin/end around each session.
        self.notes = notes
        self.safety = safety
        self.scanner = scanner
        self.capture_policy = capture_policy
        self.config = config
        self.on_scan = on_scan
        self._clock = clock
        # None: drive when the controller allows autonomous moves. True forces the attempt
        # (the controller still decides); False never drives.
        self._drive = drive
        self.head_only_interval_s = head_only_interval_s
        self.body_wait_s = body_wait_s
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lease: Lease | None = None
        self._deadline = 0.0
        self._turn_sign = 1
        self._head_only = False
        self._requested = False
        self.head_only_reason = ""
        # `run_goal`: the session runs on the caller's thread, with its own scan callback, and never pauses.
        self._goal_running = False
        self._goal_on_scan: OnScan | None = None
        self._pausable = True
        self._turned = False
        self._since_scan = 0
        self._blocked_run = 0
        # What the last session did (for logs and tests).
        self.steps = 0
        self.scans = 0
        self.avoided = 0
        self.end_reason = ""
        self.refusal = ""
        self.results: deque[CommandResult] = deque(maxlen=RESULT_HISTORY)

    # --- public ----------------------------------------------------------------

    @property
    def active(self) -> bool:
        thread = self._thread
        return self._goal_running or (thread is not None and thread.is_alive())

    @property
    def driving(self) -> bool:
        """Whether a new session would drive (else it only looks around)."""
        return not self.drive_block()

    def drive_block(self, requested: bool = False) -> str:
        """Why a new session would keep the wheels still, in words for the console; '' when it would drive.

        `requested`: somebody asked ("esplora la casa"), so an unknown battery and a used-up driving budget
        do not stop it (the controller agrees, see `Lease.requested`).
        """
        if self._drive is not None:
            return "" if self._drive else "driving is turned off for this explorer"
        safety = self.safety
        if not safety.enabled:
            return "driving is off ([robot] allow_driving or --no-driving)"
        if not safety.allow_autonomous:
            return "autonomous driving is off ([robot] allow_autonomous_driving)"
        battery = safety.battery_state
        if battery is BatteryState.LOW or battery is BatteryState.CRITICAL:
            return f"the battery is {battery.value}"
        if battery is BatteryState.UNKNOWN and not requested:
            return "the battery voltage cannot be read (battery unknown)"
        block = safety.autonomous_block()
        if block == "paused":
            return "paused ('resta ferma')"
        if block == "drive_budget" and not requested:
            return "the driving budget of the hour or the day is used up"
        return ""

    @property
    def mode(self) -> str:
        return "head_only" if self._head_only or (not self.active and not self.driving) else "driving"

    def start(self, duration_s: float | None = None, requested: bool = False) -> bool:
        """Begin exploring for at most `duration_s` (capped by `max_duration_s`).

        True when a session started. It waits for the body if a conversation holds it.
        False (see `refusal`) when exploration is off, already running or not safe now.
        `requested`: somebody asked for it, see `drive_block`. `head_only_reason` says why the wheels stay still.
        """
        with self._lock:
            self.refusal = ""
            if self.active:
                self.refusal = "already exploring"
            elif not self.config.enabled:
                self.refusal = "exploration is turned off"
            elif (blocked := self._blocked()) is not None:
                self.refusal = f"not safe now ({blocked})"
            if self.refusal:
                return False
            limit = self.config.max_duration_s
            duration = limit if duration_s is None else max(1.0, min(float(duration_s), limit))
            self._stop = threading.Event()
            self.steps = self.scans = self.avoided = 0
            self.end_reason = ""
            self._turn_sign = 1
            self._turned = False
            self._since_scan = self._blocked_run = 0
            self._requested = requested
            self.head_only_reason = self.drive_block(requested)
            self._head_only = bool(self.head_only_reason)
            self._thread = threading.Thread(
                target=self._run, args=(duration, self._stop), name="laika-explore", daemon=True
            )
            self._thread.start()
            return True

    def run_goal(
        self,
        lease: Lease,
        duration_s: float,
        on_scan: OnScan,
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> str:
        """Explore on the caller's thread, under `lease`, until `on_scan` says stop. Returns why it ended.

        For a search (#43): the lease is the caller's (a conversation's), so nothing here asks the arbiter
        for the body and a pre-emption ends the session instead of pausing it. The controller's
        autonomous gate does not apply to such a lease, so this checks `allow_autonomous` itself and
        refuses with "autonomous_disabled". `should_stop` is polled by a small watcher thread that stops
        the move in flight. Refusals come back as the reason ("already exploring", ...), never raised.
        """
        with self._lock:
            self.refusal = ""
            if self.active:
                return "already exploring"
            if not self.config.enabled:
                return "exploration is turned off"
            if not (self.safety.enabled and self.safety.allow_autonomous):
                return "autonomous_disabled"
            if (blocked := self._blocked()) is not None:
                return blocked
            if self.safety.autonomous_block():
                return self.safety.autonomous_block()  # drive_budget: she would only look around
            stop = self._stop = threading.Event()
            self.steps = self.scans = self.avoided = 0
            self.end_reason = ""
            self._turn_sign = 1
            self._turned = False
            self._since_scan = self._blocked_run = 0
            self._head_only = False
            self._goal_running = True
            self._goal_on_scan = on_scan
            self._pausable = False
            self._lease = lease
            self._deadline = self._clock() + max(1.0, float(duration_s))
        done = threading.Event()
        watcher = None
        if should_stop is not None:
            watcher = threading.Thread(
                target=self._watch, args=(should_stop, stop, done, lease), name="laika-search-watch", daemon=True
            )
            watcher.start()
        reason = "error"
        try:
            lease.kick()
            reason = self._loop(lease, stop)
        except Exception:  # noqa: BLE001 - the caller's turn must go on
            log.exception("goal exploration failed")
        finally:
            done.set()
            if watcher is not None:
                watcher.join(1.0)
            if self.arbiter.holder is lease:
                self.safety.stop("stopped")
            with self._lock:
                self._goal_running = False
                self._goal_on_scan = None
                self._pausable = True
                self._lease = None
                self.end_reason = reason
        print(
            f"[laika] search exploration ended: {reason} "
            f"({self.steps} steps, {self.scans} scans, {self.avoided} avoided)"
        )
        return reason

    def _watch(
        self, should_stop: Callable[[], bool], stop: threading.Event, done: threading.Event, lease: Lease
    ) -> None:
        while not done.wait(0.03):
            if stop.is_set():
                return
            if should_stop() or not lease.active:
                stop.set()
                self.safety.stop("stopped")
                return

    def stop(self, timeout_s: float = JOIN_TIMEOUT_S) -> None:
        """End exploration now: stop the move in flight, give the body back. Idempotent."""
        self._stop.set()
        lease = self._lease
        if lease is not None and self.arbiter.holder is lease:
            self.safety.stop("stopped")
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout_s)

    # --- the session --------------------------------------------------------------

    def _blocked(self) -> str | None:
        safety = self.safety
        if safety.emergency_stopped:
            return "emergency_stop"
        if safety.watchdog_expired:
            return "watchdog"
        if not safety.ai_available:
            return "ai_unavailable"
        if not safety.hardware_ok:
            return "hardware"
        if safety.battery_state is BatteryState.CRITICAL:
            return "battery_critical"  # she stays still, head included
        if safety.paused:
            return "paused"  # "resta ferma": nothing autonomous, head included
        return None

    def _run(self, duration_s: float, stop: threading.Event) -> None:
        lease: Lease | None = None
        reason = "error"
        try:
            lease, reason = self._acquire(stop)
            if lease is None:
                return
            lease.requested = self._requested
            self._lease = lease
            self._deadline = self._clock() + duration_s
            mode = "driving" if not self._head_only else f"head-only, the wheels stay still: {self.head_only_reason}"
            print(f"[laika] exploration started: {mode}, up to {duration_s:g} s")
            self._notes("begin")
            self.capture_policy.set_exploring(True)
            reason = self._loop(lease, stop)
        except Exception:  # noqa: BLE001 - the thread must end cleanly, and the body must be released
            log.exception("exploration failed")
            reason = "error"
        finally:
            self.capture_policy.set_exploring(False)
            if lease is not None:
                lease.release()
                self._notes("end", steps=self.steps, scans=self.scans, driving=not self._head_only, reason=reason)
            self._lease = None
            self.end_reason = reason
            print(
                f"[laika] exploration ended: {reason} "
                f"({self.steps} steps, {self.scans} scans, {self.avoided} avoided)"
            )

    def _notes(self, what: str, **kwargs: object) -> None:
        if self.notes is None:
            return
        try:
            getattr(self.notes, what)(**kwargs)
        except Exception:  # noqa: BLE001 - notes are never a reason for exploring to fail
            log.exception("place notes failed (%s)", what)

    def _acquire(self, stop: threading.Event) -> tuple[Lease | None, str]:
        give_up = self._clock() + self.body_wait_s
        while not stop.is_set():
            if (blocked := self._blocked()) is not None:
                return None, blocked
            lease = self.arbiter.request(NAME, Priority.EXPLORATION, resumable=True)
            if lease is not None:
                return lease, ""
            if self._clock() >= give_up:
                return None, "no_body"
            stop.wait(BODY_POLL_S)  # a conversation has the body: start when it ends
        return None, "stopped"

    def _loop(self, lease: Lease, stop: threading.Event) -> str:
        while True:
            if stop.is_set():
                return "stopped"
            if (reason := self._gate(lease, stop)) is not None:
                return reason
            if (blocked := self._blocked()) is not None:
                return blocked
            if self._clock() >= self._deadline:
                return "duration"
            if self.steps >= self.config.max_steps:
                return "max_steps"
            lease.kick()
            if (reason := self._leave_no_go(lease, stop)) is not None:
                return reason
            if (reason := self._step(lease, stop)) is not None:
                return reason

    def _gate(self, lease: Lease, stop: threading.Event) -> str | None:
        """None when the lease is ours. While pre-empted wait for the body; else the reason it ended."""
        paused_at: float | None = None
        while True:
            try:
                lease.check()
            except Preempted as gone:
                if not lease.waiting or not self._pausable:
                    return lease.end_reason or gone.reason
                if paused_at is None:
                    paused_at = self._clock()
                    self.capture_policy.set_exploring(False)
                    print("[laika] exploration paused: a more important behaviour has the body")
                if stop.is_set():
                    return "stopped"
                lease.wait_until_active(BODY_POLL_S)
                continue
            if paused_at is not None:
                self._deadline += self._clock() - paused_at  # a pause does not use up the time
                self.capture_policy.set_exploring(True)
                print("[laika] exploration resumed")
            return None

    # --- one step -------------------------------------------------------------------

    def _step(self, lease: Lease, stop: threading.Event) -> str | None:
        """One step and, when due, one scan. Returns why exploration must end, else None."""
        if self._head_only and self._goal_running:
            return "autonomous_disabled"  # a search has nothing to do without wheels: it scanned already
        if self._head_only:
            self._wait(self.head_only_interval_s, lease, stop)
            if stop.is_set() or not lease.active:
                return None
            self.steps += 1
            return self._scan(lease, stop)
        if (reason := self._drive_step(lease, stop)) is not None:
            return reason
        if self._head_only:  # the controller just refused autonomous moves: scans only from now on
            return None
        if self._turned or self._since_scan >= self.config.scan_every_steps:
            return self._scan(lease, stop)
        return None

    def _drive_step(self, lease: Lease, stop: threading.Event) -> str | None:
        result = self._move(lease, Direction.FORWARD, self.config.speed, self.config.step_ms)
        outcome = result.result
        reason = result.reason
        if outcome is ResultStatus.SUCCESS:
            self.steps += 1
            self._since_scan += 1
            self._blocked_run = 0
            return None
        if reason in HEAD_ONLY_REASONS:
            self._head_only = True
            self.head_only_reason = reason
            print(f"[laika] exploration: the wheels stay still ({reason}), looking around only")
            return None
        if reason in AVOID_REASONS:
            return self._avoid(lease, stop, reason)
        if reason == NO_GO_REASON:
            return self._leave_no_go(lease, stop) or self._stuck_check()
        if reason in RETRY_REASONS:
            return self._stuck_check()
        return reason or outcome.value  # emergency stop, watchdog, outage, hardware, disabled, ...

    def _stuck_check(self) -> str | None:
        self._blocked_run += 1
        return "stuck" if self._blocked_run >= MAX_BLOCKED else None

    def _avoid(self, lease: Lease, stop: threading.Event, reason: str) -> str | None:
        """Stop (the controller already did), back off from a cliff, turn away. Bounded."""
        self.avoided += 1
        if self._blocked_run == 0:  # a new episode turns the other way; a repeat keeps turning
            self._turn_sign = -self._turn_sign
        print(f"[laika] exploration: {reason} ahead, turning away")
        if reason == "cliff":
            if (end := self._side_move(lease, Direction.BACKWARD, BACKOFF_MS, 0)) is not None:
                return end
        steering = TURN_STEERING * self._turn_sign
        if (end := self._side_move(lease, Direction.BACKWARD, TURN_MS, steering)) is not None:
            return end
        self._turned = True
        stop.wait(0.05)  # never a busy loop, even when everything is refused
        return self._stuck_check()

    def _no_go_room(self) -> str | None:
        """The room she is in when a house rule forbids it (best effort: the world model may lag or not know)."""
        rules = self.safety.house_rules
        return rules.blocked_room() if rules is not None and not self._head_only else None

    def _leave_no_go(self, lease: Lease, stop: threading.Event) -> str | None:
        """In a no-go room: reverse out (retracing the way in), turn away, record it. Bounded.

        None when she was not in one, or got out and may carry on; else why the session must end
        ("no_go_room": she could not get out, so her lease ends rather than driving about in there).
        """
        room = self._no_go_room()
        if room is None:
            return None
        rules = self.safety.house_rules
        assert rules is not None
        self.avoided += 1
        print(f"[laika] exploration: the {room.replace('_', ' ')} is off limits, backing out")
        try:
            rules.record_turned_back(room)
        except Exception:  # noqa: BLE001 - the episode is a note, never a reason not to back out
            log.exception("could not record the turn back")
        for _ in range(MAX_LEAVE_MOVES):
            if stop.is_set() or not lease.active:
                return None
            if (end := self._side_move(lease, Direction.BACKWARD, LEAVE_MS, 0)) is not None:
                return end
            stop.wait(0.05)  # the room change reaches the world model through the event bus
            if self._no_go_room() is None:
                break
        else:
            return NO_GO_REASON
        self._turn_sign = -self._turn_sign
        for _ in range(TURN_AWAY_MOVES):  # one arc turns the nose only a little: face away from the door
            if stop.is_set() or not lease.active:
                return None
            if (end := self._side_move(lease, Direction.BACKWARD, TURN_MS, TURN_STEERING * self._turn_sign)) is not None:
                return end
        self._turned = True
        return None

    def _side_move(
        self, lease: Lease, direction: Direction, duration_ms: int, steering: float
    ) -> str | None:
        """A back-off or turn move. Refusals that are not fatal are left to the next step."""
        result = self._move(lease, direction, TURN_SPEED, duration_ms, steering)
        reason = result.reason
        if result.result is ResultStatus.SUCCESS or reason in AVOID_REASONS or reason in RETRY_REASONS:
            return None
        if reason in HEAD_ONLY_REASONS:
            self._head_only = True
            self.head_only_reason = reason
            return None
        return reason or result.result.value

    def _move(
        self,
        lease: Lease,
        direction: Direction,
        speed: int,
        duration_ms: int,
        steering: float = 0,
    ) -> CommandResult:
        """One fresh move under the lease, and wait for how it ended."""
        lease.kick()
        command = MoveCommand(
            direction=direction,
            speed=max(1, min(int(speed), 40)),
            duration_ms=max(1, min(int(duration_ms), 1500)),
            steering=steering,
        )
        # Her own path, even under a conversation lease (a search): the pause, budget and no-go rooms apply.
        result = self.safety.execute(command, lease=lease, self_directed=True)
        lease.kick()
        if result is None:  # execute(wait=True) always reports; be safe about it
            result = CommandResult(command.id, ResultStatus.FAILED, "no result")
        self.results.append(result)
        return result

    # --- looking around ----------------------------------------------------------------

    def _scan(self, lease: Lease, stop: threading.Event) -> str | None:
        """Stop and look around with the head only; hand the result to `on_scan`."""
        self._since_scan = 0
        self._turned = False
        if not lease.active or stop.is_set():
            return None
        try:
            with self._heartbeat(lease):
                scan = self.scanner.scan(keep_going=lambda: lease.active and not stop.is_set())
        except ScanBusy:
            return None
        except Exception as error:  # noqa: BLE001 - a head or camera failure must not leave her driving on
            log.exception("exploration scan failed")
            return f"scan failed: {error}"
        if not lease.active or stop.is_set():
            return None  # the body was taken away mid-scan: the partial result is not used
        self.scans += 1
        lease.kick()
        return self._decide(scan)

    def _decide(self, scan: EnvironmentScan) -> str | None:
        on_scan = self._goal_on_scan or self.on_scan
        if on_scan is None:
            return None
        try:
            answer = on_scan(scan)
        except Exception:  # noqa: BLE001 - the seam must not break the loop
            log.exception("on_scan failed")
            return None
        action = answer.action if isinstance(answer, ExploreDecision) else answer
        if action == "stop":
            return "decision"
        return None

    @contextmanager
    def _heartbeat(self, lease: Lease) -> Iterator[None]:
        """Keep the lease alive while a scan blocks this thread (a scan is bounded by the Scanner)."""
        done = threading.Event()
        period = max(0.05, self.arbiter.lease_timeout_s / 4)

        def beat() -> None:
            while not done.wait(period):
                lease.kick()

        thread = threading.Thread(target=beat, name="laika-explore-beat", daemon=True)
        thread.start()
        try:
            yield
        finally:
            done.set()
            thread.join(1.0)

    def _wait(self, seconds: float, lease: Lease, stop: threading.Event) -> None:
        """Wait without moving, kicking the lease, until time is up, stop, or the body is taken."""
        end = self._clock() + seconds
        while not stop.is_set() and lease.active and self._clock() < end:
            lease.kick()
            stop.wait(min(BODY_POLL_S, max(0.0, end - self._clock())))

