"""The behaviour priority hierarchy: who owns the body (wheels, head, gestures) right now.

The PRD ranks what Laika does and says lower priority must never interfere with
higher priority. `Arbiter` enforces that. A behaviour asks for the body with
`request` and gets a `Lease`, or None when something at least as important
already holds it. A strictly more important request pre-empts the holder: its
lease is revoked, the robot is stopped, queued gestures are dropped, and the
holder is told through `on_preempt`. If the holder is resumable it waits on a
small bounded list and gets the body back (same lease object, `on_resume`) when
the pre-empting behaviour releases.

Safety is not a behaviour. It is the safety controller itself: it needs no lease,
and every move made under a lease still passes all its checks. The arbiter only
listens to it: an emergency stop, watchdog expiry, AI outage or hardware error
ends every lease below CONVERSATION at once (the user can still talk), and
autonomous requests are refused while any of those holds. A low or critical battery
(`SafetyController.set_battery_state`) ends them the same way; only a critical one also refuses new
requests: when she is low, head-only activities may go on (the controller refuses their wheel moves).

When the user paused her ("resta ferma", #51) every autonomous request (CURIOSITY and below) is
refused and the autonomous leases are revoked at once; the conversation and the user's commands are
not touched. Only the user lifts the pause (`SafetyController.resume_autonomy`).

Autonomous behaviours (CURIOSITY and below) must also prove they are alive:
they `kick()` their lease at least every `lease_timeout_s`, or a small timer
thread revokes it and stops the robot. This is the per-behaviour heartbeat
planned in docs/decisions/0002-watchdog-strategy.md.

Callbacks run outside the arbiter's lock, and never on the safety threads'
critical path: the safety-driven revocation flips the leases inline (so the next
move is refused at once) and runs the callbacks on a short-lived thread.
"""

from __future__ import annotations

import enum
import itertools
import logging
import threading
import time
from typing import Callable

from laika.actions import ActionPlayer
from laika.battery import BatteryState
from laika.safety import SafetyController

log = logging.getLogger(__name__)

DEFAULT_LEASE_TIMEOUT_S = 2.0
MAX_WAITING = 4
JOIN_TIMEOUT_S = 2.0


class Priority(enum.IntEnum):
    """PRD order; a lower number is more important."""

    SAFETY = 0
    USER_COMMAND = 1
    CONVERSATION = 2
    FOLLOW_UP = 3
    IMPORTANT_OBSERVATION = 4
    CURIOSITY = 5
    SOCIAL = 6
    EXPLORATION = 7


# Ended at once by a safety outcome (everything below an active conversation).
SAFETY_REVOKES_BELOW = Priority.CONVERSATION
# Must kick its lease to keep it.
HEARTBEAT_FROM = Priority.CURIOSITY
# Moves of these leases are autonomous: the safety controller refuses them unless it was
# built with `allow_autonomous` (docs/behaviours.md, "The autonomous-driving gate").
AUTONOMOUS_FROM = HEARTBEAT_FROM


class Preempted(Exception):
    """The lease is no longer active: pre-empted (maybe resumable) or ended."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _State(enum.Enum):
    ACTIVE = "active"
    WAITING = "waiting"  # pre-empted, offered the body again when it frees up
    ENDED = "ended"


class Lease:
    """A behaviour's right to use the body. Created by `Arbiter.request`."""

    def __init__(
        self,
        cond: threading.Condition,
        name: str,
        priority: Priority,
        *,
        on_preempt: Callable[[Lease], None] | None = None,
        on_resume: Callable[[Lease], None] | None = None,
        resumable: bool = True,
        id: int = 0,
        arbiter: Arbiter | None = None,
    ) -> None:
        self.name = name
        self.priority = priority
        self.id = id
        self.resumable = resumable
        self.on_preempt = on_preempt
        self.on_resume = on_resume
        self.end_reason = ""
        self._state = _State.ACTIVE
        self._cond = cond
        self._arbiter = arbiter
        self._last_kick = time.monotonic()

    @classmethod
    def denied(cls, name: str, priority: Priority) -> Lease:
        """A lease that is already over: every move made with it is refused."""
        lease = cls(threading.Condition(), name, priority, resumable=False)
        lease._state = _State.ENDED
        lease.end_reason = "denied"
        return lease

    @property
    def autonomous(self) -> bool:
        """True for curiosity, social and exploration: nobody asked for these moves."""
        return self.priority >= AUTONOMOUS_FROM

    @property
    def active(self) -> bool:
        return self._state is _State.ACTIVE

    @property
    def waiting(self) -> bool:
        return self._state is _State.WAITING

    @property
    def ended(self) -> bool:
        return self._state is _State.ENDED

    def check(self) -> None:
        """Raise `Preempted` unless the lease is active. Behaviours call it between steps."""
        if self._state is not _State.ACTIVE:
            raise Preempted(self.end_reason or "preempted")

    def kick(self) -> None:
        """Tell the arbiter the behaviour is alive (heartbeat)."""
        self._last_kick = time.monotonic()

    def wait_until_active(self, timeout: float | None = None) -> bool:
        """Block while pre-empted. True once the body is back, False if the lease ended or timed out."""
        with self._cond:
            self._cond.wait_for(lambda: self._state is not _State.WAITING, timeout)
            return self._state is _State.ACTIVE

    def release(self) -> None:
        if self._arbiter is not None:
            self._arbiter.release(self)

    def __repr__(self) -> str:
        return f"Lease({self.name!r}, {self.priority.name}, {self._state.value})"


class Arbiter:
    """Owns the body. One holder at a time; more important requests pre-empt."""

    def __init__(
        self,
        safety: SafetyController,
        actions: ActionPlayer | None = None,
        *,
        lease_timeout_s: float = DEFAULT_LEASE_TIMEOUT_S,
        max_waiting: int = MAX_WAITING,
    ) -> None:
        self.safety = safety
        self.actions = actions
        self.lease_timeout_s = lease_timeout_s
        self.max_waiting = max(0, max_waiting)
        self._cond = threading.Condition()
        self._holder: Lease | None = None
        self._waiting: list[Lease] = []
        self._ids = itertools.count(1)
        self._closed = False
        self._stop = threading.Event()
        self._timer: threading.Thread | None = None
        safety.add_halt_listener(self._on_safety)

    # --- inspection -----------------------------------------------------------

    @property
    def holder(self) -> Lease | None:
        return self._holder

    @property
    def waiting(self) -> list[Lease]:
        with self._cond:
            return list(self._waiting)

    # --- requesting and releasing ---------------------------------------------

    def request(
        self,
        name: str,
        priority: Priority,
        *,
        on_preempt: Callable[[Lease], None] | None = None,
        on_resume: Callable[[Lease], None] | None = None,
        resumable: bool = True,
    ) -> Lease | None:
        """Ask for the body. None when the holder is at least as important (retry later)."""
        priority = Priority(priority)
        if priority is Priority.SAFETY:
            raise ValueError("safety is the safety controller; it holds no lease")
        if self._closed or (priority > SAFETY_REVOKES_BELOW and self._autonomy_blocked()):
            return None
        if priority >= AUTONOMOUS_FROM and self.safety.paused:
            return None
        preempted: Lease | None = None
        with self._cond:
            holder = self._holder
            if holder is not None:
                if priority >= holder.priority:
                    return None
                preempted = holder
                self._demote(holder)
            elif self._waiting and priority >= min(w.priority for w in self._waiting):
                return None  # a waiting behaviour is about to get the body back
            lease = Lease(
                self._cond,
                name,
                priority,
                on_preempt=on_preempt,
                on_resume=on_resume,
                resumable=resumable,
                id=next(self._ids),
                arbiter=self,
            )
            self._holder = lease
            self._cond.notify_all()
        if preempted is not None:
            # Before the new holder sees its lease, so its first move is not cancelled.
            print(f"[laika] behaviour {preempted.name} preempted by {name}")
            self._after_preempt(preempted, "preempted")
        if priority >= HEARTBEAT_FROM:
            self._ensure_timer()
        return lease

    def release(self, lease: Lease) -> None:
        """Give the body back. The most important waiting behaviour gets it. Idempotent."""
        resumed: Lease | None = None
        with self._cond:
            if lease.ended:
                return
            if lease is self._holder:
                self._holder = None
                resumed = self._promote()
            elif lease in self._waiting:
                self._waiting.remove(lease)
            self._end(lease, "released")
            self._cond.notify_all()
        self._resumed(resumed)

    def close(self) -> None:
        self._closed = True
        self._stop.set()
        timer = self._timer
        if timer is not None and timer is not threading.current_thread():
            timer.join(JOIN_TIMEOUT_S)

    # --- state changes (arbiter lock held) -------------------------------------

    def _demote(self, holder: Lease) -> None:
        if holder.resumable and len(self._waiting) < self.max_waiting:
            holder._state = _State.WAITING
            self._waiting.append(holder)
        else:
            self._end(holder, "preempted")

    def _end(self, lease: Lease, reason: str) -> None:
        lease._state = _State.ENDED
        lease.end_reason = reason

    def _promote(self) -> Lease | None:
        if not self._waiting:
            return None
        best = min(self._waiting, key=lambda lease: lease.priority)
        self._waiting.remove(best)
        best._state = _State.ACTIVE
        best.kick()
        self._holder = best
        return best

    # --- reactions (called without the lock) ------------------------------------

    def _after_preempt(self, old: Lease, reason: str) -> None:
        """Stop what the old holder was doing, then tell it."""
        try:
            self.safety.stop(reason)
            if self.actions is not None:
                self.actions.cancel_pending()
        except Exception:  # noqa: BLE001 - the new holder must still get its lease
            log.exception("could not stop after preempting %s", old.name)
        self._call(old.on_preempt, old)

    def _resumed(self, lease: Lease | None) -> None:
        if lease is not None:
            print(f"[laika] behaviour {lease.name} resumed")
            self._call(lease.on_resume, lease)

    @staticmethod
    def _call(callback: Callable[[Lease], None] | None, lease: Lease) -> None:
        if callback is None:
            return
        try:
            callback(lease)
        except Exception:  # noqa: BLE001 - a failing behaviour callback must not break the arbiter
            log.exception("behaviour callback failed for %s", lease.name)

    # --- safety ------------------------------------------------------------------

    def _autonomy_blocked(self) -> bool:
        safety = self.safety
        return (
            safety.emergency_stopped
            or safety.watchdog_expired
            or not safety.ai_available
            or not safety.hardware_ok
            or safety.battery_state is BatteryState.CRITICAL
        )

    def _on_safety(self, reason: str) -> None:
        """A safety outcome that must stop autonomy. Called from safety threads: no blocking."""
        revoked: list[Lease] = []
        # A pause ends the autonomous behaviours only; a follow-up or an observation is not "moving around".
        floor = AUTONOMOUS_FROM - 1 if reason == "paused" else SAFETY_REVOKES_BELOW
        with self._cond:
            holder = self._holder
            if holder is not None and holder.priority > floor:
                self._holder = None
                revoked.append(holder)
            for lease in list(self._waiting):
                if lease.priority > floor:
                    self._waiting.remove(lease)
                    revoked.append(lease)
            for lease in revoked:
                self._end(lease, reason)
            if revoked:
                self._cond.notify_all()
        if revoked:
            threading.Thread(
                target=self._notify_revoked, args=(reason, revoked), name="laika-revoke", daemon=True
            ).start()

    def _notify_revoked(self, reason: str, revoked: list[Lease]) -> None:
        names = ", ".join(lease.name for lease in revoked)
        print(f"[laika] behaviour {names} stopped by safety ({reason})")
        if self.actions is not None:
            try:
                self.actions.cancel_pending()
            except Exception:  # noqa: BLE001
                log.exception("could not cancel gestures")
        for lease in revoked:
            self._call(lease.on_preempt, lease)

    # --- heartbeat ----------------------------------------------------------------

    def _ensure_timer(self) -> None:
        with self._cond:
            if self._timer is not None or self._closed:
                return
            self._timer = threading.Thread(target=self._run_timer, name="laika-leases", daemon=True)
            self._timer.start()

    def _run_timer(self) -> None:
        period = max(0.02, self.lease_timeout_s / 4)
        while not self._stop.wait(period):
            try:
                self._expire_stalled()
            except Exception:  # noqa: BLE001 - the timer must survive anything
                log.exception("lease check failed")

    def _expire_stalled(self) -> None:
        with self._cond:
            holder = self._holder
            if (
                holder is None
                or holder.priority < HEARTBEAT_FROM
                or time.monotonic() - holder._last_kick <= self.lease_timeout_s
            ):
                return
            self._holder = None
            self._end(holder, "lease_expired")
            self._cond.notify_all()
        print(f"[laika] behaviour {holder.name} stalled: lease expired, robot stopped")
        self._after_preempt(holder, "lease_expired")  # stop first, so a resumed move survives
        with self._cond:
            resumed = self._promote() if self._holder is None else None
            self._cond.notify_all()
        self._resumed(resumed)
