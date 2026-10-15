"""Laika starts a conversation by herself (#40): she asks a pending question when a person is near.

The loop this closes: she notices something new (curiosity), looks into it and queues a question
(`PendingQuestions`), and later, when someone is near and the deterministic interaction policy
(#39) agrees, she turns to them, asks, and keeps the answer as memory.

This module is only the trigger. It listens for a person arriving, for something interesting and for
a periodic re-check while someone is near, and every trigger asks `InteractionPolicy.decide()`.
Nothing else can start a proactive interaction: the agent has no tool for it. When the policy says yes
it records the attempt FIRST (so a crash cannot cause a retry storm) and hands a `ProactiveRequest` to
the `Conversation`, which runs it on its own main thread. This thread never speaks or moves anything.

Refusals are logged by the policy, once per change of reason.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Protocol

from laika.behaviour import Priority
from laika.config import InteractionConfig
from laika.events import Event, EventBus, EventType
from laika.interaction_policy import Context, InteractionPolicy

if TYPE_CHECKING:
    from laika.behaviour import Arbiter
    from laika.investigate import PendingQuestions
    from laika.presence import PresenceTracker

log = logging.getLogger(__name__)

SLICE_S = 0.2  # how often the worker looks at the clock; events wake it at once
CLOSE_TIMEOUT_S = 2.0
# Just after the settle time, so the check that follows an arrival finds the person settled.
SETTLE_MARGIN_S = 0.05
REASON = "pending_question"


@dataclass(frozen=True)
class ProactiveRequest:
    """One question Laika is about to ask, and where the person is."""

    question_id: str
    text: str
    reason: str = REASON
    trigger: str = "person_detected"
    pan: float = 0.0
    tilt: float = 0.0
    room: str | None = None
    created: float = field(default_factory=time.monotonic)


class ProactiveTarget(Protocol):
    """What the Conversation offers: a thread-safe hand-off, and whether it is busy."""

    @property
    def awake(self) -> bool: ...

    def request_proactive(self, request: ProactiveRequest) -> bool: ...


class ProactiveInteraction:
    def __init__(
        self,
        policy: InteractionPolicy,
        presence: PresenceTracker,
        questions: PendingQuestions,
        conversation: ProactiveTarget,
        arbiter: Arbiter | None,
        events: EventBus,
        config: InteractionConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy = policy
        self.presence = presence
        self.questions = questions
        self.conversation = conversation
        self.arbiter = arbiter
        self.events = events
        self.config = config
        self._clock = clock
        self._cond = threading.Condition()
        self._check_lock = threading.Lock()
        self._trigger: str | None = None
        self._next_check = 0.0
        self._settle_check: float | None = None
        self._direction: tuple[float, float] = (0.0, 0.0)
        self._room: str | None = None
        self._closed = False
        self._thread: threading.Thread | None = None
        self._unsubscribe: Callable[[], None] | None = None

    # --- lifecycle ---------------------------------------------------------------

    def start(self) -> None:
        """Begin listening. With `[interaction] proactive = false` nothing runs at all."""
        if not self.config.proactive:
            log.info("proactive questions are switched off ([interaction] proactive)")
            return
        with self._cond:
            if self._thread is not None or self._closed:
                return
            self._next_check = self._clock() + self.config.recheck_s
            self._thread = threading.Thread(target=self._run, name="proactive", daemon=True)
            self._thread.start()
        self._unsubscribe = self.events.subscribe(
            self._on_event,
            EventType.PERSON_DETECTED,
            EventType.PERSON_LOST,
            EventType.INTERESTING_CHANGE_DETECTED,
        )

    def close(self, timeout_s: float = CLOSE_TIMEOUT_S) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        with self._cond:
            self._closed = True
            self._cond.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout_s)

    # --- triggers ----------------------------------------------------------------

    def _on_event(self, event: Event) -> None:
        """Runs on the event thread: note what happened and wake the worker, nothing more."""
        now = self._clock()
        with self._cond:
            if event.type is EventType.PERSON_DETECTED:
                data = event.data
                try:
                    self._direction = (float(data.get("pan", 0.0) or 0.0), float(data.get("tilt", 0.0) or 0.0))
                except (TypeError, ValueError):
                    pass
                self._room = data.get("room")
                self._trigger = "person_detected"
                # The policy waits `settle_s` before it lets her speak: look again right after.
                self._settle_check = now + self.config.settle_s + SETTLE_MARGIN_S
            elif event.type is EventType.INTERESTING_CHANGE_DETECTED and self.presence.present:
                self._trigger = self._trigger or "interesting_change"
            self._cond.notify_all()

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._closed and self._trigger is None:
                    now = self._clock()
                    settled = self._settle_check is not None and now >= self._settle_check
                    if self.presence.present and (settled or now >= self._next_check):
                        self._trigger = "recheck"
                        break
                    self._cond.wait(SLICE_S)
                if self._closed:
                    return
                trigger, self._trigger = self._trigger, None
                now = self._clock()
                self._next_check = now + self.config.recheck_s
                if self._settle_check is not None and now >= self._settle_check:
                    self._settle_check = None
            try:
                self.check(trigger or "recheck")
            except Exception:  # noqa: BLE001 - a failed check must not end the loop
                log.exception("proactive check failed")

    # --- the decision ------------------------------------------------------------

    def check(self, trigger: str = "recheck") -> bool:
        """Ask the policy now. True when a request was handed to the conversation."""
        with self._check_lock:
            pending = self.questions.pending()
            decision = self.policy.decide(
                Context(pending_questions=len(pending), conversation_active=self._body_busy())
            )
            if not decision.allowed:
                return False
            if not pending:
                return False  # interesting, but there is nothing to ask yet
            question = pending[0]  # the oldest: the one the agent and the memory selector show first
            # First: whatever happens next, the cooldown and the hourly limit already count it.
            self.policy.record_attempt()
            pan, tilt = self._where()
            request = ProactiveRequest(
                question.id, question.text, REASON, trigger, pan=pan, tilt=tilt, room=self._room or question.room
            )
            handed = self.conversation.request_proactive(request)
            if not handed:
                log.info("proactive question %s not started: the conversation is busy", question.id)
            return handed

    def _body_busy(self) -> bool:
        if getattr(self.conversation, "awake", False):
            return True
        holder = self.arbiter.holder if self.arbiter is not None else None
        return holder is not None and holder.priority <= Priority.CONVERSATION

    def _where(self) -> tuple[float, float]:
        """Where the head pointed when the person was last seen: the tracker knows, else the event did."""
        direction = getattr(self.presence, "direction", None)
        if direction is not None:
            return direction
        with self._cond:
            return self._direction
