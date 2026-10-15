"""Laika's gestures and the worker that plays them while she talks.

Gestures are adapted from SunFounder's `picarx/preset_actions.py`, rewritten
against the `Robot` protocol. The AI may only pick names from `ACTIONS`;
anything else is dropped. Driving goes through the `SafetyController`.
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from laika.commands import MAX_MOVE_SPEED, Direction, MoveCommand
from laika.robot import (
    CAMERA_PAN,
    CAMERA_TILT,
    DRIVE,
    SPEAKER,
    STEER,
    ULTRASONIC,
    Robot,
)
from laika.safety import SafetyController

PAN = frozenset({CAMERA_PAN})
TILT = frozenset({CAMERA_TILT})
LOOK = PAN | TILT
STEERING = frozenset({STEER})
DRIVING = frozenset({DRIVE, STEER})


@dataclass(frozen=True)
class Action:
    name: str
    description: str
    run: Callable[["ActionContext"], None]
    # Capabilities the robot must advertise for the AI to be offered this action.
    requires: frozenset[str]


@dataclass
class ActionContext:
    robot: Robot
    safety: SafetyController
    honk_volume: int = 80


def _sequence(robot: Robot, poses: list[tuple[float, float]], delay: float) -> None:
    for pan, tilt in poses:
        robot.look(pan, tilt)
        time.sleep(delay)


def nod(ctx: ActionContext) -> None:
    _sequence(ctx.robot, [(0, 5), (0, -25), (0, 5), (0, -25), (0, 0)], 0.15)


def shake_head(ctx: ActionContext) -> None:
    _sequence(ctx.robot, [(50, 0), (-40, 0), (30, 0), (-20, 0), (10, 0), (0, 0)], 0.12)


# Fixed (pan, tilt) poses, shared with the look_* agent tools. Negative pan is left.
LOOK_POSES: dict[str, tuple[float, float]] = {
    "left": (-60, 0),
    "right": (60, 0),
    "up": (0, 35),
    "down": (0, -25),
}


def look_left(ctx: ActionContext) -> None:
    ctx.robot.look(*LOOK_POSES["left"])


def look_right(ctx: ActionContext) -> None:
    ctx.robot.look(*LOOK_POSES["right"])


def look_up(ctx: ActionContext) -> None:
    ctx.robot.look(*LOOK_POSES["up"])


def look_down(ctx: ActionContext) -> None:
    ctx.robot.look(*LOOK_POSES["down"])


def look_forward(ctx: ActionContext) -> None:
    ctx.robot.look(0, 0)


def look_around(ctx: ActionContext) -> None:
    _sequence(ctx.robot, [(60, 10), (0, 10), (-60, 10), (0, 0)], 0.5)


def think(ctx: ActionContext) -> None:
    ctx.robot.look(20, 20)
    ctx.robot.steer(10)


def curious(ctx: ActionContext) -> None:
    _sequence(ctx.robot, [(15, 10), (-15, 10), (0, 15)], 0.25)


def wave(ctx: ActionContext) -> None:
    ctx.robot.look(0, 20)
    for _ in range(3):
        ctx.robot.steer(-25)
        time.sleep(0.12)
        ctx.robot.steer(25)
        time.sleep(0.12)
    ctx.robot.steer(0)
    ctx.robot.look(0, 0)


def celebrate(ctx: ActionContext) -> None:
    for side in (1, -1):
        ctx.robot.look(60 * side, 20)
        ctx.robot.steer(30 * side)
        time.sleep(0.3)
        ctx.robot.look(30 * side, 20)
        ctx.robot.steer(10 * side)
        time.sleep(0.12)
        ctx.robot.look(60 * side, 20)
        ctx.robot.steer(30 * side)
        time.sleep(0.3)
    ctx.robot.steer(0)
    ctx.robot.look(0, 0)


def dance(ctx: ActionContext) -> None:
    # Head and front wheels only, so it is safe anywhere and works with --no-driving.
    for _ in range(2):
        for side in (1, -1):
            ctx.robot.look(45 * side, 25)
            ctx.robot.steer(25 * side)
            time.sleep(0.25)
            ctx.robot.look(20 * side, -10)
            time.sleep(0.15)
        _sequence(ctx.robot, [(0, 30), (0, -15), (0, 30), (0, 0)], 0.15)
    for angle in (-30, 30, -30, 30):
        ctx.robot.steer(angle)
        time.sleep(0.1)
    ctx.robot.steer(0)
    ctx.robot.look(0, 0)


def sad(ctx: ActionContext) -> None:
    _sequence(ctx.robot, [(0, 10), (0, -20), (0, -10), (0, -25)], 0.2)
    time.sleep(1.0)
    ctx.robot.look(0, 0)


def honk(ctx: ActionContext) -> None:
    ctx.robot.honk(ctx.honk_volume)
    time.sleep(0.6)


def _move(ctx: ActionContext, direction: Direction, duration_ms: int, steering: float = 0) -> None:
    # The controller caps the speed to its configured drive speed.
    ctx.safety.execute(
        MoveCommand(
            direction=direction,
            speed=MAX_MOVE_SPEED,
            duration_ms=duration_ms,
            steering=steering,
        )
    )


def forward(ctx: ActionContext) -> None:
    _move(ctx, Direction.FORWARD, 1000)


def backward(ctx: ActionContext) -> None:
    _move(ctx, Direction.BACKWARD, 600)


def spin_left(ctx: ActionContext) -> None:
    _move(ctx, Direction.FORWARD, 1200, steering=-30)


def spin_right(ctx: ActionContext) -> None:
    _move(ctx, Direction.FORWARD, 1200, steering=30)


ACTIONS: dict[str, Action] = {
    a.name: a
    for a in [
        Action("nod", "annuisci con la testa (sì)", nod, TILT),
        Action("shake_head", "scuoti la testa (no)", shake_head, PAN),
        Action("look_left", "guarda a sinistra", look_left, PAN),
        Action("look_right", "guarda a destra", look_right, PAN),
        Action("look_up", "guarda in alto", look_up, TILT),
        Action("look_down", "guarda in basso", look_down, TILT),
        Action("look_forward", "guarda dritto davanti", look_forward, LOOK),
        Action("look_around", "guardati intorno", look_around, LOOK),
        Action("curious", "inclina la testa, curiosa", curious, LOOK),
        Action("think", "posa pensierosa", think, LOOK | STEERING),
        Action("wave", "saluta muovendo le ruote davanti", wave, TILT | STEERING),
        Action("celebrate", "festeggia muovendo testa e ruote", celebrate, LOOK | STEERING),
        Action(
            "dance",
            "balla con testa e ruote (quando ti chiedono di ballare)",
            dance,
            LOOK | STEERING,
        ),
        Action("sad", "abbassa la testa, triste", sad, TILT),
        Action("honk", "suona il clacson", honk, frozenset({SPEAKER})),
        Action("forward", "avanza un pochino (solo se richiesto)", forward, DRIVING | {ULTRASONIC}),
        Action("backward", "indietreggia un pochino (solo se richiesto)", backward, DRIVING),
        Action("spin_left", "fai una curva a sinistra (solo se richiesto)", spin_left, DRIVING),
        Action("spin_right", "fai una curva a destra (solo se richiesto)", spin_right, DRIVING),
    ]
}

DRIVING_ACTIONS = {"forward", "backward", "spin_left", "spin_right"}
MAX_ACTIONS_PER_REPLY = 4


def available_actions(capabilities: frozenset[str], allow_driving: bool) -> dict[str, Action]:
    """The gestures this robot can do and the config allows: the only ones the AI is offered."""
    return {
        name: action
        for name, action in ACTIONS.items()
        if (allow_driving or name not in DRIVING_ACTIONS) and action.requires <= capabilities
    }


def validate_actions(names: list[str], allowed: dict[str, Action]) -> list[str]:
    """Keep known action names only, in order, capped per reply."""
    valid = [n.strip().lower() for n in names if isinstance(n, str)]
    return [n for n in valid if n in allowed][:MAX_ACTIONS_PER_REPLY]


TALK_PAN_RANGE = 8
TALK_TILT_RANGE = 6
TALK_STEP_S = (0.25, 0.55)


class ActionPlayer:
    """Plays gestures on one background thread, the only one that drives the hardware.

    While Laika is talking and no gesture is playing, the head makes small
    random movements around wherever it points, then settles back there.
    """

    def __init__(
        self, ctx: ActionContext, allowed: dict[str, Action], *, talk_motion: bool = True
    ) -> None:
        self.ctx = ctx
        self.allowed = allowed
        self.talk_motion = talk_motion
        self._waiting: deque[str] = deque()
        self._running = False
        self._closed = False
        self._talking = False
        # Where the head pointed when the talking motion started; None when settled.
        self._talk_base: tuple[float, float] | None = None
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._run, name="laika-actions", daemon=True)
        self._thread.start()

    def play(self, *names: str) -> None:
        """Replace whatever is still waiting with these gestures."""
        with self._cond:
            self._waiting.clear()
            self._waiting.extend(validate_actions(list(names), self.allowed))
            self._cond.notify_all()

    def enqueue(self, *names: str) -> None:
        """Queue gestures after the ones already waiting (the agent's gesture tool)."""
        with self._cond:
            self._waiting.extend(validate_actions(list(names), self.allowed))
            self._cond.notify_all()

    def cancel_pending(self) -> None:
        """Drop the gestures still waiting (AI outage). A running gesture finishes;
        any drive it starts is refused by the controller."""
        with self._cond:
            self._waiting.clear()
            self._cond.notify_all()

    def set_talking(self, talking: bool) -> None:
        with self._cond:
            self._talking = talking and self.talk_motion
            self._cond.notify_all()

    def wait_done(self, timeout: float = 15.0) -> bool:
        """Wait until gestures are done and, once talking stops, the head has settled."""
        with self._cond:
            return self._cond.wait_for(
                lambda: not self._waiting
                and not self._running
                and (self._talking or self._talk_base is None),
                timeout,
            )

    def close(self) -> None:
        with self._cond:
            self._waiting.clear()
            self._closed = True
            self._cond.notify_all()
        self._thread.join(timeout=5)
        self.ctx.robot.stop()

    def _has_work(self) -> bool:
        return bool(
            self._waiting or self._closed or self._talking or self._talk_base is not None
        )

    def _talk_step(self) -> None:
        """Called with the lock held, between gestures while talking."""
        if not self._talking:
            # Talking ended: settle back where the head was pointing.
            self.ctx.robot.look(*self._talk_base)
            self._talk_base = None
            self._cond.notify_all()
            return
        if self._talk_base is None:
            self._talk_base = self.ctx.robot.head_pose()
        pan, tilt = self._talk_base
        self.ctx.robot.look(
            pan + random.uniform(-TALK_PAN_RANGE, TALK_PAN_RANGE),
            tilt + random.uniform(-TALK_TILT_RANGE / 2, TALK_TILT_RANGE),
        )
        # Waiting on the condition lets a new gesture or the end of speech cut in.
        self._cond.wait_for(
            lambda: self._waiting or self._closed or not self._talking,
            random.uniform(*TALK_STEP_S),
        )

    def _run(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(self._has_work)
                if self._closed:
                    return
                if not self._waiting:
                    try:
                        self._talk_step()
                    except Exception as error:
                        print(f"[laika] talking motion failed: {error}")
                        self._talk_base = None
                        self._cond.notify_all()
                    continue
                name = self._waiting.popleft()
                # A gesture moves the head; talking motion restarts from its end pose.
                self._talk_base = None
                self._running = True
            try:
                self.allowed[name].run(self.ctx)
            except Exception as error:  # A failed gesture must not kill the worker.
                print(f"[laika] gesture {name} failed: {error}")
                self.ctx.robot.stop()
            finally:
                with self._cond:
                    self._running = False
                    self._cond.notify_all()
