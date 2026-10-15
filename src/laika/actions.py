"""Laika's gestures and the worker that plays them while she talks.

Gestures are adapted from SunFounder's `picarx/preset_actions.py`, rewritten
against the `Robot` protocol. The AI may only pick names from `ACTIONS`;
anything else is dropped. Driving goes through `MovementGuard`.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from laika.robot import Robot
from laika.safety import MovementGuard


@dataclass(frozen=True)
class Action:
    name: str
    description: str
    run: Callable[["ActionContext"], None]


@dataclass
class ActionContext:
    robot: Robot
    guard: MovementGuard
    honk_volume: int = 80


def _sequence(robot: Robot, poses: list[tuple[float, float]], delay: float) -> None:
    for pan, tilt in poses:
        robot.look(pan, tilt)
        time.sleep(delay)


def nod(ctx: ActionContext) -> None:
    _sequence(ctx.robot, [(0, 5), (0, -25), (0, 5), (0, -25), (0, 0)], 0.15)


def shake_head(ctx: ActionContext) -> None:
    _sequence(ctx.robot, [(50, 0), (-40, 0), (30, 0), (-20, 0), (10, 0), (0, 0)], 0.12)


def look_left(ctx: ActionContext) -> None:
    ctx.robot.look(60, 0)


def look_right(ctx: ActionContext) -> None:
    ctx.robot.look(-60, 0)


def look_up(ctx: ActionContext) -> None:
    ctx.robot.look(0, 35)


def look_down(ctx: ActionContext) -> None:
    ctx.robot.look(0, -25)


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


def forward(ctx: ActionContext) -> None:
    ctx.guard.drive(forward=True, duration_s=1.0)


def backward(ctx: ActionContext) -> None:
    ctx.guard.drive(forward=False, duration_s=0.6)


def spin_left(ctx: ActionContext) -> None:
    ctx.guard.drive(forward=True, duration_s=1.2, steer=-30)


def spin_right(ctx: ActionContext) -> None:
    ctx.guard.drive(forward=True, duration_s=1.2, steer=30)


ACTIONS: dict[str, Action] = {
    a.name: a
    for a in [
        Action("nod", "annuisci con la testa (sì)", nod),
        Action("shake_head", "scuoti la testa (no)", shake_head),
        Action("look_left", "guarda a sinistra", look_left),
        Action("look_right", "guarda a destra", look_right),
        Action("look_up", "guarda in alto", look_up),
        Action("look_down", "guarda in basso", look_down),
        Action("look_forward", "guarda dritto davanti", look_forward),
        Action("look_around", "guardati intorno", look_around),
        Action("curious", "inclina la testa, curiosa", curious),
        Action("think", "posa pensierosa", think),
        Action("wave", "saluta muovendo le ruote davanti", wave),
        Action("celebrate", "festeggia muovendo testa e ruote", celebrate),
        Action("dance", "balla con testa e ruote (quando ti chiedono di ballare)", dance),
        Action("sad","abbassa la testa, triste", sad),
        Action("honk", "suona il clacson", honk),
        Action("forward", "avanza un pochino (solo se richiesto)", forward),
        Action("backward", "indietreggia un pochino (solo se richiesto)", backward),
        Action("spin_left", "fai una curva a sinistra (solo se richiesto)", spin_left),
        Action("spin_right", "fai una curva a destra (solo se richiesto)", spin_right),
    ]
}

DRIVING_ACTIONS = {"forward", "backward", "spin_left", "spin_right"}
MAX_ACTIONS_PER_REPLY = 4


def available_actions(allow_driving: bool) -> dict[str, Action]:
    return {
        name: action
        for name, action in ACTIONS.items()
        if allow_driving or name not in DRIVING_ACTIONS
    }


def validate_actions(names: list[str], allowed: dict[str, Action]) -> list[str]:
    """Keep known action names only, in order, capped per reply."""
    valid = [n.strip().lower() for n in names if isinstance(n, str)]
    return [n for n in valid if n in allowed][:MAX_ACTIONS_PER_REPLY]


class ActionPlayer:
    """Plays gestures on one background thread, the only one that drives the hardware."""

    def __init__(self, ctx: ActionContext, allowed: dict[str, Action]) -> None:
        self.ctx = ctx
        self.allowed = allowed
        self._waiting: deque[str] = deque()
        self._running = False
        self._closed = False
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._run, name="laika-actions", daemon=True)
        self._thread.start()

    def play(self, *names: str) -> None:
        """Replace whatever is still waiting with these gestures."""
        with self._cond:
            self._waiting.clear()
            self._waiting.extend(validate_actions(list(names), self.allowed))
            self._cond.notify_all()

    def wait_done(self, timeout: float = 15.0) -> bool:
        with self._cond:
            return self._cond.wait_for(lambda: not self._waiting and not self._running, timeout)

    def close(self) -> None:
        with self._cond:
            self._waiting.clear()
            self._closed = True
            self._cond.notify_all()
        self._thread.join(timeout=5)
        self.ctx.robot.stop()

    def _run(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._waiting or self._closed)
                if self._closed:
                    return
                name = self._waiting.popleft()
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
