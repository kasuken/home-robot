"""Status LEDs: what she is doing, at a glance, from across the room (docs/leds.md).

Two LEDs, each shows one pattern at a time, chosen from what she is doing:

- the **status LED** (red, by default the Raspberry Pi's own PWR LED): a heartbeat (two quick thumps, a pause)
  while she is fine; a slow blink when the battery is LOW, a fast blink when it is CRITICAL; solid on after
  an emergency stop (the wheels are refused until it is lifted);
- the **activity LED** (by default the Robot HAT's user LED): a fast blink while she is thinking (a request
  to the model is in flight), a slow blink while she is learning a new skill, solid on while she is awake in a
  conversation, off otherwise.

Nothing here can reach the motors. The LEDs are written from one small thread that polls the state sources
(plain callables) every `TICK_S`, so no other thread ever waits on a LED and a failing LED never breaks a
turn. A write happens only when the wanted value changes.

Which hardware: `hat` is the Robot HAT's user LED (`hardware.HatLed`, the one adapter that imports SunFounder), `pi_red` and
`pi_green` are the Raspberry Pi's PWR and ACT LEDs through /sys/class/leds (root; her normal trigger is saved
and put back when she stops). The HAT's *charge* LED is wired to the charger chip, not to a pin, so it cannot
be driven from software, and a robot with no such LED simply gets `NullLed`.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Protocol

log = logging.getLogger(__name__)

TICK_S = 0.02
HEARTBEAT_PERIOD_S = 1.4
SYSFS_ROOT = Path("/sys/class/leds")
SYSFS_NAMES = {"pi_red": ("PWR", "led1"), "pi_green": ("ACT", "led0")}
CHOICES = ("hat", "pi_red", "pi_green", "none")

# Battery states (laika.battery.BatteryState values) as text, so this module needs nothing from it.
BATTERY_LOW = "low"
BATTERY_CRITICAL = "critical"


def pattern_value(name: str, t: float) -> bool:
    """Whether the LED is on at time `t` seconds, for a named pattern."""
    if name == "solid":
        return True
    if name == "heartbeat":  # lub-dub: two 100 ms thumps, then rest
        phase = t % HEARTBEAT_PERIOD_S
        return phase < 0.10 or 0.20 <= phase < 0.30
    if name == "fast":  # 5 Hz
        return t % 0.2 < 0.1
    if name == "alarm":  # 4 Hz with short flashes: a stuttering, urgent rhythm
        return t % 0.25 < 0.08
    if name == "slow":  # 1 Hz
        return t % 1.0 < 0.5
    return False


def status_pattern(*, stopped: bool, battery: str | None) -> str:
    """What the status LED shows: the most urgent thing wins."""
    if stopped:
        return "solid"
    if battery == BATTERY_CRITICAL:
        return "alarm"
    if battery == BATTERY_LOW:
        return "slow"
    return "heartbeat"


def activity_pattern(*, thinking: bool, learning: bool, awake: bool) -> str:
    if thinking:
        return "fast"
    if learning:
        return "slow"
    return "solid" if awake else "off"


class Led(Protocol):
    def set(self, on: bool) -> None: ...
    def close(self) -> None: ...


class NullLed:
    def set(self, on: bool) -> None:
        pass

    def close(self) -> None:
        pass


class SysfsLed:
    """A Raspberry Pi LED under /sys/class/leds. Takes the LED over (trigger "none") and gives it back."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._trigger_file = directory / "trigger"
        self._brightness = directory / "brightness"
        self._saved = self._active_trigger()
        self._trigger_file.write_text("none")  # raises OSError without permission: the caller falls back

    def _active_trigger(self) -> str | None:
        try:
            text = self._trigger_file.read_text()
        except OSError:
            return None
        for word in text.split():
            if word.startswith("[") and word.endswith("]"):
                return word[1:-1]
        return None

    def set(self, on: bool) -> None:
        self._brightness.write_text("1" if on else "0")

    def close(self) -> None:
        try:
            if self._saved:
                self._trigger_file.write_text(self._saved)
        except OSError:
            pass


def open_led(kind: str, root: Path = SYSFS_ROOT) -> Led:
    """The LED named `kind` (see CHOICES); NullLed for "none" and when the hardware is not there (said once)."""
    if kind == "none":
        return NullLed()
    try:
        if kind == "hat":
            from laika.hardware import HatLed  # the adapter that may import SunFounder (hardware.py)

            return HatLed()
        for name in SYSFS_NAMES.get(kind, ()):
            if (root / name).is_dir():
                return SysfsLed(root / name)
        raise OSError(f"no {kind} LED under {root}")
    except (ImportError, OSError) as error:
        print(f"[laika] leds: {kind} is not available ({error}); that LED stays off")
        return NullLed()


class StatusLeds:
    """Drives the two LEDs from the state sources, on one thread."""

    def __init__(
        self,
        status: Led,
        activity: Led,
        *,
        thinking: Callable[[], bool] = lambda: False,
        awake: Callable[[], bool] = lambda: False,
        learning: Callable[[], bool] = lambda: False,
        stopped: Callable[[], bool] = lambda: False,
        battery: Callable[[], str | None] = lambda: None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.status = status
        self.activity = activity
        self._sources = (thinking, awake, learning, stopped, battery)
        self._clock = clock
        self._shown: list[bool | None] = [None, None]
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def patterns(self) -> tuple[str, str]:
        """(status pattern, activity pattern) for the state now. A source that fails counts as off."""
        thinking, awake, learning, stopped, battery = (self._read(source) for source in self._sources)
        return (
            status_pattern(stopped=bool(stopped), battery=battery),
            activity_pattern(thinking=bool(thinking), learning=bool(learning), awake=bool(awake)),
        )

    @staticmethod
    def _read(source: Callable[[], object]) -> object:
        try:
            return source()
        except Exception:  # noqa: BLE001 - a broken source must not stop the LEDs
            return None

    def tick(self, now: float | None = None) -> None:
        """Compute and write both LEDs once (the thread calls this; tests call it with a time)."""
        t = self._clock() if now is None else now
        for index, (led, name) in enumerate(zip((self.status, self.activity), self.patterns())):
            wanted = pattern_value(name, t)
            if wanted != self._shown[index]:
                try:
                    led.set(wanted)
                    self._shown[index] = wanted
                except Exception:  # noqa: BLE001 - a failing LED must not break anything else
                    log.exception("led write failed")
                    self._shown[index] = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="laika-leds", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(TICK_S):
            self.tick()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(1.0)
        for led in (self.status, self.activity):
            try:
                led.close()
            except Exception:  # noqa: BLE001
                pass
