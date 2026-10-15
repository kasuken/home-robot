"""Know the battery level (#50): sample the voltage slowly, smooth it, and say OK, LOW or CRITICAL.

`BatteryMonitor` runs on its own small timer thread, never on the safety thread: it reads
`Robot.battery_voltage()` every `[battery] sample_s` seconds and keeps the last few readings.
The level is their median, so one noisy reading changes nothing. The voltage of a pack sags while the
motors pull current, so a reading taken while the wheels turn is skipped (and tried again a little
later); only after a long drive does a loaded reading count, so an empty pack is never ignored.

The state has hysteresis, so a voltage hovering around a threshold does not flap:

    OK        -> LOW       below low_v
    LOW       -> OK        above ok_v
    any       -> CRITICAL  below critical_v
    CRITICAL  -> LOW       at or above low_v (or OK, above ok_v)

Every change publishes BatteryLow, BatteryCritical or BatteryOk and calls the listeners; the app wires
the safety controller to them (`SafetyController.set_battery_state`), which refuses autonomous driving
when LOW and every wheel move when CRITICAL. The thresholds are for two 18650 cells in series and must
be calibrated on the robot against a multimeter (docs/safety-calibration.md).
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import deque
from enum import Enum
from typing import Any, Callable

from laika.config import BatteryConfig
from laika.events import EventBus, EventType, emit
from laika.robot import Robot

log = logging.getLogger(__name__)

SMOOTHING = 5  # the level is the median of this many readings
MIN_READINGS = 3  # no state change before this many readings (a single bad one must not decide)
WARMUP_S = 2.0  # until then, read often so the state is known soon after the start
RETRY_S = 2.0  # after a reading skipped because the wheels were turning
MAX_SKIPS = 30  # about a minute of driving, then a loaded reading counts
POLL_S = 1.0
FULL_V = 8.4  # two 18650 cells, fully charged
# Readings outside this are not a battery: a pack that is not connected reads about 0 V.
MIN_VALID_V = 3.0
MAX_VALID_V = 12.0
JOIN_TIMEOUT_S = 2.0
# After this many failed reads in a row the state becomes UNKNOWN.
FAILED_READS_UNKNOWN = 3


class BatteryState(str, Enum):
    # Not known yet, or the voltage can't be read: treated like LOW for driving on her own, so a
    # broken reading can never let her wander until she is stranded. She still talks and looks.
    UNKNOWN = "unknown"
    OK = "ok"
    LOW = "low"
    CRITICAL = "critical"


_EVENTS = {
    BatteryState.OK: EventType.BATTERY_OK,
    BatteryState.LOW: EventType.BATTERY_LOW,
    BatteryState.CRITICAL: EventType.BATTERY_CRITICAL,
}


class BatteryMonitor:
    def __init__(
        self,
        robot: Robot,
        events: EventBus | None,
        config: BatteryConfig,
        clock: Callable[[], float] = time.monotonic,
        is_driving: Callable[[], bool] | None = None,
    ) -> None:
        self.robot = robot
        self.events = events
        self.config = config
        self._clock = clock
        self._is_driving = is_driving or self._robot_driving
        self._lock = threading.Lock()
        self._window: deque[float] = deque(maxlen=SMOOTHING)
        self._state = BatteryState.UNKNOWN
        self._failed_reads = 0
        self._lowest: float | None = None
        self._skips = 0
        self._next = clock()
        self._read_failed = False
        self._listeners: list[Callable[[BatteryState, BatteryState], None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- what the rest of Laika reads ---------------------------------------------

    @property
    def state(self) -> BatteryState:
        return self._state

    @property
    def voltage(self) -> float | None:
        """The smoothed voltage, or None before the first reading."""
        with self._lock:
            return self._median()

    @property
    def lowest_v(self) -> float | None:
        """The lowest smoothed voltage seen since the start."""
        return self._lowest

    @property
    def percent(self) -> int | None:
        """A ROUGH estimate: a straight line from critical_v (0) to a full pack (8.4 V, 100).

        The real curve of a lithium cell is not a line and depends on the load; use it to talk
        about the battery, never to decide anything.
        """
        voltage = self.voltage
        if voltage is None:
            return None
        span = FULL_V - self.config.critical_v
        return int(round(100 * min(1.0, max(0.0, (voltage - self.config.critical_v) / span))))

    @property
    def low(self) -> bool:
        """LOW or CRITICAL: she needs charging. UNKNOWN is not low: nothing to tell anyone."""
        return self._state in (BatteryState.LOW, BatteryState.CRITICAL)

    def snapshot(self) -> dict[str, Any]:
        """For tool results: the voltage, the state and the rough percentage."""
        voltage = self.voltage
        return {
            "voltage": None if voltage is None else round(voltage, 2),
            "state": self._state.value,
            "percent": self.percent,
        }

    def add_listener(self, listener: Callable[[BatteryState, BatteryState], None]) -> None:
        """Call `listener(new, previous)` on every state change. It must not block."""
        self._listeners = [*self._listeners, listener]

    # --- sampling ----------------------------------------------------------------------

    def poll(self) -> bool:
        """Sample when it is time (the thread calls this every second; tests call it with a fake clock)."""
        if self._clock() < self._next:
            return False
        self.sample()
        return True

    def sample(self) -> float | None:
        """Read once. Returns the reading counted (with the offset), or None when nothing was counted."""
        now = self._clock()
        volts = self._read()
        if volts is None:
            self._failed_reads += 1
            self._next = now + self.config.sample_s
            if self._failed_reads >= FAILED_READS_UNKNOWN and self._state is not BatteryState.UNKNOWN:
                with self._lock:
                    previous, self._state = self._state, BatteryState.UNKNOWN
                    self._window.clear()
                self._announce(BatteryState.UNKNOWN, previous)
            return None
        self._failed_reads = 0
        if self._is_driving_now() and self._skips < MAX_SKIPS:
            self._skips += 1
            self._next = now + RETRY_S
            return None
        self._skips = 0
        volts += self.config.voltage_offset
        change = None
        with self._lock:
            self._window.append(volts)
            smooth = self._median()
            if smooth is not None and (self._lowest is None or smooth < self._lowest):
                self._lowest = smooth
            if len(self._window) >= MIN_READINGS and smooth is not None:
                new = self._next_state(smooth)
                if new is not self._state:
                    change, self._state = (new, self._state), new
            warming = len(self._window) < MIN_READINGS
        self._next = now + (WARMUP_S if warming else self.config.sample_s)
        if change is not None:
            self._announce(*change)
        return volts

    def _read(self) -> float | None:
        try:
            volts = self.robot.battery_voltage()
        except Exception as error:  # noqa: BLE001 - a failed read must never crash the timer
            volts = None
            reason = str(error)
        else:
            reason = "no reading"
            if volts is not None and not MIN_VALID_V <= volts <= MAX_VALID_V:
                reason = f"{volts:.2f} V is not a battery pack voltage"
                volts = None
        if volts is None:
            if not self._read_failed:
                log.warning("battery voltage not available: %s", reason)
            self._read_failed = True
        else:
            self._read_failed = False
        return volts

    def _is_driving_now(self) -> bool:
        try:
            return bool(self._is_driving())
        except Exception:  # noqa: BLE001 - unknown counts as not driving: the reading is used
            return False

    def _robot_driving(self) -> bool:
        return self.robot.get_state().speed != 0

    def _median(self) -> float | None:
        return statistics.median(self._window) if self._window else None

    def _next_state(self, volts: float) -> BatteryState:
        config, state = self.config, self._state
        if volts < config.critical_v:
            return BatteryState.CRITICAL
        if state is BatteryState.CRITICAL:
            if volts > config.ok_v:
                return BatteryState.OK
            return BatteryState.LOW if volts >= config.low_v else BatteryState.CRITICAL
        if state is BatteryState.LOW:
            return BatteryState.OK if volts > config.ok_v else BatteryState.LOW
        return BatteryState.LOW if volts < config.low_v else BatteryState.OK

    def _announce(self, new: BatteryState, previous: BatteryState) -> None:
        voltage, lowest = self.voltage, self._lowest
        log.info("battery %s -> %s at %s V", previous.value, new.value, voltage)
        routine = previous is BatteryState.UNKNOWN and new is BatteryState.OK
        if routine:
            pass  # the first good reading at startup (or after a glitch): nothing to tell
        elif voltage is None:
            print(f"[laika] battery {new.value}: the voltage can't be read, no driving on her own")
        else:
            print(f"[laika] battery {new.value} ({voltage:.2f} V, about {self.percent} %)")
        if new in _EVENTS and not routine:
            emit(
                self.events,
                _EVENTS[new],
                voltage=None if voltage is None else round(voltage, 2),
                percent=self.percent,
                previous=previous.value,
                lowestV=None if lowest is None else round(lowest, 2),
            )
        for listener in self._listeners:
            try:
                listener(new, previous)
            except Exception:  # noqa: BLE001 - a failing listener must not stop the monitor
                log.exception("battery listener failed")

    # --- lifecycle -----------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self.prime()
        self._thread = threading.Thread(target=self._run, name="laika-battery", daemon=True)
        self._thread.start()

    def prime(self) -> BatteryState:
        """Take the first readings straight away, so the state is known (or UNKNOWN) at startup."""
        for _ in range(max(MIN_READINGS, FAILED_READS_UNKNOWN)):
            self.sample()
            if self._state is not BatteryState.UNKNOWN:
                break
        return self._state

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(JOIN_TIMEOUT_S)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll()
            except Exception:  # noqa: BLE001 - the timer must survive anything
                log.exception("battery sample failed")
            self._stop.wait(POLL_S)
