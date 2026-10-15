import time

from laika.simulator import SimulatedRobot


class RecordingRobot(SimulatedRobot):
    def __init__(self, **kwargs):
        super().__init__(verbose=False, **kwargs)
        self.speeds = []

    def drive(self, speed):
        super().drive(speed)
        self.speeds.append(self.speed)


class FakeClock:
    """A clock that only moves when the test says so."""

    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class LookRecordingRobot(RecordingRobot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.poses = []

    def look(self, pan, tilt):
        super().look(pan, tilt)
        self.poses.append(self.head_pose())


def settled_events(bus, seen, terminal_types, timeout=2.0):
    """Wait until a move's final event was delivered, then return what was seen.

    The safety monitor wakes the caller of execute() before it publishes the
    move's final event, so a test must wait for that event, not just drain.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        bus.drain(timeout)
        if any(e.type in terminal_types for e in seen):
            break
        time.sleep(0.01)
    bus.drain(timeout)
    return list(seen)
