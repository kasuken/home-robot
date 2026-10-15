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
