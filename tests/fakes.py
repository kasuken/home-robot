from laika.simulator import SimulatedRobot


class RecordingRobot(SimulatedRobot):
    def __init__(self, **kwargs):
        super().__init__(verbose=False, **kwargs)
        self.speeds = []

    def drive(self, speed):
        super().drive(speed)
        self.speeds.append(self.speed)


class LookRecordingRobot(RecordingRobot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.poses = []

    def look(self, pan, tilt):
        super().look(pan, tilt)
        self.poses.append(self.head_pose())
