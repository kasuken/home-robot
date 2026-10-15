"""The PiCar-X adapter: the only module that calls the SunFounder hardware APIs.

Every vendor call is wrapped so any failure surfaces as `HardwareError`, which
the safety layer reports cleanly. `reset` and `close` never raise.
"""

from __future__ import annotations

import io
import logging
import threading
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import Any, Iterator

from laika import __version__
from laika.audio import Mouth
from laika.config import REPO_ROOT
from laika.robot import (
    ALL_CAPABILITIES,
    PAN_MAX,
    PAN_MIN,
    SPEED_MAX,
    STEER_MAX,
    STEER_MIN,
    TILT_MAX,
    TILT_MIN,
    HardwareError,
    ImageObservation,
    RobotIdentity,
    RobotSensorState,
    RobotState,
    clamp,
)

log = logging.getLogger(__name__)

HORN_SOUND = REPO_ROOT / "references" / "sunfounder" / "picar-x" / "sounds" / "car-double-horn.wav"
DEFAULT_IMAGE_SIZE = (640, 480)
DEFAULT_JPEG_QUALITY = 80


@contextmanager
def _hardware(what: str) -> Iterator[None]:
    """Turn any SunFounder failure into a HardwareError."""
    try:
        yield
    except HardwareError:
        raise
    except Exception as error:  # noqa: BLE001 - the vendor libraries raise whatever they like
        raise HardwareError(f"{what}: {error}") from error


def _battery_reader() -> Callable[[], float]:
    """The vendor's battery read. Newer robot_hat has `device.get_battery_voltage`; older only `utils`."""
    try:
        from robot_hat.device import get_battery_voltage
    except ImportError:
        from robot_hat.utils import get_battery_voltage
    return get_battery_voltage


class PiCarXRobot:
    def __init__(
        self,
        mouth: Mouth | None = None,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        cliff_reference: Sequence[float] | None = None,
    ) -> None:
        # Imported here so the rest of Laika runs on machines without the Robot HAT.
        from picarx import Picarx
        from robot_hat.music import Music

        with _hardware("cannot initialise the PiCar-X"):
            self._car = Picarx()
            self._music = Music()
        if cliff_reference is not None:
            # Overrides the value read from /opt/picar-x/picar-x.conf when the vendor library started.
            with _hardware("cannot set the cliff reference"):
                self._car.set_cliff_reference([float(value) for value in cliff_reference])
        self._mouth = mouth
        self._image_size = image_size
        self._jpeg_quality = jpeg_quality
        self._camera: Any = None
        self._pose = (0.0, 0.0)
        self._steering = 0.0
        self._speed = 0
        self._lock = threading.RLock()
        self._battery_failed = False
        self._camera_lock = threading.Lock()
        self.reset()

    def identity(self) -> RobotIdentity:
        return RobotIdentity(
            robot_id="laika-01",
            name="Laika",
            version=__version__,
            capabilities=ALL_CAPABILITIES,
        )

    def get_sensors(self) -> RobotSensorState:
        with self._lock, _hardware("sensor read failed"):
            distance = self._car.get_distance()
            grayscale = self._car.get_grayscale_data()
            cliff = bool(self._car.get_cliff_status(grayscale))
            values = tuple(int(v) for v in grayscale[:3])
        if len(values) != 3:
            raise HardwareError(f"grayscale sensor returned {len(values)} values, expected 3")
        # The sensor returns -1 or -2 on timeouts and other failed reads.
        valid = distance if distance is not None and distance > 0 else None
        return RobotSensorState(distance_cm=valid, cliff_detected=cliff, grayscale=values)

    def get_state(self) -> RobotState:
        with self._lock:
            return RobotState(
                head_pan=self._pose[0],
                head_tilt=self._pose[1],
                steering=self._steering,
                speed=self._speed,
                sensors=self.get_sensors(),
                battery_v=self.battery_voltage(),
            )

    def battery_voltage(self) -> float | None:
        """The pack voltage from the Robot HAT's ADC (channel A4, through a divider of 3).

        None when the read fails: the first failure is logged, later ones stay quiet until a read
        works again, so a broken ADC neither crashes anything nor floods the log.
        """
        with self._lock:
            try:
                volts = float(_battery_reader()())
            except Exception as error:  # noqa: BLE001 - the vendor libraries raise whatever they like
                if not self._battery_failed:
                    log.error("cannot read the battery voltage: %s", error)
                self._battery_failed = True
                return None
        self._battery_failed = False
        return volts

    def look(self, pan: float, tilt: float) -> None:
        pose = (clamp(pan, PAN_MIN, PAN_MAX), clamp(tilt, TILT_MIN, TILT_MAX))
        with self._lock, _hardware("head servos failed"):
            self._car.set_cam_pan_angle(pose[0])
            self._car.set_cam_tilt_angle(pose[1])
            self._pose = pose

    def capture_image(self) -> ImageObservation:
        # The camera has its own lock: starting it and capturing take long enough
        # that holding the hardware lock would stall the safety checks.
        with self._camera_lock:
            try:
                if self._camera is None:
                    self._camera = self._start_camera()
                pan, tilt = self.head_pose()
                buffer = io.BytesIO()
                self._camera.capture_file(buffer, format="jpeg")
            except Exception as error:  # noqa: BLE001
                # Drop the camera so the next request starts a fresh one.
                self._release_camera()
                raise HardwareError(f"camera failed: {error}") from error
        return ImageObservation(
            jpeg=buffer.getvalue(),
            width=self._image_size[0],
            height=self._image_size[1],
            camera_pan=pan,
            camera_tilt=tilt,
        )

    def _start_camera(self) -> Any:
        from picamera2 import Picamera2

        camera = Picamera2()
        try:
            camera.configure(camera.create_still_configuration(main={"size": self._image_size}))
            camera.start()
        except Exception:
            camera.close()
            raise
        # picamera2 reads the JPEG quality from this dict at capture time.
        options = getattr(camera, "options", None)
        if isinstance(options, dict):
            options["quality"] = self._jpeg_quality
        return camera

    def _release_camera(self) -> None:
        """Close the camera (camera lock held). Never raises."""
        camera, self._camera = self._camera, None
        if camera is not None:
            try:
                camera.close()
            except Exception as error:  # noqa: BLE001
                log.error("could not close the camera: %s", error)

    def speak(self, text: str) -> None:
        # Not under the lock: speech takes seconds and must not block the safety loop.
        if self._mouth is not None:
            self._mouth.say(text)
        else:
            print(f"Laika: {text}")

    def head_pose(self) -> tuple[float, float]:
        with self._lock:
            return self._pose

    def steer(self, angle: float) -> None:
        angle = clamp(angle, STEER_MIN, STEER_MAX)
        with self._lock, _hardware("steering servo failed"):
            self._car.set_dir_servo_angle(angle)
            self._steering = angle

    def drive(self, speed: int) -> None:
        speed = int(clamp(speed, -SPEED_MAX, SPEED_MAX))
        with self._lock, _hardware("motors failed"):
            if speed > 0:
                self._car.forward(speed)
            elif speed < 0:
                self._car.backward(-speed)
            else:
                self._car.stop()
            self._speed = speed

    def stop(self) -> None:
        with self._lock, _hardware("could not stop the motors"):
            self._car.stop()
            self._speed = 0

    def distance_cm(self) -> float | None:
        return self.get_sensors().distance_cm

    def cliff_detected(self) -> bool:
        with self._lock, _hardware("sensor read failed"):
            return bool(self._car.get_cliff_status(self._car.get_grayscale_data()))

    def honk(self, volume: int) -> None:
        with self._lock, _hardware("horn failed"):
            self._music.sound_play_threading(str(HORN_SOUND), volume)

    def reset(self) -> None:
        """Stop, centre the wheels and the head. Never raises: each part is tried on its own."""
        with self._lock:
            parts = (
                ("motors", self._car.stop),
                ("steering", lambda: self._car.set_dir_servo_angle(0)),
                ("camera tilt", lambda: self._car.set_cam_tilt_angle(0)),
                ("camera pan", lambda: self._car.set_cam_pan_angle(0)),
            )
            for name, action in parts:
                try:
                    action()
                except Exception as error:  # noqa: BLE001
                    log.error("reset: %s failed: %s", name, error)
            self._pose = (0.0, 0.0)
            self._steering = 0.0
            self._speed = 0

    def close(self) -> None:
        """Leave the hardware stopped and release it. Never raises; safe to call twice."""
        self.reset()
        with self._lock:
            try:
                self._car.close()
            except Exception as error:  # noqa: BLE001
                log.error("close: could not release the PiCar-X: %s", error)
        with self._camera_lock:
            self._release_camera()
