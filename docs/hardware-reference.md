# PiCar-X hardware reference

Laika will use the existing SunFounder Python stack on the robot. The owner confirmed that it is installed and works with the vendor examples; see [current status](current-status.md).

## Official sources

- [SunFounder PiCar-X repository](https://github.com/sunfounder/picar-x).
- [PiCar-X documentation](https://docs.sunfounder.com/projects/picar-x-v20/en/latest/).
- [Reviewed PiCar-X source snapshot](https://github.com/sunfounder/picar-x/blob/897698daf2719d830a5fb0b83951704bfee07d0c/picarx/picarx.py).

The source was reviewed on 15 October 2026 against branch `2.1.x`, whose head at sync was `897698daf2719d830a5fb0b83951704bfee07d0c`. This is a reference snapshot, not a claim about the version installed on the robot.

## Adapter surface

The reviewed `Picarx` class exposes:

| Laika capability | Vendor API |
| --- | --- |
| Drive and stop | `forward(speed)`, `backward(speed)`, `stop()` |
| Steering | `set_dir_servo_angle(value)` |
| Camera orientation | `set_cam_pan_angle(value)`, `set_cam_tilt_angle(value)` |
| Distance | `get_distance()` |
| Ground sensing | `get_grayscale_data()`, `get_cliff_status(gm_val_list)` |

Camera capture and audio need companion libraries rather than this class alone. The [upstream setup notes](https://github.com/sunfounder/picar-x#installation) identify `robot_hat`, `vilib`, and other dependencies.

## Consequences for Laika

Motor speed is a control value, not measured travel speed. Movement duration must not imply a known distance. Cliff detection compares grayscale readings with configurable thresholds.

Wrap these APIs behind Laika's Python adapter. Keep protocol commands independent of vendor method names and require local safety checks before execution. Reuse the working installation; confirm its API version and calibration when integrating.
