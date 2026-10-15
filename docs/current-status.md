# Current project status

Recorded: 15 October 2026 (Europe/Zurich).

## Confirmed baseline

The owner supplied the [project brief](prd.md), identified [sunfounder/picar-x](https://github.com/sunfounder/picar-x) as the hardware reference, and confirmed in this project's setup conversation:

> everything is installed on the robot

> everything is verified with the example and it works

Treat the robot as provisioned and working with the SunFounder examples. Reinstalling the vendor stack or repeating initial hardware bring-up is not a prerequisite for Laika development.

| Item | Recorded status | Evidence |
| --- | --- | --- |
| Robot | SunFounder PiCar-X | Owner's hardware reference |
| Board | Raspberry Pi 4, 1 GB RAM | Original project brief |
| Vendor software | Installed | Owner confirmation |
| Vendor examples | Verified and working | Owner confirmation |
| Laika application | Planned; this repository currently contains documentation | Repository contents at initial sync |

The hardware verification is owner-reported. No connection to the robot or hardware test was performed during this documentation sync. The owner did not supply individual test logs or example filenames.

## Integration details to record when needed

These details were not supplied and remain TBC; they do not change the confirmed working baseline.

- Robot hostname/address and the chosen access method.
- Installed OS, Python, and SunFounder library versions.
- Exact Robot HAT and camera revisions; microphone configuration for spoken input.
- Backend location, AI endpoint/model, and authentication configuration.
- Laika-specific operating limits, timeouts, and calibration values to reuse from the working setup.

Store credentials outside tracked documentation and source files.

## Scope of the next verification

Vendor examples establish the hardware baseline. Future integration work must verify Laika's adapter, protocol handling, and local safety behavior, including command expiry, lost backend heartbeats, invalid commands, sensor faults, and emergency stopping.

The original brief places the safety controller after the physical adapter in its approximate implementation list. A planning recommendation from the setup discussion is to implement and exercise those stopping rules in simulation before enabling Laika-controlled physical movement. This recommendation supplements the original brief; it is not an implemented feature.

## Provenance

The existing `docs/prd.md` matched the owner's pasted project brief byte for byte at initial sync. It is preserved unchanged, including illustrative examples and proposed interfaces. Those examples describe intended behavior rather than implemented functionality or observations from the house.
