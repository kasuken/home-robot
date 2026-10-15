# Current project status

Recorded: 15 October 2026 (Europe/Zurich).

## Confirmed baseline

The owner supplied the [initial project brief](archive/prd-initial.md), identified [sunfounder/picar-x](https://github.com/sunfounder/picar-x) as the hardware reference, and confirmed in this project's setup conversation:

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
- External AI endpoint/model and authentication configuration.
- Local service configuration, data paths, and retention limits.
- Laika-specific operating limits, timeouts, and calibration values to reuse from the working setup.

Store credentials outside tracked documentation and source files.

## Application architecture

On 15 October 2026, the owner selected Python to simplify implementation using the working official examples, then confirmed that all Laika application components will run on the Raspberry Pi with no separate backend. The owner also confirmed that model inference will use an external AI endpoint, matching the original model plan.

The agent, tools, memory, observations, world model, curiosity engine, interaction policy, hardware adapter, and safety controller will run locally. The simulator is a development option using the same agent interface. The [current PRD](prd.md) reflects these decisions; its package layout and interfaces remain proposals, and application code has not been built yet.

## Scope of the next verification

Vendor examples establish the hardware baseline. Future integration work must verify Laika's adapter, local command handling, and safety behavior, including command expiry, local watchdog failures, AI endpoint outages, invalid commands, sensor faults, and emergency stopping.

The current PRD places safety and watchdog verification in simulation before Laika-controlled physical movement. These are requirements to implement and validate, not existing Laika features.

## Provenance

At initial sync, `docs/prd.md` matched the owner's pasted project brief byte for byte. That version is preserved unchanged in [the initial archive](archive/prd-initial.md). The [intermediate Python revision](archive/prd-python-distributed.md) preserves the requirements before removing the separate backend. The active `docs/prd.md` tracks the current local application architecture. Illustrative examples and proposed interfaces describe intended behavior rather than implemented functionality or observations from the house.
