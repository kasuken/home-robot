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
| Laika application | V0.1, V0.5 and V0.6 complete; V0.2 code, V0.3 vision and V0.4 exploration in `src/laika/`: Italian voice conversation, tool-calling agent, behaviour priorities, safety controller and watchdog, house simulator, camera and vision, exploration, curiosity and investigation, searching for objects and people (find_object, find_person, approach), persistent memory, person detection, the interaction policy, proactive questions and a configurable personality; not yet run on the robot, and autonomous driving is off on it by default | 1082 unit tests and simulator runs on a development computer, CI on every push |

The hardware verification is owner-reported. No connection to the robot or hardware test was performed during this documentation sync. The owner did not supply individual test logs or example filenames.

## Integration details to record when needed

These details were not supplied and remain TBC; they do not change the confirmed working baseline.

- Robot hostname/address and the chosen access method.
- Installed OS, Python, and SunFounder library versions.
- Exact Robot HAT and camera revisions; microphone configuration for spoken input.
- External AI endpoint/model and authentication configuration.
- Verification of the systemd service on the robot (issue #18; the unit and scripts are in [service](service.md)). Data paths and retention limits are defined in [storage and retention](storage-and-retention.md).
- Laika-specific operating limits, timeouts, and calibration values to reuse from the working setup. The settings, ranges and calibration procedure are in [safety calibration](safety-calibration.md); the values measured on the robot are still TBC there.

Store credentials outside tracked documentation and source files.

## Application architecture

On 15 October 2026, the owner selected Python to simplify implementation using the working official examples, then confirmed that all Laika application components will run on the Raspberry Pi with no separate backend. The owner also confirmed that model inference will use an external AI endpoint, matching the original model plan.

The agent, tools, memory, observations, world model, curiosity engine, interaction policy, hardware adapter, and safety controller will run locally. The simulator is a development option using the same agent interface. The [current PRD](prd.md) reflects these decisions; its package layout and interfaces remain proposals. V0.1 follows that layout: synchronous Python with threads ([ADR 0001](decisions/0001-threads-not-asyncio.md)), the vendor's Vosk speech recognition, the vendor's online Edge voice with its offline Piper voice as fallback, and Laika's own provider-independent AI client with tool calling in place of `picarx.llm`. The model acts only through validated tools; wheel movement goes through the safety controller. SunFounder libraries are imported only by the hardware and audio adapters. Vision sends selected camera frames, one at a time, to the configured vision model and validates the reply as an observation; in the simulator, observations come from the house's ground truth. A behaviour arbiter ([behaviours](behaviours.md)) enforces the priority hierarchy: the conversation holds a lease, higher priorities pre-empt lower ones, and safety outcomes revoke autonomous leases. Basic exploration (#27) runs as the lowest-priority behaviour in the simulator and stops on every safety condition; autonomous driving is gated by `[robot] allow_autonomous_driving`, which stays false on the robot until V0.2 is verified (exploration is then head-only). Facts and episodes (#34) are remembered across restarts and recalled through the `remember`, `recall` and `forget` tools; Interestingness is scored and announced as `InterestingChangeDetected` ([curiosity](curiosity.md)), and the investigator (#29) looks into interesting things (another picture, maybe a short move closer, then a queued question or a remembered sighting) without ever speaking on its own; whether she may interrupt is decided by the deterministic interaction policy (#39, [behaviours](behaviours.md#interaction-policy)); the proactive questions (#40, [behaviours](behaviours.md#proactive-questions)) call it before she says anything: she turns to a person, asks the pending question and keeps the answer. The local watchdog and AI-outage handling are implemented and tested in simulation; the strategy is in [ADR 0002](decisions/0002-watchdog-strategy.md). The headless service ([service](service.md)) stops the motors on every exit and restarts Laika after a crash; this is implemented and tested on a development computer, pending verification on the robot (issues #18 and #15), and unattended movement stays disabled until then.

## Navigation

Laika has no map and no position. Rooms come from what the vision model reports: every frame carries a room guess and the `RoomTracker` votes over recent frames to decide which room she is in (#53, [room recognition](behaviours.md#room-recognition)); its accuracy on the robot is still to be measured in the field trial. The evaluation of navigation options (AprilTags, SLAM, LiDAR, ROS 2) and the recommendation are in [navigation](navigation.md); only step 1 (room recognition) is built.

## Field trial of the core loop

Issue #44 is the owner's multi-day trial of the core loop (sees, understands, notices, acts, asks, a human answers, remembers) in the real house. The kit to run and judge it is in place: `python3 run-laika.py --report` summarises the event log and the store read-only (loops completed and open, safety stops, restarts, AI outages, conversations, per day), and [field trial](field-trial.md) has the setup, the daily checklist, a suggested bar for "completes repeatedly without supervision issues" and a template for posting findings. The trial itself has not been run: no result is recorded here.

## Scope of the next verification

Vendor examples establish the hardware baseline. Future integration work must verify Laika's adapter, local command handling, and safety behavior, including command expiry, local watchdog failures, AI endpoint outages, invalid commands, sensor faults, and emergency stopping.

The current PRD places safety and watchdog verification in simulation before Laika-controlled physical movement. These are requirements to implement and validate, not existing Laika features.

## Provenance

At initial sync, `docs/prd.md` matched the owner's pasted project brief byte for byte. That version is preserved unchanged in [the initial archive](archive/prd-initial.md). The [intermediate Python revision](archive/prd-python-distributed.md) preserves the requirements before removing the separate backend. The active `docs/prd.md` tracks the current local application architecture. Illustrative examples and proposed interfaces describe intended behavior rather than implemented functionality or observations from the house.
