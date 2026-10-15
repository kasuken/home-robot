# Laika Home Robot

Laika is a home robot intended to observe its surroundings, learn about the house, remember useful information, and interact when it has a reason to. Its personality is concise, curious, and direct, with dry humour inspired by TARS from *Interstellar*.

The repository is named `home-robot`; the robot is named **Laika**.

## Current status

The owner confirmed on 15 October 2026 that the robot's software is installed and the SunFounder examples have been verified and work. The platform is a SunFounder PiCar-X with a Raspberry Pi 4 and 1 GB RAM, as recorded in the project brief.

This repository currently contains project documentation and a PiCar-X reference snapshot. Laika's local application, simulator, and hardware integration are planned work.

## Documentation

- [Current PRD and roadmap](docs/prd.md): requirements for the Python application running on the Raspberry Pi.
- [Initial project brief](docs/archive/prd-initial.md): the original supplied brief, preserved unchanged for historical reference.
- [Current status and confirmed baseline](docs/current-status.md): installation and example verification, evidence boundaries, and outstanding integration details.
- [PiCar-X hardware reference](docs/hardware-reference.md): official sources and the Python APIs relevant to Laika.
- [Local SunFounder reference](references/sunfounder/README.md): a pinned copy of the upstream library and samples, with source and licence information.

## Intended architecture

Laika runs as one Python application on the Raspberry Pi. The agent, tools, observations, memory, world model, curiosity engine, and interaction policy all run locally, alongside hardware, sensors, camera, audio, and safety.

The Pi calls an external OpenAI-compatible endpoint directly for language and vision inference. That endpoint may run in the cloud or on another machine on the home network. There is no separate Laika backend, robot WebSocket connection, or remote application deployment.

**The local agent proposes actions; the safety controller decides how and whether to execute them.** Only the hardware adapter calls the SunFounder APIs. AI must never control GPIO or motors directly, or override local stopping rules.

The same agent and robot abstraction should support both a development simulator and the physical robot. Model requests must not block safety monitoring; network failures and stalled local control must stop autonomous movement.

## Development direction

The first roadmap milestone is V0.1: local application and simulator, including the robot abstraction, local command contract, events, safety behavior, basic agent tools, and a basic world model. Physical integration follows in V0.2 using the already working hardware stack.

The intended product loop is:

```text
observe -> understand -> notice -> act -> ask -> learn -> remember
```

See the [current PRD](docs/prd.md#development-roadmap) for V0.1 through V1 and the features intentionally deferred.
