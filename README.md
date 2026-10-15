# Laika Home Robot

Laika is a home robot intended to observe its surroundings, learn about the house, remember useful information, and interact when it has a reason to. Its personality is concise, curious, and direct, with dry humour inspired by TARS from *Interstellar*.

The repository is named `home-robot`; the robot is named **Laika**.

## Current status

The owner confirmed on 15 October 2026 that the robot's software is installed and the SunFounder examples have been verified and work. The platform is a SunFounder PiCar-X with a Raspberry Pi 4 and 1 GB RAM, as recorded in the project brief.

This repository currently contains project documentation and a PiCar-X reference snapshot. Laika's backend, simulator, protocol, and custom runtime are planned work.

## Documentation

- [Original project brief and roadmap](docs/prd.md): the supplied brief, preserved unchanged.
- [Current status and confirmed baseline](docs/current-status.md): installation and example verification, evidence boundaries, and outstanding integration details.
- [PiCar-X hardware reference](docs/hardware-reference.md): official sources and the Python APIs relevant to Laika.
- [Local SunFounder reference](references/sunfounder/README.md): a pinned copy of the upstream library and samples, with source and licence information.

## Intended architecture

The ASP.NET Core / .NET 10 backend will handle the agent, AI access, vision interpretation, observations, memory, and interaction policy. AI access should support cloud and local providers through an OpenAI-compatible abstraction.

The Python runtime on the Raspberry Pi will handle hardware, sensors, camera, audio, and safety through the existing SunFounder libraries. It will initiate an outbound WebSocket connection to the backend.

**The backend decides what to request; the local runtime decides how and whether to execute it safely.** AI must never control GPIO or motors directly, or override local stopping rules.

The same agent and robot abstraction should support both a simulator and the physical robot. AI models will run outside the Pi.

## Development direction

The first roadmap milestone is V0.1: backend and simulator, including the robot abstraction, command protocol, events, basic agent tools, and a basic world model. Physical integration follows in V0.2 using the already working hardware stack.

The intended product loop is:

```text
observe -> understand -> notice -> act -> ask -> learn -> remember
```

See the [project brief](docs/prd.md#development-roadmap) for V0.1 through V1 and the features intentionally deferred.
