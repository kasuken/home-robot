# Laika Home Robot

Laika is a home robot intended to observe its surroundings, learn about the house, remember useful information, and interact when it has a reason to. Its personality is concise, curious, and direct, with dry humour inspired by TARS from *Interstellar*.

The repository is named `home-robot`; the robot is named **Laika**.

## Current status

The owner confirmed on 15 October 2026 that the robot's software is installed and the SunFounder examples have been verified and work. The platform is a SunFounder PiCar-X with a Raspberry Pi 4 and 1 GB RAM, as recorded in the project brief.

The first prototype is in [`src/laika/`](src/laika/): say "ehi Laika", ask anything in Italian, and Laika answers through an OpenAI-compatible model while moving her head and wheels. The rest of the PRD (vision, memory, curiosity, proactive interaction) is still planned work.

## Run the prototype

On the robot, clone the repository and create the settings file:

```bash
git clone https://github.com/kasuken/home-robot.git
cd home-robot
cp config/laika.example.toml config/laika.toml
nano config/laika.toml   # set [ai] api_key
```

Then start Laika, the same way you run the SunFounder examples:

```bash
sudo python3 run-laika.py
```

The first start downloads the Italian speech model (Vosk, about 50 MB) and the Italian voice (Piper `it_IT-paola-medium`), so it needs internet and takes a few minutes. Press Ctrl+C to stop; Laika stops the motors and centres her head.

| Say | What happens |
| --- | --- |
| "Ehi Laika" | She looks up and says "Dimmi." |
| "Ehi Laika, quanti anni hai?" | Wakes and answers in one go |
| Any follow-up within 8 seconds | Keeps the conversation going |
| "Basta" or silence | Goes back to sleep |

Options:

- `--keyboard`: type questions instead of speaking (useful to check the AI connection).
- `--no-driving`: head and steering gestures only, the wheels never spin.
- `--sim`: no robot at all, runs on a laptop with typed input and printed replies.

If the wake word isn't recognised, watch the `[laika] heard:` lines and add what Vosk writes to `wake_names` in `config/laika.toml`.

Tests run without hardware: `python3 -m unittest discover -s tests -t .`

### Prototype layout

| Module | Role |
| --- | --- |
| `app.py` | Builds everything from the settings and runs until Ctrl+C |
| `conversation.py` | Sleep → wake word → ask → answer and gesture → follow-ups |
| `wake_word.py` | Fuzzy "Laika" matching on what Vosk heard |
| `audio.py` | Vosk listening and Piper speaking, or the console |
| `ai_client.py` | OpenAI-compatible chat client with timeouts and bounded history |
| `prompts.py` | Laika's Italian personality and JSON reply format |
| `actions.py` | Gestures the AI may choose, played on one worker thread |
| `safety.py` | Short, slow drives only; stops for obstacles and cliffs |
| `robot.py`, `hardware.py`, `simulator.py` | Robot interface, PiCar-X adapter, simulated robot |

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
