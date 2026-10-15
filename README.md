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

The AI endpoint is set in the `[ai]` section: `provider = "openai"` (default), `"azure"` (Azure OpenAI, with `azure_deployment`) or `"openai-compatible"` for LM Studio, Ollama or vLLM on your network (no key needed; for example `base_url = "http://<lan-ip>:1234/v1"`). Cloud endpoints must use HTTPS; plain HTTP is accepted only for local and private addresses. See `config/laika.example.toml`.

Laika speaks with Microsoft's free online Edge voice (`it-IT-IsabellaNeural`). Install what it needs once (skip any already installed):

```bash
sudo pip3 install edge-tts --break-system-packages
sudo apt install sox libsox-fmt-mp3
```

If the Edge voice fails, for example without internet, Laika says that sentence with the offline Piper voice (`it_IT-paola-medium`). Set `tts_engine = "piper"` in `config/laika.toml` to use only the offline voice.

Then start Laika, the same way you run the SunFounder examples:

```bash
sudo python3 run-laika.py
```

The first start downloads the Italian speech model (Vosk, about 50 MB), so it needs internet and takes a few minutes. The Piper voice is downloaded the first time it's needed. Press Ctrl+C to stop; Laika stops the motors and centres her head.

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

### Check the hardware

A guided check walks through every robot capability (steering, head pan and tilt, ultrasonic, grayscale and cliff, driving, obstacle refusal, emergency stop, camera, speaker, microphone). It uses Laika's own interfaces, including the safety controller, and asks you to confirm what you saw with y, n or s (skip):

```bash
sudo python3 run-laika.py --check-hardware
```

Prepare about 1 m of free floor, a box (to put about 20 cm in front of the robot), and be ready to lift the robot in your hands. Use `--no-mic` to skip the microphone step and `--no-driving` to skip the steps where the wheels spin. `python3 run-laika.py --sim --check-hardware --yes` is a dry run on a laptop. At the end Laika prints a summary and writes `hardware-check-<timestamp>.json` and `.md` to the `logs/` folder of the data directory; the exit code is 0 only if nothing failed. Attach the report to GitHub issue #15.

### Prototype layout

By default the model acts through tools (`[ai] use_tools = true`): it calls `move`, `look`, `capture_image`, `speak`, `remember`, `recall` and so on, a local validator checks every call, and wheel movement still goes through the safety controller. Invalid or stale calls are rejected and logged as `[laika] tool rejected: ...`. If your model has no tool calling, set `use_tools = false` and it replies with the JSON `{"say", "actions"}` format instead.

| Module | Role |
| --- | --- |
| `app.py` | Builds everything from the settings and runs until Ctrl+C |
| `conversation.py` | Sleep → wake word → ask → answer and gesture → follow-ups; picks tool mode or the JSON fallback |
| `agent.py` | Tool-calling loop: asks the model, runs its tool calls one by one, bounded rounds and history, cancels stale turns |
| `tools.py` | The tools the AI acts through (move, stop, look, capture_image, speak, remember, recall, gesture, ...) and the validator that checks every call first |
| `wake_word.py` | Fuzzy "Laika" matching on what Vosk heard |
| `audio.py` | Vosk listening; Edge voice with Piper fallback, or Piper only; or the console |
| `ai_client.py` | Provider-independent chat client (OpenAI, Azure OpenAI, LM Studio, Ollama, vLLM) with tool calls, timeouts and bounded retries |
| `prompts.py` | Laika's Italian personality: the tool-mode prompt (answers in plain text) and the older JSON reply prompt |
| `actions.py` | Gestures the AI may choose, played on one worker thread |
| `commands.py`, `events.py`, `world_model.py` | Typed local commands and telemetry, the event bus and AI endpoint monitor, the in-memory world model |
| `safety.py` | Safety controller: a 20 Hz monitor thread stops every move on obstacle, cliff, timeout, hardware error or emergency stop; all wheel movement goes through it |
| `camera.py` | Camera service: the one place frames are captured on request (minimum interval and per-minute cap), saved with bounded retention and announced as `ImageCaptured` |
| `storage.py` | Data directory, image store, rotating event log and optional transcripts, all with size or age limits |
| `hardware_check.py` | The guided `--check-hardware` script and its JSON and Markdown report |
| `robot.py`, `hardware.py`, `simulator.py`, `house.py`, `limits.py` | Robot interface and state models, PiCar-X adapter, simulated robot and its 2D house, shared limits |

## Documentation

- [Current PRD and roadmap](docs/prd.md): requirements for the Python application running on the Raspberry Pi.
- [Initial project brief](docs/archive/prd-initial.md): the original supplied brief, preserved unchanged for historical reference.
- [Current status and confirmed baseline](docs/current-status.md): installation and example verification, evidence boundaries, and outstanding integration details.
- [Storage and retention](docs/storage-and-retention.md): data directory, permissions, and every disk and memory limit.
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
