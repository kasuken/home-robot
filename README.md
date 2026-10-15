# Laika Home Robot

![SunFounder PiCar-X](https://docs.sunfounder.com/projects/picar-x-v20/en/latest/_images/picar-x_v2.png)

Laika is a home robot intended to observe its surroundings, learn about the house, remember useful information, and interact when it has a reason to. Its personality is concise, curious, and direct, with dry humour inspired by TARS from *Interstellar*.

The repository is named `home-robot`; the robot is named **Laika**.

## Current status

The owner confirmed on 15 October 2026 that the robot's software is installed and the SunFounder examples have been verified and work. The platform is a SunFounder PiCar-X with a Raspberry Pi 4 and 1 GB RAM, as recorded in the project brief.

The first prototype is in [`src/laika/`](src/laika/): say "ehi Laika", ask anything in Italian, and Laika answers through an OpenAI-compatible model while moving her head and wheels. Vision and interestingness scoring are in, and she asks about things she does not know when someone is near ([proactive questions](docs/behaviours.md#proactive-questions)); the rest of the PRD is still planned work.

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

The first start downloads the Italian speech model (Vosk, about 50 MB), so it needs internet and takes a few minutes. The Piper voice is downloaded the first time it's needed. Press Ctrl+C to stop; Laika stops the motors and centres her head. To start her on boot and restart her after a crash, see [running as a service](docs/service.md).

| Say | What happens |
| --- | --- |
| "Ehi Laika" | She looks up and says "Dimmi." |
| "Ehi Laika, quanti anni hai?" | Wakes and answers in one go |
| Any follow-up within 8 seconds | Keeps the conversation going |
| "Basta" or silence | Goes back to sleep |

Options:

- `--keyboard`: type questions instead of speaking (useful to check the AI connection).
- `--no-driving`: head and steering gestures only, the wheels never spin.
- `--sim`: no robot at all, runs on a laptop with typed input and printed replies. It also allows autonomous driving, for the simulator only.

If the wake word isn't recognised, watch the `[laika] heard:` lines and add what Vosk writes to `wake_names` in `config/laika.toml`.

Tests run without hardware: `python3 -m unittest discover -s tests -t .`

### Exploration

"Ehi Laika, esplora la casa" (or `[explore] idle_after_s`) makes Laika move slowly, stop, look around and repeat, for up to 3 minutes, stopping at once for any safety event (see [behaviours](docs/behaviours.md#exploration)). **On the robot she does not drive on her own until you set `[robot] allow_autonomous_driving = true`, and you should only do that after the V0.2 hardware check (#15, #17, #18) has passed.** Until then exploration only looks around with the head. Wake her up ("Ehi Laika") to interrupt it; it resumes when the conversation ends.

### Vision

When Laika takes a photo, the `[vision]` section decides how it is described: the frame is sent to a vision-capable model (`gpt-4o-mini` by default, or any OpenAI-compatible server with a vision model) with a strict-JSON prompt, and the reply is validated locally before it becomes an observation. Privacy: an image leaves the robot only to the endpoint you configured in `[ai]`, one frame at a time and only when a photo is taken, never as video; the model is told to count people and never identify them. Each photo is one billed request, so `max_per_minute` (default 10) skips extra frames and `detail = "low"` keeps them cheap. The simulator uses its own ground truth and sends nothing.

To look around ("guardati intorno"), the model calls `scan_environment`: Laika turns only her head to the left, centre and right (restoring it afterwards), takes one photo at each position, and merges the observations into one description (objects with where they were seen, the most people seen in any one frame, the room). The wheels never move. A position that could not be photographed or analysed is reported, not fatal. The `look_around` gesture is just an expressive head movement and sees nothing.

### Check the hardware

A guided check walks through every robot capability (steering, head pan and tilt, ultrasonic, grayscale and cliff, driving, obstacle refusal, emergency stop, camera, speaker, microphone). It uses Laika's own interfaces, including the safety controller, and asks you to confirm what you saw with y, n or s (skip):

```bash
sudo python3 run-laika.py --check-hardware
```

Before you change a safety value, read [safety calibration](docs/safety-calibration.md): what each `[robot]` setting does, its range, and how to calibrate it with this check.

Prepare about 1 m of free floor, a box (to put about 20 cm in front of the robot), and be ready to lift the robot in your hands. Use `--no-mic` to skip the microphone step and `--no-driving` to skip the steps where the wheels spin. `python3 run-laika.py --sim --check-hardware --yes` is a dry run on a laptop. At the end Laika prints a summary and writes `hardware-check-<timestamp>.json` and `.md` to the `logs/` folder of the data directory; the exit code is 0 only if nothing failed. Attach the report to GitHub issue #15.

### Prototype layout

By default the model acts through tools (`[ai] use_tools = true`): it calls `move`, `look`, `capture_image`, `speak`, `remember`, `recall` and so on, a local validator checks every call, and wheel movement still goes through the safety controller. Invalid or stale calls are rejected and logged as `[laika] tool rejected: ...`. If your model has no tool calling, set `use_tools = false` and it replies with the JSON `{"say", "actions"}` format instead.

| Module | Role |
| --- | --- |
| `app.py` | Builds everything from the settings and runs until Ctrl+C |
| `conversation.py` | Sleep → wake word → ask → answer and gesture → follow-ups; picks tool mode or the JSON fallback |
| `agent.py` | Tool-calling loop: asks the model, runs its tool calls one by one, bounded rounds and history, cancels stale turns |
| `tools.py` | The tools the AI acts through (move, stop, look, capture_image, scan_environment, speak, remember, recall, what_changed, forget, gesture, ...) and the validator that checks every call first |
| `wake_word.py` | Fuzzy "Laika" matching on what Vosk heard |
| `audio.py` | Vosk listening; Edge voice with Piper fallback, or Piper only; or the console |
| `ai_client.py` | Provider-independent chat client (OpenAI, Azure OpenAI, LM Studio, Ollama, vLLM) with tool calls, timeouts and bounded retries |
| `prompts.py` | Laika's Italian personality: the tool-mode prompt (answers in plain text) and the older JSON reply prompt |
| `actions.py` | Gestures the AI may choose, played on one worker thread |
| `commands.py`, `events.py`, `world_model.py` | Typed local commands and telemetry, the event bus and AI endpoint monitor, the world model (in memory, persisted to the store by `WorldModelPersister`) |
| `behaviour.py` | Behaviour priority: the `Arbiter` that owns the body, leases, pre-emption and resume, a heartbeat for autonomous behaviours, and safety overrides ([docs](docs/behaviours.md)) |
| `safety.py` | Safety controller: a 20 Hz monitor thread stops every move on obstacle, cliff, timeout, hardware error or emergency stop; all wheel movement goes through it |
| `observations.py` | Structured observations: the schema, local validation of model output (malformed output is rejected and logged), a bounded observation log, and the `Observer` that turns captured frames into `ObservationCreated` events off the caller's thread |
| `explore.py` | Basic exploration (V0.4): a low-priority behaviour that moves a little, stops, looks around and repeats, every move gated by the safety controller; `Explorer` and the idle auto-start ([docs](docs/behaviours.md#exploration)) |
| `scan.py` | `Scanner`: the `scan_environment` tool's head-only scan (left, centre, right), one fresh frame per position, merged into one `EnvironmentScan` |
| `vision.py` | Vision requests: sends one captured frame (optionally shrunk with Pillow) to the vision model through the AI client and returns a validated observation; size and per-minute limits, outages reported to the AI endpoint monitor |
| `camera.py` | Camera service: the one place frames are captured (minimum interval and per-minute cap), saved with bounded retention and announced as `ImageCaptured` |
| `curiosity.py` | Curiosity (#28): scores each observation for interestingness against the world model as it was before the frame (novelty, moved objects, people, unusual places, and more); announces `InterestingChangeDetected` above a threshold, with a cooldown per reason. Scores only ([docs](docs/curiosity.md)) |
| `changes.py` | Environment changes (#36): `ChangeDetector` compares each observation with the world model as it was before the frame (moved, new, missing, unusual), `ChangeTracker` keeps them and records "moved", "new" and "missing" episodes; the `what_changed` tool answers "what changed since yesterday?" ([docs](docs/curiosity.md#environment-changes)) |
| `memory.py` | Semantic and episodic memory (#34): `Fact`s (what people told her, where things usually stay, who owns them) and `Episode`s (seen, asked, answered, rules), kept in the store, recalled after a restart by the `remember`, `recall` and `forget` tools and shown in the prompt when relevant; `EpisodeRecorder` ([docs](docs/storage-and-retention.md#memory-facts-and-episodes)) |
| `conversation_memory.py` | Conversation memory (#35): which parts of a conversation become long-term memory. The model calls `remember` or `answer_question`; `MemorySelector` is the safety net that applies Italian patterns after each turn (what an object is, ownership, where it stays, house rules, preferences, answers to pending questions), in tool and JSON mode, with `[memory] auto_select` and `min_confidence`. Telling her what an object is makes it a known object ([docs](docs/storage-and-retention.md#what-laika-remembers-from-conversations)) |
| `investigate.py` | Investigation (#29): reacts to `InterestingChangeDetected` and exploration scans under a CURIOSITY lease: points the head, takes another picture, may move closer (autonomous gate and safety apply), then queues an Italian question for later or remembers the sighting; `PendingQuestions`, the `investigate` and `answer_question` tools ([docs](docs/behaviours.md#investigation)) |
| `presence.py` | Knows when a person is near from the observation's head count only (no faces or names): `PersonDetected` / `PersonLost` with confidence, hysteresis and a timer; idle presence photos come from the capture policy (`[presence]`, [docs](docs/behaviours.md#people)) |
| `proactive.py` | Proactive questions (#40): the trigger that asks the policy on `PersonDetected`, interest and a re-check, and hands a pending question to the conversation, which asks it with the head turned to the person ([docs](docs/behaviours.md#proactive-questions), `[interaction] proactive`) |
| `interaction_policy.py` | The deterministic interaction policy (#39): availability states (available, busy, do not disturb, away, sleeping, unknown), quiet hours, cooldown, per-hour limit, and `decide()` for the proactive path; the do-not-disturb and busy tools ([docs](docs/behaviours.md#interaction-policy), `[interaction]`) |
| `capture_policy.py` | When Laika takes photos on her own: after a completed move, when a person or something interesting appears, during investigation and exploration, and idle checks for people; per-trigger switches and cooldowns, one per-minute cap, a worker thread with a one-slot pending queue |
| `storage.py` | Data directory, image store, rotating event log and optional transcripts, all with size or age limits |
| `store.py` | Local persistent store (#32): one SQLite file (`memory/laika.db`) with versioned migrations, a bounded non-blocking write queue on its own thread, retention by count and age, observations restored after a restart ([docs](docs/storage-and-retention.md#the-database)) |
| `systemd.py` | Optional systemd notifications (ready, watchdog ping, stopping), a no-op outside a service |
| `hardware_check.py` | The guided `--check-hardware` script and its JSON and Markdown report |
| `robot.py`, `hardware.py`, `simulator.py`, `house.py`, `limits.py` | Robot interface and state models, PiCar-X adapter, simulated robot and its 2D house, shared limits |

## Documentation

- [Current PRD and roadmap](docs/prd.md): requirements for the Python application running on the Raspberry Pi.
- [Initial project brief](docs/archive/prd-initial.md): the original supplied brief, preserved unchanged for historical reference.
- [Current status and confirmed baseline](docs/current-status.md): installation and example verification, evidence boundaries, and outstanding integration details.
- [Running as a service](docs/service.md): systemd unit, install script, logs, and how the robot is stopped on exit and restart.
- [Safety calibration](docs/safety-calibration.md): every `[robot]` safety setting with its range, and the on-robot calibration procedure for the hardware check.
- [Curiosity](docs/curiosity.md): how an observation's interestingness is scored, the defaults and threshold, and the cooldown.
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
