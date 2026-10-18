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
nano config/secrets.toml   # [ai] api_key = "sk-..."  (ignored by git)
chmod 600 config/secrets.toml
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
| Any follow-up within 15 seconds | Keeps the conversation going (once someone starts talking she listens to the end) |
| "Basta" or silence | Goes back to sleep |

Options:

- `--keyboard`: type questions instead of speaking (useful to check the AI connection).
- `--no-driving`: head and steering gestures only, the wheels never spin.
- `--sim`: no robot at all, runs on a laptop with typed input and printed replies. It also allows autonomous driving, for the simulator only.

If the wake word isn't recognised, watch the `[laika] heard:` lines and add what Vosk writes to `wake_names` in `config/laika.toml`.

Speech recognition is offline by default (`[voice] stt_engine = "vosk"`): no audio leaves the robot. Vosk's small Italian model is the only one that fits a 1 GB Raspberry Pi and it often mishears, so the AI is told that the text comes from imperfect recognition, to read it by how it sounds, and to ask again when it makes no sense. What helps most: speak close to the robot, and raise the microphone level (`alsamixer`, press F4 for capture, or `amixer -c 1 sset Mic 90%`). `stt_engine = "cloud"` (opt-in) sends what is said after the wake word to OpenAI's transcription model instead (`stt_model`).

Tests run without hardware: `python3 -m unittest discover -s tests -t .`

### Games

"Ehi Laika, giochiamo?" She proposes two or three games and plays the one chosen: riddles, a quiz, guess what I'm thinking, a word chain, a story together, "Laika dice", a colour hunt with the camera, animal sounds, and three with her body: "un, due, tre... stella!" and statues (she watches with the distance sensor whether you move) and hide and seek (she counts to ten and looks for you). While a game is on she waits 30 seconds for an answer. See [games](docs/behaviours.md#games).

### Songs

"Ehi Laika, cantami una canzone!" She sings traditional children's songs (Fra Martino, Giro giro tondo, La bella lavanderina, Stella stellina, Ninna nanna, Tanti auguri with the birthday child's name) and two of her own, verse by verse, moving her head along. She sings only the songs she knows; asked for another, she offers one of hers or makes up a verse. See [songs](docs/behaviours.md#songs).

### The plan of the house

A private `config/house-plan.toml` (git ignores it; copy it to the robot) tells her the rooms, their names and which doors join them: better room recognition, answers to "come arrivo in cucina?", no-go rooms such as a balcony. See [the plan of the house](docs/floor-plan.md).

### AprilTag position fixes

Printed AprilTags at the doorways, read locally from her camera frames, make the room certain (OpenCV on the robot; tags listed in the house plan). It does not drive to them yet. See [AprilTag position fixes](docs/apriltags.md).

### Learning new skills

Ask for something she cannot do and she offers to try to learn it. On a yes, a GitHub Copilot SDK session writes the skill in her own repository; she checks it (tests, protected safety and hardware files untouched), commits, pushes and releases it, then restarts herself to load it. See [learning new skills](docs/learning.md).

### Bedtime stories

"Ehi Laika, raccontami una storia della buonanotte." She makes up a new story every time, with Leo and Ludo, two bunny siblings, for children from 2 to 7: three to five minutes, told calmly paragraph by paragraph. Every story is saved on the robot. At the end she asks whether you want another one and waits 15 seconds for the answer. See [bedtime stories](docs/behaviours.md#bedtime-stories).

### Exploration

"Ehi Laika, esplora la casa" (or the idle selector, below) makes Laika move slowly, stop, look around and repeat, for up to 10 minutes, stopping at once for any safety event (see [behaviours](docs/behaviours.md#exploration)). **On the robot she does not drive on her own until you set `[robot] allow_autonomous_driving = true`, and you should only do that after the V0.2 hardware check (#15, #17, #18) has passed.** Until then exploration only looks around with the head. Wake her up ("Ehi Laika") to interrupt it; it resumes when the conversation ends. What she learns about each room (how it looks, which rooms it leads to, what was new) is kept and used in later conversations ([docs](docs/behaviours.md#exploration)).

When nothing has happened for `[idle] idle_after_s` (5 minutes) she chooses what to do: explore, look out of a window, follow up on a pending question or something that changed, check on who is home (a head scan that counts people) or rest. A deterministic selector picks one by utility (time of day, what she did lately, pending questions, personality, battery, whether someone is near), never the model; quiet hours, do not disturb, a low battery and the pause always lead to rest or nothing, and the wake word ends any activity at once. On the robot every activity is head-only until you enable autonomous driving after the V0.2 check. Each run is remembered ("cosa hai fatto oggi?") and counted in the trial report ([docs](docs/behaviours.md#when-laika-is-idle)).

### Vision

When Laika takes a photo, the `[vision]` section decides how it is described: the frame is sent to a vision-capable model (`gpt-4o-mini` by default, or any OpenAI-compatible server with a vision model) with a strict-JSON prompt, and the reply is validated locally before it becomes an observation. Privacy: an image leaves the robot only to the endpoint you configured in `[ai]`, one frame at a time and only when a photo is taken, never as video; the model is told to count people and never identify them. Each photo is one billed request, so `max_per_minute` (default 10) skips extra frames and `detail = "low"` keeps them cheap. The simulator uses its own ground truth and sends nothing.

To look around ("guardati intorno"), the model calls `scan_environment`: Laika turns only her head to the left, centre and right (restoring it afterwards), takes one photo at each position, and merges the observations into one description (objects with where they were seen, the most people seen in any one frame, the room). The wheels never move. A position that could not be photographed or analysed is reported, not fatal. The `look_around` gesture is just an expressive head movement and sees nothing.

### Looking things up

Laika can look things up ("chi era Laika, la cagnolina?", "che tempo fa domani?") with two tools. `wikipedia` searches Italian Wikipedia (English as fallback) and needs no key. `web_search` uses the [Brave Search API](https://brave.com/search/api/) with strict safe search for current facts, and is offered only when `LAIKA_BRAVE_API_KEY` or `[web] brave_api_key` is set. She answers in a way suited to a child and says where she found it ("secondo Wikipedia..."). Web text is untrusted: once it enters a turn, moves, exploring and memory writes are refused until the next turn. Queries leave the robot and she is told never to search for family names or details about the home; `[web] enabled = false` turns it all off. See [privacy](docs/storage-and-retention.md#privacy-defaults).

### Check the hardware

A guided check walks through every robot capability (steering, head pan and tilt, ultrasonic, grayscale and cliff, driving, obstacle refusal, emergency stop, camera, speaker, microphone). It uses Laika's own interfaces, including the safety controller, and asks you to confirm what you saw with y, n or s (skip):

```bash
sudo python3 run-laika.py --check-hardware
```

Before you change a safety value, read [safety calibration](docs/safety-calibration.md): what each `[robot]` setting does, its range, and how to calibrate it with this check.

When she drives on her own there are limits that survive a restart ([behaviours](docs/behaviours.md#house-rules-driving-budget-and-pause)): house rules she is told ("Non entrare in camera da letto") become no-go rooms (best effort: she knows the room only from what she sees, by [room recognition](docs/behaviours.md#room-recognition); when she is not sure which room she is in, nothing is blocked), `[robot] max_autonomous_drive_s_per_hour` and `..._per_day` cap her driving time, and "Laika, resta ferma" pauses everything she does on her own until you say "puoi muoverti di nuovo". `python3 run-laika.py --unattended-check` lists what must be in place before you leave her alone ([safety calibration](docs/safety-calibration.md#unattended-checklist)).

Laika also looks out of the windows (`[outside]`, [docs](docs/behaviours.md#looking-outside)): in a room with a window she knows she points the head at it, takes a picture and asks the vision model about the weather, the light and what happens outside (people are only counted), at most every 30 minutes per window. Each look is a photo of the outside sent to the vision model, so it has a cost and privacy note in the example config. Ask "com'è fuori?" and she answers from the latest look and says when it was.

Laika also watches her battery (`[battery]`, two 18650 cells). Below `low_v` she stops driving around by herself and tells the next person she needs charging; below `critical_v` she refuses every wheel move and stays still. The thresholds are defaults: the hardware check compares the reading with a multimeter and suggests a `voltage_offset` ([calibration](docs/safety-calibration.md#battery)).

Prepare about 1 m of free floor, a box (to put about 20 cm in front of the robot), and be ready to lift the robot in your hands. Use `--no-mic` to skip the microphone step and `--no-driving` to skip the steps where the wheels spin. `python3 run-laika.py --sim --check-hardware --yes` is a dry run on a laptop. At the end Laika prints a summary and writes `hardware-check-<timestamp>.json` and `.md` to the `logs/` folder of the data directory; the exit code is 0 only if nothing failed. Attach the report to GitHub issue #15.

### Field trial report

`python3 run-laika.py --report [--days N] [--since YYYY-MM-DD]` summarises what Laika did in the last day (or N days): the core loop per stage, completed and open question loops, safety stops, restarts, AI outages and conversations. It reads the event log and the database read-only, needs no AI key or hardware, and writes `trial-report-<date>.md` and `.json` to the `logs/` folder of the data directory. It holds counts, object and room names and times, no transcripts or images. See [field trial](docs/field-trial.md).

### Prototype layout

By default the model acts through tools (`[ai] use_tools = true`): it calls `move`, `look`, `capture_image`, `speak`, `remember`, `recall` and so on, a local validator checks every call, and wheel movement still goes through the safety controller. Invalid or stale calls are rejected and logged as `[laika] tool rejected: ...`. If your model has no tool calling, set `use_tools = false` and it replies with the JSON `{"say", "actions"}` format instead.

| Module | Role |
| --- | --- |
| `app.py` | Builds everything from the settings and runs until Ctrl+C |
| `conversation.py` | Sleep → wake word → ask → answer and gesture → follow-ups; picks tool mode or the JSON fallback |
| `agent.py` | Tool-calling loop: asks the model, runs its tool calls one by one, bounded rounds and history, cancels stale turns |
| `tools.py` | The tools the AI acts through (move, stop, look, capture_image, scan_environment, find_object, find_person, approach, speak, remember, recall, what_changed, forget, gesture, ...) and the validator that checks every call first |
| `transcribe.py` | Opt-in cloud speech to text (`/audio/transcriptions`) for `CloudEar` |
| `wake_word.py` | Fuzzy "Laika" matching on what Vosk heard |
| `audio.py` | Vosk listening (optionally with cloud transcription after the wake word, `CloudEar`); Edge voice with Piper fallback, or Piper only; or the console |
| `ai_client.py` | Provider-independent chat client (OpenAI, Azure OpenAI, LM Studio, Ollama, vLLM) with tool calls, timeouts and bounded retries |
| `prompts.py` | Laika's Italian personality: the tool-mode prompt (answers in plain text) and the older JSON reply prompt |
| `actions.py` | Gestures the AI may choose, played on one worker thread |
| `commands.py`, `events.py`, `world_model.py` | Typed local commands and telemetry, the event bus and AI endpoint monitor, the world model (in memory, persisted to the store by `WorldModelPersister`) |
| `behaviour.py` | Behaviour priority: the `Arbiter` that owns the body, leases, pre-emption and resume, a heartbeat for autonomous behaviours, and safety overrides ([docs](docs/behaviours.md)) |
| `autonomy.py`, `house_rules.py`, `unattended.py` | Unattended autonomy (#51): the per-hour and per-day driving budget and the "resta ferma" pause (both kept in the store), no-go rooms read from the rule facts and `[house]`, and the `--unattended-check` checklist; all enforced by the safety controller ([docs](docs/behaviours.md#house-rules-driving-budget-and-pause)) |
| `rooms.py` | Room recognition (#53): the vision model guesses the room in every frame, `RoomTracker` votes over recent frames (confidence, landmarks, hysteresis, "unknown") and drives `world.current_room` and `RoomEntered` on the robot (shadow mode in `--sim`); "questa è la camera di Leonardo" names a room (`name_room`), and `[rooms]` tunes it ([docs](docs/behaviours.md#room-recognition)) |
| `idle.py` | What she does when nothing happens (#47): `IdleSelector` picks explore, look outside, follow up, check on the house or rest by a documented utility formula (time of day, recency, pending questions, personality, battery, presence), bounded, under leases, interrupted by the wake word, remembered as `idle` episodes ([docs](docs/behaviours.md#when-laika-is-idle)) |
| `windows.py`, `outside.py` | Looking out of the windows (#48): a registry of windows (room and camera direction, learned from vision and from "la finestra è dietro il divano"), `WindowWatcher` points the head at one, takes an `outside` picture and asks the vision model for weather, light, events and a people count (validated, never identified), remembers it as an `outside` episode, notices changes (`outside_change`, "started raining") and feeds curiosity; the agent mentions it once and `look_outside` answers "com'è fuori?" ([docs](docs/behaviours.md#looking-outside)) |
| `battery.py` | Battery monitor (#50): reads the pack voltage on its own slow timer, smooths it and says OK, LOW or CRITICAL with hysteresis; LOW stops autonomous driving and she tells people she needs charging, CRITICAL refuses every wheel move ([docs](docs/behaviours.md#battery), [calibration](docs/safety-calibration.md#battery)) |
| `safety.py` | Safety controller: a 20 Hz monitor thread stops every move on obstacle, cliff, timeout, hardware error or emergency stop; all wheel movement goes through it |
| `observations.py` | Structured observations: the schema, local validation of model output (malformed output is rejected and logged), a bounded observation log, and the `Observer` that turns captured frames into `ObservationCreated` events off the caller's thread |
| `explore.py` | Basic exploration (V0.4): a low-priority behaviour that moves a little, stops, looks around and repeats, every move gated by the safety controller; `Explorer` ([docs](docs/behaviours.md#exploration)) |
| `games.py` | Games to play together ("giochiamo?"): a catalogue of word games the model runs from their rules and three body games ("un, due, tre... stella!", statues, hide and seek) whose rounds run in code with the head and the ultrasonic sensor; `start_game`, `play_round`, `stop_game` ([docs](docs/behaviours.md#games)) |
| `songs.py` | Songs to sing ("cantami una canzone"): a catalogue of public-domain children's songs and Laika's own, sung verse by verse with gestures; the `sing` tool ([docs](docs/behaviours.md#songs)) |
| `tags.py` | AprilTag (tag36h11) position fixes: an OpenCV detector on the captured frames, size/bearing/distance, `TagReader` turns near tags the plan lists into a certain room ([docs](docs/apriltags.md)) |
| `floorplan.py` | The plan of the house: rooms, names, doors as a graph; plausible moves for the room tracker, the `house_route` tool, no-go rooms ([docs](docs/floor-plan.md)) |
| `learning.py` | Learning new skills: `propose_skill` / `learn_skill`, a background Copilot SDK job on a `skill/` branch, the gates (protected paths, tests, import check), commit, push, release ([docs](docs/learning.md)) |
| `bedtime.py` | Bedtime stories ("raccontami una storia della buonanotte"): a new story with Leo and Ludo each time, written with the owner's prompt, saved under `stories/`, told paragraph by paragraph, then "another one?" with a 15 s wait; the `bedtime_story` tool ([docs](docs/behaviours.md#bedtime-stories)) |
| `places.py` | What explorations teach her about each room: how it looks, which rooms it leads to, when it was explored, who she met; kept in the store and put in the conversation prompt; every session becomes an `explored` episode ([docs](docs/behaviours.md#exploration)) |
| `search.py` | Searching (#43): `Searcher` behind the `find_object`, `find_person` and `approach` tools: memory first, then a head scan, then (only if autonomous driving is allowed) goal-directed exploration that stops when a scan shows the target; people are counted, never identified; no map, so no "go to the kitchen" ([docs](docs/behaviours.md#searching-and-approaching)) |
| `scan.py` | `Scanner`: the `scan_environment` tool's head-only scan (left, centre, right), one fresh frame per position, merged into one `EnvironmentScan` |
| `vision.py` | Vision requests: sends one captured frame (optionally shrunk with Pillow) to the vision model through the AI client and returns a validated observation; size and per-minute limits, outages reported to the AI endpoint monitor |
| `camera.py` | Camera service: the one place frames are captured (minimum interval and per-minute cap), saved with bounded retention and announced as `ImageCaptured` |
| `curiosity.py` | Curiosity (#28): scores each observation for interestingness against the world model as it was before the frame (novelty, moved objects, people, unusual places, and more); announces `InterestingChangeDetected` above a threshold, with a cooldown per reason. Scores only ([docs](docs/curiosity.md)) |
| `changes.py` | Environment changes (#36): `ChangeDetector` compares each observation with the world model as it was before the frame (moved, new, missing, unusual), `ChangeTracker` keeps them and records "moved", "new" and "missing" episodes; the `what_changed` tool answers "what changed since yesterday?" ([docs](docs/curiosity.md#environment-changes)) |
| `memory.py` | Semantic and episodic memory (#34): `Fact`s (what people told her, where things usually stay, who owns them) and `Episode`s (seen, asked, answered, rules), kept in the store, recalled after a restart by the `remember`, `recall` and `forget` tools and shown in the prompt when relevant; `EpisodeRecorder` ([docs](docs/storage-and-retention.md#memory-facts-and-episodes)) |
| `conversation_memory.py` | Conversation memory (#35): which parts of a conversation become long-term memory. The model calls `remember` or `answer_question`; `MemorySelector` is the safety net that applies Italian patterns after each turn (what an object is, ownership, where it stays, house rules, preferences, answers to pending questions), in tool and JSON mode, with `[memory] auto_select` and `min_confidence`. Telling her what an object is makes it a known object ([docs](docs/storage-and-retention.md#what-laika-remembers-from-conversations)) |
| `investigate.py` | Investigation (#29): reacts to `InterestingChangeDetected` and exploration scans under a CURIOSITY lease: points the head, takes another picture, may move closer (autonomous gate and safety apply), then queues an Italian question for later or remembers the sighting; `PendingQuestions`, the `investigate` and `answer_question` tools ([docs](docs/behaviours.md#investigation)) |
| `presence.py` | Knows when a person is near from the observation's head count only (no faces or names): `PersonDetected` / `PersonLost` with confidence, hysteresis and a timer; idle presence photos come from the capture policy (`[presence]`, [docs](docs/behaviours.md#people)) |
| `encounter.py` | Meeting people while she is out and about (#49): a person appearing during exploration (or an investigation or a look out of a window) stops her, turns her head to them and asks the policy (a short hello counts as something to say for an encounter only); allowed: a proactive conversation, optionally after one short approach move; not allowed: silence, at most a nod ([docs](docs/behaviours.md#meeting-people), `[encounter]`) |
| `proactive.py` | Proactive questions (#40): the trigger that asks the policy on `PersonDetected`, interest and a re-check, and hands a pending question to the conversation, which asks it with the head turned to the person ([docs](docs/behaviours.md#proactive-questions), `[interaction] proactive`) |
| `interaction_policy.py` | The deterministic interaction policy (#39): availability states (available, busy, do not disturb, away, sleeping, unknown), quiet hours, cooldown, per-hour limit, and `decide()` for the proactive path; the do-not-disturb and busy tools ([docs](docs/behaviours.md#interaction-policy), `[interaction]`) |
| `capture_policy.py` | When Laika takes photos on her own: after a completed move, when a person or something interesting appears, during investigation and exploration, and idle checks for people; per-trigger switches and cooldowns, one per-minute cap, a worker thread with a one-slot pending queue |
| `storage.py` | Data directory, image store, rotating event log and optional transcripts, all with size or age limits |
| `store.py` | Local persistent store (#32): one SQLite file (`memory/laika.db`) with versioned migrations, a bounded non-blocking write queue on its own thread, retention by count and age, observations restored after a restart ([docs](docs/storage-and-retention.md#the-database)) |
| `web.py` | Looking things up (#52): `WikipediaClient` and `BraveSearchClient` (requests only, HTTPS, timeouts, one retry, results stripped and capped), a small in-memory cache and an hourly limit; behind the `wikipedia` and `web_search` tools, which taint the turn so web text can never move the robot or write memory |
| `systemd.py` | Optional systemd notifications (ready, watchdog ping, stopping), a no-op outside a service |
| `hardware_check.py` | The guided `--check-hardware` script and its JSON and Markdown report |
| `trial_report.py` | The `--report` field-trial report (#44): reads the event log and the store read-only and summarises the core loop, completed and open loops, supervision issues and conversations, per day ([docs](docs/field-trial.md)) |
| `robot.py`, `hardware.py`, `simulator.py`, `house.py`, `limits.py` | Robot interface and state models, PiCar-X adapter, simulated robot and its 2D house, shared limits |

## Documentation

- [Current PRD and roadmap](docs/prd.md): requirements for the Python application running on the Raspberry Pi.
- [Initial project brief](docs/archive/prd-initial.md): the original supplied brief, preserved unchanged for historical reference.
- [Current status and confirmed baseline](docs/current-status.md): installation and example verification, evidence boundaries, and outstanding integration details.
- [Field trial](docs/field-trial.md): how to run the multi-day trial in the house, the daily `--report` checklist, a measurable bar for "the loop completes without supervision", and how to post findings to issue #44.
- [Running as a service](docs/service.md): systemd unit, install script, logs, and how the robot is stopped on exit and restart.
- [Safety calibration](docs/safety-calibration.md): every `[robot]` safety setting with its range, and the on-robot calibration procedure for the hardware check.
- [Curiosity](docs/curiosity.md): how an observation's interestingness is scored, the defaults and threshold, and the cooldown.
- [Storage and retention](docs/storage-and-retention.md): data directory, permissions, and every disk and memory limit.
- [Navigation options](docs/navigation.md): what Laika can and cannot do today, an evaluation of AprilTags, room recognition, SLAM, LiDAR and ROS 2 for this hardware, `return_to_base`, and the recommendation (issue #45).
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
