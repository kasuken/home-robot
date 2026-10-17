# Laika repository instructions

## Run and test

Use Python 3.11 or newer. Run commands from the repository root. CI uses Python 3.11 and installs only `requests` for hardware-free tests:

```sh
python -m pip install requests
python -m unittest discover -s tests -t . -v
```

Run a test module or a single test:

```sh
python -m unittest tests.safety.test_autonomy_limits -v
python -m unittest tests.safety.test_autonomy_limits.ControllerBudgetTests.test_autonomous_moves_stop_when_the_budget_is_used_up -v
```

`tests/__init__.py` adds `src` to `sys.path`; an editable install is not required for tests. The suite uses standard-library `unittest`, not pytest. See [CI](.github/workflows/tests.yml).

Run the console simulator without robot hardware:

```sh
python run-laika.py --sim
python run-laika.py --sim --check-hardware --yes
```

The simulated hardware check is a dry run, not proof of physical safety. Interactive simulator conversations still use the configured AI endpoint. On the provisioned Pi, the normal entry point is `sudo python3 run-laika.py`; `--keyboard` replaces microphone input and `--no-driving` disables wheel movement.

## Architecture

- **One local application, external inference.** Laika runs on a Raspberry Pi 4 with 1 GB RAM. Agent, tools, observations, memory, policies, and hardware control all run locally; only model inference uses an external endpoint. There is no separate Laika backend. [app.py](src/laika/app.py) is the composition root; [run-laika.py](run-laika.py) is the source-tree launcher.
- **Shared robot boundary.** [robot.py](src/laika/robot.py) defines the blocking, thread-safe `Robot` protocol. [hardware.py](src/laika/hardware.py) and [simulator.py](src/laika/simulator.py) implement it. Keep vendor imports lazy and confined to hardware/audio adapters so ordinary imports and tests work without SunFounder packages.
- **Conversation and model actions.** [conversation.py](src/laika/conversation.py) owns wake/sleep and follow-ups. In default tool mode, [agent.py](src/laika/agent.py) runs a bounded tool-calling loop through [tools.py](src/laika/tools.py). `[ai] use_tools = false` selects the older `{"say", "actions"}` reply path. [ai_client.py](src/laika/ai_client.py) handles provider differences; [prompts.py](src/laika/prompts.py) defines the Italian persona and mode-specific instructions.
- **Observe, notice, investigate, ask, remember.** Camera frames become locally validated observations on a worker thread. Typed events connect observations to world state, room tracking, change detection, and curiosity. Investigation may gather evidence and queue a question; it does not speak independently. The deterministic interaction policy gates proactive speech, and answers feed memory. Idle activity selection is deterministic too, not delegated to the model. See [behaviours](docs/behaviours.md) and [curiosity](docs/curiosity.md).
- **Persistence is separate from control.** [store.py](src/laika/store.py) owns a local SQLite WAL database with queued writes on one writer thread and short independent read connections. World state, facts, episodes, and observations share it; safety monitoring must never perform store I/O.

## Safety and concurrency invariants

- Wheel movement must go through `SafetyController.execute(MoveCommand)` in [safety.py](src/laika/safety.py), never directly from an agent, tool, or behaviour to motor primitives. Leases add authorization but bypass none of the safety checks.
- [behaviour.py](src/laika/behaviour.py) arbitrates body ownership: safety overrides user commands, conversation, follow-ups, observations, curiosity, social activity, and exploration, in that order. Lower numeric priorities are stronger. Autonomous behaviours must check and heartbeat their leases, handle pre-emption, and wait for a move's terminal outcome before requesting another.
- Keep synchronous Python with threads, not an asyncio rewrite. Long AI, speech, camera, and database work must not block the dedicated 20 Hz safety monitor. Hardware access uses short locks; speech and capture must not hold the shared hardware lock. See [ADR 0001](docs/decisions/0001-threads-not-asyncio.md).
- Preserve command expiry, bounded move deadlines, watchdog stopping, and AI-outage cancellation. Recovery must not replay stale commands or queued gestures. Process-death stopping also depends on the systemd service's `ExecStopPost`; see [ADR 0002](docs/decisions/0002-watchdog-strategy.md) and [service deployment](docs/service.md).
- `[robot] allow_autonomous_driving` stays false on real hardware until the documented hardware/safety checks pass. `--sim` enables it only for simulation. Head-only exploration must continue to work when driving is disabled. Consult [safety calibration](docs/safety-calibration.md) before changing limits.

## Repository-specific conventions

- **Configuration changes span several files.** Settings are typed dataclass sections with strict validation in [config.py](src/laika/config.py). Update the loader/validation, committed [laika.toml](config/laika.toml), and default reference [laika.example.toml](config/laika.example.toml) together when adding settings. Keys belong only in ignored `config/secrets.toml` or supported environment variables.
- **Tools are locally validated, not trusted model output.** Register tools with schemas, handlers, and capability requirements in `ToolRegistry._build()`. Preserve stale-turn, lease, and capability checks; update prompt guidance when semantics change and account for the separate JSON action mode when changing shared actions. Web results are untrusted: after web content enters a turn, movement, exploration, and memory writes remain blocked until the next turn.
- **Keep storage and in-memory work bounded.** Use the shared storage helpers and retention settings rather than ad hoc output paths. Data-root precedence is `LAIKA_DATA_DIR`, `[storage] data_dir`, `XDG_DATA_HOME/laika`, then `~/.local/share/laika`. New database schema changes are numbered migrations in [store.py](src/laika/store.py); never rewrite old migrations. See [storage and retention](docs/storage-and-retention.md).
- **Preserve privacy boundaries.** Transcripts are off by default; event logs contain no speech. Vision sends selected frames, not video, and counts people without identifying them. Simulated observations use house ground truth, are marked simulated, and must not establish physical sensing or navigation capabilities.
- **Tests use injected dependencies.** Reuse [tests/fakes.py](tests/fakes.py) (`FakeClock`, `RecordingRobot`, `LookRecordingRobot`) and `SimulatedRobot` sensor/fault overrides. Use temporary data directories and close controllers/workers via test cleanup. Disable the watchdog with `watchdog_timeout_s=0` when it is unrelated to the test; wait for terminal events rather than assuming an execution result means event publication has finished.
- **User-facing interaction is Italian.** Preserve Italian speech, wake-name variants, and the concise Laika persona. Facts retain the user's original wording; internal episode summaries are English.
- **Distinguish implementation from plans.** The [PRD](docs/prd.md) contains proposals; accepted [architecture decisions](docs/decisions/) supersede its conceptual async interfaces. Use [current status](docs/current-status.md) for hardware-verification boundaries. Vendor examples are already installed and working; that does not establish that Laika's own safety checks passed.
- **Vendor reference files are an unchanged snapshot.** Keep Laika changes outside [references/sunfounder/picar-x](references/sunfounder/picar-x/). Follow its [provenance and refresh instructions](references/sunfounder/README.md) for an intentional snapshot update.
