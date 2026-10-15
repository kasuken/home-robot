# 0002: Watchdog strategy and AI endpoint outage handling

Status: accepted

## Context

The PRD ([Local Application Boundaries](../prd.md), [Safety](../prd.md)) requires a verified watchdog strategy for stalled application control before unattended movement is enabled, and requires that loss of the AI endpoint stops autonomous movement without replaying stale actions later. Laika is synchronous Python with threads ([ADR 0001](0001-threads-not-asyncio.md)): the safety monitor (`SafetyController`, 20 Hz, its own thread) is the only code that ends a move. Three things can leave the motors running: the application thread that asked for the move stalls, the monitor itself stalls or dies, or the whole process dies.

## Decision

### 1. Dead-man's switch for agent control (existing, now tested end to end)

Every move carries its own deadline, at most 1.5 s (0.6 s backwards), enforced by the monitor, not by the caller. A command older than `command_max_age_s` (1 s) is EXPIRED and never starts. Movement therefore continues only while fresh commands keep arriving: a hung agent, conversation or AI-request thread can keep the robot moving for at most one command duration plus two monitor ticks. Test: `DeadMansSwitchTests` (a thread issues a long move and then hangs forever; the robot is stopped by the deadline plus two ticks).

### 2. Supervising the safety monitor: `SafetyWatchdog`

The remaining gap is the monitor thread itself stalling (for example a robot call that never returns) or dying. A separate lightweight thread, `SafetyWatchdog` in `safety.py`, polls every 100 ms (`watchdog_period_s`) and compares `time.monotonic()` with the monitor's last-tick time, which the monitor stamps at the start and end of every tick. It expires when the last tick is older than `watchdog_timeout_s` (default 0.5 s, ten missed ticks) or when the monitor thread is no longer alive. On expiry it:

1. calls `robot.stop()` directly, wrapped in try/except, WITHOUT taking the controller lock (a stalled monitor may hold it; the robot's own lock is short). It repeats this on every poll while the monitor stays stalled;
2. latches `watchdog_expired`: new moves are REJECTED with reason `watchdog`, again without taking the controller lock;
3. cancels the active move (CANCELLED `watchdog`). If the lock is free this goes through the normal path; if the monitor holds it, the move is settled without the lock so a caller blocked in `execute` or `wait` is released, and the monitor discards it if it ever resumes;
4. publishes `LocalWatchdogExpired` once per expiry (reason `monitor_stalled` or `monitor_dead`, last-tick age, timeout).

Clearing: the flag clears by itself as soon as the monitor ticks again (a stalled monitor that recovers). A cancelled move is never restarted: after recovery only new commands can move the robot. A dead monitor thread is not restarted, so the controller stays expired and refuses moves until the process is restarted.

Guaranteed time to stop: `watchdog_timeout_s` plus one watchdog period plus the monitor's own tick period (0.5 + 0.1 + 0.05 s by default, about 0.65 s), assuming `robot.stop()` itself returns. `watchdog_timeout_s` is configurable in `[robot]` (0 disables it). Nothing supervises the watchdog thread; it only reads a timestamp and calls `robot.stop()`, and it catches every exception.

`SafetyController.close()` and `wait()` also avoid blocking on a stuck monitor.

### 3. Process exit

If the Python process is killed (SIGKILL, power-cycle of the Pi's service, crash of the interpreter) the Robot HAT keeps its last PWM output and the wheels keep turning. No code in this process can handle that. `app.py` stops the robot in `finally` for normal exits and Ctrl+C; for everything else the service supervisor stops the motors on exit. This is implemented by the headless service in issue #18 ([service](../service.md)): the unit's `ExecStopPost` runs `run-laika.py --stop-motors`, `Restart=on-failure` restarts Laika, and a `Type=notify` systemd watchdog (pinged from the safety monitor) restarts it when the monitor hangs. SIGTERM is handled in-process so a normal stop runs the usual shutdown. Unattended movement must not be enabled before this is verified on the robot.

### 4. Per-behaviour lease (deferred)

A heartbeat or lease for long autonomous behaviours (exploration) is deferred to issue #27 (exploration loop). Such a behaviour would have to renew a lease on every step; when it lapses the controller stops the robot. Today every move is a single bounded command, so the dead-man's switch above is sufficient.

### AI endpoint outage

`AIEndpointMonitor` accepts optional callbacks run synchronously on the unavailable and recovered transitions, before the event is published: the safety reaction does not depend on the event bus, so a slow subscriber cannot delay it. The events `AIEndpointUnavailable` and `AIEndpointRecovered` are still published.

- Unavailable: `SafetyController.set_ai_available(False)` stops the motors, cancels the active move (CANCELLED `ai_unavailable`) and rejects new moves (REJECTED `ai_unavailable`) until recovery. `ActionPlayer.cancel_pending()` drops every queued gesture; a gesture that is already running finishes, but any drive it starts is rejected. Head and steering gestures still work, so the Italian "no connection" message keeps its `sad` gesture; the conversation cancels pending gestures before playing it.
- Recovered: `set_ai_available(True)`. Nothing is replayed: the queue was cleared, and new commands are created fresh (old ones would be EXPIRED anyway).
- Today every move originates from the AI (gestures chosen by the model), so "autonomous movement" means all wheel movement. Direct user commands, when they exist, could be exempted. The local `stop` command always works.
- The endpoint is only probed by real requests, so recovery is noticed on the next successful reply.

## Consequences

- Stalled application threads, a stalled or dead monitor and an AI outage each stop the simulated robot within a defined time; tests in `tests/safety/` cover them.
- The monitor's per-tick work must stay short: anything that can block for more than `watchdog_timeout_s` belongs on another thread.
- A hung `robot.stop()` cannot be handled in-process; the hardware adapter must keep its calls short.
- The supervisor requirement (stop on process exit) is implemented in `deploy/laika.service`, pending verification on the robot (#18, #15). Unattended movement stays disabled until that is confirmed.
