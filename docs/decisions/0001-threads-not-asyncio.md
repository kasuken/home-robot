# 0001: Threads, not asyncio

Status: accepted

## Context

The PRD sketches the `Robot` interface with `async` methods. The SunFounder libraries (`picarx`, `robot_hat`, Vosk, the TTS engines) are blocking and not async-aware. The Raspberry Pi 4 has 1 GB of RAM, and a safety loop (issue #6) must keep running while the AI request, speech and camera work are in progress.

## Decision

Laika stays synchronous Python with threads. The `Robot` interface is plain blocking methods (`get_state`, `get_sensors`, `look`, `capture_image`, `speak`, ...). Long work (AI requests, audio, camera) runs on its own threads, and the safety loop gets a dedicated thread.

Robot implementations are thread-safe: each guards hardware access with an internal `threading.RLock`. Speech and the camera are the exceptions: `speak` and `capture_image` can block for seconds, so they do not hold the hardware lock (the camera has its own) and never delay the safety loop.

## Consequences

- No event loop to share with blocking vendor calls, so a slow call cannot freeze the safety checks. Each thread is easy to reason about.
- Every shared object (robots, queues, state) must be safe to call from several threads, and bounded.
- Wheel movement goes through `MovementGuard.execute(MoveCommand)`; the robot has no `move()` of its own.
- If a later component needs async (for example a network server), it can wrap the blocking interface with `asyncio.to_thread` at the edge instead of changing the core.
- The PRD's `async` sketch is conceptual; this decision supersedes it for the implementation.
