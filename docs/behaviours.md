# Behaviour priority

The PRD ranks what Laika does and says lower priority must never interfere with higher priority. `src/laika/behaviour.py` enforces it with an `Arbiter` that owns the body (wheels, head, gestures).

| Priority | Value | Holds a lease? |
| --- | --- | --- |
| Safety | 0 | No: it is the safety controller itself and overrides everyone |
| Direct user command | 1 | Yes |
| Active conversation | 2 | Yes (the wake word until the chat ends) |
| Pending follow-up | 3 | Yes |
| Important observation | 4 | Yes |
| Curiosity | 5 | Yes, with heartbeat |
| Social interaction | 6 | Yes, with heartbeat |
| Random exploration | 7 | Yes, with heartbeat |

A lower value is more important.

## Requesting, pre-empting, resuming

`arbiter.request(name, priority, on_preempt=..., on_resume=..., resumable=True)` returns a `Lease`, or `None` when the current holder is at least as important (the behaviour retries later). A strictly more important request pre-empts the holder:

1. the holder's lease is revoked, so its next `controller.execute(..., lease=lease)` is REJECTED `preempted`;
2. the arbiter calls `safety.stop("preempted")`: a move in flight ends CANCELLED `preempted`; and `actions.cancel_pending()` drops queued gestures;
3. `on_preempt(lease)` runs, outside the arbiter's lock;
4. only then does the new holder get its lease, so its first move is not cancelled by the stop.

A resumable holder waits on a small bounded list (4). When the more important behaviour calls `arbiter.release(lease)` the most important waiting behaviour gets the body back: the same `Lease` becomes active again and `on_resume` runs. A behaviour loop calls `lease.check()` between steps (it raises `Preempted`) and `lease.wait_until_active(timeout)` while pre-empted; `wait_until_active` returns False when the lease ended. A non-resumable holder just ends.

`USER_COMMAND` exists for the future direct-command path; there is no separate command parser yet (a request heard during a conversation is handled by the agent under the CONVERSATION lease), so it is exercised through the arbiter directly in the tests.

## Leases never bypass safety

`SafetyController.execute(command, lease=...)` adds one check, "is the lease still active", and removes none: obstacle, cliff, emergency stop, watchdog, AI outage, expiry and every other check still apply. Moves without a lease (gestures that drive) behave as before; the arbiter cancels queued gestures on pre-emption.

The conversation holds a CONVERSATION lease from the wake word to the sleep word or silence, and the agent's tools run under it (`Agent.respond(..., lease=...)`): once the lease is pre-empted the turn goes stale and `move`, `look` and `gesture` are refused. If the body is held by something more important when Laika wakes, she still talks but every move is refused.

## Safety overrides

The arbiter registers with `SafetyController.add_halt_listener`. On an emergency stop, watchdog expiry, AI endpoint outage or hardware error it revokes every lease below CONVERSATION at once (holder and waiting) and calls their `on_preempt`; it does not call `safety.stop()` itself because the safety layer already stopped the motors, and it never blocks a safety thread (callbacks run on a short-lived thread). Conversation and user-command leases stay, so Laika can still talk, but their moves remain subject to the controller. While one of those conditions holds, requests below CONVERSATION are refused.

## Heartbeat

Autonomous leases (curiosity, social, exploration) must call `lease.kick()` at least every `lease_timeout_s` (default 2 s). A small timer thread revokes a stalled lease (end reason `lease_expired`), calls `safety.stop("lease_expired")` and offers the body to the next waiting behaviour. This is the per-behaviour lease planned in [ADR 0002](decisions/0002-watchdog-strategy.md); the exploration loop (#27, below) uses it.


## The autonomous-driving gate

Moves made under an autonomous lease (curiosity and below: `lease.autonomous`) are REJECTED `autonomous_disabled` unless the `SafetyController` was built with `allow_autonomous=True`. Conversation, user-command and lease-less moves are not affected.

The gate follows `[robot] allow_autonomous_driving`, which is **false by default** and must stay false on the real robot until the V0.2 hardware check (#15, #17, #18) has passed. `run-laika.py --sim` turns it on for the simulator only; nothing turns it on for the PiCar-X. To enable it after the check, set `allow_autonomous_driving = true` in `config/laika.toml` and restart. With the gate closed exploration still works head-only, see below.

## Exploration

`src/laika/explore.py`: `Explorer` is the `EXPLORATION` behaviour (lowest priority, resumable). Start it by asking ("Laika, esplora la casa", the `start_exploring` tool, optional `duration_s` 10 to 300) or with `[explore] idle_after_s` (start after that long with nobody talking, 0 = never). `stop_exploring` ends it. The tool is called during a conversation, which holds the body, so exploration starts moving only when the conversation ends. It runs on its own thread and does one step at a time:

1. kick the lease and check it;
2. one fresh `MoveCommand` (forward, `[explore] speed` 15, `step_ms` 800, both clamped by the safety layer) under the lease; wait for how it ended before making the next. Nothing is ever queued;
3. every `scan_every_steps` steps, or after turning away, stop and `Scanner.scan()` (head only; the lease is kept alive during the scan) and hand the `EnvironmentScan` to `on_scan` (the seam for #28/#29: it may answer `"continue"` or `ExploreDecision("stop")`). The capture policy's exploration timer is on while exploring (`set_exploring`).

What it does on each outcome of a move:

| Outcome | Reaction |
| --- | --- |
| success | next step |
| `obstacle` | back away on an arc (a short reverse with steering, direction kept until a step succeeds, flipped after that), scan, continue |
| `cliff` | straight back a little, then the arc, scan, continue. A backward move may start with the edge in view; it stops if a cliff comes back after clearing |
| `autonomous_disabled` | head-only for the rest of the session, no retry |
| `preempted` (the wake word, a user command) | pause: no move, no capture; resumes when offered the body, the pause does not count against the duration |
| emergency stop, watchdog, AI outage, hardware error, `lease_expired`, `disabled` | exploration ENDS and logs why; it does not resume by itself when the condition clears |
| six steps in a row that went nowhere | ends `stuck` (for example a cliff reading that never clears) |

It also ends after `[explore] max_duration_s` (default 180 s; a longer request is cut to it), after `max_steps`, on `stop()` or on an `on_scan` "stop". Its lease needs a heartbeat: if the thread stalls for more than the lease timeout (2 s) the arbiter revokes the lease and stops the robot.

**Head-only mode.** With the gate closed (the default on the robot), `--no-driving` or `allow_driving = false`, exploration does not drive: it waits, scans the surroundings with the head every 10 s and says so in the log (`head-only`). That is useful on the robot before V0.2 is verified. Each scan is three photos and, with `[vision]` on, up to three vision requests, so mind `[vision] max_per_minute`.
