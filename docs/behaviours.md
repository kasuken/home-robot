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
3. every `scan_every_steps` steps, or after turning away, stop and `Scanner.scan()` (head only; the lease is kept alive during the scan) and hand the `EnvironmentScan` to `on_scan` (the seam the investigator uses, see below: it may answer `"continue"` or `ExploreDecision("stop")`). The capture policy's exploration timer is on while exploring (`set_exploring`).

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

## Investigation

`src/laika/investigate.py`: when something is interesting, Laika looks into it without always interrupting anybody (#29). The `Investigator` runs on its own worker thread, one investigation at a time, and **never speaks on its own**: whether and when to interrupt someone is the interaction policy's job (V0.6).

**Triggers.** `InterestingChangeDetected` events ([curiosity](curiosity.md)); the explorer's `on_scan` seam (`Investigator.consider_scan` queues every scan observation whose score reached the threshold and answers "continue"); and the `investigate` tool. Only objects the score is about become targets (new, moved, unknown or in an unusual place, read back from the reasons); "person nearby" alone does not. A target is queued (size 3, the newest replaces the oldest) unless the observation was already handled, that object is already queued or it is cooling down. The same observation reaching the investigator through the event and the scan is investigated once.

**Steps** (under a CURIOSITY lease, `kick()` and `check()` between steps, a heartbeat while it waits for the camera or the vision model):

1. inspect: point the head at the pan and tilt of the observation that raised interest;
2. take another picture (camera reason `investigate`) and wait, bounded, for its observation: the object must still be there, and the confidence is refreshed. If it is not, nothing else happens (no question, no memory);
3. move closer, only if the object was roughly ahead (|pan| at most 20 degrees) and the distance reading ahead is above `approach_stop_cm`: one short forward move (speed 20, 800 ms, clamped by the safety layer) under the lease, then another picture. At most `max_approach_moves` (1 by default, 0 turns it off). This is an autonomous move like exploration's, so **with the autonomous-driving gate closed (the robot's default) the safety layer answers `autonomous_disabled` and the step is skipped**; looking and taking pictures still happen;
4. outcome: an object that is new to Laika, unknown or doubtful (confidence below `[curiosity] unknown_object_confidence`, or a vague name) and that the user never described becomes a queued question. Anything else is remembered as a memory fact ("Saw backpack next to sofa in the living room"). Nothing else is published: the pictures and observations travel as usual (`ImageCaptured` with reason `investigate`, `ObservationCreated`). The log says `[laika] investigating <object> in <room>` and how it ended.

A pre-empting behaviour (the wake word) ends the investigation at the next step: no question and no memory fact is made, the head is not touched again. The lease is not resumable. A pre-empted explorer resumes by itself when the investigation releases the body. The investigation also ends after 60 s, and a queued one that cannot get the body for 10 s (a chat has it) is dropped.

**Limits** (`[investigate]`, validated): `max_per_hour` investigations started on her own (12), `object_cooldown_s` per object (600), `approach_stop_cm` (50), `max_approach_moves` (1), `question_ttl_h` (24). The capture policy's `investigate()` switch is not used: the investigator takes its pictures through the camera directly, because it must wait for the result; the camera's own rate limits still apply.

**Pending questions.** `PendingQuestions` holds at most 5 questions (the oldest goes first), one per object, expiring after `question_ttl_h`. The text is a template, not the model: "Ho visto una cosa nuova in soggiorno: {object}. Che cos'è?" (the simulator's rooms are translated: soggiorno, cucina, studio, camera da letto; any other room is "in casa"). During a conversation the agent's system prompt lists up to three of them, with an instruction to ask at most one, briefly, only if the moment fits. When the user answers, the model calls `answer_question(question_id, answer)`: the answer becomes the object's note in the world model (`learn`) and a memory fact, and the question is removed, so the object counts as known from then on.

**The `investigate` tool.** `investigate(object?)` investigates the named object at its last known place (head pointed there if a pan is known), or the latest interesting observation when there is no name, and returns what was found, whether it moved closer and whether a question was queued. It runs on the agent's thread under the turn's **CONVERSATION** lease, which an autonomous investigation could not pre-empt. Moving closer is still Laika's own decision (how far, and whether to), so it follows `allow_autonomous_driving` like every autonomous move: with the switch off (the robot's default) the investigation points the head and takes pictures but never drives. Only the `move` tool, where the user asks for a specific move, drives with the switch off. It is not counted against `max_per_hour` or the cooldown, and it still queues a question for an unknown object. If a background investigation is running it waits up to 3 s for it.

## People

Laika knows when someone is near, from a head count only (`presence.py`, `[presence]`). It does not recognise anyone.

**How it works.** The vision model counts the people in each frame (`people` in the observation; the prompt forbids identifying anyone). In the simulator the count is ground truth from the house. `PresenceTracker` listens to `ObservationCreated`:

- a frame with `people > 0` and confidence at least `min_confidence` (0.5) marks "a person is here". The change from absent to present publishes `PersonDetected` once, with `count`, `confidence`, `room`, `pan`, `tilt`, `since` (wall time) and `duration_s` (0 on arrival);
- `PersonLost` (same data, `duration_s` = how long the visit lasted) follows when no such frame arrived for `lost_after_s` (30). It is checked on every observation and by a small timer thread that sleeps until the deadline, so it fires even if no more frames come;
- frames showing nobody, or only low-confidence people, do not end a visit by themselves and do not extend it: one blurry frame never flaps, and a visit ends only after `lost_after_s` without a good sighting. A person who comes back after a longer gap is a new arrival.

The tracker exposes `present`, `count`, `last_seen` and `since` for the interaction policy (#39). The simulator no longer publishes person events itself: this is the only source.

**Idle pictures.** Frames only exist when she takes pictures, so the capture policy has a `presence` trigger: while nobody is known to be near and no conversation holds the body, one picture every `presence_interval_s` (60; 0 = off); while someone is near, one every `presence_check_s` (30; 0 = off) to notice they left. They share the `[capture] max_auto_per_minute` cap and the `[camera]` and `[vision]` limits, and any other automatic picture counts as a presence check too. `PersonDetected` also triggers the usual `person` picture (`on_person`).

**Cost.** With `[vision]` on, every idle picture is a vision request: the default 60 s is up to 60 an hour while nobody is around (the same for `--sim`, which uses ground truth and costs nothing). For a cheaper robot set `presence_interval_s = 300` (12 an hour) or `0` (she then notices people only in pictures taken for other reasons, and `PersonLost` still fires after `lost_after_s`).

**Privacy.** Only a count, the room and where the head pointed are kept in the event and the observation. No faces, names or identities. Pictures are stored only under the normal image retention (`[storage] max_images`, `max_images_mb`); nothing extra is kept for people.
