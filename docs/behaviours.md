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

`Explorer.run_goal` runs the same loop on the caller's thread under the caller's lease, for the search tools (see "Searching and approaching"). It also ends after `[explore] max_duration_s` (default 180 s; a longer request is cut to it), after `max_steps`, on `stop()` or on an `on_scan` "stop". Its lease needs a heartbeat: if the thread stalls for more than the lease timeout (2 s) the arbiter revokes the lease and stops the robot.

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

## Searching and approaching

`src/laika/search.py`: `Searcher` backs the `find_object(name, max_seconds?)`, `find_person(max_seconds?)` and `approach(target)` tools (#43). They run on the agent's thread under the turn's **CONVERSATION** lease, one call at a time, like `investigate`. With no conversation (a test, a script) the searcher asks the arbiter for a CONVERSATION lease for the length of the call. `find_*` need a camera with a pan head, `approach` also needs the wheels, steering and the distance sensor; `[robot] allow_driving = false` hides `approach`.

**Order: memory first, then a head scan, then exploring.**

1. *Memory.* `find_object` reads `WorldModel.where_is` (an exact name, else the most recently seen whole-word match), `usual_room`, the newest episodes about it and any location fact the user told her. They become `last_known` in the result (room, `when`, `ago`, usual room, note, what she was told). `find_person` asks the presence tracker: if someone is near she points the head where they were and confirms with a picture (`how: "presence"`); otherwise `last_known` is the latest observation that counted people.
2. *Head scan.* A scan with the head only (wheels never move), when she is in the room the object was last seen in, does not know where she is, has nothing in memory, may not drive at all, or is looking for a person. If memory says the object is in another room and she may drive, she skips this and goes exploring.
3. *Exploration* (`Explorer.run_goal`), only when she may drive: she wanders and scans as she does when exploring, and the exploration **stops as soon as a scan shows the target**. It is bounded by `max_seconds` (`[search] default_max_s` 60, cut to `max_max_s` 180). It runs under the same conversation lease for the duration of the call: the body is never handed to an EXPLORATION lease mid-turn. If the turn is cancelled (the agent's `cancel()`: AI outage, a new wake) or a more important behaviour takes the body, the search stops within one step, the wheels stop and the result says `cancelled` or `preempted`. (She cannot hear "basta" while a tool runs: the ear is not listening during a turn.)

**Result.** `found` (seen *now*, by `scan`, `exploration` or `presence`), `how`, `where` (`room` is the room she was in when she saw it, `direction` left/centre/right, `pan`, `tilt`; the head is left pointing at it), `last_known`, `searched` (the steps she took), `drove`, `autonomous_driving` and `notes`. Not found: `how` is `memory` when something is known, `confirmed: false`, and the notes say where it was last seen and that she cannot navigate there. Times are `when` (absolute) and `ago` (relative), like `recall`.

**No map, so no "go to the kitchen".** There is no reliable localisation yet (PRD, Navigation). A thing last seen in another room is reported honestly ("last seen in the kitchen, but she cannot navigate to a room yet"); exploration is a wander that scans, not a route. The model phrases this in Italian. Matching is by name: case, accents and underscores folded, whole words (`backpack` finds `black backpack`); synonyms are not handled.

**People are counted, never identified.** `find_person` has no name argument ("find Leonardo" finds *a* person). The result carries a count and a note that she cannot tell people apart; no name ever reaches the result.

**`approach(target)`** (an object name, or `person`) uses the latest observation of it, which must be at most 5 minutes old and from the room she is in:

1. if it was seen more than 20 degrees off the nose, one short turning move towards it (steering 30, forward 800 ms), then a picture with the head at the estimated new direction; if it is not seen, or still off the nose, she stops (`lost_sight`, `not_ahead`);
2. then up to `[search] approach_moves` (3) short forward moves (speed 20, 800 ms, clamped by the safety layer), each only while the distance reading ahead is above `approach_stop_cm` (40), each followed by a picture that must still show it (`lost_sight` otherwise). Every move goes through `safety.execute`, so an obstacle or a cliff stops it (`reason: "obstacle"` / `"cliff"`).

The result has `moves`, `turned`, `reached`, `distance_cm` and the `reason` it ended (`close_enough`, `max_moves`, `obstacle`, ...).

**The autonomous-driving switch applies.** Exploring and approaching are Laika's own decisions to drive, even though the user asked for the search, and the conversation lease is not autonomous for the safety controller, so the searcher and `Explorer.run_goal` check `safety.allow_autonomous` themselves, exactly like the investigator's approach. With `[robot] allow_autonomous_driving` off (the robot's default) a search is memory plus one head scan, with a note `autonomous_disabled` so she can say she only looked around from where she is, and `approach` returns `{"moved": false, "reason": "autonomous_disabled"}` without driving. Only the `move` tool, where the user asks for a specific move, drives with the switch off. `--sim` turns the switch on. On the real robot the switch stays off until the V0.2 hardware check; the exploring and approaching paths are tested in the simulator and ready for the owner to verify in a real room once it is on.

**Limits** (`[search]`, validated): `default_max_s` (60), `max_max_s` (180), `approach_moves` (3, 1 to 5), `approach_stop_cm` (40, 20 to 300). `approach_stop_cm` here is separate from `[investigate] approach_stop_cm`.

## Battery

`src/laika/battery.py` (`[battery]`, #50) keeps Laika from being stranded in the middle of the room. `BatteryMonitor` reads `Robot.battery_voltage()` every `sample_s` (30 s) on its own small thread, never on the safety thread, and says OK, LOW, CRITICAL or UNKNOWN. It starts UNKNOWN and takes its first readings at once, so the state is known within milliseconds of starting. UNKNOWN (no reading yet, or three failed reads in a row, or a pack that isn't connected) is treated like LOW for driving on her own: moves under an autonomous lease are refused as `battery_unknown` and exploration is head-only, so a broken reading can never let her wander until she is stranded. She still talks, looks and follows the user's own move requests, and she doesn't ask to be charged.

**The reading.** On the robot it is the Robot HAT's own battery read (`robot_hat.device.get_battery_voltage()`, older versions `robot_hat.utils`): ADC channel A4 through a divider of 3. A failed read returns None, is logged once and never crashes anything. The simulator models a pack that starts at 8.2 V, drains slowly with time and faster while driving, and reads 0.15 V lower while the motors run (`SimulatedRobot.set_battery(volts)` for tests and demos).

**The level.** The median of the last 5 readings, so one noisy reading changes nothing, and no state change before 3 readings (the first ones are taken every 2 s). Voltage sags under load, so a reading taken while the wheels turn is skipped and tried again two seconds later; only after about a minute of continuous driving does a loaded reading count. A reading under 3 V or over 12 V is not a battery (a pack that is not connected reads about 0 V) and is ignored. `percent` is a rough straight line from `critical_v` to 8.4 V: good for talking, not for deciding.

**The states, with hysteresis** (defaults for two 18650 cells in series; calibrate them, see [safety calibration](safety-calibration.md#battery)):

| State | Enters | Leaves |
| --- | --- | --- |
| OK | above `ok_v` (7.3 V) | below `low_v` (7.0 V) |
| LOW | below `low_v` | above `ok_v` |
| CRITICAL | below `critical_v` (6.6 V) | at or above `low_v` (goes to LOW) |

Every change publishes `BatteryLow`, `BatteryCritical` or `BatteryOk` (voltage, rough percent, previous state, lowest voltage since the start) and appears in the field-trial report.

**LOW: she rests.**

- Moves under an autonomous lease are REJECTED `battery_low`; the user's and the conversation's moves still run.
- The arbiter ends the autonomous leases on the transition, like any other halt (the explorer ends with reason `battery_low`). New autonomous requests are still granted: head-only activities (looking around, scans) may continue, and exploration starts head-only (`Explorer.driving` is false).
- At the start of a conversation her context carries an Italian note that her battery is low and that she should say so once, in her own words ("Ho bisogno di essere ricaricata"). The note goes in until a turn of that conversation has answered, and again after the conversation ends.
- `InteractionPolicy` counts a low battery as something worth saying while someone is near, but only within its normal rules: quiet hours, do-not-disturb, busy, cooldown and the hourly limit all still win. A proactive conversation still needs a question to ask, so in practice she tells people when they start talking to her.
- `get_robot_state` and `get_sensors` carry `battery` (voltage, state, percent).

**CRITICAL: she stays still.** Every move is REJECTED `battery_critical`, from autonomous behaviours, from the agent's `move` tool and from gestures that drive; the move in progress is cancelled; exploration and new autonomous requests are refused. `stop` and the emergency stop are never refused. The conversation note says the wheels are still.

**Turning it off.** `[battery] enabled = false` starts no monitor: nothing is reported and nothing is limited. If the voltage cannot be read at all she stays OK (and logs it once), so a broken ADC never strands her; check the log after the hardware check.

## People

Laika knows when someone is near, from a head count only (`presence.py`, `[presence]`). It does not recognise anyone.

**How it works.** The vision model counts the people in each frame (`people` in the observation; the prompt forbids identifying anyone). In the simulator the count is ground truth from the house. `PresenceTracker` listens to `ObservationCreated`:

- a frame with `people > 0` and confidence at least `min_confidence` (0.5) marks "a person is here". The change from absent to present publishes `PersonDetected` once, with `count`, `confidence`, `room`, `pan`, `tilt`, `since` (wall time) and `duration_s` (0 on arrival);
- `PersonLost` (same data, `duration_s` = how long the visit lasted) follows when no such frame arrived for `lost_after_s` (600, which must be longer than `presence_check_s`). It is checked on every observation and by a small timer thread that sleeps until the deadline, so it fires even if no more frames come;
- frames showing nobody, or only low-confidence people, do not end a visit by themselves and do not extend it: one blurry frame never flaps, and a visit ends only after `lost_after_s` without a good sighting. A person who comes back after a longer gap is a new arrival.

The tracker exposes `present`, `count`, `last_seen` and `since` for the interaction policy (#39). The simulator no longer publishes person events itself: this is the only source.

**Idle pictures.** Frames only exist when she takes pictures, so the capture policy has a `presence` trigger: while nobody is known to be near and no conversation holds the body, one picture every `presence_interval_s` (300, every 5 minutes; 0 = off); while someone is near, one every `presence_check_s` (300; 0 = off) to notice they left. They share the `[capture] max_auto_per_minute` cap and the `[camera]` and `[vision]` limits, and any other automatic picture counts as a presence check too. `PersonDetected` also triggers the usual `person` picture (`on_person`).

**Cost and privacy.** With `[vision]` on, every idle picture is a photo of the home sent to the vision model: the default of one every 5 minutes is at most 12 an hour while nobody is around (`--sim` uses ground truth and sends nothing). The trade-off is that she may take up to 5 minutes to notice someone walking in. Lower `presence_interval_s` to notice people sooner, or set it to `0` to stop idle pictures (she then notices people only in pictures taken for other reasons, and `PersonLost` still fires after `lost_after_s`).

**Privacy.** Only a count, the room and where the head pointed are kept in the event and the observation. No faces, names or identities. Pictures are stored only under the normal image retention (`[storage] max_images`, `max_images_mb`); nothing extra is kept for people.

## Interaction policy

`src/laika/interaction_policy.py` (`[interaction]`, #39) decides **whether** Laika may start talking on her own. The model only decides **what** to say. The user can always talk to her; only her initiative is limited. The speaking itself is [proactive questions](#proactive-questions) (#40): that controller calls `InteractionPolicy.decide()` before it generates anything. Nothing here speaks.

**States**, first match wins:

| State | When |
| --- | --- |
| `SLEEPING` | inside `quiet_hours` (local time, `"21:30-07:30"`, may cross midnight; empty = none) |
| `DO_NOT_DISTURB` | the user asked for it and it has not expired |
| `BUSY` | a conversation or other USER/CONVERSATION behaviour holds the body, or the user said they are busy and it has not expired |
| `AWAY` | nobody seen for `away_after_s` (900); counted from startup if nobody was ever seen |
| `AVAILABLE` | a person is near now and has been for at least the settle time (`settle_s`, 5 by default, scaled by interruption tolerance), so someone walking past is not ambushed |
| `UNKNOWN` | anything else (nobody seen yet, just left, just arrived) |

**The rule** (the only one for V0.6). `decide()` returns `Decision(allowed, reason, state)` and allows only when all of these hold; the first that fails is the reason:

1. `enabled` (else `disabled`);
2. the state is `AVAILABLE` (else the state's own name: `sleeping`, `do_not_disturb`, `busy`, `away`, `unknown`);
3. no interaction within the cooldown (`cooldown_s`, 1800 or 30 minutes by default, scaled by initiative: see [Personality](#personality)), else `cooldown`. A chat counts from its end (`InteractionStarted` / `InteractionCompleted`), and so does every proactive attempt the caller reports with `record_attempt()`;
4. something worth saying, else `nothing_to_say`: a pending question, or an interest score at or above the `[curiosity] threshold` within the last `interest_window_s` (600), read from `InterestingChangeDetected`;
5. fewer than the hourly limit (`max_per_hour`, 2 by default, from initiative: see [Personality](#personality)) proactive attempts in the last hour, else `rate_limit`.

`decide()` only reads: it uses no randomness and only the injected clocks, so the same inputs give the same answer. Decisions are logged at debug level, and a refusal is logged once each time the reason changes.

**Do not disturb and busy.** The agent has `set_do_not_disturb(minutes 5 to 720)`, `clear_do_not_disturb()`, `set_busy(minutes)` and `clear_busy()`, so "Laika, non disturbarmi per un'ora" works. Busy is the softer one: Laika just does not start, and the user can still talk to her. A shorter request never shortens a longer one still running. Both are stored in the local store (kv namespace `interaction`, as wall-clock end times), so they survive a restart and expire by themselves.

**Why the model cannot override it.** The tools only make Laika quieter or clear the user's own setting. There is no tool that starts an interaction, records an attempt or touches the cooldown, the threshold or the quiet hours (those are settings in the file). The proactive path asks the policy before it calls the model, and the model never sees a "start a conversation" tool.

## Personality

`[personality]` sets six numbers from 0 to 100 (whole numbers; anything else fails at startup and names the setting). The defaults are the PRD's example: humor 70, sarcasm 45, verbosity 20, initiative 60, curiosity 75, interruption_tolerance 25. Edit `config/laika.toml` and restart: no code changes. At startup she prints them: `[laika] personality: humor 70, sarcasm 45, ...`.

| Parameter | What it changes | Bands: 0-30 / 31-69 / 70-100 |
|---|---|---|
| `humor` | prompts (tool mode, JSON mode, proactive question) | no jokes / occasional dry humour / frequent dry humour |
| `sarcasm` | prompts | never sarcastic / a light ironic touch / ironic, TARS-style. Always gentle and never mean, especially with children, whatever the value |
| `verbosity` | prompts | one or two short sentences / up to three / up to five. Always speech-friendly: no lists, no markdown |
| `curiosity` | prompts (how often she asks a follow-up question), the curiosity threshold and the investigator's hourly limit | no follow-ups unless needed / now and then / often |
| `initiative` | the policy: cooldown and hourly limit | see the bounds below |
| `interruption_tolerance` | the policy: how long someone must be near before she speaks | see the bounds below |

The core character does not change with the numbers: concise, direct, curious, Italian, kid-friendly, "if you do not know something, say so".

**Bounds.** Every mapping is deterministic and stays inside these limits, at 0 and at 100 too:

- **Cooldown** (`cooldown_s`, 1800): multiplied by 2.0 at initiative 0, 1.0 at 50 and 0.5 at 100, and never below ten minutes (or `cooldown_s` itself when that is shorter). Default: 1620 s.
- **Hourly limit** (`max_per_hour`): from initiative, 1 (0 to 24), 2 (25 to 49), 3 (50 to 74) or 4 (75 to 100), and never above `[interaction] max_per_hour`, which is the ceiling you set (2 by default). Default: 2.
- **Settle time** (`settle_s`, 5): multiplied by 2.0 at tolerance 0, 1.0 at 50 and 0.5 at 100. Default: 7.5 s.
- **Presence floor**: below an interruption tolerance of 20 a person must have been near for at least 30 s, whatever the settle time.
- **Curiosity threshold**: `threshold * (1.3 - 0.6 * curiosity / 100)`, clamped to 0.2 to 0.6. Default: 0.35 becomes about 0.30 (see [curiosity](curiosity.md#threshold-and-cooldown)).
- **Investigator hourly limit** (`[investigate] max_per_hour`, 12): `round(12 * (0.5 + curiosity / 100))`, so 0.5x at curiosity 0 to 1.5x at 100, and at least 1. Default: 15.

**Never affected.** Do-not-disturb, busy, quiet hours and the safety layer (the motors, the watchdog, the autonomous-driving switch) read only the user's own settings and the sensors. No personality value reaches them, so a do-not-disturb with initiative 100 still refuses, and quiet hours still apply at tolerance 100.

## Proactive questions

`src/laika/proactive.py` and `Conversation.converse_proactive` (`[interaction] proactive`, `recheck_s`, #40) close the core loop: she notices something new, looks into it, and later, when someone is near and the policy agrees, she asks about it and keeps the answer.

```text
new object -> InterestingChangeDetected -> Investigator -> pending question (Italian template)
person near -> PersonDetected -> InteractionPolicy.decide() -> allowed
            -> turn the head to the person -> ask -> the answer -> known object + "answer" episode
```

**The trigger.** `ProactiveInteraction` is the only thing that can start one. It checks on `PersonDetected` (again right after `settle_s`, because the policy waits for the person to settle), on `InterestingChangeDetected` while someone is near, and every `recheck_s` (15) while someone is near, since people often arrive before there is anything to say. Every check calls `decide()` with the real number of pending questions and whether the body is held by a conversation. When it allows, the attempt is recorded **first** (cooldown and hourly limit count it even if a crash follows), then a `ProactiveRequest` is handed to the `Conversation`. A trigger with nothing to ask (an interest score but no queued question) starts nothing and records nothing. The question asked is always the oldest pending one, which is also the one the agent and the memory selector look at first.

**The hand-off.** The trigger thread never speaks or moves. `Conversation.request_proactive()` stores the request and calls `ear.interrupt()`, so the sleeping `wait_for_wake` returns the request and the main thread runs it. A wake word heard at the same moment wins, a request older than 30 s is dropped, and none is accepted while a conversation is going. `VoskEar.interrupt()` stops the recogniser, and `ConsoleEar` reads stdin on a small thread so `listen(timeout)` can time out and `interrupt()` can wake it.

**The conversation.** She takes a `SOCIAL` lease (with its heartbeat); if something more important holds the body she gives up quietly and says nothing. Then she:

1. turns the head to where the person was seen (pan and tilt of the `PersonDetected` frame; head only, the wheels never move);
2. emits `InteractionStarted` with `{"initiator": "laika", "reason": "pending_question", "questionId": ...}` (a chat the user starts emits `initiator: "user"`);
3. asks the model for the opening line with a short instruction that holds the question text. That turn has a **restricted registry**: only `look`, `look_left/right/up/down` and `gesture`. No `move`, no exploring, no `investigate`, and none of `set_do_not_disturb`, `clear_do_not_disturb`, `set_busy`, `clear_busy`: a turn Laika starts herself can never loosen the policy. If the AI is down or the model says nothing, she speaks the question's own Italian template instead;
4. listens for `[voice] follow_up_s`. If the person answers, the lease becomes a normal `CONVERSATION` lease and it is an ordinary chat: the answer goes to the agent with all its tools and to the `MemorySelector`, which stores it (`record_answer`: the object becomes a known object with the answer as its note, plus a fact and an `answer` episode) and removes the question;
5. emits `InteractionCompleted` with `{"initiator": "laika", "answered": true|false, "questionId": ...}`.

**What she never does.** She does not nag: if nobody answers within `follow_up_s` she says nothing more and goes back to sleep, the question stays pending, and the cooldown (the attempt counts) keeps her quiet for `cooldown_s` (30 minutes by default). She does not start during do-not-disturb, busy or quiet hours, while someone is only passing (`settle_s`), while a conversation is going, more than `max_per_hour` times an hour, or without a pending question. She never drives to a person and never says anything the model was not asked to word, except the question template.

**Turning it off.** `[interaction] proactive = false` stops the trigger altogether (the policy and the do-not-disturb tools stay); `[interaction] enabled = false` makes the policy refuse everything. The user can say "non disturbarmi per un'ora" at any time.

**Seam for the personality (#41).** The wording of the opening line is `prompts.proactive_instruction()` (verbosity and humour would go there); initiative and interruption tolerance belong to the policy's settings. Neither is parametrised yet.
