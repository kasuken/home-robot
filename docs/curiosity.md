# Curiosity: interestingness scores

Every observation gets an interestingness score (PRD, Curiosity Engine). The code is `src/laika/curiosity.py`; the settings are `[curiosity]` in `config/laika.toml`.

The score says **how interesting** an observation is. It does not decide what Laika does. When the total reaches the threshold, Laika publishes `InterestingChangeDetected`; the interaction policy (#29) decides whether to look, take a photo, move, ask or say something. Automatic photos already react to that event through the capture policy's `on_interesting` switch.

## Where it runs

The `Observer` scores each observation on its worker thread **before** it publishes `ObservationCreated`. The world model learns from that event afterwards, so the score compares the frame with what Laika knew before it. Scoring after the update would make everything look known.

1. The observation is validated (as before).
2. The scorer compares it with the world model and memory, reading only. The "previous" observation is the newest logged one with the same location and a camera pan within 20 degrees.
3. `ObservationCreated` carries the score as `interest`: `{"total", "components", "reasons"}`.
4. The score is kept with the observation in the bounded `ObservationLog` (`score_of(id)`), so #29 can read it.
5. If the total is at or above the threshold and some reason is not in cooldown, `InterestingChangeDetected` is published with `observation_id`, `location`, `total`, `reasons` and `objects`.

## Formula

Nine inputs, each between 0 and 1:

    total = sum(weight * component) / sum(weight)

It is a weighted average, so the total stays in 0 to 1 whatever scale the weights use. Components are rounded to three decimals only when they are written out.

| Input | Weight | Component (0 to 1) |
|---|---|---|
| `novelty` | 0.30 | Share of the observation's objects the world model has never seen. 0 with no objects. |
| `environmental_change` | 0.20 | Share of objects last known in a room other than the observation's room (moved). 0 without a location. |
| `uncertainty` | 0.05 | 1 minus the observation's confidence. |
| `person` | 0.15 | 1 if any person is in view, else 0. |
| `unknown_object` | 0.10 | Share of objects with confidence below `unknown_object_confidence`, or whose name is "unknown", "object" or "thing". |
| `unusual_location` | 0.10 | Share of objects seen in a room that is not their usual room (below). |
| `movement` | 0.05 | Compared with the previous look at the same view: objects that appeared or vanished, and a change in the number of people, as a share of all the objects and people in either look. |
| `sound` | 0.00 | Always 0. There is no microphone-level sensing yet; the weight keeps the formula complete. |
| `user_relevance` | 0.05 | Share of the observation's objects (plus its room) mentioned in a remembered fact, or with a note from the user in the world model. |

Heuristics:

- **Usual room.** An object's rooms are its current room and its `previous_rooms` (the last few). The usual room is the one it appears in most often; ties go to the most recent. A room counts as a habit only after two sightings, so one move is not "unusual".
- **Moved.** The world model keeps only the last known room. "Known in this room before and now in another" and "last known room differs from this observation's room" are the same check, seen from the object.
- **Remembered facts.** Matched by substring, as `recall` does. A very short name matches many facts.
- **Reasons** are short English strings such as `new object: backpack`, `moved: keys from kitchen to office`, `person nearby`, `unusual location: mug in office (usually kitchen)`. They are sorted, so the same inputs give the same list.

The scorer is deterministic: no randomness, and no clock. The timestamps come from the observation.

## Defaults

| Setting | Default | Meaning |
|---|---|---|
| weights | the table above | Relative importance. Each is 0 to 10. At least one must be above 0. |
| `threshold` | 0.35 | Announce at or above this total (0 to 1). |
| `unknown_object_confidence` | 0.5 | Confidence below this makes an object "unknown". |
| `cooldown_s` | 120 | A reason is announced at most once per this many seconds (0 to 86400; 0 turns the cooldown off). |

With the defaults, a new object alone (0.30) stays below the threshold. A new object with a person (0.45) passes it, and so does a moved object with a person (0.35).

## Threshold and cooldown

`InterestGate` publishes the event when the total is at or above the threshold and at least one reason is not still cooling down. The fresh reasons start their cooldown; a reason still cooling down does not restart it. So a person who stays in view produces one event every `cooldown_s`, not one per frame. Reasons that stop being true are forgotten when their cooldown ends.

## Simulator

In `--sim` the first scan of each room is very novel: every object is new, so `novelty` is 1.0 there. That is expected, and it is why a first scan alone does not reach the threshold. Later scans of the same room are not novel.

## Not done yet

- Long-term patterns (the PRD's "long-term pattern" input) are not scored.
- The interaction policy that acts on the score is #29.
- The world model is updated by the event bus thread, so a frame scored right after the previous one may not see that frame yet. On the robot, frames are seconds apart, so this does not matter in practice; the tests drain the bus between frames.
