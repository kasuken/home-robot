# Curiosity: interestingness scores

Every observation gets an interestingness score (PRD, Curiosity Engine). The code is `src/laika/curiosity.py`; the settings are `[curiosity]` in `config/laika.toml`.

The score says **how interesting** an observation is. It does not decide what Laika does. When the total reaches the threshold, Laika publishes `InterestingChangeDetected`; the investigator ([#29](behaviours.md#investigation)) decides whether to look, take another photo, move closer, queue a question or remember it; it never speaks on its own. Automatic photos already react to that event through the capture policy's `on_interesting` switch.

## Where it runs

The `Observer` scores each observation on its worker thread **before** it publishes `ObservationCreated`. The world model learns from that event afterwards, so the score compares the frame with what Laika knew before it. Scoring after the update would make everything look known.

1. The observation is validated (as before).
2. The change detector (below) and then the scorer compare it with the world model and memory, reading only. The "previous" observation is the newest logged one with the same location and a camera pan within 20 degrees.
3. `ObservationCreated` carries the score as `interest`: `{"total", "components", "reasons"}`, and the changes as `changes` (see below).
4. The score is kept with the observation in the bounded `ObservationLog` (`score_of(id)`), so the investigator can read it.
5. If the total is at or above the threshold and some reason is not in cooldown, `InterestingChangeDetected` is published with `observation_id`, `location`, `total`, `reasons` and `objects`.

## Formula

Nine inputs, each between 0 and 1:

    total = sum(weight * component) / sum(weight)

It is a weighted average, so the total stays in 0 to 1 whatever scale the weights use. Components are rounded to three decimals only when they are written out.

| Input | Weight | Component (0 to 1) |
|---|---|---|
| `novelty` | 0.15 | Share of the observation's objects the world model has never seen. 0 with no objects. |
| `environmental_change` | 0.40 | 1 if anything changed in a familiar room: an object last known in another room (moved here), or a never-seen object in a room Laika has known for `familiar_room_after_s` (it arrived). One change is enough; it is not diluted by unchanged objects. 0 without a location. |
| `uncertainty` | 0.03 | 1 minus the observation's confidence. |
| `person` | 0.15 | 1 if any person is in view, else 0. |
| `unknown_object` | 0.10 | Share of objects with confidence below `unknown_object_confidence`, or whose name is "unknown", "object" or "thing"; an object the user explained (it has a note in the world model) never counts (#35). |
| `unusual_location` | 0.10 | Share of objects seen in a room that is not their usual room (below). |
| `movement` | 0.04 | Compared with the previous look at the same view: objects that appeared or vanished, and a change in the number of people, as a share of all the objects and people in either look. |
| `sound` | 0.00 | Always 0. There is no microphone-level sensing yet; the weight keeps the formula complete. |
| `user_relevance` | 0.03 | Share of the observation's objects (plus its room) mentioned in a remembered fact (its subject or its words, ignoring case and accents), or with a note from the user in the world model. |

Heuristics:

- **Usual room.** (`WorldModel.usual_room`.) Each object counts its sightings per room (`room_sightings`, the 8 most relevant rooms). The usual room is the one it was seen in most often; ties go to the most recent. A room counts as a habit only after two sightings, so the sofa seen three times in the living room usually stays there, and keys seen five times in the kitchen and once in the office are unusual in the office. Objects saved before these counts existed fall back to their current room and `previous_rooms`.
- **Familiar room.** (`WorldModel.is_familiar`.) A room is familiar once some object in it was first seen at least `familiar_room_after_s` (300 s) before this observation, measured on the observation's own timestamp. Until then a never-seen object there is just part of discovering the room.
- **Moved.** The world model keeps only the last known room. "Known in this room before and now in another" and "last known room differs from this observation's room" are the same check, seen from the object.
- **Remembered facts.** Matched by the fact's subject or a substring of its text, ignoring case and accents, as `recall` does. A very short name matches many facts.
- **Reasons** are short English strings such as `new object: backpack`, `moved: keys from kitchen to office`, `person nearby`, `unusual location: mug in office (usually kitchen)`. They are sorted, so the same inputs give the same list.

The scorer is deterministic: no randomness, and no clock. The timestamps come from the observation.

## Environment changes

The code is `src/laika/changes.py` (#36). `ChangeDetector.detect` answers "is this normally here? what changed? is this new?" for one observation. It runs in the `Observer` just before the scorer, so it too sees the world as it was before the frame; it only reads. The scorer then **uses the detector's changes** instead of repeating the checks: `environmental_change` is 1 when there is a "moved" or "new" change, and `unusual_location` is the share of objects with an "unusual" change. The scores are the same as before the detector existed.

| Kind | When | Example text |
|---|---|---|
| `moved` | The object was last known in another room (`previous_location`). | `backpack moved from living room to kitchen` |
| `new` | Never seen before, in a familiar room. | `lamp appeared in office` |
| `missing` | In the previous look at the same view (same location, pan within 20 degrees), absent now. Only judged when the current observation has confidence of at least 0.6, so a blurry frame does not make things vanish. | `mug is no longer in kitchen` |
| `unusual` | Seen in a room that is not its usual room (`previous_location` is the usual one). | `mug in kitchen, usually living room` |

A change has `kind`, `object`, `location` (where the object is now; for `missing`, where it was), `previous_location`, `timestamp`, `observation_id` and `text`. `ObservationCreated` carries all of an observation's changes as `changes`. `missing` is not a scoring input (the `movement` component already counts what vanished).

`ChangeTracker` keeps the recent changes (at most 100) and records `moved`, `new` and `missing` ones as episodes whose summary is the change text; `unusual` is a score input, not an event. The same change for the same object is recorded once until its state changes: the world model learns from the bus thread, so two quick looks can both see the same move, and a missing object that comes back can go missing again.

Questions, as tools:

- `what_changed` (`since` "today" or "yesterday", meaning local midnight, or `since_hours` 1 to 168, default 24) lists the change episodes since then, newest first, at most 20, each with an absolute local time and an `ago` hint. It reads the store as well as the in-memory window, so it works after a restart.
- `recall` adds `first_seen` (have I seen this before? is it new?), `times_seen`, `usual_room` and `is_usual_here` (true or false; null until a room is a habit).

## Defaults

| Setting | Default | Meaning |
|---|---|---|
| weights | the table above | Relative importance. Each is 0 to 10. At least one must be above 0. |
| `threshold` | 0.35 | Announce at or above this total (0 to 1). |
| `unknown_object_confidence` | 0.5 | Confidence below this makes an object "unknown". |
| `cooldown_s` | 120 | A reason is announced at most once per this many seconds (0 to 86400; 0 turns the cooldown off). |
| `familiar_room_after_s` | 300 | How long Laika must have known something in a room before a new object there counts as a change (0 to 86400). |

With the defaults, a change in a familiar room passes the threshold on its own: something new on the sofa, or the keys in the wrong room, scores at least 0.40. Discovering a new room does not: novelty alone is at most 0.15, a person alone 0.15, so a first look around a house is not "interesting" everywhere.

## Threshold and cooldown

The `threshold` above is the base. `[personality] curiosity` shifts it at startup to `threshold * (1.3 - 0.6 * curiosity / 100)`, clamped to 0.2 to 0.6, and the same shifted value reaches the interaction policy and the investigator (docs/behaviours.md, [Personality](behaviours.md#personality)).

`InterestGate` publishes the event when the total is at or above the threshold and at least one reason is not still cooling down. The fresh reasons start their cooldown; a reason still cooling down does not restart it. So a person who stays in view produces one event every `cooldown_s`, not one per frame. Reasons that stop being true are forgotten when their cooldown ends.

## Simulator

In `--sim` the first scan of each room is very novel: every object is new, so `novelty` is 1.0 there, but the room is not familiar yet, so there is no change and the total (0.15) stays below the threshold. To see an investigation in the simulator, let Laika know a room for more than `familiar_room_after_s` and then add something new to it (`house.add_thing`), or lower `familiar_room_after_s`.

## Not done yet

- Long-term patterns (the PRD's "long-term pattern" input) are not scored.
- Deciding when to interrupt someone with what was found is the interaction policy (V0.6); the investigator (#29) only queues questions.
- The world model is updated by the event bus thread, so a frame scored right after the previous one may not see that frame yet. On the robot, frames are seconds apart, so this does not matter in practice; the tests drain the bus between frames.
