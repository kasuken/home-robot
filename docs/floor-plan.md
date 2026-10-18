# The plan of the house

`src/laika/floorplan.py`. Laika has no odometry ([navigation options](navigation.md)), so a metric plan would have nothing to attach to: she cannot say where she is on it. What a plan *can* give her is the shape of the house, a **graph of rooms**: which rooms there are, what people call them and which doors join them. It does not drive her anywhere.

## What it does

- **Names.** Every name of a room is taught to the world model (`seed_world`), so "la cucina", "la camera dei bambini" and the room ids the vision model uses are one room, and the vision prompt lists the rooms with the household's names.
- **Plausible moves.** A room that is not next door to the current one is a jump: the room tracker ([room recognition](behaviours.md#room-recognition)) then asks for one more vote and 0.4 more margin (`JUMP_EXTRA_MARGIN`). Two frames of a lookalike room on the other side of the flat no longer move her there.
- **Questions.** The plan (rooms, sizes, what they join, the home spot) is in the model's prompt, and the `house_route` tool answers "dov'è il bagno?" / "come arrivo in cucina?" with the rooms to cross. It only talks: nothing moves.
- **Rules.** A room marked `no_go` (a balcony) is added to the house rules, and routes never pass through it.
- **Home spot.** `home_room` is where she rests; it is told to the model, nothing more yet.

## The file

`config/house-plan.toml` next to `laika.toml` (`[house] plan`; empty = never read one). It describes a home and the repository is public, so git ignores it: **copy it to the robot by hand** (`scp config/house-plan.toml ema@<robot>:/home/ema/home-robot/config/`). [config/house-plan.example.toml](../config/house-plan.example.toml) shows the format with an invented flat. A missing file means no plan; a broken one is reported at start (`house plan ignored: ...`) and ignored.

```toml
[plan]
name = "..."          # one line, English
home_room = "office"  # where she rests

[[room]]
id = "kitchen"                 # English, lowercase; the vision model uses it
names = ["cucina"]             # up to 5, Italian; the first is how she says it
area_m2 = 9.04                 # optional
about = "..."                  # English, for the prompt
links = ["office"]             # doors and openings; one direction is enough
no_go = false                  # true: she never drives in
```

An open space (living, dining, hall) is one room; a corner people call something else (a home office with no door) can be its own room joined to it, which also matches what the vision model says it sees.

## What it does not do

No coordinates, no position, no path planning: `plausible_move` and `route` work on rooms. The position fixes are printed AprilTags at the doorways ([AprilTag position fixes](apriltags.md)): the plan lists them as `[[tag]]` (`id`, `marks`, `seen_from`) and says what each one means. Driving to a room is not built. The simulator still uses its own invented house.

## Not verified on the robot

Whether the extra margin for jumps helps or hurts in the field trial (it can only slow a real move down, by a frame or two); the plan's doorways were read from a drawing, so check `links`.
