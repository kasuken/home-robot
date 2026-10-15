# Safety calibration

Laika's safety thresholds are set in the `[robot]` section of `config/laika.toml` (see `config/laika.example.toml`). This page says what each one does, the range Laika accepts, and how to calibrate them on the robot with the guided hardware check (issue #17). The code is in `src/laika/safety.py` (the controller), `src/laika/hardware.py` (the cliff reference) and `src/laika/config.py` (the ranges).

Calibration is the owner's step on the robot. The defaults below are the prototype values; the values recorded for this robot are in [Recorded values](#recorded-values).

## What each setting does

| Setting | Default | Allowed | What it does |
| --- | --- | --- | --- |
| `allow_autonomous_driving` | false | true or false | Lets Laika drive on her own (exploration, and later any behaviour nobody asked for). Leave it false until this hardware check, the service check and the first-drive check (#15, #17, #18) have passed on the robot. While false the safety controller refuses every autonomous move (`autonomous_disabled`); moves you ask for still work. `--sim` turns it on for the simulator only. |
| `drive_speed` | 25 | 1 to 40 (whole number) | Wheel speed of every move. The safety controller also caps it at 40. |
| `max_drive_s` | 1.5 | 0.1 to 3 | Longest forward move. A single move command is also capped at 1.5 s by the command format, so a larger value has no effect today. |
| `backward_max_s` | 0.6 | 0.1 to 1 | Longest backward move. There is no rear sensor, so backward moves are kept short. |
| `min_front_distance_cm` | 25 | 10 to 100 | A forward move is refused (`obstacle`) when the ultrasonic reads closer than this, and stopped if an obstacle comes closer while it runs. |
| `ultrasonic_invalid_limit` | 5 | 0 (off) or 1 to 50 | A forward move is stopped (`ultrasonic_invalid`) after this many invalid ultrasonic readings in a row. At 20 Hz, 5 readings is 0.25 s. An invalid reading (`-1`, `-2`, `0` or a timeout) is not treated as clear. Backward moves are not affected. |
| `command_max_age_s` | 1.0 | 0.2 to 5 | A move command older than this when it arrives is expired and never starts. |
| `safety_cadence_hz` | 20 | 10 to 50 | How often the safety monitor reads the sensors while a move runs. 20 Hz is a 50 ms tick, so the stop comes within about one tick of a hazard. |
| `watchdog_timeout_s` | 0.5 | 0 (off) or 0.2 to 2 | If the safety monitor stalls this long, a separate watchdog stops the motors. Off is not recommended. See [ADR 0002](decisions/0002-watchdog-strategy.md). |
| `cliff_reference` | not set | three values, each 0 to 4095 | Grayscale values at or below these read as a cliff, one per sensor. Not set: the value from the vendor file `/opt/picar-x/picar-x.conf` is used (500, 500, 500 by default). |

Laika checks these at startup. A value outside its range stops her with a message that names the setting, for example `[robot] drive_speed must be 1 to 40, got 90`, and nothing moves.

Settings that are not in `[robot]`:

- **Emergency stop**: no setting. A stop cancels the move and refuses every new move until it is reset. It is part of the hardware check (the "Stop and emergency stop" step).
- **Cliff and obstacle handling**: no setting beyond `cliff_reference` and `min_front_distance_cm`. A cliff stops a move and refuses a new forward one. The cliff sensors are at the front, so a backward move may start with the edge in view (the only way away from it); it is ignored until the sensors read clear once, after which a cliff stops it again.

## Before you start

- Stop the service if it is running, because the check needs the robot: `sudo systemctl stop laika`.
- Have about 1 m of free floor, a box (for the obstacle step), and be ready to lift the robot in your hands (for the cliff step). The check asks before each physical step and answers `s` skips it.
- Note the current `[robot]` values, so you can compare them with the report.

## Procedure on the robot

1. **Run the check** from the clone:

   ```bash
   sudo python3 run-laika.py --check-hardware
   ```

   Answer `y`, `n` or `s` to each question. Add a note after the answer if something looked odd (`n the box was on the left`).

2. **Ultrasonic** (step 3). The check prints the readings with the front clear and with a hand about 15 cm away, and the invalid-reading rate for each. Look at:
   - `invalid readings: X% with the front clear`. A high rate means the sensor often gets no echo here (soft surfaces, a bad position, a loose cable). A forward move stops after `ultrasonic_invalid_limit` invalid readings in a row, so a high rate makes Laika stop more often, not drive blind.
   - The median with a hand in front, which must be below the clear median and below 30 cm.

3. **Obstacle** (step 6). The box is placed about 20 cm in front. The check prints `refused at N cm (limit M cm)`: N is the distance the safety controller saw when it refused the forward move, M is `min_front_distance_cm`. Compare N with the distance you measured with a ruler. If they agree within a few centimetres, keep the limit. Choose a new `min_front_distance_cm` (10 to 100) only if the readings are consistently off or the stop comes too late or too early for the room.

4. **Grayscale and cliff** (step 4). The check reads the floor, then asks you to lift the robot into the air with your hands so the sensors see nothing below, and reads again. Never hold it over a real table edge: the step then asks for a forward move that must be refused, and if the sensor missed the edge the wheels would turn. It prints:
   - `floor`: the three grayscale values on the floor, and whether the floor reads as a cliff (it must not).
   - `lifted`: the three values over the edge, and whether `cliff_detected` became true (it must).
   - `suggested cliff_reference [a, b, c]`: halfway between each floor and edge value, so each sensor has the same margin on both sides. It is also written to the report under `suggestedSettings`, with the exact line to paste.
   - `forward move with the edge in view: rejected (cliff)`: a forward move is refused while the edge is in view. A refused move never reaches the motors, so the wheels do not turn.

   If the check says `no cliff reference`, the floor does not read at least 100 counts above the edge on every sensor. Clean the sensors, check the surface (a dark mat or a shiny floor both make this harder), and run the check again. Do not paste a value in that case.

5. **Paste the suggestion.** Open `config/laika.toml`, find `[robot]`, and add the line the check printed, for example:

   ```toml
   cliff_reference = [525, 525, 525]
   ```

   When this line is set, Laika writes the value into the vendor file `/opt/picar-x/picar-x.conf` at each start, so the two agree. Removing the line later does not restore the old value in that file; to go back, change `cliff_reference` in the file by hand, or keep the line.

6. **Speed and durations** (step 5). The drive step runs 0.5 s forward and 0.5 s backward at `drive_speed`. Measure how far the robot goes; if that is more than you want for a 1.5 s move, lower `drive_speed` or `max_drive_s`. Raising them needs the same measurement.

7. **Emergency stop** (step 7). A 1.5 s forward move starts, and after 0.3 s Laika calls the emergency stop. The report must show `cancelled (emergency_stop)` and motor speed 0, and the robot must stop at once. Answer `n` if it kept rolling.

8. **Verify.** Run the check again with the new values. The report must show `PASS` for Ultrasonic, Grayscale and cliff, Obstacle refusal and Stop and emergency stop, and the Grayscale step must now show the edge refusal. Then record the values in the table below and attach the report (`hardware-check-<timestamp>.md` and `.json` in the data directory's `logs/` folder) to issue #17.

Restart the service after changing the settings: `sudo systemctl restart laika`, or start Laika again by hand.

## Reading the report

The Markdown report has these parts:

- **Summary**: the result, and the steps as a table.
- **Issues to fix**: the operator's `n` answers that failed a step.
- **Suggested settings**: the lines to paste under `[robot]`. The JSON carries the same values under `suggestedSettings`.
- **Limits used**: the values the check ran with, including `ultrasonic_invalid_limit`, `cliff_reference` (or `vendor value from picar-x.conf`) and `min_front_distance_cm`.
- **Details**: each step's measurements. Ultrasonic has `clear`, `hand` and `invalid_rate`. Grayscale has `floor`, `lifted`, `suggested_cliff_reference` and `move_over_edge`. Obstacle has `refused_at_cm` and `limit_cm`.

## Recorded values

Fill these in after the calibration on the robot. `TBC` means not yet recorded.

| Setting | Default | Value on the robot | Date | Report file |
| --- | --- | --- | --- | --- |
| `drive_speed` | 25 | TBC | TBC | TBC |
| `max_drive_s` | 1.5 | TBC | TBC | TBC |
| `backward_max_s` | 0.6 | TBC | TBC | TBC |
| `min_front_distance_cm` | 25 | TBC | TBC | TBC |
| `ultrasonic_invalid_limit` | 5 | TBC | TBC | TBC |
| `command_max_age_s` | 1.0 | TBC | TBC | TBC |
| `safety_cadence_hz` | 20 | TBC | TBC | TBC |
| `watchdog_timeout_s` | 0.5 | TBC | TBC | TBC |
| `cliff_reference` | vendor file (500, 500, 500) | TBC | TBC | TBC |

| Verification | Expected | Result on the robot | Date |
| --- | --- | --- | --- |
| Invalid ultrasonic rate, front clear | low, see step 2 | TBC | TBC |
| Obstacle: forward move refused at | close to the measured distance, below the limit | TBC cm | TBC |
| Cliff: floor reads as floor, edge reads as cliff | yes, yes | TBC | TBC |
| Cliff: forward move refused over the edge | `rejected (cliff)` | TBC | TBC |
| Emergency stop: move cancelled, motors stopped | `cancelled (emergency_stop)`, speed 0 | TBC | TBC |
