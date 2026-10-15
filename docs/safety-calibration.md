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
| `max_autonomous_drive_s_per_hour`, `max_autonomous_drive_s_per_day` | 120, 900 | 0 (no limit) or 10 to 3600 / 86400 | The most she drives by herself in an hour and in a day: see [Driving budget](#driving-budget). |
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

## Battery

The `[battery]` thresholds are for two 18650 cells in series (8.4 V full, 7.4 V nominal) and are **defaults, not measurements**. Calibrate them on the robot, because the divider on the Robot HAT is not exact and a pack sags when the motors pull current.

| Setting | Default | Range | What it does |
| --- | --- | --- | --- |
| `enabled` | true | true or false | false: no battery monitor at all |
| `sample_s` | 30 | 5 to 3600 | Seconds between readings; readings taken while the wheels turn are skipped for up to a minute |
| `low_v` | 7.0 | 5 to 9 | Below this she stops driving around by herself (`battery_low`) and asks to be charged |
| `ok_v` | 7.3 | 5 to 9 | She is OK again only above this, so the level does not flap |
| `critical_v` | 6.6 | 5 to 9 | Below this every wheel move is refused (`battery_critical`) and she stays still |
| `voltage_offset` | 0.0 | -2 to 2 | Volts added to every reading so it matches the multimeter |

The rule is `critical_v < low_v < ok_v`; the settings are refused at startup otherwise. The reading comes from `robot_hat.device.get_battery_voltage()` (ADC channel A4, times 3); verify it as follows.

**Multimeter procedure (V0.2 hardware check, the "Battery voltage" step)**

1. Charge the pack, switch the robot on and leave the wheels still: a reading under load is lower.
2. Run `python3 run-laika.py --check-hardware`. The "Battery voltage" step prints the reading and asks for the multimeter value.
3. Put the multimeter on the battery connector (red on +, black on -) and type the volts it shows. The step records both values and the difference.
4. A difference under 0.5 V is an offset: paste the suggested `voltage_offset = ...` under `[battery]`. A bigger one is a wiring or divider problem and fails the step.
5. Run the check again: the reading with the offset must match the multimeter within a few hundredths of a volt.
6. Calibrate the thresholds for your cells: run Laika (exploring, with `allow_autonomous_driving` on) and note the voltage at which she gets sluggish or the Raspberry Pi browns out under load. Set `critical_v` a few tenths of a volt above that, `low_v` about 0.4 V above `critical_v` (enough to get back to the charger), and `ok_v` about 0.3 V above `low_v`.
7. With a bench supply or a nearly empty pack, check the behaviour: below `low_v` the log says `battery low`, exploration stops driving and a move from the explorer is rejected `battery_low`; below `critical_v` the `move` tool answers `battery_critical`; `stop` always works.

Record the final values in the table below.

## Driving budget

Even once she may drive on her own, there is a cap on how much (#51). Only autonomous driving counts (exploration, investigations that move closer, searches she wanders on); a move you ask for does not, and is never refused for it. The time is the real run time of each move.

| Setting | Default | Allowed | What it does |
| --- | --- | --- | --- |
| `max_autonomous_drive_s_per_hour` | 120 | 0 (no limit) or 10 to 3600 | Seconds of autonomous driving in any sliding hour. Used up: autonomous moves are REJECTED `drive_budget` until the hour has room. |
| `max_autonomous_drive_s_per_day` | 900 | 0 (no limit) or 10 to 86400 | Seconds of autonomous driving in the local day (midnight to midnight). Kept in the store, so a restart does not reset it; the hour window is in memory and starts empty after a restart. |

The hour must not be more than the day. At the exploration speed a step of 0.8 s moves her a few centimetres, so 900 s is a quarter of an hour of driving a day: set the numbers from what the field trial shows (the report lists `drive_budget` refusals and the stored day total under "Autonomy limits"), not from this table. Exploration that has used up the budget does not stop: it looks around with the head, like with a low battery.

## Unattended checklist

Letting her drive around while nobody watches is the owner's decision, so Laika does not decide it for you: she tells you. `python3 run-laika.py --unattended-check` is read-only (it does not start the robot), prints the list and exits 1 while a required item is not met. At startup, with `allow_autonomous_driving = true` and the list not met, one warning line is printed; it never blocks.

| Item | Met when | How to meet it |
| --- | --- | --- |
| Hardware check passed on the robot | the newest `hardware-check-*.json` in `<data_dir>/logs/` has `mode: robot` and `result: pass` (a simulator report does not count) | `python3 run-laika.py --check-hardware` on the robot (#15, #17) |
| Battery calibrated | `[battery] voltage_offset` is set, or a passed report records a suggested offset, and `[battery] enabled` is true | the multimeter step of the hardware check, then [Battery](#battery) |
| Service installed and enabled | Linux only, best effort: `/etc/systemd/system/laika.service` exists and is enabled; skipped elsewhere | `sudo deploy/install-service.sh` ([service](service.md), #18) |
| `allow_autonomous_driving` turned on deliberately | `[robot] allow_autonomous_driving = true` | set it last, after the items above |
| Watchdog enabled | `[robot] watchdog_timeout_s` is above 0 | [ADR 0002](decisions/0002-watchdog-strategy.md) |
| Driving budgets set | both `max_autonomous_drive_s_per_*` are above 0 | [Driving budget](#driving-budget) |
| House rules | information only: how many no-go rooms are set or taught | [house rules](behaviours.md#house-rules-driving-budget-and-pause) |

The check can only read files. It cannot tell that the cliff and obstacle values you recorded below are right, or that the service really stops the motors on your robot: those are what the hardware check steps and your own test exercise. Keep the pause in mind for the first days: "Laika, resta ferma" stops everything she does on her own, survives a restart, and only "puoi muoverti di nuovo" lifts it.

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
| `[battery] voltage_offset` | 0.0 | TBC | TBC | TBC |
| `[battery] low_v` / `ok_v` / `critical_v` | 7.0 / 7.3 / 6.6 | TBC | TBC | TBC |

| Verification | Expected | Result on the robot | Date |
| --- | --- | --- | --- |
| Invalid ultrasonic rate, front clear | low, see step 2 | TBC | TBC |
| Obstacle: forward move refused at | close to the measured distance, below the limit | TBC cm | TBC |
| Cliff: floor reads as floor, edge reads as cliff | yes, yes | TBC | TBC |
| Cliff: forward move refused over the edge | `rejected (cliff)` | TBC | TBC |
| Emergency stop: move cancelled, motors stopped | `cancelled (emergency_stop)`, speed 0 | TBC | TBC |
| Battery reading against a multimeter | within a few hundredths of a volt after the offset | TBC | TBC |
