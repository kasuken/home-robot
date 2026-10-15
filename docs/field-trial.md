# Field trial: validating the core loop in the house

Issue #44 asks one question: does the core product loop of the [PRD](prd.md#things-not-to-build-yet) complete, repeatedly, in the real house, without anyone having to supervise it?

```text
Laika sees -> understands -> notices something interesting -> acts -> asks -> a human answers -> Laika remembers
```

The trial is yours: several days with Laika running in the house. This page is the kit that makes it easy to run and to judge. The judging stays with you; nothing here closes the issue.

## What the trial covers

| Stage | What she does | Where you can see it |
| --- | --- | --- |
| sees | takes pictures: idle presence checks, after moves, scans, investigations | `ImageCaptured` events, stored observations |
| understands | the vision model turns a picture into a validated observation | `ObservationCreated` events |
| notices | something new, moved or missing scores above the curiosity threshold | `InterestingChangeDetected`, "moved", "new", "missing" episodes |
| acts | looks at it again, maybe moves closer (only where autonomous driving is allowed), queues a question | investigation pictures, `Movement*` events, "question" episodes |
| asks | when a person is near and the interaction policy agrees, she turns and asks the pending question | `InteractionStarted` with `initiator: laika` and a `questionId` |
| human answers | someone replies | `InteractionCompleted` with `answered: true`, "answer" episodes |
| remembers | the answer becomes a fact and the object becomes known | facts and world objects in the store |

## Set up

1. Install the service as described in [running as a service](service.md): `sudo deploy/install-service.sh`, set the key in `/etc/laika/laika.env`, `sudo systemctl restart laika`. The service restarts her after a crash and stops the motors on every exit, so you can leave her alone.
2. Create `config/laika.toml` (copy `config/laika.example.toml`) and start with these values:

```toml
[robot]
# Stays false until the V0.2 hardware check (#15, #17, #18) has passed. With it off she only looks
# around with the head, so no wheel ever turns on its own: this is the safe way to start the trial.
allow_autonomous_driving = false

[vision]
enabled = true                 # without vision there are no observations, so no loop

[presence]
presence_interval_s = 300      # an idle picture every 5 minutes, to notice someone walking in
presence_check_s = 300

[interaction]
enabled = true
proactive = true               # she may start a conversation to ask a pending question
quiet_hours = "21:30-07:30"    # set the hours when the house sleeps

[investigate]
enabled = true

[storage]
# The event log is what the report reads. At the default 5 MB x 4 files, several days of pictures and
# observations can rotate the first day away. Keep a week or more:
log_max_mb = 20
log_backups = 10
keep_transcripts = false       # leave off: the report never needs what was said
```

3. The service keeps its data in `/var/lib/laika` (`LAIKA_DATA_DIR`, set by the unit). The report commands below use that path.
4. Optional but useful: enable driving only in a **second phase**, after the head-only days went well and the V0.2 verification is done. Say so in your notes, because the bar below changes (there will be moves, obstacles and cliff stops to explain).

Put the robot somewhere with people passing by and a few objects that change (a bag, a cup, a chair). The loop needs something new to notice, and a person to ask.

## House rules, pause and budgets for the trial

Before she drives on her own (only after the V0.2 hardware check, and with `python3 run-laika.py --unattended-check` showing what is still missing):

- Tell her the rules in Italian, once, as you would to a person: "Non entrare in camera da letto", "Stai fuori dallo studio". She keeps them (they are in the store) and `forget` lifts one. Best effort: she knows the room from what she sees, and she notices she has crossed a doorway only after the fact ([house rules](behaviours.md#house-rules-driving-budget-and-pause)). On the robot the room comes from room recognition ([behaviours](behaviours.md#room-recognition)): a rule acts only when she is confident which room she is in, so walk her through the house first, correct her ("No, questa è la cucina") and name the rooms ("Questa è la camera di Leonardo") before you trust a no-go room; until then keep `allow_autonomous_driving` off or stay near.
- Say "Laika, resta ferma" the moment you need her to stop wandering (a visitor, a child on the floor, a cable on the carpet). It stops everything she does on her own, also after a restart, until you say "puoi muoverti di nuovo". Count each use in your notes: a trial where you pause her every day is a finding.
- Keep the driving budgets in `[robot]` (`max_autonomous_drive_s_per_hour`, `..._per_day`) and watch the refusals in the report; they cap how much she can do unseen.

## Daily checklist

Once a day, at the same time (the report covers the last 24 hours by default):

```bash
sudo LAIKA_DATA_DIR=/var/lib/laika python3 run-laika.py --report
```

It needs no AI key and no hardware, it reads the data directory read-only (the database is opened in read-only mode, never written) and it prints a summary and writes `logs/trial-report-<date>.md` and `.json` in the data directory. Then:

1. Read **Supervision issues** first. For each line, write down what happened and what you did, if anything. A safety stop needs a reason you can name ("box left in the corridor"), not just a count.
2. Read **Loops**: completed loops are the success. Open loops show questions asked and not answered, or queued and never asked, with their age. A queued question that is older than 24 hours, or older than the last restart, is lost (pending questions live in memory), so it will not be asked.
3. Look at **Conversations**: who started them, the answered rate, and whether any proactive ask fell inside the quiet hours (it should never).
3a. Look at **Autonomy limits**: moves refused for `paused`, `drive_budget` or `no_go_room`, how often she turned back from a no-go room, whether the pause is still on, and the driving time stored for the day. A budget that is used up before lunch is a number to change, not a failure.
3a2. Look at **Room recognition**: room changes decided by vision, corrections by a person, and how often she did not know the room. Walk her through the house once and write down every time she names the wrong room; that, and the corrections, is the accuracy (the report cannot know the true room on the robot).
3b. Look at **Battery**: the low, critical and back-to-OK changes and the lowest voltage the events mention (only changes are logged, not every reading). A critical battery during a day means she needed charging.
4. Note anything you had to do by hand: restart the service, move her, pick her up, press the power switch, ssh in. Each of these counts against the trial, even when the report shows nothing.
5. Check the lines the report cannot see: `journalctl -u laika --since yesterday | grep -E "rejected|refused|paused|stalled"` (observation rejections, policy refusals, a paused or stalled exploration).
6. Keep the Markdown report. For several days use `--days 3` or `--since 2026-10-20`.

`--days N` covers the last N days; `--since YYYY-MM-DD` starts at local midnight of that date, and runs to now (or for `--days` days). From a laptop, copy `/var/lib/laika/logs` and `/var/lib/laika/memory` and run the report there with `LAIKA_DATA_DIR` pointing at the copy; copy the whole folder, or the database and its `-wal` file can be out of step.

### What the report cannot tell you

- It links a queued question to the ask that carries its `questionId` by order within one run, because nothing in the event log records that a question was queued. After a restart the numbering starts again at q-1, and the report knows where runs start from `HardwareReady`. If the log rotated past the start of a run, those asks show as "asked, but no matching queued question".
- It cannot see the words spoken. That is on purpose: the report holds counts, object names, room names and times, no transcripts, images or identities.
- Observation rejections, policy refusals and "stuck" explorations are only printed to the journal.

## What "the loop completes repeatedly without supervision issues" means

A suggested bar, measurable from the report, for **5 consecutive days** with the settings above. It is a suggestion; change it if you want, but write down what you changed before you start.

| Check | Bar |
| --- | --- |
| Completed loops (question queued, asked, answered, remembered) | At least 10 over the 5 days (an average of 2 a day), on at least 3 different days |
| Local watchdog expiries (`LocalWatchdogExpired`) | 0 |
| Hardware errors (`HardwareUnavailable`) | 0, or each one explained and fixed |
| Restarts (starts after the first) | 0 unexplained: each one matches something you did (an update, a config change, a reboot) or is understood and fixed |
| Safety stops | Every obstacle, cliff, emergency stop and `ai_unavailable` stop is explained |
| Stuck robot | No day on which Laika had to be rescued (picked up, freed, power-cycled) |
| Interruptions | No proactive ask inside the quiet hours (record the answered rate of her asks, but it is not part of the bar) |
| AI outages | Each one shorter than you tolerate, and she recovered by herself (no restart needed) |

The report prints a **Suggested bar** table for the period you ask for. It computes the number of completed loops (the target scales with the days: two a day), the watchdog expiries, the hardware errors and the asks inside quiet hours; for restarts and safety stops it shows the count and asks you to explain each one; "stuck" and the spread over days are yours to judge.

The first days may show zero completed loops because nothing new appeared or nobody was near when a question was queued. That is a finding too: write it down, then change something in the room (put an unknown object on the table) and try the loop again by hand ("Ehi Laika, che cosa vedi?", or just walk by).

## Posting findings to issue #44

Do not close the issue yourself and do not paste transcripts. Post this, with the Markdown report of the whole period (`--days 5`) pasted below it:

```markdown
### Field trial: <first date> to <last date>

Setup: autonomous driving <off/on>, presence every 5 min, vision on, quiet hours <hours>, laika <commit>.

Result against the bar: <met / not met, per line>

Completed loops: <n> over <days> days (<which days>)
Things I had to do by hand: <list with dates, or "none">
Safety stops explained: <list>
Restarts explained: <list>
What did not work, or surprised me: <list>
Next step: <what to change or fix>

<details><summary>Report (<first date> to <last date>)</summary>

(paste the contents of trial-report-<date>.md here)

</details>
```

The report holds counts, object names, room names and times only, so it is safe to paste. Check object and room names before you post them if your house is private.
