# Running Laika as a service

Laika can run headless under systemd instead of being started by hand: it starts on boot, restarts after a crash, logs to the journal, and the robot is stopped before it starts and after it ends, however it ended. The files are in [`deploy/`](../deploy/) (issue #18). This is implemented and tested on a development computer; verifying it on the robot is the owner's step, see [Verify on the Pi](#verify-on-the-pi).

## Install

On the Pi, from the clone:

```bash
sudo deploy/install-service.sh        # uses the clone it is run from; or pass another path
sudo nano /etc/laika/laika.env        # set OPENAI_API_KEY (or use [ai] api_key in config/laika.toml)
sudo systemctl restart laika
```

The script writes `/etc/systemd/system/laika.service` with the clone's real path, creates `/etc/laika/laika.env` (0600, no key in it) only if missing, runs `daemon-reload` and `enable --now`. Running it again refreshes the unit and keeps your env file. Create `config/laika.toml` as described in the README first if you use it. Remove everything with `sudo deploy/uninstall-service.sh` (it keeps `/etc/laika` and `/var/lib/laika`).

The first start downloads the Vosk model (about 50 MB), so it needs internet and can take minutes; `TimeoutStartSec=600` allows for it.

## Day to day

```bash
sudo systemctl status laika
sudo systemctl stop laika        # stops the robot, then exits
sudo systemctl restart laika
sudo systemctl disable laika     # do not start on boot
journalctl -u laika -f           # live logs
journalctl -u laika -b           # this boot
```

All output (`[laika] ...` lines, warnings, tracebacks) goes to the journal; nothing is written to a log file for the service.

## Where things live

| What | Where |
| --- | --- |
| Unit | `/etc/systemd/system/laika.service` (copied from `deploy/laika.service`) |
| Secrets | `/etc/laika/laika.env`, 0600, root only; systemd reads it as `EnvironmentFile` |
| Data (images, event log, transcripts) | `/var/lib/laika`, 0700, created by `StateDirectory=laika`; the unit sets `LAIKA_DATA_DIR` to it. Limits: [storage and retention](storage-and-retention.md) |
| Settings | `config/laika.toml` in the clone, as when started by hand |

## How the robot is stopped

The Robot HAT keeps its last motor output when a process dies, so three layers make sure the wheels stop:

1. **SIGTERM / SIGHUP (normal stop, `systemctl stop|restart`).** Laika installs a handler: it first emergency-stops the motors through the safety controller, then raises in the main thread so the usual shutdown runs (gesture worker, safety monitor, `robot.close()`: motors stopped, head and wheels centred). It exits 0, so systemd does not count it as a failure. A repeated signal during shutdown is ignored.
2. **`ExecStopPost`, the safety net.** After the process is gone, for any reason (crash, watchdog kill, `kill -9`, a main thread stuck in a C call that never ran the handler, `TimeoutStopSec` expiring), systemd runs `run-laika.py --stop-motors`. It builds the robot, stops it, centres it and releases it. `ExecStartPre` runs the same command, so Laika never starts with the wheels turning.
3. **systemd watchdog.** The unit is `Type=notify` with `WatchdogSec=5`. Laika sends `READY=1` once everything is up and `WATCHDOG=1` from the safety monitor loop (at most every `WatchdogSec/3`), only while the monitor ticks. If the monitor hangs, pings stop, systemd kills Laika (SIGABRT, then SIGKILL after the stop timeout) and `Restart=on-failure` brings it back after `RestartSec=3`; `ExecStopPost` stops the motors in between. The in-process watchdog ([ADR 0002](decisions/0002-watchdog-strategy.md)) stops the motors much sooner; this covers the process as a whole. A hung conversation thread is not a watchdog failure: moves are individually bounded, so the robot cannot keep driving.

`--stop-motors` always exits 0, even when the hardware cannot be reached, because a non-zero `ExecStopPost` would mark the stop as failed. It prints what it did or what failed (`[stop-motors] ...`), which shows in the journal. Use `--stop-motors --sim` to try it on a laptop.

`Restart=on-failure` restarts after crashes, non-zero exits and watchdog timeouts, but not after a clean stop. `StartLimitBurst=5` in `StartLimitIntervalSec=120` stops a crash loop; reset it with `sudo systemctl reset-failed laika`.

## Headless notes

- Voice mode reads nothing from stdin; the service has none.
- `--keyboard` (and `--sim`) read typed input. Under systemd stdin is empty, so Laika would exit at once: do not combine them with the service. Add options such as `--no-driving` by editing `ExecStart` in the unit (`sudo systemctl edit --full laika`).
- Laika runs as **root**, like `sudo python3 run-laika.py`, because the SunFounder libraries need GPIO, I2C, PWM and audio. The unit has only conservative hardening (`NoNewPrivileges`, kernel and cgroup protection); `ProtectHome` is off because the clone is under `/home`. The unit contains a commented-out `User=` and `SupplementaryGroups=gpio i2c spi audio video` alternative: try it later, and verify every sensor, the camera, microphone and speaker before relying on it.

## Verify on the Pi

Done when: Laika starts on boot and restarts after a crash with the robot stopped. Check:

1. `sudo deploy/install-service.sh`, then `systemctl status laika` shows `active (running)` and the journal shows `[laika] sleeping`.
2. `sudo reboot`; after boot, `systemctl status laika` is active and she answers "ehi Laika" without anyone logging in.
3. Start a move (ask her to drive forward), then `sudo systemctl kill -s SIGKILL laika`. The wheels stop, the unit restarts about 3 seconds later; the journal shows `[stop-motors] robot stopped` and a fresh start.
4. Hang test (optional): `sudo systemctl kill -s SIGSTOP laika`; within about 5 seconds systemd restarts it (journal: watchdog timeout), the motors stopped.
5. `sudo systemctl stop laika`: the journal shows `SIGTERM received, stopping the robot`, `[robot] closed` and `[stop-motors] robot stopped`.

Report the result on issues #18 and #15.
