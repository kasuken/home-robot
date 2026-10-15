# Storage and retention

Laika runs on a Raspberry Pi 4 with 1 GB RAM and an SD card, so everything she keeps is bounded, on disk and in memory. The code is in `src/laika/storage.py`; settings are in the `[storage]` section of `config/laika.toml` (see `config/laika.example.toml`).

## Directory layout

The data directory is the first of these that is set:

1. `$LAIKA_DATA_DIR`
2. `[storage] data_dir`
3. `$XDG_DATA_HOME/laika`
4. `~/.local/share/laika`

Under `sudo`, `~` is `/root`. The systemd service (issue #18) sets `LAIKA_DATA_DIR` explicitly, for example `/var/lib/laika`. Laika prints the directory at startup.

```text
<data dir>/
  images/    camera frames (img-YYYYMMDD-HHMMSS-micros.jpg)
  logs/      events.jsonl (+ .1 .2 .3 rotated), transcripts-YYYY-MM-DD.jsonl (optional)
  memory/    laika.db: the local database (SQLite, WAL journal)
```

`config/laika.toml` (settings, and the API key if you do not use an environment variable) stays in the repository folder and is ignored by git.

## The database

Observations, and later the world model and memories, live in one SQLite file, `memory/laika.db` (mode `0600`), written by `src/laika/store.py`. Nothing is sent anywhere: there is no remote database. It uses the standard library `sqlite3` with the WAL journal and `synchronous=NORMAL`, which keeps SD card writes low.

What is stored:

| Table | Content |
| --- | --- |
| `observations` | One row per observation: id, time, room, description, objects (JSON), people, confidence, camera pan and tilt, image path, source and the interest score (JSON) |
| `world_objects` | The world model (#33): one row per known object, `name` and its JSON document (`KnownObject.to_dict()`: room, times seen, first and last seen, note, source, previous rooms) |
| `world_rooms` | The rooms Laika knows: name, when first seen, when last visited |
| `kv` | Generic JSON values by namespace and key, for memories and pending questions of later issues |
| `schema_version` | The migrations already applied |

How it behaves:

- Writes are queued and done by one writer thread, so the caller is never blocked and the safety controller and its monitor never touch the store. There are two queues. The observation queue holds 256 writes; when it is full the oldest observation write is dropped and counted. The world model queue holds 5000 writes and is never emptied to make room for observations: a world write is dropped only if that queue is full too, which needs thousands of unwritten changes, and it is counted separately (`world_dropped`). Observations are a log that can lose its oldest entries; the world model is the house's state and should not.
- Reads use their own short connections and never wait for the writer.
- Retention runs at startup and about every 200 writes or 10 minutes: only the newest `max_observations` rows (default 5000) are kept, and none older than `max_observation_days` (default 90). Images are pruned separately by the image limits, so a row can point to a picture that is gone; that is fine.
- If the file is corrupt, Laika moves it to `laika.db.corrupt-<time>` next to it, logs an error and starts a new one. If it cannot be opened or the disk is full, errors are logged and counted and Laika keeps running without storing.
- Schema changes are new numbered migrations in `store.py`; old ones are never edited. A database written by a newer Laika is left alone and not used.
- After many rows were pruned the file does not shrink by itself. `Store.vacuum()` does that, only when called.

Inspect it (the `sqlite3` command line tool may need `sudo apt install sqlite3`):

```bash
sqlite3 ~/.local/share/laika/memory/laika.db '.tables'
sqlite3 ~/.local/share/laika/memory/laika.db 'SELECT datetime(timestamp, "unixepoch", "localtime"), location, description FROM observations ORDER BY timestamp DESC LIMIT 10'
```

### What survives a restart

- **The world model** (`world_objects`, `world_rooms`): every object and room, with times seen, the room history of each object (the last five rooms), notes that people gave, and the room visit times. Laika starts from it, and each change is written back as it happens (a sighting, a move, a note, an eviction). After a restart the model matches the one before it, apart from the current room.
- **Not the current room.** Laika does not know where she is after a restart. She learns it when she sees something or enters a room; until then the prompt says she does not know which room she is in.
- **Observations**: the newest ones are reloaded into the in-memory log, and the full history stays in the table.
- Eviction (500 objects) deletes the object's row too. At startup the stored objects are loaded as they are; the 500 limit is applied in memory only.

Use `$LAIKA_DATA_DIR/memory/laika.db` if that variable is set. To wipe it, stop Laika and delete `laika.db`, `laika.db-wal` and `laika.db-shm` (see "Wiping data" below).

Privacy: descriptions of the home, what Laika saw and where, stay on the Pi in this file. Anyone with access to the Pi account can read them; the file is private to that user.

## Camera

The camera takes single photos only, through `src/laika/camera.py`, when the agent asks or when a trigger in the next section fires; there is no continuous video and nothing is uploaded. Settings are in `[camera]`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `width`, `height` | 640 x 480 | Frame size; clamped to at most 1280 x 960 |
| `jpeg_quality` | 80 | JPEG quality (10 to 95) |
| `min_interval_s` | 0.5 | A request sooner than this after the last photo gets the previous observation back instead of a new frame |
| `max_per_minute` | 30 | At most this many real captures in any 60 seconds; extra requests get the previous observation |
| `save_images` | true | false: the frame is processed but no file is written |

Each real capture is saved to `images/` and published as one `ImageCaptured` event (path, size, pan, tilt, scene, room, reason). Retention is the `images/` row in the table below: the newest `max_images` files and `max_images_mb` megabytes are kept, older ones are deleted at every save. A camera failure returns no image and publishes `HardwareUnavailable` once until the camera works again. The model never receives image bytes from the `capture_image` tool. Each new frame is also handed to the `Observer`, which makes a validated observation and publishes `ObservationCreated`; the world model is fed from that event, not from `ImageCaptured`.

## Permissions

Directories are created `0o700` and files `0o600` (with an explicit `chmod`, so the umask does not matter). On Windows, `chmod` does nothing; that only matters for development. At startup Laika warns if `config/laika.toml` is readable by every user on POSIX; fix it with `chmod 600 config/laika.toml`.

## Credentials

The API key lives in `config/laika.toml` (git-ignored) or in `OPENAI_API_KEY` / `LAIKA_AI_API_KEY`. It is never printed or logged, and the AI client does not log request headers. Event logs hold events only (movement, rooms, errors), not keys.

## Disk limits

| What | Where | Limit | Setting |
| --- | --- | --- | --- |
| Camera images | `images/` | newest 200 files and 100 MB, oldest deleted first; at most 30 captures per minute | `max_images`, `max_images_mb`, `[camera]` |
| Event log | `logs/events.jsonl` | rotates at 5 MB, keeps 3 old files: at most about 20 MB | `log_max_mb`, `log_backups` |
| Transcripts | `logs/transcripts-*.jsonl` | off by default; when on, files older than 7 days are deleted at startup and each new day; each line is cut at 2000 characters | `keep_transcripts`, `max_transcript_days` |
| Observation database | `memory/laika.db` | newest 5000 observations, none older than 90 days; checked at startup and about every 200 writes | `max_observations`, `max_observation_days` |

With the defaults, images (100 MB) plus event logs (20 MB) cap at about 120 MB, plus a few days of transcripts if you turn them on.

## Memory limits (RAM)

| What | Bound | Where |
| --- | --- | --- |
| Agent conversation history | `max_history_turns` question/answer pairs (default 8); tool messages are not kept between turns | `agent.py`, `[ai] max_history_turns` |
| Chat client history (JSON mode) | the same `max_history_turns` pairs | `ai_client.py` |
| Agent tool calls per turn | 8 calls, 4 rounds | `agent.py` |
| Event bus queue | 256 events; the oldest is dropped when full | `events.py` |
| World model | 500 objects, 5 previous rooms per object, 600 characters in the prompt summary | `world_model.py` |
| Observation log (the newest ones; the rest are in the database) | 50 observations; observer queue of 8 frames, the oldest is dropped when full; 50 recent rejections | `observations.py` |
| Remembered facts (`remember` tool) | 50 facts, 200 characters each | `tools.py` |
| Tool rejection history | 50 entries | `tools.py` |
| Safety controller | 32 finished moves, 100 monitor interval samples | `safety.py` |
| Simulator spoken lines (tests and `--sim`) | 100 lines | `simulator.py` |

Images are not held in memory beyond the latest frame, which the camera service keeps as `Camera.last`.

## When Laika takes pictures

Photos come from three places: the agent's `capture_image` tool (always, subject only to the `[camera]` limits), and the automatic triggers in `src/laika/capture_policy.py`, configured in `[capture]`:

| Trigger | When | Default | Cooldown setting (default) |
| --- | --- | --- | --- |
| `after_movement` | a move completed | on | `after_movement_cooldown_s` (10 s) |
| `person` | a person came into view | on | `person_cooldown_s` (30 s) |
| `interesting` | something interesting changed | on | `interesting_cooldown_s` (20 s) |
| `investigate` | an investigation takes its own pictures through the camera (#29), not through this policy | always | the investigation caps (`[investigate]`) |
| `exploration` | while exploring (#27), on a timer | only while exploring | `exploration_interval_s` (30 s) |

All automatic triggers share one limit, `max_auto_per_minute` (6): any photo beyond it is skipped. Automatic photos are sent to the vision model when `[vision]` is on, so this number is the main cost limit. A photo that `[camera]` refuses (interval or `max_per_minute`) is not counted. Automatic captures run on their own thread with one pending slot, so a burst of events never queues more than one photo.

## Privacy defaults

- Transcripts are off: they would hold what people said, children included. Turn them on with `keep_transcripts = true` only when you need to debug a conversation.
- The event log contains no speech. Events carry robot activity such as movement, rooms and errors.
- Camera images stay on the Pi. With `[vision]` enabled, a frame is sent to the configured `[ai]` endpoint only when a photo is taken: one frame per request, resized in memory, never video and never stored by Laika for this purpose. Set `[vision] enabled = false` to send nothing. What the provider keeps is governed by the provider's own policy.

## Wiping data

Stop Laika, then delete the directory contents:

```bash
rm -rf "$LAIKA_DATA_DIR"/images/* "$LAIKA_DATA_DIR"/logs/* "$LAIKA_DATA_DIR"/memory/*
```

Use the path printed at startup if the variable is not set. Laika recreates the folders on the next start.
