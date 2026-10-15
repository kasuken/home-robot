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
  memory/    reserved for persistent memory (not written yet)
```

`config/laika.toml` (settings, and the API key if you do not use an environment variable) stays in the repository folder and is ignored by git.

## Camera

The camera is used on request only, through `src/laika/camera.py`; there is no continuous video and nothing is uploaded. Settings are in `[camera]`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `width`, `height` | 640 x 480 | Frame size; clamped to at most 1280 x 960 |
| `jpeg_quality` | 80 | JPEG quality (10 to 95) |
| `min_interval_s` | 0.5 | A request sooner than this after the last photo gets the previous observation back instead of a new frame |
| `max_per_minute` | 30 | At most this many real captures in any 60 seconds; extra requests get the previous observation |
| `save_images` | true | false: the frame is processed but no file is written |

Each real capture is saved to `images/` and published as one `ImageCaptured` event (path, size, pan, tilt, scene, room, reason). Retention is the `images/` row in the table below: the newest `max_images` files and `max_images_mb` megabytes are kept, older ones are deleted at every save. A camera failure returns no image and publishes `HardwareUnavailable` once until the camera works again. The model never receives image bytes from the `capture_image` tool.

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
| Memory | `memory/` | nothing is written yet | (later issue) |

With the defaults, images (100 MB) plus event logs (20 MB) cap at about 120 MB, plus a few days of transcripts if you turn them on.

## Memory limits (RAM)

| What | Bound | Where |
| --- | --- | --- |
| Agent conversation history | `max_history_turns` question/answer pairs (default 8); tool messages are not kept between turns | `agent.py`, `[ai] max_history_turns` |
| Chat client history (JSON mode) | the same `max_history_turns` pairs | `ai_client.py` |
| Agent tool calls per turn | 8 calls, 4 rounds | `agent.py` |
| Event bus queue | 256 events; the oldest is dropped when full | `events.py` |
| World model | 500 objects, 5 previous rooms per object, 600 characters in the prompt summary | `world_model.py` |
| Remembered facts (`remember` tool) | 50 facts, 200 characters each | `tools.py` |
| Tool rejection history | 50 entries | `tools.py` |
| Safety controller | 32 finished moves, 100 monitor interval samples | `safety.py` |
| Simulator spoken lines (tests and `--sim`) | 100 lines | `simulator.py` |

Images are not held in memory beyond the latest frame, which the camera service keeps as `Camera.last`.

## Privacy defaults

- Transcripts are off: they would hold what people said, children included. Turn them on with `keep_transcripts = true` only when you need to debug a conversation.
- The event log contains no speech. Events carry robot activity such as movement, rooms and errors.
- Camera images stay on the Pi. They are sent to the AI endpoint only when a tool or feature explicitly does so.

## Wiping data

Stop Laika, then delete the directory contents:

```bash
rm -rf "$LAIKA_DATA_DIR"/images/* "$LAIKA_DATA_DIR"/logs/* "$LAIKA_DATA_DIR"/memory/*
```

Use the path printed at startup if the variable is not set. Laika recreates the folders on the next start.
