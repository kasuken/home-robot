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
  stories/   bedtime stories told (story-YYYYMMDD-HHMMSS.txt)
  learning/  logs of the skill-learning jobs (skill-*.log) and history.json
```

`config/laika.toml` (the settings) is committed to git, so `git pull` brings every change to the robot. The keys never go there: they live in `config/secrets.toml` next to it, which git ignores, or in environment variables.

## The database

Observations, the world model, facts and episodes live in one SQLite file, `memory/laika.db` (mode `0600`), written by `src/laika/store.py`. Nothing is sent anywhere: there is no remote database. It uses the standard library `sqlite3` with the WAL journal and `synchronous=NORMAL`, which keeps SD card writes low.

What is stored:

| Table | Content |
| --- | --- |
| `observations` | One row per observation: id, time, room, description, objects (JSON), people, confidence, camera pan and tilt, image path, source and the interest score (JSON) |
| `world_objects` | The world model (#33): one row per known object, `name` and its JSON document (`KnownObject.to_dict()`: room, times seen, first and last seen, note, source, previous rooms) |
| `world_rooms` | The rooms Laika knows: name, when first seen, when last visited, and the names people gave the room (`aliases`, a JSON list of up to 5, added by migration 4, #53) |
| `facts` | Semantic memory (#34): id, text as the user said it, subject, kind (`location`, `ownership`, `preference`, `rule`, `general`), source (`user`, `vision`, `investigation`), created, updated, confidence |
| `episodes` | Episodic memory (#34): id, time, kind (`seen`, `moved`, `conversation`, `question`, `answer`, `rule`), short English summary, subjects (JSON, plus a folded copy for searching), room |
| `kv` | Generic JSON values by namespace and key, for small values of later issues |
| `schema_version` | The migrations already applied |

How it behaves:

- Writes are queued and done by one writer thread, so the caller is never blocked and the safety controller and its monitor never touch the store. There are two queues. The observation queue holds 256 writes; when it is full the oldest observation write is dropped and counted. The world model queue (shared with facts and episodes) holds 5000 writes and is never emptied to make room for observations: a world write is dropped only if that queue is full too, which needs thousands of unwritten changes, and it is counted separately (`world_dropped`). Observations are a log that can lose its oldest entries; the world model and what Laika was told should not.
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
- **Facts and episodes**: all facts and the newest 200 episodes are reloaded; older episodes stay in the table and `recall` finds them there.
- **Observations**: the newest ones are reloaded into the in-memory log, and the full history stays in the table.
- Eviction (500 objects) deletes the object's row too. At startup the stored objects are loaded as they are; the 500 limit is applied in memory only.

### Memory: facts and episodes

Code: `src/laika/memory.py`. Facts and episodes are loaded at startup and written through the same queue as the world model.

- **Facts** keep the user's own words (any language) with an optional subject and a kind. A new fact with the same subject, kind `location` or `ownership` and source replaces the old one (a usual place or an owner changes, it does not pile up); other kinds are added, and identical text only refreshes the time. Facts are never aged out: they stay until replaced, evicted or forgotten. At most 500 are kept; beyond that the least recently updated one is evicted, in memory and in the table.
- **Episodes** are events. The `EpisodeRecorder` records a `seen` episode for each object in an observation, at most once per object and room per `[storage] seen_episode_interval_s` (default 3600 s), so a camera staring at a sofa adds one row an hour. The `remember` tool records a `conversation` or `rule` episode, `answer_question` an `answer`, and a queued question a `question`. The change tracker (#36) records `moved`, `new` and `missing` episodes (summary: the change text, subject: the object, location: where it is now or was), and the `what_changed` tool reads them by time. The newest 200 are held in memory; the table keeps `[storage] max_episodes` (default 20000) and none older than `max_episode_days` (default 365), checked with the observation retention. Episode kinds: `seen`, `moved`, `new`, `missing`, `conversation`, `question`, `answer`, `rule`.
- **Recall.** The `recall` tool returns facts (subject match before text match, ignoring case and accents), the world model's last known room and time, and the latest episodes about the subject, each with an absolute local time (minute precision) and an English hint such as "2 hours ago" or "yesterday" for the model to phrase in Italian. Each object also has `first_seen`, `usual_room` and `is_usual_here`. The result is bounded (8 facts, 8 objects, 5 episodes). The agent's prompt also lists the five facts that best match the user's message (at most 600 characters).
- **Outside.** The window watcher (#48) adds an `outside` episode per look and `outside_change` episodes for what changed; the windows themselves (room, pan, tilt) are one row in the key/value table (`windows`/`registry`) and the latest look another (`outside`/`latest`). `recall` finds them by `outside` or `weather`.
- **Forget.** `forget` takes a subject or a fact id and deletes those facts from memory and the table. It does not touch the world model's record of the object, nor past episodes, which age out by the retention above; to remove everything, wipe the database (see "Wiping data").
- **Privacy.** Facts and episodes can contain the names of family members and what they said. They stay in `memory/laika.db` on the Pi and are sent to the AI endpoint only as part of the prompt or a tool result when relevant to a conversation.

### What Laika remembers from conversations

Code: `src/laika/conversation_memory.py`. Not every message becomes long-term memory (docs/prd.md, "Conversation memory"). Two layers decide, the model first and local rules as a safety net.

- **The model.** The tool-mode prompt tells Laika to call `remember` only when someone tells her what an object is or what it is called, where it usually stays, who owns it, a preference or a house rule ("non entrare in camera"), with the right `kind` and `subject` (and `room` for a place). Small talk, questions, jokes and one-off requests are not remembered. A reply to a question she was waiting to ask goes through `answer_question`.
- **The rules.** After each completed turn, if the model called none of `remember`, `answer_question` or `forget`, `MemorySelector` reads what the user said with a few Italian patterns (case, accents and apostrophes ignored). It works in JSON mode (`use_tools = false`) too, where it is the only path.

| Pattern (examples) | Stored as | Confidence |
| --- | --- | --- |
| a reply to a pending question: "È uno stabilizzatore per la fotocamera", "Uno stabilizzatore", "È di Leonardo", "Si chiama Rex" | the object becomes known (see below); a `general` or `ownership` fact; an `answer` episode; the question is removed | 0.7 to 0.85 |
| "non entrare in camera", "non devi andare nello studio" | `rule`, subject = the place | 0.9 |
| "il tablet di solito sta in cucina", "lo zaino sta vicino all'ingresso" | `location` (replaces the old one); with a known room (soggiorno, salotto, cucina, studio, ufficio, camera) the object moves there | 0.85 (with "di solito", "sempre", "normalmente"), else 0.65 |
| "lo zaino nero è di Leonardo", "la chitarra appartiene a Anna", "questo è di Leonardo" | `ownership`; "questo" means the one object in view | 0.8 |
| "il gatto si chiama Briciola" | `general` | 0.8 |
| "questo è un cacciavite" with exactly one object in view (seen in the last two minutes) | `general` about that object, which becomes known | 0.7 |
| "mi piace la musica classica", "preferisco il tè" | `preference` | 0.75 |

Only a candidate at or above `[memory] min_confidence` (default 0.6) is stored; source is `user` and the text is the user's own words. Each stored item is logged as `[laika] remembered: <kind> <subject>: <text>`.

- **Telling Laika what an object is makes it a known object.** Both the selector and the `remember` and `answer_question` tools set the object's note in the world model (source `conversation`), for a `general` or `ownership` fact about an object the world model knows, one a pending question is about, or the one in view. The curiosity score then ignores its low camera confidence (`unknown_object` is 0 for an object with a note) and the investigator does not queue a new question about it.
- **Never stored:** questions (a final "?", or a first word such as "cosa", "dove", "come", "chi", since speech recognition writes no question mark), greetings, remarks of fewer than three words (unless they answer a pending question), anything with "dimentica", "scordati" or "cancella", a "non so"/"boh" answer, and everything when `[memory] auto_select = false` (then only the model decides).
- **Bounded conversation history.** The agent keeps the last `[ai] max_history_turns` question and answer pairs (default 8) and forgets them when the chat ends; tool messages are dropped after each turn. The JSON-mode client keeps the same number of pairs. At most 5 pending questions are kept, and they expire after `[investigate] question_ttl_h`. Transcripts are off by default (`[storage] keep_transcripts`), so what people said is not written to disk unless you turn them on; only the selected facts are.
- **Make her forget.** Say "dimentica ..." (she calls `forget` with the subject or the fact id), or wipe the database (see "Wiping data"). `forget` does not remove the world model's note or past episodes.
- **Privacy.** Only the facts, the notes and the `answer` or `conversation` episodes are kept, as the user's words, on the Pi in `memory/laika.db`. Set `auto_select = false` to keep only what the model chooses, or raise `min_confidence` to keep less.

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

Each real capture is saved to `images/` and published as one `ImageCaptured` event (path, size, pan, tilt, scene, room, reason). Retention is the `images/` row in the table below: the newest `max_images` files and `max_images_mb` megabytes are kept, and no picture older than `max_image_days` (7). Old pictures are deleted at every save, at startup and every hour, so they go away even when Laika takes no new pictures and the SD card never fills up; half-written files left by a crash are removed after an hour. A camera failure returns no image and publishes `HardwareUnavailable` once until the camera works again. The model never receives image bytes from the `capture_image` tool. Each new frame is also handed to the `Observer`, which makes a validated observation and publishes `ObservationCreated`; the world model is fed from that event, not from `ImageCaptured`.

## Permissions

Directories are created `0o700` and files `0o600` (with an explicit `chmod`, so the umask does not matter). On Windows, `chmod` does nothing; that only matters for development. At startup Laika warns if `config/laika.toml` is readable by every user on POSIX; fix it with `chmod 600 config/laika.toml`.

## Credentials

The keys live in `config/secrets.toml` (git-ignored; it may only hold `[ai] api_key` and `[web] brave_api_key`; run `chmod 600` on it) or in `OPENAI_API_KEY` / `LAIKA_AI_API_KEY` and `LAIKA_BRAVE_API_KEY`, which win. `config/laika.toml` is committed to a public repository, so Laika warns at startup if a key was put there, and a test fails if one is ever committed. It is never printed or logged, and the AI client does not log request headers. Event logs hold events only (movement, rooms, errors), not keys.

## Disk limits

| What | Where | Limit | Setting |
| --- | --- | --- | --- |
| Camera images | `images/` | newest 200 files and 100 MB, none older than 7 days; cleaned at every save, at startup and hourly; at most 30 captures per minute | `max_images`, `max_images_mb`, `max_image_days`, `[camera]` |
| Bedtime stories | `stories/` | every story is kept, up to the newest 1000 files and 20 MB (a story is about 4 KB: years of evenings); checked at every save | none (`bedtime.py` constants) |
| Skill-learning logs | `learning/` | newest 30 `skill-*.log` files and 5 MB; `history.json` keeps only the starts of the last 24 h | none (`learning.py` constants) |
| Event log | `logs/events.jsonl` | rotates at 5 MB, keeps 3 old files: at most about 20 MB | `log_max_mb`, `log_backups` |
| Transcripts | `logs/transcripts-*.jsonl` | off by default; when on, files older than 7 days are deleted at startup and each new day; each line is cut at 2000 characters | `keep_transcripts`, `max_transcript_days` |
| Observation database | `memory/laika.db` | newest 5000 observations, none older than 90 days; checked at startup and about every 200 writes | `max_observations`, `max_observation_days` |
| Episodes | `memory/laika.db` | newest 20000 episodes, none older than 365 days; facts: 500, none aged out | `max_episodes`, `max_episode_days` |

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
| Facts (`remember` tool) | 500 facts, 300 characters each; 5 shown in the prompt | `memory.py` |
| Episodes | the newest 200 in memory; "seen" dedupe table of 2000 keys | `memory.py` |
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
| `outside` | the window watcher looks out of a window (#48); not through this policy and not an ordinary observation | `[outside] enabled` | `[outside] min_interval_s` (1800 s per window) and `max_per_day` (24) |

All automatic triggers share one limit, `max_auto_per_minute` (6): any photo beyond it is skipped. Automatic photos are sent to the vision model when `[vision]` is on, so this number is the main cost limit. A photo that `[camera]` refuses (interval or `max_per_minute`) is not counted. Automatic captures run on their own thread with one pending slot, so a burst of events never queues more than one photo.

## Privacy defaults

- Transcripts are off: they would hold what people said, children included. Turn them on with `keep_transcripts = true` only when you need to debug a conversation.
- The event log contains no speech. Events carry robot activity such as movement, rooms and errors.
- Camera images stay on the Pi. With `[vision]` enabled, a frame is sent to the configured `[ai]` endpoint only when a photo is taken: one frame per request, resized in memory, never video and never stored by Laika for this purpose. Set `[vision] enabled = false` to send nothing. What the provider keeps is governed by the provider's own policy.
- Looking out of the windows (`[outside]`, #48) sends a photo of the outside to the same vision endpoint, at most every 30 minutes per window and 24 times a day by default, with a prompt that asks for the weather, the light, short events and only the **number** of people outside: never who they are or what they look like (an instruction to the model, backed by a schema that has no place for a description). The picture is saved with the other pictures and deleted by the normal image retention; a window can see a neighbour's garden or a street, so point it where you are comfortable, or set `[outside] enabled = false` to send nothing.
- Looking things up (`[web]`, #52) sends the query text to Wikipedia (`wikipedia` tool) and, with a Brave key, to the Brave Search API (`web_search`, strict safe search), together with a User-Agent that names Laika. The prompt tells Laika never to search for names of family members, addresses or details about the home, but that is an instruction to the model, not a filter: set `[web] enabled = false` to send nothing. Results are never stored: they live in a 64-entry in-memory cache for one hour and are gone after a restart. The Brave key is read from `LAIKA_BRAVE_API_KEY` or `config/laika.toml` and is never printed or logged. Web content cannot move the robot or write memory in the turn that read it.

## Wiping data

Stop Laika, then delete the directory contents:

```bash
rm -rf "$LAIKA_DATA_DIR"/images/* "$LAIKA_DATA_DIR"/logs/* "$LAIKA_DATA_DIR"/memory/* "$LAIKA_DATA_DIR"/stories/*
```

Use the path printed at startup if the variable is not set. Laika recreates the folders on the next start.
