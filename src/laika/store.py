"""The local persistent store: one SQLite file on the Pi, no remote database (#32).

Observations and the world model survive a restart; a small generic `kv` table is there
for pending questions and other small values. Facts and episodes (#34) have their own tables.

Writes go through one dedicated writer thread, so storage work never blocks the caller
and never reaches the safety thread. There are two bounded queues in one writer: when
the observation queue is full the oldest observation write is dropped and counted, but
world model writes have their own, much larger queue and are never dropped to make room
for observations (see `_enqueue`); facts and episodes (#34) share the world queue, so what Laika
was told is not lost to a flood of sightings. Reads use short-lived connections
(SQLite's WAL journal lets them run while the writer works). Nothing here raises into
callers: if the file is corrupt it is moved aside and a fresh one is started, and if the
disk is full or the file cannot be opened Laika keeps running without a store.

Schema changes: append a new entry to `MIGRATIONS`, never edit an old one.
Each migration is a list of SQL statements applied in one transaction at open.
`VACUUM` (shrinking the file after big prunes) only runs when `Store.vacuum()` is called.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from laika.events import Event, EventBus, EventType
from laika.observations import LOG_SIZE, InvalidObservation, Observation, ObservationLog, fold, normalise
from laika.storage import FILE_MODE

log = logging.getLogger(__name__)

DB_NAME = "laika.db"
QUEUE_LIMIT = 256
# World writes have their own queue: one change per object and room touched, so this is
# far more than a stuck writer needs to hit before anything is lost.
WORLD_QUEUE_LIMIT = 5000
WORLD_KINDS = frozenset({"world_object", "world_delete", "world_room", "world_room_aliases"})
MEMORY_KINDS = frozenset({"fact_put", "fact_delete", "episode_put"})
# Writes that must not be dropped to make room for observations.
PRIORITY_KINDS = WORLD_KINDS | MEMORY_KINDS
BUSY_TIMEOUT_MS = 5000
RETENTION_EVERY_WRITES = 200
MAINTENANCE_INTERVAL_S = 600.0
CLOSE_TIMEOUT_S = 3.0
MAX_LOGGED_FAILURES = 5
DAY_S = 86400.0

MIGRATIONS: list[list[str]] = [
    # 1: observations (#32) and a generic key/value table for later issues.
    [
        """CREATE TABLE observations (
            id TEXT PRIMARY KEY,
            timestamp REAL NOT NULL,
            location TEXT,
            description TEXT NOT NULL,
            objects TEXT NOT NULL,
            people INTEGER NOT NULL DEFAULT 0,
            confidence REAL NOT NULL,
            camera_pan REAL NOT NULL DEFAULT 0,
            camera_tilt REAL NOT NULL DEFAULT 0,
            image_path TEXT,
            source TEXT NOT NULL,
            interest TEXT
        )""",
        "CREATE INDEX idx_observations_timestamp ON observations (timestamp)",
        "CREATE INDEX idx_observations_location ON observations (location)",
        """CREATE TABLE kv (
            namespace TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            updated REAL NOT NULL,
            PRIMARY KEY (namespace, key)
        )""",
    ],
    # 2: the world model (#33). `world_objects` holds one JSON document per object. Its rowid
    # order is the model's least-recently-seen order: every write replaces the row (new rowid).
    [
        """CREATE TABLE world_objects (
            name TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            updated REAL NOT NULL
        )""",
        """CREATE TABLE world_rooms (
            name TEXT PRIMARY KEY,
            first_seen REAL NOT NULL,
            last_visit REAL NOT NULL
        )""",
    ],
    # 3: semantic and episodic memory (#34). `search` holds the folded subjects as "|a|b|" so a
    # substring test finds an episode whatever the accents or case.
    [
        """CREATE TABLE facts (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            subject TEXT,
            kind TEXT NOT NULL,
            source TEXT NOT NULL,
            created REAL NOT NULL,
            updated REAL NOT NULL,
            confidence REAL NOT NULL
        )""",
        "CREATE INDEX idx_facts_subject ON facts (subject)",
        "CREATE INDEX idx_facts_updated ON facts (updated)",
        """CREATE TABLE episodes (
            id TEXT PRIMARY KEY,
            timestamp REAL NOT NULL,
            kind TEXT NOT NULL,
            summary TEXT NOT NULL,
            subjects TEXT NOT NULL,
            search TEXT NOT NULL,
            location TEXT
        )""",
        "CREATE INDEX idx_episodes_timestamp ON episodes (timestamp)",
    ],
    # 4: the names people gave a room (#53), a JSON list of aliases kept with the room.
    [
        "ALTER TABLE world_rooms ADD COLUMN aliases TEXT NOT NULL DEFAULT '[]'",
    ],
]
SCHEMA_VERSION = len(MIGRATIONS)

_COLUMNS = (
    "id, timestamp, location, description, objects, people, confidence, "
    "camera_pan, camera_tilt, image_path, source, interest"
)
_NAME_OF = "CASE j.type WHEN 'object' THEN json_extract(j.value, '$.name') ELSE j.value END"


@dataclass(frozen=True)
class StoreStats:
    available: bool
    queued: int
    written: int
    dropped: int
    failed: int
    world_dropped: int = 0


@dataclass
class _Op:
    kind: str  # "observation", "kv", "vacuum", "flush" or one of WORLD_KINDS
    args: tuple[Any, ...] = ()
    done: threading.Event | None = None


class StoreVersionError(sqlite3.OperationalError):
    """The file was written by a newer Laika; it is left untouched and the store stays off."""


def _connect(path: Path) -> sqlite3.Connection:
    # Autocommit: transactions are explicit. The writer connection is used by one thread at a time.
    conn = sqlite3.connect(
        path, timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None, check_same_thread=False
    )
    try:
        conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    except BaseException:
        conn.close()  # an open handle would stop a corrupt file from being moved aside on Windows
        raise
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    """Apply the migrations the file has not seen yet, each in its own transaction."""
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    current = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] or 0
    if current > SCHEMA_VERSION:
        raise StoreVersionError(f"database schema {current} is newer than this Laika ({SCHEMA_VERSION})")
    for version in range(current + 1, SCHEMA_VERSION + 1):
        conn.execute("BEGIN IMMEDIATE")
        try:
            for statement in MIGRATIONS[version - 1]:
                conn.execute(statement)
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise


class Store:
    """Observations and key/value data in `path`. Safe to call from any thread; never raises."""

    def __init__(
        self,
        path: Path,
        *,
        max_observations: int = 5000,
        max_observation_days: float = 90.0,
        max_episodes: int = 20000,
        max_episode_days: float = 365.0,
        queue_limit: int = QUEUE_LIMIT,
        world_queue_limit: int = WORLD_QUEUE_LIMIT,
        retention_every: int = RETENTION_EVERY_WRITES,
        maintenance_interval_s: float = MAINTENANCE_INTERVAL_S,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path)
        self.max_observations = max(1, int(max_observations))
        self.max_observation_days = max(0.0, float(max_observation_days))
        self.max_episodes = max(1, int(max_episodes))
        self.max_episode_days = max(0.0, float(max_episode_days))
        self.queue_limit = max(1, queue_limit)
        self.retention_every = max(1, retention_every)
        self.maintenance_interval_s = maintenance_interval_s
        self._clock = clock
        self._cond = threading.Condition()
        self.world_queue_limit = max(1, world_queue_limit)
        self._queue: deque[_Op] = deque()
        self._queued_data = 0  # observation, kv and vacuum writes waiting in `_queue`
        self._queued_world = 0  # world model writes waiting in `_queue`
        self._closed = False
        self._written = 0
        self._dropped = 0
        self._world_dropped = 0
        self._failed = 0
        self._writes_since_retention = 0
        self._last_maintenance = time.monotonic()
        self._conn: sqlite3.Connection | None = self._open()
        self._thread: threading.Thread | None = None
        if self._conn is not None:
            self._thread = threading.Thread(target=self._run, name="store-writer", daemon=True)
            self._thread.start()

    # -- opening ---------------------------------------------------------------------------

    def _open(self) -> sqlite3.Connection | None:
        try:
            return self._open_file()
        except StoreVersionError as error:
            log.error("%s: %s; running without a store", self.path, error)
        except sqlite3.OperationalError as error:  # cannot open, locked, read-only: not corruption
            log.error("cannot open the store %s: %s; running without a store", self.path, error)
        except sqlite3.DatabaseError as error:
            log.error("the store %s is corrupt (%s); starting a fresh one", self.path, error)
            if not self._move_aside():
                return None
            try:
                return self._open_file()
            except (sqlite3.Error, OSError) as again:
                log.error("cannot start a fresh store: %s; running without a store", again)
        except OSError as error:
            log.error("cannot open the store %s: %s; running without a store", self.path, error)
        return None

    def _open_file(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            os.close(os.open(self.path, os.O_CREAT | os.O_WRONLY, FILE_MODE))
        try:
            os.chmod(self.path, FILE_MODE)
        except OSError:
            pass
        conn = _connect(self.path)
        try:
            migrate(conn)
        except BaseException:
            conn.close()
            raise
        return conn

    def _move_aside(self) -> bool:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self._clock()))
        target = self.path.with_name(f"{self.path.name}.corrupt-{stamp}")
        try:
            os.replace(self.path, target)
            for suffix in ("-wal", "-shm"):
                self.path.with_name(self.path.name + suffix).unlink(missing_ok=True)
        except OSError as error:
            log.error("could not move the corrupt store aside: %s", error)
            return False
        log.error("the corrupt store was moved to %s", target)
        return True

    @property
    def available(self) -> bool:
        return self._conn is not None and not self._closed

    def stats(self) -> StoreStats:
        with self._cond:
            return StoreStats(
                self.available,
                self._queued_data + self._queued_world,
                self._written,
                self._dropped,
                self._failed,
                self._world_dropped,
            )

    # -- writing (queued) ------------------------------------------------------------------

    def add_observation(self, observation: Observation, interest: dict[str, Any] | None = None) -> None:
        """Queue an observation (with its interest score as a dict). Never blocks, never raises."""
        try:
            objects = json.dumps([o.to_json() for o in observation.objects], ensure_ascii=False)
            score = None if interest is None else json.dumps(interest, ensure_ascii=False)
            row = (
                observation.id, observation.timestamp, observation.location, observation.description,
                objects, observation.people, observation.confidence, observation.camera_pan,
                observation.camera_tilt, observation.image_path, observation.source, score,
            )
        except Exception:  # noqa: BLE001
            log.exception("could not prepare an observation for the store")
            return
        self._enqueue(_Op("observation", row))

    def kv_put(self, namespace: str, key: str, value: Any) -> None:
        """Queue `value` (anything JSON) under `namespace`/`key`. Never blocks, never raises."""
        try:
            text = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            log.warning("store: value for %s/%s is not JSON, not stored", namespace, key)
            return
        self._enqueue(_Op("kv", (namespace, key, text, self._clock())))

    def put_world_object(self, data: Mapping[str, Any]) -> None:
        """Queue one object's document (`KnownObject.to_dict()`), replacing the stored one."""
        try:
            name = str(data["name"])
            text = json.dumps(dict(data), ensure_ascii=False)
        except (KeyError, TypeError, ValueError):
            log.warning("store: a world object is not storable, not stored")
            return
        self._enqueue(_Op("world_object", (name, text, self._clock())))

    def delete_world_object(self, name: str) -> None:
        self._enqueue(_Op("world_delete", (str(name),)))

    def put_world_room(self, name: str, first_seen: float, last_visit: float) -> None:
        try:
            row = (str(name), float(first_seen), float(last_visit))
        except (TypeError, ValueError):
            log.warning("store: a world room has no usable times, not stored")
            return
        self._enqueue(_Op("world_room", row))

    def put_world_room_aliases(self, name: str, aliases: Iterable[str]) -> None:
        try:
            row = (json.dumps([str(a) for a in aliases], ensure_ascii=False), str(name))
        except (TypeError, ValueError):
            log.warning("store: room aliases are not storable, not stored")
            return
        self._enqueue(_Op("world_room_aliases", row))

    def put_fact(self, data: Mapping[str, Any]) -> None:
        """Queue one fact (`Fact.to_dict()`), replacing the stored one with the same id."""
        try:
            row = (
                str(data["id"]), str(data["text"]), data.get("subject"), str(data["kind"]),
                str(data["source"]), float(data["created"]), float(data["updated"]),
                float(data["confidence"]),
            )
        except (KeyError, TypeError, ValueError):
            log.warning("store: a fact is not storable, not stored")
            return
        self._enqueue(_Op("fact_put", row))

    def delete_fact(self, fact_id: str) -> None:
        self._enqueue(_Op("fact_delete", (str(fact_id),)))

    def put_episode(self, data: Mapping[str, Any]) -> None:
        """Queue one episode (`Episode.to_dict()`)."""
        try:
            subjects = [str(s) for s in data.get("subjects", ())]
            search = "|" + "|".join(fold(s) for s in subjects) + "|" if subjects else ""
            row = (
                str(data["id"]), float(data["timestamp"]), str(data["kind"]), str(data["summary"]),
                json.dumps(subjects, ensure_ascii=False), search, data.get("location"),
            )
        except (KeyError, TypeError, ValueError):
            log.warning("store: an episode is not storable, not stored")
            return
        self._enqueue(_Op("episode_put", row))

    def vacuum(self) -> None:
        """Shrink the file after large prunes. Only runs when asked; it rewrites the whole database."""
        self._enqueue(_Op("vacuum"))

    def _enqueue(self, op: _Op) -> None:
        """Queue a write. Observation-type writes and world writes have separate limits.

        A full observation queue drops its oldest write (observations are a log, and the
        newest ones matter most). World writes are never dropped to make room for
        observations; only a world queue that is full too drops its oldest world write,
        which needs thousands of unwritten changes and is counted separately.
        """
        with self._cond:
            if self._closed or self._conn is None:
                return
            world = op.kind in PRIORITY_KINDS
            if world and self._queued_world >= self.world_queue_limit:
                self._drop_oldest(world=True)
            elif not world and self._queued_data >= self.queue_limit:
                self._drop_oldest(world=False)
            self._queue.append(op)
            if world:
                self._queued_world += 1
            else:
                self._queued_data += 1
            self._cond.notify_all()

    def _drop_oldest(self, *, world: bool) -> None:
        """Make room in one queue: the oldest observation write goes first, else the oldest of the other kinds."""
        groups = (PRIORITY_KINDS,) if world else (("observation",), ("kv", "vacuum"))
        for kinds in groups:
            for index, queued in enumerate(self._queue):
                if queued.kind in kinds:
                    del self._queue[index]
                    self._forget(queued)
                    if world:
                        self._world_dropped += 1
                        dropped = self._world_dropped
                    else:
                        self._dropped += 1
                        dropped = self._dropped
                    if dropped in (1, 10, 100) or dropped % 1000 == 0:
                        log.warning("the store write queue is full: %d writes dropped so far", dropped)
                    return

    def _forget(self, op: _Op) -> None:
        """Account for an op that left the queue (written or dropped). Caller holds `_cond`."""
        if op.kind in PRIORITY_KINDS:
            self._queued_world -= 1
        elif op.kind != "flush":
            self._queued_data -= 1

    def flush(self, timeout: float = 2.0) -> bool:
        """Wait until everything queued so far is written. False on timeout (or no store)."""
        marker = _Op("flush", done=threading.Event())
        with self._cond:
            if self._closed or self._conn is None:
                return False
            self._queue.append(marker)
            self._cond.notify_all()
        assert marker.done is not None
        return marker.done.wait(max(0.0, timeout))

    def close(self, timeout: float = CLOSE_TIMEOUT_S) -> None:
        """Write what is queued (within `timeout`), then close. Idempotent."""
        self.flush(timeout)
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._cond.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                log.warning("the store writer is stuck; leaving it")
                return
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.execute("PRAGMA optimize")
                conn.close()
            except sqlite3.Error:
                log.debug("store close failed", exc_info=True)

    # -- the writer thread -----------------------------------------------------------------

    def _run(self) -> None:
        self._guarded(self._retention)
        while True:
            with self._cond:
                if not self._queue and not self._closed:
                    self._cond.wait(self.maintenance_interval_s)
                if self._closed and not self._queue:
                    return
                op = self._queue.popleft() if self._queue else None
                if op is not None:
                    self._forget(op)
            if op is not None:
                self._apply(op)
                if op.done is not None:
                    op.done.set()
            self._maintain()

    def _apply(self, op: _Op) -> None:
        if op.kind != "flush" and self._guarded(lambda: self._write(op)):
            with self._cond:
                self._written += 1

    def _write(self, op: _Op) -> None:
        conn = self._conn
        assert conn is not None
        if op.kind == "observation":
            conn.execute(
                f"INSERT OR REPLACE INTO observations ({_COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                op.args,
            )
            self._writes_since_retention += 1
        elif op.kind == "kv":
            conn.execute(
                "INSERT OR REPLACE INTO kv (namespace, key, value, updated) VALUES (?,?,?,?)", op.args
            )
        elif op.kind == "vacuum":
            conn.execute("VACUUM")
        elif op.kind == "world_object":
            # Replacing the row (not updating it) moves it to the end of rowid order: the model's LRU order.
            conn.execute("INSERT OR REPLACE INTO world_objects (name, data, updated) VALUES (?,?,?)", op.args)
        elif op.kind == "world_delete":
            conn.execute("DELETE FROM world_objects WHERE name = ?", op.args)
        elif op.kind == "world_room":
            conn.execute(
                "INSERT INTO world_rooms (name, first_seen, last_visit) VALUES (?,?,?) "
                "ON CONFLICT (name) DO UPDATE SET last_visit = excluded.last_visit",
                op.args,
            )
        elif op.kind == "world_room_aliases":
            conn.execute("UPDATE world_rooms SET aliases = ? WHERE name = ?", op.args)
        elif op.kind in MEMORY_KINDS:
            self._write_memory(op)

    def _write_memory(self, op: _Op) -> None:
        conn = self._conn
        assert conn is not None
        if op.kind == "fact_put":
            conn.execute(
                "INSERT OR REPLACE INTO facts "
                "(id, text, subject, kind, source, created, updated, confidence) VALUES (?,?,?,?,?,?,?,?)",
                op.args,
            )
        elif op.kind == "fact_delete":
            conn.execute("DELETE FROM facts WHERE id = ?", op.args)
        elif op.kind == "episode_put":
            conn.execute(
                "INSERT OR REPLACE INTO episodes "
                "(id, timestamp, kind, summary, subjects, search, location) VALUES (?,?,?,?,?,?,?)",
                op.args,
            )
            self._writes_since_retention += 1

    def _guarded(self, action: Callable[[], object]) -> bool:
        """Run storage work; a failure (disk full, I/O error) is counted and logged, never raised."""
        try:
            action()
            return True
        except Exception as error:  # noqa: BLE001 - the store must never take Laika down
            with self._cond:
                self._failed += 1
                failed = self._failed
            if failed <= MAX_LOGGED_FAILURES or failed % 100 == 0:
                log.error("store write failed (%d so far): %s", failed, error)
            return False

    def _maintain(self) -> None:
        due = time.monotonic() - self._last_maintenance >= self.maintenance_interval_s
        if self._writes_since_retention >= self.retention_every or due:
            self._guarded(self._retention)
        if due:
            self._last_maintenance = time.monotonic()
            self._guarded(lambda: self._conn is not None and self._conn.execute("PRAGMA optimize"))

    def _retention(self) -> None:
        """Keep at most `max_observations` rows, none older than `max_observation_days`;
        the same for episodes with `max_episodes` and `max_episode_days`. Facts are never pruned here.

        Image files are not touched here: `ImageStore` prunes them on its own limits, so a
        row may point to an image that is gone. Readers treat `image_path` as a hint only.
        """
        conn = self._conn
        if conn is None:
            return
        self._writes_since_retention = 0
        if self.max_observation_days > 0:
            cutoff = self._clock() - self.max_observation_days * DAY_S
            conn.execute("DELETE FROM observations WHERE timestamp < ?", (cutoff,))
        conn.execute(
            "DELETE FROM observations WHERE id IN (SELECT id FROM observations "
            "ORDER BY timestamp DESC, rowid DESC LIMIT -1 OFFSET ?)",
            (self.max_observations,),
        )
        if self.max_episode_days > 0:
            conn.execute(
                "DELETE FROM episodes WHERE timestamp < ?", (self._clock() - self.max_episode_days * DAY_S,)
            )
        conn.execute(
            "DELETE FROM episodes WHERE id IN (SELECT id FROM episodes "
            "ORDER BY timestamp DESC, rowid DESC LIMIT -1 OFFSET ?)",
            (self.max_episodes,),
        )

    # -- reading (own short-lived connection, never waits for the queue) --------------------

    def _read(self, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        if not self.available:
            return []
        try:
            conn = sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None)
            try:
                return conn.execute(sql, params).fetchall()
            finally:
                conn.close()
        except (sqlite3.Error, OSError) as error:
            log.warning("store read failed: %s", error)
            return []

    def _observations(self, sql: str, params: tuple[Any, ...]) -> list[Observation]:
        found = []
        for row in self._read(sql, params):
            try:
                found.append(_row_to_observation(row))
            except (InvalidObservation, ValueError, TypeError):
                log.warning("store: skipping an unreadable observation row %s", row[0])
        return found

    def recent_observations(self, limit: int = 20, location: str | None = None) -> list[Observation]:
        """The newest observations, newest first, optionally only from one room."""
        limit = max(0, int(limit))
        if location is None:
            return self._observations(
                f"SELECT {_COLUMNS} FROM observations ORDER BY timestamp DESC, rowid DESC LIMIT ?",
                (limit,),
            )
        return self._observations(
            f"SELECT {_COLUMNS} FROM observations WHERE location = ? "
            "ORDER BY timestamp DESC, rowid DESC LIMIT ?",
            (normalise(location), limit),
        )

    def observations_with(self, object_name: str, limit: int = 20) -> list[Observation]:
        """The newest observations that list `object_name`, newest first."""
        return self._observations(
            f"SELECT {_COLUMNS} FROM observations o WHERE EXISTS "
            f"(SELECT 1 FROM json_each(o.objects) j WHERE {_NAME_OF} = ?) "
            "ORDER BY timestamp DESC, rowid DESC LIMIT ?",
            (normalise(object_name), max(0, int(limit))),
        )

    def last_seen(self, object_name: str) -> Observation | None:
        found = self.observations_with(object_name, 1)
        return found[0] if found else None

    def count_observations(self) -> int:
        rows = self._read("SELECT COUNT(*) FROM observations")
        return int(rows[0][0]) if rows else 0

    def kv_get(self, namespace: str, key: str, default: Any = None) -> Any:
        rows = self._read("SELECT value FROM kv WHERE namespace = ? AND key = ?", (namespace, key))
        return _loads(rows[0][0], default) if rows else default

    def kv_items(self, namespace: str) -> dict[str, Any]:
        rows = self._read("SELECT key, value FROM kv WHERE namespace = ? ORDER BY key", (namespace,))
        return {key: _loads(value, None) for key, value in rows}

    def load_facts(self, limit: int = 500) -> list[dict[str, Any]]:
        """The stored facts, the most recently updated `limit` of them, oldest first."""
        rows = self._read(
            "SELECT id, text, subject, kind, source, created, updated, confidence FROM facts "
            "ORDER BY updated DESC, rowid DESC LIMIT ?",
            (max(0, int(limit)),),
        )
        keys = ("id", "text", "subject", "kind", "source", "created", "updated", "confidence")
        return [dict(zip(keys, row)) for row in reversed(rows)]

    def recent_episodes(self, limit: int = 200) -> list[dict[str, Any]]:
        """The newest episodes, oldest first."""
        return self._episodes("", (), limit)[::-1]

    def episodes_about(self, subject: str, limit: int = 10) -> list[dict[str, Any]]:
        """The newest episodes whose subjects contain `subject` (folded substring), newest first."""
        needle = fold(subject)
        if not needle:
            return []
        return self._episodes("WHERE instr(search, ?) > 0", (needle,), limit)

    def episodes_since(self, since: float, kinds: Iterable[str], limit: int = 20) -> list[dict[str, Any]]:
        """The newest episodes of these kinds at or after `since`, newest first."""
        wanted = tuple(kinds)
        if not wanted:
            return []
        marks = ",".join("?" for _ in wanted)
        return self._episodes(f"WHERE timestamp >= ? AND kind IN ({marks})", (float(since), *wanted), limit)

    def count_episodes(self) -> int:
        rows = self._read("SELECT COUNT(*) FROM episodes")
        return int(rows[0][0]) if rows else 0

    def _episodes(self, where: str, params: tuple[Any, ...], limit: int) -> list[dict[str, Any]]:
        rows = self._read(
            "SELECT id, timestamp, kind, summary, subjects, location FROM episodes "
            f"{where} ORDER BY timestamp DESC, rowid DESC LIMIT ?",
            (*params, max(0, int(limit))),
        )
        return [
            {
                "id": i, "timestamp": t, "kind": k, "summary": summary,
                "subjects": _loads(subjects, []), "location": location,
            }
            for i, t, k, summary, subjects, location in rows
        ]

    def load_world(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """The stored world model: object documents (least recently seen first) and rooms (first seen first)."""
        objects: list[dict[str, Any]] = []
        for (text,) in self._read("SELECT data FROM world_objects ORDER BY rowid"):
            document = _loads(text, None)
            if isinstance(document, dict):
                objects.append(document)
            else:
                log.warning("store: skipping an unreadable world object")
        rooms = [
            {"name": name, "first_seen": first, "last_visit": last, "aliases": _loads(aliases, [])}
            for name, first, last, aliases in self._read(
                "SELECT name, first_seen, last_visit, aliases FROM world_rooms ORDER BY rowid"
            )
        ]
        return objects, rooms


def _loads(text: str, default: Any) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return default


def _row_to_observation(row: tuple[Any, ...]) -> Observation:
    obs_id, timestamp, location, description, objects, people, confidence, pan, tilt, image, source, _ = row
    return Observation.from_dict(
        {
            "id": obs_id,
            "timestamp": timestamp,
            "location": location,
            "description": description,
            "objects": json.loads(objects),
            "confidence": confidence,
            "people": people,
            "cameraPan": pan,
            "cameraTilt": tilt,
            "imagePath": image,
            "source": source,
        }
    )


class ObservationArchiver:
    """Event subscriber that queues every `ObservationCreated` for the store (cheap, never blocks)."""

    def __init__(self, store: Store, bus: EventBus) -> None:
        self.store = store
        self._unsubscribe = bus.subscribe(self._on_event, EventType.OBSERVATION_CREATED)

    def _on_event(self, event: Event) -> None:
        try:
            data = dict(event.data)
            interest = data.pop("interest", None)
            self.store.add_observation(Observation.from_dict(data), interest)
        except (InvalidObservation, ValueError, TypeError):
            log.warning("could not archive an observation event", exc_info=True)

    def close(self) -> None:
        self._unsubscribe()


def seed_observation_log(store: Store, observations: ObservationLog, limit: int = LOG_SIZE) -> int:
    """Fill the in-memory log with the newest stored observations (oldest first), so a restart remembers."""
    found = store.recent_observations(limit)
    for observation in reversed(found):
        observations.add(observation)
    return len(found)
