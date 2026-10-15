from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path

from laika.config import StorageConfig, validate_storage
from laika.events import EventBus, EventType
from laika.observations import Observation, ObservationLog, SeenObject
from laika.safety import SafetyController
from laika.store import DB_NAME, MIGRATIONS, SCHEMA_VERSION, ObservationArchiver, Store, seed_observation_log
from tests.fakes import RecordingRobot

POSIX = os.name == "posix"
NOW = 1_800_000_000.0
DAY = 86400.0


def obs(n: int = 0, *, objects=("cup",), location="kitchen", timestamp=None, **kwargs) -> Observation:
    return Observation(
        description=f"I see things {n}",
        objects=tuple(o if isinstance(o, SeenObject) else SeenObject(o) for o in objects),
        confidence=0.9,
        location=location,
        timestamp=NOW - 1000 + n if timestamp is None else timestamp,
        id=f"obs-{n:08x}",
        **kwargs,
    )


class StoreCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name) / "memory"
        self.path = self.dir / DB_NAME
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def open(self, **kwargs) -> Store:
        kwargs.setdefault("clock", lambda: NOW)
        store = Store(self.path, **kwargs)
        self.addCleanup(store.close)
        return store


class SchemaTests(StoreCase):
    def tables(self) -> set[str]:
        conn = sqlite3.connect(self.path)
        try:
            return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            conn.close()

    def test_migrations_create_the_schema_and_are_idempotent(self):
        store = self.open()
        store.close()
        self.assertTrue({"observations", "kv", "schema_version"} <= self.tables())
        for _ in range(2):
            self.open().close()
        conn = sqlite3.connect(self.path)
        versions = [r[0] for r in conn.execute("SELECT version FROM schema_version")]
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        self.assertEqual(versions, list(range(1, SCHEMA_VERSION + 1)))
        self.assertEqual(mode, "wal")
        self.assertEqual(SCHEMA_VERSION, len(MIGRATIONS))

    def test_a_newer_database_is_left_alone(self):
        self.open().close()
        conn = sqlite3.connect(self.path)
        conn.execute("INSERT INTO schema_version VALUES (99)")
        conn.commit()
        conn.close()
        store = self.open()
        self.assertFalse(store.available)
        store.add_observation(obs())  # no store: silently ignored
        self.assertEqual(store.recent_observations(), [])
        self.assertTrue(self.path.exists())

    @unittest.skipUnless(POSIX, "POSIX permissions")
    def test_the_file_is_private(self):
        self.open()
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)


class RoundTripTests(StoreCase):
    def test_observation_survives_a_restart_unchanged(self):
        rich = obs(
            1,
            objects=("cup", SeenObject("book", 0.4, "left", "blue")),
            people=2,
            camera_pan=15.0,
            camera_tilt=-5.0,
            image_path="/data/images/x.jpg",
            source="simulator",
        )
        store = self.open()
        store.add_observation(rich, {"total": 0.7, "reasons": ["new"]})
        store.close()
        again = self.open()
        self.assertEqual(again.recent_observations(5), [rich])
        self.assertEqual(again.count_observations(), 1)

    def test_kv_round_trip_and_replace(self):
        store = self.open()
        store.kv_put("world", "kitchen", {"objects": ["cup"]})
        store.kv_put("world", "kitchen", {"objects": ["cup", "pan"]})
        store.kv_put("other", "x", 1)
        store.close()
        again = self.open()
        self.assertEqual(again.kv_get("world", "kitchen"), {"objects": ["cup", "pan"]})
        self.assertEqual(again.kv_items("world"), {"kitchen": {"objects": ["cup", "pan"]}})
        self.assertEqual(again.kv_get("world", "missing", "d"), "d")

    def test_queries(self):
        store = self.open()
        store.add_observation(obs(1, objects=("cup", "book"), location="kitchen"))
        store.add_observation(obs(2, objects=(SeenObject("cup", 0.5),), location="salotto"))
        store.add_observation(obs(3, objects=("keys",), location="kitchen"))
        self.assertTrue(store.flush())
        self.assertEqual([o.id for o in store.recent_observations(10)], [obs(3).id, obs(2).id, obs(1).id])
        self.assertEqual([o.id for o in store.recent_observations(10, "Kitchen")], [obs(3).id, obs(1).id])
        self.assertEqual(len(store.recent_observations(1)), 1)
        self.assertEqual([o.id for o in store.observations_with("Cup", 10)], [obs(2).id, obs(1).id])
        self.assertEqual(store.last_seen("cup").id, obs(2).id)
        self.assertIsNone(store.last_seen("sofa"))
        self.assertEqual(store.observations_with("cu"), [])
        # Injection attempts are just names.
        self.assertEqual(store.observations_with("x' OR '1'='1"), [])
        self.assertEqual(store.count_observations(), 3)


class RetentionTests(StoreCase):
    def test_keeps_the_newest_by_count(self):
        store = self.open(max_observations=3, retention_every=1)
        for n in range(6):
            store.add_observation(obs(n))
        store.flush()
        self.assertEqual([o.id for o in store.recent_observations(10)], [obs(5).id, obs(4).id, obs(3).id])

    def test_drops_rows_older_than_the_limit(self):
        store = self.open(max_observation_days=10, retention_every=1)
        store.add_observation(obs(1, timestamp=NOW - 11 * DAY))
        store.add_observation(obs(2, timestamp=NOW - 9 * DAY))
        store.flush()
        self.assertEqual([o.id for o in store.recent_observations(10)], [obs(2).id])

    def test_retention_runs_at_startup(self):
        first = self.open(max_observations=100)
        for n in range(5):
            first.add_observation(obs(n))
        first.close()
        second = self.open(max_observations=2)
        second.flush()
        self.assertEqual(second.count_observations(), 2)

    def test_a_row_whose_image_is_gone_still_reads(self):
        store = self.open()
        store.add_observation(obs(1, image_path=str(self.dir / "pruned.jpg")))
        store.flush()
        self.assertEqual(store.recent_observations(1)[0].image_path, str(self.dir / "pruned.jpg"))

    def test_config_limits_are_validated(self):
        validate_storage(StorageConfig())
        for bad in (dict(max_observations=0), dict(max_observation_days=0), dict(max_observations=1.5)):
            with self.assertRaises(ValueError):
                validate_storage(StorageConfig(**bad))


class BlockedWriterTests(StoreCase):
    def blocked(self, **kwargs):
        store = self.open(**kwargs)
        release = threading.Event()
        self.addCleanup(release.set)
        entered = threading.Event()
        original = store._write

        def slow(op):
            entered.set()
            release.wait(10)
            original(op)

        store._write = slow
        store.add_observation(obs(0))
        self.assertTrue(entered.wait(5))
        return store, release

    def test_the_queue_is_bounded_and_callers_never_block(self):
        store, release = self.blocked(queue_limit=8)
        started = time.monotonic()
        for n in range(1, 101):
            store.add_observation(obs(n))
            store.kv_put("ns", "k", n)
        self.assertLess(time.monotonic() - started, 2.0)
        stats = store.stats()
        self.assertLessEqual(stats.queued, 8)
        self.assertEqual(stats.dropped, 200 - 8)
        release.set()
        self.assertTrue(store.flush(5))

    def test_the_oldest_observation_is_dropped_first(self):
        store, release = self.blocked(queue_limit=3)
        for n in range(1, 6):
            store.add_observation(obs(n))
        release.set()
        store.flush(5)
        ids = {o.id for o in store.recent_observations(10)}
        self.assertEqual(ids, {obs(n).id for n in (0, 3, 4, 5)})
        self.assertEqual(store.stats().dropped, 2)

    def test_reads_do_not_wait_for_the_writer(self):
        store, _ = self.blocked()
        started = time.monotonic()
        self.assertEqual(store.count_observations(), 0)
        self.assertLess(time.monotonic() - started, 1.0)

    def test_flush_times_out_when_stuck(self):
        store, _ = self.blocked()
        self.assertFalse(store.flush(0.1))


class FailureTests(StoreCase):
    def test_a_failing_write_is_counted_not_raised(self):
        store = self.open()

        def broken(op):
            raise sqlite3.OperationalError("database or disk is full")

        store._write = broken
        store.add_observation(obs(1))
        store.kv_put("a", "b", 1)
        self.assertTrue(store.flush())
        self.assertEqual(store.stats().failed, 2)
        store.add_observation(obs(2))  # still running
        self.assertTrue(store.flush())
        self.assertEqual(store.stats().failed, 3)

    def test_closed_store_ignores_everything(self):
        store = self.open()
        store.close()
        store.close()
        store.add_observation(obs(1))
        store.kv_put("a", "b", 1)
        self.assertFalse(store.flush(0.1))
        self.assertEqual(store.recent_observations(), [])
        self.assertEqual(store.count_observations(), 0)
        self.assertIsNone(store.kv_get("a", "b"))

    def test_an_unusable_location_leaves_laika_running(self):
        blocker = Path(self.dir.parent) / "file"
        blocker.write_text("not a directory")
        store = Store(blocker / "memory" / DB_NAME)
        self.assertFalse(store.available)
        store.add_observation(obs(1))
        self.assertEqual(store.recent_observations(), [])
        store.close()

    def test_a_corrupt_file_is_moved_aside_and_a_fresh_one_works(self):
        self.dir.mkdir(parents=True)
        self.path.write_bytes(b"this is definitely not a sqlite database" * 100)
        store = self.open()
        self.assertTrue(store.available)
        store.add_observation(obs(1))
        store.flush()
        self.assertEqual(store.count_observations(), 1)
        moved = [p for p in self.dir.iterdir() if ".corrupt-" in p.name]
        self.assertEqual(len(moved), 1)
        self.assertTrue(moved[0].read_bytes().startswith(b"this is"))


class ArchiverTests(StoreCase):
    def test_observation_events_reach_the_store_with_their_interest(self):
        store = self.open()
        bus = EventBus()
        bus.start()
        self.addCleanup(bus.close)
        ObservationArchiver(store, bus)
        observation = obs(1, objects=("cup", SeenObject("pan", 0.5)))
        data = observation.to_dict()
        data["interest"] = {"total": 0.8, "components": {}, "reasons": ["new"]}
        bus.emit(EventType.OBSERVATION_CREATED, **data)
        bus.emit(EventType.ROOM_ENTERED, room="kitchen")
        bus.emit(EventType.OBSERVATION_CREATED, description="")  # malformed: ignored
        self.assertTrue(bus.drain())
        self.assertTrue(store.flush())
        [stored] = store.recent_observations(5)
        self.assertEqual(stored.id, observation.id)
        self.assertEqual(stored.object_names, ["cup", "pan"])
        self.assertEqual(stored.location, "kitchen")
        conn = sqlite3.connect(self.path)
        self.assertIn('"total": 0.8', conn.execute("SELECT interest FROM observations").fetchone()[0])
        conn.close()

    def test_the_log_is_seeded_after_a_restart(self):
        first = self.open()
        for n in range(5):
            first.add_observation(obs(n))
        first.close()
        store = self.open()
        log = ObservationLog()
        self.assertEqual(seed_observation_log(store, log, 3), 3)
        self.assertEqual([o.id for o in log.recent(10)], [obs(4).id, obs(3).id, obs(2).id])
        self.assertEqual(log.last().id, obs(4).id)


class SafetyIsolationTests(StoreCase):
    def test_a_blocked_store_does_not_slow_the_safety_ticks(self):
        store = self.open(queue_limit=4)
        release = threading.Event()
        self.addCleanup(release.set)
        store._write = lambda op: release.wait(10)
        bus = EventBus()
        bus.start()
        self.addCleanup(bus.close)
        ObservationArchiver(store, bus)
        robot = RecordingRobot(distance_cm=100)
        controller = SafetyController(robot, speed=25)
        self.addCleanup(controller.close)
        controller.start()
        for n in range(50):
            bus.emit(EventType.OBSERVATION_CREATED, **obs(n).to_dict())
            time.sleep(0.01)
        time.sleep(0.5)
        self.assertLessEqual(controller.stats().mean_interval_s, 0.08)


if __name__ == "__main__":
    unittest.main()
