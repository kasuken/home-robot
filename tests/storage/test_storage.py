from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from datetime import date, timedelta
from pathlib import Path

from laika.config import StorageConfig, load_config, permission_warning
from laika.events import Event, EventType
from laika.robot import ImageObservation
from laika.storage import (
    DataPaths,
    EventFileLogger,
    ImageStore,
    TranscriptLog,
    attach_storage,
    prune_directory,
    resolve_data_dir,
)

POSIX = os.name == "posix"


def total_size(directory: Path) -> int:
    return sum(f.stat().st_size for f in directory.iterdir() if f.is_file())


def frame(size: int = 100, timestamp: float = 1_700_000_000.0) -> ImageObservation:
    return ImageObservation(
        jpeg=b"x" * size, width=1, height=1, camera_pan=0, camera_tilt=0, timestamp=timestamp
    )


class TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)


class ResolveDataDirTests(unittest.TestCase):
    def test_env_wins_over_config_and_xdg(self):
        env = {"LAIKA_DATA_DIR": "/srv/laika", "XDG_DATA_HOME": "/xdg"}
        self.assertEqual(resolve_data_dir("/cfg", env), Path("/srv/laika"))

    def test_config_then_xdg_then_home(self):
        self.assertEqual(resolve_data_dir("/cfg", {"XDG_DATA_HOME": "/xdg"}), Path("/cfg"))
        self.assertEqual(resolve_data_dir("", {"XDG_DATA_HOME": "/xdg"}), Path("/xdg") / "laika")
        self.assertEqual(resolve_data_dir("", {}), Path.home() / ".local" / "share" / "laika")


class DataPathsTests(TempDirCase):
    def test_creates_subfolders(self):
        paths = DataPaths(self.tmp / "data").ensure()
        for folder in (paths.root, paths.images, paths.logs, paths.memory):
            self.assertTrue(folder.is_dir())
        paths.ensure()  # idempotent

    @unittest.skipUnless(POSIX, "POSIX permissions")
    def test_directories_are_private(self):
        paths = DataPaths(self.tmp / "data").ensure()
        for folder in (paths.root, paths.images, paths.logs, paths.memory):
            self.assertEqual(folder.stat().st_mode & 0o777, 0o700)


class PruneTests(TempDirCase):
    def make(self, count: int, size: int = 10) -> list[Path]:
        files = []
        for i in range(count):
            f = self.tmp / f"f{i:03d}.bin"
            f.write_bytes(b"x" * size)
            os.utime(f, (1000 + i, 1000 + i))
            files.append(f)
        return files

    def test_prune_by_count_removes_oldest(self):
        files = self.make(5)
        removed = prune_directory(self.tmp, max_files=2)
        self.assertEqual(removed, files[:3])
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()), ["f003.bin", "f004.bin"])

    def test_prune_by_size_removes_oldest(self):
        files = self.make(5, size=10)
        removed = prune_directory(self.tmp, max_bytes=25)
        self.assertEqual(removed, files[:3])
        self.assertLessEqual(total_size(self.tmp), 25)

    def test_both_limits_and_no_limits(self):
        self.make(5)
        self.assertEqual(prune_directory(self.tmp), [])
        prune_directory(self.tmp, max_files=4, max_bytes=20)
        self.assertEqual(len(list(self.tmp.iterdir())), 2)

    def test_pattern_leaves_other_files(self):
        self.make(3)
        (self.tmp / "keep.txt").write_text("hi")
        prune_directory(self.tmp, max_files=0, pattern="*.bin")
        self.assertEqual([p.name for p in self.tmp.iterdir()], ["keep.txt"])

    def test_missing_directory_is_fine(self):
        self.assertEqual(prune_directory(self.tmp / "nope", max_files=1), [])


class ImageStoreTests(TempDirCase):
    def test_saves_timestamped_jpeg(self):
        store = ImageStore(self.tmp, max_images=5)
        path = store.save(frame())
        self.assertIsNotNone(path)
        self.assertTrue(path.name.startswith("img-") and path.suffix == ".jpg")
        self.assertEqual(path.read_bytes(), b"x" * 100)

    def test_no_jpeg_saves_nothing(self):
        obs = ImageObservation(jpeg=None, width=1, height=1, camera_pan=0, camera_tilt=0)
        self.assertIsNone(ImageStore(self.tmp).save(obs))
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_same_timestamp_does_not_overwrite(self):
        store = ImageStore(self.tmp, max_images=5)
        a, b = store.save(frame()), store.save(frame())
        self.assertNotEqual(a, b)
        self.assertEqual(len(list(self.tmp.iterdir())), 2)

    def test_long_run_stays_bounded_by_count(self):
        store = ImageStore(self.tmp, max_images=20, max_mb=100)
        last = None
        for i in range(1000):
            last = store.save(frame(50, 1_700_000_000.0 + i))
            self.assertLessEqual(len(list(self.tmp.iterdir())), 20)
        self.assertTrue(last.exists())
        self.assertEqual(len(list(self.tmp.iterdir())), 20)

    def age(self, path, days, now):
        os.utime(path, (now - days * 86400, now - days * 86400))

    def test_pictures_older_than_max_days_are_deleted(self):
        now = time.time()
        store = ImageStore(self.tmp, max_images=100, max_days=7, clock=lambda: now)
        old, recent = store.save(frame(10, 1.0)), store.save(frame(10, 2.0))
        self.age(old, 8, now)
        self.age(recent, 6, now)
        self.assertEqual(store.prune(), [old])
        self.assertFalse(old.exists())
        self.assertTrue(recent.exists())

    def test_no_age_limit_with_zero_days(self):
        now = time.time()
        store = ImageStore(self.tmp, max_images=100, max_days=0, clock=lambda: now)
        path = store.save(frame(10, 1.0))
        self.age(path, 400, now)
        self.assertEqual(store.prune(), [])
        self.assertTrue(path.exists())

    def test_cleanup_runs_at_start_without_any_new_picture(self):
        now = time.time()
        writer = ImageStore(self.tmp, max_images=100, max_days=0)
        paths = [writer.save(frame(10, float(i))) for i in range(3)]
        for path in paths:
            self.age(path, 30, now)
        store = ImageStore(self.tmp, max_images=100, max_days=7, clock=lambda: now)
        self.addCleanup(store.close)
        store.start_cleanup(interval_s=3600)
        self.assertEqual(list(self.tmp.glob("img-*.jpg")), [])

    def test_cleanup_repeats_on_its_interval(self):
        now = [time.time()]
        store = ImageStore(self.tmp, max_images=100, max_days=7, clock=lambda: now[0])
        self.addCleanup(store.close)
        path = store.save(frame(10, 1.0))
        self.age(path, 1, now[0])
        store.start_cleanup(interval_s=1)
        self.assertTrue(path.exists())  # one day old: kept
        now[0] += 7 * 86400  # a week later, with no new picture taken
        deadline = time.monotonic() + 5
        while path.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        self.assertFalse(path.exists())

    def test_stale_half_written_files_are_removed(self):
        now = time.time()
        store = ImageStore(self.tmp, clock=lambda: now)
        stale, fresh = self.tmp / ".img-1.jpg.tmp", self.tmp / ".img-2.jpg.tmp"
        for path in (stale, fresh):
            path.write_bytes(b"x")
        self.age(stale, 1, now)
        os.utime(fresh, (now, now))
        store.prune()
        self.assertFalse(stale.exists())
        self.assertTrue(fresh.exists())

    def test_long_run_stays_bounded_by_size(self):
        store = ImageStore(self.tmp, max_images=1000, max_mb=0.001)  # about 1048 bytes
        for i in range(300):
            store.save(frame(100, 1_700_000_000.0 + i))
        self.assertLessEqual(total_size(self.tmp), 1048)

    def test_oversized_image_is_not_kept(self):
        store = ImageStore(self.tmp, max_mb=0.0001)
        self.assertIsNone(store.save(frame(10_000)))
        self.assertEqual(list(self.tmp.iterdir()), [])

    @unittest.skipUnless(POSIX, "POSIX permissions")
    def test_files_are_private(self):
        path = ImageStore(self.tmp).save(frame())
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)


class EventFileLoggerTests(TempDirCase):
    def test_writes_one_json_line_per_event(self):
        logger = EventFileLogger(self.tmp, max_mb=1, backups=1)
        logger(Event(EventType.ROOM_ENTERED, {"room": "cucina"}))
        logger(Event(EventType.PERSON_LOST, {"name": "è"}))
        logger.close()
        lines = (self.tmp / "events.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(x)["type"] for x in lines], ["RoomEntered", "PersonLost"])

    def test_rotation_keeps_disk_use_bounded(self):
        logger = EventFileLogger(self.tmp, max_mb=0.001, backups=2)  # clamped to 1 KiB
        for i in range(2000):
            logger(Event(EventType.OBSTACLE_DETECTED, {"cm": i, "pad": "y" * 50}))
        logger.close()
        self.assertLessEqual(len(list(self.tmp.iterdir())), 3)
        self.assertLessEqual(total_size(self.tmp), 3 * 1024)

    @unittest.skipUnless(POSIX, "POSIX permissions")
    def test_log_is_private(self):
        logger = EventFileLogger(self.tmp)
        logger(Event(EventType.ROOM_ENTERED, {}))
        logger.close()
        self.assertEqual((self.tmp / "events.jsonl").stat().st_mode & 0o777, 0o600)


class TranscriptTests(TempDirCase):
    def test_disabled_by_default_writes_nothing(self):
        log = TranscriptLog(self.tmp)
        log.record("user", "ciao")
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_default_config_keeps_no_transcripts(self):
        paths = DataPaths(self.tmp / "d").ensure()
        event_file, transcript = attach_storage(StorageConfig(), paths)
        event_file.close()
        self.assertFalse(transcript.enabled)

    def test_enabled_appends_lines_per_day(self):
        day = [date(2026, 10, 15)]
        log = TranscriptLog(self.tmp, enabled=True, today=lambda: day[0])
        log.record("user", "ciao")
        log.record("assistant", "Ciao!")
        day[0] = date(2026, 10, 16)
        log.record("user", "x")
        first = (self.tmp / "transcripts-2026-10-15.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(x)["role"] for x in first], ["user", "assistant"])
        self.assertTrue((self.tmp / "transcripts-2026-10-16.jsonl").exists())

    def test_old_files_pruned_at_startup_and_on_new_day(self):
        today = date(2026, 10, 15)
        for age in (0, 3, 7, 30):
            name = f"transcripts-{(today - timedelta(days=age)).isoformat()}.jsonl"
            (self.tmp / name).write_text("{}")
        (self.tmp / "other.txt").write_text("keep")
        day = [today]
        log = TranscriptLog(self.tmp, enabled=True, max_days=7, today=lambda: day[0])
        self.assertEqual(
            sorted(p.name for p in self.tmp.iterdir()),
            ["other.txt", "transcripts-2026-10-12.jsonl", "transcripts-2026-10-15.jsonl"],
        )
        day[0] = today + timedelta(days=5)
        log.record("user", "ciao")
        names = sorted(p.name for p in self.tmp.iterdir())
        self.assertNotIn("transcripts-2026-10-12.jsonl", names)
        self.assertIn("transcripts-2026-10-20.jsonl", names)

    def test_long_text_is_cut(self):
        log = TranscriptLog(self.tmp, enabled=True)
        log.record("user", "a" * 10_000)
        file = next(self.tmp.iterdir())
        self.assertLess(len(json.loads(file.read_text(encoding="utf-8"))["text"]), 2001)


class ConfigTests(TempDirCase):
    def test_defaults(self):
        storage = load_config(self.tmp / "missing.toml").storage
        self.assertEqual(storage.data_dir, "")
        self.assertFalse(storage.keep_transcripts)
        self.assertEqual((storage.max_images, storage.log_backups), (200, 3))

    def test_parses_storage_section(self):
        path = self.tmp / "laika.toml"
        path.write_text(
            '[storage]\ndata_dir = "/var/lib/laika"\nmax_images = 10\nmax_images_mb = 2.5\n'
            "log_max_mb = 1\nlog_backups = 1\nkeep_transcripts = true\nmax_transcript_days = 2\n",
            encoding="utf-8",
        )
        s = load_config(path).storage
        self.assertEqual((s.data_dir, s.max_images, s.max_images_mb), ("/var/lib/laika", 10, 2.5))
        self.assertEqual(
            (s.log_max_mb, s.log_backups, s.keep_transcripts, s.max_transcript_days),
            (1, 1, True, 2),
        )

    def test_unknown_storage_key_is_rejected(self):
        path = self.tmp / "laika.toml"
        path.write_text("[storage]\nbogus = 1\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_config(path)

    def test_example_config_loads(self):
        example = Path(__file__).resolve().parents[2] / "config" / "laika.example.toml"
        self.assertEqual(load_config(example).storage.max_images, 200)

    @unittest.skipUnless(POSIX, "POSIX permissions")
    def test_world_readable_config_warns(self):
        path = self.tmp / "laika.toml"
        path.write_text("")
        os.chmod(path, 0o644)
        self.assertIn("chmod 600", permission_warning(path))
        os.chmod(path, 0o600)
        self.assertIsNone(permission_warning(path))

    def test_missing_config_does_not_warn(self):
        self.assertIsNone(permission_warning(self.tmp / "nope.toml"))


if __name__ == "__main__":
    unittest.main()
