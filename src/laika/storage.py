"""Where Laika keeps files on the Pi, and how that storage stays bounded.

Everything lives under one data directory: `images/`, `logs/` and `memory/`.
Every writer here has a limit (file count, bytes, rotation or age), so a long
run cannot fill the SD card. Directories are created 0o700 and files 0o600;
on Windows `chmod` does nothing useful and that is fine for development.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Mapping

from laika.config import StorageConfig
from laika.events import Event
from laika.robot import ImageObservation

log = logging.getLogger(__name__)

DIR_MODE = 0o700
FILE_MODE = 0o600
MB = 1024 * 1024
MAX_TRANSCRIPT_CHARS = 2000
EVENT_LOG_NAME = "events.jsonl"
TRANSCRIPT_PATTERN = re.compile(r"^transcripts-(\d{4}-\d{2}-\d{2})\.jsonl$")


def resolve_data_dir(configured: str = "", environ: Mapping[str, str] | None = None) -> Path:
    """`$LAIKA_DATA_DIR`, else the configured path, else `$XDG_DATA_HOME/laika`, else `~/.local/share/laika`.

    Under sudo `~` is /root; the service sets LAIKA_DATA_DIR explicitly.
    """
    env = os.environ if environ is None else environ
    if env.get("LAIKA_DATA_DIR"):
        return Path(env["LAIKA_DATA_DIR"]).expanduser()
    if configured:
        return Path(configured).expanduser()
    if env.get("XDG_DATA_HOME"):
        return Path(env["XDG_DATA_HOME"]).expanduser() / "laika"
    return Path.home() / ".local" / "share" / "laika"


def _chmod(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)  # not subject to the umask, unlike mkdir(mode=)
    except OSError:
        pass  # Windows, or a filesystem without modes


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @property
    def images(self) -> Path:
        return self.root / "images"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def memory(self) -> Path:
        return self.root / "memory"

    def ensure(self) -> DataPaths:
        """Create the directories (0o700). Raises OSError if the root cannot be created."""
        for directory in (self.root, self.images, self.logs, self.memory):
            directory.mkdir(parents=True, exist_ok=True)
            _chmod(directory, DIR_MODE)
        return self


def prune_directory(
    path: Path, max_files: int | None = None, max_bytes: int | None = None, pattern: str = "*"
) -> list[Path]:
    """Delete the oldest files (by mtime, then name) until within both limits. Returns what was removed."""
    entries: list[tuple[float, str, Path, int]] = []
    for file in path.glob(pattern):
        try:
            if file.is_file():
                stat = file.stat()
                entries.append((stat.st_mtime, file.name, file, stat.st_size))
        except OSError:
            continue
    entries.sort(key=lambda entry: (entry[0], entry[1]))
    count = len(entries)
    total = sum(entry[3] for entry in entries)
    removed: list[Path] = []
    for _, _, file, size in entries:
        too_many = max_files is not None and count > max(0, max_files)
        too_big = max_bytes is not None and total > max(0, max_bytes)
        if not (too_many or too_big):
            break
        try:
            file.unlink()
        except OSError:
            log.warning("could not delete %s", file)
            continue
        count -= 1
        total -= size
        removed.append(file)
    return removed


IMAGE_PATTERN = "img-*.jpg"
# Half-written images (a crash between write and rename) older than this are removed.
STALE_TEMP_S = 3600.0
# How often old pictures are deleted even when no new ones arrive.
IMAGE_CLEANUP_INTERVAL_S = 3600.0
DAY_S = 86400.0


def _delete_older_than(path: Path, pattern: str, cutoff: float) -> list[Path]:
    """Delete files matching `pattern` whose mtime is before `cutoff`. Returns what was removed."""
    removed = []
    for file in path.glob(pattern):
        try:
            if file.is_file() and file.stat().st_mtime < cutoff:
                file.unlink()
                removed.append(file)
        except OSError:
            log.warning("could not delete %s", file)
    return removed


class ImageStore:
    """Camera frames on disk, newest kept: at most `max_images` files and `max_mb` megabytes,
    and none older than `max_days` (0 = no age limit).

    Pruning runs on every save, and `start_cleanup()` also runs it at startup and every hour,
    so old pictures go away even when Laika takes no new ones.
    """

    def __init__(
        self,
        directory: Path,
        max_images: int = 200,
        max_mb: float = 100.0,
        max_days: float = 7.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.directory = directory
        self.max_images = max(1, max_images)
        self.max_bytes = int(max(0.0, max_mb) * MB)
        self.max_days = max(0.0, max_days)
        self._clock = clock
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._cleaner: threading.Thread | None = None

    def prune(self) -> list[Path]:
        """Delete pictures beyond the count, size and age limits, and stale temporary files."""
        with self._lock:
            return self._prune()

    def _prune(self) -> list[Path]:
        if not self.directory.exists():
            return []
        now = self._clock()
        removed = []
        if self.max_days:
            removed += _delete_older_than(self.directory, IMAGE_PATTERN, now - self.max_days * DAY_S)
        removed += _delete_older_than(self.directory, ".img-*.tmp", now - STALE_TEMP_S)
        removed += prune_directory(self.directory, self.max_images, self.max_bytes, IMAGE_PATTERN)
        return removed

    def start_cleanup(self, interval_s: float = IMAGE_CLEANUP_INTERVAL_S) -> None:
        """Prune now, then every `interval_s` on a small daemon thread, until `close()`."""
        self._run_cleanup()
        if self._cleaner is None:
            self._cleaner = threading.Thread(
                target=self._cleanup_loop, args=(interval_s,), name="laika-image-cleanup", daemon=True
            )
            self._cleaner.start()

    def _cleanup_loop(self, interval_s: float) -> None:
        while not self._stop.wait(max(1.0, interval_s)):
            self._run_cleanup()

    def _run_cleanup(self) -> None:
        try:
            removed = self.prune()
        except Exception:  # noqa: BLE001 - cleaning up must never break Laika
            log.warning("could not clean up old images", exc_info=True)
            return
        if removed:
            log.info("deleted %d old image(s)", len(removed))

    def close(self) -> None:
        self._stop.set()
        if self._cleaner is not None:
            self._cleaner.join(timeout=2)

    def save(self, observation: ImageObservation) -> Path | None:
        """Write the JPEG and prune. None when there is nothing to save or it did not fit."""
        if not observation.jpeg:
            return None
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            timestamp = observation.timestamp
            while True:
                moment = datetime.fromtimestamp(timestamp)
                target = self.directory / f"img-{moment:%Y%m%d-%H%M%S}-{moment.microsecond:06d}.jpg"
                if not target.exists():
                    break
                timestamp += 1e-6
            temp = self.directory / f".{target.name}.tmp"
            try:
                temp.write_bytes(observation.jpeg)
                _chmod(temp, FILE_MODE)
                os.replace(temp, target)
            except OSError:
                log.warning("could not save image %s", target, exc_info=True)
                temp.unlink(missing_ok=True)
                return None
            self._prune()
            return target if target.exists() else None


class _PrivateRotatingFileHandler(RotatingFileHandler):
    def _open(self):  # type: ignore[override]
        stream = super()._open()
        _chmod(Path(self.baseFilename), FILE_MODE)
        return stream


class EventFileLogger:
    """Event subscriber writing one JSON line per event to `events.jsonl`, rotated by size.

    Disk use stays under `max_mb * (backups + 1)` plus at most one line.
    """

    def __init__(self, directory: Path, max_mb: float = 5.0, backups: int = 3) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / EVENT_LOG_NAME
        self._handler = _PrivateRotatingFileHandler(
            self.path,
            maxBytes=max(1024, int(max_mb * MB)),
            backupCount=max(0, backups),
            encoding="utf-8",
            delay=True,
        )
        self._handler.setFormatter(logging.Formatter("%(message)s"))
        # Not registered with `logging`: it must never reach the root logger or the console.
        self._logger = logging.Logger("laika.event-file", logging.INFO)
        self._logger.addHandler(self._handler)
        self._logger.propagate = False

    def __call__(self, event: Event) -> None:
        self._logger.info(event.to_json())

    def close(self) -> None:
        self._logger.removeHandler(self._handler)
        self._handler.close()


class TranscriptLog:
    """Optional record of what was said, one JSONL file per day. Off by default (privacy).

    Files older than `max_days` are deleted at startup and whenever a new day file starts.
    """

    def __init__(
        self,
        directory: Path,
        enabled: bool = False,
        max_days: int = 7,
        today: Callable[[], date] = date.today,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.directory = directory
        self.enabled = enabled
        self.max_days = max(1, max_days)
        self._today = today
        self._now = now
        self._current: date | None = None
        self._lock = threading.Lock()
        if enabled:
            self._prune()

    def record(self, role: str, text: str) -> None:
        if not self.enabled or not text:
            return
        try:
            with self._lock:
                day = self._today()
                self.directory.mkdir(parents=True, exist_ok=True)
                if day != self._current:
                    self._current = day
                    self._prune()
                line = json.dumps(
                    {"time": self._now(), "role": role, "text": text[:MAX_TRANSCRIPT_CHARS]},
                    ensure_ascii=False,
                )
                path = self.directory / f"transcripts-{day.isoformat()}.jsonl"
                with path.open("a", encoding="utf-8") as file:
                    file.write(line + "\n")
                _chmod(path, FILE_MODE)
        except OSError:
            log.warning("could not write transcript", exc_info=True)

    def _prune(self) -> None:
        today = self._today()
        try:
            files = list(self.directory.iterdir())
        except OSError:
            return
        for file in files:
            match = TRANSCRIPT_PATTERN.match(file.name)
            if not match:
                continue
            try:
                age_days = (today - date.fromisoformat(match.group(1))).days
                if age_days >= self.max_days:
                    file.unlink()
            except (ValueError, OSError):
                continue


def attach_storage(storage: StorageConfig, paths: DataPaths) -> tuple[EventFileLogger, TranscriptLog]:
    """Build the event logger and transcript log from the `[storage]` settings."""
    return (
        EventFileLogger(paths.logs, storage.log_max_mb, storage.log_backups),
        TranscriptLog(
            paths.logs, enabled=storage.keep_transcripts, max_days=storage.max_transcript_days
        ),
    )
