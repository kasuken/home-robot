"""The limits on what Laika does by herself: a driving budget and a pause the user can call (#51).

`DriveBudget` counts autonomous driving time (the real run time of the moves nobody asked for) in a
sliding hour and in the local day, and says when a limit is used up. The day's total is kept in the
store, so a restart does not give the whole day back; the hour is only in memory (after a restart the
hour limit starts again, the day limit still holds). `AutonomyPause` is the "Laika, resta ferma"
switch: it is kept in the store too and only the user lifts it. Both are read by the safety controller
(docs/behaviours.md, "House rules, driving budget and pause").
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from laika.observations import fold

if TYPE_CHECKING:
    from laika.store import Store

log = logging.getLogger(__name__)

NAMESPACE = "autonomy"
DAY_KEY = "drive_day"
PAUSE_KEY = "paused"
HOUR_S = 3600.0
MAX_HOUR_ENTRIES = 4000


def local_day(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).date().isoformat()


class DriveBudget:
    """Autonomous driving seconds in the last hour and in today's local day. A limit of 0 is off."""

    def __init__(
        self,
        per_hour_s: float = 0.0,
        per_day_s: float = 0.0,
        store: Store | None = None,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.per_hour_s = max(0.0, float(per_hour_s))
        self.per_day_s = max(0.0, float(per_day_s))
        self.store = store
        self._wall = wall_clock
        self._lock = threading.Lock()
        self._recent: deque[tuple[float, float]] = deque(maxlen=MAX_HOUR_ENTRIES)
        self._day = local_day(wall_clock())
        self._day_s = 0.0
        self._load()

    def _load(self) -> None:
        if self.store is None:
            return
        saved = self.store.kv_get(NAMESPACE, DAY_KEY)
        if not isinstance(saved, dict) or saved.get("date") != self._day:
            return
        seconds = saved.get("seconds")
        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and math.isfinite(seconds):
            self._day_s = max(0.0, float(seconds))

    def _roll(self, now: float) -> None:
        """Start a new day's count when the local date changed, and drop what left the hour (lock held)."""
        today = local_day(now)
        if today != self._day:
            self._day, self._day_s = today, 0.0
        while self._recent and now - self._recent[0][0] >= HOUR_S:
            self._recent.popleft()

    def add(self, seconds: float) -> None:
        """Count `seconds` of autonomous driving that just happened."""
        if not math.isfinite(seconds) or seconds <= 0:
            return
        now = self._wall()
        with self._lock:
            self._roll(now)
            self._recent.append((now, seconds))
            self._day_s += seconds
            day, total = self._day, self._day_s
        if self.store is not None:
            self.store.kv_put(NAMESPACE, DAY_KEY, {"date": day, "seconds": round(total, 3)})

    def exhausted(self) -> str:
        """"hour" or "day" when that limit is used up, else ''."""
        now = self._wall()
        with self._lock:
            self._roll(now)
            if self.per_hour_s and self._hour_used() >= self.per_hour_s:
                return "hour"
            if self.per_day_s and self._day_s >= self.per_day_s:
                return "day"
        return ""

    def _hour_used(self) -> float:
        return sum(seconds for _, seconds in self._recent)

    def snapshot(self) -> dict[str, Any]:
        now = self._wall()
        with self._lock:
            self._roll(now)
            hour, day = self._hour_used(), self._day_s
        left = [limit - used for limit, used in ((self.per_hour_s, hour), (self.per_day_s, day)) if limit]
        return {
            "hour_used_s": round(hour, 1),
            "hour_limit_s": self.per_hour_s or None,
            "day_used_s": round(day, 1),
            "day_limit_s": self.per_day_s or None,
            "left_s": round(max(0.0, min(left)), 1) if left else None,
            "exhausted": self.exhausted() or None,
        }


class AutonomyPause:
    """"Laika, resta ferma": every autonomous activity is refused until the user lifts it. Survives a restart."""

    def __init__(self, store: Store | None = None, wall_clock: Callable[[], float] = time.time) -> None:
        self.store = store
        self._wall = wall_clock
        self._lock = threading.Lock()
        self._since: float | None = None
        if store is not None:
            saved = store.kv_get(NAMESPACE, PAUSE_KEY)
            since = saved.get("since") if isinstance(saved, dict) else None
            if isinstance(saved, dict) and saved.get("paused") is True:
                ok = isinstance(since, (int, float)) and not isinstance(since, bool) and math.isfinite(since)
                self._since = float(since) if ok else self._wall()

    @property
    def paused(self) -> bool:
        return self._since is not None

    @property
    def since(self) -> float | None:
        return self._since

    def pause(self) -> bool:
        """Pause. True when it was not paused before."""
        with self._lock:
            changed = self._since is None
            if changed:
                self._since = self._wall()
            since = self._since
        if changed and self.store is not None:
            self.store.kv_put(NAMESPACE, PAUSE_KEY, {"paused": True, "since": since})
        return changed

    def resume(self) -> bool:
        """Lift the pause. True when it was paused."""
        with self._lock:
            changed, self._since = self._since is not None, None
        if changed and self.store is not None:
            self.store.kv_put(NAMESPACE, PAUSE_KEY, {"paused": False, "since": None})
        return changed


# --- the user's own words ---------------------------------------------------------------------------

MAX_PHRASE_WORDS = 9
PAUSE_PHRASES = (
    re.compile(r"\b(?:resta|stai|rimani|restare|stare|rimanere)\s+(?:li\s+)?ferma\b"),
    re.compile(r"\bnon\s+(?:ti\s+muovere|muoverti)\b"),
)
RESUME_PHRASES = (
    re.compile(r"\bpuoi\s+(?:di\s+nuovo\s+|ancora\s+|pure\s+|adesso\s+|ora\s+)?(?:muoverti|girare|esplorare)\b"),
    re.compile(r"\b(?:riprendi|ricomincia|torna)\s+a\s+(?:muoverti|girare|esplorare)\b"),
    re.compile(r"\bmuoviti\s+pure\b"),
)


def autonomy_command(text: str) -> str | None:
    """"pause" or "resume" when a short utterance is the user asking for it, else None.

    A local rule, so it works in JSON mode and when the model is slow or down. Only the user's own
    words reach it (never web text or her own proactive lines).
    """
    words = fold(text.replace("?", " ").replace("!", " ").replace(",", " ").replace(".", " "))
    if not words or len(words.split()) > MAX_PHRASE_WORDS:
        return None
    if any(p.search(words) for p in RESUME_PHRASES):
        return "resume"
    if any(p.search(words) for p in PAUSE_PHRASES):
        return "pause"
    return None
