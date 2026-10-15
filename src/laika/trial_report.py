"""The field-trial report: `python3 run-laika.py --report [--days N] [--since YYYY-MM-DD]` (#44).

Issue #44 asks whether the core loop (sees, understands, notices, acts, asks, a human answers,
remembers) completes repeatedly in the real house without supervision. The judging is the owner's;
this module reads what Laika already wrote and turns it into one page that makes the judging quick.

Sources, all read-only and all local:

* the event log `logs/events.jsonl` and its rotated files `events.jsonl.1..N`,
* the SQLite store `memory/laika.db`, opened with `mode=ro` (it is never created, migrated or written).

It needs no AI key and no hardware, and works on a copy of the data dir from another machine. Bad,
partial or unknown log lines are skipped and counted; a missing store only removes the store-based
numbers. The report holds counts, object names, room names and times: no transcripts, no pictures,
no people identities, and not even what an answer said (episode summaries and fact texts are never read).

Gaps, said in the report itself: nothing in the event log records that a question was queued, so a
queued question is linked to the proactive ask that carries its `questionId` by order within one run
(ids restart at q-1 on every start, and every run begins with HardwareReady). Observation rejections,
policy refusals and "stuck" explorations are printed to the console/journal only.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

from laika import __version__
from laika.config import parse_quiet_hours
from laika.events import EventType
from laika.storage import EVENT_LOG_NAME, FILE_MODE, DataPaths, resolve_data_dir
from laika.store import DB_NAME

log = logging.getLogger(__name__)

REPORT_VERSION = 1
DAY_S = 86400.0
QUESTION_TTL_S = 24 * 3600.0  # PendingQuestions drops a question after this long
GAP_S = 1800.0  # a silence in the event log this long is listed
MAX_LISTED = 25  # rows per list in the Markdown report; the JSON has them all
NAME_CHARS = 60
ANSWER_WINDOW_S = 5.0  # an answer episode this close after the end of a proactive ask belongs to it
# The suggested bar of docs/field-trial.md.
BAR_LOOPS_PER_DAY = 2.0

PRIVACY = (
    "This report contains counts, object names, room names and times. It holds no transcripts, no "
    "images, no people identities and none of what anyone said (answers and fact texts are not read)."
)

# Why a move ended. These need a person's eyes; the rest (preempted, superseded...) is normal traffic.
ATTENTION_REASONS = frozenset(
    {"obstacle", "cliff", "emergency_stop", "watchdog", "ai_unavailable", "hardware error",
     "ultrasonic_invalid", "monitor error"}
)

NOT_VISIBLE = (
    "Observation rejections (`[laika] observation rejected`) are printed, not logged as events.",
    "Interaction-policy refusals (`interaction policy: refused (...)`) go to the journal only.",
    "A stuck or stalled exploration (`exploration paused`, `stalled: lease expired`) is printed only.",
    "Pending questions live in memory: one that was never asked and was lost in a restart looks like a "
    "queued question that was never asked.",
    "Do-not-disturb is shown as its stored end time only; its history is not recorded.",
    "Search for those lines with `journalctl -u laika --since yesterday | grep -E \"rejected|refused|paused|stalled\"`.",
)

BATTERY_EVENTS = {
    EventType.BATTERY_LOW.value: "low",
    EventType.BATTERY_CRITICAL.value: "critical",
    EventType.BATTERY_OK.value: "ok",
}

_EVENT_FILE = re.compile(rf"^{re.escape(EVENT_LOG_NAME)}(?:\.(\d+))?$")
_KNOWN_TYPES = frozenset(t.value for t in EventType)


# --- reading the event log ----------------------------------------------------------------------


@dataclass(frozen=True)
class Ev:
    ts: float
    type: str
    data: Mapping[str, Any]


@dataclass
class EventLog:
    events: list[Ev] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    lines: int = 0
    bad_lines: int = 0
    unknown_types: Counter = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)

    @property
    def first(self) -> float | None:
        return self.events[0].ts if self.events else None

    @property
    def last(self) -> float | None:
        return self.events[-1].ts if self.events else None


def event_files(logs: Path) -> list[Path]:
    """`events.jsonl.N` (oldest, highest N) ... `events.jsonl.1`, then `events.jsonl`."""
    found: list[tuple[int, Path]] = []
    try:
        names = list(logs.iterdir())
    except OSError:
        return []
    for path in names:
        match = _EVENT_FILE.match(path.name)
        if match and path.is_file():
            found.append((int(match.group(1)) if match.group(1) else 0, path))
    found.sort(key=lambda item: (item[0] == 0, -item[0]))
    return [path for _, path in found]


def _parse_line(text: str) -> Ev | None:
    try:
        raw = json.loads(text)
    except ValueError:
        return None
    if not isinstance(raw, dict):
        return None
    kind, stamp = raw.get("type"), raw.get("timestamp")
    if not isinstance(kind, str) or isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
        return None
    if not math.isfinite(stamp):
        return None
    data = raw.get("data")
    return Ev(float(stamp), kind, data if isinstance(data, dict) else {})


def load_events(logs: Path) -> EventLog:
    """Every readable event, oldest first. Lines that are not events are counted, never raised."""
    result = EventLog()
    for path in event_files(logs):
        result.files.append(path.name)
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    result.lines += 1
                    event = _parse_line(line)
                    if event is None:
                        result.bad_lines += 1
                        continue
                    if event.type not in _KNOWN_TYPES:
                        result.unknown_types[event.type] += 1
                        continue
                    result.events.append(event)
        except OSError as error:
            result.errors.append(f"could not read {path.name}: {error}")
    result.events.sort(key=lambda event: event.ts)  # stable: same-time events keep file order
    return result


# --- reading the store (read-only) --------------------------------------------------------------


@dataclass
class StoreData:
    available: bool = False
    note: str = ""
    observations: list[dict[str, Any]] = field(default_factory=list)  # in the period
    episodes: list[dict[str, Any]] = field(default_factory=list)  # in the period or the question lookback
    facts: list[dict[str, Any]] = field(default_factory=list)  # touched in the period
    objects: list[dict[str, Any]] = field(default_factory=list)  # first seen in the period
    kv: dict[str, Any] = field(default_factory=dict)
    totals: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _open_readonly(path: Path) -> tuple[sqlite3.Connection | None, str]:
    """A connection that cannot write. A WAL file that cannot be read in place falls back to an
    immutable open, which ignores the WAL (the newest writes may be missing, and the note says so)."""
    base = path.resolve().as_uri()
    for query, note in (("?mode=ro", ""), ("?mode=ro&immutable=1", "opened immutable: writes still in the WAL file may be missing")):
        conn = None
        try:
            conn = sqlite3.connect(base + query, uri=True, timeout=2.0)
            conn.execute("PRAGMA query_only = ON")
            conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
            return conn, note
        except sqlite3.Error:
            if conn is not None:
                conn.close()
    return None, "the database could not be opened"


def _loads(text: Any, default: Any) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


def _query(conn: sqlite3.Connection, data: StoreData, sql: str, params: tuple[Any, ...] = ()) -> list[tuple]:
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.Error as error:
        message = str(error).split("\n")[0]
        if message not in data.errors:
            data.errors.append(message)
        return []


def read_store(db_path: Path, start: float, end: float) -> StoreData:
    data = StoreData()
    if not db_path.is_file():
        data.note = "no database (memory/laika.db): store-based numbers are missing"
        return data
    conn, note = _open_readonly(db_path)
    if conn is None:
        data.note = note
        return data
    data.available = True
    data.note = note
    try:
        for ts, source, objects, people, location in _query(
            conn, data,
            "SELECT timestamp, source, objects, people, location FROM observations "
            "WHERE timestamp >= ? AND timestamp < ?", (start, end),
        ):
            names = [o.get("name") if isinstance(o, dict) else o for o in _loads(objects, []) if o]
            data.observations.append(
                {"ts": ts, "source": source, "objects": [n for n in names if isinstance(n, str)],
                 "people": people or 0, "location": location}
            )
        # Summaries are never selected: they can quote what a person said.
        for rowid, ts, kind, subjects, location in _query(
            conn, data,
            "SELECT rowid, timestamp, kind, subjects, location FROM episodes "
            "WHERE (timestamp >= ? AND timestamp < ?) OR (kind IN ('question', 'answer') AND timestamp < ?) "
            "ORDER BY timestamp, rowid", (start, end, end),
        ):
            subjects = [s for s in _loads(subjects, []) if isinstance(s, str)]
            data.episodes.append({"ts": ts, "kind": kind, "subjects": subjects, "location": location})
        # Fact texts are never selected either.
        for fid, subject, kind, source, created, updated in _query(
            conn, data,
            "SELECT id, subject, kind, source, created, updated FROM facts WHERE updated >= ? AND updated < ?",
            (start, end),
        ):
            data.facts.append(
                {"id": fid, "subject": subject, "kind": kind, "source": source,
                 "created": created, "updated": updated, "new": start <= created < end}
            )
        for (text,) in _query(conn, data, "SELECT data FROM world_objects"):
            document = _loads(text, None)
            if not isinstance(document, dict) or not isinstance(document.get("name"), str):
                continue
            first = document.get("first_seen")
            if isinstance(first, (int, float)) and start <= first < end:
                data.objects.append(
                    {"name": document["name"], "room": document.get("room"), "first_seen": float(first),
                     "source": document.get("source"), "explained": bool(document.get("note"))}
                )
        for key, value in _query(conn, data, "SELECT key, value FROM kv WHERE namespace = 'interaction'"):
            data.kv[key] = _loads(value, None)
        for table in ("observations", "episodes", "facts", "world_objects"):
            rows = _query(conn, data, f"SELECT COUNT(*) FROM {table}")
            if rows:
                data.totals[table] = int(rows[0][0])
    finally:
        conn.close()
    return data


# --- small helpers ------------------------------------------------------------------------------


def _in_quiet_hours(ts: float, window: tuple[int, int]) -> bool:
    moment = datetime.fromtimestamp(ts)
    minute = moment.hour * 60 + moment.minute
    first, last = window
    return first <= minute < last if first < last else minute >= first or minute < last


def _clip(value: Any, limit: int = NAME_CHARS) -> str:
    text = " ".join(str(value).split()).replace("|", "/")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _iso(ts: float | None) -> str | None:
    return None if ts is None else datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def _when(ts: float | None) -> str:
    return "-" if ts is None else datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _dur(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    if seconds < 2 * DAY_S:
        hours = seconds / 3600
        return f"{hours:.1f} h"
    return f"{seconds / DAY_S:.1f} days"


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts).date().isoformat()


def _days_between(start: float, end: float) -> list[str]:
    first, last = datetime.fromtimestamp(start).date(), datetime.fromtimestamp(max(start, end - 1e-6)).date()
    return [(first + timedelta(days=n)).isoformat() for n in range((last - first).days + 1)]


def _text(value: Any, default: str = "unknown") -> str:
    return _clip(value) if isinstance(value, str) and value.strip() else default


def _object_names(data: Mapping[str, Any]) -> list[str]:
    names = []
    for raw in data.get("objects") or []:
        name = raw.get("name") if isinstance(raw, Mapping) else raw
        if isinstance(name, str) and name.strip():
            names.append(_clip(name))
    return names


def _counter(counter: Counter) -> dict[str, int]:
    return {str(k): v for k, v in sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))}


def _reason_group(reason: str) -> str:
    """A curiosity reason ("new in kitchen: cup") reduced to what kind of thing it was."""
    head = reason.split(":", 1)[0].strip().lower()
    for prefix, name in (("new in ", "new"), ("missing from ", "missing")):
        if head.startswith(prefix):
            return name
    return head[:40] or "unknown"


# --- the analysis -------------------------------------------------------------------------------


def _restart_starts(events: Iterable[Ev]) -> list[float]:
    return [e.ts for e in events if e.type == EventType.HARDWARE_READY.value]


def _run_of(ts: float, starts: list[float]) -> int:
    """How many starts happened at or before `ts`: events of one run share a number, which is also the
    scope of the question ids (they restart at q-1)."""
    return sum(1 for s in starts if s <= ts)


def _question_ordinals(queued: list[dict[str, Any]], starts: list[float]) -> None:
    """Give each queued question the id the app would have given it: q-1, q-2... within its run."""
    counters: Counter = Counter()
    for item in queued:
        run = _run_of(item["ts"], starts)
        counters[run] += 1
        item["run"] = run
        item["id"] = f"q-{counters[run]}"


def build_loops(
    queued: list[dict[str, Any]],
    asks: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    user_windows: list[tuple[float, float]],
    starts: list[float],
    start: float,
    end: float,
) -> dict[str, Any]:
    """Chain queued -> asked -> answered -> remembered, and list what is still open (see module doc)."""
    queued = sorted(queued, key=lambda q: q["ts"])
    _question_ordinals(queued, starts)
    by_key = {(q["run"], q["id"]): q for q in queued}
    for q in queued:
        q.update(asks=[], answer=None, via=None)
    orphans = []
    for ask in sorted(asks, key=lambda a: a["ts"]):
        ask["run"] = _run_of(ask["ts"], starts)
        # Run 0 began before the log did: its question numbers cannot be told, so it is not linked.
        target = by_key.get((ask["run"], ask["question_id"])) if ask["run"] else None
        if target is not None and target["ts"] <= ask["ts"]:
            target["asks"].append(ask)
        else:
            orphans.append(ask)
    used: set[int] = set()
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in queued:
        by_object[q["object"]].append(q)
    for series in by_object.values():
        for index, q in enumerate(series):
            limit = series[index + 1]["ts"] if index + 1 < len(series) else float("inf")
            for position, answer in enumerate(answers):
                if position in used or answer["object"] != q["object"] or not q["ts"] <= answer["ts"] < limit:
                    continue
                used.add(position)
                q["answer"] = answer
                break
    for q in queued:
        answer = q["answer"]
        if answer is None:
            continue
        proactive = next(
            (a for a in q["asks"] if a["ts"] - ANSWER_WINDOW_S <= answer["ts"] <= (a["completed"] or answer["ts"]) + ANSWER_WINDOW_S),
            None,
        )
        if proactive is not None:
            q["via"] = "laika asked"
        elif any(a <= answer["ts"] <= b + ANSWER_WINDOW_S for a, b in user_windows):
            q["via"] = "asked in a conversation (inferred)"
        else:
            q["via"] = "ask not traced"
        q["fact"] = next(
            (f for f in facts if f["subject"] == q["object"] and f["source"] == "user" and f["updated"] >= answer["ts"] - ANSWER_WINDOW_S),
            None,
        )

    def row(q: dict[str, Any]) -> dict[str, Any]:
        first = q["asks"][0] if q["asks"] else None
        answer = q["answer"]
        return {
            "object": q["object"], "room": q["room"], "question_id": q["id"],
            "queued": _iso(q["ts"]), "asked": _iso(first["ts"]) if first else None,
            "ask_count": len(q["asks"]),
            "answered": _iso(answer["ts"]) if answer else None,
            "remembered": answer is not None,  # the "answer" episode is itself a memory
            "fact": bool(q.get("fact")), "via": q["via"],
            "age_s": None if answer else round(max(0.0, end - q["ts"]), 1),
            "queued_to_answer_s": round(answer["ts"] - q["ts"], 1) if answer else None,
        }

    completed = [q for q in queued if q["answer"] is not None and start <= q["answer"]["ts"] < end]
    open_unasked, open_unanswered = [], []
    restarts_after = lambda ts: any(s > ts for s in starts)  # noqa: E731 - pending questions are in memory only
    for q in queued:
        if q["answer"] is not None or not start - QUESTION_TTL_S <= q["ts"] < end:
            continue
        item = row(q)
        item["likely_lost"] = bool(end - q["ts"] > QUESTION_TTL_S or restarts_after(q["ts"]))
        (open_unanswered if q["asks"] else open_unasked).append(item)
    return {
        "completed": [row(q) for q in completed],
        "open_asked_unanswered": open_unanswered,
        "open_queued_never_asked": open_unasked,
        "asked_without_queue_record": [
            {"question_id": a["question_id"], "asked": _iso(a["ts"]), "answered": a["answered"]} for a in orphans
        ],
        "answers_without_queue": sum(1 for p, a in enumerate(answers) if p not in used and start <= a["ts"] < end),
    }


def _number_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def build_battery(period: Iterable[Ev]) -> dict[str, Any]:
    """Battery transitions (BatteryLow, BatteryCritical, BatteryOk) and the lowest voltage they report.

    Only changes are logged, not every reading, so the lowest voltage is the lowest one the
    events mention: the voltage at each change and `lowestV`, the lowest the monitor had seen by then.
    """
    rows = [e for e in period if e.type in BATTERY_EVENTS]
    transitions = [
        {
            "at": _iso(e.ts),
            "state": BATTERY_EVENTS[e.type],
            "voltage": _number_or_none(e.data.get("voltage")),
            "percent": _number_or_none(e.data.get("percent")),
        }
        for e in rows
    ]
    seen = [
        v for e in rows for v in (_number_or_none(e.data.get("voltage")), _number_or_none(e.data.get("lowestV")))
        if v is not None
    ]
    counts = Counter(BATTERY_EVENTS[e.type] for e in rows)
    return {
        "transitions": transitions,
        "low": counts["low"],
        "critical": counts["critical"],
        "ok": counts["ok"],
        "lowest_v": round(min(seen), 2) if seen else None,
    }


def build_report(
    data_dir: Path,
    start: float,
    end: float,
    *,
    now: float | None = None,
    quiet_hours: str | None = None,
) -> dict[str, Any]:
    """Everything the report says, as JSON-friendly data. Reads only; never raises for bad input files."""
    now = time.time() if now is None else now
    paths = DataPaths(data_dir)
    log_data = load_events(paths.logs)
    store = read_store(paths.memory / DB_NAME, start, end)
    all_events = log_data.events
    period = [e for e in all_events if start <= e.ts < end]
    starts_all = _restart_starts(all_events)
    days = _days_between(start, end)
    per_day: dict[str, Counter] = {d: Counter() for d in days}

    def tick(ts: float, key: str, amount: float = 1) -> None:
        bucket = per_day.get(_day(ts))
        if bucket is not None:
            bucket[key] += amount

    by_type: dict[str, list[Ev]] = defaultdict(list)
    for event in period:
        by_type[event.type].append(event)

    def of(kind: EventType) -> list[Ev]:
        return by_type.get(kind.value, [])

    # sees and understands
    images = of(EventType.IMAGE_CAPTURED)
    by_reason = Counter(_text(e.data.get("reason"), "unknown") for e in images)
    created = of(EventType.OBSERVATION_CREATED)
    seen_objects: Counter = Counter()
    event_changes: Counter = Counter()
    for event in created:
        seen_objects.update(set(_object_names(event.data)))
        for change in event.data.get("changes") or []:
            if isinstance(change, Mapping) and isinstance(change.get("kind"), str):
                event_changes[change["kind"]] += 1
    for event in images:
        tick(event.ts, "images")
    for event in created:
        tick(event.ts, "observations")
    store_sources = Counter(_text(o["source"]) for o in store.observations)
    people_frames = sum(1 for o in store.observations if o["people"]) or sum(
        1 for e in created if (e.data.get("people") or 0)
    )
    # notices
    interesting = of(EventType.INTERESTING_CHANGE_DETECTED)
    reasons: Counter = Counter()
    interesting_objects: Counter = Counter()
    for event in interesting:
        tick(event.ts, "interesting")
        for reason in event.data.get("reasons") or []:
            if isinstance(reason, str):
                reasons[_reason_group(reason)] += 1
        interesting_objects.update(set(_object_names(event.data)))
    episode_kinds = Counter(e["kind"] for e in store.episodes if start <= e["ts"] < end)
    # acts
    moves = {
        "started": len(of(EventType.MOVEMENT_STARTED)),
        "completed": len(of(EventType.MOVEMENT_COMPLETED)),
        "failed": len(of(EventType.MOVEMENT_FAILED)),
    }
    directions = Counter(_text(e.data.get("direction")) for e in of(EventType.MOVEMENT_STARTED))
    failed_reasons = Counter(_text(e.data.get("reason"), "unknown") for e in of(EventType.MOVEMENT_FAILED))
    for event in of(EventType.MOVEMENT_STARTED):
        tick(event.ts, "moves")
    stops = Counter(failed_reasons)
    stop_rows = [e for e in of(EventType.MOVEMENT_FAILED) if _text(e.data.get("reason"), "unknown") in ATTENTION_REASONS]
    for event in stop_rows:
        tick(event.ts, "safety_stops")

    # interactions
    started_events = [e for e in period if e.type == EventType.INTERACTION_STARTED.value]
    completed_events = [e for e in period if e.type == EventType.INTERACTION_COMPLETED.value]
    open_by: dict[str, list[Ev]] = defaultdict(list)
    interactions: list[dict[str, Any]] = []
    unfinished = Counter()
    for event in sorted(started_events + completed_events, key=lambda e: e.ts):
        who = _text(event.data.get("initiator"), "user")
        if event.type == EventType.INTERACTION_STARTED.value:
            open_by[who].append(event)
        elif open_by[who]:
            began = open_by[who].pop(0)
            interactions.append({"who": who, "start": began.ts, "end": event.ts, "data": began.data, "done": event.data})
    for who, leftover in open_by.items():
        if leftover:
            unfinished[who] += len(leftover)
    user_windows = [(i["start"], i["end"]) for i in interactions if i["who"] == "user"]
    laika_done = [i for i in interactions if i["who"] == "laika"]
    answered = [i for i in laika_done if i["done"].get("answered") is True]
    asks: list[dict[str, Any]] = []
    for event in started_events:
        if _text(event.data.get("initiator"), "user") != "laika":
            continue
        qid = event.data.get("questionId")
        match = next((i for i in laika_done if i["start"] == event.ts and i["data"].get("questionId") == qid), None)
        asks.append(
            {"ts": event.ts, "question_id": _text(qid), "completed": match["end"] if match else None,
             "answered": bool(match and match["done"].get("answered") is True) if match else None}
        )
        tick(event.ts, "asks")
    for i in answered:
        tick(i["end"], "answers")
    for i in interactions:
        if i["who"] == "user":
            tick(i["start"], "user_conversations")

    # loops
    queued = [
        {"ts": e["ts"], "object": e["subjects"][0] if e["subjects"] else "unknown",
         "room": e["location"]}
        for e in store.episodes if e["kind"] == "question"
    ]
    answer_rows = [
        {"ts": e["ts"], "object": e["subjects"][0] if e["subjects"] else "unknown"}
        for e in store.episodes if e["kind"] == "answer"
    ]
    for q in queued:
        q["object"], q["room"] = _text(q["object"]), (_text(q["room"]) if q["room"] else None)
    for a in answer_rows:
        a["object"] = _text(a["object"])
    queued_in_period = [q for q in queued if start <= q["ts"] < end]
    for q in queued_in_period:
        tick(q["ts"], "questions_queued")
    loops = build_loops(
        queued, asks, answer_rows,
        [{**f, "subject": _text(f["subject"]) if f["subject"] else None} for f in store.facts],
        user_windows, starts_all, start, end,
    )
    for item in loops["completed"]:
        stamp = datetime.fromisoformat(item["answered"]).timestamp()
        tick(stamp, "loops_completed")

    # remembers
    for fact in store.facts:
        if fact["new"]:
            tick(fact["created"], "facts_created")
    facts_new = [f for f in store.facts if f["new"]]
    facts_updated = [f for f in store.facts if not f["new"]]

    # supervision
    ai_events = sorted(
        [e for e in period if e.type in (EventType.AI_ENDPOINT_UNAVAILABLE.value, EventType.AI_ENDPOINT_RECOVERED.value,
                                          EventType.HARDWARE_READY.value)],
        key=lambda e: e.ts,
    )
    outages: list[dict[str, Any]] = []
    opened: Ev | None = None
    for event in ai_events:
        if event.type == EventType.AI_ENDPOINT_UNAVAILABLE.value and opened is None:
            opened = event
        elif opened is not None and event.type != EventType.AI_ENDPOINT_UNAVAILABLE.value:
            how = "recovered" if event.type == EventType.AI_ENDPOINT_RECOVERED.value else "ended by a restart"
            outages.append({"start": opened.ts, "end": event.ts, "duration_s": round(event.ts - opened.ts, 1), "ended": how})
            tick(opened.ts, "outage_s", event.ts - opened.ts)
            opened = None
    if opened is not None:
        outages.append({"start": opened.ts, "end": end, "duration_s": round(max(0.0, min(end, now) - opened.ts), 1), "ended": "still down at the end of the period"})
        tick(opened.ts, "outage_s", max(0.0, min(end, now) - opened.ts))
    watchdog = [
        {"at": _iso(e.ts), "reason": _text(e.data.get("reason")), "age_s": e.data.get("ageS")}
        for e in of(EventType.LOCAL_WATCHDOG_EXPIRED)
    ]
    for event in of(EventType.LOCAL_WATCHDOG_EXPIRED):
        tick(event.ts, "watchdog")
    hardware = [
        {"at": _iso(e.ts), "device": _text(e.data.get("device"), "robot"), "error": _clip(e.data.get("error", ""), 100)}
        for e in of(EventType.HARDWARE_UNAVAILABLE)
    ]
    for event in of(EventType.HARDWARE_UNAVAILABLE):
        tick(event.ts, "hardware_unavailable")
    restarts: list[dict[str, Any]] = []
    for event in of(EventType.HARDWARE_READY):
        before = [e for e in all_events if e.ts < event.ts]
        previous = before[-1] if before else None
        if previous is None:
            continue  # the first start the log knows about
        restarts.append(
            {"at": _iso(event.ts), "ts": event.ts, "silent_before_s": round(event.ts - previous.ts, 1),
             "previous_event": previous.type, "ai_down": any(o["start"] <= event.ts <= o["end"] for o in outages)}
        )
        tick(event.ts, "restarts")
    gaps = []
    for earlier, later in zip(all_events, all_events[1:]):
        if later.ts - earlier.ts >= GAP_S and start <= later.ts < end and later.type != EventType.HARDWARE_READY.value:
            gaps.append({"from": _iso(earlier.ts), "to": _iso(later.ts), "duration_s": round(later.ts - earlier.ts, 1)})
    gaps.sort(key=lambda g: -g["duration_s"])
    interventions: list[str] = []
    for event in stop_rows:
        if _text(event.data.get("reason")) in ("emergency_stop", "watchdog", "hardware error"):
            interventions.append(f"{_when(event.ts)}: move ended by {_text(event.data.get('reason'))}")
    for item in watchdog:
        interventions.append(f"{item['at'][:16].replace('T', ' ')}: local watchdog expired ({item['reason']})")
    for item in restarts:
        interventions.append(f"{item['at'][:16].replace('T', ' ')}: restart after {_dur(item['silent_before_s'])} of silence")
    for item in hardware:
        interventions.append(f"{item['at'][:16].replace('T', ' ')}: hardware unavailable ({item['device']})")
    interventions.sort()

    # conversations
    user_n = sum(1 for i in interactions if i["who"] == "user")
    laika_n = len(laika_done)
    durations = [i["end"] - i["start"] for i in interactions]
    quiet = {"configured": quiet_hours or None, "do_not_disturb_until": None, "busy_until": None}
    for key in ("do_not_disturb_until", "busy_until"):
        value = store.kv.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= start:
            quiet[key] = _iso(float(value))
    try:
        window = parse_quiet_hours(quiet_hours) if quiet_hours else None
    except ValueError:
        window = None
    conversations = {
        "asks": [
            {"at": _iso(a["ts"]), "question_id": a["question_id"], "answered": a["answered"]} for a in asks
        ],
        "asks_in_quiet_hours": None if window is None else sum(1 for a in asks if _in_quiet_hours(a["ts"], window)),
        "user_initiated": user_n,
        "laika_initiated": laika_n,
        "laika_answered": len(answered),
        "answered_rate": round(len(answered) / laika_n, 3) if laika_n else None,
        "started_not_completed": dict(unfinished),
        "mean_duration_s": round(sum(durations) / len(durations), 1) if durations else None,
        "do_not_disturb": quiet,
    }

    # the suggested bar
    span_days = max(1.0, (end - start) / DAY_S)
    target = max(1, round(BAR_LOOPS_PER_DAY * span_days))
    attention = len(stop_rows)
    bar = [
        {"check": f"completed loops (at least {target})", "value": len(loops["completed"]),
         "status": "met" if len(loops["completed"]) >= target else "not met"},
        {"check": "local watchdog expiries (none)", "value": len(watchdog), "status": "met" if not watchdog else "not met"},
        {"check": "hardware unavailable (none)", "value": len(hardware), "status": "met" if not hardware else "not met"},
        {"check": "proactive asks inside quiet hours (none)", "value": conversations["asks_in_quiet_hours"],
         "status": "not known" if window is None else ("met" if not conversations["asks_in_quiet_hours"] else "not met")},
        {"check": "restarts, each one explained", "value": len(restarts), "status": "met" if not restarts else "needs your explanation"},
        {"check": "safety stops, each one explained", "value": attention, "status": "met" if not attention else "needs your explanation"},
        {"check": "robot stuck and needing rescue (none)", "value": None, "status": "note it yourself"},
    ]

    sources = {
        "data_dir": str(data_dir),
        "event_files": log_data.files,
        "event_lines": log_data.lines,
        "bad_lines": log_data.bad_lines,
        "unknown_event_types": _counter(log_data.unknown_types),
        "events_in_period": len(period),
        "log_first_event": _iso(log_data.first),
        "log_last_event": _iso(log_data.last),
        # Only a log that has rotated can have lost the start of the period, and one that begins with a
        # start (HardwareReady) has lost nothing: it simply began then.
        "log_rotated_past_start": bool(
            log_data.events
            and log_data.first > start
            and len(log_data.files) > 1
            and log_data.events[0].type != EventType.HARDWARE_READY.value
        ),
        "read_errors": log_data.errors,
        "store_available": store.available,
        "store_note": store.note,
        "store_errors": store.errors,
        "store_totals": store.totals,
    }
    return {
        "version": REPORT_VERSION,
        "laika_version": __version__,
        "privacy": PRIVACY,
        "generated_at": _iso(now),
        "period": {"start": _iso(start), "end": _iso(end), "days": round((end - start) / DAY_S, 2)},
        "core_loop": {
            "sees": {"images": len(images), "images_by_reason": _counter(by_reason),
                     "observations_stored": len(store.observations) if store.available else None},
            "understands": {
                "observations_created": len(created),
                "images_to_observations": round(len(created) / len(images), 3) if images else None,
                "rejections_logged": None,
                "distinct_objects": len(seen_objects),
                "top_objects": _counter(Counter(dict(seen_objects.most_common(10)))),
                "frames_with_people": people_frames,
                "stored_sources": _counter(store_sources),
            },
            "notices": {
                "interesting_changes": len(interesting),
                "reasons": _counter(reasons),
                "objects": _counter(Counter(dict(interesting_objects.most_common(10)))),
                "changes_in_observations": _counter(event_changes),
                "change_episodes": {k: episode_kinds.get(k, 0) for k in ("moved", "new", "missing")},
            },
            "acts": {
                "moves": moves, "directions": _counter(directions), "failed_by_reason": _counter(failed_reasons),
                "investigation_pictures": by_reason.get("investigate", 0),
                "scan_pictures": by_reason.get("scan", 0),
                "search_pictures": by_reason.get("search", 0),
                "obstacles": len(of(EventType.OBSTACLE_DETECTED)),
                "cliffs": len(of(EventType.CLIFF_DETECTED)),
            },
            "asks": {
                "questions_queued": len(queued_in_period) if store.available else None,
                "proactive_asks": len(asks),
                "proactive_answered": len(answered),
            },
            "human_answers": {
                "answers_in_proactive_asks": len(answered),
                "answer_episodes": sum(1 for a in answer_rows if start <= a["ts"] < end) if store.available else None,
            },
            "remembers": {
                "facts_created": len(facts_new) if store.available else None,
                "facts_updated": len(facts_updated) if store.available else None,
                "facts_by_source": _counter(Counter(_text(f["source"]) for f in store.facts)),
                "objects_became_known": [
                    {"name": _clip(o["name"]), "room": o["room"], "source": o["source"], "explained": o["explained"]}
                    for o in sorted(store.objects, key=lambda o: o["first_seen"])
                ],
            },
            "presence": {"person_detected": len(of(EventType.PERSON_DETECTED)), "person_lost": len(of(EventType.PERSON_LOST))},
        },
        "loops": loops,
        "supervision": {
            "safety_stops_by_reason": _counter(stops),
            "attention_stops": attention,
            "local_watchdog_expired": watchdog,
            "hardware_unavailable": hardware,
            "ai_outages": outages,
            "ai_outage_total_s": round(sum(o["duration_s"] for o in outages), 1),
            "ai_unavailable_events": len(of(EventType.AI_ENDPOINT_UNAVAILABLE)),
            "ai_recovered_events": len(of(EventType.AI_ENDPOINT_RECOVERED)),
            "hardware_ready": len(of(EventType.HARDWARE_READY)),
            "restarts": [{k: v for k, v in r.items() if k != "ts"} for r in restarts],
            "silent_gaps": gaps[:MAX_LISTED],
            "possible_interventions": interventions,
        },
        "conversations": conversations,
        "battery": build_battery(period),
        "per_day": [{"date": d, **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in sorted(per_day[d].items())}} for d in days],
        "bar": bar,
        "not_visible": list(NOT_VISIBLE),
        "sources": sources,
    }


# --- Markdown -----------------------------------------------------------------------------------


def _battery_summary(battery: Mapping[str, Any]) -> str:
    lowest = battery["lowest_v"]
    return (
        f"{battery['low']} low, {battery['critical']} critical, {battery['ok']} back to OK; lowest voltage "
        + ("not known (no battery event)" if lowest is None else f"{lowest:.2f} V")
    )


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join("-" if cell is None or cell == "" else str(cell) for cell in row) + " |")
    return out


def _counts(mapping: Mapping[str, int], empty: str = "none") -> str:
    return ", ".join(f"{k} {v}" for k, v in mapping.items()) or empty


def _days(count: float) -> str:
    return f"{count:g} day" + ("" if count == 1 else "s")


def _percent(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate:.0%}"


def _stamp(iso: str | None) -> str:
    return "-" if not iso else iso[:16].replace("T", " ")


def render_markdown(report: Mapping[str, Any]) -> str:
    core, sup, conv, loops = report["core_loop"], report["supervision"], report["conversations"], report["loops"]
    period, src = report["period"], report["sources"]
    lines = [
        "# Laika field-trial report",
        "",
        f"Period: {_stamp(period['start'])} to {_stamp(period['end'])} ({_days(period['days'])}). "
        f"Generated {_stamp(report['generated_at'])} by Laika {report['laika_version']}.",
        "",
        f"> {report['privacy']}",
        "",
        "## Summary",
        "",
        f"- Completed loops (question, asked, answered, remembered): **{len(loops['completed'])}**; "
        f"open: {len(loops['open_asked_unanswered'])} asked and unanswered, "
        f"{len(loops['open_queued_never_asked'])} queued and never asked.",
        f"- Supervision: {sup['attention_stops']} safety stop(s) that need an explanation, "
        f"{len(sup['local_watchdog_expired'])} watchdog expiry(ies), {len(sup['hardware_unavailable'])} hardware error(s), "
        f"{len(sup['restarts'])} restart(s), AI down {_dur(sup['ai_outage_total_s'])} in {len(sup['ai_outages'])} outage(s).",
        f"- Battery: {_battery_summary(report['battery'])}.",
        f"- Conversations: {conv['user_initiated']} started by a person, {conv['laika_initiated']} by Laika "
        f"(answered rate {_percent(conv['answered_rate'])}).",
        f"- Log lines read: {src['event_lines']} ({src['bad_lines']} unreadable, skipped); "
        f"events in the period: {src['events_in_period']}.",
    ]
    if src["log_rotated_past_start"] and src["log_first_event"]:
        lines.append(
            f"- Warning: the oldest event on disk is from {_stamp(src['log_first_event'])}, after the start of the "
            "period: older log files were rotated away. Raise `[storage] log_max_mb` and `log_backups`."
        )
    if not src["store_available"]:
        lines.append(f"- Warning: {src['store_note'] or 'the store could not be read'}.")
    lines += ["", "## Core loop", ""]
    see, und, notice, acts = core["sees"], core["understands"], core["notices"], core["acts"]
    asks, ans, rem = core["asks"], core["human_answers"], core["remembers"]
    moves = acts["moves"]
    rate = und["images_to_observations"]
    lines += _table(
        ["Stage", "Count", "Detail"],
        [
            ["sees", see["images"], f"pictures by reason: {_counts(see['images_by_reason'])}; stored observations: {see['observations_stored'] if see['observations_stored'] is not None else 'n/a'}"],
            ["understands", und["observations_created"],
             f"{'n/a' if rate is None else f'{rate:.0%}'} of pictures became an observation; {und['distinct_objects']} distinct objects; "
             f"{und['frames_with_people']} frames with people (count only); rejections: not logged. Top: {_counts(und['top_objects'])}"],
            ["notices", notice["interesting_changes"],
             f"reasons: {_counts(notice['reasons'])}; change episodes moved {notice['change_episodes']['moved']} / "
             f"new {notice['change_episodes']['new']} / missing {notice['change_episodes']['missing']}"],
            ["acts", moves["started"],
             f"moves started {moves['started']}, completed {moves['completed']}, failed {moves['failed']} ({_counts(acts['failed_by_reason'])}); "
             f"investigation pictures {acts['investigation_pictures']}, scan pictures {acts['scan_pictures']}, search pictures {acts['search_pictures']}"],
            ["asks", asks["proactive_asks"],
             f"questions queued: {asks['questions_queued'] if asks['questions_queued'] is not None else 'n/a'}; proactive asks by Laika: {asks['proactive_asks']}"],
            ["human answers", ans["answers_in_proactive_asks"],
             f"answered proactive asks {ans['answers_in_proactive_asks']}; answer episodes in the store: {ans['answer_episodes'] if ans['answer_episodes'] is not None else 'n/a'}"],
            ["remembers", rem["facts_created"] if rem["facts_created"] is not None else "n/a",
             f"facts created {rem['facts_created']}, updated {rem['facts_updated']} ({_counts(rem['facts_by_source'])}); objects that became known: {len(rem['objects_became_known'])}"],
        ],
    )
    if rem["objects_became_known"]:
        lines += ["", "Objects that became known: " + ", ".join(
            f"{o['name']}{' (explained)' if o['explained'] else ''}" for o in rem["objects_became_known"][:MAX_LISTED]
        ) + "."]
    lines += ["", "## Loops", ""]
    lines += ["Question queued, then asked, then answered, then remembered. Linking uses the order of questions within one "
              "run, as the event log has no queued event.", ""]
    if loops["completed"]:
        lines += ["Completed:", ""]
        lines += _table(
            ["Object", "Room", "Queued", "Asked", "Answered", "Via", "Fact", "Took"],
            [[c["object"], c["room"], _stamp(c["queued"]), _stamp(c["asked"]), _stamp(c["answered"]), c["via"],
              "yes" if c["fact"] else "no", _dur(c["queued_to_answer_s"])] for c in loops["completed"][:MAX_LISTED]],
        )
    else:
        lines.append("No completed loop in this period.")
    for title, key in (("Asked and not answered", "open_asked_unanswered"), ("Queued and never asked", "open_queued_never_asked")):
        if loops[key]:
            lines += ["", f"{title}:", ""]
            lines += _table(
                ["Object", "Room", "Queued", "Asked", "Age", "Note"],
                [[o["object"], o["room"], _stamp(o["queued"]), _stamp(o["asked"]), _dur(o["age_s"]),
                  "probably lost (restart or older than 24 h)" if o["likely_lost"] else ""] for o in loops[key][:MAX_LISTED]],
            )
    if loops["asked_without_queue_record"]:
        lines += ["", "Asked, but no matching queued question in the store: " + ", ".join(
            f"{a['question_id']} at {_stamp(a['asked'])}" for a in loops["asked_without_queue_record"][:MAX_LISTED]) + "."]
    if loops["answers_without_queue"]:
        lines += ["", f"Answer episodes with no queued question found: {loops['answers_without_queue']} (told unprompted or the question was queued earlier)."]

    lines += ["", "## Supervision issues", ""]
    lines.append(f"Safety stops by reason: {_counts(sup['safety_stops_by_reason'])}. "
                 f"Needing an explanation (obstacle, cliff, emergency stop, watchdog, ai unavailable, hardware error): {sup['attention_stops']}.")
    lines.append(f"Obstacle events {acts['obstacles']}, cliff events {acts['cliffs']}.")
    lines.append("")
    if sup["possible_interventions"]:
        lines += ["Anything that suggests a person had to step in:", ""] + [f"- {text}" for text in sup["possible_interventions"][:MAX_LISTED]]
    else:
        lines.append("Nothing suggests that a person had to step in.")
    lines += ["", f"Local watchdog expiries: {len(sup['local_watchdog_expired'])}. Hardware unavailable: {len(sup['hardware_unavailable'])}."]
    for item in sup["local_watchdog_expired"][:MAX_LISTED]:
        lines.append(f"- {_stamp(item['at'])} watchdog: {item['reason']}")
    for item in sup["hardware_unavailable"][:MAX_LISTED]:
        lines.append(f"- {_stamp(item['at'])} {item['device']}: {item['error']}")
    lines += ["", f"AI endpoint: {sup['ai_unavailable_events']} unavailable, {sup['ai_recovered_events']} recovered, total down {_dur(sup['ai_outage_total_s'])}."]
    for item in sup["ai_outages"][:MAX_LISTED]:
        lines.append(f"- {_stamp(_iso(item['start']))}: down {_dur(item['duration_s'])} ({item['ended']})")
    lines += ["", f"Starts (HardwareReady): {sup['hardware_ready']}; restarts: {len(sup['restarts'])}."]
    for item in sup["restarts"][:MAX_LISTED]:
        lines.append(f"- {_stamp(item['at'])}: after {_dur(item['silent_before_s'])} without events (last was {item['previous_event']})")
    if sup["silent_gaps"]:
        lines += ["", f"Silences of {_dur(GAP_S)} or more in the event log:"]
        lines += [f"- {_stamp(g['from'])} to {_stamp(g['to'])} ({_dur(g['duration_s'])})" for g in sup["silent_gaps"][:10]]

    battery = report["battery"]
    lines += ["", "## Battery", ""]
    lines.append(f"{_battery_summary(battery)}. Only changes are logged, so the lowest voltage is the lowest one "
                 "the battery events mention.")
    if battery["transitions"]:
        lines.append("")
        lines += _table(
            ["When", "State", "Voltage", "Percent (rough)"],
            [[_stamp(t["at"]), t["state"], "n/a" if t["voltage"] is None else f"{t['voltage']:.2f} V",
              "n/a" if t["percent"] is None else f"{t['percent']:.0f} %"] for t in battery["transitions"][:MAX_LISTED]],
        )

    lines += ["", "## Conversations", ""]
    lines.append(f"Started by a person: {conv['user_initiated']}. Started by Laika: {conv['laika_initiated']}, "
                 f"answered {conv['laika_answered']}"
                 + ("" if conv["answered_rate"] is None else f" ({conv['answered_rate']:.0%})") + ".")
    if conv["started_not_completed"]:
        lines.append(f"Started but never completed (a restart or a crash): {_counts(conv['started_not_completed'])}.")
    if conv["asks_in_quiet_hours"]:
        lines.append(f"Warning: {conv['asks_in_quiet_hours']} proactive ask(s) started inside quiet hours.")
    dnd = conv["do_not_disturb"]
    lines.append(f"Quiet hours: {dnd['configured'] or 'not known (not read from the config)'}; "
                 f"do-not-disturb stored until {_stamp(dnd['do_not_disturb_until'])}; busy until {_stamp(dnd['busy_until'])}.")

    if len(report["per_day"]) > 1 and period["days"] > 1.0001:
        lines += ["", "## Per day", ""]
        keys = ["images", "observations", "interesting", "moves", "safety_stops", "questions_queued", "asks", "answers",
                "loops_completed", "facts_created", "restarts", "watchdog", "hardware_unavailable", "outage_s"]
        names = ["pictures", "observations", "interesting", "moves", "safety stops", "queued", "asks", "answers",
                 "loops", "facts", "restarts", "watchdog", "hw errors", "AI down"]
        rows = []
        for day in report["per_day"]:
            rows.append([day["date"]] + [(_dur(day.get(k, 0)) if k == "outage_s" and day.get(k) else day.get(k, 0)) for k in keys])
        lines += _table(["Day"] + names, rows)

    lines += ["", "## Against the suggested bar", ""]
    lines += _table(["Check", "Value", "Status"], [[b["check"], b["value"], b["status"]] for b in report["bar"]])
    lines += ["", "The bar is in docs/field-trial.md; it is a suggestion, and the verdict is yours.", ""]
    lines += ["## Not visible in this report", ""] + [f"- {text}" for text in report["not_visible"]]
    lines += ["", "## Data read", ""]
    lines.append(f"- Data directory: `{src['data_dir']}`")
    lines.append(f"- Event files: {', '.join(src['event_files']) or 'none'}; first event {_stamp(src['log_first_event'])}, last {_stamp(src['log_last_event'])}")
    if src["unknown_event_types"]:
        lines.append(f"- Event types this version does not know (ignored): {_counts(src['unknown_event_types'])}")
    for error in src["read_errors"]:
        lines.append(f"- {error}")
    if src["store_available"]:
        lines.append(f"- Store (read-only): {_counts(src['store_totals'])} rows in total" + (f"; {src['store_note']}" if src["store_note"] else ""))
        for error in src["store_errors"]:
            lines.append(f"- Store query problem: {error}")
    else:
        lines.append(f"- Store: {src['store_note']}")
    return "\n".join(lines).rstrip() + "\n"


# --- writing and the command line ---------------------------------------------------------------


def write_report(report: Mapping[str, Any], directory: Path, when: date) -> tuple[Path, Path]:
    """`trial-report-<date>.json` and `.md` in `directory`; a second run on the same day replaces them."""
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"trial-report-{when.isoformat()}.json"
    md_path = directory / f"trial-report-{when.isoformat()}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    for path in (json_path, md_path):
        try:
            os.chmod(path, FILE_MODE)
        except OSError:
            pass
    return json_path, md_path


def parse_since(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"--since must look like 2026-10-15, got {text!r}") from error


def resolve_period(days: int | None, since: date | None, now: float) -> tuple[float, float]:
    """`--days N` ending now, or from the start of `--since` (local midnight) for N days or until now."""
    if days is not None and days < 1:
        raise ValueError("--days must be at least 1")
    if since is None:
        return now - (days or 1) * DAY_S, now
    start = datetime(since.year, since.month, since.day).timestamp()
    end = now if days is None else min(now, (datetime(since.year, since.month, since.day) + timedelta(days=days)).timestamp())
    if start >= end:
        raise ValueError(f"--since {since.isoformat()} is not in the past")
    return start, end


def summary_text(report: Mapping[str, Any]) -> str:
    sup, loops, conv, core = report["supervision"], report["loops"], report["conversations"], report["core_loop"]
    period = report["period"]
    return "\n".join(
        [
            f"Laika trial report: {_stamp(period['start'])} to {_stamp(period['end'])} ({_days(period['days'])})",
            f"  sees {core['sees']['images']} pictures, understands {core['understands']['observations_created']} observations, "
            f"notices {core['notices']['interesting_changes']} interesting changes, moves {core['acts']['moves']['started']}",
            f"  questions queued {core['asks']['questions_queued'] if core['asks']['questions_queued'] is not None else 'n/a'}, "
            f"asked by Laika {core['asks']['proactive_asks']}, answered {conv['laika_answered']}",
            f"  completed loops {len(loops['completed'])}; open: {len(loops['open_asked_unanswered'])} asked and unanswered, "
            f"{len(loops['open_queued_never_asked'])} never asked",
            f"  supervision: {sup['attention_stops']} stop(s) to explain, {len(sup['local_watchdog_expired'])} watchdog, "
            f"{len(sup['hardware_unavailable'])} hardware, {len(sup['restarts'])} restart(s), "
            f"AI down {_dur(sup['ai_outage_total_s'])}",
            f"  battery: {_battery_summary(report['battery'])}",
            f"  log lines skipped as unreadable: {report['sources']['bad_lines']}",
        ]
    )


def run_from_args(args: Any, config: Any, *, now: float | None = None) -> int:
    """`--report`: exit 0 when the report was written, 2 for a bad option or a missing data directory."""
    current = time.time() if now is None else now
    try:
        since = parse_since(args.since) if args.since else None
        start, end = resolve_period(args.days, since, current)
    except ValueError as error:
        print(f"[report] {error}")
        return 2
    data_dir = resolve_data_dir(config.storage.data_dir)
    if not data_dir.is_dir():
        print(f"[report] the data directory {data_dir} does not exist. Set LAIKA_DATA_DIR (the service uses /var/lib/laika).")
        return 2
    quiet = getattr(getattr(config, "interaction", None), "quiet_hours", None)
    report = build_report(data_dir, start, end, now=current, quiet_hours=quiet)
    try:
        json_path, md_path = write_report(report, DataPaths(data_dir).logs, datetime.fromtimestamp(current).date())
    except OSError as error:
        print(f"[report] could not write the report: {error}")
        return 1
    print(summary_text(report))
    print(f"  report: {md_path}")
    print(f"  data:   {json_path}")
    return 0
