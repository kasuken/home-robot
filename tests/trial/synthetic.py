"""A synthetic data directory for the trial report tests: three days, one rotated log file, a store."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from laika.events import Event, EventType
from laika.observations import Observation, SeenObject
from laika.store import DB_NAME, Store

E = EventType


def at(day: int, hh: int, mm: int = 0, ss: int = 0) -> float:
    """Local time on 2026-10-(10 + day)."""
    return datetime(2026, 10, 10 + day, hh, mm, ss).timestamp()


START = at(0, 0)
END = at(2, 23)  # "now" of the scenario


def line(kind: EventType, ts: float, **data) -> str:
    return Event(kind, data, ts, id=f"evt-{abs(hash((kind.value, ts))) % 10**8:08d}").to_json()


def old_events() -> list[str]:
    """Day 0 up to the first obstacle: this goes to the rotated file."""
    return [
        line(E.HARDWARE_READY, at(0, 8), robotId="laika-sim", name="Laika"),
        line(E.IMAGE_CAPTURED, at(0, 8, 5), reason="presence", room="kitchen"),
        line(E.OBSERVATION_CREATED, at(0, 8, 5, 5), objects=["cup", {"name": "backpack"}], people=0,
             changes=[{"kind": "new", "object": "backpack"}]),
        line(E.INTERESTING_CHANGE_DETECTED, at(0, 8, 5, 6), reasons=["new in kitchen: backpack"],
             objects=["backpack"], total=0.6),
        line(E.IMAGE_CAPTURED, at(0, 8, 6), reason="investigate", room="kitchen"),
        line(E.IMAGE_CAPTURED, at(0, 8, 6, 10), reason="investigate", room="kitchen"),
        line(E.MOVEMENT_STARTED, at(0, 8, 7), direction="forward"),
        line(E.MOVEMENT_COMPLETED, at(0, 8, 7, 1), result="success"),
        line(E.MOVEMENT_STARTED, at(0, 8, 10), direction="forward"),
        line(E.OBSTACLE_DETECTED, at(0, 8, 10, 1), distanceCm=9),
        line(E.MOVEMENT_FAILED, at(0, 8, 10, 1), result="cancelled", reason="obstacle"),
    ]


def new_events() -> list[str]:
    return [
        line(E.PERSON_DETECTED, at(0, 9, 30), count=1),
        line(E.INTERACTION_STARTED, at(0, 9, 31), initiator="laika", reason="pending_question", questionId="q-1"),
        line(E.INTERACTION_COMPLETED, at(0, 9, 32), initiator="laika", answered=True, questionId="q-1"),
        line(E.INTERACTION_STARTED, at(0, 10, 30), initiator="laika", reason="pending_question", questionId="q-3"),
        line(E.INTERACTION_COMPLETED, at(0, 10, 31), initiator="laika", answered=False, questionId="q-3"),
        line(E.AI_ENDPOINT_UNAVAILABLE, at(0, 14), error="timeout"),
        line(E.AI_ENDPOINT_RECOVERED, at(0, 14, 25)),
        # day 1: a restart, a conversation, a watchdog expiry, a hardware error, an outage over a restart
        line(E.HARDWARE_READY, at(1, 7), robotId="laika-sim", name="Laika"),
        line(E.INTERACTION_STARTED, at(1, 8), initiator="user"),
        line(E.INTERACTION_COMPLETED, at(1, 8, 5), initiator="user"),
        line(E.LOCAL_WATCHDOG_EXPIRED, at(1, 11), reason="monitor_stalled", ageS=1.2, timeoutS=1.0),
        line(E.MOVEMENT_FAILED, at(1, 11), result="cancelled", reason="watchdog"),
        line(E.HARDWARE_UNAVAILABLE, at(1, 11, 1), device="camera", error="boom"),
        line(E.AI_ENDPOINT_UNAVAILABLE, at(1, 22), error="dns"),
        # day 2: restart ends the outage, a proactive ask without a queued question, a new outage left open
        line(E.HARDWARE_READY, at(2, 6), robotId="laika-sim", name="Laika"),
        line(E.INTERACTION_STARTED, at(2, 7), initiator="laika", reason="pending_question", questionId="q-9"),
        line(E.INTERACTION_COMPLETED, at(2, 7, 1), initiator="laika", answered=False, questionId="q-9"),
        line(E.IMAGE_CAPTURED, at(2, 8), reason="presence", room="kitchen"),
        line(E.INTERACTION_STARTED, at(2, 9), initiator="user"),
        line(E.INTERACTION_COMPLETED, at(2, 9, 2), initiator="user"),
        line(E.MOVEMENT_FAILED, at(2, 10), result="rejected", reason="preempted"),
        line(E.AI_ENDPOINT_UNAVAILABLE, at(2, 15), error="timeout"),
    ]


def write_logs(data_dir: Path, *, damage: bool = True) -> None:
    logs = data_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "events.jsonl.1").write_text("\n".join(old_events()) + "\n", encoding="utf-8")
    current = new_events()
    if damage:
        current[3:3] = [
            "this is not json",
            json.dumps({"id": "evt-x", "type": "InteractionStarted", "data": {}}),  # no timestamp
            line(E.HARDWARE_READY, at(0, 12), robotId="x", name="x").replace("HardwareReady", "FutureThing"),
        ]
    text = "\n".join(current) + "\n"
    if damage:
        text += '{"id": "evt-cut", "type": "ImageCap'  # a half-written last line
    (logs / "events.jsonl").write_text(text, encoding="utf-8")


def _episode(n: int, ts: float, kind: str, subject: str, room: str | None) -> dict:
    return {"id": f"ep-{n}", "timestamp": ts, "kind": kind, "summary": f"SECRET summary about {subject}",
            "subjects": [subject], "location": room}


def write_store(data_dir: Path, *, keep_open: bool = False) -> Store:
    memory = data_dir / "memory"
    memory.mkdir(parents=True, exist_ok=True)
    store = Store(memory / DB_NAME, max_observation_days=0, max_episode_days=0)
    episodes = [
        _episode(1, at(0, 8, 7), "question", "backpack", "kitchen"),  # q-1 of the first run
        _episode(2, at(0, 8, 20), "question", "vase", "living_room"),  # q-2, never asked
        _episode(3, at(0, 9, 31, 30), "answer", "backpack", "kitchen"),
        _episode(4, at(0, 10), "question", "toy", "bedroom"),  # q-3, asked, never answered
        _episode(5, at(0, 8, 5, 6), "new", "backpack", "kitchen"),
        _episode(6, at(0, 8, 5, 7), "moved", "cup", "kitchen"),
        _episode(7, at(1, 7, 30), "question", "lamp", "office"),  # q-1 of the second run
        _episode(8, at(1, 8, 3), "answer", "lamp", "office"),  # answered during the user's conversation
        _episode(9, at(0, 8, 5), "seen", "cup", "kitchen"),
    ]
    for episode in episodes:
        store.put_episode(episode)

    def fact(fid, subject, kind, source, created, updated):
        store.put_fact({"id": fid, "text": f"SECRET text {subject}", "subject": subject, "kind": kind,
                        "source": source, "created": created, "updated": updated, "confidence": 1.0})

    fact("f-1", "backpack", "general", "user", at(0, 9, 31, 30), at(0, 9, 31, 30))
    fact("f-2", "lamp", "general", "user", at(1, 8, 3), at(1, 8, 3))
    fact("f-3", "cup", "location", "investigation", at(0, 8, 8), at(1, 9))
    for name, first, source, note in (
        ("backpack", at(0, 8, 5), "conversation", "SECRET note"),
        ("cup", at(0, 8, 5), "vision", None),
        ("sofa", at(0, 8) - 5 * 86400, "vision", None),
    ):
        store.put_world_object({"name": name, "room": "kitchen", "first_seen": first, "last_seen": first,
                                "source": source, "note": note})
    for n, ts in enumerate((at(0, 8, 5, 5), at(0, 8, 6), at(0, 9), at(2, 8))):
        store.add_observation(
            Observation("I see", (SeenObject("cup"),), 0.9, "kitchen", 1 if n == 2 else 0, ts, f"obs-{n:08x}")
        )
    store.kv_put("interaction", "do_not_disturb_until", at(2, 22))
    store.flush()
    if not keep_open:
        store.close()
    return store
