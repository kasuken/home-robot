"""The unattended checklist: is the robot ready to drive around the house without anyone watching? (#51)

`run-laika.py --unattended-check` prints it and exits non-zero when a required item is not met. It
only reads: the hardware-check reports under `<data_dir>/logs/`, the settings, and (on Linux, best
effort) whether the systemd unit is installed and enabled. It never starts the robot. It does not
decide for the owner: turning `allow_autonomous_driving` on is deliberate, and at startup a missing
item only prints a warning (docs/safety-calibration.md, "Unattended checklist").
"""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from laika.house_rules import canonical_room, rooms_in_rule

if TYPE_CHECKING:
    from laika.config import Config

OK, FAIL, SKIP, INFO = "ok", "fail", "skip", "info"
UNIT_NAME = "laika.service"
UNIT_PATH = Path("/etc/systemd/system") / UNIT_NAME
ENABLED_LINK = Path("/etc/systemd/system/multi-user.target.wants") / UNIT_NAME
REPORT_GLOB = "hardware-check-*.json"


@dataclass(frozen=True)
class Item:
    name: str
    status: str  # ok, fail (required and not met), skip (cannot be checked here), info (not a gate)
    detail: str


def latest_hardware_report(logs: Path) -> tuple[Path | None, dict | None]:
    """The newest hardware-check JSON report in `logs` and what it holds (None when unreadable)."""
    try:
        reports = sorted(logs.glob(REPORT_GLOB), key=lambda p: p.name)
    except OSError:
        return None, None
    if not reports:
        return None, None
    path = reports[-1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return path, None
    return path, data if isinstance(data, dict) else None


def taught_no_go_rooms(db_path: Path) -> set[str]:
    """The no-go rooms in the rule facts of the store, read-only (the robot may be running). Empty when unreadable."""
    if not db_path.is_file():
        return set()
    try:
        conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2.0)
    except sqlite3.Error:
        return set()
    try:
        rows = conn.execute("SELECT text FROM facts WHERE kind = 'rule'").fetchall()
    except sqlite3.Error:
        return set()
    finally:
        conn.close()
    rooms: set[str] = set()
    for (text,) in rows:
        rooms |= rooms_in_rule(str(text))
    return rooms


def _hardware_item(report: tuple[Path | None, dict | None]) -> Item:
    path, data = report
    name = "hardware check passed on the robot"
    if path is None:
        return Item(name, FAIL, "no report in the logs folder: run `run-laika.py --check-hardware` on the robot")
    if data is None:
        return Item(name, FAIL, f"{path.name} cannot be read")
    if data.get("mode") != "robot":
        return Item(name, FAIL, f"the latest report ({path.name}) is from the simulator: run the check on the robot")
    if data.get("result") != "pass":
        return Item(name, FAIL, f"the latest report ({path.name}) did not pass: fix it and run the check again")
    return Item(name, OK, f"{path.name}")


def _battery_item(config: Config, data: dict | None) -> Item:
    name = "battery calibrated"
    if not config.battery.enabled:
        return Item(name, FAIL, "[battery] enabled is false: she would not rest when the battery is low")
    if config.battery.voltage_offset != 0:
        return Item(name, OK, f"[battery] voltage_offset = {config.battery.voltage_offset:g}")
    if data is not None and data.get("result") == "pass":
        recorded = (data.get("suggestedSettings") or {}).get("voltage_offset")
        used = (data.get("limits") or {}).get("battery_voltage_offset")
        if recorded is not None or (isinstance(used, (int, float)) and used != 0):
            return Item(name, OK, "a voltage offset is recorded in the latest passed hardware-check report")
    return Item(
        name, FAIL,
        "compare the reading with a multimeter in the hardware check and set [battery] voltage_offset "
        "(0 only counts when the report records it)",
    )


def default_service_probe() -> tuple[str, str]:
    """(status, detail) for the systemd unit: best effort, skipped away from Linux."""
    if not sys.platform.startswith("linux"):
        return SKIP, "not checked: systemd is Linux only"
    if not UNIT_PATH.exists():
        return FAIL, f"{UNIT_PATH} is missing: run `sudo deploy/install-service.sh` (docs/service.md)"
    if not (ENABLED_LINK.exists() or ENABLED_LINK.is_symlink()):
        return FAIL, f"{UNIT_NAME} is installed but not enabled: run `sudo systemctl enable {UNIT_NAME}`"
    return OK, f"{UNIT_NAME} is installed and enabled"


def evaluate(
    config: Config,
    data_dir: Path,
    *,
    service_probe: Callable[[], tuple[str, str]] | None = None,
    taught_rooms: set[str] | None = None,
) -> list[Item]:
    """The checklist for these settings and this data directory. `taught_rooms` are the no-go rooms
    people taught her (the store's rule facts); the configured rooms are counted too."""
    report = latest_hardware_report(data_dir / "logs")
    robot = config.robot
    items = [
        _hardware_item(report),
        _battery_item(config, report[1]),
    ]
    status, detail = (service_probe or default_service_probe)()
    items.append(Item("service installed and enabled (stops the motors on every exit)", status, detail))
    items.append(
        Item(
            "allow_autonomous_driving turned on deliberately",
            OK if robot.allow_autonomous_driving else FAIL,
            "[robot] allow_autonomous_driving = true"
            if robot.allow_autonomous_driving
            else "[robot] allow_autonomous_driving is false: idle activities only look around",
        )
    )
    items.append(
        Item(
            "watchdog enabled",
            OK if robot.watchdog_timeout_s > 0 else FAIL,
            f"[robot] watchdog_timeout_s = {robot.watchdog_timeout_s:g}"
            if robot.watchdog_timeout_s > 0
            else "[robot] watchdog_timeout_s is 0: a stalled safety monitor would not stop the motors",
        )
    )
    hour, day = robot.max_autonomous_drive_s_per_hour, robot.max_autonomous_drive_s_per_day
    items.append(
        Item(
            "driving budgets set",
            OK if hour and day else FAIL,
            f"{hour:g} s per hour, {day:g} s per day" if hour and day else
            "set both [robot] max_autonomous_drive_s_per_hour and max_autonomous_drive_s_per_day (0 means no limit)",
        )
    )
    configured = {room for room in (canonical_room(r) for r in config.house.no_go_rooms) if room}
    taught = taught_rooms or set()
    items.append(
        Item(
            "house rules",
            INFO,
            f"{len(configured | taught)} no-go room(s): {len(configured)} in [house], {len(taught)} taught by voice"
            + ("" if configured or taught else " (nothing is off limits)"),
        )
    )
    return items


def unmet(items: list[Item]) -> list[Item]:
    return [item for item in items if item.status == FAIL]


def render(items: list[Item]) -> str:
    marks = {OK: "[ok]  ", FAIL: "[FAIL]", SKIP: "[skip]", INFO: "[info]"}
    lines = ["Laika unattended checklist", ""]
    lines += [f"{marks[item.status]} {item.name}: {item.detail}" for item in items]
    missing = unmet(items)
    lines.append("")
    lines.append(
        f"Not met: {len(missing)}. Do not leave her driving on her own until they are."
        if missing
        else "All required items are met. The decision to let her drive unattended is still yours."
    )
    return "\n".join(lines)


def startup_warning(config: Config, data_dir: Path) -> str | None:
    """One line for the console when autonomous driving is on but the checklist is not met (never blocks)."""
    if not config.robot.allow_autonomous_driving:
        return None
    missing = unmet(evaluate(config, data_dir))
    if not missing:
        return None
    names = "; ".join(item.name for item in missing)
    return (
        f"autonomous driving is on but the unattended checklist is not met ({names}): "
        "run `run-laika.py --unattended-check` and do not leave her alone yet"
    )


def run_from_args(config: Config) -> int:
    """`run-laika.py --unattended-check`: print the checklist; 0 when every required item is met, else 1."""
    from laika.storage import DataPaths, resolve_data_dir
    from laika.store import DB_NAME

    paths = DataPaths(resolve_data_dir(config.storage.data_dir))
    items = evaluate(config, paths.root, taught_rooms=taught_no_go_rooms(paths.memory / DB_NAME))
    print(render(items))
    return 1 if unmet(items) else 0
