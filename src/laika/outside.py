"""Life outside the windows (#48): the weather, the light and what happens on the street.

`WindowWatcher` is an activity under a CURIOSITY lease. In a room with a known window (`windows.py`) it points
the head there, tilted up a little, takes one picture with the reason "outside" and asks the vision model what
is outside with a dedicated prompt. The reply is strict JSON, validated here like an observation:

    {"weather": sunny|cloudy|rain|snow|fog|unknown, "light": day|dusk|night|unknown,
     "description": "one short English sentence", "events": ["a car passed", ...],
     "people_outside": whole number, "confidence": 0 to 1}

People outside are only counted: the prompt forbids describing or naming them and the schema has no room for
it. An `OutsideObservation` is kept as an `outside` episode; compared with the previous one it can also give
`outside_change` episodes ("started raining", "it got dark", "the street got busy") and an
InterestingChangeDetected through the curiosity gate, so the interaction policy may mention it later. The
watcher itself never speaks. The agent's context carries the latest outside observation, once, and the
`look_outside` tool answers "com'è fuori?" (docs/behaviours.md, "Looking outside").

Cost and privacy: every look is a photo of the outside sent to the vision model, so a window is looked out of at
most every `[outside] min_interval_s`, all windows at most `max_per_day` times a day, and the frame shares the
vision limits. The picture is saved like any other (normal image retention). In the simulator the answer comes
from the simulated house's ground truth and nothing leaves the machine.

Without navigation she can only watch a window of the room she is in. When to look is the idle selector's
business (#47): it asks `due_windows()` and calls `watch`; nothing in here runs on a timer of its own.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Mapping

from laika.behaviour import Arbiter, Lease, Preempted, Priority
from laika.camera import Camera
from laika.config import OutsideConfig
from laika.curiosity import InterestGate, InterestScore
from laika.investigate import _heartbeat
from laika.limits import TILT_MAX
from laika.memory import iso_minutes, relative_time
from laika.observations import (
    MAX_PEOPLE,
    InvalidObservation,
    Observation,
    _load_json,
    _unit,
    normalise,
)
from laika.robot import ImageObservation, Robot
from laika.windows import Window, WindowRegistry

if TYPE_CHECKING:
    from laika.memory import Memory
    from laika.store import Store
    from laika.vision import VisionDescriber
    from laika.world_model import WorldModel

log = logging.getLogger(__name__)
# The smallest gap between two looks' timestamps (see `_remember`).
MIN_LOOK_STEP_S = 0.001

NAME = "window-watch"
CAPTURE_REASON = "outside"
NAMESPACE = "outside"
LATEST_KEY = "latest"
WEATHER = ("sunny", "cloudy", "rain", "snow", "fog", "unknown")
LIGHT = ("day", "dusk", "night", "unknown")
MAX_DESCRIPTION_CHARS = 300
MAX_EVENTS = 5
MAX_EVENT_CHARS = 80
# The change episodes are about the street: a look that finds it very different from the last one.
BUSY_EXTRA_PEOPLE = 3
# A look is compared with the previous one only when they are this close in time: "it got dark" after a
# night's sleep is not news.
CHANGE_MAX_AGE_S = 6 * 3600.0
# A look this recent is reused when somebody asks how it is outside (saves a vision request).
REUSE_S = 120.0
# The agent mentions what she saw only while it is still news.
NOTE_MAX_AGE_S = 6 * 3600.0
NOTE_MAX_CHANGES = 3
SETTLE_S = 0.8
TILT_UP_DEG = 10.0  # "tilt up a little": windows are above where a scan looks
OBSERVATION_TIMEOUT_S = 90.0
HISTORY = 16
EPISODE_SUBJECTS = ("outside", "weather", "window")

# The English sentence of a weather change, "from" "to": the reason string is "outside: " + this.
_WEATHER_CHANGES = {
    ("rain", None): "stopped raining",
    ("snow", None): "stopped snowing",
    ("fog", None): "the fog cleared",
    (None, "rain"): "started raining",
    (None, "snow"): "started snowing",
    (None, "fog"): "fog came in",
    ("cloudy", "sunny"): "the sun came out",
    ("rain", "sunny"): "stopped raining, the sun came out",
    ("sunny", "cloudy"): "it clouded over",
}
_LIGHT_CHANGES = {
    ("day", "dusk"): "it is getting dark",
    ("day", "night"): "it got dark",
    ("dusk", "night"): "it got dark",
    ("night", "dusk"): "it is getting light",
    ("night", "day"): "it got light",
    ("dusk", "day"): "it got light",
}


# --- the observation -----------------------------------------------------------------------------


@dataclass(frozen=True)
class OutsideObservation:
    weather: str
    light: str
    description: str
    events: tuple[str, ...] = ()
    # A count and nothing else: people outside are never described or identified.
    people_outside: int = 0
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: f"out-{uuid.uuid4().hex[:8]}")
    room: str | None = None
    window: str | None = None
    pan: float = 0.0
    tilt: float = 0.0
    image_path: str | None = None
    source: str = "vision"

    @property
    def known(self) -> bool:
        """Whether the look told anything: a frame of a wall or a curtain says "unknown" twice."""
        return self.weather != "unknown" or self.light != "unknown"

    @property
    def summary(self) -> str:
        """Short English for episodes and the agent: "rain, dusk. Wet street. A car passed. 2 people outside"."""
        parts = [f"{self.weather}, {self.light}." if self.known else "Nothing could be made out."]
        parts.append(self.description.rstrip(".") + ".")
        if self.events:
            parts.append("; ".join(self.events).capitalize() + ".")
        if self.people_outside:
            parts.append(f"{self.people_outside} {'person' if self.people_outside == 1 else 'people'} outside.")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "timestamp": self.timestamp, "weather": self.weather, "light": self.light,
            "description": self.description, "events": list(self.events), "people_outside": self.people_outside,
            "confidence": self.confidence, "room": self.room, "window": self.window, "pan": self.pan,
            "tilt": self.tilt, "image_path": self.image_path, "source": self.source,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> OutsideObservation:
        """Rebuild from `to_dict` output (stored); raises InvalidObservation."""
        if not isinstance(raw, Mapping):
            raise InvalidObservation("an outside observation must be an object")
        fields = _validate_outside(raw)
        stamp = raw.get("timestamp")
        if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
            raise InvalidObservation("timestamp must be a number")
        room, window, path = raw.get("room"), raw.get("window"), raw.get("image_path")
        return cls(
            **fields,
            timestamp=float(stamp),
            id=str(raw.get("id") or f"out-{uuid.uuid4().hex[:8]}"),
            room=normalise(room) if isinstance(room, str) and room.strip() else None,
            window=window if isinstance(window, str) else None,
            pan=float(raw.get("pan", 0.0) or 0.0),
            tilt=float(raw.get("tilt", 0.0) or 0.0),
            image_path=path if isinstance(path, str) else None,
            source="simulator" if raw.get("source") == "simulator" else "vision",
        )


def _choice(value: Any, allowed: tuple[str, ...], label: str) -> str:
    if not isinstance(value, str) or " ".join(value.split()).lower() not in allowed:
        raise InvalidObservation(f"{label} must be one of: {', '.join(allowed)}")
    return " ".join(value.split()).lower()


def _validate_outside(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Check the model's fields (or the stored ones): the shared core of the parsers."""
    missing = [key for key in ("weather", "light", "description", "confidence") if key not in raw]
    if missing:
        raise InvalidObservation(f"missing keys: {', '.join(missing)}")
    description = raw["description"]
    if not isinstance(description, str):
        raise InvalidObservation("description must be a string")
    description = " ".join(description.split())
    if not description:
        raise InvalidObservation("description is empty")
    if len(description) > MAX_DESCRIPTION_CHARS:
        raise InvalidObservation(f"description is longer than {MAX_DESCRIPTION_CHARS} characters")
    items = raw.get("events", [])
    if not isinstance(items, (list, tuple)):
        raise InvalidObservation("events must be a list")
    if len(items) > MAX_EVENTS:
        raise InvalidObservation(f"too many events: {len(items)} (limit {MAX_EVENTS})")
    events: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise InvalidObservation("each event must be a short text")
        text = " ".join(item.split())
        if not text or len(text) > MAX_EVENT_CHARS:
            raise InvalidObservation(f"an event must have 1 to {MAX_EVENT_CHARS} characters")
        events.append(text)
    people = raw.get("people_outside", 0)
    # A count only: a list of people or a description there is refused, never trimmed to something else.
    if isinstance(people, bool) or not isinstance(people, int) or not 0 <= people <= MAX_PEOPLE:
        raise InvalidObservation(f"people_outside must be a whole number from 0 to {MAX_PEOPLE}")
    return {
        "weather": _choice(raw["weather"], WEATHER, "weather"),
        "light": _choice(raw["light"], LIGHT, "light"),
        "description": description,
        "events": tuple(dict.fromkeys(events)),
        "people_outside": people,
        "confidence": _unit(raw["confidence"], "confidence"),
    }


def parse_outside_observation(
    output: str | Mapping[str, Any],
    *,
    image: ImageObservation,
    room: str | None,
    window: Window | None = None,
    image_path: str | None = None,
) -> OutsideObservation:
    """Validate untrusted model output into an OutsideObservation, or raise InvalidObservation.

    The model only supplies the fields of the schema; time, place, camera pose and the saved picture come
    from the robot. Unknown keys are ignored, so a model that adds "people" descriptions gets them dropped.
    """
    raw = _load_json(output) if isinstance(output, str) else output
    if not isinstance(raw, Mapping):
        raise InvalidObservation("model output must be a JSON object")
    fields = _validate_outside(raw)
    extra = set(raw) - {"weather", "light", "description", "events", "people_outside", "confidence"}
    if extra:
        log.debug("ignoring unknown outside keys: %s", sorted(extra))
    return OutsideObservation(
        **fields,
        timestamp=image.timestamp,
        room=normalise(room) if room else None,
        window=window.id if window else None,
        pan=image.camera_pan,
        tilt=image.camera_tilt,
        image_path=image_path,
        source="vision",
    )


def outside_from_scene(
    image: ImageObservation, room: str | None, window: Window | None = None, image_path: str | None = None
) -> OutsideObservation | None:
    """An outside observation from the simulator's ground truth, or None when no window is in the frame."""
    truth = image.outside
    if truth is None:
        return None
    weather, light = truth.get("weather", "unknown"), truth.get("light", "unknown")
    events = tuple(str(e) for e in truth.get("events", ()))[:MAX_EVENTS]
    description = f"{weather.capitalize()} outside, {light}time." if weather != "unknown" else "Cannot tell outside."
    return OutsideObservation(
        weather=weather if weather in WEATHER else "unknown",
        light=light if light in LIGHT else "unknown",
        description=description,
        events=events,
        people_outside=min(int(truth.get("people", 0) or 0), MAX_PEOPLE),
        confidence=1.0,
        timestamp=image.timestamp,
        room=normalise(room) if room else None,
        window=window.id if window else None,
        pan=image.camera_pan,
        tilt=image.camera_tilt,
        image_path=image_path,
        source="simulator",
    )


def detect_changes(previous: OutsideObservation | None, current: OutsideObservation) -> list[tuple[str, str]]:
    """What differs from the last look, as (kind, English text): "weather", "light" or "street".

    Only looks that told something are compared, and only when they are close in time.
    """
    if previous is None or not previous.known or not current.known:
        return []
    if not 0 <= current.timestamp - previous.timestamp <= CHANGE_MAX_AGE_S:
        return []
    changes: list[tuple[str, str]] = []
    if previous.weather not in ("unknown", current.weather) and current.weather != "unknown":
        text = (
            _WEATHER_CHANGES.get((previous.weather, current.weather))
            or _WEATHER_CHANGES.get((previous.weather, None))
            or _WEATHER_CHANGES.get((None, current.weather))
            or f"the weather changed from {previous.weather} to {current.weather}"
        )
        changes.append(("weather", text))
    if previous.light not in ("unknown", current.light) and current.light != "unknown":
        text = _LIGHT_CHANGES.get((previous.light, current.light))
        if text is not None:
            changes.append(("light", text))
    busy_before = previous.people_outside + len(previous.events)
    busy_now = current.people_outside + len(current.events)
    if busy_now >= busy_before + BUSY_EXTRA_PEOPLE:
        changes.append(("street", "the street got busy"))
    return changes


# --- asking the vision model --------------------------------------------------------------------

OUTSIDE_SYSTEM_PROMPT = f"""You are the eyes of Laika, a small home robot. You receive one camera frame \
taken through a window and describe what is happening OUTSIDE the window: the weather, the light, and \
anything notable on the street, in the garden or in the sky.

Reply with STRICT JSON only: no markdown, no code fences, no text outside the object. Schema:
{{"weather": "sunny" | "cloudy" | "rain" | "snow" | "fog" | "unknown",
 "light": "day" | "dusk" | "night" | "unknown",
 "description": "one short English sentence about the outside",
 "events": ["short English items, for example a car passed, birds on the balcony"],
 "people_outside": whole number,
 "confidence": 0.0-1.0}}

Rules:
- Describe only what you can really see outside; the room, the curtains and the glass itself do not count. \
If the frame shows no outside (a wall, a closed shutter, a reflection), use "unknown" for weather and light \
and lower the confidence.
- At most {MAX_EVENTS} events, each a few words, only things that are actually happening or just happened.
- People: only count the people you can see outside, in "people_outside". Never identify anyone, never name \
anyone, never describe faces, clothes or other identifying features, not even in the events.
- "confidence" is how sure you are of the whole answer; lower it when it is dark, blurry or unclear."""


def outside_user_prompt(room: str | None) -> str:
    where = f" The window is in the {room.replace('_', ' ')}." if room else ""
    return f"Describe what is outside this window.{where} Reply with the JSON object only."


class OutsideDescriber:
    """Callable `(image, room, window, image_path) -> OutsideObservation | None`: the dedicated outside prompt.

    It goes through `VisionDescriber.ask`, so the frame limits, the per-minute cap and the outage handling are
    the vision ones. A malformed reply raises InvalidObservation.
    """

    def __init__(self, vision: VisionDescriber) -> None:
        self.vision = vision

    def __call__(
        self, image: ImageObservation, room: str | None, window: Window | None, image_path: str | None
    ) -> OutsideObservation | None:
        text = self.vision.ask(image, OUTSIDE_SYSTEM_PROMPT, outside_user_prompt(room))
        if text is None:
            return None
        return parse_outside_observation(text, image=image, room=room, window=window, image_path=image_path)


Describe = Callable[[ImageObservation, "str | None", "Window | None", "str | None"], "OutsideObservation | None"]


# --- the watcher -----------------------------------------------------------------------------------


@dataclass
class WatchResult:
    """How one look went: "done", "not_seen", "no_answer", "no_camera", "refused", "aborted", "error"."""

    status: str
    window: Window | None = None
    observation: OutsideObservation | None = None
    changes: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "done"


def local_day(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).date().isoformat()


class _Reserved(Exception):
    """A look that the interval or the daily cap does not allow."""


class WindowWatcher:
    def __init__(
        self,
        arbiter: Arbiter,
        robot: Robot,
        camera: Camera,
        windows: WindowRegistry,
        world: WorldModel,
        memory: Memory,
        config: OutsideConfig,
        *,
        describe: Describe | None = None,
        gate: InterestGate | None = None,
        store: Store | None = None,
        clock: Callable[[], float] = time.time,
        settle_s: float = SETTLE_S,
        sleep: Callable[[float], None] = time.sleep,
        timeout_s: float = OBSERVATION_TIMEOUT_S,
    ) -> None:
        self.arbiter = arbiter
        self.robot = robot
        self.camera = camera
        self.windows = windows
        self.world = world
        self.memory = memory
        self.config = config
        # Without a describe callable only the simulator's ground truth can be read.
        self.describe = describe
        self.gate = gate
        self.store = store
        self._clock = clock
        self.settle_s = max(0.0, settle_s)
        self._sleep = sleep
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._one = threading.Lock()  # one look at a time, from any thread
        self._latest: OutsideObservation | None = None
        self._last: dict[str, float] = {}  # window id -> when it was last looked out of
        self._today: deque[float] = deque()
        # What was done (for logs and tests).
        self.results: deque[WatchResult] = deque(maxlen=HISTORY)
        self.rejections: deque[str] = deque(maxlen=HISTORY)
        self._load()

    # --- what is known -----------------------------------------------------------------

    def _load(self) -> None:
        if self.store is None:
            return
        saved = self.store.kv_get(NAMESPACE, LATEST_KEY)
        if saved is None:
            return
        try:
            self._latest = OutsideObservation.from_dict(saved)
        except (InvalidObservation, TypeError, ValueError):
            log.warning("outside: skipping an unreadable stored observation")
            return
        # A restart does not start the intervals again: the window was looked out of when she last did.
        if self._latest.window:
            self._last[self._latest.window] = self._latest.timestamp

    def now(self) -> float:
        return self._clock()

    def latest(self) -> OutsideObservation | None:
        with self._lock:
            return self._latest

    def windows_here(self) -> list[Window]:
        return self.windows.in_room(self.world.current_room)

    def changes_since(self, since: float, limit: int = NOTE_MAX_CHANGES) -> list[tuple[float, str]]:
        """The newest `outside_change` episodes after `since` as (time, text), oldest first."""
        found = [e for e in self.memory.episodes.recent(200) if e.kind == "outside_change" and e.timestamp > since]
        return [(e.timestamp, e.summary.removeprefix("Outside: ")) for e in reversed(found[:limit])]

    def snapshot(self, observation: OutsideObservation | None = None) -> dict[str, Any]:
        """The latest look for a tool result: the picture of what is outside and when it was seen."""
        observation = observation or self.latest()
        now = self._clock()
        if observation is None:
            return {}
        return {
            "when": iso_minutes(observation.timestamp),
            "ago": relative_time(observation.timestamp, now),
            "weather": observation.weather,
            "light": observation.light,
            "description": observation.description,
            "events": list(observation.events),
            "people_outside": observation.people_outside,
            "confidence": observation.confidence,
            "room": observation.room,
        }

    # --- the rate limits -------------------------------------------------------------------

    def _forget_old_days(self, now: float) -> None:
        today = local_day(now)
        while self._today and local_day(self._today[0]) != today:
            self._today.popleft()

    def due(self, window: Window, now: float | None = None) -> bool:
        """Whether `window` may be looked out of now by herself: the interval and the daily cap."""
        now = self._clock() if now is None else now
        with self._lock:
            self._forget_old_days(now)
            if len(self._today) >= self.config.max_per_day:
                return False
            last = self._last.get(window.id)
            return last is None or now - last >= self.config.min_interval_s

    def _reserve(self, window: Window, *, interval: bool) -> None:
        """Count a look before it is made, so a failing one is not retried at once. Raises `_Reserved`."""
        now = self._clock()
        with self._lock:
            self._forget_old_days(now)
            if len(self._today) >= self.config.max_per_day:
                raise _Reserved("daily limit reached")
            last = self._last.get(window.id)
            if interval and last is not None and now - last < self.config.min_interval_s:
                raise _Reserved("looked out of this window a short while ago")
            self._last[window.id] = now
            self._today.append(now)

    # --- the activity ----------------------------------------------------------------------

    def due_windows(self) -> list[Window]:
        """The windows of this room that may be looked out of now by herself (the idle selector asks this)."""
        if not self.config.enabled:
            return []
        return [w for w in self.windows_here() if self.due(w)]

    def poll(self) -> WatchResult | None:
        """Look out of a window of this room if one is due and nobody needs the body. None when nothing is done.

        Pause, a critical battery, a conversation or another activity make the arbiter refuse the lease,
        and nothing happens.
        """
        if self._one.locked() or self.arbiter.holder is not None or self.arbiter.waiting:
            return None
        due = self.due_windows()
        if not due:
            return None
        result = self.watch(due[0])
        return None if result.status == "refused" else result

    def watch(self, window: Window, lease: Lease | None = None) -> WatchResult:
        """Look out of `window` now, honouring the rate limits. Without a `lease` it takes a CURIOSITY one."""
        if lease is None:
            return self._autonomous(window)
        return self._guarded(window, lease, interval=False)

    def _autonomous(self, window: Window) -> WatchResult:
        mine = self.arbiter.request(NAME, Priority.CURIOSITY, resumable=False)
        if mine is None:
            return self._finish(WatchResult("refused", window, reason="the body is busy, paused or the battery is empty"))
        try:
            return self._guarded(window, mine, interval=True)
        finally:
            mine.release()

    def _guarded(self, window: Window, lease: Lease, *, interval: bool) -> WatchResult:
        if not self._one.acquire(blocking=False):
            return WatchResult("refused", window, reason="already looking out of a window")
        try:
            try:
                self._reserve(window, interval=interval)
            except _Reserved as limit:
                return self._finish(WatchResult("refused", window, reason=str(limit)))
            return self._finish(self._look(window, lease))
        finally:
            self._one.release()

    def _finish(self, result: WatchResult) -> WatchResult:
        self.results.append(result)
        if result.status in ("done", "not_seen"):
            outcome = result.observation.summary if result.observation else ""
            print(f"[laika] looked out of the {result.window.label if result.window else 'window'}: "
                  f"{outcome or result.status}")
        else:
            log.info("window watch: %s %s", result.status, result.reason)
        return result

    def _look(self, window: Window, lease: Lease) -> WatchResult:
        original = self.robot.head_pose()
        pan = window.pan if window.pan is not None else 0.0
        image: ImageObservation | None = None
        try:
            with _heartbeat(lease, self.arbiter.lease_timeout_s):
                lease.check()
                self.robot.look(pan, min(TILT_MAX, window.tilt + TILT_UP_DEG))
                self._sleep(self.settle_s)
                lease.check()
                image = self._frame(lease)
                if image is None:
                    return WatchResult("no_camera", window, reason="the camera is unavailable")
                observation = self._read(image, window)
                lease.check()  # pre-empted while the picture was read: nothing is concluded
        except Preempted as gone:
            return WatchResult("aborted", window, reason=gone.reason)
        except InvalidObservation as error:
            self.rejections.append(str(error))
            print(f"[laika] outside observation rejected: {error}")
            return WatchResult("error", window, reason=f"invalid answer: {error}")
        except Exception:  # noqa: BLE001 - a head, camera or vision failure ends this look only
            log.exception("could not look out of a window")
            return WatchResult("error", window, reason="the look failed")
        finally:
            if lease.active:
                try:
                    self.robot.look(*original)
                except Exception:  # noqa: BLE001
                    log.exception("could not restore the head after looking out")
        if observation is None:
            if image is not None and image.simulated:
                return WatchResult("not_seen", window, reason="no window in the picture")
            return WatchResult("no_answer", window, reason="no answer from the vision model")
        if not observation.known:
            return WatchResult("not_seen", window, observation, reason="no outside in the picture")
        return WatchResult("done", window, observation, self._remember(observation))

    def _frame(self, lease: Lease) -> ImageObservation | None:
        """A fresh frame, waiting once for the camera's minimum interval; the outside never reaches the Observer."""
        image = self.camera.capture(CAPTURE_REASON, observe=False)
        if image is not None and self.camera.reused:
            self._sleep(max(self.camera.min_interval_s, self.settle_s) + 0.05)
            lease.check()
            image = self.camera.capture(CAPTURE_REASON, observe=False)
            if image is not None and self.camera.reused:
                return None
        return image

    def _read(self, image: ImageObservation, window: Window) -> OutsideObservation | None:
        path = str(self.camera.last_path) if self.camera.last_path else None
        room = self.world.current_room or window.room
        if image.simulated:
            return outside_from_scene(image, room, window, path)
        if self.describe is None:
            return None
        return self.describe(image, room, window, path)

    # --- remembering -------------------------------------------------------------------------

    def _remember(self, observation: OutsideObservation) -> list[str]:
        """Keep the look, and what changed since the last one. Returns the changes (English)."""
        with self._lock:
            previous = self._latest
            if previous is not None and observation.timestamp <= previous.timestamp:
                # Each look is strictly later than the last: on a coarse clock two looks can share a
                # tick, and "what changed since the last look" compares times.
                observation = replace(observation, timestamp=previous.timestamp + MIN_LOOK_STEP_S)
            self._latest = observation
        if self.store is not None:
            self.store.kv_put(NAMESPACE, LATEST_KEY, observation.to_dict())
        room = observation.room
        self.memory.episodes.record(
            "outside", f"Outside: {observation.summary}", EPISODE_SUBJECTS, room, observation.timestamp
        )
        changes = detect_changes(previous, observation)
        for kind, text in changes:
            self.memory.episodes.record(
                "outside_change", f"Outside: {text}", ["outside", kind], room, observation.timestamp
            )
        if changes:
            print(f"[laika] outside: {', '.join(text for _, text in changes)}")
            self._interest(observation, [text for _, text in changes])
        return [text for _, text in changes]

    def _interest(self, observation: OutsideObservation, changes: list[str]) -> None:
        """Feed curiosity: one InterestingChangeDetected for the changes, through the same gate as every sighting."""
        interest = self.config.change_interest
        if self.gate is None or interest <= 0:
            return
        seen = Observation(
            description=observation.description,
            confidence=observation.confidence,
            location=observation.room,
            timestamp=observation.timestamp,
            camera_pan=observation.pan,
            camera_tilt=observation.tilt,
            image_path=observation.image_path,
            source=observation.source,
        )
        score = InterestScore(total=interest, components={}, reasons=[f"outside: {text}" for text in changes])
        try:
            self.gate.consider(seen, score, source="outside")
        except Exception:  # noqa: BLE001 - announcing must never fail a look
            log.exception("could not announce a change outside")

    # --- the look_outside tool ----------------------------------------------------------------

    def look_now(self, lease: Lease | None = None) -> dict[str, Any]:
        """"Com'è fuori?": look out of a window of this room now, else answer from the latest look.

        The result says which it was ("source": "looked_now" or "latest_observation") and when it was seen.
        A look this recent, the daily cap or a failing look answer from the latest too, with the reason.
        Runs on the caller's thread under the caller's lease (the agent's turn holds the CONVERSATION lease).
        """
        here = self.windows_here()
        note = "there is no window in this room" if not here else ""
        if here:
            latest = self.latest()
            if latest is not None and 0 <= self._clock() - latest.timestamp < REUSE_S and latest.known:
                return self._answer(latest, "latest_observation", "looked out of the window a moment ago")
            active = lease or Lease(threading.Condition(), NAME, Priority.USER_COMMAND, resumable=False)
            result = self.watch(here[0], active)
            if result.ok and result.observation is not None:
                answer = self._answer(result.observation, "looked_now", "")
                if result.changes:
                    answer["changes"] = result.changes
                return answer
            note = {
                "refused": f"could not look now ({result.reason})",
                "aborted": "a more important behaviour took over",
                "not_seen": "the window showed nothing to judge by",
            }.get(result.status, f"the look failed ({result.reason or result.status})")
        latest = self.latest()
        if latest is None:
            return {"ok": False, "error": f"{note}, and nothing outside has been seen yet" if note else "nothing outside has been seen yet"}
        return self._answer(latest, "latest_observation", note)

    def _answer(self, observation: OutsideObservation, source: str, note: str) -> dict[str, Any]:
        result: dict[str, Any] = {"ok": True, "source": source, **self.snapshot(observation)}
        if note:
            result["note"] = note
        recent = [text for _, text in self.changes_since(observation.timestamp - CHANGE_MAX_AGE_S)]
        if recent:
            result["recent_changes"] = recent
        return result

    # --- the agent's context ---------------------------------------------------------------------

    def briefing(
        self, told_until: float = 0.0, told_id: str | None = None
    ) -> tuple[OutsideObservation, list[tuple[float, str]]] | None:
        """The latest look and its changes, when it is not the look already offered.

        `told_id` is that look's id and `told_until` its timestamp. The id decides: two looks can
        share a timestamp on a coarse clock (Windows ticks about every 15 ms).
        """
        latest = self.latest()
        now = self._clock()
        if latest is None or now - latest.timestamp > NOTE_MAX_AGE_S:
            return None
        if latest.id == told_id or latest.timestamp < told_until:
            return None
        if told_id is None and latest.timestamp <= told_until:
            return None
        return latest, self.changes_since(max(told_until, now - NOTE_MAX_AGE_S))

