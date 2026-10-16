"""The tools the AI acts through, and the validator every call passes first.

The model only proposes tool calls (`ai_client.ToolCall`). `ToolRegistry.run`
checks each one (known and offered tool, stale turn, schema, then the typed
`laika.commands` parsing) before any handler touches the robot. Wheel movement
goes through `SafetyController.execute_raw`, so the safety layer has the last
word. `run` never raises: the model gets `{"ok": true, ...}` or
`{"ok": false, "error": "..."}` back and can recover.
"""

from __future__ import annotations

import copy
import math
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

from laika.actions import DRIVING_ACTIONS, LOOK_POSES, ActionPlayer
from laika.ai_client import ToolCall, ToolSpec
from laika.commands import (
    MAX_MOVE_DURATION_MS,
    MAX_MOVE_SPEED,
    MAX_SPEAK_CHARS,
    InvalidCommand,
    LookCommand,
    ResultStatus,
    SpeakCommand,
)
from laika.camera import Camera
from laika.interaction_policy import MAX_MINUTES, MIN_MINUTES
from laika.limits import PAN_MAX, PAN_MIN, STEER_MAX, STEER_MIN, TILT_MAX, TILT_MIN
from laika.robot import (
    BATTERY,
    CAMERA,
    CAMERA_PAN,
    CAMERA_TILT,
    DRIVE,
    SPEAKER,
    STEER,
    ULTRASONIC,
    Robot,
)
from laika.conversation_memory import record_answer, store_user_fact
from laika.memory import FACT_KINDS, Fact, Memory, iso_minutes, relative_time
from laika.safety import SafetyController
from laika.observations import Observer, fold
from laika.rooms import name_current_room
from laika.scan import DEFAULT_SETTLE_S, ScanBusy, Scanner
from laika.windows import WindowRegistry, is_window_name, parse_window_statement, teach_window
from laika.world_model import KnownObject, WorldModel

if TYPE_CHECKING:
    from laika.battery import BatteryMonitor
    from laika.behaviour import Lease
    from laika.explore import Explorer
    from laika.interaction_policy import InteractionPolicy
    from laika.investigate import Investigator
    from laika.outside import WindowWatcher
    from laika.rooms import RoomTracker
    from laika.search import Searcher
    from laika.web import WebLookup

MAX_GESTURES_PER_TURN = 3
MAX_TEXT_CHARS = 200
MAX_RECALL_RESULTS = 8
MAX_RECALL_EPISODES = 5
MAX_CHANGES = 20
DEFAULT_CHANGE_HOURS = 24
MAX_CHANGE_HOURS = 168
REJECTION_HISTORY = 50
OBSERVATION_WAIT_S = 2.0
# What the opening line of a proactive question may use (#40): the head and the gestures. No wheels,
# no exploring, and none of the tools that change the interaction policy: a turn Laika starts herself
# must never loosen it.
PROACTIVE_TOOLS = frozenset({"look", "look_left", "look_right", "look_up", "look_down", "gesture"})
# Tools that bring in text from the internet. Once one of them returned content, the turn is tainted.
WEB_TOOLS = frozenset({"wikipedia", "web_search"})
# Refused for the rest of a tainted turn: anything that moves the body (stop is always allowed, as is
# stop_exploring) and anything that writes memory or changes a setting. Web text is data, never orders.
TAINT_BLOCKED = frozenset(
    {
        "move", "approach", "start_exploring", "investigate", "find_object", "find_person",
        "remember", "answer_question", "forget", "name_room", "look_outside",
        "set_do_not_disturb", "clear_do_not_disturb", "set_busy", "clear_busy",
        "resume_autonomy",
    }
)
MAX_QUERY_CHARS = 200
# A move ended for one of these reasons: nothing else the model proposed this turn should run.
ABORTING_REASONS = frozenset(
    {"emergency_stop", "watchdog", "ai_unavailable", "closed", "preempted", "lease_expired"}
)


# --- turns ----------------------------------------------------------------


class Turn:
    """One agent turn. It goes stale when the agent cancels it or a move is aborted."""

    def __init__(
        self,
        generation: int = 0,
        current: Callable[[], int] = lambda: 0,
        lease: Lease | None = None,
    ) -> None:
        self.generation = generation
        self._current = current
        # The behaviour lease the turn acts under (the conversation's), if any.
        self.lease = lease
        self._aborted: str | None = None
        self.gestures = 0
        # True once web content entered the turn (see TAINT_BLOCKED).
        self.tainted = False

    def abort(self, reason: str) -> None:
        self._aborted = self._aborted or reason

    @property
    def cancelled(self) -> bool:
        return self.generation != self._current()

    @property
    def stale_reason(self) -> str | None:
        if self._aborted:
            return f"movement was cancelled ({self._aborted})"
        if self.generation != self._current():
            return "the turn was cancelled"
        if self.lease is not None and not self.lease.active:
            return "a more important behaviour took over"
        return None


# --- schema validation (the small JSON-schema subset the tools use) --------


def validate_arguments(schema: Mapping[str, Any], arguments: Mapping[str, Any]) -> str | None:
    """Return why `arguments` do not fit `schema`, or None. Unknown keys are rejected."""
    properties = schema.get("properties", {})
    missing = [key for key in schema.get("required", []) if key not in arguments]
    if missing:
        return f"missing parameters: {', '.join(missing)}"
    unknown = sorted(set(arguments) - set(properties))
    if unknown:
        return f"unknown parameters: {', '.join(unknown)}"
    for key, value in arguments.items():
        problem = _check_value(key, properties[key], value)
        if problem:
            return problem
    return None


def _check_value(name: str, spec: Mapping[str, Any], value: Any) -> str | None:
    kind = spec.get("type")
    if kind == "string":
        if not isinstance(value, str):
            return f"{name} must be a string"
        if len(value.strip()) < spec.get("minLength", 0):
            return f"{name} is too short"
        if len(value) > spec.get("maxLength", math.inf):
            return f"{name} must be at most {spec['maxLength']} characters"
    elif kind in ("integer", "number"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{name} must be a number"
        if not math.isfinite(value):
            return f"{name} must be finite"
        if kind == "integer" and not float(value).is_integer():
            return f"{name} must be a whole number"
        low, high = spec.get("minimum", -math.inf), spec.get("maximum", math.inf)
        if not low <= value <= high:
            return f"{name} must be between {low} and {high}, got {value}"
    elif kind == "boolean":
        if not isinstance(value, bool):
            return f"{name} must be true or false"
    if "enum" in spec and value not in spec["enum"]:
        return f"{name} must be one of: {', '.join(map(str, spec['enum']))}"
    return None


def schema(required: tuple[str, ...] = (), **properties: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        result["required"] = list(required)
    return result


# --- the tools ------------------------------------------------------------


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]
    # Capabilities the robot must advertise for the model to be offered this tool.
    requires: frozenset[str] = frozenset()

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)


def _since(now: float, since: str | None, hours: int | None) -> float:
    """The start of the period `what_changed` covers: local midnight today or yesterday, or `hours` back."""
    if since in ("today", "yesterday"):
        midnight = datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0)
        if since == "yesterday":
            midnight -= timedelta(days=1)
        return midnight.timestamp()
    return now - (hours if hours is not None else DEFAULT_CHANGE_HOURS) * 3600.0


def _object_dict(obj: KnownObject) -> dict[str, Any]:
    return {"name": obj.name, "room": obj.room, "note": obj.note, "source": obj.source}


class ToolRegistry:
    """The tools offered to the model for one robot, and the gate every call goes through."""

    def __init__(
        self,
        robot: Robot,
        safety: SafetyController,
        *,
        world: WorldModel | None = None,
        actions: ActionPlayer | None = None,
        speak: Callable[[str], None] | None = None,
        memory: Memory | None = None,
        clock: Callable[[], float] = time.time,
        camera: Camera | None = None,
        observer: Observer | None = None,
        observation_wait_s: float = OBSERVATION_WAIT_S,
        scan_settle_s: float = DEFAULT_SETTLE_S,
        allow_driving: bool = True,
        explorer: Explorer | None = None,
        investigator: Investigator | None = None,
        interaction: InteractionPolicy | None = None,
        searcher: Searcher | None = None,
        web: WebLookup | None = None,
        battery: BatteryMonitor | None = None,
        rooms: RoomTracker | None = None,
        outside: WindowWatcher | None = None,
        windows: WindowRegistry | None = None,
    ) -> None:
        self.robot = robot
        self.rooms = rooms
        self.outside = outside
        self.windows = windows if windows is not None else (outside.windows if outside is not None else None)
        self.battery = battery
        self.safety = safety
        self.world = world if world is not None else WorldModel()
        self.actions = actions
        self.memory = memory if memory is not None else Memory()
        self._clock = clock
        self.camera = camera if camera is not None else Camera(robot)
        self.observer = observer
        # How long capture_image waits for the frame's observation: a vision request is slower
        # than the simulator's ground truth, and the model wants the description in the reply.
        self.observation_wait_s = observation_wait_s
        self.scanner = Scanner(
            robot, self.camera, observer, settle_s=scan_settle_s, observation_wait_s=observation_wait_s
        )
        self._speak = speak or robot.speak
        self.allow_driving = allow_driving
        self.explorer = explorer
        self.investigator = investigator
        self.interaction = interaction
        self.searcher = searcher
        self.web = web
        # Recent rejections as (tool, reason), for logs and tests.
        self.rejections: deque[tuple[str, str]] = deque(maxlen=REJECTION_HISTORY)
        # The lease of the turn being run (the agent runs one turn at a time per thread).
        self._running = threading.local()
        capabilities = robot.identity().capabilities
        self._tools = {
            tool.name: tool
            for tool in self._build()
            if tool.requires <= capabilities and (allow_driving or DRIVE not in tool.requires)
        }

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in self._tools.values()]

    def restricted(self, allowed: Iterable[str]) -> ToolRegistry:
        """A registry that offers (and runs) only the tools in `allowed`; everything else is unknown."""
        keep = frozenset(allowed)
        narrowed = copy.copy(self)
        narrowed._tools = {name: tool for name, tool in self._tools.items() if name in keep}
        return narrowed

    def run(self, call: ToolCall, turn: Turn | None = None) -> dict[str, Any]:
        """Validate and execute one proposed call. Always returns a result dict."""
        turn = turn or Turn()
        stale = turn.stale_reason
        if stale:
            return self.reject(call.name, f"stale: {stale}; the proposal was not executed")
        tool = self._tools.get(call.name)
        if tool is None:
            offered = ", ".join(self._tools)
            return self.reject(call.name, f"unknown tool {call.name!r}; available: {offered}")
        if turn.tainted and call.name in TAINT_BLOCKED:
            return self.reject(
                call.name, "not allowed after reading web content in the same turn; ask the user again"
            )
        if not call.valid:
            return self.reject(call.name, f"invalid arguments: {call.error}")
        problem = validate_arguments(tool.parameters, call.arguments)
        if problem:
            return self.reject(call.name, problem)
        try:
            result = self._dispatch(tool, call.arguments, turn)
        except InvalidCommand as error:
            return self.reject(call.name, f"invalid: {error}")
        except Exception as error:  # noqa: BLE001 - a failing tool must not break the turn
            print(f"[laika] tool {call.name} failed: {error}")
            return {"ok": False, "error": f"{type(error).__name__}: {error}"}
        if not result.get("ok", False):
            print(f"[laika] tool {call.name} not done: {result.get('error')}")
        return result

    def _dispatch(self, tool: Tool, arguments: dict[str, Any], turn: Turn) -> dict[str, Any]:
        if tool.name == "gesture":
            if turn.gestures >= MAX_GESTURES_PER_TURN:
                return {"ok": False, "error": f"at most {MAX_GESTURES_PER_TURN} gestures per answer"}
            turn.gestures += 1
        self._running.lease = turn.lease
        self._running.turn = turn
        try:
            result = tool.handler(arguments)
        finally:
            self._running.lease = None
            self._running.turn = None
        if tool.name in WEB_TOOLS and result.get("ok"):
            turn.tainted = True
        reason = result.get("reason")
        if reason in ABORTING_REASONS:
            turn.abort(str(reason))
        return result

    def reject(self, name: str, reason: str) -> dict[str, Any]:
        self.rejections.append((name, reason))
        print(f"[laika] tool rejected: {name or '?'}: {reason}")
        return {"ok": False, "error": reason}

    # --- building ---------------------------------------------------------

    def _build(self) -> list[Tool]:
        look_poses = [
            Tool(
                f"look_{side}",
                f"Gira la testa a {label} (posa fissa).",
                schema(),
                lambda _a, pose=pose: self._look(*pose),
                requires=frozenset({CAMERA_PAN if side in ("left", "right") else CAMERA_TILT}),
            )
            for side, label, pose in (
                ("left", "sinistra", LOOK_POSES["left"]),
                ("right", "destra", LOOK_POSES["right"]),
                ("up", "in alto", LOOK_POSES["up"]),
                ("down", "in basso", LOOK_POSES["down"]),
            )
        ]
        tools = [
            Tool(
                "get_robot_state",
                "Stato del robot: identità, capacità, posizione della testa, sterzo, "
                "velocità, distanza, stanza e batteria.",
                schema(),
                self._get_robot_state,
            ),
            Tool(
                "move",
                "Muovi le ruote per un tempo breve, solo se te lo chiedono. "
                "Il controllo di sicurezza può fermare o rifiutare il movimento.",
                schema(
                    ("direction", "duration_ms"),
                    direction={"type": "string", "enum": ["forward", "backward"]},
                    duration_ms={"type": "integer", "minimum": 1, "maximum": MAX_MOVE_DURATION_MS},
                    steering={"type": "number", "minimum": STEER_MIN, "maximum": STEER_MAX},
                    speed={"type": "integer", "minimum": 1, "maximum": MAX_MOVE_SPEED},
                ),
                self._move,
                requires=frozenset({DRIVE, STEER}),
            ),
            Tool("stop", "Ferma subito le ruote.", schema(), self._stop),
            Tool(
                "look",
                "Punta la testa: pan negativo a sinistra, positivo a destra; "
                "tilt negativo in basso, positivo in alto.",
                schema(
                    ("pan", "tilt"),
                    pan={"type": "number", "minimum": PAN_MIN, "maximum": PAN_MAX},
                    tilt={"type": "number", "minimum": TILT_MIN, "maximum": TILT_MAX},
                ),
                lambda a: self._look(a["pan"], a["tilt"]),
                requires=frozenset({CAMERA_PAN, CAMERA_TILT}),
            ),
            *look_poses,
            Tool(
                "capture_image",
                "Scatta una foto con la telecamera e dì cosa si vede.",
                schema(),
                self._capture_image,
                requires=frozenset({CAMERA}),
            ),
            Tool(
                "scan_environment",
                "Guardati intorno: muove solo la testa (sinistra, centro, destra), scatta una foto "
                "in ogni posizione e riassume cosa c'è e dove. Richiede qualche secondo.",
                schema(),
                self._scan_environment,
                requires=frozenset({CAMERA, CAMERA_PAN}),
            ),
            Tool(
                "get_distance",
                "Distanza in centimetri dall'ostacolo davanti.",
                schema(),
                self._get_distance,
                requires=frozenset({ULTRASONIC}),
            ),
            Tool(
                "get_sensors",
                "Leggi tutti i sensori: distanza, bordo (scale), scala di grigi e batteria.",
                schema(),
                self._get_sensors,
            ),
            Tool(
                "speak",
                "Di' subito una frase breve ad alta voce, per esempio prima di una "
                "azione lunga. La risposta finale in testo viene letta comunque.",
                schema(("text",), text={"type": "string", "minLength": 1, "maxLength": MAX_SPEAK_CHARS}),
                self._say,
                requires=frozenset({SPEAKER}),
            ),
            Tool(
                "remember",
                "Ricorda un fatto che ti dicono, con le parole dell'utente (text). subject è l'oggetto, "
                "la persona o la stanza di cui parla (per esempio 'zaino nero'); kind è location (dove "
                "sta di solito), ownership (di chi è), preference, rule (una regola da rispettare) o "
                "general. Con kind location puoi dare anche room (la stanza, in inglese, per esempio "
                "living_room). Un nuovo luogo o proprietario dello stesso subject sostituisce il vecchio.",
                schema(
                    ("text",),
                    text={"type": "string", "minLength": 1, "maxLength": MAX_TEXT_CHARS},
                    subject={"type": "string", "minLength": 1, "maxLength": 60},
                    kind={"type": "string", "enum": list(FACT_KINDS)},
                    room={"type": "string", "minLength": 1, "maxLength": 60},
                ),
                self._remember,
            ),
            Tool(
                "recall",
                "Cerca nei ricordi un oggetto, una persona o una stanza: fatti che ti hanno detto, "
                "dove l'hai visto l'ultima volta e cosa è successo (con ora e quanto tempo fa).",
                schema(("query",), query={"type": "string", "minLength": 1, "maxLength": 100}),
                self._recall,
            ),
            Tool(
                "what_changed",
                "Racconta cosa è cambiato in casa da un certo momento: oggetti spostati, nuovi o spariti "
                "che hai notato, e cosa hai fatto nei momenti liberi (activities: hai esplorato, "
                "guardato fuori dalla finestra, riposato...). since è 'today' (da stamattina) o 'yesterday' (da ieri mattina); "
                "oppure since_hours (da 1 a 168 ore fa); senza niente guardi le ultime 24 ore. "
                "Le ore sono assolute, usa 'ago' per dire 'due ore fa' o 'ieri'.",
                schema(
                    since={"type": "string", "enum": ["today", "yesterday"]},
                    since_hours={"type": "integer", "minimum": 1, "maximum": MAX_CHANGE_HOURS},
                ),
                self._what_changed,
            ),
            Tool(
                "forget",
                "Dimentica i fatti su una cosa o una persona, se te lo chiedono: target è il subject "
                "oppure l'id di un fatto (per esempio f-1a2b3c4d).",
                schema(("target",), target={"type": "string", "minLength": 1, "maxLength": 60}),
                self._forget,
            ),
            # A memory write like remember: refused after web content (TAINT_BLOCKED) and never offered to
            # a proactive turn (PROACTIVE_TOOLS is an allow-list).
            Tool(
                "name_room",
                "Impari il nome della stanza in cui sei ora, quando te lo dicono (\"questa è la camera di "
                "Leonardo\", \"qui è lo studio\", \"questa stanza si chiama taverna\") o ti correggono (\"no, "
                "questa è la cucina\"). name è il nome come l'ha detto l'utente, senza articolo. Poi "
                "\"non entrare nella camera di Leonardo\" vale per quella stanza.",
                schema(("name",), name={"type": "string", "minLength": 1, "maxLength": 60}),
                self._name_room,
            ),
        ]
        if self.explorer is not None and self.explorer.config.enabled:
            tools += [
                Tool(
                    "start_exploring",
                    "Vai a esplorare la casa da sola, piano e con prudenza, guardandoti intorno. "
                    "Parte quando finisci di parlare; duration_s è la durata in secondi.",
                    schema(duration_s={"type": "integer", "minimum": 10, "maximum": 300}),
                    self._start_exploring,
                    requires=frozenset({CAMERA, CAMERA_PAN}),
                ),
                Tool(
                    "stop_exploring",
                    "Smetti di esplorare e ferma le ruote.",
                    schema(),
                    self._stop_exploring,
                ),
            ]
        if self.investigator is not None and self.investigator.config.enabled:
            tools += [
                Tool(
                    "investigate",
                    "Guarda meglio una cosa: punta la testa dove l'hai vista, scatta un'altra foto e, "
                    "se serve, si avvicina un poco. Senza object indaga sull'ultima cosa interessante, "
                    "con object su quell'oggetto. Se è una cosa che non conosci mette in coda una "
                    "domanda da fare più tardi.",
                    schema(object={"type": "string", "minLength": 1, "maxLength": 60}),
                    self._investigate,
                    requires=frozenset({CAMERA}),
                ),
                Tool(
                    "answer_question",
                    "Salva la risposta dell'utente a una domanda in sospeso (question_id, per esempio "
                    "q-1, e answer, la risposta breve).",
                    schema(
                        ("question_id", "answer"),
                        question_id={"type": "string", "minLength": 1, "maxLength": 20},
                        answer={"type": "string", "minLength": 1, "maxLength": MAX_TEXT_CHARS},
                    ),
                    self._answer_question,
                ),
            ]
        if self.searcher is not None:
            seconds = {"type": "integer", "minimum": 5, "maximum": int(self.searcher.config.max_max_s)}
            tools += [
                Tool(
                    "find_object",
                    "Cerca una cosa in casa: prima nei ricordi, poi guardandoti intorno con la testa, "
                    "poi (se può guidare) andando a esplorare. name è l'oggetto come nei ricordi (in "
                    "inglese, per esempio backpack); max_seconds è quanto cercare. Dice se l'ha vista, "
                    "dove (stanza e direzione, la testa è puntata lì) e come l'ha trovata; se no, dov'era "
                    "l'ultima volta. Non sa andare in una stanza precisa.",
                    schema(("name",), name={"type": "string", "minLength": 1, "maxLength": 60}, max_seconds=seconds),
                    self._find_object,
                    requires=frozenset({CAMERA, CAMERA_PAN}),
                ),
                Tool(
                    "find_person",
                    "Cerca una persona qualsiasi: chi è vicino, poi guardandoti intorno, poi (se può "
                    "guidare) esplorando. Non distingue le persone e non sa i nomi: le conta soltanto. "
                    "max_seconds è quanto cercare.",
                    schema(max_seconds=seconds),
                    self._find_person,
                    requires=frozenset({CAMERA, CAMERA_PAN}),
                ),
                Tool(
                    "approach",
                    "Avvicinati a quello che hai visto da poco, con pochi movimenti brevi e sicuri: target "
                    "è il nome di un oggetto (come nei ricordi) oppure person. Se le ruote non possono "
                    "muoversi da sole risponde autonomous_disabled.",
                    schema(("target",), target={"type": "string", "minLength": 1, "maxLength": 60}),
                    self._approach,
                    requires=frozenset({CAMERA, DRIVE, STEER, ULTRASONIC}),
                ),
            ]
        if self.interaction is not None:
            # These only make her quieter or clear the user's own setting; no tool starts an
            # interaction or touches the cooldown (the policy decides that, interaction_policy.py).
            minutes = {"type": "integer", "minimum": MIN_MINUTES, "maximum": MAX_MINUTES}
            tools += [
                Tool(
                    "set_do_not_disturb",
                    "L'utente non vuole essere disturbato (\"non disturbarmi per un'ora\"): da quel "
                    "momento non inizi più conversazioni da sola. Risponde sempre se ti parlano.",
                    schema(("minutes",), minutes=minutes),
                    self._set_do_not_disturb,
                ),
                Tool(
                    "clear_do_not_disturb",
                    "L'utente dice che puoi di nuovo parlare da sola: toglie il non disturbare.",
                    schema(),
                    self._clear_do_not_disturb,
                ),
                Tool(
                    "set_busy",
                    "L'utente è occupato (\"sono occupato per mezz'ora\"): non inizi conversazioni "
                    "da sola, ma se ti parla rispondi.",
                    schema(("minutes",), minutes=minutes),
                    self._set_busy,
                ),
                Tool(
                    "clear_busy",
                    "L'utente dice che non è più occupato.",
                    schema(),
                    self._clear_busy,
                ),
            ]
        tools += [
            # Only the user lifts a pause: this tool is not offered to a proactive turn (PROACTIVE_TOOLS)
            # and is refused after web content entered the turn (TAINT_BLOCKED).
            Tool(
                "pause_autonomy",
                "L'utente ti dice di stare ferma (\"Laika, resta ferma\"): da quel momento non ti muovi, "
                "non esplori e non inizi niente da sola, anche dopo un riavvio, finché non ti dice che "
                "puoi muoverti di nuovo. Se te lo chiede lui, i suoi comandi di movimento funzionano ancora.",
                schema(),
                self._pause_autonomy,
            ),
            Tool(
                "resume_autonomy",
                "Solo se l'utente lo chiede espressamente (\"puoi muoverti di nuovo\"): toglie la pausa e "
                "puoi tornare a muoverti da sola. Mai di tua iniziativa.",
                schema(),
                self._resume_autonomy,
            ),
        ]
        if self.outside is not None:
            tools.append(
                Tool(
                    "look_outside",
                    "Com'è fuori? Se in questa stanza c'è una finestra, la guardi (punta la testa, scatta una "
                    "foto e la fa descrivere: tempo, luce, cosa succede, quante persone); altrimenti rispondi "
                    "con l'ultima occhiata che avevi dato, con l'ora. Il risultato dice quale dei due è "
                    "(source: looked_now o latest_observation) e quando (ago). Le persone fuori le conti soltanto.",
                    schema(),
                    self._look_outside,
                    requires=frozenset({CAMERA}),
                )
            )
        if self.web is not None:
            query = {"type": "string", "minLength": 2, "maxLength": MAX_QUERY_CHARS}
            tools.append(
                Tool(
                    "wikipedia",
                    "Cerca su Wikipedia (italiano, poi inglese) fatti su persone, luoghi, animali e cose: "
                    "\"chi è\", \"cos'è\", \"come funziona\". query sono poche parole chiave. Restituisce "
                    "brevi riassunti con il link. Non cercare mai nomi dei familiari o dettagli sulla casa.",
                    schema(("query",), query=query),
                    self._wikipedia,
                )
            )
            if self.web.has_brave:
                tools.append(
                    Tool(
                        "web_search",
                        "Cerca in rete cose di attualità: notizie, meteo, sport, orari di apertura. query sono "
                        "poche parole chiave. Risultati filtrati per bambini. Non cercare mai nomi dei "
                        "familiari o dettagli sulla casa.",
                        schema(("query",), query=query),
                        self._web_search,
                    )
                )
        gestures = self._gesture_names()
        if gestures:
            tools.append(
                Tool(
                    "gesture",
                    "Fai un gesto espressivo senza bloccarti (da zero a "
                    f"{MAX_GESTURES_PER_TURN} per risposta).",
                    schema(("name",), name={"type": "string", "enum": gestures}),
                    self._gesture,
                )
            )
        return tools

    def _gesture_names(self) -> list[str]:
        if self.actions is None:
            return []
        return [name for name in self.actions.allowed if name not in DRIVING_ACTIONS]

    # --- handlers ---------------------------------------------------------

    def _get_robot_state(self, _args: dict[str, Any]) -> dict[str, Any]:
        state = self.robot.get_state()
        telemetry = state.to_telemetry().to_dict()
        battery = self._battery_info(state.battery_v)
        if battery is not None and battery["voltage"] is not None:
            telemetry["batteryV"] = battery["voltage"]  # calibrated and smoothed, not the raw reading
        result = {
            "ok": True,
            "robot": self.robot.identity().to_dict(),
            "state": telemetry,
            "room": self.world.current_room,
            "driving_enabled": "move" in self._tools and self.safety.enabled,
            "emergency_stopped": self.safety.emergency_stopped,
            "autonomy": self.safety.autonomy_status(),
        }
        if battery is not None:
            result["battery"] = battery
        return result

    def _battery_info(self, raw_v: float | None = None) -> dict[str, Any] | None:
        """Voltage, state and a rough percentage; None when nothing is known about the battery."""
        if self.battery is not None:
            return self.battery.snapshot()
        if raw_v is None and BATTERY in self.robot.identity().capabilities:
            raw_v = self.robot.battery_voltage()
        return None if raw_v is None else {"voltage": round(raw_v, 2), "state": "unknown", "percent": None}

    def _move(self, args: dict[str, Any]) -> dict[str, Any]:
        command = {
            "type": "command",
            "command": "move",
            "parameters": {
                "direction": args["direction"],
                "speed": args.get("speed", MAX_MOVE_SPEED),  # the controller caps it further
                "durationMs": args["duration_ms"],
                "steering": args.get("steering", 0),
            },
        }
        outcome = self.safety.execute_raw(command, lease=self._lease())
        if outcome is None:
            return {"ok": False, "error": "the move did not report a result"}
        if outcome.reason.startswith("invalid:"):
            raise InvalidCommand(outcome.reason.removeprefix("invalid:").strip())
        result: dict[str, Any] = {"ok": outcome.result is ResultStatus.SUCCESS}
        result["result"] = outcome.result.value
        if outcome.reason:
            result["reason"] = outcome.reason
        if not result["ok"]:
            result["error"] = f"move {outcome.result.value}: {outcome.reason or 'not completed'}"
        result["distance_cm"] = self.robot.distance_cm()
        return result

    def _lease(self) -> Lease | None:
        return getattr(self._running, "lease", None)

    def _preempted(self) -> dict[str, Any] | None:
        lease = self._lease()
        if lease is not None and not lease.active:
            return {"ok": False, "reason": "preempted", "error": "a more important behaviour took over"}
        return None

    def _stop(self, _args: dict[str, Any]) -> dict[str, Any]:
        self.safety.stop()
        return {"ok": True}

    def _look(self, pan: float, tilt: float) -> dict[str, Any]:
        command = LookCommand(pan=pan, tilt=tilt)
        if (gone := self._preempted()) is not None:
            return gone
        self.robot.look(command.pan, command.tilt)
        return {"ok": True, "pan": command.pan, "tilt": command.tilt}

    def _capture_image(self, _args: dict[str, Any]) -> dict[str, Any]:
        # Never returns image bytes: the model gets what was recognised, not the frame.
        image = self.camera.capture("request")
        if image is None:
            return {"ok": False, "error": "the camera is unavailable"}
        result: dict[str, Any] = {
            "ok": True,
            "scene": list(image.scene),
            "room": self.world.current_room,
            "pan": image.camera_pan,
            "tilt": image.camera_tilt,
            "saved": self.camera.last_path is not None,
        }
        observation = (
            self.observer.wait_for(image, self.observation_wait_s) if self.observer is not None else None
        )
        if observation is not None:
            result["description"] = observation.description
            result["objects"] = observation.object_names
            result["people"] = observation.people
            result["confidence"] = observation.confidence
        if self.camera.reused:
            result["note"] = "photo taken a moment ago, reused: the camera was asked too often"
        elif not image.scene:
            result["note"] = "nothing recognised: full image understanding is not available yet"
        return result

    def _scan_environment(self, _args: dict[str, Any]) -> dict[str, Any]:
        # Never returns image bytes: only what was recognised at each position.
        try:
            scan = self.scanner.scan()
        except ScanBusy as error:
            return {"ok": False, "error": str(error)}
        result = scan.to_dict()
        if not scan.ok:
            reasons = sorted({p.reason or "no observation" for p in scan.positions})
            result["error"] = f"nothing could be seen: {', '.join(reasons)}"
        return result

    def _start_exploring(self, args: dict[str, Any]) -> dict[str, Any]:
        assert self.explorer is not None  # the tool is only built with an Explorer
        # Somebody asked: an unknown battery or a used-up driving budget does not keep the wheels still.
        if not self.explorer.start(args.get("duration_s"), requested=True):
            return {"ok": False, "error": f"cannot explore: {self.explorer.refusal}"}
        result: dict[str, Any] = {
            "ok": True,
            "started": True,
            "mode": self.explorer.mode,
            "note": "she starts as soon as the conversation is over",
        }
        if self.explorer.mode == "head_only":
            result["note"] += f"; the wheels stay still ({self.explorer.head_only_reason}), she only looks around"
        return result

    def _find_object(self, args: dict[str, Any]) -> dict[str, Any]:
        assert self.searcher is not None  # the tool is only built with a Searcher
        return self.searcher.find_object(
            args["name"], lease=self._lease(), max_s=args.get("max_seconds"), should_stop=self._turn_stale
        )

    def _find_person(self, args: dict[str, Any]) -> dict[str, Any]:
        assert self.searcher is not None
        return self.searcher.find_person(
            lease=self._lease(), max_s=args.get("max_seconds"), should_stop=self._turn_stale
        )

    def _approach(self, args: dict[str, Any]) -> dict[str, Any]:
        assert self.searcher is not None
        return self.searcher.approach(args["target"], lease=self._lease(), should_stop=self._turn_stale)

    def _wikipedia(self, args: dict[str, Any]) -> dict[str, Any]:
        assert self.web is not None  # the tool is only built with a WebLookup
        return self.web.wikipedia.search(args["query"])

    def _web_search(self, args: dict[str, Any]) -> dict[str, Any]:
        assert self.web is not None and self.web.brave is not None
        return self.web.brave.search(args["query"])

    def _turn_stale(self) -> bool:
        """Whether the turn running this tool was cancelled (polled by a long search)."""
        turn = getattr(self._running, "turn", None)
        return turn is not None and turn.stale_reason is not None

    def _set_do_not_disturb(self, args: dict[str, Any]) -> dict[str, Any]:
        assert self.interaction is not None
        return {"ok": True, "until": iso_minutes(self.interaction.set_do_not_disturb(args["minutes"]))}

    def _clear_do_not_disturb(self, _args: dict[str, Any]) -> dict[str, Any]:
        assert self.interaction is not None
        return {"ok": True, "was_set": self.interaction.clear_do_not_disturb()}

    def _set_busy(self, args: dict[str, Any]) -> dict[str, Any]:
        assert self.interaction is not None
        return {"ok": True, "until": iso_minutes(self.interaction.set_busy(args["minutes"]))}

    def _clear_busy(self, _args: dict[str, Any]) -> dict[str, Any]:
        assert self.interaction is not None
        return {"ok": True, "was_set": self.interaction.clear_busy()}

    def _pause_autonomy(self, _args: dict[str, Any]) -> dict[str, Any]:
        was_paused = self.safety.paused
        self.safety.pause_autonomy()
        return {"ok": True, "paused": True, "was_paused": was_paused}

    def _resume_autonomy(self, _args: dict[str, Any]) -> dict[str, Any]:
        was_paused = self.safety.resume_autonomy()
        return {"ok": True, "paused": False, "was_paused": was_paused}

    def _stop_exploring(self, _args: dict[str, Any]) -> dict[str, Any]:
        assert self.explorer is not None
        was_exploring = self.explorer.active
        self.explorer.stop()
        return {"ok": True, "was_exploring": was_exploring}

    def _investigate(self, args: dict[str, Any]) -> dict[str, Any]:
        assert self.investigator is not None  # the tool is only built with an Investigator
        # Under the turn's CONVERSATION lease: the user asked, so its moves are not autonomous.
        return self.investigator.investigate_now(args.get("object"), lease=self._lease())

    def _answer_question(self, args: dict[str, Any]) -> dict[str, Any]:
        assert self.investigator is not None
        questions = self.investigator.questions
        question = questions.get(args["question_id"].strip())
        if question is None:
            return {"ok": False, "error": "no such question (it may have expired or been answered)"}
        record_answer(self.world, self.memory, question, args["answer"], self._clock())
        questions.remove(question.id)
        return {"ok": True, "stored": f"object: {question.object}", "remaining": len(questions)}

    def _get_distance(self, _args: dict[str, Any]) -> dict[str, Any]:
        distance = self.robot.distance_cm()
        result: dict[str, Any] = {"ok": True, "distance_cm": distance}
        if distance is None:
            result["note"] = "the reading is invalid"
        return result

    def _get_sensors(self, _args: dict[str, Any]) -> dict[str, Any]:
        sensors = self.robot.get_sensors()
        result: dict[str, Any] = {
            "ok": True,
            "distance_cm": sensors.distance_cm,
            "cliff_detected": sensors.cliff_detected,
            "grayscale": list(sensors.grayscale),
        }
        battery = self._battery_info()
        if battery is not None:
            result["battery"] = battery
        return result

    def _say(self, args: dict[str, Any]) -> dict[str, Any]:
        command = SpeakCommand(text=args["text"])
        self._speak(command.text)
        return {"ok": True}

    def _remember(self, args: dict[str, Any]) -> dict[str, Any]:
        text = " ".join(args["text"].split())
        subject = (args.get("subject") or "").strip() or None
        room = (args.get("room") or "").strip() or None
        kind = args.get("kind") or ("location" if room else "general")
        if subject and is_window_name(subject) and self.windows is not None and kind in ("location", "general"):
            return self._remember_window(
                text, fold(room).replace(" ", "_") if room else None, self._clock()
            )
        if room and (subject is None or kind != "location"):
            return {"ok": False, "error": "room goes with a subject and kind location"}
        fact = store_user_fact(
            self.world, self.memory, text=text, subject=subject, kind=kind, now=self._clock(),
            room=fold(room).replace(" ", "_") if room else None,
            questions=self.investigator.questions if self.investigator is not None else None,
        )
        return {"ok": True, "stored": {"id": fact.id, "kind": fact.kind, "subject": fact.subject}}

    def _look_outside(self, _args: dict[str, Any]) -> dict[str, Any]:
        assert self.outside is not None  # the tool is only built with a WindowWatcher
        if (gone := self._preempted()) is not None:
            return gone
        return self.outside.look_now(self._lease())

    def _remember_window(
        self, text: str, room: str | None, now: float
    ) -> dict[str, Any]:
        """A window somebody told her about: the registry, the world model and a fact (windows.teach_window)."""
        assert self.windows is not None
        where = room or self.world.current_room
        if where is None:
            return {"ok": False, "error": "which room is the window in? give room"}
        hint = parse_window_statement(fold(text).replace("'", " ")) or ("somewhere", None)
        window, fact = teach_window(
            self.windows, self.world, self.memory, room=where, text=text, pan=hint[1], now=now
        )
        return {
            "ok": True,
            "stored": {"id": fact.id, "kind": fact.kind, "subject": fact.subject},
            "window": window.label if window is not None else "no room for another window",
        }

    def _name_room(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            naming = name_current_room(
                self.world, self.memory, self.rooms, args["name"], self._clock(),
                said=f"La stanza si chiama {args['name'].strip()}",
            )
        except ValueError as error:
            return {"ok": False, "error": str(error)}
        return {
            "ok": True, "action": naming.action, "room": naming.room, "name": naming.name,
            "previous_room": naming.previous, "names": naming.labels,
        }

    def _forget(self, args: dict[str, Any]) -> dict[str, Any]:
        removed = self.memory.facts.forget(args["target"])
        result: dict[str, Any] = {"ok": True, "forgotten": len(removed)}
        if not removed:
            result["note"] = "no fact with that id or subject"
        return result

    def _recall(self, args: dict[str, Any]) -> dict[str, Any]:
        query = args["query"]
        needle = fold(query)
        now = self._clock()
        objects: dict[str, KnownObject] = {}
        exact = self.world.where_is(query)
        if exact is not None:
            objects[exact.name] = exact
        named = self.world.room_for_alias(query)  # "la camera di Leonardo" is a room people named
        for room in self.world.rooms():
            if fold(room) == needle or room == named:
                for obj in self.world.objects_in(room):
                    objects.setdefault(obj.name, obj)
        for raw in self.world.to_dict()["objects"]:
            known = KnownObject.from_dict(raw)
            if needle in fold(known.name) or (known.note and needle in fold(known.note)):
                objects.setdefault(known.name, known)
        found = [self._object_result(o, now) for o in list(objects.values())[:MAX_RECALL_RESULTS]]
        facts = [self._fact_result(f, now) for f in self.memory.facts.search(query, MAX_RECALL_RESULTS)]
        episodes = [
            {
                "when": iso_minutes(e.timestamp),
                "ago": relative_time(e.timestamp, now),
                "kind": e.kind,
                "summary": e.summary,
                "location": e.location,
            }
            for e in self.memory.episodes.about(query, MAX_RECALL_EPISODES)
        ]
        result: dict[str, Any] = {"ok": True, "found": bool(found or facts or episodes), "now": iso_minutes(now)}
        result["facts"] = facts
        result["objects"] = found
        result["episodes"] = episodes
        if not result["found"]:
            result["summary"] = self.world.summary()
        return result

    def _what_changed(self, args: dict[str, Any]) -> dict[str, Any]:
        now = self._clock()
        since = _since(now, args.get("since"), args.get("since_hours"))
        episodes = self.memory.episodes.changes_since(since, MAX_CHANGES)
        changes = [
            {
                "when": iso_minutes(e.timestamp),
                "ago": relative_time(e.timestamp, now),
                "kind": e.kind,
                "object": e.subjects[0] if e.subjects else None,
                "summary": e.summary,
                "location": e.location,
            }
            for e in episodes
        ]
        result: dict[str, Any] = {
            "ok": True,
            "now": iso_minutes(now),
            "since": iso_minutes(since),
            "count": len(changes),
            "changes": changes,
        }
        if not changes:
            result["note"] = "no changes noticed in that period"
        # What she did with her free time (#47): explored, looked out of a window, rested, ...
        activities = [
            {
                "when": iso_minutes(e.timestamp),
                "ago": relative_time(e.timestamp, now),
                "summary": e.summary,
                "location": e.location,
            }
            for e in self.memory.episodes.activities_since(since, MAX_CHANGES)
        ]
        if activities:
            result["activities"] = activities
        return result

    @staticmethod
    def _fact_result(fact: Fact, now: float) -> dict[str, Any]:
        return {
            "id": fact.id,
            "text": fact.text,
            "subject": fact.subject,
            "kind": fact.kind,
            "source": fact.source,
            "updated": iso_minutes(fact.updated),
            "ago": relative_time(fact.updated, now),
        }

    def _object_result(self, obj: KnownObject, now: float) -> dict[str, Any]:
        result = _object_dict(obj)
        result["last_seen"] = iso_minutes(obj.last_seen)
        result["ago"] = relative_time(obj.last_seen, now)
        # Have I seen it before, and is it normally here? usual_room is None until a room is a habit.
        usual = self.world.usual_room(obj.name)
        result["first_seen"] = iso_minutes(obj.first_seen)
        result["first_seen_ago"] = relative_time(obj.first_seen, now)
        result["times_seen"] = obj.times_seen
        result["usual_room"] = usual
        result["is_usual_here"] = None if usual is None or obj.room is None else usual == obj.room
        if obj.previous_rooms:
            result["previous_rooms"] = list(obj.previous_rooms[-2:])
        return result

    def _gesture(self, args: dict[str, Any]) -> dict[str, Any]:
        assert self.actions is not None  # the tool is only built with an ActionPlayer
        if (gone := self._preempted()) is not None:
            return gone
        self.actions.enqueue(args["name"])
        return {"ok": True, "gesture": args["name"]}
