"""The tools the AI acts through, and the validator every call passes first.

The model only proposes tool calls (`ai_client.ToolCall`). `ToolRegistry.run`
checks each one (known and offered tool, stale turn, schema, then the typed
`laika.commands` parsing) before any handler touches the robot. Wheel movement
goes through `SafetyController.execute_raw`, so the safety layer has the last
word. `run` never raises: the model gets `{"ok": true, ...}` or
`{"ok": false, "error": "..."}` back and can recover.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping

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
from laika.limits import PAN_MAX, PAN_MIN, STEER_MAX, STEER_MIN, TILT_MAX, TILT_MIN
from laika.robot import (
    CAMERA,
    CAMERA_PAN,
    CAMERA_TILT,
    DRIVE,
    SPEAKER,
    STEER,
    ULTRASONIC,
    Robot,
)
from laika.safety import SafetyController
from laika.observations import Observer
from laika.scan import DEFAULT_SETTLE_S, ScanBusy, Scanner
from laika.world_model import KnownObject, WorldModel, normalise

if TYPE_CHECKING:
    from laika.behaviour import Lease
    from laika.explore import Explorer
    from laika.investigate import Investigator

MAX_GESTURES_PER_TURN = 3
MAX_FACTS = 50
MAX_TEXT_CHARS = 200
MAX_RECALL_RESULTS = 8
REJECTION_HISTORY = 50
OBSERVATION_WAIT_S = 2.0
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


# --- small free-text memory -----------------------------------------------


class Memory:
    """A bounded list of free-text facts, for what does not fit an object in the world model."""

    def __init__(self, max_facts: int = MAX_FACTS) -> None:
        self._facts: deque[str] = deque(maxlen=max(1, max_facts))
        self._lock = threading.Lock()

    def add(self, fact: str) -> None:
        text = " ".join(fact.split())
        with self._lock:
            if text in self._facts:
                self._facts.remove(text)
            self._facts.append(text)

    def search(self, query: str) -> list[str]:
        """Facts containing `query`, newest first."""
        needle = normalise(query)
        with self._lock:
            return [f for f in reversed(self._facts) if needle in f.lower()]

    def __len__(self) -> int:
        return len(self._facts)


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
        camera: Camera | None = None,
        observer: Observer | None = None,
        observation_wait_s: float = OBSERVATION_WAIT_S,
        scan_settle_s: float = DEFAULT_SETTLE_S,
        allow_driving: bool = True,
        explorer: Explorer | None = None,
        investigator: Investigator | None = None,
    ) -> None:
        self.robot = robot
        self.safety = safety
        self.world = world if world is not None else WorldModel()
        self.actions = actions
        self.memory = memory if memory is not None else Memory()
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
        try:
            result = tool.handler(arguments)
        finally:
            self._running.lease = None
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
                "velocità, distanza e stanza.",
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
                "Leggi tutti i sensori: distanza, bordo (scale) e scala di grigi.",
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
                "Ricorda qualcosa che ti dicono: un oggetto (name, con room e note) "
                "oppure un fatto libero (fact).",
                schema(
                    name={"type": "string", "minLength": 1, "maxLength": 60},
                    room={"type": "string", "minLength": 1, "maxLength": 60},
                    note={"type": "string", "minLength": 1, "maxLength": MAX_TEXT_CHARS},
                    fact={"type": "string", "minLength": 1, "maxLength": MAX_TEXT_CHARS},
                ),
                self._remember,
            ),
            Tool(
                "recall",
                "Cerca nei ricordi: un oggetto, una stanza o una parola dei fatti.",
                schema(("query",), query={"type": "string", "minLength": 1, "maxLength": 100}),
                self._recall,
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
        return {
            "ok": True,
            "robot": self.robot.identity().to_dict(),
            "state": state.to_telemetry().to_dict(),
            "room": self.world.current_room,
            "driving_enabled": "move" in self._tools and self.safety.enabled,
            "emergency_stopped": self.safety.emergency_stopped,
        }

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
        if not self.explorer.start(args.get("duration_s")):
            return {"ok": False, "error": f"cannot explore: {self.explorer.refusal}"}
        result: dict[str, Any] = {
            "ok": True,
            "started": True,
            "mode": self.explorer.mode,
            "note": "she starts as soon as the conversation is over",
        }
        if self.explorer.mode == "head_only":
            result["note"] += "; the wheels stay still (autonomous driving is off), she only looks around"
        return result

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
        answer = " ".join(args["answer"].split())
        self.world.learn(question.object, room=question.room, note=answer)
        self.memory.add(f"{question.object}: {answer}")
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
        return {
            "ok": True,
            "distance_cm": sensors.distance_cm,
            "cliff_detected": sensors.cliff_detected,
            "grayscale": list(sensors.grayscale),
        }

    def _say(self, args: dict[str, Any]) -> dict[str, Any]:
        command = SpeakCommand(text=args["text"])
        self._speak(command.text)
        return {"ok": True}

    def _remember(self, args: dict[str, Any]) -> dict[str, Any]:
        name, fact = args.get("name"), args.get("fact")
        if not (name and name.strip()) and not (fact and fact.strip()):
            return {"ok": False, "error": "give a name (an object) or a fact to remember"}
        stored: list[str] = []
        if name and name.strip():
            self.world.learn(name, room=args.get("room"), note=args.get("note"))
            stored.append(f"object: {normalise(name)}")
        if fact and fact.strip():
            self.memory.add(fact)
            stored.append("fact")
        return {"ok": True, "stored": stored}

    def _recall(self, args: dict[str, Any]) -> dict[str, Any]:
        query = normalise(args["query"])
        objects: dict[str, KnownObject] = {}
        exact = self.world.where_is(query)
        if exact is not None:
            objects[exact.name] = exact
        room = query.replace(" ", "_")
        if room in self.world.rooms():
            for obj in self.world.objects_in(room):
                objects.setdefault(obj.name, obj)
        for raw in self.world.to_dict()["objects"]:
            known = KnownObject.from_dict(raw)
            if query in known.name or (known.note and query in known.note.lower()):
                objects.setdefault(known.name, known)
        found = [_object_dict(o) for o in list(objects.values())[:MAX_RECALL_RESULTS]]
        facts = self.memory.search(query)[:MAX_RECALL_RESULTS]
        result: dict[str, Any] = {"ok": True, "found": bool(found or facts)}
        result["objects"] = found
        result["facts"] = facts
        if not result["found"]:
            result["summary"] = self.world.summary()
        return result

    def _gesture(self, args: dict[str, Any]) -> dict[str, Any]:
        assert self.actions is not None  # the tool is only built with an ActionPlayer
        if (gone := self._preempted()) is not None:
            return gone
        self.actions.enqueue(args["name"])
        return {"ok": True, "gesture": args["name"]}
