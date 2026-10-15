"""Build Laika from the configuration and run her until Ctrl+C."""

from __future__ import annotations

import argparse
import signal
import threading
from collections.abc import Callable
from pathlib import Path

from laika import __version__
from laika.actions import MAX_ACTIONS_PER_REPLY, ActionContext, ActionPlayer, available_actions
from laika.agent import Agent
from laika.ai_client import AIConfigError, ChatClient, Endpoint
from laika.camera import Camera
from laika.capture_policy import CapturePolicy
from laika.behaviour import Arbiter
from laika.audio import TTS_ENGINES, ConsoleEar, ConsoleMouth, VoskEar, make_mouth
from laika.config import DEFAULT_CONFIG_PATH, load_config, permission_warning
from laika.conversation import Conversation
from laika.curiosity import InterestGate, InterestScorer
from laika.explore import Explorer, IdleStarter
from laika.events import EventBus, EventType, log_events_to_console
from laika.house import default_house
from laika.investigate import Investigator
from laika.prompts import build_system_prompt, build_tool_system_prompt
from laika.safety import SafetyController
from laika.scan import Scanner
from laika.simulator import SimulatedRobot
from laika.storage import DataPaths, ImageStore, attach_storage, resolve_data_dir
from laika.store import DB_NAME, ObservationArchiver, Store, seed_observation_log
from laika.systemd import Notifier
from laika.memory import EpisodeRecorder, Memory
from laika.tools import MAX_GESTURES_PER_TURN, OBSERVATION_WAIT_S, ToolRegistry
from laika.observations import ObservationLog, Observer
from laika.vision import VisionDescriber, build_vision_client
from laika.world_model import WorldModel, WorldModelPersister, WorldModelUpdater


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="run-laika", description="Start Laika.")
    parser.add_argument(
        "--sim",
        action="store_true",
        help="no robot: simulated hardware, typed input and printed replies (works on a laptop)",
    )
    parser.add_argument(
        "--keyboard",
        action="store_true",
        help="type instead of speaking; Laika still talks and moves",
    )
    parser.add_argument(
        "--no-driving", action="store_true", help="head and steering gestures only, wheels never spin"
    )
    parser.add_argument(
        "--events", action="store_true", help="print every robot event as one JSON line"
    )
    parser.add_argument(
        "--check-hardware",
        action="store_true",
        help="guided check of every robot capability, with a report (add --sim for a dry run)",
    )
    parser.add_argument(
        "--yes", action="store_true", help="with --check-hardware: answer yes to every question"
    )
    parser.add_argument(
        "--no-mic", action="store_true", help="with --check-hardware: skip the microphone step"
    )
    parser.add_argument(
        "--stop-motors",
        action="store_true",
        help="only put the robot in a stopped state and exit 0 (used by the systemd unit)",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="settings file (TOML)"
    )
    return parser.parse_args(argv)


class Terminated(KeyboardInterrupt):
    """Raised in the main thread by SIGTERM or SIGHUP so the normal shutdown path runs."""


HALT_WAIT_S = 1.0


def install_shutdown_handlers(halt: Callable[[], None]) -> Callable[[], None]:
    """Turn SIGTERM and SIGHUP into `Terminated`; returns a function that undoes it.

    systemd stops a service with SIGTERM and Python's default would exit without
    running `finally`. The handler first halts the motors (on a helper thread,
    in case the main thread holds a lock the halt needs) and then raises. Later
    signals are ignored so they cannot interrupt the shutdown itself. A main
    thread stuck in a C call never runs the handler: the unit's TimeoutStopSec
    and ExecStopPost cover that.
    """
    fired = threading.Event()

    def handler(signum: int, _frame: object) -> None:
        if fired.is_set():
            return
        fired.set()
        name = signal.Signals(signum).name
        print(f"\n[laika] {name} received, stopping the robot")
        worker = threading.Thread(target=halt, name="laika-halt", daemon=True)
        worker.start()
        worker.join(HALT_WAIT_S)
        raise Terminated(name)

    previous: dict[int, object] = {}
    for name in ("SIGTERM", "SIGHUP"):  # SIGHUP does not exist on Windows
        number = getattr(signal, name, None)
        if number is not None:
            previous[number] = signal.signal(number, handler)

    def restore() -> None:
        for number, old in previous.items():
            signal.signal(number, old)  # type: ignore[arg-type]

    return restore


def stop_motors(sim: bool) -> int:
    """Put the robot in a stopped state and release it. Always returns 0.

    Runs as ExecStartPre and ExecStopPost. A failure here (no hardware, another
    process holds it) is printed but must not mark the service as failed or block
    its start: exit 0 on purpose.
    """
    try:
        if sim:
            robot = SimulatedRobot()
        else:
            from laika.hardware import PiCarXRobot

            robot = PiCarXRobot()
    except Exception as error:  # noqa: BLE001
        print(f"[stop-motors] cannot reach the robot: {error}")
        return 0
    failed = False
    for name, action in (("stop", robot.stop), ("reset", robot.reset), ("close", robot.close)):
        try:
            action()
        except Exception as error:  # noqa: BLE001
            failed = True
            print(f"[stop-motors] {name} failed: {error}")
    print("[stop-motors] done with errors" if failed else "[stop-motors] robot stopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stop_motors:
        return stop_motors(args.sim)
    try:
        config = load_config(args.config)
    except ValueError as error:
        print(f"[laika] settings error in {args.config}: {error}")
        return 2
    if args.sim:
        # Only the simulator: on the robot it stays what [robot] allow_autonomous_driving says.
        config.robot.allow_autonomous_driving = True
    warning = permission_warning(args.config)
    if warning:
        print(f"[laika] warning: {warning}")
    if args.check_hardware:
        from laika.hardware_check import run_from_args

        print(f"Laika {__version__} hardware check ({'simulator' if args.sim else 'PiCar-X'})")
        return run_from_args(args, config)
    if not config.ai.api_key and config.ai.provider != "openai-compatible":
        print(
            "No AI key configured. Copy config/laika.example.toml to config/laika.toml "
            "and set [ai] api_key, or set the OPENAI_API_KEY environment variable."
        )
        return 2
    try:
        Endpoint.build(
            provider=config.ai.provider,
            base_url=config.ai.base_url,
            api_key=config.ai.api_key,
            model=config.ai.model,
            azure_deployment=config.ai.azure_deployment,
            azure_api_version=config.ai.azure_api_version,
        )
    except AIConfigError as error:
        print(error)
        return 2
    if config.voice.tts_engine not in TTS_ENGINES:
        print(f"[voice] tts_engine must be one of: {', '.join(TTS_ENGINES)}")
        return 2

    print(f"Laika {__version__} ({'simulator' if args.sim else 'PiCar-X'}, model {config.ai.model})")

    if args.sim:
        mouth = ConsoleMouth()
    else:
        mouth = make_mouth(
            config.voice.tts_engine,
            edge_voice=config.voice.edge_voice,
            piper_model=config.voice.tts_model,
        )
    try:
        paths = DataPaths(resolve_data_dir(config.storage.data_dir)).ensure()
        event_file, transcript = attach_storage(config.storage, paths)
    except OSError as error:
        print(f"Cannot use the data directory: {error}. Set [storage] data_dir or LAIKA_DATA_DIR.")
        return 2
    print(f"[laika] data directory: {paths.root}")
    events = EventBus()
    events.subscribe(event_file)
    if args.events:
        events.subscribe(log_events_to_console)
    events.start()
    # Observations are kept on the Pi; the store never raises and never blocks the safety layer.
    store = Store(
        paths.memory / DB_NAME,
        max_observations=config.storage.max_observations,
        max_observation_days=config.storage.max_observation_days,
        max_episodes=config.storage.max_episodes,
        max_episode_days=config.storage.max_episode_days,
    )
    observation_log = ObservationLog()
    restored = seed_observation_log(store, observation_log)
    archiver = ObservationArchiver(store, events)
    if restored:
        print(f"[laika] remembered {restored} recent observations from the last run")
    # The model grows from events in every mode; the agent reads it through the recall tool.
    # It starts from what the last run stored (#33) and every later change is written back.
    world = WorldModel()
    world.load(*store.load_world())
    known = world.to_dict()
    if known["objects"] or known["rooms"]:
        print(f"[laika] world model from the last run: {len(known['rooms'])} rooms, {len(known['objects'])} objects")
    world_persister = WorldModelPersister(world, store)
    world_updater = WorldModelUpdater(world, events)

    # The robot speaks through the mouth, so the agent only talks to the Robot interface.
    if args.sim:
        robot = SimulatedRobot(mouth=mouth, house=default_house(), events=events)
        world.enter_room(robot.current_room())
    else:
        from laika.hardware import PiCarXRobot

        robot = PiCarXRobot(
            mouth=mouth,
            image_size=config.camera.image_size(),
            jpeg_quality=config.camera.quality(),
            cliff_reference=config.robot.cliff_reference,
        )
    ear = ConsoleEar() if args.sim or args.keyboard else VoskEar(config.voice.stt_language)

    camera = Camera(
        robot,
        ImageStore(paths.images, config.storage.max_images, config.storage.max_images_mb),
        events,
        min_interval_s=config.camera.min_interval_s,
        max_per_minute=config.camera.max_per_minute,
        save_images=config.camera.save_images,
    )
    # Facts and episodes (#34) come back from the store; the tools, the agent prompt and the
    # curiosity scorer (#28) share them, and every sighting is recorded as a "seen" episode.
    memory = Memory(store)
    loaded_facts, loaded_episodes = memory.load()
    if loaded_facts or loaded_episodes:
        print(f"[laika] memory from the last run: {loaded_facts} facts, {loaded_episodes} recent episodes")
    episode_recorder = EpisodeRecorder(
        memory.episodes, events, interval_s=config.storage.seen_episode_interval_s
    )
    # Frames become validated observations off the caller's thread; the world model listens.
    # The scorer runs before each observation is published, so it sees the world as it was.
    observer = Observer(
        events,
        observation_log,
        default_location=lambda: world.current_room,
        scorer=InterestScorer(world, memory=memory, config=config.curiosity),
        gate=InterestGate(config.curiosity, events),
    )
    observer.attach(camera)
    # Automatic photos (after moves, people, changes) go through the policy, never the caller's thread.
    capture_policy = CapturePolicy(camera, events, config.capture)
    capture_policy.start()

    identity = robot.identity()
    events.emit(EventType.HARDWARE_READY, robotId=identity.robot_id, name=identity.name)
    print(
        f"[robot] {identity.name} ({identity.robot_id}) {identity.version}, "
        f"capabilities: {', '.join(sorted(identity.capabilities))}"
    )
    allowed = available_actions(
        identity.capabilities, config.robot.allow_driving and not args.no_driving
    )
    notifier = Notifier.from_env()
    safety = SafetyController(
        robot,
        enabled=config.robot.allow_driving and not args.no_driving,
        speed=config.robot.drive_speed,
        max_duration_s=config.robot.max_drive_s,
        backward_max_s=config.robot.backward_max_s,
        min_front_distance_cm=config.robot.min_front_distance_cm,
        cadence_hz=config.robot.safety_cadence_hz,
        command_max_age_s=config.robot.command_max_age_s,
        ultrasonic_invalid_limit=config.robot.ultrasonic_invalid_limit,
        events=events,
        watchdog_timeout_s=config.robot.watchdog_timeout_s,
        heartbeat=notifier.watchdog,
        allow_autonomous=config.robot.allow_autonomous_driving,
    )
    safety.start()
    if not config.robot.allow_autonomous_driving:
        print(
            "[laika] autonomous driving is off ([robot] allow_autonomous_driving): "
            "exploration only looks around, the wheels never move on their own"
        )
    actions = ActionPlayer(
        ActionContext(robot, safety, config.robot.honk_volume),
        allowed,
        talk_motion=config.robot.talk_motion,
    )
    # Who owns the body: the conversation now, exploration and the rest later.
    arbiter = Arbiter(safety, actions)
    ai = ChatClient(
        base_url=config.ai.base_url,
        api_key=config.ai.api_key,
        model=config.ai.model,
        system_prompt=build_system_prompt(allowed, MAX_ACTIONS_PER_REPLY),
        timeout_s=config.ai.timeout_s,
        max_history_turns=config.ai.max_history_turns,
        temperature=config.ai.temperature,
        provider=config.ai.provider,
        azure_deployment=config.ai.azure_deployment,
        azure_api_version=config.ai.azure_api_version,
    )

    conversation = Conversation(
        ear=ear,
        robot=robot,
        ai=ai,
        actions=actions,
        voice=config.voice,
        events=events,
        safety=safety,
        transcript=transcript,
        arbiter=arbiter,
    )
    observation_wait_s = OBSERVATION_WAIT_S
    if config.vision.enabled and not args.sim:
        # The simulator keeps its ground-truth path. Vision shares the conversation's endpoint
        # monitor, so an outage seen here stops movement exactly like one seen while talking.
        observer.describe = VisionDescriber(
            build_vision_client(config.ai, config.vision), config.vision, conversation.endpoint
        )
        print(f"[laika] vision: {config.vision.model or config.ai.model}, detail {config.vision.detail}")
        # Wait for the vision reply (plus the queue) rather than answering without it.
        observation_wait_s = config.vision.timeout_s + 2
    investigator: Investigator | None = None
    if config.investigate.enabled:
        # Reacts to InterestingChangeDetected and to exploration scans; never speaks on its own.
        investigator = Investigator(
            arbiter,
            safety,
            robot,
            camera,
            observer,
            world,
            memory,
            events,
            config.investigate,
            threshold=config.curiosity.threshold,
            unknown_confidence=config.curiosity.unknown_object_confidence,
            observation_wait_s=observation_wait_s,
        )
        investigator.start()
    explorer: Explorer | None = None
    idle_starter: IdleStarter | None = None
    if config.explore.enabled:
        explorer = Explorer(
            arbiter,
            safety,
            Scanner(robot, camera, observer, observation_wait_s=observation_wait_s),
            capture_policy,
            config=config.explore,
            on_scan=investigator.consider_scan if investigator is not None else None,
        )
        idle_starter = IdleStarter(
            explorer, arbiter, idle_after_s=config.explore.idle_after_s
        )
        idle_starter.start()
    if config.ai.use_tools:
        # The speak tool goes through the conversation, so Laika moves her head while talking.
        registry = ToolRegistry(
            robot,
            safety,
            world=world,
            actions=actions,
            memory=memory,
            camera=camera,
            observer=observer,
            observation_wait_s=observation_wait_s,
            speak=conversation.speak,
            allow_driving=config.robot.allow_driving and not args.no_driving,
            explorer=explorer,
            investigator=investigator,
        )
        driving = "move" in registry.names()
        conversation.agent = Agent(
            ai,
            registry,
            build_tool_system_prompt(
                allowed,
                MAX_GESTURES_PER_TURN,
                driving,
                "start_exploring" in registry.names(),
                "investigate" in registry.names(),
            ),
            world=world,
            questions=investigator.questions if investigator is not None else None,
            memory=memory,
            max_history_turns=config.ai.max_history_turns,
        )
        print(f"[laika] tools: {', '.join(registry.names())}")
    else:
        print("[laika] use_tools is off: the model replies with JSON actions")

    restore_signals = install_shutdown_handlers(lambda: safety.emergency_stop("shutdown signal"))
    try:
        notifier.ready()
        conversation.run()
    except KeyboardInterrupt:
        print("\n[laika] stopping")
    finally:
        notifier.stopping()
        if idle_starter is not None:
            idle_starter.close()
        if explorer is not None:
            explorer.stop()
        if investigator is not None:
            investigator.close()
        arbiter.close()
        safety.close()
        capture_policy.close()
        actions.close()
        ear.close()
        robot.close()
        observer.close()
        world_updater.close()
        world_persister.close()
        episode_recorder.close()
        events.close()
        archiver.close()
        store.close()
        event_file.close()
        restore_signals()
    return 0
