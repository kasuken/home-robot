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
from laika.audio import TTS_ENGINES, ConsoleEar, ConsoleMouth, VoskEar, make_mouth
from laika.config import DEFAULT_CONFIG_PATH, load_config, permission_warning
from laika.conversation import Conversation
from laika.events import EventBus, EventType, log_events_to_console
from laika.house import default_house
from laika.prompts import build_system_prompt, build_tool_system_prompt
from laika.safety import SafetyController
from laika.simulator import SimulatedRobot
from laika.storage import DataPaths, ImageStore, attach_storage, resolve_data_dir
from laika.systemd import Notifier
from laika.tools import MAX_GESTURES_PER_TURN, OBSERVATION_WAIT_S, ToolRegistry
from laika.observations import Observer
from laika.vision import VisionDescriber, build_vision_client
from laika.world_model import WorldModel, WorldModelUpdater


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
    # The model grows from events in every mode; the agent reads it through the recall tool.
    world = WorldModel()
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
    # Frames become validated observations off the caller's thread; the world model listens.
    observer = Observer(events, default_location=lambda: world.current_room)
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
    )
    safety.start()
    actions = ActionPlayer(
        ActionContext(robot, safety, config.robot.honk_volume),
        allowed,
        talk_motion=config.robot.talk_motion,
    )
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
    if config.ai.use_tools:
        # The speak tool goes through the conversation, so Laika moves her head while talking.
        registry = ToolRegistry(
            robot,
            safety,
            world=world,
            actions=actions,
            camera=camera,
            observer=observer,
            observation_wait_s=observation_wait_s,
            speak=conversation.speak,
            allow_driving=config.robot.allow_driving and not args.no_driving,
        )
        driving = "move" in registry.names()
        conversation.agent = Agent(
            ai,
            registry,
            build_tool_system_prompt(allowed, MAX_GESTURES_PER_TURN, driving),
            world=world,
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
        safety.close()
        capture_policy.close()
        actions.close()
        ear.close()
        robot.close()
        observer.close()
        world_updater.close()
        events.close()
        event_file.close()
        restore_signals()
    return 0
