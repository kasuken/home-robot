"""Build Laika from the configuration and run her until Ctrl+C."""

from __future__ import annotations

import argparse
from pathlib import Path

from laika import __version__
from laika.actions import MAX_ACTIONS_PER_REPLY, ActionContext, ActionPlayer, available_actions
from laika.agent import Agent
from laika.ai_client import AIConfigError, ChatClient, Endpoint
from laika.audio import TTS_ENGINES, ConsoleEar, ConsoleMouth, VoskEar, make_mouth
from laika.config import DEFAULT_CONFIG_PATH, load_config, permission_warning
from laika.conversation import Conversation
from laika.events import EventBus, EventType, log_events_to_console
from laika.house import default_house
from laika.prompts import build_system_prompt, build_tool_system_prompt
from laika.safety import SafetyController
from laika.simulator import SimulatedRobot
from laika.storage import DataPaths, attach_storage, resolve_data_dir
from laika.tools import MAX_GESTURES_PER_TURN, ToolRegistry
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
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="settings file (TOML)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    warning = permission_warning(args.config)
    if warning:
        print(f"[laika] warning: {warning}")
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

        robot = PiCarXRobot(mouth=mouth)
    ear = ConsoleEar() if args.sim or args.keyboard else VoskEar(config.voice.stt_language)

    identity = robot.identity()
    events.emit(EventType.HARDWARE_READY, robotId=identity.robot_id, name=identity.name)
    print(
        f"[robot] {identity.name} ({identity.robot_id}) {identity.version}, "
        f"capabilities: {', '.join(sorted(identity.capabilities))}"
    )
    allowed = available_actions(
        identity.capabilities, config.robot.allow_driving and not args.no_driving
    )
    safety = SafetyController(
        robot,
        enabled=config.robot.allow_driving and not args.no_driving,
        speed=config.robot.drive_speed,
        max_duration_s=config.robot.max_drive_s,
        min_front_distance_cm=config.robot.min_front_distance_cm,
        events=events,
        watchdog_timeout_s=config.robot.watchdog_timeout_s,
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
    if config.ai.use_tools:
        # The speak tool goes through the conversation, so Laika moves her head while talking.
        registry = ToolRegistry(
            robot,
            safety,
            world=world,
            actions=actions,
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

    try:
        conversation.run()
    except KeyboardInterrupt:
        print("\n[laika] stopping")
    finally:
        safety.close()
        actions.close()
        ear.close()
        robot.close()
        world_updater.close()
        events.close()
        event_file.close()
    return 0
