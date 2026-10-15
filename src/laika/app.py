"""Build Laika from the configuration and run her until Ctrl+C."""

from __future__ import annotations

import argparse
from pathlib import Path

from laika import __version__
from laika.actions import MAX_ACTIONS_PER_REPLY, ActionContext, ActionPlayer, available_actions
from laika.ai_client import ChatClient
from laika.audio import TTS_ENGINES, ConsoleEar, ConsoleMouth, VoskEar, make_mouth
from laika.config import DEFAULT_CONFIG_PATH, load_config
from laika.conversation import Conversation
from laika.prompts import build_system_prompt
from laika.safety import MovementGuard
from laika.simulator import SimulatedRobot


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
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="settings file (TOML)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if not config.ai.api_key:
        print(
            "No AI key configured. Copy config/laika.example.toml to config/laika.toml "
            "and set [ai] api_key, or set the OPENAI_API_KEY environment variable."
        )
        return 2
    if config.voice.tts_engine not in TTS_ENGINES:
        print(f"[voice] tts_engine must be one of: {', '.join(TTS_ENGINES)}")
        return 2

    print(f"Laika {__version__} ({'simulator' if args.sim else 'PiCar-X'}, model {config.ai.model})")

    if args.sim:
        robot = SimulatedRobot()
    else:
        from laika.hardware import PiCarXRobot

        robot = PiCarXRobot()

    allowed = available_actions(config.robot.allow_driving and not args.no_driving)
    guard = MovementGuard(
        robot,
        enabled=config.robot.allow_driving and not args.no_driving,
        speed=config.robot.drive_speed,
        max_duration_s=config.robot.max_drive_s,
        min_front_distance_cm=config.robot.min_front_distance_cm,
    )
    actions = ActionPlayer(
        ActionContext(robot, guard, config.robot.honk_volume),
        allowed,
        talk_motion=config.robot.talk_motion,
    )
    ear = ConsoleEar() if args.sim or args.keyboard else VoskEar(config.voice.stt_language)
    if args.sim:
        mouth = ConsoleMouth()
    else:
        mouth = make_mouth(
            config.voice.tts_engine,
            edge_voice=config.voice.edge_voice,
            piper_model=config.voice.tts_model,
        )
    ai = ChatClient(
        base_url=config.ai.base_url,
        api_key=config.ai.api_key,
        model=config.ai.model,
        system_prompt=build_system_prompt(allowed, MAX_ACTIONS_PER_REPLY),
        timeout_s=config.ai.timeout_s,
        max_history_turns=config.ai.max_history_turns,
        temperature=config.ai.temperature,
    )

    try:
        Conversation(ear=ear, mouth=mouth, ai=ai, actions=actions, voice=config.voice).run()
    except KeyboardInterrupt:
        print("\n[laika] stopping")
    finally:
        actions.close()
        ear.close()
        robot.close()
    return 0
