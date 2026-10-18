"""Build Laika from the configuration and run her until Ctrl+C."""

from __future__ import annotations

import argparse
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from laika import __version__
from laika.actions import MAX_ACTIONS_PER_REPLY, ActionContext, ActionPlayer, available_actions
from laika.robot import BATTERY
from laika.agent import Agent
from laika.ai_client import AIConfigError, ChatClient, Endpoint
from laika.camera import Camera
from laika.capture_policy import CapturePolicy
from laika.presence import PresenceTracker
from laika.battery import BatteryMonitor
from laika.behaviour import Arbiter, Priority
from laika.audio import TTS_ENGINES, CloudEar, ConsoleEar, ConsoleMouth, VoskEar, make_mouth
from laika.config import DEFAULT_CONFIG_PATH, Config, committed_key_warning, load_config, permission_warning
from laika.conversation import Conversation
from laika.conversation_memory import MemorySelector
from laika.changes import ChangeDetector, ChangeTracker
from laika.curiosity import InterestGate, InterestScorer
from laika.explore import Explorer
from laika.events import EventBus, EventType, log_events_to_console
from laika.autonomy import AutonomyPause, DriveBudget
from laika.house import default_house
from laika.house_rules import HouseRules
from laika.idle import IdleSelector
from laika.interaction_policy import InteractionPolicy
from laika.encounter import Encounters
from laika.investigate import Investigator, PendingQuestions
from laika.outside import OutsideDescriber, WindowWatcher
from laika.proactive import ProactiveInteraction
from laika.rooms import RoomTracker, prompt_rooms
from laika.prompts import build_system_prompt, build_tool_system_prompt
from laika.safety import SafetyController
from laika.search import Searcher
from laika.web import WebLookup
from laika.scan import Scanner
from laika.simulator import SimulatedRobot
from laika.storage import DataPaths, ImageStore, attach_storage, resolve_data_dir
from laika.store import DB_NAME, ObservationArchiver, Store, seed_observation_log
from laika.systemd import Notifier
from laika.personality import curiosity_threshold, investigate_max_per_hour
from laika.memory import EpisodeRecorder, Memory
from laika.places import PlaceNotes
from laika.transcribe import Transcriber
from laika.wake_word import WakeWordMatcher
from laika.floorplan import load_plan, seed_world
from laika.tags import OpenCVTagDetector, TagReader
from laika.learning import LearningSettings, SkillLearner
from laika.bedtime import STORY_TIMEOUT_S, BedtimeStories, StoryArchive
from laika.games import GameSession
from laika.tools import MAX_GESTURES_PER_TURN, OBSERVATION_WAIT_S, ToolRegistry
from laika.observations import ObservationLog, Observer
from laika.vision import VisionDescriber, build_vision_client
from laika.windows import WindowRegistry
from laika.world_model import WorldModel, WorldModelPersister, WorldModelUpdater


VIEW_MAX_AGE_S = 120.0


def objects_in_view(log: ObservationLog) -> list[str]:
    """The objects of the latest observation, when it is recent enough to be what she is looking at."""
    latest = log.last()
    if latest is None or time.time() - latest.timestamp > VIEW_MAX_AGE_S:
        return []
    return latest.object_names


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
        "--unattended-check",
        action="store_true",
        help="print the checklist for letting her drive around alone; exit 1 when it is not met (read-only)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="field-trial report from the event log and the store (read-only, needs no AI key or hardware)",
    )
    parser.add_argument(
        "--days", type=int, default=None, metavar="N", help="with --report: how many days to cover (default 1)"
    )
    parser.add_argument(
        "--since", metavar="YYYY-MM-DD", default=None, help="with --report: start at this local date"
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


def make_ear(config: Config) -> Any:
    """The robot's ear: Vosk for the wake word plus cloud transcription ([voice] stt_engine), else Vosk alone."""
    voice, ai = config.voice, config.ai
    if voice.stt_engine == "cloud":
        if ai.provider != "openai" or not ai.api_key:
            print("[laika] speech: cloud recognition needs [ai] provider \"openai\" and a key; using Vosk only")
        else:
            try:
                ear = CloudEar(
                    voice.stt_language,
                    Transcriber(base_url=ai.base_url, api_key=ai.api_key, model=voice.stt_model, language=voice.stt_language),
                    wants_cloud=WakeWordMatcher(voice.wake_names).match,
                )
            except Exception as error:  # noqa: BLE001 - no sounddevice/vosk, no microphone: Vosk as before
                print(f"[laika] speech: cloud recognition unavailable ({error}); using Vosk only")
            else:
                print(f"[laika] speech: Vosk waits for the wake word, then {voice.stt_model} transcribes")
                return ear
    return VoskEar(voice.stt_language)


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
    if args.report:
        from laika.trial_report import run_from_args as run_report

        return run_report(args, config)
    if args.unattended_check:
        from laika.unattended import run_from_args as run_unattended_check

        return run_unattended_check(config)
    if args.sim:
        # Only the simulator: on the robot it stays what [robot] allow_autonomous_driving says.
        config.robot.allow_autonomous_driving = True
    for warning in (permission_warning(args.config), committed_key_warning(args.config)):
        if warning:
            print(f"[laika] warning: {warning}")
    # The personality (#41) shifts curiosity: the threshold and the investigator's hourly limit, bounded.
    config.curiosity = replace(
        config.curiosity, threshold=curiosity_threshold(config.curiosity.threshold, config.personality.curiosity)
    )
    config.investigate = replace(
        config.investigate,
        max_per_hour=investigate_max_per_hour(config.investigate.max_per_hour, config.personality.curiosity),
    )
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
    print(f"[laika] personality: {config.personality.describe()}")

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
    if not args.sim:
        from laika.unattended import startup_warning

        unattended = startup_warning(config, paths.root)
        if unattended:
            print(f"[laika] warning: {unattended}")
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
    # The plan of the house, if there is one: room names for the world model, no-go rooms for the house rules.
    plan = None
    if config.house.plan:
        try:
            plan = load_plan(args.config.parent / config.house.plan)
        except ValueError as error:
            print(f"[laika] house plan ignored: {error}")
    if plan is not None:
        seed_world(world, plan)
        config.house.no_go_rooms = list(dict.fromkeys([*config.house.no_go_rooms, *plan.no_go_rooms()]))
        print(f"[laika] house plan: {len(plan)} rooms, home spot {plan.home_room or 'none'}")
    world.load(*store.load_world())
    known = world.to_dict()
    if known["objects"] or known["rooms"]:
        print(f"[laika] world model from the last run: {len(known['rooms'])} rooms, {len(known['objects'])} objects")
    world_persister = WorldModelPersister(world, store)
    world_updater = WorldModelUpdater(world, events)
    # Where the windows are (#48): learned from observations and from what people say, kept in the store.
    windows = WindowRegistry(store, events)
    if known_windows := windows.load():
        print(f"[laika] windows from the last run: {known_windows}")

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
    ear = ConsoleEar() if args.sim or args.keyboard else make_ear(config)

    image_store = ImageStore(
        paths.images,
        config.storage.max_images,
        config.storage.max_images_mb,
        config.storage.max_image_days,
    )
    # Old pictures are deleted now and every hour, not only when a new one is saved.
    image_store.start_cleanup()
    camera = Camera(
        robot,
        image_store,
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
    # What each exploration teaches her about the rooms (how they look, where they lead), kept for later.
    places = PlaceNotes(events, store, world=world, memory=memory)
    if places.places():
        print(f"[laika] place notes from earlier explorations: {len(places.places())} rooms")
    # Frames become validated observations off the caller's thread; the world model listens.
    # The change detector (#36) and the scorer run before each observation is published, so they
    # see the world as it was; the changes are remembered as "moved", "new" and "missing" episodes.
    # Room recognition (#53): on the robot the tracker decides which room she is in from the vision model's
    # guesses; in the simulator the ground truth decides and the tracker only runs beside it (shadow mode).
    room_tracker: RoomTracker | None = None
    if config.rooms.enabled and (not args.sim or config.rooms.shadow_in_sim):
        room_tracker = RoomTracker(
            world,
            events,
            config.rooms,
            drive=not args.sim,
            truth=(lambda: robot.current_room()) if args.sim else None,
            plan=plan,
        )
    observer = Observer(
        events,
        observation_log,
        default_location=lambda: world.current_room,
        rooms=room_tracker,
        scorer=InterestScorer(world, memory=memory, config=config.curiosity),
        gate=InterestGate(config.curiosity, events),
        detector=ChangeDetector(world, observation_log, config.curiosity),
        tracker=ChangeTracker(memory.episodes),
    )
    observer.attach(camera)
    # AprilTag fixes: the tags the plan lists, read from the same frames, on their own thread.
    tag_reader = None
    if config.tags.enabled and plan is not None and plan.tags and room_tracker is not None:
        try:
            tag_reader = TagReader(plan, room_tracker, OpenCVTagDetector(), config.tags)
            tag_reader.attach(camera)
            print(f"[laika] tags: reading {len(plan.tags)} AprilTags from the camera frames")
        except ImportError as error:
            print(f"[laika] tags: OpenCV is not installed, so no tag fixes ({error})")
    # Who is near (#38): PersonDetected and PersonLost come from observations only, in every mode.
    presence = PresenceTracker(events, config.presence)
    presence.start()

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
    # The limits for unattended driving (#51): what people said is off limits, a budget, and the pause.
    house_rules = HouseRules(memory, world, config.house)
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
        drive_budget=DriveBudget(
            config.robot.max_autonomous_drive_s_per_hour, config.robot.max_autonomous_drive_s_per_day, store
        ),
        pause=AutonomyPause(store),
        house_rules=house_rules,
    )
    safety.start()
    if not config.robot.allow_autonomous_driving:
        print(
            "[laika] autonomous driving is off ([robot] allow_autonomous_driving): "
            "exploration only looks around, the wheels never move on their own"
        )
    if safety.paused:
        print("[laika] autonomy is paused (the user said 'resta ferma'): she stays still until told she can move again")
    if house_rules.count():
        print(f"[laika] house rules: no-go rooms: {', '.join(sorted(house_rules.no_go_rooms()))}")
    actions = ActionPlayer(
        ActionContext(robot, safety, config.robot.honk_volume),
        allowed,
        talk_motion=config.robot.talk_motion,
    )
    # Who owns the body: the conversation now, exploration and the rest later.
    arbiter = Arbiter(safety, actions)
    # The battery (#50): sampled slowly on its own thread; LOW stops autonomous driving, CRITICAL all driving.
    battery: BatteryMonitor | None = None
    if config.battery.enabled and BATTERY in identity.capabilities:
        battery = BatteryMonitor(robot, events, config.battery, is_driving=lambda: safety.moving)
        battery.add_listener(lambda new, _previous: safety.set_battery_state(new))
        battery.start()  # takes the first readings now, so the state is known before anything moves
        safety.set_battery_state(battery.state)
        print(
            f"[laika] battery: sampled every {config.battery.sample_s:g} s, low below {config.battery.low_v:g} V, "
            f"critical below {config.battery.critical_v:g} V (calibrate them on the robot)"
        )
    # Automatic photos (after moves, people, changes, idle presence checks) go through the policy,
    # never the caller's thread. No presence photos while a conversation holds the body.
    capture_policy = CapturePolicy(
        camera,
        events,
        config.capture,
        presence=config.presence,
        person_present=lambda: presence.present,
        conversation_active=lambda: (arbiter.holder is not None and arbiter.holder.priority <= Priority.CONVERSATION),
    )
    capture_policy.start()
    ai = ChatClient(
        base_url=config.ai.base_url,
        api_key=config.ai.api_key,
        model=config.ai.model,
        system_prompt=build_system_prompt(allowed, MAX_ACTIONS_PER_REPLY, config.personality),
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
        personality=config.personality,
    )
    observation_wait_s = OBSERVATION_WAIT_S
    if config.vision.enabled and not args.sim:
        # The simulator keeps its ground-truth path. Vision shares the conversation's endpoint
        # monitor, so an outage seen here stops movement exactly like one seen while talking.
        observer.describe = VisionDescriber(
            build_vision_client(config.ai, config.vision),
            config.vision,
            conversation.endpoint,
            known_rooms=(lambda: prompt_rooms(world)) if room_tracker is not None else None,
        )
        print(f"[laika] vision: {config.vision.model or config.ai.model}, detail {config.vision.detail}")
        # Wait for the vision reply (plus the queue) rather than answering without it.
        observation_wait_s = config.vision.timeout_s + 2
    # Looking out of the windows (#48): the vision model reads the outside, the simulator's ground truth in --sim.
    window_watcher: WindowWatcher | None = None
    if config.outside.enabled and (args.sim or isinstance(observer.describe, VisionDescriber)):
        window_watcher = WindowWatcher(
            arbiter,
            robot,
            camera,
            windows,
            world,
            memory,
            config.outside,
            describe=OutsideDescriber(observer.describe) if isinstance(observer.describe, VisionDescriber) else None,
            gate=InterestGate(config.curiosity, events),
            store=store,
        )
        print(
            f"[laika] windows: she looks out at most every {config.outside.min_interval_s:g} s per window, "
            f"{config.outside.max_per_day} times a day (on the robot each look is a vision request)"
        )
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
    if config.explore.enabled:
        explorer = Explorer(
            arbiter,
            safety,
            Scanner(robot, camera, observer, observation_wait_s=observation_wait_s),
            capture_policy,
            config=config.explore,
            on_scan=investigator.consider_scan if investigator is not None else None,
            notes=places,
        )
        block = explorer.drive_block()
        print(
            "[laika] exploring on her own: "
            + (f"head only, the wheels stay still ({block})" if block else "she drives")
            + (f"; when asked: head only ({asked})" if (asked := explorer.drive_block(requested=True)) else "; when asked: she drives")
        )
    # Whether she may start talking by herself (#39). The proactive speaking is #40's; it will call
    # interaction.decide() before generating anything. Reads the same presence and pending questions.
    interaction = InteractionPolicy(
        config.interaction,
        presence,
        events=events,
        store=store,
        threshold=config.curiosity.threshold,
        pending_questions=(lambda: len(investigator.questions)) if investigator is not None else None,
        body_busy=lambda: arbiter.holder is not None and arbiter.holder.priority <= Priority.CONVERSATION,
        personality=config.personality,
        battery_low=(lambda: battery.low) if battery is not None else None,
    )
    interaction.start()
    # After each turn, local rules keep what was said when the model did not (#35); in JSON mode too.
    # Only the agent shows pending questions to the user, so only then can a reply answer one.
    asking = investigator is not None and config.ai.use_tools
    conversation.selector = MemorySelector(
        world,
        memory,
        investigator.questions if asking else None,
        in_view=lambda: objects_in_view(observation_log),
        config=config.memory,
        rooms=room_tracker,
        windows=windows,
    )
    # find_object, find_person and approach (#43): memory, a head scan and, only where autonomous driving
    # is allowed, exploring and moving closer. Own scanner: the explorer's may be busy in the background.
    searcher = Searcher(
        robot,
        safety,
        arbiter,
        Scanner(robot, camera, observer, observation_wait_s=observation_wait_s),
        explorer,
        world,
        memory,
        presence,
        observer,
        config.search,
        observation_wait_s=observation_wait_s,
    )
    # What she does when nothing happens (#47): a deterministic selector (never the model) picks explore,
    # look outside, follow up, check on the house or rest. It replaces the old idle timers.
    idle_selector: IdleSelector | None = None
    if config.idle.enabled and config.idle.idle_after_s > 0:
        idle_selector = IdleSelector(
            config.idle,
            arbiter=arbiter,
            safety=safety,
            explorer=explorer,
            windows=window_watcher,
            searcher=searcher,
            questions=investigator.questions if investigator is not None else None,
            changes=observer.tracker,
            presence=presence,
            policy=interaction,
            world=world,
            memory=memory,
            personality=config.personality,
            robot=robot,
            events=events,
            search_max_s=config.search.default_max_s,
        )
        idle_selector.start()
        print(
            f"[laika] idle: after {config.idle.idle_after_s:g} s with nothing to do she chooses what to do "
            "(explore, look outside, follow up, check on the house, rest); "
            + ("on her own wheels" if config.robot.allow_autonomous_driving else "head only: autonomous driving is off")
        )
    web = WebLookup.from_config(config.web) if config.web.enabled else None
    if config.ai.use_tools:
        # The speak tool goes through the conversation, so Laika moves her head while talking.
        # Games to play together ("giochiamo?"): the model runs the word games, play_round the body ones.
        games = GameSession()
        # Bedtime stories: written by their own client, with a longer timeout than a conversation reply.
        story_client = ChatClient(
            base_url=config.ai.base_url,
            api_key=config.ai.api_key,
            model=config.ai.model,
            system_prompt="",
            timeout_s=max(config.ai.timeout_s, STORY_TIMEOUT_S),
            retries=0,  # a timed-out story is not written twice while the child waits
            temperature=config.ai.temperature,
            provider=config.ai.provider,
            azure_deployment=config.ai.azure_deployment,
            azure_api_version=config.ai.azure_api_version,
        )
        stories = BedtimeStories(story_client, StoryArchive(paths.stories), conversation.speak)
        # Learning new skills with the Copilot SDK: offered only when there is a clone to learn in.
        learner = None
        if config.learning.enabled:
            learner = SkillLearner(
                LearningSettings.from_config(config.learning), conversation.speak, data_dir=paths.learning
            )
            if not learner.available:
                print(f"[laika] learning: {learner.settings.repo_dir} is not a git clone, so no new skills")
                learner = None
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
            interaction=interaction,
            searcher=searcher,
            web=web,
            battery=battery,
            rooms=room_tracker,
            outside=window_watcher,
            windows=windows,
            games=games,
            stories=stories,
            learner=learner,
            plan=plan,
        )
        conversation.games = games
        conversation.stories = stories
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
                "set_do_not_disturb" in registry.names(),
                "find_object" in registry.names(),
                "approach" in registry.names(),
                web="wikipedia" in registry.names(),
                web_search="web_search" in registry.names(),
                personality=config.personality,
                autonomy=True,
                outside="look_outside" in registry.names(),
                games="start_game" in registry.names(),
                singing="sing" in registry.names(),
                bedtime="bedtime_story" in registry.names(),
                learning="learn_skill" in registry.names(),
            ),
            world=world,
            questions=investigator.questions if investigator is not None else None,
            memory=memory,
            battery=battery,
            outside=window_watcher,
            places=places,
            plan=plan,
            max_history_turns=config.ai.max_history_turns,
        )
        print(f"[laika] tools: {', '.join(registry.names())}")
    else:
        print("[laika] use_tools is off: the model replies with JSON actions")

    # Curiosity closes the loop (#40): a pending question is asked when the policy says so. Only with
    # the agent, which shows the question to the memory selector so the answer is kept.
    proactive: ProactiveInteraction | None = None
    encounters: Encounters | None = None
    questions_ask = asking and conversation.agent is not None and investigator is not None
    if questions_ask or config.encounter.enabled:
        if questions_ask:
            conversation.questions = investigator.questions
        waiting = investigator.questions if questions_ask else PendingQuestions()
        proactive = ProactiveInteraction(
            interaction, presence, waiting, conversation, arbiter, events, config.interaction
        )
        # Meeting someone while she explores (#49): she stops, turns to them and the policy decides.
        if config.encounter.enabled:
            encounters = Encounters(
                config.encounter,
                config.interaction,
                interaction,
                presence,
                arbiter,
                safety,
                conversation,
                waiting,
                events,
                robot,
                actions=actions,
                memory=memory,
                battery_low=(lambda: battery.low) if battery is not None else None,
            )
            encounters.start()
            proactive.encounters = encounters
        proactive.start()

    restore_signals = install_shutdown_handlers(lambda: safety.emergency_stop("shutdown signal"))
    try:
        notifier.ready()
        conversation.run()
    except KeyboardInterrupt:
        print("\n[laika] stopping")
    finally:
        notifier.stopping()
        if idle_selector is not None:
            idle_selector.close()
        if explorer is not None:
            explorer.stop()
        if investigator is not None:
            investigator.close()
        arbiter.close()
        if battery is not None:
            battery.close()
        safety.close()
        capture_policy.close()
        if proactive is not None:
            proactive.close()
        if encounters is not None:
            encounters.close()
        interaction.close()
        presence.close()
        actions.close()
        ear.close()
        robot.close()
        if tag_reader is not None:
            tag_reader.close()
        observer.close()
        image_store.close()
        windows.close()
        world_updater.close()
        world_persister.close()
        episode_recorder.close()
        places.close()
        events.close()
        archiver.close()
        store.close()
        event_file.close()
        restore_signals()
    return 0
