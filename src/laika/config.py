"""Load Laika's settings from a TOML file, with environment overrides for secrets."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from laika.ai_client import IMAGE_DETAILS
from laika.personality import Personality

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "laika.toml"
# The keys live next to the settings in secrets.toml, which git ignores: laika.toml is committed.
SECRETS_NAME = "secrets.toml"
# The only settings secrets.toml may hold, so it can't quietly become a second settings file.
SECRET_SETTINGS = {"ai": ("api_key",), "web": ("brave_api_key",), "learning": ("github_token",)}


@dataclass
class AIConfig:
    # "openai", "azure" or "openai-compatible" (LM Studio, Ollama, vLLM; the key is optional).
    provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    timeout_s: float = 20.0
    max_history_turns: int = 8
    temperature: float = 0.8
    # false: ask for a JSON {"say", "actions"} reply instead of tool calls (models without tools).
    use_tools: bool = True
    # Azure OpenAI only: the deployment name stands in for the model.
    azure_deployment: str = ""
    azure_api_version: str = "2024-10-21"


@dataclass
class VoiceConfig:
    stt_language: str = "it"
    # "vosk": everything offline with Vosk's small model (the default: no audio leaves the robot).
    # "cloud" (opt-in): Vosk still waits for the wake word, then what is said goes to the transcription
    #          model below on [ai]'s endpoint (better Italian); Vosk's text when the cloud fails.
    stt_engine: str = "vosk"
    stt_model: str = "gpt-4o-transcribe"
    # "edge": Microsoft's online voices, falling back to Piper when offline.
    # "piper": the offline voice only.
    tts_engine: str = "edge"
    edge_voice: str = "it-IT-IsabellaNeural"
    # Piper voice, used by the "piper" engine and as Edge's offline fallback.
    tts_model: str = "it_IT-paola-medium"
    # Vosk's Italian model does not know the word "Laika", so it usually hears
    # something close to it. Any word similar to one of these wakes Laika up.
    wake_names: list[str] = field(default_factory=lambda: ["laika", "laica"])
    wake_greeting: str = "Dimmi."
    welcome: str = "Laika è online. Per svegliarmi, dite: ehi Laika."
    goodbye: str = "Va bene. Torno a fare la guardia."
    sleep_words: list[str] = field(
        default_factory=lambda: ["basta", "stop", "dormi", "fine", "a dopo"]
    )
    # Seconds to wait for a follow-up question before going back to sleep.
    follow_up_s: float = 15.0


@dataclass
class RobotConfig:
    # Every safety knob is checked by `validate_robot`; docs/safety-calibration.md has the ranges.
    allow_driving: bool = True
    # Autonomous driving (exploration) stays OFF until the V0.2 hardware check passes
    # (#15, #17, #18). `--sim` turns it on for the simulator only.
    allow_autonomous_driving: bool = False
    drive_speed: int = 25
    max_drive_s: float = 1.5
    # Backward moves are shorter: there is no rear sensor.
    backward_max_s: float = 0.6
    # Forward moves are refused or stopped when something is closer than this.
    min_front_distance_cm: float = 25.0
    # A move that waits longer than this before it starts is expired, never run.
    command_max_age_s: float = 1.0
    # How often the safety monitor reads the sensors while a move is active.
    safety_cadence_hz: float = 20.0
    # Consecutive invalid ultrasonic readings during a forward move that stop it (0 = off).
    ultrasonic_invalid_limit: int = 5
    honk_volume: int = 80
    # The safety monitor must tick at least this often or the watchdog stops the motors (0 = off).
    watchdog_timeout_s: float = 0.5
    # Grayscale values at or below these read as a cliff. None keeps the vendor's file value.
    cliff_reference: list[float] | None = None
    # Small random head movements while Laika talks.
    talk_motion: bool = True
    # The most she drives by herself (exploration, idle activities; never what you ask for) in any
    # hour and in a local day, in seconds of real driving time (0 = no limit). Over it, autonomous
    # moves are refused ("drive_budget"). docs/safety-calibration.md, "Driving budget".
    max_autonomous_drive_s_per_hour: float = 1200.0
    max_autonomous_drive_s_per_day: float = 7200.0


# The Robot HAT's grayscale ADC is 12 bit.
GRAYSCALE_MAX = 4095


@dataclass
class StorageConfig:
    # Empty: $LAIKA_DATA_DIR, else $XDG_DATA_HOME/laika, else ~/.local/share/laika.
    data_dir: str = ""
    max_images: int = 200
    max_images_mb: float = 100.0
    # Pictures older than this many days are deleted (0 = no age limit, only the count and size).
    max_image_days: float = 7.0
    log_max_mb: float = 5.0
    log_backups: int = 3
    # Off by default: transcripts hold what people said, including children.
    keep_transcripts: bool = False
    max_transcript_days: int = 7
    # The observation database (memory/laika.db): the newest rows within both limits are kept.
    max_observations: int = 5000
    max_observation_days: int = 90
    # Episodic memory (#34), in the same database: the newest rows within both limits are kept.
    max_episodes: int = 20000
    max_episode_days: int = 365
    # A "seen" episode is recorded at most this often per object and room (0: every sighting).
    seen_episode_interval_s: float = 3600.0


@dataclass
class LearningConfig:
    # Learning new skills with the GitHub Copilot SDK (docs/learning.md). When Laika is asked for something
    # no tool covers she offers to try to learn it; on a yes, a Copilot session edits her own repository.
    enabled: bool = True
    # The clone Copilot works in (and that the service runs from).
    repo_dir: str = "/home/ema/home-robot"
    # The Copilot model (empty: the Copilot default).
    model: str = "gpt-5"
    # A GitHub token with Copilot access (and contents write, for push and release). Keep this empty: the
    # key goes in config/secrets.toml ([learning] github_token) or in LAIKA_GITHUB_TOKEN, which wins.
    github_token: str = ""
    # Seconds a Copilot session may run, and the unit tests that check its work (30 to 7200).
    timeout_s: float = 1500.0
    test_timeout_s: float = 900.0
    # New skills started per day at most (1 to 20).
    max_per_day: int = 3
    # Seconds a "do you want me to try to learn this?" waits for the yes (30 to 3600).
    proposal_ttl_s: float = 600.0
    # After the tests pass: push the branch, and create a GitHub release for it.
    push: bool = True
    release: bool = True


MAX_IMAGE_SIZE = (1280, 960)


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    jpeg_quality: int = 80
    # A capture request sooner than this after the last one gets the previous frame back.
    min_interval_s: float = 0.5
    max_per_minute: int = 30
    save_images: bool = True

    def image_size(self) -> tuple[int, int]:
        """The configured size, clamped to something the Pi can handle."""
        return (
            max(64, min(int(self.width), MAX_IMAGE_SIZE[0])),
            max(48, min(int(self.height), MAX_IMAGE_SIZE[1])),
        )

    def quality(self) -> int:
        return max(10, min(int(self.jpeg_quality), 95))


@dataclass
class VisionConfig:
    enabled: bool = True
    # Empty: use [ai] model. Must be able to read images (e.g. gpt-4o-mini).
    # With provider "azure" this is the deployment name.
    model: str = ""
    # "low" is cheap and enough for 640x480; "high" costs several times more.
    detail: str = "low"
    # A JPEG larger than this (after the optional downscale) is not sent.
    max_image_kb: int = 400
    # With Pillow installed, frames are shrunk so their longest side fits this.
    max_side_px: int = 768
    jpeg_quality: int = 80
    timeout_s: float = 30.0
    # Cost guard, separate from the camera's rate: frames over it are skipped, not queued.
    max_per_minute: int = 10


@dataclass
class CaptureConfig:
    # Automatic photos only: the agent's capture_image tool is not limited by these.
    after_movement: bool = True
    after_movement_cooldown_s: float = 10.0
    on_person: bool = True
    person_cooldown_s: float = 30.0
    on_interesting: bool = True
    interesting_cooldown_s: float = 20.0
    # While exploring (#27): one photo every this many seconds.
    exploration_interval_s: float = 30.0
    # All automatic triggers together; each one is a vision request when [vision] is on.
    max_auto_per_minute: int = 6


@dataclass
class ExploreConfig:
    # false: Laika never explores, whatever the agent or the idle timer asks.
    enabled: bool = True
    # Wheel speed and length of one exploration step: slow and short, the safety layer clamps both.
    speed: int = 25
    step_ms: int = 800
    # Stop and look around (head only) after this many steps.
    scan_every_steps: int = 6
    # One exploration session ends after this long (a longer request is clamped to it) ...
    max_duration_s: float = 600.0
    # ... or after this many steps.
    max_steps: int = 1000
    # Deprecated: read as [idle] idle_after_s (the idle selector, #47, decides what to do when nothing
    # happens, and exploring is only one of the things it may choose).
    idle_after_s: float = 0.0


@dataclass
class PresenceConfig:
    # A frame counts as "a person is here" only when the observation's confidence is at least this.
    min_confidence: float = 0.5
    # PersonLost after this many seconds without a frame that shows a person. Longer than
    # presence_check_s, or she would think someone left between two photos.
    lost_after_s: float = 600.0
    # Idle pictures to notice someone arriving: every this many seconds while nobody is known to be
    # near and nobody is talking to her (0 = off). With [vision] on, each one is a vision request:
    # one every 5 minutes is at most 12 an hour.
    presence_interval_s: float = 300.0
    # While someone is near: how often to look again to notice they left (0 = off).
    presence_check_s: float = 300.0


@dataclass
class InteractionConfig:
    # false: Laika never starts a conversation on her own (the user can always talk to her).
    enabled: bool = True
    # No proactive interaction within this many seconds of any interaction (a chat or her own attempt).
    cooldown_s: float = 1800.0
    # A person must have been near this long before she may start (someone walking past is not ambushed).
    settle_s: float = 5.0
    # Nobody seen for this long: the user is away.
    away_after_s: float = 900.0
    # Local time she keeps quiet, "HH:MM-HH:MM" (may cross midnight); empty = no quiet hours.
    quiet_hours: str = "21:30-07:30"
    # Something interesting counts as worth saying for this many seconds.
    interest_window_s: float = 600.0
    # The most proactive attempts in any hour. Initiative (`[personality]`) picks 1 to 4 within this.
    max_per_hour: int = 2
    # Master switch for the questions she asks on her own (#40); false keeps the policy but never speaks first.
    proactive: bool = True
    # While someone is near she asks the policy again this often (people often arrive before there is
    # something to say).
    recheck_s: float = 15.0


# The nine inputs of the interestingness score (docs/curiosity.md); one weight each.
CURIOSITY_INPUTS = (
    "novelty",
    "environmental_change",
    "uncertainty",
    "person",
    "unknown_object",
    "unusual_location",
    "movement",
    "sound",
    "user_relevance",
)


@dataclass
class CuriosityConfig:
    # Weights of the inputs, 0 to 10 each. The total divides by their sum, so only the ratios matter.
    # A change in a familiar room (something new or moved) alone passes the threshold;
    # seeing a room for the first time (only novelty) does not.
    novelty: float = 0.15
    environmental_change: float = 0.40
    uncertainty: float = 0.03
    person: float = 0.15
    unknown_object: float = 0.10
    unusual_location: float = 0.10
    movement: float = 0.04
    # No microphone-level sensing yet: always 0, kept so the formula is complete.
    sound: float = 0.0
    user_relevance: float = 0.03
    # At or above this total (0 to 1) an observation is interesting enough to announce.
    threshold: float = 0.35
    # A room is familiar once Laika has known something in it for this long; only then is a
    # never-seen object there a change rather than part of discovering the room.
    familiar_room_after_s: float = 300.0
    # An object seen with less confidence than this counts as unknown.
    unknown_object_confidence: float = 0.5
    # The same reason is announced at most once per this many seconds (0 = no cooldown).
    cooldown_s: float = 120.0


@dataclass
class InvestigateConfig:
    # false: Laika never investigates on her own and the investigate tool is not offered.
    enabled: bool = True
    # Investigations she starts by herself, per hour (one asked for by the user is not counted).
    max_per_hour: int = 12
    # The same object is investigated at most once per this many seconds.
    object_cooldown_s: float = 600.0
    # She only moves closer while the front distance reading is above this (cm).
    approach_stop_cm: float = 50.0
    # Short forward moves while investigating one thing (0 = never move closer).
    max_approach_moves: int = 1
    # A question she has not been able to ask expires after this many hours.
    question_ttl_h: float = 24.0


@dataclass
class SearchConfig:
    # How long find_object and find_person look when the model gives no time, in seconds ...
    default_max_s: float = 60.0
    # ... and the most any request may ask for (a longer one is cut to it).
    max_max_s: float = 180.0
    # Short forward moves of one `approach` call (it also stops at approach_stop_cm).
    approach_moves: int = 3
    # `approach` moves closer only while the distance reading ahead is above this (cm).
    approach_stop_cm: float = 40.0


@dataclass
class MemoryConfig:
    # Local rules that keep what people tell her (what an object is, where it stays, who owns it,
    # a house rule) when the model did not call remember itself. false: only the model decides.
    auto_select: bool = True
    # A rule's guess is stored only at or above this confidence (0 to 1).
    min_confidence: float = 0.6


@dataclass
class WebConfig:
    # false: neither the wikipedia nor the web_search tool is offered and nothing leaves the robot.
    enabled: bool = True
    wikipedia_lang: str = "it"
    # Used when the first language has no result; empty = no fallback.
    fallback_lang: str = "en"
    # Brave Search API key; without one only Wikipedia is offered. $LAIKA_BRAVE_API_KEY wins.
    brave_api_key: str = ""
    # "strict" or "moderate".
    safesearch: str = "strict"
    max_results: int = 5
    timeout_s: float = 8.0
    # Lookups sent in any hour (cache hits are free).
    max_per_hour: int = 30


@dataclass
class BatteryConfig:
    # Know the battery level and rest when it is low (docs/safety-calibration.md, "Battery").
    # false: no monitor at all; Laika neither reports the battery nor limits her driving for it.
    enabled: bool = True
    # How often the voltage is read, in seconds (5 to 3600). Slow on purpose: the battery changes slowly.
    sample_s: float = 30.0
    # Volts for the pack of two 18650 cells in series (8.4 V full, 7.4 V nominal). CALIBRATE THEM on the
    # robot against a multimeter: below low_v she stops driving around by herself, below critical_v
    # she stays still, and she is only OK again above ok_v (the gap stops flapping).
    low_v: float = 7.0
    ok_v: float = 7.3
    critical_v: float = 6.6
    # Added to every reading so it matches the multimeter (the hardware check suggests it), in volts.
    voltage_offset: float = 0.0


@dataclass
class HouseConfig:
    # Rooms she never drives into by herself, besides the house rules people tell her ("non entrare in
    # camera"). Italian words (cucina, camera da letto, ...) or the names the world model uses
    # (kitchen, bedroom, living_room, office, bathroom, entrance). docs/behaviours.md, "House rules".
    no_go_rooms: list[str] = field(default_factory=list)


@dataclass
class RoomsConfig:
    # Which room she is in, from what she sees (docs/behaviours.md, "Room recognition"). The vision model
    # guesses the room in every frame; the RoomTracker votes over the recent guesses before changing the room.
    # false: no tracker, the room only comes from the simulator or from being told.
    enabled: bool = True
    # Only guesses newer than this many seconds count (10 to 3600).
    window_s: float = 120.0
    # A different room needs at least this many recent votes before she believes she moved (1 to 5).
    min_votes: int = 2
    # ...and its votes must outweigh the current room's and every other room's by this much (0 to 3).
    switch_margin: float = 0.4
    # A frame votes only when its evidence for a room (the model's confidence plus landmarks) reaches this (0.1 to 1).
    min_confidence: float = 0.5
    # In --sim the ground truth stays in charge; the tracker runs beside it and logs how often it agrees.
    shadow_in_sim: bool = True


@dataclass
class OutsideConfig:
    # Look out of the windows (docs/behaviours.md, "Looking outside"). Each look is a photo of the outside
    # sent to the vision model, so it costs like any vision request (docs/storage-and-retention.md).
    # false: she never looks out of the windows and the look_outside tool is not offered.
    enabled: bool = True
    # She looks out of the same window at most this often (60 to 86400 seconds) ...
    min_interval_s: float = 1800.0
    # ... and at most this many times in a local day, all windows together (1 to 200).
    max_per_day: int = 24
    # How interesting a change outside is (started raining, it got dark, a busier street), 0 to 1. It goes
    # through the same gate as every other interesting thing, so it only reaches the interaction policy when
    # it is at or above [curiosity] threshold. 0 = a change outside never counts as interesting.
    change_interest: float = 0.6


@dataclass
class EncounterConfig:
    # Meeting people while she is out and about (docs/behaviours.md, "Meeting people"). A person appearing
    # while she explores stops her and turns her head; whether she then speaks is still the interaction
    # policy's call ([interaction]: cooldown, quiet hours, do-not-disturb, hourly limit, settle time).
    # false: she carries on as if nobody came (a person who arrives while she is idle is handled as before).
    enabled: bool = True
    # With nothing else to say (no question, nothing noticed) a short hello counts as something to say,
    # for an encounter only. false: she only speaks to someone she meets when she has a real reason.
    greet_when_nothing_to_say: bool = True
    # Come a little closer before talking when the person is ahead and farther than approach_distance_cm.
    # One short move behind [robot] allow_autonomous_driving (and the pause, budget and no-go rooms).
    approach: bool = True
    approach_distance_cm: float = 120.0
    # Never closer than this to a person (the ultrasonic reading ahead).
    stop_cm: float = 80.0
    # When the policy says no she may still turn to the person and nod (silent), except during quiet
    # hours, do-not-disturb or when busy, where she does nothing at all.
    gesture_when_quiet: bool = True


@dataclass
class IdleConfig:
    # What Laika does when nothing happens (docs/behaviours.md, "When Laika is idle"). A deterministic selector
    # picks one activity by utility; the model never chooses. Everything stays behind the usual rules: quiet
    # hours, do not disturb, the pause, the battery and [robot] allow_autonomous_driving (without it she only
    # looks around with the head).
    enabled: bool = True
    # Start choosing after this many seconds with nobody talking to her and no behaviour holding the body
    # (5 to 86400; 0 = never on her own).
    idle_after_s: float = 300.0
    # Wait this long after an activity before choosing the next one.
    between_activities_s: float = 60.0
    # One activity ends after this long at the most.
    max_activity_s: float = 600.0
    # The same activity is never chosen more than this many times in a row (unless nothing else is possible).
    max_repeats: int = 2
    # Seeds the tie-breaking generator, so a run can be repeated (0 = seeded from the time).
    seed: int = 0
    # Each activity can be turned off, and has a base weight (0 to 10) that the formula multiplies.
    explore: bool = True
    explore_weight: float = 1.0
    look_outside: bool = True
    look_outside_weight: float = 0.6
    follow_up: bool = True
    follow_up_weight: float = 0.8
    check_home: bool = True
    check_home_weight: float = 0.4
    rest: bool = True
    rest_weight: float = 0.5


IDLE_ACTIVITIES = ("explore", "look_outside", "follow_up", "check_home", "rest")


@dataclass
class Config:
    ai: AIConfig = field(default_factory=AIConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    explore: ExploreConfig = field(default_factory=ExploreConfig)
    curiosity: CuriosityConfig = field(default_factory=CuriosityConfig)
    investigate: InvestigateConfig = field(default_factory=InvestigateConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    presence: PresenceConfig = field(default_factory=PresenceConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    web: WebConfig = field(default_factory=WebConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    house: HouseConfig = field(default_factory=HouseConfig)
    rooms: RoomsConfig = field(default_factory=RoomsConfig)
    outside: OutsideConfig = field(default_factory=OutsideConfig)
    encounter: EncounterConfig = field(default_factory=EncounterConfig)
    idle: IdleConfig = field(default_factory=IdleConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    personality: Personality = field(default_factory=Personality)


def secrets_path(config_path: Path) -> Path:
    """Where the keys for `config_path` live: secrets.toml in the same folder."""
    return config_path.parent / SECRETS_NAME


def permission_warning(path: Path) -> str | None:
    """A warning when the secrets file next to `path` is world-readable on POSIX."""
    if os.name != "posix":
        return None
    secrets = secrets_path(path)
    try:
        mode = secrets.stat().st_mode
    except OSError:
        return None
    if mode & 0o004:
        return f"{secrets} is readable by every user and holds your keys; run: chmod 600 {secrets}"
    return None


def committed_key_warning(path: Path) -> str | None:
    """A warning when a key was put in the settings file, which is committed to git (a public repo)."""
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    found = [
        f"[{section}] {key}"
        for section, keys in SECRET_SETTINGS.items()
        for key in keys
        if str(data.get(section, {}).get(key, "") or "").strip()
    ]
    if not found:
        return None
    return (
        f"{path} holds {', '.join(found)} but is committed to git: move the key to "
        f"{secrets_path(path)} (ignored by git) or an environment variable, and empty it here"
    )


def _load_secrets(config: "Config", path: Path) -> None:
    """Read the keys from secrets.toml; anything else there is an error."""
    if not path.exists():
        return
    with path.open("rb") as file:
        data = tomllib.load(file)
    for section, values in data.items():
        allowed = SECRET_SETTINGS.get(section)
        if allowed is None or not isinstance(values, dict):
            raise ValueError(f"{path.name} may only hold {_secret_list()}, not [{section}]")
        for key, value in values.items():
            if key not in allowed:
                raise ValueError(f"{path.name} may only hold {_secret_list()}, not [{section}] {key}")
            if not isinstance(value, str):
                raise ValueError(f"[{section}] {key} in {path.name} must be a string")
            if value.strip():
                setattr(getattr(config, section), key, value.strip())


def _secret_list() -> str:
    return ", ".join(f"[{section}] {key}" for section, keys in SECRET_SETTINGS.items() for key in keys)


def _apply(section: Any, values: dict[str, Any], name: str) -> None:
    known = {f.name for f in fields(section)}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"Unknown setting [{name}] {key}")
        setattr(section, key, value)


def _number(
    name: str,
    value: Any,
    low: float,
    high: float,
    *,
    integer: bool = False,
    off: bool = False,
    section: str = "robot",
) -> None:
    """Reject a [robot] value outside its safe range. `off` also accepts 0, which disables the feature."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"[{section}] {name} must be a number, got {value!r}")
    if integer and not isinstance(value, int):
        raise ValueError(f"[{section}] {name} must be a whole number, got {value!r}")
    if off and value == 0:
        return
    if not low <= value <= high:
        allowed = f"{low:g} to {high:g}" + (" (or 0 to turn it off)" if off else "")
        raise ValueError(f"[{section}] {name} must be {allowed}, got {value:g}")


def validate_storage(storage: StorageConfig) -> None:
    """Fail at startup, with the setting named, when a retention limit is unusable."""
    _number("max_image_days", storage.max_image_days, 0, 3650, section="storage")
    _number("max_observations", storage.max_observations, 10, 1_000_000, integer=True, section="storage")
    _number("max_observation_days", storage.max_observation_days, 1, 3650, integer=True, section="storage")
    _number("max_episodes", storage.max_episodes, 100, 1_000_000, integer=True, section="storage")
    _number("max_episode_days", storage.max_episode_days, 1, 3650, integer=True, section="storage")
    _number("seen_episode_interval_s", storage.seen_episode_interval_s, 0, 86400, section="storage")


def validate_robot(robot: RobotConfig) -> None:
    """Fail at startup, with the setting named, when a safety value is unsafe or meaningless."""
    if not isinstance(robot.allow_autonomous_driving, bool):
        raise ValueError("[robot] allow_autonomous_driving must be true or false")
    _number("drive_speed", robot.drive_speed, 1, 40, integer=True)
    _number("max_drive_s", robot.max_drive_s, 0.1, 3)
    _number("backward_max_s", robot.backward_max_s, 0.1, 1)
    _number("min_front_distance_cm", robot.min_front_distance_cm, 10, 100)
    _number("command_max_age_s", robot.command_max_age_s, 0.2, 5)
    _number("safety_cadence_hz", robot.safety_cadence_hz, 10, 50)
    _number("ultrasonic_invalid_limit", robot.ultrasonic_invalid_limit, 1, 50, integer=True, off=True)
    _number("watchdog_timeout_s", robot.watchdog_timeout_s, 0.2, 2, off=True)
    _number("max_autonomous_drive_s_per_hour", robot.max_autonomous_drive_s_per_hour, 10, 3600, off=True)
    _number("max_autonomous_drive_s_per_day", robot.max_autonomous_drive_s_per_day, 10, 86400, off=True)
    hour, day = robot.max_autonomous_drive_s_per_hour, robot.max_autonomous_drive_s_per_day
    if hour and day and hour > day:
        raise ValueError(
            "[robot] max_autonomous_drive_s_per_hour must not be more than max_autonomous_drive_s_per_day "
            f"(got {hour:g} and {day:g})"
        )
    if robot.cliff_reference is not None:
        values = robot.cliff_reference
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError("[robot] cliff_reference must be a list of three numbers, e.g. [400, 400, 400]")
        for value in values:
            _number("cliff_reference", value, 0, GRAYSCALE_MAX)


def validate_vision(vision: VisionConfig) -> None:
    """Fail at startup, with the setting named, when a vision value is unusable."""
    if not isinstance(vision.enabled, bool):
        raise ValueError("[vision] enabled must be true or false")
    if not isinstance(vision.model, str):
        raise ValueError("[vision] model must be a string")
    if vision.detail not in IMAGE_DETAILS:
        raise ValueError(f"[vision] detail must be one of: {', '.join(IMAGE_DETAILS)}")
    _number("max_image_kb", vision.max_image_kb, 10, 4096, integer=True, section="vision")
    _number("max_side_px", vision.max_side_px, 64, 2048, integer=True, section="vision")
    _number("jpeg_quality", vision.jpeg_quality, 10, 95, integer=True, section="vision")
    _number("timeout_s", vision.timeout_s, 1, 120, section="vision")
    _number("max_per_minute", vision.max_per_minute, 1, 120, integer=True, section="vision")


def validate_capture(capture: CaptureConfig) -> None:
    """Fail at startup, with the setting named, when an automatic capture value is unusable."""
    for name in ("after_movement", "on_person", "on_interesting"):
        if not isinstance(getattr(capture, name), bool):
            raise ValueError(f"[capture] {name} must be true or false")
    for name in ("after_movement_cooldown_s", "person_cooldown_s", "interesting_cooldown_s"):
        _number(name, getattr(capture, name), 0, 600, section="capture")
    _number("exploration_interval_s", capture.exploration_interval_s, 5, 300, section="capture")
    _number("max_auto_per_minute", capture.max_auto_per_minute, 1, 60, integer=True, section="capture")


def validate_explore(explore: ExploreConfig) -> None:
    """Fail at startup, with the setting named, when an exploration value is unusable."""
    if not isinstance(explore.enabled, bool):
        raise ValueError("[explore] enabled must be true or false")
    _number("speed", explore.speed, 1, 40, integer=True, section="explore")
    _number("step_ms", explore.step_ms, 100, 1500, integer=True, section="explore")
    _number("scan_every_steps", explore.scan_every_steps, 1, 100, integer=True, section="explore")
    _number("max_duration_s", explore.max_duration_s, 10, 3600, section="explore")
    _number("max_steps", explore.max_steps, 1, 10000, integer=True, section="explore")
    _number("idle_after_s", explore.idle_after_s, 30, 86400, off=True, section="explore")


def validate_curiosity(curiosity: CuriosityConfig) -> None:
    """Fail at startup, with the setting named, when an interestingness value is unusable."""
    for name in CURIOSITY_INPUTS:
        _number(name, getattr(curiosity, name), 0, 10, section="curiosity")
    if sum(getattr(curiosity, name) for name in CURIOSITY_INPUTS) <= 0:
        raise ValueError("[curiosity] at least one input weight must be above 0")
    _number("threshold", curiosity.threshold, 0, 1, section="curiosity")
    _number("unknown_object_confidence", curiosity.unknown_object_confidence, 0, 1, section="curiosity")
    _number("cooldown_s", curiosity.cooldown_s, 0, 86400, section="curiosity")
    _number("familiar_room_after_s", curiosity.familiar_room_after_s, 0, 86400, section="curiosity")


def validate_investigate(investigate: InvestigateConfig) -> None:
    """Fail at startup, with the setting named, when an investigation value is unusable."""
    if not isinstance(investigate.enabled, bool):
        raise ValueError("[investigate] enabled must be true or false")
    _number("max_per_hour", investigate.max_per_hour, 1, 120, integer=True, section="investigate")
    _number("object_cooldown_s", investigate.object_cooldown_s, 0, 86400, section="investigate")
    _number("approach_stop_cm", investigate.approach_stop_cm, 20, 300, section="investigate")
    _number("max_approach_moves", investigate.max_approach_moves, 0, 2, integer=True, section="investigate")
    _number("question_ttl_h", investigate.question_ttl_h, 1, 168, section="investigate")


def validate_search(search: SearchConfig) -> None:
    """Fail at startup, with the setting named, when a search value is unusable."""
    _number("default_max_s", search.default_max_s, 5, 600, section="search")
    _number("max_max_s", search.max_max_s, 10, 600, section="search")
    _number("approach_moves", search.approach_moves, 1, 5, integer=True, section="search")
    _number("approach_stop_cm", search.approach_stop_cm, 20, 300, section="search")
    if search.default_max_s > search.max_max_s:
        raise ValueError(
            f"[search] default_max_s must not be longer than max_max_s (got {search.default_max_s:g} and {search.max_max_s:g})"
        )


def validate_web(web: WebConfig) -> None:
    """Fail at startup, with the setting named, when a web lookup value is unusable."""
    if not isinstance(web.enabled, bool):
        raise ValueError("[web] enabled must be true or false")
    for name in ("wikipedia_lang", "fallback_lang"):
        value = getattr(web, name)
        if not isinstance(value, str) or (value and not re.fullmatch(r"[a-z]{2,3}(-[a-z]{2,8})?", value)):
            raise ValueError(f"[web] {name} must be a Wikipedia language code such as it or en, got {value!r}")
    if not web.wikipedia_lang:
        raise ValueError("[web] wikipedia_lang must not be empty")
    if not isinstance(web.brave_api_key, str):
        raise ValueError("[web] brave_api_key must be a string")
    if web.safesearch not in ("strict", "moderate"):
        raise ValueError('[web] safesearch must be "strict" or "moderate"')
    _number("max_results", web.max_results, 1, 10, integer=True, section="web")
    _number("timeout_s", web.timeout_s, 1, 60, section="web")
    _number("max_per_hour", web.max_per_hour, 1, 600, integer=True, section="web")


def validate_battery(battery: BatteryConfig) -> None:
    """Fail at startup, with the setting named, when a battery value is unusable."""
    if not isinstance(battery.enabled, bool):
        raise ValueError("[battery] enabled must be true or false")
    _number("sample_s", battery.sample_s, 5, 3600, section="battery")
    for name in ("critical_v", "low_v", "ok_v"):
        _number(name, getattr(battery, name), 5, 9, section="battery")
    _number("voltage_offset", battery.voltage_offset, -2, 2, section="battery")
    if not battery.critical_v < battery.low_v < battery.ok_v:
        raise ValueError(
            "[battery] the thresholds must satisfy critical_v < low_v < ok_v "
            f"(got {battery.critical_v:g}, {battery.low_v:g} and {battery.ok_v:g})"
        )


MAX_NO_GO_ROOMS = 20


def validate_house(house: HouseConfig) -> None:
    """Fail at startup, with the setting named, when a house setting is unusable."""
    rooms = house.no_go_rooms
    if not isinstance(rooms, list) or len(rooms) > MAX_NO_GO_ROOMS:
        raise ValueError(f"[house] no_go_rooms must be a list of at most {MAX_NO_GO_ROOMS} room names")
    for room in rooms:
        if not isinstance(room, str) or not room.strip() or len(room) > 60:
            raise ValueError(f"[house] no_go_rooms must hold room names (text up to 60 characters), got {room!r}")


def validate_rooms(rooms: RoomsConfig) -> None:
    """Fail at startup, with the setting named, when a room recognition value is unusable."""
    for name in ("enabled", "shadow_in_sim"):
        if not isinstance(getattr(rooms, name), bool):
            raise ValueError(f"[rooms] {name} must be true or false")
    _number("window_s", rooms.window_s, 10, 3600, section="rooms")
    _number("min_votes", rooms.min_votes, 1, 5, integer=True, section="rooms")
    _number("switch_margin", rooms.switch_margin, 0, 3, section="rooms")
    _number("min_confidence", rooms.min_confidence, 0.1, 1, section="rooms")


def validate_outside(outside: OutsideConfig) -> None:
    """Fail at startup, with the setting named, when a window-watching value is unusable."""
    if not isinstance(outside.enabled, bool):
        raise ValueError("[outside] enabled must be true or false")
    _number("min_interval_s", outside.min_interval_s, 60, 86400, section="outside")
    _number("max_per_day", outside.max_per_day, 1, 200, integer=True, section="outside")
    _number("change_interest", outside.change_interest, 0, 1, section="outside")


def validate_idle(idle: IdleConfig) -> None:
    """Fail at startup, with the setting named, when an idle-selector value is unusable."""
    for name in ("enabled", *IDLE_ACTIVITIES):
        if not isinstance(getattr(idle, name), bool):
            raise ValueError(f"[idle] {name} must be true or false")
    _number("idle_after_s", idle.idle_after_s, 5, 86400, off=True, section="idle")
    _number("between_activities_s", idle.between_activities_s, 0, 86400, section="idle")
    _number("max_activity_s", idle.max_activity_s, 10, 3600, section="idle")
    _number("max_repeats", idle.max_repeats, 1, 20, integer=True, section="idle")
    _number("seed", idle.seed, 0, 2**31, integer=True, section="idle")
    for name in IDLE_ACTIVITIES:
        _number(f"{name}_weight", getattr(idle, f"{name}_weight"), 0, 10, section="idle")


# One approach move covers about this much (speed 20 for 0.8 s); stop_cm keeps room for it.
APPROACH_MOVE_CM = 12.0


def validate_encounter(encounter: EncounterConfig) -> None:
    """Fail at startup, with the setting named, when an encounter value is unusable."""
    for name in ("enabled", "greet_when_nothing_to_say", "approach", "gesture_when_quiet"):
        if not isinstance(getattr(encounter, name), bool):
            raise ValueError(f"[encounter] {name} must be true or false")
    _number("stop_cm", encounter.stop_cm, 40, 300, section="encounter")
    _number("approach_distance_cm", encounter.approach_distance_cm, 60, 500, section="encounter")
    if encounter.approach_distance_cm < encounter.stop_cm + APPROACH_MOVE_CM:
        raise ValueError(
            "[encounter] approach_distance_cm must be at least stop_cm + "
            f"{APPROACH_MOVE_CM:g} (one move), got {encounter.approach_distance_cm:g} and {encounter.stop_cm:g}"
        )


def validate_memory(memory: MemoryConfig) -> None:
    """Fail at startup, with the setting named, when a conversation-memory value is unusable."""
    if not isinstance(memory.auto_select, bool):
        raise ValueError("[memory] auto_select must be true or false")
    _number("min_confidence", memory.min_confidence, 0, 1, section="memory")


def validate_presence(presence: PresenceConfig) -> None:
    """Fail at startup, with the setting named, when a presence value is unusable."""
    _number("min_confidence", presence.min_confidence, 0, 1, section="presence")
    _number("lost_after_s", presence.lost_after_s, 5, 3600, section="presence")
    _number("presence_interval_s", presence.presence_interval_s, 5, 3600, off=True, section="presence")
    _number("presence_check_s", presence.presence_check_s, 5, 3600, off=True, section="presence")
    if presence.presence_check_s and presence.lost_after_s <= presence.presence_check_s:
        raise ValueError(
            "[presence] lost_after_s must be longer than presence_check_s, or Laika would think "
            f"someone left between two photos (got {presence.lost_after_s} and {presence.presence_check_s})"
        )


QUIET_HOURS_FORMAT = re.compile(r"^(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})$")


def parse_quiet_hours(text: str) -> tuple[int, int] | None:
    """"21:30-07:30" as (start, end) minutes since midnight; None for no quiet hours. Raises ValueError."""
    if not isinstance(text, str):
        raise ValueError(f"[interaction] quiet_hours must be text like 21:30-07:30, got {text!r}")
    text = text.strip()
    if not text:
        return None
    match = QUIET_HOURS_FORMAT.match(text)
    if match is None:
        raise ValueError(f"[interaction] quiet_hours must look like 21:30-07:30 (or be empty), got {text!r}")
    h1, m1, h2, m2 = (int(part) for part in match.groups())
    if h1 > 23 or h2 > 23 or m1 > 59 or m2 > 59:
        raise ValueError(f"[interaction] quiet_hours has a time that does not exist: {text!r}")
    start, end = h1 * 60 + m1, h2 * 60 + m2
    if start == end:
        raise ValueError("[interaction] quiet_hours start and end must differ (leave it empty for none)")
    return start, end


def validate_interaction(interaction: InteractionConfig) -> None:
    """Fail at startup, with the setting named, when an interaction-policy value is unusable."""
    if not isinstance(interaction.enabled, bool):
        raise ValueError("[interaction] enabled must be true or false")
    _number("cooldown_s", interaction.cooldown_s, 0, 86400, section="interaction")
    _number("settle_s", interaction.settle_s, 0, 600, section="interaction")
    _number("away_after_s", interaction.away_after_s, 10, 86400, section="interaction")
    _number("interest_window_s", interaction.interest_window_s, 10, 86400, section="interaction")
    _number("max_per_hour", interaction.max_per_hour, 1, 60, integer=True, section="interaction")
    _number("recheck_s", interaction.recheck_s, 1, 3600, section="interaction")
    if not isinstance(interaction.proactive, bool):
        raise ValueError("[interaction] proactive must be true or false")
    parse_quiet_hours(interaction.quiet_hours)


def _read_deprecated_idle_after(config: Config, data: dict[str, Any]) -> None:
    """`[explore] idle_after_s` (before #47) is read as `[idle] idle_after_s`, with a note; [idle] wins."""
    if "idle_after_s" not in data.get("explore", {}):
        return
    if "idle_after_s" in data.get("idle", {}):
        print("[laika] settings: [explore] idle_after_s is deprecated and ignored: [idle] idle_after_s is set")
        return
    config.idle.idle_after_s = float(config.explore.idle_after_s)
    validate_idle(config.idle)
    print(
        "[laika] settings: [explore] idle_after_s is deprecated, read as [idle] idle_after_s = "
        f"{config.explore.idle_after_s:g} (the idle selector now chooses what to do, see docs/behaviours.md)"
    )


def validate_learning(learning: LearningConfig) -> None:
    for name in ("enabled", "push", "release"):
        if not isinstance(getattr(learning, name), bool):
            raise ValueError(f"[learning] {name} must be true or false")
    for name in ("repo_dir", "model"):
        if not isinstance(getattr(learning, name), str):
            raise ValueError(f"[learning] {name} must be text")
    _number("timeout_s", learning.timeout_s, 30, 7200, section="learning")
    _number("test_timeout_s", learning.test_timeout_s, 30, 7200, section="learning")
    _number("max_per_day", learning.max_per_day, 1, 20, integer=True, section="learning")
    _number("proposal_ttl_s", learning.proposal_ttl_s, 30, 3600, section="learning")


def validate_voice(voice: VoiceConfig) -> None:
    if voice.stt_engine not in ("cloud", "vosk"):
        raise ValueError('[voice] stt_engine must be "cloud" or "vosk"')
    if not isinstance(voice.stt_model, str) or not voice.stt_model.strip():
        raise ValueError("[voice] stt_model must be a model name, for example gpt-4o-transcribe")


def load_config(path: Path | None = None) -> Config:
    config = Config()
    path = path or DEFAULT_CONFIG_PATH
    if path.exists():
        with path.open("rb") as file:
            data = tomllib.load(file)
        for name in ("ai", "voice", "robot", "storage", "camera", "vision", "capture", "explore", "curiosity", "investigate", "memory", "presence", "search", "interaction", "web", "battery", "house", "rooms", "outside", "encounter", "idle", "learning"):
            _apply(getattr(config, name), data.get(name, {}), name)
        validate_voice(config.voice)
        validate_storage(config.storage)
        validate_robot(config.robot)
        validate_vision(config.vision)
        validate_capture(config.capture)
        validate_explore(config.explore)
        validate_curiosity(config.curiosity)
        validate_investigate(config.investigate)
        validate_memory(config.memory)
        validate_presence(config.presence)
        validate_search(config.search)
        validate_interaction(config.interaction)
        validate_web(config.web)
        validate_battery(config.battery)
        validate_house(config.house)
        validate_rooms(config.rooms)
        validate_outside(config.outside)
        validate_encounter(config.encounter)
        validate_idle(config.idle)
        validate_learning(config.learning)
        _read_deprecated_idle_after(config, data)
        config.personality = Personality.from_settings(data.get("personality", {}))

    # Keys: environment variables win over secrets.toml, which wins over the settings file.
    _load_secrets(config, secrets_path(path))
    config.ai.api_key = (
        os.environ.get("LAIKA_AI_API_KEY") or os.environ.get("OPENAI_API_KEY") or config.ai.api_key
    )
    config.ai.base_url = os.environ.get("LAIKA_AI_BASE_URL", config.ai.base_url)
    config.ai.model = os.environ.get("LAIKA_AI_MODEL", config.ai.model)
    config.web.brave_api_key = os.environ.get("LAIKA_BRAVE_API_KEY") or config.web.brave_api_key
    config.ai.provider = os.environ.get("LAIKA_AI_PROVIDER", config.ai.provider)
    config.learning.github_token = os.environ.get("LAIKA_GITHUB_TOKEN") or config.learning.github_token
    return config
