"""Load Laika's settings from a TOML file, with environment overrides for secrets."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from laika.ai_client import IMAGE_DETAILS

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "laika.toml"


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
    follow_up_s: float = 8.0


@dataclass
class RobotConfig:
    # Every safety knob is checked by `validate_robot`; docs/safety-calibration.md has the ranges.
    allow_driving: bool = True
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


# The Robot HAT's grayscale ADC is 12 bit.
GRAYSCALE_MAX = 4095


@dataclass
class StorageConfig:
    # Empty: $LAIKA_DATA_DIR, else $XDG_DATA_HOME/laika, else ~/.local/share/laika.
    data_dir: str = ""
    max_images: int = 200
    max_images_mb: float = 100.0
    log_max_mb: float = 5.0
    log_backups: int = 3
    # Off by default: transcripts hold what people said, including children.
    keep_transcripts: bool = False
    max_transcript_days: int = 7


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
class Config:
    ai: AIConfig = field(default_factory=AIConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)


def permission_warning(path: Path) -> str | None:
    """A warning when the settings file (which may hold the API key) is world-readable on POSIX."""
    if os.name != "posix":
        return None
    try:
        mode = path.stat().st_mode
    except OSError:
        return None
    if mode & 0o004:
        return f"{path} is readable by every user and may hold your API key; run: chmod 600 {path}"
    return None


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


def validate_robot(robot: RobotConfig) -> None:
    """Fail at startup, with the setting named, when a safety value is unsafe or meaningless."""
    _number("drive_speed", robot.drive_speed, 1, 40, integer=True)
    _number("max_drive_s", robot.max_drive_s, 0.1, 3)
    _number("backward_max_s", robot.backward_max_s, 0.1, 1)
    _number("min_front_distance_cm", robot.min_front_distance_cm, 10, 100)
    _number("command_max_age_s", robot.command_max_age_s, 0.2, 5)
    _number("safety_cadence_hz", robot.safety_cadence_hz, 10, 50)
    _number("ultrasonic_invalid_limit", robot.ultrasonic_invalid_limit, 1, 50, integer=True, off=True)
    _number("watchdog_timeout_s", robot.watchdog_timeout_s, 0.2, 2, off=True)
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


def load_config(path: Path | None = None) -> Config:
    config = Config()
    path = path or DEFAULT_CONFIG_PATH
    if path.exists():
        with path.open("rb") as file:
            data = tomllib.load(file)
        for name in ("ai", "voice", "robot", "storage", "camera", "vision", "capture"):
            _apply(getattr(config, name), data.get(name, {}), name)
        validate_robot(config.robot)
        validate_vision(config.vision)
        validate_capture(config.capture)

    # Environment variables win, so the key can stay out of the file.
    config.ai.api_key = (
        os.environ.get("LAIKA_AI_API_KEY") or os.environ.get("OPENAI_API_KEY") or config.ai.api_key
    )
    config.ai.base_url = os.environ.get("LAIKA_AI_BASE_URL", config.ai.base_url)
    config.ai.model = os.environ.get("LAIKA_AI_MODEL", config.ai.model)
    config.ai.provider = os.environ.get("LAIKA_AI_PROVIDER", config.ai.provider)
    return config
