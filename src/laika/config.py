"""Load Laika's settings from a TOML file, with environment overrides for secrets."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

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
    allow_driving: bool = True
    drive_speed: int = 25
    max_drive_s: float = 1.5
    min_front_distance_cm: float = 25.0
    honk_volume: int = 80
    # The safety monitor must tick at least this often or the watchdog stops the motors (0 = off).
    watchdog_timeout_s: float = 0.5
    # Small random head movements while Laika talks.
    talk_motion: bool = True


@dataclass
class Config:
    ai: AIConfig = field(default_factory=AIConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)


def _apply(section: Any, values: dict[str, Any], name: str) -> None:
    known = {f.name for f in fields(section)}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"Unknown setting [{name}] {key}")
        setattr(section, key, value)


def load_config(path: Path | None = None) -> Config:
    config = Config()
    path = path or DEFAULT_CONFIG_PATH
    if path.exists():
        with path.open("rb") as file:
            data = tomllib.load(file)
        for name in ("ai", "voice", "robot"):
            _apply(getattr(config, name), data.get(name, {}), name)

    # Environment variables win, so the key can stay out of the file.
    config.ai.api_key = (
        os.environ.get("LAIKA_AI_API_KEY") or os.environ.get("OPENAI_API_KEY") or config.ai.api_key
    )
    config.ai.base_url = os.environ.get("LAIKA_AI_BASE_URL", config.ai.base_url)
    config.ai.model = os.environ.get("LAIKA_AI_MODEL", config.ai.model)
    config.ai.provider = os.environ.get("LAIKA_AI_PROVIDER", config.ai.provider)
    return config
