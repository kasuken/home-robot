"""Laika's personality: six parameters from 0 to 100, set in `[personality]` (#41).

The parameters shape two things, and only these:

- the prompts: humour, sarcasm, verbosity and curiosity become short Italian sentences, picked
  from three bands (0-30, 31-69, 70-100). The core character stays fixed in the prompt text.
- the behaviour of her own initiative: initiative and interruption tolerance scale the interaction
  policy (cooldown, hourly limit, how long someone must be near), and curiosity shifts the
  interestingness threshold and the investigator's hourly limit.

Safety, do-not-disturb, busy and quiet hours are never affected by any parameter. Every mapping
here is a pure function with hard bounds, documented in docs/behaviours.md ("Personality").
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

MIN_VALUE = 0
MAX_VALUE = 100
# The cooldown never drops below ten minutes (or the configured cooldown when that is shorter).
MIN_COOLDOWN_S = 600.0
# Initiative picks 1 to this many proactive attempts per hour, within [interaction] max_per_hour.
MAX_PROACTIVE_PER_HOUR = 4
# Below this interruption tolerance a person must have been near for at least `LOW_TOLERANCE_PRESENT_S`.
LOW_TOLERANCE = 20
LOW_TOLERANCE_PRESENT_S = 30.0
# The curiosity-shifted threshold stays between these, whatever the base.
THRESHOLD_BOUNDS = (0.2, 0.6)


@dataclass(frozen=True)
class Personality:
    humor: int = 70
    sarcasm: int = 70  # the PRD's example has 45; the owner raised it on 15 October 2026
    verbosity: int = 20
    initiative: int = 60
    curiosity: int = 75
    interruption_tolerance: int = 25

    @classmethod
    def from_settings(cls, values: Mapping[str, Any]) -> Personality:
        """Build from the `[personality]` table; every value must be a whole number from 0 to 100."""
        if not isinstance(values, Mapping):
            raise ValueError("[personality] must be a table of 0 to 100 values")
        known = {f.name for f in fields(cls)}
        checked: dict[str, int] = {}
        for key, value in values.items():
            if key not in known:
                raise ValueError(f"Unknown setting [personality] {key}")
            checked[key] = _whole_number(key, value)
        return cls(**checked)

    def describe(self) -> str:
        """One line for the startup log, e.g. "humor 70, sarcasm 70, ..."."""
        return ", ".join(f"{f.name.replace('_', ' ')} {getattr(self, f.name)}" for f in fields(self))


def _whole_number(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"[personality] {name} must be a whole number from 0 to 100, got {value!r}")
    if not MIN_VALUE <= value <= MAX_VALUE:
        raise ValueError(f"[personality] {name} must be 0 to 100, got {value}")
    return value


def band(value: int) -> int:
    """0 for 0 to 30, 1 for 31 to 69, 2 for 70 to 100."""
    return 0 if value <= 30 else (1 if value <= 69 else 2)


# --- the prompt sentences (Italian, one per band) ----------------------------------

HUMOR_LINES = (
    "Non fai battute.",
    "Ogni tanto fai una battuta asciutta.",
    "Fai spesso battute asciutte, con umorismo secco.",
)
SARCASM_LINES = (
    "Non sei mai sarcastica.",
    "Ogni tanto un tocco di ironia leggera.",
    "Sei ironica, in stile TARS.",
)
# Said whatever the sarcasm value: the child-safety rule is not a setting.
SARCASM_ALWAYS = "Sei sempre gentile e mai cattiva, soprattutto con i bambini: non prendi in giro nessuno."
VERBOSITY_LINES = (
    "Rispondi con una o due frasi brevi.",
    "Rispondi con al massimo tre frasi.",
    "Puoi spiegare un po' di più, fino a cinque frasi.",
)
CURIOSITY_LINES = (
    "Fai domande di approfondimento solo quando serve davvero.",
    "Ogni tanto fai una domanda di approfondimento.",
    "Fai spesso una domanda di approfondimento, quando ti interessa davvero.",
)


def persona(p: Personality) -> str:
    """The personality paragraph of the system prompts. The core character is fixed here."""
    return (
        "Personalità: ispirata a TARS di Interstellar, ma con un carattere tutto suo. Sei concisa, "
        f"diretta e curiosa. {HUMOR_LINES[band(p.humor)]} {SARCASM_LINES[band(p.sarcasm)]} "
        f"{SARCASM_ALWAYS} {VERBOSITY_LINES[band(p.verbosity)]} {CURIOSITY_LINES[band(p.curiosity)]} "
        "Spesso parli con un bambino: usa parole semplici, sii incoraggiante, niente argomenti "
        "spaventosi o inadatti. Se non sai una cosa, dillo."
    )


def spoken_style(p: Personality) -> str:
    """The style sentences for a question Laika starts herself (no curiosity line: the question is given)."""
    return (
        f"{HUMOR_LINES[band(p.humor)]} {SARCASM_LINES[band(p.sarcasm)]} {SARCASM_ALWAYS} "
        f"{VERBOSITY_LINES[band(p.verbosity)]} Rispondi in italiano."
    )


# --- the mappings used by the interaction policy and curiosity ---------------------


def _scale(value: int) -> float:
    """2.0 at 0, 1.0 at 50, 0.5 at 100: straight lines either side of the neutral 50."""
    return 2.0 - value / 50 if value <= 50 else 1.0 - (value - 50) / 100


def cooldown_factor(initiative: int) -> float:
    """More initiative, shorter wait between her own attempts: 0.5x at 100, 2x at 0."""
    return _scale(initiative)


def max_per_hour_for(initiative: int) -> int:
    """Initiative picks 1 to MAX_PROACTIVE_PER_HOUR attempts per hour: 1 at 0-24, 4 at 75-100."""
    return min(MAX_PROACTIVE_PER_HOUR, 1 + initiative // 25)


def settle_factor(interruption_tolerance: int) -> float:
    """Less tolerance, longer wait before addressing someone: 2x at 0, 0.5x at 100."""
    return _scale(interruption_tolerance)


def present_floor_s(interruption_tolerance: int) -> float:
    """Below LOW_TOLERANCE a person must have been near this long, whatever settle_s says."""
    return LOW_TOLERANCE_PRESENT_S if interruption_tolerance < LOW_TOLERANCE else 0.0


def curiosity_threshold(base: float, curiosity: int) -> float:
    """The interestingness threshold shifted by curiosity: base * (1.3 - 0.6 * curiosity / 100), clamped."""
    low, high = THRESHOLD_BOUNDS
    return min(high, max(low, base * (1.3 - 0.6 * curiosity / 100)))


def investigate_max_per_hour(base: int, curiosity: int) -> int:
    """The investigator's hourly limit scaled by curiosity: between 0.5x and 1.5x of the configured value."""
    return max(1, round(base * (0.5 + curiosity / 100)))
