from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(slots=True)
class BrainConfig:
    connectome_dir: Path
    state_file: Path
    seed: int = 67
    leak: float = 0.86
    propagation_gain: float = 0.72
    noise: float = 0.018
    plasticity_lr: float = 0.0025
    plasticity_decay: float = 0.998
    max_bias: float = 0.35
    steps_per_event: int = 3
    idle_steps: int = 1


@dataclass(slots=True)
class LanguageConfig:
    database: Path
    min_tokens_before_speaking: int = 450
    min_unique_tokens_before_speaking: int = 80
    max_generated_tokens: int = 28
    spontaneous_text: bool = True
    reply_cooldown_seconds: int = 35
    spontaneous_cooldown_seconds: int = 180
    learn_from_bots: bool = False


@dataclass(slots=True)
class VoiceConfig:
    enabled: bool = True
    poll_seconds: int = 15
    minimum_dwell_seconds: int = 60
    move_threshold: float = 0.67
    join_threshold: float = 0.72
    leave_threshold: float = 0.82
    move_margin: float = 0.05
    include_empty_channels: bool = True
    exclude_afk_channel: bool = True


@dataclass(slots=True)
class BehaviorConfig:
    idle_tick_seconds: int = 5
    speak_threshold: float = 0.70
    reaction_threshold: float = 0.73
    save_every_seconds: int = 45


@dataclass(slots=True)
class DiscordConfig:
    command_prefix: str = "!mucha "


@dataclass(slots=True)
class Config:
    brain: BrainConfig
    language: LanguageConfig
    voice: VoiceConfig
    behavior: BehaviorConfig
    discord: DiscordConfig


def load_config(path: str | Path = "config.toml") -> Config:
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)

    b = raw["brain"]
    l = raw["language"]
    v = raw["voice"]
    beh = raw["behavior"]
    d = raw["discord"]

    return Config(
        brain=BrainConfig(**{**b, "connectome_dir": Path(b["connectome_dir"]), "state_file": Path(b["state_file"])}),
        language=LanguageConfig(**{**l, "database": Path(l["database"])}),
        voice=VoiceConfig(**v),
        behavior=BehaviorConfig(**beh),
        discord=DiscordConfig(**d),
    )
