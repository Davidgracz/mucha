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
    backend: str = "auto"
    gpu_device: int = 0


@dataclass(slots=True)
class LanguageConfig:
    database: Path
    min_chars_before_speaking: int = 1200
    min_unique_chars_before_speaking: int = 18
    max_generated_chars: int = 220
    spontaneous_text: bool = True
    reply_cooldown_seconds: int = 35
    spontaneous_cooldown_seconds: int = 180
    learn_from_bots: bool = False


@dataclass(slots=True)
class VoiceConfig:
    enabled: bool = True
    poll_seconds: int = 15
    minimum_dwell_seconds: int = 60
    maximum_dwell_seconds: int = 300
    overstay_punish_amount: float = 0.5
    overstay_punish_interval_seconds: int = 60
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
    reaction_cooldown_seconds: int = 20
    reaction_candidate_sample: int = 64
    save_every_seconds: int = 45


@dataclass(slots=True)
class DiscordConfig:
    command_prefix: str = "!mucha "


@dataclass(slots=True)
class ConsoleUIConfig:
    mode: str = "dashboard"
    refresh_seconds: float = 1.0
    top_neurons: int = 8


@dataclass(slots=True)
class WebUIConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8765
    auto_open: bool = True
    refresh_ms: int = 500
    history_points: int = 180


@dataclass(slots=True)
class Config:
    brain: BrainConfig
    language: LanguageConfig
    voice: VoiceConfig
    behavior: BehaviorConfig
    discord: DiscordConfig
    console_ui: ConsoleUIConfig
    web_ui: WebUIConfig


def load_config(path: str | Path = "config.toml") -> Config:
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)

    b = raw["brain"]
    l = dict(raw["language"])
    # Backward compatibility with pre-character language configs.
    if "min_chars_before_speaking" not in l:
        legacy = int(l.pop("min_tokens_before_speaking", 450))
        l["min_chars_before_speaking"] = max(600, legacy * 3)
    else:
        l.pop("min_tokens_before_speaking", None)
    if "min_unique_chars_before_speaking" not in l:
        l.pop("min_unique_tokens_before_speaking", None)
        l["min_unique_chars_before_speaking"] = 18
    else:
        l.pop("min_unique_tokens_before_speaking", None)
    if "max_generated_chars" not in l:
        legacy_max = int(l.pop("max_generated_tokens", 28))
        l["max_generated_chars"] = max(80, legacy_max * 7)
    else:
        l.pop("max_generated_tokens", None)
    v = raw["voice"]
    beh = raw["behavior"]
    d = raw["discord"]
    cui = raw.get("console_ui", {})
    wui = raw.get("web_ui", {})

    return Config(
        brain=BrainConfig(**{**b, "connectome_dir": Path(b["connectome_dir"]), "state_file": Path(b["state_file"])}),
        language=LanguageConfig(**{**l, "database": Path(l["database"])}),
        voice=VoiceConfig(**v),
        behavior=BehaviorConfig(**beh),
        discord=DiscordConfig(**d),
        console_ui=ConsoleUIConfig(**cui),
        web_ui=WebUIConfig(**wui),
    )
