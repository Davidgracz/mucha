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
    threat_ramp_seconds: int = 120
    threat_magnitude: float = 1.2
    threat_move_boost: float = 0.28
    threat_affinity_relaxation: float = 0.18
    threat_escape_reward: float = 0.35
    threat_steps: int = 3
    deadly_channel_seconds: int = 600
    deadly_threat_magnitude: float = 1.6
    exploration_memory_seconds: int = 1800
    exploration_novelty_bonus: float = 0.32
    exploration_recent_penalty: float = 0.38
    exploration_temperature: float = 0.18
    exploration_min_candidates: int = 3
    random_audio_enabled: bool = True
    random_audio_file: str = "assets/random_audio.mp3"
    random_audio_chance_denominator: int = 10000
    random_audio_volume: float = 0.8
    ffmpeg_executable: str = "ffmpeg"
    tts_enabled: bool = True
    tts_interval_seconds: int = 10
    tts_rate: int = 185
    tts_volume: float = 0.9
    tts_engine: str = "piper"
    tts_piper_model: str = "voices/pl_PL-gosia-medium.onnx"
    tts_piper_length_scale: float = 1.0
    tts_voice_name: str = "pl"
    tts_max_chars: int = 180
    stt_enabled: bool = True
    stt_model: str = "base"
    stt_language: str = "pl"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_cpu_threads: int = 2
    stt_download_root: str = "state/whisper"
    stt_silence_seconds: float = 0.9
    stt_min_segment_seconds: float = 0.7
    stt_max_segment_seconds: float = 12.0
    stt_min_chars: int = 2
    stt_beam_size: int = 1
    chaser_enabled: bool = True
    chaser_bot_id: int = 0
    chaser_name_hint: str = "chaser"
    chaser_confirm_hits: int = 2
    chaser_follow_window_seconds: float = 12.0
    chaser_panic_seconds: float = 35.0
    chaser_suspicion_seconds: float = 8.0
    chaser_escape_delay_min_seconds: float = 0.15
    chaser_escape_delay_max_seconds: float = 0.75
    chaser_channel_avoid_seconds: float = 90.0
    chaser_threat_magnitude: float = 2.2
    chaser_escape_reward: float = 0.25
    chaser_scream_enabled: bool = True
    chaser_scream_file: str = "assets/scream.mp3"
    chaser_scream_volume: float = 1.25
    chaser_scream_text: str = "AAAAAAAA!"
    move_threshold: float = 0.67
    join_threshold: float = 0.72
    leave_threshold: float = 0.82
    move_margin: float = 0.05
    blocked_voice_channel_ids: tuple[int, ...] = ()
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
    social_learning_enabled: bool = True
    social_window_seconds: int = 900
    word_reuse_reward: float = 0.20
    phrase_reuse_reward: float = 0.35
    direct_reply_reward: float = 0.10
    self_repeat_penalty: float = 0.10
    user_affinity_positive_step: float = 0.08
    user_affinity_negative_step: float = 0.12
    direct_reply_affinity_step: float = 0.015
    mention_affinity_step: float = 0.010
    continued_conversation_affinity_step: float = 0.008
    word_reuse_affinity_step: float = 0.015
    phrase_reuse_affinity_step: float = 0.025
    voice_join_affinity_step: float = 0.005
    voice_stay_affinity_step: float = 0.008
    voice_stay_seconds: int = 30
    tts_stay_affinity_step: float = 0.004
    tts_stay_seconds: int = 20
    voice_leave_after_join_affinity_step: float = 0.015
    voice_leave_after_join_seconds: int = 20
    tts_leave_affinity_step: float = 0.008
    tts_leave_seconds: int = 10
    negative_contact_cooldown_seconds: int = 20
    negative_streak_window_seconds: int = 600
    negative_streak_multiplier_step: float = 0.15
    negative_streak_max_multiplier: float = 1.50
    positive_contact_cooldown_seconds: int = 45
    familiar_affinity_threshold: float = 0.10
    user_avoid_threshold: float = -0.35
    ignore_disliked_users_text: bool = True
    avoid_disliked_users_on_voice: bool = True


@dataclass(slots=True)
class DiscordConfig:
    command_prefix: str = "!mucha "
    blocked_text_channel_ids: tuple[int, ...] = ()


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
    auth_enabled: bool = True
    auth_username: str = "admin"
    auth_password_env: str = "MUCHA_DASHBOARD_PASSWORD"
    session_hours: int = 168
    chaser_status_file: str = "/opt/mucha-chaser/state/chaser_status.json"


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
    v = dict(raw["voice"])
    if "blocked_voice_channel_ids" in v:
        v["blocked_voice_channel_ids"] = tuple(
            int(x) for x in v["blocked_voice_channel_ids"]
        )
    beh = raw["behavior"]
    d = dict(raw["discord"])
    if "blocked_text_channel_ids" in d:
        d["blocked_text_channel_ids"] = tuple(
            int(x) for x in d["blocked_text_channel_ids"]
        )
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
