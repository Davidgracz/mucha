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
    synaptic_plasticity_enabled: bool = True
    synaptic_plasticity_lr: float = 0.0015
    synaptic_plasticity_max_delta: float = 0.08
    synaptic_plasticity_trace_neurons: int = 256
    synaptic_plasticity_max_edges: int = 40000
    consolidation_enabled: bool = True
    consolidation_interval_seconds: int = 300
    bias_forgetting_half_life_hours: float = 72.0
    synaptic_forgetting_half_life_hours: float = 168.0
    synaptic_consolidation_gain: float = 0.08
    synaptic_consolidation_decay_half_life_days: float = 30.0
    synaptic_consolidation_protection: float = 5.0
    synaptic_prune_threshold: float = 0.00001
    synaptic_consolidated_threshold: float = 0.35
    neuromodulation_enabled: bool = True
    neuromodulatory_direct_residual: float = 0.12
    dopamine_plasticity_gain: float = 1.60
    serotonin_stability_gain: float = 0.10
    octopamine_arousal_gain: float = 0.35
    neuromodulator_smoothing: float = 0.90
    internal_states_enabled: bool = True
    internal_state_pool_size: int = 192
    internal_state_entry_width: int = 160
    internal_state_recurrent_gain: float = 0.32
    internal_state_level_gain: float = 4.0
    internal_state_arousal_gain: float = 0.18
    internal_state_stress_gain: float = 0.16
    internal_state_satiety_stability_gain: float = 0.08
    action_policy_enabled: bool = True
    action_policy_lr: float = 0.06
    action_policy_max_bias: float = 0.55
    action_policy_decay: float = 0.999
    steps_per_event: int = 3
    idle_steps: int = 1
    backend: str = "auto"
    gpu_device: int = 0


@dataclass(slots=True)
class LanguageConfig:
    database: Path
    min_chars_before_speaking: int = 800
    min_unique_chars_before_speaking: int = 16
    max_generated_chars: int = 120
    spontaneous_text: bool = True
    reply_cooldown_seconds: int = 35
    spontaneous_cooldown_seconds: int = 180
    learn_from_bots: bool = False
    hybrid_word_enabled: bool = True
    word_model_probability: float = 0.90
    word_max_tokens: int = 14
    word_recent_window_seconds: int = 3600
    word_recent_boost: float = 2.00
    word_frequency_exponent: float = 0.95
    word_arousal_flatten: float = 0.12
    char_frequency_exponent: float = 0.90
    char_arousal_flatten: float = 0.15
    word_reward_scale: float = 0.12
    connectome_word_control_enabled: bool = True
    connectome_word_control_min_vocab: int = 1500
    connectome_word_control_strength: float = 0.35
    connectome_word_control_candidates: int = 24
    connectome_word_feedback_enabled: bool = True
    connectome_word_feedback_steps: int = 2
    connectome_word_feedback_magnitude: float = 0.18


@dataclass(slots=True)
class VoiceConfig:
    enabled: bool = True
    poll_seconds: int = 15
    connectome_voice_control_enabled: bool = True
    voice_sensory_enabled: bool = True
    voice_sensory_interval_seconds: float = 0.50
    voice_sensory_speaker_timeout_seconds: float = 0.55
    voice_sensory_reply_window_seconds: float = 15.0
    voice_sensory_base_magnitude: float = 0.55
    voice_sensory_steps: int = 1
    social_drive_enabled: bool = True
    social_drive_start_seconds: int = 60
    social_drive_ramp_seconds: int = 300
    social_drive_max_magnitude: float = 1.40
    social_drive_stay_punish: float = 0.035
    social_drive_learning_interval_seconds: int = 45
    social_join_reward: float = 0.12
    reward_opportunity_enabled: bool = True
    reward_opportunity_ttl_seconds: int = 90
    reward_opportunity_min_strength: float = 0.65
    reward_opportunity_max_strength: float = 1.35
    reward_opportunity_success_chance: float = 0.55
    reward_opportunity_reward: float = 0.18
    reward_opportunity_stay_punish: float = 0.025
    reward_opportunity_stay_punish_interval_seconds: int = 45
    motivation_propagation_steps: int = 4
    homeostasis_enabled: bool = True
    social_fatigue_start_seconds: int = 120
    social_fatigue_ramp_seconds: int = 240
    social_fatigue_max_magnitude: float = 1.10
    habituation_enabled: bool = True
    habituation_half_life_seconds: int = 180
    habituation_max_suppression: float = 0.70
    habituation_change_magnitude: float = 0.90
    exploration_drive_enabled: bool = True
    exploration_drive_start_seconds: int = 90
    exploration_drive_ramp_seconds: int = 300
    exploration_drive_max_magnitude: float = 1.00
    episodic_prediction_enabled: bool = True
    episodic_database: str = "state/voice_episodes.sqlite3"
    episodic_memory_size: int = 256
    episodic_max_persisted_events: int = 10000
    episodic_recall_magnitude: float = 0.85
    prediction_learning_rate: float = 0.20
    prediction_error_scale: float = 0.20
    prediction_error_max_correction: float = 0.15
    prediction_max_age_seconds: int = 180
    prediction_credit_queue_size: int = 8
    prediction_credit_decay_seconds: float = 60.0
    memory_replay_enabled: bool = True
    memory_replay_idle_seconds: int = 180
    memory_replay_interval_seconds: int = 120
    memory_replay_batch_size: int = 2
    memory_replay_magnitude: float = 0.25
    memory_replay_reward_scale: float = 0.10
    memory_replay_steps: int = 6
    memory_replay_max_age_days: int = 14
    sleep_enabled: bool = True
    sleep_idle_seconds: int = 900
    sleep_cycle_interval_seconds: int = 15
    sleep_max_cycles: int = 8
    sleep_replay_batch_size: int = 4
    sleep_replay_magnitude_multiplier: float = 1.60
    sleep_reward_scale_multiplier: float = 1.50
    sleep_steps_multiplier: float = 2.00
    episodic_consolidation_gain: float = 0.08
    episodic_forgetting_half_life_days: float = 14.0
    episodic_forgetting_interval_seconds: int = 300
    episodic_consolidated_threshold: float = 0.35
    semantic_memory_enabled: bool = True
    semantic_recall_min_observations: int = 2
    semantic_recall_magnitude: float = 0.85
    semantic_recall_steps: int = 2
    uncertainty_exploration_enabled: bool = True
    uncertainty_curiosity_magnitude: float = 1.10
    uncertainty_curiosity_steps: int = 3
    uncertainty_target_weight: float = 0.30
    information_gain_reward_scale: float = 0.20
    information_gain_reward_max: float = 0.08
    information_gain_min_delta: float = 0.01
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
    blocked_voice_guild_ids: tuple[int, ...] = (
        1552972475170689146,
    )
    include_empty_channels: bool = True
    exclude_afk_channel: bool = True


@dataclass(slots=True)
class BehaviorConfig:
    idle_tick_seconds: int = 5
    attention_enabled: bool = True
    attention_half_life_seconds: float = 45.0
    working_memory_seconds: int = 120
    attention_max_items: int = 8
    attention_reinject_magnitude: float = 0.32
    attention_topic_words: int = 5
    attention_mention_boost: float = 0.35
    speak_threshold: float = 0.70
    reaction_threshold: float = 0.73
    reaction_cooldown_seconds: int = 20
    reaction_candidate_sample: int = 64
    save_every_seconds: int = 45
    social_learning_enabled: bool = True
    neural_social_memory_enabled: bool = True
    neural_affinity_weight: float = 0.70
    neural_social_learning_scale: float = 1.00
    person_model_enabled: bool = True
    person_model_min_observations: int = 2
    person_model_sensory_magnitude: float = 0.35
    channel_model_enabled: bool = True
    channel_model_min_observations: int = 2
    channel_model_sensory_magnitude: float = 0.32
    social_scene_model_enabled: bool = True
    social_scene_min_observations: int = 2
    social_scene_sensory_magnitude: float = 0.36
    voice_dynamics_learning_enabled: bool = True
    voice_dynamics_min_observations: int = 2
    voice_dynamics_sensory_magnitude: float = 0.38
    voice_dynamics_seen_cooldown_seconds: int = 45
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
    ignored_reply_affinity_step: float = 0.003
    ignored_reply_seconds: int = 45
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
    refresh_ms: int = 250
    history_points: int = 180
    auth_enabled: bool = True
    auth_username: str = "admin"
    auth_password_env: str = "MUCHA_DASHBOARD_PASSWORD"
    session_hours: int = 168
    chaser_status_file: str = "/opt/mucha-chaser/state/chaser_status.json"
    public_readonly_enabled: bool = False
    service_unit: str = "mucha.service"


@dataclass(slots=True)
class Config:
    brain: BrainConfig
    language: LanguageConfig
    voice: VoiceConfig
    behavior: BehaviorConfig
    discord: DiscordConfig
    console_ui: ConsoleUIConfig
    web_ui: WebUIConfig


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path = "config.toml") -> Config:
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)

    local_path = path.with_name("config.local.toml")
    if local_path.exists():
        with local_path.open("rb") as f:
            local_raw = tomllib.load(f)
        raw = _deep_merge(raw, local_raw)

    b = raw["brain"]
    l = dict(raw["language"])
    # Backward compatibility with pre-character language configs.
    if "min_chars_before_speaking" not in l:
        legacy = int(l.pop("min_tokens_before_speaking", 450))
        l["min_chars_before_speaking"] = max(
            600,
            min(1200, legacy * 2),
        )
    else:
        l.pop("min_tokens_before_speaking", None)
    if "min_unique_chars_before_speaking" not in l:
        l.pop("min_unique_tokens_before_speaking", None)
        l["min_unique_chars_before_speaking"] = 16
    else:
        l.pop("min_unique_tokens_before_speaking", None)
    if "max_generated_chars" not in l:
        legacy_max = int(l.pop("max_generated_tokens", 28))
        l["max_generated_chars"] = max(
            80,
            min(140, legacy_max * 4),
        )
    else:
        l.pop("max_generated_tokens", None)
    v = dict(raw["voice"])
    if "blocked_voice_channel_ids" in v:
        v["blocked_voice_channel_ids"] = tuple(
            int(x) for x in v["blocked_voice_channel_ids"]
        )
    if "blocked_voice_guild_ids" in v:
        v["blocked_voice_guild_ids"] = tuple(
            int(x) for x in v["blocked_voice_guild_ids"]
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
