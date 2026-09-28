from __future__ import annotations

import asyncio
import logging
import math
import random
import re
import shutil
import subprocess
import threading
import time
import tomllib
import wave
from dataclasses import dataclass
from pathlib import Path

import discord
import numpy as np
import emoji as emoji_lib
import pyttsx3
import tomli_w
from discord.ext import tasks
from scipy.signal import resample_poly

try:
    from discord.ext import voice_recv
except ImportError:
    voice_recv = None

try:
    from piper import PiperVoice, SynthesisConfig
except ImportError:
    PiperVoice = None
    SynthesisConfig = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

from .brain import FlyBrain
from .config import Config
from .connectome import Connectome
from .console_ui import ConsoleBrainUI
from .episodic import VoiceEpisodicMemory
from .language import OnlineLanguage
from .web_ui import WebDashboard

log = logging.getLogger("mucha")

POSITIVE = {"👍", "❤️", "❤", "😂", "🤣", "🔥", "🪰", "💚", "👏"}
NEGATIVE = {"👎", "😡", "🤮", "💩", "😒"}
POSITIVE_REACTION_WEIGHT = {
    "❤️": 1.0, "❤": 1.0, "😂": 0.9, "🤣": 0.9,
    "👍": 0.8, "🔥": 0.8, "💚": 0.8, "👏": 0.7, "🪰": 0.6,
}
NEGATIVE_REACTION_WEIGHT = {
    "🤮": 1.0, "😡": 0.9, "👎": 0.8, "💩": 0.7, "😒": 0.6,
}

# Natural-language rejection aimed at Mucha. Severity is 0..1 and controls
# both social affinity damage and the negative reward of the triggering trace.
VERBAL_REJECTION_PATTERNS: tuple[tuple[re.Pattern[str], str, float], ...] = (
    (re.compile(r"\bwypierdal(?:aj|ać|ac)?\b", re.I), "wypierdalaj", 1.00),
    (re.compile(r"\bspierdal(?:aj|ać|ac)?\b", re.I), "spierdalaj", 1.00),
    (re.compile(r"\bodpierdol\s*(?:się|sie)?\b", re.I), "odpierdol się", 1.00),
    (re.compile(r"\bpierdol\s*(?:się|sie)\b", re.I), "pierdol się", 0.95),
    (re.compile(r"\bjeb\s*(?:się|sie)\b", re.I), "jeb się", 0.95),
    (re.compile(r"\bzamknij\s+(?:mordę|morde|ryj|pysk)\b", re.I), "zamknij mordę", 0.95),
    (re.compile(r"\bstul\s+(?:mordę|morde|ryj|pysk)\b", re.I), "stul pysk", 0.95),
    (re.compile(r"\bprzestań\s+pierdolić\b", re.I), "przestań pierdolić", 0.90),
    (re.compile(r"\bprzestan\s+pierdolic\b", re.I), "przestań pierdolić", 0.90),
    (re.compile(r"\bnie\s+pierdol\b", re.I), "nie pierdol", 0.85),
    (re.compile(r"\bskończ\s+pierdolić\b", re.I), "skończ pierdolić", 0.85),
    (re.compile(r"\bskoncz\s+pierdolic\b", re.I), "skończ pierdolić", 0.85),
    (re.compile(r"\bcicho\s+kurwa\b", re.I), "cicho kurwa", 0.85),
    (re.compile(r"\bkurwa\s+(?:cicho|zamknij\s+się|zamknij\s+sie)\b", re.I), "kurwa cicho", 0.85),
    (re.compile(r"\bco\s+ty\s+pierdolisz\b", re.I), "co ty pierdolisz", 0.75),
    (re.compile(r"\bale\s+pierdolisz\b", re.I), "ale pierdolisz", 0.70),
    (re.compile(r"\bzamknij\s+(?:się|sie)\b", re.I), "zamknij się", 0.70),
    (re.compile(r"\bweź\s+się\s+zamknij\b", re.I), "weź się zamknij", 0.75),
    (re.compile(r"\bwez\s+sie\s+zamknij\b", re.I), "weź się zamknij", 0.75),
    (re.compile(r"\bnie\s+odzywaj\s+(?:się|sie)\b", re.I), "nie odzywaj się", 0.70),
    (re.compile(r"\bzamilcz\b", re.I), "zamilcz", 0.65),
    (re.compile(r"\bjapa\b", re.I), "japa", 0.65),
    (re.compile(r"\bstul\s+się\b", re.I), "stul się", 0.65),
    (re.compile(r"\bstul\s+sie\b", re.I), "stul się", 0.65),
    (re.compile(r"\bprzestań\b", re.I), "przestań", 0.45),
    (re.compile(r"\bprzestan\b", re.I), "przestań", 0.45),
    (re.compile(r"\bdaj\s+spokój\b", re.I), "daj spokój", 0.35),
    (re.compile(r"\bdaj\s+spokoj\b", re.I), "daj spokój", 0.35),
    (re.compile(r"\bgłupia\s+mucha\b", re.I), "głupia mucha", 0.55),
    (re.compile(r"\bglupia\s+mucha\b", re.I), "głupia mucha", 0.55),
    (re.compile(r"\bdebilna\s+mucha\b", re.I), "debilna mucha", 0.70),
    (re.compile(r"\bidiotyczna\s+mucha\b", re.I), "idiotyczna mucha", 0.65),
    (re.compile(r"\bzamknij\s+kurwa\s+(?:mordę|morde|ryj|pysk|japę|jape)\b", re.I), "zamknij kurwa mordę", 1.00),
    (re.compile(r"\bstul\s+kurwa\s+(?:mordę|morde|ryj|pysk|japę|jape)\b", re.I), "stul kurwa pysk", 1.00),
    (re.compile(r"\bstul\s+(?:japę|jape)\b", re.I), "stul japę", 0.90),
    (re.compile(r"\bzamknij\s+(?:japę|jape)\b", re.I), "zamknij japę", 0.90),
    (re.compile(r"\b(?:jebana|pierdolona)\s+mucha\b", re.I), "jebana/pierdolona mucha", 0.90),
    (re.compile(r"\bjebać\s+muchę\b", re.I), "jebać muchę", 1.00),
    (re.compile(r"\bjebac\s+muche\b", re.I), "jebać muchę", 1.00),
    (re.compile(r"\bgówno\s+(?:gadasz|mówisz|piszesz)\b", re.I), "gówno gadasz", 0.85),
    (re.compile(r"\bgowno\s+(?:gadasz|mowisz|piszesz)\b", re.I), "gówno gadasz", 0.85),
    (re.compile(r"\b(?:co|ale)\s+za\s+gówno\b", re.I), "co za gówno", 0.80),
    (re.compile(r"\b(?:co|ale)\s+za\s+gowno\b", re.I), "co za gówno", 0.80),
    (re.compile(r"\bty\s+(?:debilu|idioto|idiotko|kretynie)\b", re.I), "obraźliwe wyzwisko", 0.80),
    (re.compile(r"\bty\s+(?:kurwo|szmato)\b", re.I), "mocne wyzwisko", 1.00),
    (re.compile(r"\b(?:debilna|jebana|pierdolona)\s+botka\b", re.I), "obraźliwe określenie bota", 0.90),
    (re.compile(r"\b(?:weź|wez)\s+wypierdalaj\b", re.I), "weź wypierdalaj", 1.00),
    (re.compile(r"\bwypierdalaj\s+stąd\b", re.I), "wypierdalaj stąd", 1.00),
    (re.compile(r"\bspierdalaj\s+(?:stąd|mi\s+stąd)\b", re.I), "spierdalaj stąd", 1.00),
)


@dataclass
class SentTrace:
    trigrams: list[tuple[str,str,str]]
    created: float
    action: str
    learning_trace: tuple
    text: str = ""
    guild_id: int | None = None
    channel_id: int | None = None


class MuchaClient(discord.Client):
    # Permanent no-write channels. These IDs are intentionally hard-coded so
    # neither config.local.toml nor the web panel can accidentally enable them.
    HARD_BLOCKED_TEXT_CHANNEL_IDS = frozenset({
        344519890083774475,
        506193122460434443,
        784590857633267713,
    })

    def __init__(self, cfg: Config):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.voice_states = True
        super().__init__(intents=intents)
        self.cfg = cfg
        self.connectome = Connectome.load(cfg.brain.connectome_dir)
        self.brain = FlyBrain(self.connectome, cfg.brain)
        self.language = OnlineLanguage(
            cfg.language.database,
            cfg.language.min_chars_before_speaking,
            cfg.language.min_unique_chars_before_speaking,
            cfg.language.max_generated_chars,
            seed=cfg.brain.seed,
            hybrid_word_enabled=cfg.language.hybrid_word_enabled,
            word_model_probability=cfg.language.word_model_probability,
            word_max_tokens=cfg.language.word_max_tokens,
            word_recent_window_seconds=cfg.language.word_recent_window_seconds,
            word_recent_boost=cfg.language.word_recent_boost,
            word_frequency_exponent=cfg.language.word_frequency_exponent,
            word_arousal_flatten=cfg.language.word_arousal_flatten,
            char_frequency_exponent=cfg.language.char_frequency_exponent,
            char_arousal_flatten=cfg.language.char_arousal_flatten,
            word_reward_scale=cfg.language.word_reward_scale,
            connectome_word_control_enabled=(
                cfg.language.connectome_word_control_enabled
            ),
            connectome_word_control_min_vocab=(
                cfg.language.connectome_word_control_min_vocab
            ),
            connectome_word_control_strength=(
                cfg.language.connectome_word_control_strength
            ),
            connectome_word_control_candidates=(
                cfg.language.connectome_word_control_candidates
            ),
        )
        self._language_start_diag = self.language.diagnostics()
        self._stt_transcripts_since_start = 0
        self.random = random.Random(cfg.brain.seed + 1)
        episodic_path = Path(
            cfg.voice.episodic_database
            or "state/voice_episodes.sqlite3"
        )
        if not episodic_path.is_absolute():
            episodic_path = (
                Path(__file__).resolve().parents[1]
                / episodic_path
            )
        self.voice_episodes = VoiceEpisodicMemory(
            max_events=cfg.voice.episodic_memory_size,
            learning_rate=cfg.voice.prediction_learning_rate,
            database=episodic_path,
            max_persisted_events=(
                cfg.voice.episodic_max_persisted_events
            ),
            consolidation_gain=(
                cfg.voice.episodic_consolidation_gain
            ),
            forgetting_half_life_days=(
                cfg.voice.episodic_forgetting_half_life_days
            ),
            forgetting_interval_seconds=(
                cfg.voice.episodic_forgetting_interval_seconds
            ),
            consolidated_threshold=(
                cfg.voice.episodic_consolidated_threshold
            ),
        )
        self._voice_prediction_pending: dict[int, list[dict]] = {}
        self._voice_prediction_corrections: dict[int, list[dict]] = {}
        self._voice_prediction_last: dict[int, dict] = {}
        self._last_external_activity = time.monotonic()
        self._memory_replay_last = 0.0
        self._memory_replay_count = 0
        self._memory_replay_recent_keys: list[str] = []
        self._memory_replay_debug: dict = {
            "enabled": bool(cfg.voice.memory_replay_enabled),
            "state": "WAITING",
            "reason": "startup",
            "count": 0,
            "last_at": 0.0,
            "last": [],
        }
        self.last_reply: dict[int, float] = {}
        self.last_spontaneous: dict[int, float] = {}
        self.last_text_channel: dict[int, int] = {}
        self.last_text_context: dict[int, str] = {}
        self.last_text_author: dict[int, int] = {}
        self.voice_arrived: dict[int, float] = {}
        self.sent: dict[int, SentTrace] = {}
        self.paused = False
        self._last_save = time.monotonic()
        self._brain_lock = asyncio.Lock()
        self._last_presence_text: str | None = None
        self._last_brain_event = "startup"
        self._last_brain_action = "brak"
        self._voice_debug: dict[int, dict] = {}
        self._reaction_debug: dict = {
            "score": 0.0,
            "threshold": cfg.behavior.reaction_threshold,
            "decision": "BRAK DANYCH",
            "emoji": None,
            "target": None,
            "cooldown_remaining": 0.0,
        }
        self._last_reaction: dict[int, float] = {}
        self._social_positive_last: dict[tuple[int, str], float] = {}
        self._social_positive_streak: dict[int, dict] = {}
        self._voice_social_stay_tasks: dict[tuple[int, int], asyncio.Task] = {}
        self._tts_social_stay_tasks: dict[tuple[int, int], asyncio.Task] = {}
        self._social_negative_last: dict[tuple[int, str], float] = {}
        self._social_negative_streak: dict[int, dict] = {}
        self._voice_arrival_members: dict[int, set[int]] = {}
        self._voice_arrival_channel: dict[int, int] = {}
        self._voice_arrival_learning: dict[int, tuple[str, tuple, float, int]] = {}
        self._pending_direct_replies: dict[tuple[int, int], dict] = {}
        self._social_debug: dict = {
            "event": "BRAK",
            "detail": "",
            "amount": 0.0,
            "user_id": None,
            "user_name": None,
            "affinity": 0.0,
            "updated_at": 0.0,
        }
        self._last_overstay_punish: dict[int, float] = {}
        self._last_social_drive_punish: dict[int, float] = {}
        self._last_reward_opportunity_stay_punish: dict[int, float] = {}
        self._voice_reward_opportunity: dict[int, dict] = {}
        self._deadly_voice_until: dict[tuple[int, int], float] = {}
        self._voice_last_visit: dict[tuple[int, int], float] = {}
        self._voice_homeostasis_state: dict[int, dict] = {}
        self._chaser_follow_state: dict[tuple[int, int], dict] = {}
        self._chaser_confirmed: dict[int, int] = {}
        self._chaser_panic_until: dict[int, float] = {}
        self._chaser_escape_tasks: dict[int, asyncio.Task] = {}
        self._chaser_scream_tasks: dict[int, asyncio.Task] = {}
        self._random_audio_missing_warned = False
        self._piper_voice = None
        self._piper_model_path: str | None = None
        self._piper_lock = threading.Lock()
        self._piper_warning_shown = False
        self._stt_buffers: dict[tuple[int, int], dict] = {}
        self._stt_buffer_lock = threading.Lock()
        self._stt_model = None
        self._stt_model_key: tuple | None = None
        self._stt_model_lock = threading.Lock()
        self._stt_inference_lock = asyncio.Lock()
        self._stt_pending = 0
        self._last_tts_trace: dict[int, SentTrace] = {}
        self._last_tts_audience: dict[int, set[int]] = {}
        self._stt_debug: dict = {
            "enabled": bool(cfg.voice.stt_enabled),
            "status": "IDLE",
            "model": cfg.voice.stt_model,
            "device": cfg.voice.stt_device,
            "compute_type": cfg.voice.stt_compute_type,
            "user": None,
            "user_id": None,
            "guild": None,
            "channel": None,
            "text": "",
            "duration": 0.0,
            "language": cfg.voice.stt_language,
            "language_probability": None,
            "pending": 0,
            "error": "",
            "updated_at": time.time(),
        }
        self._audio_playback_token = 0
        self._audio_debug: dict = {
            "status": "STARTUP",
            "stage": "init",
            "error": "",
            "ffmpeg": "",
            "guild": None,
            "channel": None,
            "file": None,
            "file_size": 0,
            "text": "",
            "updated_at": time.time(),
        }
        self._unicode_emojis = [
            char
            for char, data in emoji_lib.EMOJI_DATA.items()
            if data.get("status") == emoji_lib.STATUS["fully_qualified"]
        ]
        self._action_history: list[dict] = []
        self._reward_history: list[dict] = []
        self._last_reinforceable: dict[int, tuple[str, tuple]] = {}
        self._guild_learning_context: dict[int, dict] = {}
        self.console_ui = ConsoleBrainUI(
            mode=cfg.console_ui.mode,
            top_neurons=cfg.console_ui.top_neurons,
        )
        self.web_ui = WebDashboard(
            snapshot_provider=self._console_snapshot,
            connectome_provider=self._connectome_dashboard_snapshot,
            neuromap_provider=self._neuromap_dashboard_snapshot,
            association_provider=self._association_dashboard_snapshot,
            public_readonly_enabled=cfg.web_ui.public_readonly_enabled,
            service_unit=cfg.web_ui.service_unit,
            host=cfg.web_ui.host,
            port=cfg.web_ui.port,
            auto_open=cfg.web_ui.auto_open,
            refresh_ms=cfg.web_ui.refresh_ms,
            history_points=cfg.web_ui.history_points,
            auth_enabled=cfg.web_ui.auth_enabled,
            auth_username=cfg.web_ui.auth_username,
            auth_password_env=cfg.web_ui.auth_password_env,
            session_hours=cfg.web_ui.session_hours,
            chaser_status_file=cfg.web_ui.chaser_status_file,
            config_provider=self._dashboard_config_snapshot,
            config_updater=self._dashboard_update_config,
        )

    def _dashboard_config_snapshot(self) -> dict:
        brain_fields = [
            "synaptic_plasticity_enabled",
            "synaptic_plasticity_lr",
            "synaptic_plasticity_max_delta",
            "synaptic_plasticity_trace_neurons",
            "synaptic_plasticity_max_edges",
            "consolidation_enabled",
            "consolidation_interval_seconds",
            "bias_forgetting_half_life_hours",
            "synaptic_forgetting_half_life_hours",
            "synaptic_consolidation_gain",
            "synaptic_consolidation_decay_half_life_days",
            "synaptic_consolidation_protection",
            "synaptic_prune_threshold",
            "synaptic_consolidated_threshold",
            "neuromodulation_enabled",
            "neuromodulatory_direct_residual",
            "dopamine_plasticity_gain",
            "serotonin_stability_gain",
            "octopamine_arousal_gain",
            "neuromodulator_smoothing",
            "internal_states_enabled",
            "internal_state_pool_size",
            "internal_state_entry_width",
            "internal_state_recurrent_gain",
            "internal_state_level_gain",
            "internal_state_arousal_gain",
            "internal_state_stress_gain",
            "internal_state_satiety_stability_gain",
        ]
        language_fields = [
            "min_chars_before_speaking",
            "min_unique_chars_before_speaking",
            "max_generated_chars",
            "spontaneous_text",
            "reply_cooldown_seconds",
            "spontaneous_cooldown_seconds",
            "learn_from_bots",
            "hybrid_word_enabled",
            "word_model_probability",
            "word_max_tokens",
            "word_recent_window_seconds",
            "word_recent_boost",
            "word_frequency_exponent",
            "word_arousal_flatten",
            "char_frequency_exponent",
            "char_arousal_flatten",
            "word_reward_scale",
            "connectome_word_control_enabled",
            "connectome_word_control_min_vocab",
            "connectome_word_control_strength",
            "connectome_word_control_candidates",
            "connectome_word_feedback_enabled",
            "connectome_word_feedback_steps",
            "connectome_word_feedback_magnitude",
        ]
        behavior_fields = [
            "speak_threshold",
            "reaction_threshold",
            "reaction_cooldown_seconds",
            "social_learning_enabled",
            "neural_social_memory_enabled",
            "neural_affinity_weight",
            "neural_social_learning_scale",
            "social_window_seconds",
            "word_reuse_reward",
            "phrase_reuse_reward",
            "direct_reply_reward",
            "self_repeat_penalty",
            "user_affinity_positive_step",
            "user_affinity_negative_step",
            "direct_reply_affinity_step",
            "mention_affinity_step",
            "continued_conversation_affinity_step",
            "word_reuse_affinity_step",
            "phrase_reuse_affinity_step",
            "voice_join_affinity_step",
            "voice_stay_affinity_step",
            "voice_stay_seconds",
            "tts_stay_affinity_step",
            "tts_stay_seconds",
            "voice_leave_after_join_affinity_step",
            "voice_leave_after_join_seconds",
            "tts_leave_affinity_step",
            "tts_leave_seconds",
            "ignored_reply_affinity_step",
            "ignored_reply_seconds",
            "negative_contact_cooldown_seconds",
            "negative_streak_window_seconds",
            "negative_streak_multiplier_step",
            "negative_streak_max_multiplier",
            "positive_contact_cooldown_seconds",
            "familiar_affinity_threshold",
            "user_avoid_threshold",
            "ignore_disliked_users_text",
            "avoid_disliked_users_on_voice",
        ]
        voice_fields = [
            "poll_seconds",
            "connectome_voice_control_enabled",
            "social_drive_enabled",
            "social_drive_start_seconds",
            "social_drive_ramp_seconds",
            "social_drive_max_magnitude",
            "social_drive_stay_punish",
            "social_drive_learning_interval_seconds",
            "social_join_reward",
            "reward_opportunity_enabled",
            "reward_opportunity_ttl_seconds",
            "reward_opportunity_min_strength",
            "reward_opportunity_max_strength",
            "reward_opportunity_success_chance",
            "reward_opportunity_reward",
            "reward_opportunity_stay_punish",
            "reward_opportunity_stay_punish_interval_seconds",
            "motivation_propagation_steps",
            "homeostasis_enabled",
            "social_fatigue_start_seconds",
            "social_fatigue_ramp_seconds",
            "social_fatigue_max_magnitude",
            "habituation_enabled",
            "habituation_half_life_seconds",
            "habituation_max_suppression",
            "habituation_change_magnitude",
            "exploration_drive_enabled",
            "exploration_drive_start_seconds",
            "exploration_drive_ramp_seconds",
            "exploration_drive_max_magnitude",
            "episodic_prediction_enabled",
            "episodic_database",
            "episodic_memory_size",
            "episodic_max_persisted_events",
            "episodic_recall_magnitude",
            "prediction_learning_rate",
            "prediction_error_scale",
            "prediction_error_max_correction",
            "prediction_max_age_seconds",
            "prediction_credit_queue_size",
            "prediction_credit_decay_seconds",
            "memory_replay_enabled",
            "memory_replay_idle_seconds",
            "memory_replay_interval_seconds",
            "memory_replay_batch_size",
            "memory_replay_magnitude",
            "memory_replay_reward_scale",
            "memory_replay_steps",
            "memory_replay_max_age_days",
            "episodic_consolidation_gain",
            "episodic_forgetting_half_life_days",
            "episodic_forgetting_interval_seconds",
            "episodic_consolidated_threshold",
            "minimum_dwell_seconds",
            "maximum_dwell_seconds",
            "overstay_punish_amount",
            "overstay_punish_interval_seconds",
            "threat_ramp_seconds",
            "threat_magnitude",
            "threat_move_boost",
            "threat_affinity_relaxation",
            "threat_escape_reward",
            "threat_steps",
            "move_threshold",
            "join_threshold",
            "leave_threshold",
            "include_empty_channels",
            "tts_enabled",
            "tts_interval_seconds",
            "tts_volume",
            "stt_enabled",
            "stt_model",
            "stt_language",
            "stt_device",
            "stt_compute_type",
            "stt_cpu_threads",
            "stt_silence_seconds",
            "stt_min_segment_seconds",
            "stt_max_segment_seconds",
            "stt_min_chars",
            "stt_beam_size",
            "random_audio_enabled",
        ]
        data = {
            "brain": {
                key: getattr(self.cfg.brain, key)
                for key in brain_fields
            },
            "language": {
                key: getattr(self.cfg.language, key)
                for key in language_fields
            },
            "behavior": {
                key: getattr(self.cfg.behavior, key)
                for key in behavior_fields
            },
            "voice": {
                key: getattr(self.cfg.voice, key)
                for key in voice_fields
            },
            "discord": {
                "blocked_text_channel_ids": list(
                    self.cfg.discord.blocked_text_channel_ids
                ),
            },
            "blocked_voice_channel_ids": list(
                self.cfg.voice.blocked_voice_channel_ids
            ),
            "blocked_voice_guild_ids": list(
                self.cfg.voice.blocked_voice_guild_ids
            ),
            "channels": {
                "text": [],
                "voice": [],
            },
            "guilds": [],
        }

        blocked_text = (
            set(self.cfg.discord.blocked_text_channel_ids)
            | set(self.HARD_BLOCKED_TEXT_CHANNEL_IDS)
        )
        blocked_voice = set(self.cfg.voice.blocked_voice_channel_ids)
        blocked_voice_guilds = set(self.cfg.voice.blocked_voice_guild_ids)
        for guild in self.guilds:
            data["guilds"].append({
                "id": guild.id,
                "name": guild.name,
                "voice_blocked": guild.id in blocked_voice_guilds,
            })
            for channel in guild.text_channels:
                data["channels"]["text"].append({
                    "id": channel.id,
                    "name": channel.name,
                    "guild": guild.name,
                    "blocked": channel.id in blocked_text,
                    "hard_blocked": (
                        channel.id
                        in self.HARD_BLOCKED_TEXT_CHANNEL_IDS
                    ),
                })
            for channel in guild.voice_channels:
                data["channels"]["voice"].append({
                    "id": channel.id,
                    "name": channel.name,
                    "guild": guild.name,
                    "blocked": channel.id in blocked_voice,
                })
        return data

    def _dashboard_update_config(self, payload: dict) -> dict:
        allowed: dict[tuple[str, str], tuple[type, float | None, float | None]] = {
            ("brain", "synaptic_plasticity_enabled"): (
                bool, None, None
            ),
            ("brain", "synaptic_plasticity_lr"): (
                float, 0.0, 0.05
            ),
            ("brain", "synaptic_plasticity_max_delta"): (
                float, 0.001, 0.5
            ),
            ("brain", "synaptic_plasticity_trace_neurons"): (
                int, 32, 1024
            ),
            ("brain", "synaptic_plasticity_max_edges"): (
                int, 1000, 250000
            ),
            ("brain", "consolidation_enabled"): (
                bool, None, None
            ),
            ("brain", "consolidation_interval_seconds"): (
                int, 10, 86400
            ),
            ("brain", "bias_forgetting_half_life_hours"): (
                float, 1.0, 8760.0
            ),
            ("brain", "synaptic_forgetting_half_life_hours"): (
                float, 1.0, 8760.0
            ),
            ("brain", "synaptic_consolidation_gain"): (
                float, 0.0, 1.0
            ),
            ("brain", "synaptic_consolidation_decay_half_life_days"): (
                float, 0.25, 3650.0
            ),
            ("brain", "synaptic_consolidation_protection"): (
                float, 0.0, 50.0
            ),
            ("brain", "synaptic_prune_threshold"): (
                float, 0.0, 0.05
            ),
            ("brain", "synaptic_consolidated_threshold"): (
                float, 0.0, 1.0
            ),
            ("brain", "neuromodulation_enabled"): (
                bool, None, None
            ),
            ("brain", "neuromodulatory_direct_residual"): (
                float, 0.0, 1.0
            ),
            ("brain", "dopamine_plasticity_gain"): (
                float, 0.0, 4.0
            ),
            ("brain", "serotonin_stability_gain"): (
                float, 0.0, 0.30
            ),
            ("brain", "octopamine_arousal_gain"): (
                float, 0.0, 1.5
            ),
            ("brain", "neuromodulator_smoothing"): (
                float, 0.0, 0.999
            ),
            ("brain", "internal_states_enabled"): (
                bool, None, None
            ),
            ("brain", "internal_state_pool_size"): (
                int, 24, 1024
            ),
            ("brain", "internal_state_entry_width"): (
                int, 16, 1024
            ),
            ("brain", "internal_state_recurrent_gain"): (
                float, 0.0, 2.0
            ),
            ("brain", "internal_state_level_gain"): (
                float, 0.1, 20.0
            ),
            ("brain", "internal_state_arousal_gain"): (
                float, 0.0, 1.5
            ),
            ("brain", "internal_state_stress_gain"): (
                float, 0.0, 1.5
            ),
            ("brain", "internal_state_satiety_stability_gain"): (
                float, 0.0, 0.5
            ),
            ("language", "min_chars_before_speaking"): (int, 100, 1000000),
            ("language", "min_unique_chars_before_speaking"): (int, 5, 500),
            ("language", "max_generated_chars"): (int, 24, 700),
            ("language", "spontaneous_text"): (bool, None, None),
            ("language", "reply_cooldown_seconds"): (int, 0, 3600),
            ("language", "spontaneous_cooldown_seconds"): (int, 1, 86400),
            ("language", "learn_from_bots"): (bool, None, None),
            ("language", "hybrid_word_enabled"): (bool, None, None),
            ("language", "word_model_probability"): (float, 0.0, 1.0),
            ("language", "word_max_tokens"): (int, 3, 60),
            ("language", "word_recent_window_seconds"): (int, 60, 604800),
            ("language", "word_recent_boost"): (float, 1.0, 5.0),
            ("language", "word_frequency_exponent"): (float, 0.1, 2.0),
            ("language", "word_arousal_flatten"): (float, 0.0, 1.0),
            ("language", "char_frequency_exponent"): (float, 0.1, 2.0),
            ("language", "char_arousal_flatten"): (float, 0.0, 1.0),
            ("language", "word_reward_scale"): (float, 0.0, 1.0),
            ("language", "connectome_word_control_enabled"): (
                bool, None, None
            ),
            ("language", "connectome_word_control_min_vocab"): (
                int, 8, 100000
            ),
            ("language", "connectome_word_control_strength"): (
                float, 0.0, 2.0
            ),
            ("language", "connectome_word_control_candidates"): (
                int, 4, 96
            ),
            ("language", "connectome_word_feedback_enabled"): (
                bool, None, None
            ),
            ("language", "connectome_word_feedback_steps"): (
                int, 1, 8
            ),
            ("language", "connectome_word_feedback_magnitude"): (
                float, 0.0, 1.0
            ),
            ("behavior", "speak_threshold"): (float, 0.0, 1.0),
            ("behavior", "reaction_threshold"): (float, 0.0, 1.0),
            ("behavior", "reaction_cooldown_seconds"): (int, 0, 3600),
            ("behavior", "social_learning_enabled"): (bool, None, None),
            ("behavior", "neural_social_memory_enabled"): (
                bool, None, None
            ),
            ("behavior", "neural_affinity_weight"): (
                float, 0.0, 1.0
            ),
            ("behavior", "neural_social_learning_scale"): (
                float, 0.0, 3.0
            ),
            ("behavior", "social_window_seconds"): (int, 30, 86400),
            ("behavior", "word_reuse_reward"): (float, 0.0, 1.0),
            ("behavior", "phrase_reuse_reward"): (float, 0.0, 1.0),
            ("behavior", "direct_reply_reward"): (float, 0.0, 1.0),
            ("behavior", "self_repeat_penalty"): (float, 0.0, 1.0),
            ("behavior", "user_affinity_positive_step"): (float, 0.0, 1.0),
            ("behavior", "user_affinity_negative_step"): (float, 0.0, 1.0),
            ("behavior", "direct_reply_affinity_step"): (float, 0.0, 0.25),
            ("behavior", "mention_affinity_step"): (float, 0.0, 0.25),
            ("behavior", "continued_conversation_affinity_step"): (float, 0.0, 0.25),
            ("behavior", "word_reuse_affinity_step"): (float, 0.0, 0.25),
            ("behavior", "phrase_reuse_affinity_step"): (float, 0.0, 0.25),
            ("behavior", "voice_join_affinity_step"): (float, 0.0, 0.25),
            ("behavior", "voice_stay_affinity_step"): (float, 0.0, 0.25),
            ("behavior", "voice_stay_seconds"): (int, 5, 3600),
            ("behavior", "tts_stay_affinity_step"): (float, 0.0, 0.25),
            ("behavior", "tts_stay_seconds"): (int, 5, 3600),
            ("behavior", "voice_leave_after_join_affinity_step"): (float, 0.0, 0.25),
            ("behavior", "voice_leave_after_join_seconds"): (int, 1, 300),
            ("behavior", "tts_leave_affinity_step"): (float, 0.0, 0.25),
            ("behavior", "tts_leave_seconds"): (int, 1, 300),
            ("behavior", "ignored_reply_affinity_step"): (float, 0.0, 0.10),
            ("behavior", "ignored_reply_seconds"): (int, 5, 600),
            ("behavior", "negative_contact_cooldown_seconds"): (int, 1, 3600),
            ("behavior", "negative_streak_window_seconds"): (int, 30, 86400),
            ("behavior", "negative_streak_multiplier_step"): (float, 0.0, 1.0),
            ("behavior", "negative_streak_max_multiplier"): (float, 1.0, 3.0),
            ("behavior", "positive_contact_cooldown_seconds"): (int, 1, 3600),
            ("behavior", "familiar_affinity_threshold"): (float, -1.0, 1.0),
            ("behavior", "user_avoid_threshold"): (float, -1.0, 1.0),
            ("behavior", "ignore_disliked_users_text"): (bool, None, None),
            ("behavior", "avoid_disliked_users_on_voice"): (bool, None, None),
            ("voice", "poll_seconds"): (int, 1, 3600),
            ("voice", "connectome_voice_control_enabled"): (
                bool, None, None
            ),
            ("voice", "social_drive_enabled"): (
                bool, None, None
            ),
            ("voice", "social_drive_start_seconds"): (
                int, 0, 86400
            ),
            ("voice", "social_drive_ramp_seconds"): (
                int, 1, 86400
            ),
            ("voice", "social_drive_max_magnitude"): (
                float, 0.0, 4.0
            ),
            ("voice", "social_drive_stay_punish"): (
                float, 0.0, 0.5
            ),
            ("voice", "social_drive_learning_interval_seconds"): (
                int, 5, 3600
            ),
            ("voice", "social_join_reward"): (
                float, 0.0, 1.0
            ),
            ("voice", "reward_opportunity_enabled"): (
                bool, None, None
            ),
            ("voice", "reward_opportunity_ttl_seconds"): (
                int, 10, 3600
            ),
            ("voice", "reward_opportunity_min_strength"): (
                float, 0.0, 4.0
            ),
            ("voice", "reward_opportunity_max_strength"): (
                float, 0.0, 4.0
            ),
            ("voice", "reward_opportunity_success_chance"): (
                float, 0.0, 1.0
            ),
            ("voice", "reward_opportunity_reward"): (
                float, 0.0, 1.0
            ),
            ("voice", "reward_opportunity_stay_punish"): (
                float, 0.0, 0.5
            ),
            ("voice", "reward_opportunity_stay_punish_interval_seconds"): (
                int, 5, 3600
            ),
            ("voice", "motivation_propagation_steps"): (
                int, 2, 12
            ),
            ("voice", "homeostasis_enabled"): (
                bool, None, None
            ),
            ("voice", "social_fatigue_start_seconds"): (
                int, 0, 86400
            ),
            ("voice", "social_fatigue_ramp_seconds"): (
                int, 1, 86400
            ),
            ("voice", "social_fatigue_max_magnitude"): (
                float, 0.0, 4.0
            ),
            ("voice", "habituation_enabled"): (
                bool, None, None
            ),
            ("voice", "habituation_half_life_seconds"): (
                int, 1, 86400
            ),
            ("voice", "habituation_max_suppression"): (
                float, 0.0, 0.95
            ),
            ("voice", "habituation_change_magnitude"): (
                float, 0.0, 4.0
            ),
            ("voice", "exploration_drive_enabled"): (
                bool, None, None
            ),
            ("voice", "exploration_drive_start_seconds"): (
                int, 0, 86400
            ),
            ("voice", "exploration_drive_ramp_seconds"): (
                int, 1, 86400
            ),
            ("voice", "exploration_drive_max_magnitude"): (
                float, 0.0, 4.0
            ),
            ("voice", "episodic_prediction_enabled"): (
                bool, None, None
            ),
            ("voice", "episodic_database"): (
                str, None, None
            ),
            ("voice", "episodic_memory_size"): (
                int, 16, 5000
            ),
            ("voice", "episodic_max_persisted_events"): (
                int, 16, 1000000
            ),
            ("voice", "episodic_recall_magnitude"): (
                float, 0.0, 4.0
            ),
            ("voice", "prediction_learning_rate"): (
                float, 0.001, 1.0
            ),
            ("voice", "prediction_error_scale"): (
                float, 0.0, 1.0
            ),
            ("voice", "prediction_error_max_correction"): (
                float, 0.0, 0.5
            ),
            ("voice", "prediction_max_age_seconds"): (
                int, 5, 3600
            ),
            ("voice", "prediction_credit_queue_size"): (
                int, 1, 64
            ),
            ("voice", "prediction_credit_decay_seconds"): (
                float, 1.0, 3600.0
            ),
            ("voice", "memory_replay_enabled"): (
                bool, None, None
            ),
            ("voice", "memory_replay_idle_seconds"): (
                int, 10, 86400
            ),
            ("voice", "memory_replay_interval_seconds"): (
                int, 30, 86400
            ),
            ("voice", "memory_replay_batch_size"): (
                int, 1, 8
            ),
            ("voice", "memory_replay_magnitude"): (
                float, 0.0, 1.5
            ),
            ("voice", "memory_replay_reward_scale"): (
                float, 0.0, 0.5
            ),
            ("voice", "memory_replay_steps"): (
                int, 1, 24
            ),
            ("voice", "memory_replay_max_age_days"): (
                int, 1, 365
            ),
            ("voice", "episodic_consolidation_gain"): (
                float, 0.0, 1.0
            ),
            ("voice", "episodic_forgetting_half_life_days"): (
                float, 0.25, 3650.0
            ),
            ("voice", "episodic_forgetting_interval_seconds"): (
                int, 30, 86400
            ),
            ("voice", "episodic_consolidated_threshold"): (
                float, 0.0, 1.0
            ),
            ("voice", "minimum_dwell_seconds"): (int, 0, 86400),
            ("voice", "maximum_dwell_seconds"): (int, 1, 86400),
            ("voice", "overstay_punish_amount"): (float, 0.0, 1.0),
            ("voice", "overstay_punish_interval_seconds"): (int, 1, 3600),
            ("voice", "threat_ramp_seconds"): (int, 1, 86400),
            ("voice", "threat_magnitude"): (float, 0.0, 4.0),
            ("voice", "threat_move_boost"): (float, 0.0, 1.0),
            ("voice", "threat_affinity_relaxation"): (float, 0.0, 1.0),
            ("voice", "threat_escape_reward"): (float, 0.0, 1.0),
            ("voice", "threat_steps"): (int, 1, 12),
            ("voice", "move_threshold"): (float, 0.0, 1.0),
            ("voice", "join_threshold"): (float, 0.0, 1.0),
            ("voice", "leave_threshold"): (float, 0.0, 1.0),
            ("voice", "include_empty_channels"): (bool, None, None),
            ("voice", "tts_enabled"): (bool, None, None),
            ("voice", "tts_interval_seconds"): (int, 1, 3600),
            ("voice", "tts_volume"): (float, 0.0, 2.0),
            ("voice", "stt_enabled"): (bool, None, None),
            ("voice", "stt_model"): (str, None, None),
            ("voice", "stt_language"): (str, None, None),
            ("voice", "stt_device"): (str, None, None),
            ("voice", "stt_compute_type"): (str, None, None),
            ("voice", "stt_cpu_threads"): (int, 1, 64),
            ("voice", "stt_silence_seconds"): (float, 0.2, 5.0),
            ("voice", "stt_min_segment_seconds"): (float, 0.2, 10.0),
            ("voice", "stt_max_segment_seconds"): (float, 2.0, 60.0),
            ("voice", "stt_min_chars"): (int, 1, 100),
            ("voice", "stt_beam_size"): (int, 1, 10),
            ("voice", "random_audio_enabled"): (bool, None, None),
        }

        config_path = Path(__file__).resolve().parents[1] / "config.local.toml"
        if config_path.exists():
            with config_path.open("rb") as handle:
                raw = tomllib.load(handle)
        else:
            raw = {}

        changed = []
        for (section, key), (kind, minimum, maximum) in allowed.items():
            section_payload = payload.get(section, {})
            if key not in section_payload:
                continue
            value = section_payload[key]
            if kind is bool:
                value = bool(value)
            elif kind is int:
                value = int(value)
            elif kind is str:
                value = str(value).strip()
            else:
                value = float(value)
            if minimum is not None:
                value = max(minimum, value)
            if maximum is not None:
                value = min(maximum, value)

            current = getattr(getattr(self.cfg, section), key)
            raw.setdefault(section, {})[key] = value
            setattr(getattr(self.cfg, section), key, value)
            if current != value:
                changed.append(f"{section}.{key}")

        if "blocked_text_channel_ids" in payload:
            ids = tuple(
                sorted({
                    int(value)
                    for value in payload.get(
                        "blocked_text_channel_ids",
                        [],
                    )
                })
            )
            raw.setdefault("discord", {})[
                "blocked_text_channel_ids"
            ] = list(ids)
            current_ids = tuple(self.cfg.discord.blocked_text_channel_ids)
            self.cfg.discord.blocked_text_channel_ids = ids
            if current_ids != ids:
                changed.append("discord.blocked_text_channel_ids")

        if "blocked_voice_channel_ids" in payload:
            ids = tuple(
                sorted({
                    int(value)
                    for value in payload.get(
                        "blocked_voice_channel_ids",
                        [],
                    )
                })
            )
            raw.setdefault("voice", {})[
                "blocked_voice_channel_ids"
            ] = list(ids)
            current_ids = tuple(self.cfg.voice.blocked_voice_channel_ids)
            self.cfg.voice.blocked_voice_channel_ids = ids
            if current_ids != ids:
                changed.append("voice.blocked_voice_channel_ids")

        if "blocked_voice_guild_ids" in payload:
            ids = tuple(
                sorted({
                    int(value)
                    for value in payload.get(
                        "blocked_voice_guild_ids",
                        [],
                    )
                })
            )
            raw.setdefault("voice", {})[
                "blocked_voice_guild_ids"
            ] = list(ids)
            current_ids = tuple(self.cfg.voice.blocked_voice_guild_ids)
            self.cfg.voice.blocked_voice_guild_ids = ids
            if current_ids != ids:
                changed.append("voice.blocked_voice_guild_ids")

        temp_path = config_path.with_suffix(".toml.tmp")
        temp_path.write_text(
            tomli_w.dumps(raw),
            encoding="utf-8",
        )
        temp_path.replace(config_path)

        self.language.min_chars = max(
            1,
            int(self.cfg.language.min_chars_before_speaking),
        )
        self.language.min_unique_chars = max(
            1,
            int(self.cfg.language.min_unique_chars_before_speaking),
        )
        self.language.max_chars = max(
            24,
            int(self.cfg.language.max_generated_chars),
        )
        self.language.hybrid_word_enabled = bool(
            self.cfg.language.hybrid_word_enabled
        )
        self.language.word_model_probability = max(
            0.0,
            min(1.0, float(self.cfg.language.word_model_probability)),
        )
        self.language.word_max_tokens = max(
            3,
            int(self.cfg.language.word_max_tokens),
        )
        self.language.word_recent_window_seconds = max(
            1,
            int(self.cfg.language.word_recent_window_seconds),
        )
        self.language.word_recent_boost = max(
            1.0,
            float(self.cfg.language.word_recent_boost),
        )
        self.language.word_frequency_exponent = max(
            0.1,
            float(self.cfg.language.word_frequency_exponent),
        )
        self.language.word_arousal_flatten = max(
            0.0,
            float(self.cfg.language.word_arousal_flatten),
        )
        self.language.char_frequency_exponent = max(
            0.1,
            float(self.cfg.language.char_frequency_exponent),
        )
        self.language.char_arousal_flatten = max(
            0.0,
            float(self.cfg.language.char_arousal_flatten),
        )
        self.language.word_reward_scale = max(
            0.0,
            min(1.0, float(self.cfg.language.word_reward_scale)),
        )
        self.language.connectome_word_control_enabled = bool(
            self.cfg.language.connectome_word_control_enabled
        )
        self.language.connectome_word_control_min_vocab = max(
            8,
            int(self.cfg.language.connectome_word_control_min_vocab),
        )
        self.language.connectome_word_control_strength = max(
            0.0,
            min(
                2.0,
                float(self.cfg.language.connectome_word_control_strength),
            ),
        )
        self.language.connectome_word_control_candidates = max(
            4,
            min(
                96,
                int(self.cfg.language.connectome_word_control_candidates),
            ),
        )

        self.voice_loop.change_interval(
            seconds=max(1, int(self.cfg.voice.poll_seconds))
        )
        self.tts_loop.change_interval(
            seconds=max(1, int(self.cfg.voice.tts_interval_seconds))
        )

        stt_model_fields = {
            "voice.stt_model",
            "voice.stt_language",
            "voice.stt_device",
            "voice.stt_compute_type",
            "voice.stt_cpu_threads",
        }
        if any(key in stt_model_fields for key in changed):
            with self._stt_model_lock:
                self._stt_model = None
                self._stt_model_key = None

        self._stt_debug.update({
            "enabled": bool(self.cfg.voice.stt_enabled),
            "model": self.cfg.voice.stt_model,
            "device": self.cfg.voice.stt_device,
            "compute_type": self.cfg.voice.stt_compute_type,
            "language": self.cfg.voice.stt_language,
            "updated_at": time.time(),
        })

        if self.is_ready():
            if self.cfg.voice.stt_enabled:
                if not self.stt_segment_loop.is_running():
                    self.stt_segment_loop.start()
                for vc in self.voice_clients:
                    self._ensure_voice_listener(vc)
                asyncio.create_task(self._warm_stt_model())
            elif self.stt_segment_loop.is_running():
                self.stt_segment_loop.cancel()

            if self.cfg.voice.tts_enabled:
                if (
                    self.cfg.voice.enabled
                    and not self.tts_loop.is_running()
                ):
                    self.tts_loop.start()
            elif self.tts_loop.is_running():
                self.tts_loop.cancel()

            if self.cfg.voice.random_audio_enabled:
                if (
                    self.cfg.voice.enabled
                    and not self.random_audio_loop.is_running()
                ):
                    self.random_audio_loop.start()
            elif self.random_audio_loop.is_running():
                self.random_audio_loop.cancel()

        if "voice.blocked_voice_guild_ids" in changed and self.is_ready():
            for guild in self.guilds:
                if self._is_voice_guild_blocked(guild):
                    asyncio.create_task(self._voice_decision(guild))

        self._record_action(
            "config",
            ", ".join(changed) if changed else "brak zmian",
        )
        result_config = self._dashboard_config_snapshot()
        result_config["config_path"] = str(config_path)
        return {
            "ok": True,
            "changed": changed,
            "config": result_config,
            "config_path": str(config_path),
        }

    def _record_action(
        self,
        kind: str,
        detail: str,
        guild: discord.Guild | None = None,
    ) -> None:
        self._action_history.append({
            "time": time.time(),
            "kind": kind,
            "detail": detail,
            "guild_id": guild.id if guild else None,
            "guild": guild.name if guild else None,
        })
        if len(self._action_history) > 80:
            del self._action_history[:-80]

    def _queue_voice_prediction(
        self,
        guild_id: int,
        pending: dict,
    ) -> None:
        guild_id = int(guild_id)
        now = time.monotonic()
        max_age = max(
            5.0,
            float(self.cfg.voice.prediction_max_age_seconds),
        )
        queue = [
            item
            for item in self._voice_prediction_pending.get(guild_id, [])
            if now - float(item.get("time", 0.0)) <= max_age
        ]
        queue.append(pending)
        queue_size = max(
            1,
            int(self.cfg.voice.prediction_credit_queue_size),
        )
        self._voice_prediction_pending[guild_id] = queue[-queue_size:]

    def _record_reward(
        self,
        amount: float,
        action: str | None,
        source: str,
        guild: discord.Guild | None = None,
    ) -> None:
        self._reward_history.append({
            "time": time.time(),
            "amount": float(amount),
            "action": action,
            "source": source,
            "trace": float(self.brain.reward_trace),
            "guild_id": guild.id if guild else None,
            "guild": guild.name if guild else None,
        })
        if len(self._reward_history) > 120:
            del self._reward_history[:-120]

        if (
            guild is None
            or action not in {
                "stay",
                "voice_join",
                "voice_move",
                "voice_leave",
            }
            or not self.cfg.voice.episodic_prediction_enabled
        ):
            return

        queue = list(
            self._voice_prediction_pending.get(guild.id, [])
        )
        if not queue:
            return

        now = time.monotonic()
        max_age = max(
            5.0,
            float(self.cfg.voice.prediction_max_age_seconds),
        )
        decay = max(
            1.0,
            float(self.cfg.voice.prediction_credit_decay_seconds),
        )
        valid: list[tuple[dict, float, float]] = []
        for pending in queue:
            age = max(
                0.0,
                now - float(pending.get("time", 0.0)),
            )
            if age > max_age:
                continue
            action_match = (
                1.0
                if str(pending.get("action")) == str(action)
                else 0.35
            )
            weight = action_match * math.exp(-age / decay)
            if weight > 1e-6:
                valid.append((pending, age, weight))

        if not valid:
            self._voice_prediction_pending.pop(guild.id, None)
            return

        total_weight = sum(item[2] for item in valid)
        correction_queue = self._voice_prediction_corrections.setdefault(
            guild.id,
            [],
        )
        last_episode = None
        for pending, age, raw_weight in valid:
            credit_share = raw_weight / max(1e-9, total_weight)
            pending_action = str(pending.get("action", action))
            episode = self.voice_episodes.observe(
                guild_id=guild.id,
                context=str(pending["context"]),
                action=pending_action,
                actual_reward=float(amount),
                source=(
                    f"{source} • temporal-credit "
                    f"{credit_share:.3f}"
                ),
                predicted_reward=float(
                    pending.get("predicted_reward", 0.0)
                ),
                channel_id=pending.get("channel_id"),
                channel_name=str(
                    pending.get("channel_name", "")
                ),
                user_ids=list(
                    pending.get("user_ids", [])
                ),
                user_names=list(
                    pending.get("user_names", [])
                ),
                scene_key=str(
                    pending.get("scene_key", "")
                ),
            )
            episode["credit_share"] = float(credit_share)
            episode["decision_age_seconds"] = float(age)
            last_episode = episode

            error = float(episode["prediction_error"])
            limit = max(
                0.0,
                min(
                    0.5,
                    float(
                        self.cfg.voice
                        .prediction_error_max_correction
                    ),
                ),
            )
            correction = max(
                -limit,
                min(
                    limit,
                    error
                    * max(
                        0.0,
                        float(
                            self.cfg.voice
                            .prediction_error_scale
                        ),
                    )
                    * credit_share,
                ),
            )
            if (
                abs(correction) > 1e-9
                and pending.get("trace") is not None
            ):
                correction_queue.append({
                    "amount": correction,
                    "action": pending_action,
                    "trace": pending["trace"],
                    "prediction_error": error,
                    "credit_share": credit_share,
                    "decision_age_seconds": age,
                    "source": source,
                })

        if last_episode is not None:
            self._voice_prediction_last[guild.id] = last_episode

        max_corrections = max(
            2,
            int(self.cfg.voice.prediction_credit_queue_size) * 2,
        )
        if correction_queue:
            self._voice_prediction_corrections[guild.id] = (
                correction_queue[-max_corrections:]
            )
        else:
            self._voice_prediction_corrections.pop(guild.id, None)

        # A reward event closes the current temporal-credit window. New
        # decisions made after it start a fresh causal chain.
        self._voice_prediction_pending.pop(guild.id, None)

    def _set_reinforceable(
        self,
        guild: discord.Guild,
        action: str,
        trace: tuple,
        detail: str = "",
    ) -> None:
        self._last_reinforceable[guild.id] = (action, trace)
        self._guild_learning_context[guild.id] = {
            "guild_id": guild.id,
            "guild": guild.name,
            "action": action,
            "detail": detail,
            "time": time.time(),
        }

    @staticmethod
    def _social_words(text: str) -> list[str]:
        normalized = OnlineLanguage.normalize(text).lower()
        return re.findall(r"[^\W_]{3,}", normalized, flags=re.UNICODE)

    def _user_affinity_components(self, user_id: int) -> dict:
        user_id = int(user_id)
        legacy = float(
            self.language.get_user_affinity(user_id)
        )
        if not self.cfg.behavior.neural_social_memory_enabled:
            return {
                "legacy": legacy,
                "neural": 0.0,
                "maturity": 0.0,
                "weight": 0.0,
                "effective": legacy,
                "memory": None,
            }

        memory = self.brain.user_memory_diagnostics(user_id)
        maturity = max(
            0.0,
            min(1.0, float(memory.get("maturity", 0.0))),
        )
        weight = max(
            0.0,
            min(
                1.0,
                float(self.cfg.behavior.neural_affinity_weight)
                * maturity,
            ),
        )
        neural = max(
            -1.0,
            min(
                1.0,
                float(memory.get("neural_affinity", 0.0)),
            ),
        )
        effective = max(
            -1.0,
            min(
                1.0,
                (1.0 - weight) * legacy
                + weight * neural,
            ),
        )
        return {
            "legacy": legacy,
            "neural": neural,
            "maturity": maturity,
            "weight": weight,
            "effective": effective,
            "memory": memory,
        }

    def _user_affinity(self, user_id: int) -> float:
        return float(
            self._user_affinity_components(user_id)["effective"]
        )

    def _write_neural_social_memory(
        self,
        user_id: int,
        affinity_delta: float,
    ) -> dict | None:
        if (
            not self.cfg.behavior.social_learning_enabled
            or not self.cfg.behavior.neural_social_memory_enabled
        ):
            return None
        scaled = float(affinity_delta) * max(
            0.0,
            float(self.cfg.behavior.neural_social_learning_scale),
        )
        return self.brain.reinforce_user_memory(
            int(user_id),
            scaled,
        )

    def _is_disliked_user(self, user_id: int) -> bool:
        return (
            self._user_affinity(user_id)
            <= float(self.cfg.behavior.user_avoid_threshold)
        )

    def _disliked_members(
        self,
        members: list[discord.Member],
    ) -> list[tuple[discord.Member, float]]:
        threshold = float(self.cfg.behavior.user_avoid_threshold)
        disliked = []
        for member in members:
            affinity = self._user_affinity(member.id)
            if affinity <= threshold:
                disliked.append((member, affinity))
        return disliked

    def _remember_social_event(
        self,
        event: str,
        detail: str,
        amount: float,
        member: discord.abc.User | discord.Member | None = None,
    ) -> None:
        components = (
            self._user_affinity_components(member.id)
            if member is not None
            else {
                "effective": 0.0,
                "legacy": 0.0,
                "neural": 0.0,
                "maturity": 0.0,
            }
        )
        affinity = float(components["effective"])
        self._social_debug = {
            "event": event,
            "detail": detail,
            "amount": float(amount),
            "user_id": member.id if member is not None else None,
            "user_name": (
                getattr(member, "display_name", None)
                or getattr(member, "name", None)
                if member is not None
                else None
            ),
            "affinity": affinity,
            "legacy_affinity": float(
                components.get("legacy", affinity)
            ),
            "neural_affinity": float(
                components.get("neural", 0.0)
            ),
            "neural_maturity": float(
                components.get("maturity", 0.0)
            ),
            "updated_at": time.time(),
        }

    def _advance_negative_streak(
        self,
        user_id: int,
        now: float | None = None,
    ) -> tuple[int, float]:
        now = time.monotonic() if now is None else float(now)
        window = max(
            1.0,
            float(self.cfg.behavior.negative_streak_window_seconds),
        )
        streak = self._social_negative_streak.get(
            int(user_id),
            {"count": 0, "last": 0.0},
        )
        if now - float(streak.get("last", 0.0)) > window:
            streak = {"count": 0, "last": 0.0}
        streak_count = int(streak.get("count", 0)) + 1
        streak["count"] = streak_count
        streak["last"] = now
        self._social_negative_streak[int(user_id)] = streak

        multiplier = min(
            max(
                1.0,
                float(self.cfg.behavior.negative_streak_max_multiplier),
            ),
            1.0
            + max(
                0.0,
                float(self.cfg.behavior.negative_streak_multiplier_step),
            )
            * max(0, streak_count - 1),
        )
        return streak_count, float(multiplier)

    def _soften_negative_streak(
        self,
        user_id: int,
        now: float | None = None,
    ) -> None:
        now = time.monotonic() if now is None else float(now)
        streak = self._social_negative_streak.get(int(user_id))
        if streak is None:
            return
        window = max(
            1.0,
            float(self.cfg.behavior.negative_streak_window_seconds),
        )
        if now - float(streak.get("last", 0.0)) > window:
            self._social_negative_streak.pop(int(user_id), None)
            return
        count = max(0, int(streak.get("count", 0)) - 1)
        if count <= 0:
            self._social_negative_streak.pop(int(user_id), None)
        else:
            streak["count"] = count

    async def _grant_negative_social(
        self,
        member: discord.Member,
        event: str,
        stimulus: str,
        affinity_step: float,
        guild: discord.Guild,
        detail: str = "",
        source_action: str | None = None,
        source_learning_trace: tuple | None = None,
        brain_penalty: float = 0.0,
    ) -> float | None:
        if (
            not self.cfg.behavior.social_learning_enabled
            or member.bot
        ):
            return None

        now = time.monotonic()
        cooldown = max(
            1.0,
            float(self.cfg.behavior.negative_contact_cooldown_seconds),
        )
        cooldown_key = (member.id, event)
        last = self._social_negative_last.get(cooldown_key, 0.0)
        if now - last < cooldown:
            return None
        self._social_negative_last[cooldown_key] = now

        streak_count, multiplier = self._advance_negative_streak(
            member.id,
            now,
        )
        delta = -min(
            0.25,
            max(0.0, float(affinity_step)) * multiplier,
        )
        if delta >= 0.0:
            return None

        new_affinity = self.language.adjust_user_affinity(
            member.id,
            member.display_name,
            delta,
        )

        penalty = -min(
            0.35,
            max(0.0, abs(float(brain_penalty))) * multiplier,
        )
        async with self._brain_lock:
            self._write_neural_social_memory(
                member.id,
                delta,
            )
            new_affinity = self._user_affinity(member.id)
            self.brain.inject(stimulus, 0.60, 128)
            self.brain.inject(
                f"{stimulus}:user:{member.id}",
                0.48,
                96,
            )
            self.brain.inject(
                "social:user-avoids-me",
                min(1.0, 0.50 + 0.08 * streak_count),
                128,
            )
            self.brain.inject(
                f"social:user-avoids-me:user:{member.id}",
                min(1.0, 0.42 + 0.07 * streak_count),
                96,
            )
            self.brain.inject(
                "internal:social-failure",
                min(1.0, 0.42 + 0.06 * streak_count),
                112,
            )
            if streak_count >= 2:
                self.brain.inject(
                    "social:repeated-rejection",
                    min(1.0, 0.52 + 0.08 * streak_count),
                    144,
                )
                self.brain.inject(
                    f"social:repeated-rejection:user:{member.id}",
                    min(1.0, 0.45 + 0.07 * streak_count),
                    96,
                )
            if penalty < 0.0:
                if source_learning_trace is not None:
                    self.brain.reward(
                        penalty,
                        action=source_action,
                        trace=source_learning_trace,
                    )
                else:
                    self.brain.reward(
                        penalty,
                        action=source_action,
                    )
            self.brain.step(1)

        if penalty < 0.0:
            self._record_reward(
                penalty,
                source_action,
                event.lower(),
                guild,
            )

        self._record_action(
            "negative_social",
            (
                f"{event} • {member.display_name} • "
                f"affinity {new_affinity:+.3f} • "
                f"streak {streak_count} ×{multiplier:.2f}"
            ),
            guild,
        )
        self._remember_social_event(
            event,
            (
                (detail or stimulus)
                + f" • affinity {new_affinity:+.3f}"
                + f" • streak {streak_count} ×{multiplier:.2f}"
            ),
            delta,
            member,
        )
        return new_affinity

    def _note_direct_reply_activity(
        self,
        message: discord.Message,
    ) -> None:
        if message.guild is None or message.author.bot:
            return
        key = (message.guild.id, message.author.id)
        pending = self._pending_direct_replies.get(key)
        if pending is None:
            return

        age = time.monotonic() - float(pending.get("created", 0.0))
        if age > float(self.cfg.behavior.ignored_reply_seconds):
            return

        if message.channel.id == int(pending["channel_id"]):
            self._pending_direct_replies.pop(key, None)
        else:
            pending["active_elsewhere"] = True
            pending["last_elsewhere"] = time.monotonic()

    def _track_direct_reply_target(
        self,
        guild: discord.Guild,
        member: discord.Member,
        channel_id: int,
        message_id: int,
        trace: SentTrace,
    ) -> None:
        key = (guild.id, member.id)
        pending = {
            "guild_id": guild.id,
            "user_id": member.id,
            "channel_id": int(channel_id),
            "message_id": int(message_id),
            "created": time.monotonic(),
            "active_elsewhere": False,
            "last_elsewhere": 0.0,
            "trace": trace,
        }
        self._pending_direct_replies[key] = pending
        asyncio.create_task(
            self._check_ignored_direct_reply(
                guild.id,
                member.id,
                int(message_id),
            )
        )

    async def _check_ignored_direct_reply(
        self,
        guild_id: int,
        user_id: int,
        message_id: int,
    ) -> None:
        await asyncio.sleep(
            max(5, int(self.cfg.behavior.ignored_reply_seconds))
        )
        key = (int(guild_id), int(user_id))
        pending = self._pending_direct_replies.get(key)
        if (
            pending is None
            or int(pending.get("message_id", 0)) != int(message_id)
        ):
            return
        self._pending_direct_replies.pop(key, None)

        if not bool(pending.get("active_elsewhere")):
            return

        guild = self.get_guild(int(guild_id))
        if guild is None:
            return
        member = guild.get_member(int(user_id))
        if member is None or member.bot:
            return
        trace = pending.get("trace")
        if not isinstance(trace, SentTrace):
            return

        await self._grant_negative_social(
            member,
            "TEXT_IGNORED_DIRECT_REPLY",
            "social:user-ignored-me",
            self.cfg.behavior.ignored_reply_affinity_step,
            guild,
            detail=(
                "Mucha odpowiedziała tej osobie, a ona była aktywna "
                "na innym kanale bez kontynuacji rozmowy"
            ),
            source_action=trace.action,
            source_learning_trace=trace.learning_trace,
            brain_penalty=0.02,
        )

    async def _mark_social_voice_arrival(
        self,
        guild: discord.Guild,
        channel: discord.VoiceChannel,
        action: str,
        now: float | None = None,
    ) -> None:
        now = time.monotonic() if now is None else float(now)
        if self._chaser_panic_remaining(guild.id, now) > 0.0:
            self._voice_arrival_members.pop(guild.id, None)
            self._voice_arrival_channel.pop(guild.id, None)
            self._voice_arrival_learning.pop(guild.id, None)
            return

        self._voice_arrival_members[guild.id] = {
            member.id
            for member in channel.members
            if not member.bot
        }
        self._voice_arrival_channel[guild.id] = channel.id
        async with self._brain_lock:
            learning_trace = self.brain.capture_learning_trace()
        self._voice_arrival_learning[guild.id] = (
            action,
            learning_trace,
            now,
            channel.id,
        )

    def _affinity_rules_snapshot(self) -> dict:
        behavior = self.cfg.behavior

        positive = [
            {
                "event": "Bezpośredni reply do Muchy",
                "delta": float(behavior.direct_reply_affinity_step),
                "condition": "reply do wiadomości Muchy",
            },
            {
                "event": "@Mucha / powiedzenie „Mucha” na VC",
                "delta": float(behavior.mention_affinity_step),
                "condition": "mention lub rozpoznane imię na voice",
            },
            {
                "event": "Kontynuacja rozmowy",
                "delta": float(behavior.continued_conversation_affinity_step),
                "condition": "krótko po wypowiedzi Muchy",
            },
            {
                "event": "Powtórzenie słowa Muchy",
                "delta": float(behavior.word_reuse_affinity_step),
                "condition": "słowo ma co najmniej 5 znaków",
            },
            {
                "event": "Powtórzenie frazy Muchy",
                "delta": float(behavior.phrase_reuse_affinity_step),
                "condition": "fraza 2–3 wyrazy",
            },
            {
                "event": "Wejście do VC Muchy",
                "delta": float(behavior.voice_join_affinity_step),
                "condition": "użytkownik sam dołącza do jej kanału",
            },
            {
                "event": "Zostanie z Muchą na VC",
                "delta": float(behavior.voice_stay_affinity_step),
                "condition": f"po {int(behavior.voice_stay_seconds)} s",
            },
            {
                "event": "Zostanie po TTS",
                "delta": float(behavior.tts_stay_affinity_step),
                "condition": f"po {int(behavior.tts_stay_seconds)} s",
            },
        ]
        for emoji, weight in POSITIVE_REACTION_WEIGHT.items():
            positive.append({
                "event": f"Reakcja {emoji}",
                "delta": float(behavior.user_affinity_positive_step)
                * float(weight),
                "condition": "reakcja pod wiadomością Muchy",
            })

        negative = [
            {
                "event": "Wyjście/przeniesienie po wejściu Muchy",
                "delta": -float(
                    behavior.voice_leave_after_join_affinity_step
                ),
                "condition": (
                    f"do {int(behavior.voice_leave_after_join_seconds)} s "
                    "po wejściu Muchy; tylko osoba obecna przed jej wejściem"
                ),
            },
            {
                "event": "Wyjście/przeniesienie po TTS",
                "delta": -float(behavior.tts_leave_affinity_step),
                "condition": (
                    f"do {int(behavior.tts_leave_seconds)} s "
                    "od rozpoczęcia TTS"
                ),
            },
            {
                "event": "Ignorowanie bezpośredniej odpowiedzi Muchy",
                "delta": -float(behavior.ignored_reply_affinity_step),
                "condition": (
                    f"przez {int(behavior.ignored_reply_seconds)} s brak "
                    "kontynuacji w tym kanale, ale użytkownik aktywnie "
                    "pisze gdzie indziej na serwerze"
                ),
            },
        ]
        for emoji, weight in NEGATIVE_REACTION_WEIGHT.items():
            negative.append({
                "event": f"Reakcja {emoji}",
                "delta": -float(behavior.user_affinity_negative_step)
                * float(weight),
                "condition": "reakcja pod wiadomością Muchy",
            })

        verbal = []
        seen: set[tuple[str, float]] = set()
        for _, label, severity in VERBAL_REJECTION_PATTERNS:
            key = (label, float(severity))
            if key in seen:
                continue
            seen.add(key)
            delta = -min(
                0.30,
                max(
                    0.01,
                    float(behavior.user_affinity_negative_step)
                    * (0.45 + 1.05 * float(severity)),
                ),
            )
            verbal.append({
                "phrase": label,
                "severity": float(severity),
                "delta": delta,
            })
        verbal.sort(
            key=lambda row: (row["severity"], row["phrase"]),
            reverse=True,
        )

        return {
            "positive": positive,
            "negative": negative,
            "verbal_rejections": verbal,
            "positive_cooldown_seconds": int(
                behavior.positive_contact_cooldown_seconds
            ),
            "negative_cooldown_seconds": int(
                behavior.negative_contact_cooldown_seconds
            ),
            "negative_streak": {
                "window_seconds": int(
                    behavior.negative_streak_window_seconds
                ),
                "step": float(
                    behavior.negative_streak_multiplier_step
                ),
                "max_multiplier": float(
                    behavior.negative_streak_max_multiplier
                ),
            },
            "thresholds": {
                "familiar": float(
                    behavior.familiar_affinity_threshold
                ),
                "liked": 0.35,
                "avoid": float(behavior.user_avoid_threshold),
            },
            "avoid_effects": [
                "nie odpisuje użytkownikowi",
                "nie reaguje na jego wiadomości",
                "omija kanały voice z tym użytkownikiem",
                "próbuje wyjść, gdy użytkownik wejdzie na jej VC",
            ],
            "chaser_exception": (
                "Podczas aktywnej ucieczki przed Chaserem social avoidance "
                "nie ogranicza wyboru kanału i naturalne kary za uciekanie z VC "
                "są pomijane."
            ),
        }

    async def _grant_positive_social(
        self,
        member: discord.Member,
        event: str,
        stimulus: str,
        affinity_delta: float,
        guild: discord.Guild,
        detail: str = "",
        source_trace: SentTrace | None = None,
        brain_reward: float = 0.0,
    ) -> float | None:
        if (
            not self.cfg.behavior.social_learning_enabled
            or member.bot
        ):
            return None

        now = time.monotonic()
        cooldown = max(
            1.0,
            float(self.cfg.behavior.positive_contact_cooldown_seconds),
        )
        if event == "VOICE_STAY":
            cooldown = max(cooldown, 600.0)
        elif event == "TTS_STAY":
            cooldown = max(cooldown, 300.0)
        cooldown_key = (member.id, event)
        last = self._social_positive_last.get(cooldown_key, 0.0)
        if now - last < cooldown:
            return None
        self._social_positive_last[cooldown_key] = now
        self._soften_negative_streak(member.id, now)

        delta = max(0.0, min(0.25, float(affinity_delta)))
        if delta <= 0.0:
            return None

        new_affinity = self.language.adjust_user_affinity(
            member.id,
            member.display_name,
            delta,
        )

        window = max(
            cooldown,
            float(self.cfg.behavior.social_window_seconds),
        )
        streak = self._social_positive_streak.get(
            member.id,
            {"count": 0, "last": 0.0},
        )
        if now - float(streak.get("last", 0.0)) > window:
            streak = {"count": 0, "last": 0.0}
        streak["count"] = int(streak.get("count", 0)) + 1
        streak["last"] = now
        self._social_positive_streak[member.id] = streak

        repeated_bonus = 0.0
        if streak["count"] >= 3 and streak["count"] % 3 == 0:
            repeated_bonus = min(0.006, max(0.002, delta * 0.4))
            new_affinity = self.language.adjust_user_affinity(
                member.id,
                member.display_name,
                repeated_bonus,
            )

        async with self._brain_lock:
            self._write_neural_social_memory(
                member.id,
                delta + repeated_bonus,
            )
            new_affinity = self._user_affinity(member.id)
            self.brain.inject(stimulus, 0.45, 112)
            self.brain.inject(
                f"{stimulus}:user:{member.id}",
                0.35,
                80,
            )
            if repeated_bonus > 0.0:
                self.brain.inject(
                    "social:repeated-positive-contact",
                    0.60,
                    128,
                )
                self.brain.inject(
                    f"social:repeated-positive-contact:user:{member.id}",
                    0.45,
                    96,
                )
            if new_affinity >= float(
                self.cfg.behavior.familiar_affinity_threshold
            ):
                self.brain.inject(
                    "social:familiar-user",
                    min(1.0, 0.35 + abs(new_affinity)),
                    128,
                )
                self.brain.inject(
                    f"social:familiar-user:{member.id}",
                    min(1.0, 0.30 + abs(new_affinity)),
                    96,
                )
            if new_affinity >= 0.35:
                self.brain.inject(
                    "social:liked-user",
                    min(1.0, new_affinity),
                    128,
                )
                self.brain.inject(
                    f"social:liked-user:{member.id}",
                    min(1.0, new_affinity),
                    96,
                )
            reward = max(0.0, min(0.20, float(brain_reward)))
            if reward > 0.0 and source_trace is not None:
                self.brain.reward(
                    reward,
                    action=source_trace.action,
                    trace=source_trace.learning_trace,
                )
            self.brain.step(1)

        if brain_reward > 0.0 and source_trace is not None:
            self._record_reward(
                min(0.20, float(brain_reward)),
                source_trace.action,
                event.lower(),
                guild,
            )

        suffix = (
            f" • streak {streak['count']}"
            + (
                f" • bonus +{repeated_bonus:.3f}"
                if repeated_bonus > 0.0
                else ""
            )
        )
        self._record_action(
            "positive_social",
            (
                f"{event} • {member.display_name} • "
                f"affinity {new_affinity:+.3f}{suffix}"
            ),
            guild,
        )
        self._remember_social_event(
            event,
            (
                (detail or stimulus)
                + f" • affinity {new_affinity:+.3f}"
                + suffix
            ),
            delta + repeated_bonus,
            member,
        )
        return new_affinity

    def _schedule_voice_social_stay(
        self,
        guild: discord.Guild,
        member: discord.Member,
        channel_id: int,
    ) -> None:
        key = (guild.id, member.id)
        existing = self._voice_social_stay_tasks.get(key)
        if existing is not None and not existing.done():
            return
        task = asyncio.create_task(
            self._voice_social_stay_after_delay(
                guild.id,
                member.id,
                int(channel_id),
            )
        )
        self._voice_social_stay_tasks[key] = task

        def clear(done_task: asyncio.Task, task_key=key) -> None:
            if self._voice_social_stay_tasks.get(task_key) is done_task:
                self._voice_social_stay_tasks.pop(task_key, None)

        task.add_done_callback(clear)

    async def _voice_social_stay_after_delay(
        self,
        guild_id: int,
        user_id: int,
        channel_id: int,
    ) -> None:
        await asyncio.sleep(
            max(5, int(self.cfg.behavior.voice_stay_seconds))
        )
        guild = self.get_guild(guild_id)
        if guild is None or self._chaser_panic_remaining(guild_id) > 0.0:
            return
        member = guild.get_member(user_id)
        vc = guild.voice_client
        if (
            member is None
            or member.voice is None
            or member.voice.channel is None
            or vc is None
            or not vc.is_connected()
            or vc.channel is None
            or member.voice.channel.id != channel_id
            or vc.channel.id != channel_id
        ):
            return
        await self._grant_positive_social(
            member,
            "VOICE_STAY",
            "social:user-stayed-with-me",
            self.cfg.behavior.voice_stay_affinity_step,
            guild,
            detail=f"{vc.channel.name} • {self.cfg.behavior.voice_stay_seconds}s",
        )

    def _schedule_tts_social_stay(
        self,
        guild: discord.Guild,
        channel_id: int,
        user_ids: list[int],
    ) -> None:
        for user_id in user_ids:
            key = (guild.id, int(user_id))
            existing = self._tts_social_stay_tasks.get(key)
            if existing is not None and not existing.done():
                continue
            task = asyncio.create_task(
                self._tts_social_stay_after_delay(
                    guild.id,
                    int(user_id),
                    int(channel_id),
                )
            )
            self._tts_social_stay_tasks[key] = task

            def clear(done_task: asyncio.Task, task_key=key) -> None:
                if self._tts_social_stay_tasks.get(task_key) is done_task:
                    self._tts_social_stay_tasks.pop(task_key, None)

            task.add_done_callback(clear)

    async def _tts_social_stay_after_delay(
        self,
        guild_id: int,
        user_id: int,
        channel_id: int,
    ) -> None:
        await asyncio.sleep(
            max(5, int(self.cfg.behavior.tts_stay_seconds))
        )
        guild = self.get_guild(guild_id)
        if guild is None or self._chaser_panic_remaining(guild_id) > 0.0:
            return
        member = guild.get_member(user_id)
        vc = guild.voice_client
        if (
            member is None
            or member.voice is None
            or member.voice.channel is None
            or vc is None
            or not vc.is_connected()
            or vc.channel is None
            or member.voice.channel.id != channel_id
            or vc.channel.id != channel_id
        ):
            return
        await self._grant_positive_social(
            member,
            "TTS_STAY",
            "social:user-stayed-after-tts",
            self.cfg.behavior.tts_stay_affinity_step,
            guild,
            detail=f"{vc.channel.name} • został po TTS",
        )

    @staticmethod
    def _detect_verbal_rejection(text: str) -> tuple[str, float] | None:
        normalized = OnlineLanguage.normalize(text).lower()
        best: tuple[str, float] | None = None
        for pattern, label, severity in VERBAL_REJECTION_PATTERNS:
            if pattern.search(normalized):
                if best is None or severity > best[1]:
                    best = (label, float(severity))
        return best

    def _message_targets_mucha(
        self,
        message: discord.Message,
        referenced: SentTrace | None,
    ) -> bool:
        if referenced is not None:
            return True
        if self.user is not None and self.user in message.mentions:
            return True
        normalized = OnlineLanguage.normalize(message.content).lower()
        return bool(re.search(r"\bmucha\b", normalized, flags=re.UNICODE))

    async def _apply_social_message_feedback(
        self,
        message: discord.Message,
    ) -> None:
        if not self.cfg.behavior.social_learning_enabled:
            return

        window = max(
            30.0,
            float(self.cfg.behavior.social_window_seconds),
        )
        now = time.monotonic()
        reference_id = (
            message.reference.message_id
            if message.reference is not None
            else None
        )
        referenced = self.sent.get(reference_id) if reference_id else None

        verbal_rejection = self._detect_verbal_rejection(message.content)
        targeted_rejection = bool(
            verbal_rejection
            and self._message_targets_mucha(message, referenced)
        )
        if targeted_rejection and verbal_rejection is not None:
            label, severity = verbal_rejection
            streak_count, streak_multiplier = self._advance_negative_streak(
                message.author.id,
                now,
            )
            affinity_delta = -min(
                0.30,
                max(
                    0.01,
                    float(self.cfg.behavior.user_affinity_negative_step)
                    * (0.45 + 1.05 * severity)
                    * streak_multiplier,
                ),
            )
            new_affinity = self.language.adjust_user_affinity(
                message.author.id,
                message.author.display_name,
                affinity_delta,
                "negative",
            )

            source_trace = referenced
            if source_trace is None:
                recent_target = sorted(
                    (
                        trace
                        for trace in self.sent.values()
                        if trace.guild_id == message.guild.id
                        and trace.channel_id == message.channel.id
                        and now - trace.created <= min(window, 180.0)
                    ),
                    key=lambda trace: trace.created,
                    reverse=True,
                )
                source_trace = recent_target[0] if recent_target else None

            brain_penalty = -min(
                0.35,
                (0.06 + 0.24 * severity) * streak_multiplier,
            )
            async with self._brain_lock:
                self._write_neural_social_memory(
                    message.author.id,
                    affinity_delta,
                )
                new_affinity = self._user_affinity(
                    message.author.id
                )
                self.brain.inject(
                    "social:user-told-me-stop",
                    0.55 + 0.65 * severity,
                    160,
                )
                self.brain.inject(
                    "social:user-rejected-me",
                    0.50 + 0.70 * severity,
                    160,
                )
                self.brain.inject(
                    "internal:social-failure",
                    0.35 + 0.65 * severity,
                    128,
                )
                self.brain.inject(
                    f"social:user-rejected-me:user:{message.author.id}",
                    0.45 + 0.55 * severity,
                    96,
                )
                if streak_count >= 2:
                    self.brain.inject(
                        "social:repeated-rejection",
                        min(1.0, 0.52 + 0.08 * streak_count),
                        144,
                    )
                    self.brain.inject(
                        f"social:repeated-rejection:user:{message.author.id}",
                        min(1.0, 0.45 + 0.07 * streak_count),
                        96,
                    )
                if source_trace is not None:
                    self.brain.reward(
                        brain_penalty,
                        action=source_trace.action,
                        trace=source_trace.learning_trace,
                    )
                else:
                    self.brain.reward(brain_penalty)
                self.brain.step(2)

            self._record_reward(
                brain_penalty,
                source_trace.action if source_trace is not None else None,
                f"verbal rejection • {label}",
                message.guild,
            )
            self._record_action(
                "verbal_rejection",
                (
                    f"{message.author.display_name} • {label} • "
                    f"severity {severity:.2f} • streak {streak_count} "
                    f"×{streak_multiplier:.2f} • affinity {new_affinity:+.2f}"
                ),
                message.guild,
            )
            self._remember_social_event(
                "VERBAL_REJECTION",
                (
                    f"{label} • streak {streak_count} "
                    f"×{streak_multiplier:.2f} • affinity {new_affinity:+.2f}"
                ),
                affinity_delta,
                message.author,
            )
            return

        if (
            not targeted_rejection
            and referenced is not None
            and referenced.guild_id == message.guild.id
            and now - referenced.created <= window
        ):
            amount = max(
                0.0,
                min(1.0, float(self.cfg.behavior.direct_reply_reward)),
            )
            if amount > 0.0:
                if referenced.text:
                    self.language.reinforce_text(
                        referenced.text,
                        min(0.20, amount * 0.50),
                    )
                async with self._brain_lock:
                    self.brain.inject(
                        "social:direct-reply",
                        0.70,
                        128,
                    )
                    self.brain.inject(
                        f"social:direct-reply:user:{message.author.id}",
                        0.55,
                        96,
                    )
                    self.brain.reward(
                        amount,
                        action=referenced.action,
                        trace=referenced.learning_trace,
                    )
                    self.brain.step(1)
                self._record_reward(
                    amount,
                    referenced.action,
                    f"direct reply • {message.author.display_name}",
                    message.guild,
                )
                self._record_action(
                    "social_reply",
                    f"+{amount:.2f} • {message.author.display_name}",
                    message.guild,
                )
                self._remember_social_event(
                    "DIRECT_REPLY",
                    message.author.display_name,
                    amount,
                    message.author,
                )

        if (
            not targeted_rejection
            and referenced is not None
            and referenced.guild_id == message.guild.id
            and now - referenced.created <= window
        ):
            await self._grant_positive_social(
                message.author,
                "DIRECT_REPLY_AFFINITY",
                "social:user-replied",
                self.cfg.behavior.direct_reply_affinity_step,
                message.guild,
                detail="bezpośredni reply do Muchy",
            )

        if (
            not targeted_rejection
            and self.user is not None
            and self.user in message.mentions
        ):
            await self._grant_positive_social(
                message.author,
                "MENTION",
                "social:user-mentioned-me",
                self.cfg.behavior.mention_affinity_step,
                message.guild,
                detail="wspomniał Muchę",
            )

        if not targeted_rejection and referenced is None:
            recent_conversation = sorted(
                (
                    trace
                    for trace in self.sent.values()
                    if trace.guild_id == message.guild.id
                    and trace.channel_id == message.channel.id
                    and now - trace.created <= min(window, 120.0)
                ),
                key=lambda trace: trace.created,
                reverse=True,
            )
            if recent_conversation:
                source_trace = recent_conversation[0]
                await self._grant_positive_social(
                    message.author,
                    "CONTINUED_CONVERSATION",
                    "social:user-continued-conversation",
                    self.cfg.behavior.continued_conversation_affinity_step,
                    message.guild,
                    detail="kontynuował rozmowę po wypowiedzi Muchy",
                    source_trace=source_trace,
                    brain_reward=0.025,
                )

        normalized_message = OnlineLanguage.normalize(
            message.content
        ).lower()
        correction_match = re.search(
            r'\bnie\s+["„]?([^\s"”„,.;:!?]{2,})["”]?'
            r'\s*,?\s*(?:tylko|ale)\s+'
            r'["„]?([^\s"”„,.;:!?]{2,})["”]?',
            normalized_message,
            flags=re.UNICODE,
        )
        if correction_match:
            wrong_word = correction_match.group(1)
            right_word = correction_match.group(2)
            recent_for_correction = sorted(
                (
                    trace
                    for trace in self.sent.values()
                    if trace.guild_id == message.guild.id
                    and trace.channel_id == message.channel.id
                    and now - trace.created <= window
                    and wrong_word in self._social_words(trace.text)
                ),
                key=lambda trace: trace.created,
                reverse=True,
            )
            if recent_for_correction:
                source_trace = recent_for_correction[0]
                self.language.reinforce_text(wrong_word, -0.30)
                self.language.reinforce_text(right_word, 0.50)
                self.language.record_word_feedback(
                    right_word,
                    message.author.id,
                    0.50,
                )
                correction_reward = -0.10
                async with self._brain_lock:
                    self.brain.inject(
                        "social:correction",
                        0.85,
                        144,
                    )
                    self.brain.inject(
                        f"social:correction:user:{message.author.id}",
                        0.60,
                        96,
                    )
                    self.brain.reward(
                        correction_reward,
                        action=source_trace.action,
                        trace=source_trace.learning_trace,
                    )
                    self.brain.step(1)
                detail = f"{wrong_word} → {right_word}"
                self._record_reward(
                    correction_reward,
                    source_trace.action,
                    f"social:correction • {detail}",
                    message.guild,
                )
                self._record_action(
                    "social_correction",
                    (
                        f"{message.author.display_name} • {detail} • "
                        "lang -0.30/+0.50"
                    ),
                    message.guild,
                )
                self._remember_social_event(
                    "CORRECTION",
                    detail,
                    0.50,
                    message.author,
                )
                return

        user_words = self._social_words(message.content)
        if not user_words:
            return
        user_word_set = set(user_words)
        user_phrases = {
            tuple(user_words[i:i + size])
            for size in (2, 3)
            for i in range(max(0, len(user_words) - size + 1))
        }

        recent = sorted(
            (
                trace
                for trace in self.sent.values()
                if trace.guild_id == message.guild.id
                and trace.channel_id == message.channel.id
                and now - trace.created <= window
            ),
            key=lambda trace: trace.created,
            reverse=True,
        )[:12]

        matched_trace = None
        matched_phrase: tuple[str, ...] | None = None
        matched_word: str | None = None

        for trace in recent:
            sent_words = self._social_words(trace.text)
            for size in (3, 2):
                for i in range(max(0, len(sent_words) - size + 1)):
                    phrase = tuple(sent_words[i:i + size])
                    if (
                        phrase in user_phrases
                        and sum(len(x) for x in phrase) >= 8
                    ):
                        matched_trace = trace
                        matched_phrase = phrase
                        break
                if matched_phrase is not None:
                    break
            if matched_phrase is not None:
                break

            shared = [
                word
                for word in set(sent_words) & user_word_set
                if len(word) >= 5
            ]
            if shared:
                matched_trace = trace
                matched_word = max(shared, key=len)
                break

        if matched_trace is None:
            return

        if matched_phrase is not None:
            phrase_text = " ".join(matched_phrase)
            language_amount = max(
                0.0,
                min(1.0, float(self.cfg.behavior.phrase_reuse_reward)),
            )
            self.language.reinforce_text(phrase_text, language_amount)
            for word in matched_phrase:
                if len(word) >= 4:
                    self.language.record_word_feedback(
                        word,
                        message.author.id,
                        language_amount * 0.5,
                    )
            brain_amount = min(0.20, language_amount * 0.35)
            affinity_delta = float(
                self.cfg.behavior.phrase_reuse_affinity_step
            )
            event = "PHRASE_REUSE"
            detail = phrase_text
        else:
            language_amount = max(
                0.0,
                min(1.0, float(self.cfg.behavior.word_reuse_reward)),
            )
            self.language.reinforce_text(matched_word or "", language_amount)
            feedback_info = {}
            if matched_word:
                feedback_info = self.language.record_word_feedback(
                    matched_word,
                    message.author.id,
                    language_amount,
                )
            unique_users = int(feedback_info.get("unique_users", 1))
            confirmation_bonus = min(
                0.10,
                max(0, unique_users - 1) * 0.02,
            )
            if confirmation_bonus > 0.0 and matched_word:
                self.language.reinforce_text(
                    matched_word,
                    confirmation_bonus,
                )
            brain_amount = min(
                0.20,
                language_amount * 0.35 + confirmation_bonus * 0.5,
            )
            event = (
                "MULTI_USER_CONFIRM"
                if unique_users >= 2
                else "WORD_REUSE"
            )
            affinity_delta = float(
                self.cfg.behavior.word_reuse_affinity_step
            ) + min(0.010, max(0, unique_users - 1) * 0.002)
            detail = (
                f"{matched_word} • {unique_users} osób"
                if matched_word
                else ""
            )

        stimulus = {
            "WORD_REUSE": "social:word-reused",
            "PHRASE_REUSE": "social:phrase-reused",
            "MULTI_USER_CONFIRM": "social:multi-user-confirm",
        }.get(event, f"social:{event.lower().replace('_', '-')}")
        if brain_amount > 0.0:
            async with self._brain_lock:
                self.brain.inject(
                    stimulus,
                    0.65,
                    128,
                )
                self.brain.inject(
                    f"{stimulus}:user:{message.author.id}",
                    0.45,
                    96,
                )
                self.brain.reward(
                    brain_amount,
                    action=matched_trace.action,
                    trace=matched_trace.learning_trace,
                )
                self.brain.step(1)

        self._record_reward(
            brain_amount,
            matched_trace.action,
            f"{event.lower()} • {detail}",
            message.guild,
        )
        await self._grant_positive_social(
            message.author,
            event + "_AFFINITY",
            {
                "WORD_REUSE": "social:user-reused-word",
                "PHRASE_REUSE": "social:user-reused-phrase",
                "MULTI_USER_CONFIRM": "social:repeated-positive-contact",
            }.get(event, "social:positive-contact"),
            affinity_delta,
            message.guild,
            detail=detail,
        )
        self._record_action(
            "social_learn",
            (
                f"{event} • {message.author.display_name} • "
                f"{detail} • lang +{language_amount:.2f}"
            ),
            message.guild,
        )
        self._remember_social_event(
            event,
            detail,
            language_amount,
            message.author,
        )

    def _is_text_channel_blocked(self, channel: object) -> bool:
        channel_id = getattr(channel, "id", None)
        if channel_id is None:
            return False
        channel_id = int(channel_id)
        return (
            channel_id in self.HARD_BLOCKED_TEXT_CHANNEL_IDS
            or channel_id in self.cfg.discord.blocked_text_channel_ids
        )

    def _is_voice_channel_blocked(self, channel: object) -> bool:
        channel_id = getattr(channel, "id", None)
        if channel_id is None:
            return False
        return int(channel_id) in self.cfg.voice.blocked_voice_channel_ids

    def _is_voice_guild_blocked(self, guild: object) -> bool:
        guild_id = getattr(guild, "id", None)
        if guild_id is None:
            return False
        return int(guild_id) in self.cfg.voice.blocked_voice_guild_ids

    def _deadly_voice_remaining(
        self,
        guild_id: int,
        channel_id: int,
        now: float | None = None,
    ) -> float:
        now = time.monotonic() if now is None else now
        key = (int(guild_id), int(channel_id))
        expiry = self._deadly_voice_until.get(key, 0.0)
        remaining = max(0.0, expiry - now)
        if remaining <= 0.0:
            self._deadly_voice_until.pop(key, None)
            return 0.0
        return remaining

    def _mark_deadly_voice_channel(
        self,
        guild_id: int,
        channel_id: int,
        now: float | None = None,
        seconds: float | None = None,
    ) -> float:
        now = time.monotonic() if now is None else now
        duration = max(
            1.0,
            float(
                self.cfg.voice.deadly_channel_seconds
                if seconds is None
                else seconds
            ),
        )
        key = (int(guild_id), int(channel_id))
        expiry = max(
            self._deadly_voice_until.get(key, 0.0),
            now + duration,
        )
        self._deadly_voice_until[key] = expiry
        return expiry

    def _chaser_panic_remaining(
        self,
        guild_id: int,
        now: float | None = None,
    ) -> float:
        now = time.monotonic() if now is None else now
        expiry = self._chaser_panic_until.get(int(guild_id), 0.0)
        remaining = max(0.0, expiry - now)
        if remaining <= 0.0:
            self._chaser_panic_until.pop(int(guild_id), None)
            return 0.0
        return remaining

    def _current_voice_channel(
        self,
        guild: discord.Guild,
    ) -> discord.VoiceChannel | discord.StageChannel | None:
        """Return Mucha's current voice channel, preferring gateway state.

        During VoiceRecvClient reconnects, VoiceClient.channel can briefly lag
        behind Discord's guild member voice state. Chaser logic needs the
        gateway state first so a catch is not missed after an escape.
        """
        me = guild.me
        gateway_channel = (
            getattr(getattr(me, "voice", None), "channel", None)
            if me is not None
            else None
        )
        if gateway_channel is not None:
            return gateway_channel

        vc = guild.voice_client
        if vc is not None and vc.is_connected():
            return vc.channel
        return None

    def _chaser_is_named(self, member: discord.Member) -> bool:
        configured = int(self.cfg.voice.chaser_bot_id)
        if configured > 0 and member.id == configured:
            return True
        hint = self.cfg.voice.chaser_name_hint.strip().lower()
        if not hint:
            return False
        return hint in member.display_name.lower() or hint in member.name.lower()

    def _register_chaser_encounter(
        self,
        member: discord.Member,
        channel: discord.VoiceChannel,
        now: float,
    ) -> tuple[bool, int]:
        key = (member.guild.id, member.id)
        state = self._chaser_follow_state.get(key, {})
        last = float(state.get("last", 0.0))
        hits = int(state.get("hits", 0))
        window = max(
            1.0,
            float(self.cfg.voice.chaser_follow_window_seconds),
        )
        if now - last > window:
            hits = 0
        hits += 1
        self._chaser_follow_state[key] = {
            "hits": hits,
            "last": now,
            "channel_id": channel.id,
        }

        already_confirmed = (
            self._chaser_confirmed.get(member.guild.id) == member.id
        )
        confirmed = bool(
            already_confirmed
            or self._chaser_is_named(member)
            or hits >= max(1, int(self.cfg.voice.chaser_confirm_hits))
        )
        if confirmed:
            self._chaser_confirmed[member.guild.id] = member.id

        panic_seconds = (
            float(self.cfg.voice.chaser_panic_seconds)
            if confirmed
            else float(self.cfg.voice.chaser_suspicion_seconds)
        )
        self._chaser_panic_until[member.guild.id] = max(
            self._chaser_panic_until.get(member.guild.id, 0.0),
            now + max(1.0, panic_seconds),
        )
        self._mark_deadly_voice_channel(
            member.guild.id,
            channel.id,
            now,
            seconds=float(self.cfg.voice.chaser_channel_avoid_seconds),
        )
        return confirmed, hits

    def _schedule_chaser_escape(
        self,
        guild: discord.Guild,
        predator_id: int,
        learning_trace: tuple,
    ) -> None:
        existing = self._chaser_escape_tasks.get(guild.id)
        if existing is not None and not existing.done():
            return

        task = asyncio.create_task(
            self._escape_from_chaser(
                guild.id,
                predator_id,
                learning_trace,
            )
        )
        self._chaser_escape_tasks[guild.id] = task

        def clear(done_task: asyncio.Task, guild_id: int = guild.id) -> None:
            if self._chaser_escape_tasks.get(guild_id) is done_task:
                self._chaser_escape_tasks.pop(guild_id, None)

        task.add_done_callback(clear)

    async def _escape_from_chaser(
        self,
        guild_id: int,
        predator_id: int,
        learning_trace: tuple,
    ) -> None:
        """Keep evading the chaser for the whole panic window.

        A single task owns the chase. After every successful move it waits for
        the predator to catch up again and immediately chooses another channel.
        If there is nowhere else to run, Mucha disconnects from voice and stays
        out until the panic window expires.
        """
        delay_min = max(
            0.0,
            float(self.cfg.voice.chaser_escape_delay_min_seconds),
        )
        delay_max = max(
            delay_min,
            float(self.cfg.voice.chaser_escape_delay_max_seconds),
        )
        await asyncio.sleep(self.random.uniform(delay_min, delay_max))

        first_escape = True
        predator_missing_since: float | None = None
        while self._chaser_panic_remaining(guild_id) > 0.0:
            guild = self.get_guild(guild_id)
            if guild is None or self._is_voice_guild_blocked(guild):
                return

            vc = guild.voice_client
            me = guild.me
            if me is None:
                return
            current = self._current_voice_channel(guild)
            if (
                vc is None
                or not vc.is_connected()
                or current is None
            ):
                # Voice transport can disappear briefly during a Discord move
                # or reconnect. Do not kill the chase task in that window.
                await asyncio.sleep(0.15)
                continue
            predator = guild.get_member(predator_id)
            predator_channel = (
                getattr(getattr(predator, "voice", None), "channel", None)
                if predator is not None
                else None
            )

            now = time.monotonic()
            if predator_channel is None:
                if predator_missing_since is None:
                    predator_missing_since = now
                if now - predator_missing_since >= 2.5:
                    self._chaser_panic_until.pop(guild_id, None)
                    if (
                        self._audio_debug.get("stage") in {
                            "chaser_scream",
                            "playing_check",
                        }
                        and vc.is_playing()
                    ):
                        self._stop_voice_playback(vc)
                    self._audio_debug.update({
                        "status": "IDLE",
                        "stage": "chaser_finished",
                        "playing": False,
                        "connected": bool(vc.is_connected()),
                        "error": "",
                        "updated_at": time.time(),
                    })
                    self._last_brain_event = "CHASER • pościg zakończony"
                    self._last_brain_action = "PANIC END"
                    self._record_action(
                        "chaser_end",
                        f"predator {predator_id} opuścił voice",
                        guild,
                    )
                    log.info(
                        "CHASER END guild=%s predator=%s reason=left_voice",
                        guild.id,
                        predator_id,
                    )
                    return
                await asyncio.sleep(0.10)
                continue

            predator_missing_since = None
            # Keep the panic alive while the predator is genuinely present on
            # voice. The chase therefore lasts as long as the chaser does,
            # instead of expiring halfway through a long pursuit.
            self._chaser_panic_until[guild_id] = max(
                self._chaser_panic_until.get(guild_id, 0.0),
                now + 5.0,
            )

            # Do not hop endlessly on our own. Wait until the chaser is
            # actually on the same channel again.
            if predator_channel.id != current.id:
                await asyncio.sleep(0.10)
                continue

            now = time.monotonic()
            self._mark_deadly_voice_channel(
                guild.id,
                current.id,
                now,
                seconds=float(self.cfg.voice.chaser_channel_avoid_seconds),
            )

            clean: list[
                tuple[discord.VoiceChannel, list[discord.Member]]
            ] = []
            fallback: list[
                tuple[discord.VoiceChannel, list[discord.Member]]
            ] = []

            for ch in guild.voice_channels:
                if ch.id == current.id:
                    continue
                if self._is_voice_channel_blocked(ch):
                    continue
                if (
                    self.cfg.voice.exclude_afk_channel
                    and guild.afk_channel
                    and ch.id == guild.afk_channel.id
                ):
                    continue
                perms = ch.permissions_for(me)
                if not perms.view_channel or not perms.connect:
                    continue
                if any(m.id == predator_id for m in ch.members):
                    continue

                humans = [m for m in ch.members if not m.bot]
                if not humans and not self.cfg.voice.include_empty_channels:
                    continue
                item = (ch, humans)
                fallback.append(item)
                if (
                    self._deadly_voice_remaining(
                        guild.id,
                        ch.id,
                        now,
                    )
                    <= 0.0
                ):
                    clean.append(item)

            candidates = clean or fallback
            if not candidates:
                old_name = current.name
                try:
                    self._stop_voice_playback(vc)
                    await vc.disconnect(force=True)
                    self.voice_arrived[guild.id] = now
                    self._voice_arrival_members.pop(guild.id, None)
                    self._voice_arrival_channel.pop(guild.id, None)
                    self._voice_arrival_learning.pop(guild.id, None)
                    self._last_overstay_punish.pop(guild.id, None)
                    self._set_audio_disconnected(
                        guild,
                        reason="chaser_no_escape_channel",
                    )
                    self._last_brain_event = (
                        f"CHASER • brak drogi • opuszcza {old_name}"
                    )
                    self._last_brain_action = (
                        "PANIC ESCAPE → DISCONNECT"
                    )
                    self._record_action(
                        "chaser_disconnect_escape",
                        f"← {old_name} • brak kolejnego kanału",
                        guild,
                    )
                    log.info(
                        "CHASER DISCONNECT guild=%s predator=%s from=%s reason=no_escape_channel",
                        guild.id,
                        predator_id,
                        old_name,
                    )
                except (
                    discord.Forbidden,
                    discord.HTTPException,
                    asyncio.TimeoutError,
                ) as exc:
                    self._record_action(
                        "chaser_escape_error",
                        f"disconnect {type(exc).__name__}: {exc}",
                        guild,
                    )
                return

            async with self._brain_lock:
                affinities = {
                    ch.id: self.brain.channel_affinity(guild.id, ch.id)
                    for ch, _ in candidates
                }

            target, _ = self._choose_voice_target(
                guild,
                candidates,
                affinities,
                now,
                current_id=current.id,
            )
            if target is None:
                # A filtered target list can still fail selection. Treat it as
                # a trap instead of freezing on the chaser's channel.
                old_name = current.name
                try:
                    self._stop_voice_playback(vc)
                    await vc.disconnect(force=True)
                    self.voice_arrived[guild.id] = now
                    self._set_audio_disconnected(
                        guild,
                        reason="chaser_target_selection_failed",
                    )
                    self._last_brain_event = (
                        f"CHASER • brak celu • opuszcza {old_name}"
                    )
                    self._last_brain_action = (
                        "PANIC ESCAPE → DISCONNECT"
                    )
                    self._record_action(
                        "chaser_disconnect_escape",
                        f"← {old_name} • selector bez celu",
                        guild,
                    )
                    log.info(
                        "CHASER DISCONNECT guild=%s predator=%s from=%s reason=no_target",
                        guild.id,
                        predator_id,
                        old_name,
                    )
                except (
                    discord.Forbidden,
                    discord.HTTPException,
                    asyncio.TimeoutError,
                ) as exc:
                    self._record_action(
                        "chaser_escape_error",
                        f"disconnect {type(exc).__name__}: {exc}",
                        guild,
                    )
                return

            try:
                await vc.move_to(target)
                self.voice_arrived[guild.id] = now
                self._mark_voice_visit(guild.id, target.id, now)

                reward = max(
                    0.0,
                    min(
                        1.0,
                        float(self.cfg.voice.chaser_escape_reward),
                    ),
                )
                async with self._brain_lock:
                    trace = (
                        learning_trace
                        if first_escape
                        else self.brain.capture_learning_trace()
                    )
                    if reward > 0.0:
                        self.brain.reward(
                            reward,
                            action="voice_move",
                            trace=trace,
                        )
                        self.brain.step(1)

                first_escape = False
                self._last_brain_event = (
                    f"CHASER • ucieczka {current.name} → {target.name}"
                )
                self._last_brain_action = (
                    f"PANIC ESCAPE → {target.name}"
                )
                self._set_reinforceable(
                    guild,
                    "voice_move",
                    trace,
                    f"chaser escape {current.name} → {target.name}",
                )
                self._record_action(
                    "chaser_escape",
                    f"{current.name} → {target.name}",
                    guild,
                )
                log.info(
                    "CHASER ESCAPE guild=%s predator=%s %s -> %s",
                    guild.id,
                    predator_id,
                    current.name,
                    target.name,
                )
                self._ensure_chaser_scream_loop(guild)
                if reward > 0.0:
                    self._record_reward(
                        reward,
                        "voice_move",
                        "Mucha Chaser escape",
                        guild,
                    )

                # Give Discord a moment to publish both bots' voice states.
                await asyncio.sleep(0.12)
            except (
                discord.Forbidden,
                discord.HTTPException,
                asyncio.TimeoutError,
            ) as exc:
                self._record_action(
                    "chaser_escape_error",
                    f"{type(exc).__name__}: {exc}",
                    guild,
                )
                await asyncio.sleep(0.20)

    def _ensure_chaser_scream_loop(
        self,
        guild: discord.Guild,
    ) -> None:
        if not self.cfg.voice.chaser_scream_enabled:
            return

        existing = self._chaser_scream_tasks.get(guild.id)
        if existing is not None and not existing.done():
            return

        task = asyncio.create_task(
            self._chaser_scream_loop(guild.id)
        )
        self._chaser_scream_tasks[guild.id] = task

        def clear(
            done_task: asyncio.Task,
            guild_id: int = guild.id,
        ) -> None:
            if self._chaser_scream_tasks.get(guild_id) is done_task:
                self._chaser_scream_tasks.pop(guild_id, None)

        task.add_done_callback(clear)

    async def _chaser_scream_loop(self, guild_id: int) -> None:
        """Play panic audio only while the confirmed chaser is with Mucha."""
        while True:
            if self._chaser_panic_remaining(guild_id) <= 0.0:
                return

            guild = self.get_guild(guild_id)
            if guild is None:
                return

            vc = guild.voice_client
            if (
                vc is None
                or not vc.is_connected()
                or vc.channel is None
            ):
                self._set_audio_disconnected(
                    guild,
                    reason="chaser_escape_disconnected",
                )
                await asyncio.sleep(0.10)
                continue

            predator_id = self._chaser_confirmed.get(guild_id)
            predator = (
                guild.get_member(predator_id)
                if predator_id is not None
                else None
            )
            predator_channel = getattr(
                getattr(predator, "voice", None),
                "channel",
                None,
            )

            same_channel = bool(
                predator_channel is not None
                and predator_channel.id == vc.channel.id
            )
            if not same_channel:
                if (
                    self._audio_debug.get("stage") in {
                        "chaser_scream",
                        "playing_check",
                    }
                    and vc.is_playing()
                ):
                    self._stop_voice_playback(vc)
                    self._audio_debug.update({
                        "status": "IDLE",
                        "stage": "chaser_clear",
                        "playing": False,
                        "connected": True,
                        "error": "",
                        "updated_at": time.time(),
                    })
                await asyncio.sleep(0.10)
                continue

            if not self.cfg.voice.chaser_scream_enabled:
                return

            if vc.is_playing():
                await asyncio.sleep(0.05)
                continue

            await self._play_chaser_scream(guild, vc)
            await asyncio.sleep(0.05)

    def _mark_voice_visit(
        self,
        guild_id: int,
        channel_id: int,
        now: float | None = None,
    ) -> None:
        now = time.monotonic() if now is None else now
        self._voice_last_visit[(int(guild_id), int(channel_id))] = now

    @staticmethod
    def _voice_prediction_context(
        *,
        connected: bool,
        social_need: float,
        social_fatigue: float,
        habituation: float,
        exploration: float,
        human_count: int,
        alternatives: int,
    ) -> str:
        def bucket(value: float) -> int:
            return max(
                0,
                min(3, int(float(value) * 4.0)),
            )

        return (
            f"{'in' if connected else 'out'}"
            f"|need={bucket(social_need)}"
            f"|fatigue={bucket(social_fatigue)}"
            f"|hab={bucket(habituation)}"
            f"|explore={bucket(exploration)}"
            f"|humans={min(4, max(0, int(human_count)))}"
            f"|alts={min(4, max(0, int(alternatives)))}"
        )

    def _voice_homeostasis_levels(
        self,
        guild_id: int,
        current_channel_id: int | None,
        current_user_ids: list[int],
        *,
        now: float,
        dwell_elapsed: float,
        outside_seconds: float,
        alternatives: int,
    ) -> dict:
        """Derive slow internal drives without directly selecting an action."""
        guild_id = int(guild_id)
        current_users = tuple(sorted(int(x) for x in current_user_ids))
        scene_key = (
            (int(current_channel_id), current_users)
            if current_channel_id is not None
            else None
        )
        state = self._voice_homeostasis_state.get(guild_id)
        if state is None or state.get("scene_key") != scene_key:
            state = {
                "scene_key": scene_key,
                "scene_since": float(now),
                "changed_at": float(now),
            }
            self._voice_homeostasis_state[guild_id] = state

        scene_age = (
            max(0.0, float(now) - float(state["scene_since"]))
            if scene_key is not None
            else 0.0
        )
        connected = current_channel_id is not None
        homeostasis = bool(self.cfg.voice.homeostasis_enabled)

        social_fatigue = 0.0
        if homeostasis and connected and current_users:
            start = max(
                0.0,
                float(self.cfg.voice.social_fatigue_start_seconds),
            )
            ramp = max(
                1.0,
                float(self.cfg.voice.social_fatigue_ramp_seconds),
            )
            social_fatigue = max(
                0.0,
                min(1.0, (dwell_elapsed - start) / ramp),
            )

        habituation = 0.0
        if (
            homeostasis
            and self.cfg.voice.habituation_enabled
            and connected
        ):
            half_life = max(
                1.0,
                float(self.cfg.voice.habituation_half_life_seconds),
            )
            habituation = max(
                0.0,
                min(1.0, 1.0 - (2.0 ** (-scene_age / half_life))),
            )

        exploration = 0.0
        if (
            homeostasis
            and self.cfg.voice.exploration_drive_enabled
            and connected
            and alternatives > 0
        ):
            start = max(
                0.0,
                float(self.cfg.voice.exploration_drive_start_seconds),
            )
            ramp = max(
                1.0,
                float(self.cfg.voice.exploration_drive_ramp_seconds),
            )
            exploration = max(
                0.0,
                min(1.0, (scene_age - start) / ramp),
            )

        suppression = (
            habituation
            * max(
                0.0,
                min(
                    0.95,
                    float(self.cfg.voice.habituation_max_suppression),
                ),
            )
        )
        return {
            "scene_age": scene_age,
            "social_need": (
                max(0.0, float(outside_seconds))
                if not connected
                else 0.0
            ),
            "social_fatigue": social_fatigue,
            "habituation": habituation,
            "habituation_suppression": suppression,
            "exploration": exploration,
        }

    def _update_pending_voice_scene(
        self,
        guild_id: int,
        channel: discord.VoiceChannel,
    ) -> None:
        queue = self._voice_prediction_pending.get(
            int(guild_id),
            [],
        )
        if not queue:
            return
        pending = queue[-1]
        humans = [
            member
            for member in channel.members
            if not member.bot
        ]
        user_ids = sorted(int(member.id) for member in humans)
        pending["channel_id"] = int(channel.id)
        pending["channel_name"] = str(channel.name)
        pending["user_ids"] = user_ids
        pending["user_names"] = [
            member.display_name
            for member in sorted(
                humans,
                key=lambda item: int(item.id),
            )
        ]
        pending["scene_key"] = (
            self.voice_episodes.make_scene_key(
                channel.id,
                user_ids,
            )
        )

    def _voice_reward_opportunity_for(
        self,
        guild_id: int,
        candidates: list[
            tuple[discord.VoiceChannel, list[discord.Member]]
        ],
        now: float,
    ) -> dict | None:
        if not self.cfg.voice.reward_opportunity_enabled:
            self._voice_reward_opportunity.pop(int(guild_id), None)
            return None

        social_candidates = [
            (channel, humans)
            for channel, humans in candidates
            if humans
        ]
        if not social_candidates:
            self._voice_reward_opportunity.pop(int(guild_id), None)
            return None

        valid_ids = {
            int(channel.id)
            for channel, _ in social_candidates
        }
        current = self._voice_reward_opportunity.get(
            int(guild_id)
        )
        if (
            current is not None
            and float(current.get("expires_at", 0.0)) > now
            and int(current.get("channel_id", 0)) in valid_ids
        ):
            return current

        channel, humans = self.random.choice(
            social_candidates
        )
        low = max(
            0.0,
            min(
                4.0,
                float(
                    self.cfg.voice.reward_opportunity_min_strength
                ),
            ),
        )
        high = max(
            low,
            min(
                4.0,
                float(
                    self.cfg.voice.reward_opportunity_max_strength
                ),
            ),
        )
        ttl = max(
            10.0,
            float(
                self.cfg.voice.reward_opportunity_ttl_seconds
            ),
        )
        opportunity = {
            "channel_id": int(channel.id),
            "channel_name": str(channel.name),
            "human_count": int(len(humans)),
            "strength": float(
                self.random.uniform(low, high)
            ),
            "created_at": now,
            "expires_at": now + ttl,
        }
        self._voice_reward_opportunity[
            int(guild_id)
        ] = opportunity
        return opportunity

    def _voice_exploration_score(
        self,
        guild_id: int,
        channel_id: int,
        affinity: float,
        now: float,
    ) -> tuple[float, float | None, float, float]:
        memory = max(1.0, float(self.cfg.voice.exploration_memory_seconds))
        last = self._voice_last_visit.get((int(guild_id), int(channel_id)))
        if last is None:
            age = None
            novelty = 1.0
            recent = 0.0
        else:
            age = max(0.0, now - last)
            novelty = min(1.0, age / memory)
            recent = max(0.0, 1.0 - age / memory)

        score = (
            float(affinity)
            + float(self.cfg.voice.exploration_novelty_bonus) * novelty
            - float(self.cfg.voice.exploration_recent_penalty) * recent
        )
        return score, age, novelty, recent

    def _choose_voice_target(
        self,
        guild: discord.Guild,
        candidates: list[tuple[discord.VoiceChannel, list[discord.Member]]],
        affinities: dict[int, float],
        now: float,
        current_id: int | None = None,
        preferred_channel_id: int | None = None,
    ) -> tuple[discord.VoiceChannel | None, dict[int, dict]]:
        scored: list[tuple[discord.VoiceChannel, float]] = []
        debug_scores: dict[int, dict] = {}

        for ch, _ in candidates:
            if current_id is not None and ch.id == current_id:
                continue
            affinity = float(affinities.get(ch.id, 0.5))
            score, age, novelty, recent = self._voice_exploration_score(
                guild.id,
                ch.id,
                affinity,
                now,
            )
            scored.append((ch, score))
            debug_scores[ch.id] = {
                "exploration_score": score,
                "visit_age": age,
                "novelty": novelty,
                "recent_penalty_factor": recent,
            }

        if not scored:
            return None, debug_scores

        if preferred_channel_id is not None:
            preferred_channel_id = int(preferred_channel_id)
            for channel, _score in scored:
                if int(channel.id) == preferred_channel_id:
                    debug_scores[channel.id][
                        "reward_opportunity"
                    ] = True
                    return channel, debug_scores

        # Softmax-like sampling: affinity still matters, but fresh/rarely visited
        # channels can win instead of repeatedly selecting the same deterministic max.
        temperature = max(
            0.03,
            float(self.cfg.voice.exploration_temperature),
        )
        best = max(score for _, score in scored)
        weights = [
            math.exp(max(-20.0, min(20.0, (score - best) / temperature)))
            for _, score in scored
        ]

        # If there are many options, guarantee that several candidates retain a
        # meaningful chance instead of collapsing onto one or two channels.
        min_candidates = max(1, int(self.cfg.voice.exploration_min_candidates))
        if len(scored) >= min_candidates:
            floor = max(weights) * 0.08
            weights = [max(w, floor) for w in weights]

        total = sum(weights)
        pick = self.random.random() * total
        upto = 0.0
        for (ch, _), weight in zip(scored, weights):
            upto += weight
            if upto >= pick:
                return ch, debug_scores
        return scored[-1][0], debug_scores

    def _reaction_candidates(self, guild: discord.Guild) -> tuple[list[tuple[str, object]], int]:
        """Sample from the full Unicode emoji set plus usable custom guild emoji."""
        sample_size = max(8, int(self.cfg.behavior.reaction_candidate_sample))
        custom = [e for e in guild.emojis if e.available]
        custom_slots = min(len(custom), min(16, max(2, sample_size // 4)))
        unicode_slots = max(1, sample_size - custom_slots)

        if len(self._unicode_emojis) <= unicode_slots:
            unicode_sample = list(self._unicode_emojis)
        else:
            unicode_sample = self.random.sample(self._unicode_emojis, unicode_slots)

        if len(custom) <= custom_slots:
            custom_sample = custom
        else:
            custom_sample = self.random.sample(custom, custom_slots)

        candidates: list[tuple[str, object]] = [(x, x) for x in unicode_sample]
        candidates.extend((str(e), e) for e in custom_sample)
        return candidates, len(self._unicode_emojis) + len(custom)

    async def setup_hook(self) -> None:
        self.idle_loop.change_interval(seconds=self.cfg.behavior.idle_tick_seconds)
        self.voice_loop.change_interval(seconds=self.cfg.voice.poll_seconds)
        self.tts_loop.change_interval(
            seconds=max(1, self.cfg.voice.tts_interval_seconds)
        )
        self.console_loop.change_interval(seconds=max(0.25, self.cfg.console_ui.refresh_seconds))
        self.idle_loop.start()
        self.presence_loop.start()
        if self.cfg.voice.stt_enabled:
            self.stt_segment_loop.start()
        if self.cfg.voice.enabled:
            self.voice_loop.start()
            if self.cfg.voice.random_audio_enabled:
                self.random_audio_loop.start()
            if self.cfg.voice.tts_enabled:
                self.tts_loop.start()

    async def on_ready(self):
        m = self.connectome.metadata
        log.info("Zalogowano jako %s", self.user)
        log.info("Connectome: %s neuronów, %s połączeń", self.connectome.n_neurons, self.connectome.matrix.nnz)
        log.info("Źródło: %s", m.get("source", "unknown"))
        ffmpeg_cfg = self.cfg.voice.ffmpeg_executable
        ffmpeg_found = (
            str(Path(ffmpeg_cfg).resolve())
            if Path(ffmpeg_cfg).is_file()
            else shutil.which(ffmpeg_cfg)
        )
        if ffmpeg_found:
            log.info("FFmpeg audio: %s", ffmpeg_found)
            self._audio_debug.update({
                "status": "READY",
                "stage": "ffmpeg",
                "ffmpeg": ffmpeg_found,
                "error": "",
                "updated_at": time.time(),
            })
        else:
            self._audio_debug.update({
                "status": "ERROR",
                "stage": "ffmpeg",
                "ffmpeg": ffmpeg_cfg,
                "error": f"FFmpeg nie znaleziony: {ffmpeg_cfg}",
                "updated_at": time.time(),
            })
            log.error(
                "FFmpeg nie znaleziony: %s — TTS i rare audio nie zagrają",
                ffmpeg_cfg,
            )
        self.console_ui.start()
        if self.cfg.console_ui.mode != "off" and not self.console_loop.is_running():
            self.console_loop.start()
        if self.cfg.web_ui.enabled:
            try:
                await self.web_ui.start()
            except OSError:
                log.exception("Nie udało się uruchomić Web UI na %s:%s", self.cfg.web_ui.host, self.cfg.web_ui.port)
        if self.cfg.voice.stt_enabled:
            asyncio.create_task(self._warm_stt_model())
        await self._update_presence()

    async def close(self) -> None:
        try:
            if self.stt_segment_loop.is_running():
                self.stt_segment_loop.cancel()
            with self._stt_buffer_lock:
                self._stt_buffers.clear()
            await self.web_ui.stop()
            self.console_ui.stop()
            self.brain.save()
            self.language.close()
        finally:
            await super().close()

    async def on_message(self, message: discord.Message):
        if message.guild is None or message.author.id == self.user.id:
            return
        self._last_external_activity = time.monotonic()
        if message.content.startswith(self.cfg.discord.command_prefix):
            await self._admin_command(message)
            return
        if self.paused:
            return
        if message.author.bot and not self.cfg.language.learn_from_bots:
            return

        if not message.author.bot:
            self._note_direct_reply_activity(message)

        blocked_text = self._is_text_channel_blocked(message.channel)
        if not blocked_text:
            self.last_text_channel[message.guild.id] = message.channel.id
        self.last_text_context[message.guild.id] = message.content
        self.last_text_author[message.guild.id] = message.author.id
        await self._apply_social_message_feedback(message)
        self.language.learn(message.content)
        user_affinity = self._user_affinity(message.author.id)
        disliked_user = bool(
            self.cfg.behavior.ignore_disliked_users_text
            and user_affinity <= float(self.cfg.behavior.user_avoid_threshold)
        )
        mentioned = self.user in message.mentions if self.user else False
        channel_name = getattr(message.channel, "name", str(message.channel.id))
        self._last_brain_event = f"TEXT • {message.author.display_name} • #{channel_name}" + (" • mention" if mentioned else "")

        async with self._brain_lock:
            self.brain.inject_text(message.content, message.author.id, mentioned)
            familiar_threshold = float(
                self.cfg.behavior.familiar_affinity_threshold
            )
            if user_affinity >= familiar_threshold:
                self.brain.inject(
                    "social:familiar-user",
                    min(1.0, 0.35 + abs(user_affinity)),
                    128,
                )
                self.brain.inject(
                    f"social:familiar-user:{message.author.id}",
                    min(1.0, 0.30 + abs(user_affinity)),
                    96,
                )
            if user_affinity >= 0.35:
                self.brain.inject(
                    "social:liked-user",
                    min(1.0, user_affinity),
                    128,
                )
                self.brain.inject(
                    f"social:liked-user:{message.author.id}",
                    min(1.0, user_affinity),
                    96,
                )
            elif user_affinity <= float(self.cfg.behavior.user_avoid_threshold):
                self.brain.inject(
                    f"social:disliked-user:{message.author.id}",
                    min(1.0, abs(user_affinity)),
                    96,
                )
            self.brain.step(self.cfg.brain.steps_per_event)
            scores = self.brain.action_scores()

        now = time.monotonic()

        # Autonomous reaction path. The connectome decides whether to react and
        # separately scores the available emoji outputs.
        last_react = self._last_reaction.get(message.guild.id, 0.0)
        react_cooldown = max(
            0.0,
            self.cfg.behavior.reaction_cooldown_seconds - (now - last_react),
        )
        self._reaction_debug = {
            "score": scores["react"],
            "threshold": self.cfg.behavior.reaction_threshold,
            "decision": "NIE REAGUJĘ",
            "emoji": None,
            "target": f"#{channel_name} / {message.author.display_name}",
            "cooldown_remaining": react_cooldown,
            "guild_id": message.guild.id,
        }
        if (
            not disliked_user
            and scores["react"] >= self.cfg.behavior.reaction_threshold
            and react_cooldown <= 0.0
        ):
            candidates, pool_total = self._reaction_candidates(message.guild)
            async with self._brain_lock:
                ranked = sorted(
                    (
                        (
                            label,
                            reaction_obj,
                            self.brain.readout("reaction-emoji:" + label, 96),
                        )
                        for label, reaction_obj in candidates
                    ),
                    key=lambda item: item[2],
                    reverse=True,
                )
                learning_trace = self.brain.capture_learning_trace()

            self._reaction_debug["pool_total"] = pool_total
            self._reaction_debug["candidates_evaluated"] = len(ranked)
            self._reaction_debug["top_candidates"] = [
                {"emoji": label, "score": float(score)}
                for label, _, score in ranked[:10]
            ]
            self._reaction_debug["decision"] = "PRÓBUJĘ REAKCJI"

            chosen_label = None
            last_http_error = None
            for label, reaction_obj, _ in ranked[: min(10, len(ranked))]:
                try:
                    await message.add_reaction(reaction_obj)
                    chosen_label = label
                    break
                except discord.Forbidden:
                    self._reaction_debug["decision"] = "BRAK UPRAWNIEŃ"
                    break
                except discord.HTTPException as exc:
                    last_http_error = exc
                    continue

            if chosen_label is not None:
                self._last_reaction[message.guild.id] = now
                self._reaction_debug["emoji"] = chosen_label
                self._reaction_debug["decision"] = "REAKCJA DODANA"
                self._reaction_debug["cooldown_remaining"] = float(
                    self.cfg.behavior.reaction_cooldown_seconds
                )
                self._last_brain_action = f"REACTION → {chosen_label} • #{channel_name}"
                self._set_reinforceable(
                    message.guild,
                    "react",
                    learning_trace,
                    f"{chosen_label} → #{channel_name}",
                )
                self._record_action(
                    "react",
                    f"{chosen_label} → #{channel_name} / {message.author.display_name}",
                    message.guild,
                )
            elif self._reaction_debug["decision"] != "BRAK UPRAWNIEŃ":
                self._reaction_debug["decision"] = (
                    f"EMOJI ODRZUCONE: {type(last_http_error).__name__}"
                    if last_http_error is not None
                    else "BRAK KANDYDATÓW"
                )
        elif disliked_user:
            self._reaction_debug["decision"] = (
                f"SOCIAL AVOID • affinity {user_affinity:+.2f}"
            )
        elif react_cooldown > 0.0:
            self._reaction_debug["decision"] = "COOLDOWN"
        else:
            self._reaction_debug["decision"] = (
                f"react {scores['react']:.3f} < {self.cfg.behavior.reaction_threshold:.3f}"
            )

        last = self.last_reply.get(message.guild.id, 0.0)
        urge = scores["speak"] + (0.10 if mentioned else 0.0)
        if (
            not blocked_text
            and not disliked_user
            and self.language.ready()
            and urge >= self.cfg.behavior.speak_threshold
            and now - last >= self.cfg.language.reply_cooldown_seconds
        ):
            await self._send_learned(
                message.channel,
                message.content,
                scores["explore"],
                target_member=(
                    message.author
                    if isinstance(message.author, discord.Member)
                    else None
                ),
            )
            self.last_reply[message.guild.id] = now

    def _brain_word_feedback(
        self,
        token: str,
        previous_token: str | None,
    ) -> None:
        if not self.cfg.language.connectome_word_feedback_enabled:
            return
        self.brain.advance_language_word(
            token,
            previous_token,
            magnitude=float(
                self.cfg.language.connectome_word_feedback_magnitude
            ),
            steps=int(self.cfg.language.connectome_word_feedback_steps),
        )

    async def _send_learned(
        self,
        channel: discord.abc.Messageable,
        context: str,
        arousal: float,
        target_member: discord.Member | None = None,
    ):
        if self._is_text_channel_blocked(channel):
            return
        async with self._brain_lock:
            internal = self.brain.internal_state_diagnostics()
            neural_arousal = float(
                internal.get("states", {})
                .get("arousal", {})
                .get("level", 0.0)
            )
            effective_arousal = max(
                0.0,
                min(
                    1.0,
                    0.75 * neural_arousal
                    + 0.25 * float(arousal),
                ),
            )
            text, trigrams = self.language.generate(
                context=context,
                arousal=effective_arousal,
                brain_word_score=self.brain.language_word_score,
                brain_word_feedback=self._brain_word_feedback,
            )
            if text:
                self.brain.mark_language_output(text)
                learning_trace = self.brain.capture_learning_trace()
            else:
                learning_trace = None
        if not text or learning_trace is None:
            return
        try:
            sent = await channel.send(text, allowed_mentions=discord.AllowedMentions.none())
            guild = getattr(channel, "guild", None)
            self.sent[sent.id] = SentTrace(
                trigrams=trigrams,
                created=time.monotonic(),
                action="speak",
                learning_trace=learning_trace,
                text=text,
                guild_id=guild.id if isinstance(guild, discord.Guild) else None,
                channel_id=getattr(channel, "id", None),
            )

            if (
                isinstance(guild, discord.Guild)
                and target_member is not None
                and getattr(channel, "id", None) is not None
            ):
                self._track_direct_reply_target(
                    guild,
                    target_member,
                    int(channel.id),
                    sent.id,
                    self.sent[sent.id],
                )

            if self.cfg.behavior.social_learning_enabled:
                recent_self = [
                    trace
                    for mid, trace in self.sent.items()
                    if mid != sent.id
                    and trace.guild_id == self.sent[sent.id].guild_id
                    and trace.channel_id == self.sent[sent.id].channel_id
                    and time.monotonic() - trace.created
                    <= float(self.cfg.behavior.social_window_seconds)
                ]
                new_words = set(self._social_words(text))
                repeated = False
                for old_trace in recent_self[-8:]:
                    old_words = set(self._social_words(old_trace.text))
                    if (
                        len(new_words) >= 4
                        and len(old_words) >= 4
                        and len(new_words & old_words)
                        / max(1, len(new_words | old_words)) >= 0.72
                    ):
                        repeated = True
                        break
                if repeated:
                    penalty = max(
                        0.0,
                        min(
                            1.0,
                            float(self.cfg.behavior.self_repeat_penalty),
                        ),
                    )
                    self.language.reinforce_text(text, -penalty)
                    async with self._brain_lock:
                        self.brain.inject(
                            "internal:self-repeat",
                            min(1.0, 0.45 + penalty),
                            128,
                        )
                        self.brain.reward(
                            -min(0.15, penalty * 0.4),
                            action="speak",
                            trace=learning_trace,
                        )
                        self.brain.step(1)
                    self._record_action(
                        "self_repeat",
                        f"-{penalty:.2f} • {text[:100]}",
                        guild if isinstance(guild, discord.Guild) else None,
                    )
                    self._remember_social_event(
                        "SELF_REPEAT",
                        text[:100],
                        -penalty,
                    )
            channel_name = getattr(channel, "name", "kanał")
            self._last_brain_action = f"TEXT → #{channel_name}: {text[:80]}"
            if isinstance(guild, discord.Guild):
                self._set_reinforceable(
                    guild,
                    "speak",
                    learning_trace,
                    f"#{channel_name}: {text[:80]}",
                )
            self._record_action(
                "speak",
                f"#{channel_name}: {text[:120]}",
                guild if isinstance(guild, discord.Guild) else None,
            )
            if len(self.sent) > 500:
                oldest = sorted(self.sent.items(), key=lambda kv: kv[1].created)[:100]
                for mid, _ in oldest:
                    self.sent.pop(mid, None)
        except (discord.Forbidden, discord.HTTPException):
            log.exception("Nie udało się wysłać wiadomości")

    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        if self.user and payload.user_id == self.user.id:
            return
        trace = self.sent.get(payload.message_id)
        if not trace:
            return
        emoji = str(payload.emoji)
        amount = 0.0
        if emoji in POSITIVE:
            amount = 1.0
        elif emoji in NEGATIVE:
            amount = -1.0
        if amount:
            guild = self.get_guild(payload.guild_id) if payload.guild_id else None
            member = guild.get_member(payload.user_id) if guild else None
            display_name = (
                member.display_name
                if member is not None
                else str(payload.user_id)
            )
            reaction_streak_count = 0
            reaction_streak_multiplier = 1.0
            if amount > 0:
                self._soften_negative_streak(payload.user_id)
                affinity_delta = (
                    float(self.cfg.behavior.user_affinity_positive_step)
                    * POSITIVE_REACTION_WEIGHT.get(emoji, 0.6)
                )
                affinity_kind = "positive"
            else:
                (
                    reaction_streak_count,
                    reaction_streak_multiplier,
                ) = self._advance_negative_streak(payload.user_id)
                affinity_delta = -min(
                    0.30,
                    float(self.cfg.behavior.user_affinity_negative_step)
                    * NEGATIVE_REACTION_WEIGHT.get(emoji, 0.6)
                    * reaction_streak_multiplier,
                )
                affinity_kind = "negative"
            new_affinity = self.language.adjust_user_affinity(
                payload.user_id,
                display_name,
                affinity_delta,
                affinity_kind,
            )
            self._last_brain_event = (
                f"REACTION • {emoji} • reward {amount:+.0f} • "
                f"affinity {new_affinity:+.2f}"
            )
            self.language.reinforce_text(trace.text, amount)
            async with self._brain_lock:
                self._write_neural_social_memory(
                    payload.user_id,
                    affinity_delta,
                )
                new_affinity = self._user_affinity(
                    payload.user_id
                )
                if amount < 0 and reaction_streak_count >= 2:
                    self.brain.inject(
                        "social:repeated-rejection",
                        min(
                            1.0,
                            0.52 + 0.08 * reaction_streak_count,
                        ),
                        144,
                    )
                    self.brain.inject(
                        f"social:repeated-rejection:user:{payload.user_id}",
                        min(
                            1.0,
                            0.45 + 0.07 * reaction_streak_count,
                        ),
                        96,
                    )
                self.brain.reward(
                    amount,
                    action=trace.action,
                    trace=trace.learning_trace,
                )
                self.brain.step(1)
            self._record_reward(amount, trace.action, f"Discord {emoji}", guild)
            self._record_action(
                "reward",
                (
                    f"{amount:+.0f} → {trace.action} ({emoji}) • "
                    f"{display_name} affinity {new_affinity:+.2f}"
                ),
                guild,
            )
            self._remember_social_event(
                "REACTION_AFFINITY",
                f"{emoji} • {new_affinity:+.2f}",
                affinity_delta,
                member,
            )

    async def on_voice_state_update(self, member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
        self._last_external_activity = time.monotonic()
        if self.paused:
            return
        if self.user and member.id == self.user.id:
            if after.channel is None:
                vc = member.guild.voice_client
                if vc is not None:
                    try:
                        self._stop_voice_playback(vc)
                    except Exception:
                        pass
                self._set_audio_disconnected(
                    member.guild,
                    reason="discord_voice_state_left",
                )
                with self._stt_buffer_lock:
                    self._stt_buffers.clear()
            if (
                after.channel is not None
                and self._is_voice_guild_blocked(member.guild)
            ):
                vc = member.guild.voice_client
                if vc is not None and vc.is_connected():
                    try:
                        await vc.disconnect(force=False)
                    except (discord.Forbidden, discord.HTTPException):
                        log.exception(
                            "Nie udało się opuścić zablokowanego serwera voice %s",
                            member.guild.id,
                        )
                self._record_action(
                    "voice_guild_block",
                    f"zakaz voice na serwerze {member.guild.name}",
                    member.guild,
                )
            return
        before_name = getattr(before.channel, "name", "poza voice")
        after_name = getattr(after.channel, "name", "poza voice")
        self._last_brain_event = f"VOICE • {member.display_name}: {before_name} → {after_name}"
        key = f"voice-change:{member.id}:{getattr(before.channel, 'id', 0)}:{getattr(after.channel, 'id', 0)}"

        predator_encounter = False
        confirmed_chaser = False
        chaser_hits = 0
        now = time.monotonic()
        vc = member.guild.voice_client
        my_channel = self._current_voice_channel(member.guild)
        changed_channel = getattr(before.channel, "id", None) != getattr(
            after.channel,
            "id",
            None,
        )

        if (
            not member.bot
            and changed_channel
            and before.channel is not None
            and my_channel is not None
            and before.channel.id == my_channel.id
            and getattr(after.channel, "id", None) != my_channel.id
            and self._chaser_panic_remaining(member.guild.id, now) <= 0.0
        ):
            last_tts = self._last_tts_trace.get(member.guild.id)
            tts_audience = self._last_tts_audience.get(
                member.guild.id,
                set(),
            )
            tts_recent = bool(
                last_tts is not None
                and last_tts.channel_id == before.channel.id
                and member.id in tts_audience
                and now - last_tts.created
                <= float(self.cfg.behavior.tts_leave_seconds)
            )

            if tts_recent and last_tts is not None:
                await self._grant_negative_social(
                    member,
                    "VOICE_LEFT_AFTER_TTS",
                    "social:user-left-after-tts",
                    self.cfg.behavior.tts_leave_affinity_step,
                    member.guild,
                    detail=(
                        f"{before.channel.name} → {after_name} • "
                        "wyjście krótko po TTS"
                    ),
                    source_action=last_tts.action,
                    source_learning_trace=last_tts.learning_trace,
                    brain_penalty=0.06,
                )
            else:
                arrival = self._voice_arrival_learning.get(
                    member.guild.id
                )
                arrival_members = self._voice_arrival_members.get(
                    member.guild.id,
                    set(),
                )
                arrival_channel = self._voice_arrival_channel.get(
                    member.guild.id
                )
                if (
                    arrival is not None
                    and arrival_channel == before.channel.id
                    and member.id in arrival_members
                    and now - float(arrival[2])
                    <= float(
                        self.cfg.behavior.voice_leave_after_join_seconds
                    )
                ):
                    moved = after.channel is not None
                    await self._grant_negative_social(
                        member,
                        (
                            "VOICE_MOVED_AWAY"
                            if moved
                            else "VOICE_LEFT_AFTER_JOIN"
                        ),
                        (
                            "social:user-moved-away"
                            if moved
                            else "social:user-left-after-my-join"
                        ),
                        self.cfg.behavior.voice_leave_after_join_affinity_step,
                        member.guild,
                        detail=(
                            f"{before.channel.name} → {after_name} • "
                            "krótko po wejściu Muchy"
                        ),
                        source_action=arrival[0],
                        source_learning_trace=arrival[1],
                        brain_penalty=0.08,
                    )

        if (
            not member.bot
            and changed_channel
            and after.channel is not None
            and my_channel is not None
            and after.channel.id == my_channel.id
            and self._chaser_panic_remaining(member.guild.id, now) <= 0.0
        ):
            await self._grant_positive_social(
                member,
                "VOICE_JOIN_ME",
                "social:user-joined-my-voice",
                self.cfg.behavior.voice_join_affinity_step,
                member.guild,
                detail=f"wszedł na {after.channel.name}",
            )
            self._schedule_voice_social_stay(
                member.guild,
                member,
                after.channel.id,
            )

        if (
            not member.bot
            and changed_channel
            and after.channel is not None
            and my_channel is not None
            and after.channel.id == my_channel.id
            and self.cfg.behavior.avoid_disliked_users_on_voice
            and self._is_disliked_user(member.id)
            and self._chaser_panic_remaining(member.guild.id, now) <= 0.0
        ):
            self._last_brain_event = (
                f"SOCIAL AVOID • {member.display_name} wszedł na "
                f"{after.channel.name}"
            )
            asyncio.create_task(self._voice_decision(member.guild))

        if (
            self.cfg.voice.chaser_enabled
            and member.bot
            and changed_channel
            and after.channel is not None
            and my_channel is not None
            and after.channel.id == my_channel.id
        ):
            predator_encounter = True
            confirmed_chaser, chaser_hits = self._register_chaser_encounter(
                member,
                after.channel,
                now,
            )

        async with self._brain_lock:
            self.brain.inject(key, 0.65, 64)
            self.brain.inject(f"voice-user:{member.id}", 0.35, 48)
            if predator_encounter:
                magnitude = float(self.cfg.voice.chaser_threat_magnitude)
                if not confirmed_chaser:
                    magnitude *= 0.55
                self.brain.inject(
                    "internal:predator-chaser",
                    magnitude,
                    192,
                )
                self.brain.inject(
                    f"voice:predator:{member.guild.id}:{member.id}",
                    magnitude,
                    160,
                )
                self.brain.inject(
                    f"voice:danger-channel:{member.guild.id}:{after.channel.id}",
                    magnitude * 0.8,
                    128,
                )
                self.brain.step(3)
                learning_trace = self.brain.capture_learning_trace()
            else:
                self.brain.step(1)
                learning_trace = None

        if predator_encounter and learning_trace is not None:
            state = "CONFIRMED" if confirmed_chaser else "SUSPECT"
            self._last_brain_event = (
                f"CHASER {state} • {member.display_name} • "
                f"{after.channel.name} • hit {chaser_hits}"
            )
            self._last_brain_action = (
                f"PANIC • {member.display_name} wykryty"
            )
            self._record_action(
                "chaser_detected",
                f"{state} {member.display_name} • hit {chaser_hits} • "
                f"{after.channel.name}",
                member.guild,
            )
            self._schedule_chaser_escape(
                member.guild,
                member.id,
                learning_trace,
            )
            self._ensure_chaser_scream_loop(member.guild)

    async def _maybe_memory_replay(
        self,
        now: float | None = None,
    ) -> None:
        now = time.monotonic() if now is None else float(now)
        self._memory_replay_debug["enabled"] = bool(
            self.cfg.voice.memory_replay_enabled
        )
        if (
            not self.cfg.voice.memory_replay_enabled
            or not self.cfg.voice.episodic_prediction_enabled
        ):
            self._memory_replay_debug.update({
                "state": "OFF",
                "reason": "disabled",
            })
            return

        idle_seconds = max(
            10.0,
            float(self.cfg.voice.memory_replay_idle_seconds),
        )
        quiet_for = max(
            0.0,
            now - float(self._last_external_activity),
        )
        if quiet_for < idle_seconds:
            self._memory_replay_debug.update({
                "state": "WAITING",
                "reason": "activity",
                "quiet_for": quiet_for,
            })
            return

        interval = max(
            30.0,
            float(self.cfg.voice.memory_replay_interval_seconds),
        )
        since_last = max(0.0, now - self._memory_replay_last)
        if self._memory_replay_last > 0.0 and since_last < interval:
            self._memory_replay_debug.update({
                "state": "WAITING",
                "reason": "interval",
                "quiet_for": quiet_for,
                "next_in": interval - since_last,
            })
            return

        if any(
            vc is not None and vc.is_connected()
            for vc in self.voice_clients
        ):
            self._memory_replay_debug.update({
                "state": "WAITING",
                "reason": "voice-connected",
                "quiet_for": quiet_for,
            })
            return

        if any(
            self._chaser_panic_remaining(guild.id, now) > 0.0
            for guild in self.guilds
        ):
            self._memory_replay_debug.update({
                "state": "WAITING",
                "reason": "chaser-panic",
                "quiet_for": quiet_for,
            })
            return

        batch_size = max(
            1,
            min(8, int(self.cfg.voice.memory_replay_batch_size)),
        )
        candidates = self.voice_episodes.replay_candidates(
            limit=max(12, batch_size * 8),
            max_age_seconds=max(
                86400.0,
                float(self.cfg.voice.memory_replay_max_age_days)
                * 86400.0,
            ),
        )
        if not candidates:
            self._memory_replay_debug.update({
                "state": "WAITING",
                "reason": "no-episodes",
                "quiet_for": quiet_for,
            })
            return

        recent = set(self._memory_replay_recent_keys)
        selected = []
        for episode in candidates:
            key = (
                f"{float(episode.get('time', 0.0)):.6f}:"
                f"{episode.get('guild_id')}:"
                f"{episode.get('action')}:"
                f"{episode.get('scene_key')}"
            )
            if key in recent:
                continue
            row = dict(episode)
            row["_replay_key"] = key
            selected.append(row)
            if len(selected) >= batch_size:
                break
        if not selected:
            recent.clear()
            self._memory_replay_recent_keys.clear()
            for episode in candidates[:batch_size]:
                row = dict(episode)
                row["_replay_key"] = (
                    f"{float(row.get('time', 0.0)):.6f}:"
                    f"{row.get('guild_id')}:"
                    f"{row.get('action')}:"
                    f"{row.get('scene_key')}"
                )
                selected.append(row)

        magnitude = max(
            0.0,
            min(1.5, float(self.cfg.voice.memory_replay_magnitude)),
        )
        reward_scale = max(
            0.0,
            min(
                0.5,
                float(self.cfg.voice.memory_replay_reward_scale),
            ),
        )
        if magnitude <= 0.001:
            self._memory_replay_debug.update({
                "state": "WAITING",
                "reason": "zero-magnitude",
                "quiet_for": quiet_for,
            })
            return
        steps = max(
            1,
            min(24, int(self.cfg.voice.memory_replay_steps)),
        )
        replayed = []

        async with self._brain_lock:
            for episode in selected:
                guild_id = int(episode.get("guild_id") or 0)
                channel_id = episode.get("channel_id")
                user_ids = [
                    int(x)
                    for x in episode.get("user_ids", [])
                ]
                action = str(episode.get("action") or "stay")

                if channel_id is not None:
                    self.brain.inject_voice_snapshot(
                        guild_id,
                        int(channel_id),
                        user_ids,
                        sensory_scale=max(
                            0.05,
                            min(1.0, magnitude),
                        ),
                    )
                else:
                    self.brain.inject(
                        f"memory-replay:outside:guild:{guild_id}",
                        0.35 * magnitude,
                        64,
                    )
                    for user_id in user_ids[:12]:
                        self.brain.activate_user_memory(
                            user_id,
                            0.10 * magnitude,
                        )

                guided = self.brain.inject_action_guided_sensory(
                    action,
                    (
                        "memory-replay:"
                        f"{guild_id}:"
                        f"{episode.get('scene_key', '')}"
                    ),
                    magnitude,
                    width=176,
                    hops=3,
                )
                self.brain.step(steps)
                trace = self.brain.capture_learning_trace()

                actual = float(episode.get("actual_reward", 0.0))
                error = float(
                    episode.get("prediction_error", 0.0)
                )
                replay_signal = max(
                    -1.0,
                    min(1.0, 0.65 * actual + 0.35 * error),
                )
                replay_reward = replay_signal * reward_scale
                if abs(replay_reward) > 1e-9:
                    learning = self.brain.reward(
                        replay_reward,
                        action=action,
                        trace=trace,
                    )
                else:
                    learning = {}
                self.brain.step(1)
                memory_consolidation = (
                    self.voice_episodes.consolidate_replay(
                        episode,
                        replay_reward,
                    )
                )

                replayed.append({
                    "time": float(episode.get("time", 0.0)),
                    "guild_id": guild_id,
                    "channel_id": channel_id,
                    "channel_name": str(
                        episode.get("channel_name", "")
                    ),
                    "user_names": list(
                        episode.get("user_names", [])
                    ),
                    "action": action,
                    "actual_reward": actual,
                    "prediction_error": error,
                    "replay_score": float(
                        episode.get("replay_score", 0.0)
                    ),
                    "replay_reward": float(replay_reward),
                    "memory_strength": float(
                        memory_consolidation.get(
                            "strength",
                            episode.get("memory_strength", 0.0),
                        )
                    ),
                    "memory_status": str(
                        memory_consolidation.get(
                            "status",
                            episode.get("memory_status", "forming"),
                        )
                    ),
                    "memory_replays": int(
                        memory_consolidation.get(
                            "replay_count",
                            episode.get("memory_replays", 0),
                        )
                    ),
                    "steps": steps,
                    "guided_mode": str(
                        guided.get("mode", "")
                    ),
                    "guided_neurons": int(
                        guided.get("neurons", 0)
                    ),
                    "guided_reach_max": float(
                        guided.get("reach_max", 0.0)
                    ),
                    "changed_neurons": int(
                        learning.get("changed_neurons", 0)
                    ),
                    "changed_synapses": int(
                        learning.get("changed_synapses", 0)
                    ),
                })

        for episode in selected:
            self._memory_replay_recent_keys.append(
                str(episode["_replay_key"])
            )
        self._memory_replay_recent_keys = (
            self._memory_replay_recent_keys[-12:]
        )
        self._memory_replay_last = now
        self._memory_replay_count += len(replayed)
        self._memory_replay_debug.update({
            "state": "REPLAY",
            "reason": "idle-memory-consolidation",
            "quiet_for": quiet_for,
            "last_at": time.time(),
            "count": self._memory_replay_count,
            "batch": len(replayed),
            "last": replayed,
        })
        self._last_brain_event = (
            f"MEMORY REPLAY • {len(replayed)} episode(s)"
        )
        if replayed:
            last = replayed[-1]
            self._last_brain_action = (
                "REPLAY → "
                f"{last['action']} "
                f"{last['replay_reward']:+.3f}"
            )
            guild = self.get_guild(int(last["guild_id"]))
            self._record_action(
                "memory_replay",
                (
                    f"{last['action']} • "
                    f"{last['channel_name'] or 'poza VC'} • "
                    f"reward {last['replay_reward']:+.3f} • "
                    f"{steps} tick"
                ),
                guild,
            )

    @tasks.loop(seconds=5)
    async def idle_loop(self):
        await self.wait_until_ready()
        if self.paused:
            return
        now = time.monotonic()
        wall_now = time.time()
        await self._maybe_memory_replay(now)
        async with self._brain_lock:
            self.brain.consolidate_and_forget(now=wall_now)
            self.brain.inject("internal:time", 0.035, 32)
            self.brain.step(self.cfg.brain.idle_steps)
            scores = self.brain.action_scores()
        self.voice_episodes.apply_forgetting(now=wall_now)

        if now - self._last_save >= self.cfg.behavior.save_every_seconds:
            async with self._brain_lock:
                self.brain.save()
            self._last_save = now

        if not self.cfg.language.spontaneous_text or not self.language.ready():
            return
        if scores["speak"] < max(self.cfg.behavior.speak_threshold + 0.08, 0.80):
            return
        for guild in self.guilds:
            last = self.last_spontaneous.get(guild.id, 0.0)
            if now - last < self.cfg.language.spontaneous_cooldown_seconds:
                continue
            last_author = self.last_text_author.get(guild.id)
            if (
                self.cfg.behavior.ignore_disliked_users_text
                and last_author is not None
                and self._is_disliked_user(last_author)
            ):
                continue
            cid = self.last_text_channel.get(guild.id)
            channel = guild.get_channel(cid) if cid else None
            if (
                isinstance(channel, discord.TextChannel)
                and not self._is_text_channel_blocked(channel)
            ):
                await self._send_learned(channel, "", scores["explore"])
                self.last_spontaneous[guild.id] = now
                break

    @idle_loop.before_loop
    async def before_idle(self):
        await self.wait_until_ready()

    def _presence_from_brain(self, scores: dict[str, float], diag: dict[str, float | int]) -> tuple[discord.ActivityType, str]:
        """Translate current connectome readouts into a Discord presence.

        The labels are only human-readable names for neuronal readouts; the
        winning state and activity value come from the live connectome.
        """
        mean_abs = float(diag["mean_abs"])
        active = int(diag["active_abs_gt_0_1"])

        candidates = {
            "speak": scores["speak"],
            "explore": scores["explore"],
            "voice": max(scores["voice_join"], scores["voice_move"]),
            "stay": scores["stay"],
            "react": scores["react"],
        }
        dominant = max(candidates, key=candidates.get)
        strength = candidates[dominant]

        if dominant == "voice":
            activity_type = discord.ActivityType.listening
            label = "nasłuchuje kanałów"
        elif dominant == "speak":
            activity_type = discord.ActivityType.listening
            label = "uczy się rozmów"
        elif dominant == "explore":
            activity_type = discord.ActivityType.watching
            label = "eksploruje serwer"
        elif dominant == "react":
            activity_type = discord.ActivityType.watching
            label = "obserwuje reakcje"
        else:
            activity_type = discord.ActivityType.watching
            label = "przetwarza bodźce"

        text = f"🧠 {label} • a={mean_abs:.3f} • {active:,} aktywnych"
        if strength >= 0.85:
            text = "⚡ " + text[2:]
        return activity_type, text[:128]

    async def _update_presence(self) -> None:
        if not self.is_ready():
            return
        async with self._brain_lock:
            scores = self.brain.action_scores()
            diag = self.brain.diagnostics()

        activity_type, text = self._presence_from_brain(scores, diag)
        if text == self._last_presence_text:
            return

        activity = discord.Activity(type=activity_type, name=text)
        try:
            await self.change_presence(
                status=discord.Status.online,
                activity=activity,
            )
            self._last_presence_text = text
        except discord.HTTPException:
            log.exception("Nie udało się zaktualizować statusu Discord")

    async def _connectome_dashboard_snapshot(
        self,
        follow_activity: bool = False,
        action: str | None = None,
    ) -> dict:
        async with self._brain_lock:
            snap = self.brain.connectome_visual_snapshot(
                count=42 if follow_activity else 64,
                edge_limit=150 if follow_activity else 190,
                follow_activity=follow_activity,
            )
            action = str(action or "").strip()
            if action in self.brain.ACTIONS:
                snap["action_path"] = (
                    self.brain.action_path_snapshot(action)
                )
            return snap

    async def _association_dashboard_snapshot(self) -> dict:
        word_rows = self.language.association_words(limit=28)
        words = [str(row.get("word", "")) for row in word_rows]
        async with self._brain_lock:
            graph = self.brain.word_association_snapshot(
                words,
                max_nodes=28,
                edge_limit=72,
            )

        metadata = {
            str(row.get("word", "")): row
            for row in word_rows
        }
        for node in graph.get("nodes", []):
            info = metadata.get(str(node.get("id", "")), {})
            node["count"] = int(info.get("count", 0))
            node["language_reward"] = float(info.get("reward", 0.0))
            node["last_seen"] = float(info.get("last_seen", 0.0))

        graph["last_event"] = self._last_brain_event
        graph["last_action"] = self._last_brain_action
        graph["language_diag"] = self.language.diagnostics()
        return graph

    async def _neuromap_dashboard_snapshot(
        self,
        projection: str = "xy",
    ) -> dict:
        async with self._brain_lock:
            brain_map = self.brain.neuro_map_snapshot(
                count=220,
                projection=projection,
            )
            scores = self.brain.action_scores()

        language_diag = self.language.diagnostics()
        return {
            "brain_map": brain_map,
            "scores": scores,
            "last_event": self._last_brain_event,
            "last_action": self._last_brain_action,
            "language_diag": language_diag,
            "source": self.connectome.metadata.get("source", "unknown"),
        }

    async def _console_snapshot(self) -> dict:
        async with self._brain_lock:
            scores = self.brain.action_scores()
            diag = self.brain.diagnostics()
            top_neurons = self.brain.top_active_neurons(
                self.cfg.console_ui.top_neurons
            )
            learning_since_start = (
                self.brain.learning_since_start_diagnostics()
            )

        language_total, language_unique = self.language.stats()
        language_diag = self.language.diagnostics()
        language_start = self._language_start_diag
        learning_since_start.update({
            "language_chars": max(
                0,
                int(language_diag.get("chars", 0))
                - int(language_start.get("chars", 0)),
            ),
            "language_messages": max(
                0,
                int(language_diag.get("messages", 0))
                - int(language_start.get("messages", 0)),
            ),
            "language_transitions": max(
                0,
                int(language_diag.get("transitions", 0))
                - int(language_start.get("transitions", 0)),
            ),
            "language_unique_chars": max(
                0,
                int(language_diag.get("unique_chars", 0))
                - int(language_start.get("unique_chars", 0)),
            ),
            "language_word_tokens": max(
                0,
                int(language_diag.get("word_tokens", 0))
                - int(language_start.get("word_tokens", 0)),
            ),
            "language_word_vocab": max(
                0,
                int(language_diag.get("word_vocab", 0))
                - int(language_start.get("word_vocab", 0)),
            ),
            "language_word_bigrams": max(
                0,
                int(language_diag.get("word_bigrams", 0))
                - int(language_start.get("word_bigrams", 0)),
            ),
            "language_word_trigrams": max(
                0,
                int(language_diag.get("word_trigrams", 0))
                - int(language_start.get("word_trigrams", 0)),
            ),
            "voice_transcripts": int(
                self._stt_transcripts_since_start
            ),
        })
        user_affinities = self.language.user_affinities(50)
        now_mono = time.monotonic()
        streak_window = max(
            1.0,
            float(self.cfg.behavior.negative_streak_window_seconds),
        )
        for row in user_affinities:
            streak = self._social_negative_streak.get(
                int(row["user_id"])
            )
            count = 0
            multiplier = 1.0
            if (
                streak is not None
                and now_mono - float(streak.get("last", 0.0))
                <= streak_window
            ):
                count = int(streak.get("count", 0))
                multiplier = min(
                    max(
                        1.0,
                        float(
                            self.cfg.behavior.negative_streak_max_multiplier
                        ),
                    ),
                    1.0
                    + max(
                        0.0,
                        float(
                            self.cfg.behavior.negative_streak_multiplier_step
                        ),
                    )
                    * max(0, count - 1),
                )
            row["negative_streak"] = count
            row["negative_multiplier"] = multiplier

        if self.cfg.behavior.neural_social_memory_enabled:
            async with self._brain_lock:
                for row in user_affinities[:20]:
                    components = self._user_affinity_components(
                        int(row["user_id"])
                    )
                    memory = components.get("memory") or {}
                    row["legacy_affinity"] = float(
                        components["legacy"]
                    )
                    row["neural_affinity"] = float(
                        components["neural"]
                    )
                    row["neural_maturity"] = float(
                        components["maturity"]
                    )
                    row["neural_weight"] = float(
                        components["weight"]
                    )
                    row["effective_affinity"] = float(
                        components["effective"]
                    )
                    row["affinity"] = float(
                        components["effective"]
                    )
                    row["neural_memory"] = memory
        else:
            for row in user_affinities:
                row["legacy_affinity"] = float(
                    row.get("affinity", 0.0)
                )
                row["neural_affinity"] = 0.0
                row["neural_maturity"] = 0.0
                row["neural_weight"] = 0.0
                row["effective_affinity"] = float(
                    row.get("affinity", 0.0)
                )

        voice_parts = []
        for guild in self.guilds:
            vc = guild.voice_client
            if vc and vc.is_connected() and vc.channel:
                voice_parts.append(f"{guild.name}/{vc.channel.name}")
        reaction_debug = dict(self._reaction_debug)
        reaction_guild_id = reaction_debug.get("guild_id")
        if reaction_guild_id:
            last_react = self._last_reaction.get(int(reaction_guild_id), 0.0)
            reaction_debug["cooldown_remaining"] = max(
                0.0,
                self.cfg.behavior.reaction_cooldown_seconds - (time.monotonic() - last_react),
            )

        return {
            "source": self.connectome.metadata.get("source", "unknown"),
            "diag": diag,
            "scores": scores,
            "top_neurons": top_neurons,
            "language_tokens": language_total,
            "language_unique": language_unique,
            "language_ready": self.language.ready(),
            "language_diag": language_diag,
            "voice": ", ".join(voice_parts) if voice_parts else "poza voice",
            "last_event": self._last_brain_event,
            "last_action": self._last_brain_action,
            "paused": self.paused,
            "voice_debug": list(self._voice_debug.values()),
            "episodic_memory": self.voice_episodes.diagnostics(),
            "memory_replay": dict(self._memory_replay_debug),
            "audio_debug": dict(self._audio_debug),
            "stt_debug": dict(self._stt_debug),
            "reaction_debug": reaction_debug,
            "learning_debug": self.brain.learning_diagnostics(),
            "learning_since_start": learning_since_start,
            "affinity_rules": self._affinity_rules_snapshot(),
            "social_debug": dict(self._social_debug),
            "user_affinities": user_affinities,
            "word_feedback": self.language.top_word_feedback(30),
            "social_settings": {
                "user_avoid_threshold": self.cfg.behavior.user_avoid_threshold,
                "familiar_affinity_threshold": self.cfg.behavior.familiar_affinity_threshold,
                "ignore_disliked_users_text": self.cfg.behavior.ignore_disliked_users_text,
                "avoid_disliked_users_on_voice": self.cfg.behavior.avoid_disliked_users_on_voice,
                "neural_social_memory_enabled": self.cfg.behavior.neural_social_memory_enabled,
                "neural_affinity_weight": self.cfg.behavior.neural_affinity_weight,
                "neural_social_learning_scale": self.cfg.behavior.neural_social_learning_scale,
            },
            "action_history": self._action_history[-40:],
            "reward_history": self._reward_history[-80:],
            "guild_learning_context": [
                self._guild_learning_context[guild.id]
                for guild in self.guilds
                if guild.id in self._guild_learning_context
            ],
        }

    @tasks.loop(seconds=1)
    async def console_loop(self):
        await self.wait_until_ready()
        try:
            snap = await self._console_snapshot()
            self.console_ui.update(snap)
        except Exception:
            log.exception("Błąd konsolowego dashboardu")

    @console_loop.before_loop
    async def before_console(self):
        await self.wait_until_ready()

    @tasks.loop(seconds=30)
    async def presence_loop(self):
        await self.wait_until_ready()
        await self._update_presence()

    @presence_loop.before_loop
    async def before_presence(self):
        await self.wait_until_ready()

    def _make_voice_source(
        self,
        path: Path,
        volume: float,
    ) -> discord.AudioSource:
        volume = max(0.0, min(2.0, float(volume)))
        pcm = discord.FFmpegPCMAudio(
            str(path),
            executable=self.cfg.voice.ffmpeg_executable,
            before_options="-loglevel error",
            options="-vn",
        )
        return discord.PCMVolumeTransformer(
            pcm,
            volume=volume,
        )

    def _audio_playback_done(
        self,
        token: int,
        guild_id: int,
        label: str,
        error: Exception | None,
    ) -> None:
        if int(self._audio_debug.get("playback_token", -1)) != token:
            return

        guild = self.get_guild(guild_id)
        vc = guild.voice_client if guild is not None else None
        connected = bool(vc is not None and vc.is_connected())
        self._audio_debug.update({
            "status": "IDLE" if error is None else "ERROR",
            "stage": "finished",
            "playing": False,
            "connected": connected,
            "error": "" if error is None else f"{label}: {error}",
            "channel": (
                getattr(vc.channel, "name", None)
                if connected and vc is not None
                else None
            ),
            "updated_at": time.time(),
        })

    def _play_voice_source(
        self,
        vc: discord.VoiceClient,
        source: discord.AudioSource,
        label: str,
    ) -> int:
        self._audio_playback_token += 1
        token = self._audio_playback_token
        loop = asyncio.get_running_loop()
        guild_id = vc.guild.id
        self._audio_debug["playback_token"] = token

        def finished(error: Exception | None) -> None:
            loop.call_soon_threadsafe(
                self._audio_playback_done,
                token,
                guild_id,
                label,
                error,
            )

        vc.play(source, after=finished)
        return token

    async def _verify_voice_playback(
        self,
        vc: discord.VoiceClient,
        label: str,
        token: int | None = None,
    ) -> None:
        await asyncio.sleep(0.35)
        if (
            token is not None
            and int(self._audio_debug.get("playback_token", -1)) != token
        ):
            return

        connected = bool(vc.is_connected())
        playing = bool(vc.is_playing()) if connected else False
        if not connected:
            self._set_audio_disconnected(
                getattr(vc, "guild", None),
                reason=f"{label}_voice_disconnected",
            )
            return

        if not playing and self._audio_debug.get("stage") == "finished":
            return

        self._audio_debug.update({
            "playing": playing,
            "connected": connected,
            "stage": "playing_check",
            "status": "PLAYING" if playing else "STOPPED",
            "error": (
                ""
                if playing
                else f"{label}: Discord nie raportuje aktywnego playbacku po 350 ms"
            ),
            "updated_at": time.time(),
        })

    def _synthesize_piper_file(self, text: str, path: Path) -> bool:
        if self.cfg.voice.tts_engine.strip().lower() != "piper":
            return False

        if PiperVoice is None or SynthesisConfig is None:
            if not self._piper_warning_shown:
                log.warning(
                    "Piper TTS unavailable; install requirements.txt. "
                    "Falling back to espeak-ng."
                )
                self._piper_warning_shown = True
            return False

        model_path = Path(self.cfg.voice.tts_piper_model)
        config_path = Path(f"{model_path}.json")
        if not model_path.is_file() or not config_path.is_file():
            if not self._piper_warning_shown:
                log.warning(
                    "Piper voice missing: %s / %s. "
                    "Falling back to espeak-ng.",
                    model_path,
                    config_path,
                )
                self._piper_warning_shown = True
            return False

        try:
            with self._piper_lock:
                model_key = str(model_path.resolve())
                if (
                    self._piper_voice is None
                    or self._piper_model_path != model_key
                ):
                    self._piper_voice = PiperVoice.load(model_path)
                    self._piper_model_path = model_key
                    log.info("Piper TTS loaded: %s", model_path)

                syn_config = SynthesisConfig(
                    length_scale=max(
                        0.5,
                        min(
                            2.0,
                            float(self.cfg.voice.tts_piper_length_scale),
                        ),
                    ),
                    normalize_audio=True,
                    volume=1.0,
                )

                with wave.open(str(path), "wb") as wav_file:
                    self._piper_voice.synthesize_wav(
                        text,
                        wav_file,
                        syn_config=syn_config,
                    )

            self._piper_warning_shown = False
            return path.is_file() and path.stat().st_size > 44
        except Exception:
            if not self._piper_warning_shown:
                log.exception("Piper TTS synthesis failed; using fallback")
                self._piper_warning_shown = True
            return False

    def _synthesize_tts_file(self, text: str, path: Path) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

        if self._synthesize_piper_file(text, path):
            return True

        espeak = shutil.which("espeak-ng")
        if espeak:
            cmd = [
                espeak,
                "-s",
                str(max(80, min(450, int(self.cfg.voice.tts_rate)))),
                "-w",
                str(path),
            ]
            wanted = self.cfg.voice.tts_voice_name.strip()
            if wanted:
                cmd.extend(["-v", wanted])
            cmd.append(text)

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                if (
                    result.returncode == 0
                    and path.is_file()
                    and path.stat().st_size > 44
                ):
                    return True
                log.warning(
                    "espeak-ng TTS failed (code=%s): %s",
                    result.returncode,
                    (result.stderr or result.stdout).strip(),
                )
            except Exception:
                log.exception("espeak-ng TTS failed")

        engine = pyttsx3.init()
        try:
            engine.setProperty("rate", int(self.cfg.voice.tts_rate))
            wanted = self.cfg.voice.tts_voice_name.strip().lower()
            if wanted:
                for voice in engine.getProperty("voices"):
                    name = str(getattr(voice, "name", "")).lower()
                    voice_id = str(getattr(voice, "id", "")).lower()
                    if wanted in name or wanted in voice_id:
                        engine.setProperty("voice", voice.id)
                        break
            engine.save_to_file(text, str(path))
            engine.runAndWait()
        finally:
            try:
                engine.stop()
            except Exception:
                pass

        return path.is_file() and path.stat().st_size > 44

    async def _warm_stt_model(self) -> None:
        if not self.cfg.voice.stt_enabled:
            return
        if WhisperModel is None:
            self._stt_debug.update({
                "status": "ERROR",
                "error": "faster-whisper nie jest zainstalowany",
                "updated_at": time.time(),
            })
            return
        if self._stt_model is not None:
            return
        self._stt_debug.update({
            "status": "LOADING",
            "model": self.cfg.voice.stt_model,
            "device": self.cfg.voice.stt_device,
            "compute_type": self.cfg.voice.stt_compute_type,
            "error": "",
            "updated_at": time.time(),
        })
        try:
            await asyncio.to_thread(self._load_stt_model_sync)
            self._stt_debug.update({
                "status": "READY",
                "error": "",
                "updated_at": time.time(),
            })
        except Exception as exc:
            self._stt_debug.update({
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "updated_at": time.time(),
            })
            log.exception("Nie udało się załadować modelu STT")

    def _ensure_voice_listener(self, vc: discord.VoiceClient) -> None:
        if not self.cfg.voice.stt_enabled:
            return
        if voice_recv is None:
            self._stt_debug.update({
                "status": "ERROR",
                "error": "discord-ext-voice-recv nie jest zainstalowany",
                "updated_at": time.time(),
            })
            return
        if not isinstance(vc, voice_recv.VoiceRecvClient):
            self._stt_debug.update({
                "status": "WAIT_RECONNECT",
                "error": (
                    "Aktualne połączenie voice nie obsługuje odbioru audio; "
                    "potrzebny reconnect VC"
                ),
                "updated_at": time.time(),
            })
            return
        if vc.is_listening():
            return
        try:
            vc.listen(voice_recv.BasicSink(self._on_voice_pcm))
            self._stt_debug.update({
                "status": "LISTENING",
                "error": "",
                "guild": vc.guild.name,
                "channel": getattr(vc.channel, "name", None),
                "updated_at": time.time(),
            })
            log.info(
                "STT listening: %s/%s",
                vc.guild.name,
                getattr(vc.channel, "name", "voice"),
            )
        except Exception as exc:
            self._stt_debug.update({
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "updated_at": time.time(),
            })
            log.exception("Nie udało się uruchomić nasłuchu voice")

    def _set_audio_disconnected(
        self,
        guild: discord.Guild | None = None,
        *,
        reason: str = "voice_disconnected",
    ) -> None:
        # Invalidate callbacks from the AudioPlayer that belonged to the old
        # voice connection. Otherwise a late after() callback can overwrite
        # the disconnected state on the dashboard.
        self._audio_playback_token += 1
        self._audio_debug.update({
            "playback_token": self._audio_playback_token,
            "status": "IDLE",
            "stage": "disconnected",
            "playing": False,
            "connected": False,
            "error": "",
            "guild": getattr(guild, "name", self._audio_debug.get("guild")),
            "channel": None,
            "text": "",
            "disconnect_reason": reason,
            "updated_at": time.time(),
        })

    def _stop_voice_playback(self, vc: discord.VoiceClient) -> None:
        is_paused = getattr(vc, "is_paused", lambda: False)
        if not vc.is_playing() and not is_paused():
            return
        stop_playing = getattr(vc, "stop_playing", None)
        if callable(stop_playing):
            stop_playing()
        else:
            vc.stop()

    def _on_voice_pcm(self, user, data) -> None:
        if not self.cfg.voice.stt_enabled or user is None:
            return
        if getattr(user, "bot", False):
            return
        if self.user is not None and user.id == self.user.id:
            return
        pcm = getattr(data, "pcm", b"")
        if not pcm:
            return

        guild = getattr(user, "guild", None)
        voice_state = getattr(user, "voice", None)
        channel = getattr(voice_state, "channel", None)
        if guild is None or channel is None:
            return

        now = time.monotonic()
        key = (guild.id, user.id)
        max_seconds = max(
            2.0,
            float(self.cfg.voice.stt_max_segment_seconds),
        )
        hard_limit = int(max_seconds * 192000 * 1.35)

        with self._stt_buffer_lock:
            state = self._stt_buffers.get(key)
            if (
                state is None
                or state.get("channel_id") != channel.id
                or now - float(state.get("last_packet", now))
                > max(3.0, float(self.cfg.voice.stt_silence_seconds) * 4.0)
            ):
                state = {
                    "guild_id": guild.id,
                    "user_id": user.id,
                    "channel_id": channel.id,
                    "started": now,
                    "last_packet": now,
                    "pcm": bytearray(),
                }
                self._stt_buffers[key] = state

            state["last_packet"] = now
            state["pcm"].extend(pcm)
            if len(state["pcm"]) > hard_limit:
                del state["pcm"][: len(state["pcm"]) - hard_limit]

    def _load_stt_model_sync(self):
        if WhisperModel is None:
            raise RuntimeError(
                "faster-whisper nie jest zainstalowany"
            )

        model_key = (
            self.cfg.voice.stt_model,
            self.cfg.voice.stt_device,
            self.cfg.voice.stt_compute_type,
            int(self.cfg.voice.stt_cpu_threads),
            self.cfg.voice.stt_download_root,
        )
        with self._stt_model_lock:
            if (
                self._stt_model is not None
                and self._stt_model_key == model_key
            ):
                return self._stt_model

            download_root = Path(self.cfg.voice.stt_download_root)
            download_root.mkdir(parents=True, exist_ok=True)
            log.info(
                "Ładowanie STT model=%s device=%s compute=%s",
                self.cfg.voice.stt_model,
                self.cfg.voice.stt_device,
                self.cfg.voice.stt_compute_type,
            )
            model = WhisperModel(
                self.cfg.voice.stt_model,
                device=self.cfg.voice.stt_device,
                compute_type=self.cfg.voice.stt_compute_type,
                cpu_threads=max(
                    1,
                    int(self.cfg.voice.stt_cpu_threads),
                ),
                num_workers=1,
                download_root=str(download_root),
            )
            self._stt_model = model
            self._stt_model_key = model_key
            return model

    @staticmethod
    def _discord_pcm_to_whisper(pcm: bytes) -> np.ndarray:
        if len(pcm) < 4:
            return np.empty(0, dtype=np.float32)
        usable = len(pcm) - (len(pcm) % 4)
        samples = np.frombuffer(
            pcm[:usable],
            dtype="<i2",
        ).reshape(-1, 2)
        mono = samples.astype(np.float32).mean(axis=1)
        audio = resample_poly(mono, 1, 3).astype(np.float32)
        audio /= 32768.0
        return np.clip(audio, -1.0, 1.0)

    def _transcribe_pcm_sync(
        self,
        pcm: bytes,
    ) -> tuple[str, str | None, float | None]:
        audio = self._discord_pcm_to_whisper(pcm)
        if audio.size < 1600:
            return "", None, None

        model = self._load_stt_model_sync()
        language = self.cfg.voice.stt_language.strip() or None
        segments, info = model.transcribe(
            audio,
            language=language,
            beam_size=max(1, int(self.cfg.voice.stt_beam_size)),
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text.strip()
        ).strip()
        return (
            text,
            getattr(info, "language", None),
            getattr(info, "language_probability", None),
        )

    async def _transcribe_voice_segment(
        self,
        segment: dict,
        pcm: bytes,
        duration: float,
    ) -> None:
        try:
            guild = self.get_guild(int(segment["guild_id"]))
            if guild is None:
                return
            member = guild.get_member(int(segment["user_id"]))
            if member is None or member.bot:
                return
            channel = guild.get_channel(int(segment["channel_id"]))

            self._stt_debug.update({
                "status": "TRANSCRIBING",
                "user": member.display_name,
                "user_id": member.id,
                "guild": guild.name,
                "channel": getattr(channel, "name", str(segment["channel_id"])),
                "duration": duration,
                "error": "",
                "updated_at": time.time(),
            })

            async with self._stt_inference_lock:
                text, language, probability = await asyncio.to_thread(
                    self._transcribe_pcm_sync,
                    pcm,
                )

            text = re.sub(r"\s+", " ", text).strip()
            if len(text) < max(1, int(self.cfg.voice.stt_min_chars)):
                self._stt_debug.update({
                    "status": "NO_SPEECH",
                    "text": text,
                    "language": language,
                    "language_probability": probability,
                    "updated_at": time.time(),
                })
                return

            self._stt_debug.update({
                "status": "HEARD",
                "text": text,
                "language": language,
                "language_probability": probability,
                "updated_at": time.time(),
            })
            self._stt_transcripts_since_start += 1
            await self._handle_voice_transcript(
                guild,
                member,
                channel,
                text,
            )
        except Exception as exc:
            self._stt_debug.update({
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "updated_at": time.time(),
            })
            log.exception("Błąd rozpoznawania mowy z Discord voice")
        finally:
            self._stt_pending = max(0, self._stt_pending - 1)
            self._stt_debug["pending"] = self._stt_pending

    async def _handle_voice_transcript(
        self,
        guild: discord.Guild,
        member: discord.Member,
        channel,
        text: str,
    ) -> None:
        normalized = OnlineLanguage.normalize(text).lower()
        mentioned = bool(
            re.search(r"\bmucha\b", normalized, flags=re.UNICODE)
        )
        channel_name = getattr(channel, "name", "voice")

        self.last_text_context[guild.id] = text
        self.last_text_author[guild.id] = member.id
        self.language.learn(text)

        affinity = self._user_affinity(member.id)
        async with self._brain_lock:
            self.brain.inject(
                "voice:speech-heard",
                0.75,
                144,
            )
            self.brain.inject(
                f"voice:speech:user:{member.id}",
                0.55,
                96,
            )
            self.brain.inject_text(
                text,
                member.id,
                mentioned,
            )
            if affinity >= float(
                self.cfg.behavior.familiar_affinity_threshold
            ):
                self.brain.inject(
                    "social:familiar-user",
                    min(1.0, 0.35 + abs(affinity)),
                    128,
                )
            if affinity >= 0.35:
                self.brain.inject(
                    "social:liked-user",
                    min(1.0, affinity),
                    128,
                )
            elif affinity <= float(
                self.cfg.behavior.user_avoid_threshold
            ):
                self.brain.inject(
                    "social:disliked-user",
                    min(1.0, abs(affinity)),
                    128,
                )
            self.brain.step(self.cfg.brain.steps_per_event)

        self._last_brain_event = (
            f"VOICE SPEECH • {member.display_name} • "
            f"{channel_name}: {text[:80]}"
        )
        self._record_action(
            "voice_heard",
            f"{member.display_name} • {channel_name}: {text[:140]}",
            guild,
        )

        last_tts = self._last_tts_trace.get(guild.id)
        recent_tts = bool(
            last_tts is not None
            and last_tts.channel_id == getattr(channel, "id", None)
            and time.monotonic() - last_tts.created <= 120.0
        )

        rejection = self._detect_verbal_rejection(text)
        targeted_rejection = bool(
            rejection
            and (
                mentioned
                or (
                    recent_tts
                    and time.monotonic() - last_tts.created <= 20.0
                )
            )
        )
        if targeted_rejection and rejection is not None:
            label, severity = rejection
            streak_count, streak_multiplier = self._advance_negative_streak(
                member.id
            )
            affinity_delta = -min(
                0.30,
                max(
                    0.01,
                    float(self.cfg.behavior.user_affinity_negative_step)
                    * (0.45 + 1.05 * severity)
                    * streak_multiplier,
                ),
            )
            new_affinity = self.language.adjust_user_affinity(
                member.id,
                member.display_name,
                affinity_delta,
                "negative",
            )
            penalty = -min(
                0.35,
                (0.06 + 0.24 * severity) * streak_multiplier,
            )
            async with self._brain_lock:
                self._write_neural_social_memory(
                    member.id,
                    affinity_delta,
                )
                new_affinity = self._user_affinity(member.id)
                self.brain.inject(
                    "social:user-told-me-stop",
                    0.55 + 0.65 * severity,
                    160,
                )
                self.brain.inject(
                    "social:user-rejected-me",
                    0.50 + 0.70 * severity,
                    160,
                )
                self.brain.inject(
                    "internal:social-failure",
                    0.35 + 0.65 * severity,
                    128,
                )
                self.brain.inject(
                    "voice:spoken-rejection",
                    0.60 + 0.60 * severity,
                    144,
                )
                if streak_count >= 2:
                    self.brain.inject(
                        "social:repeated-rejection",
                        min(1.0, 0.52 + 0.08 * streak_count),
                        144,
                    )
                    self.brain.inject(
                        f"social:repeated-rejection:user:{member.id}",
                        min(1.0, 0.45 + 0.07 * streak_count),
                        96,
                    )
                if recent_tts and last_tts is not None:
                    self.brain.reward(
                        penalty,
                        action=last_tts.action,
                        trace=last_tts.learning_trace,
                    )
                else:
                    self.brain.reward(penalty)
                self.brain.step(2)
            self._record_reward(
                penalty,
                last_tts.action if recent_tts and last_tts else None,
                f"spoken rejection • {label}",
                guild,
            )
            self._remember_social_event(
                "VOICE_REJECTION",
                (
                    f"{label} • streak {streak_count} "
                    f"×{streak_multiplier:.2f} • affinity {new_affinity:+.2f}"
                ),
                affinity_delta,
                member,
            )
            return

        if mentioned:
            await self._grant_positive_social(
                member,
                "VOICE_MENTION",
                "social:user-mentioned-me",
                self.cfg.behavior.mention_affinity_step,
                guild,
                detail="powiedział Mucha na voice",
            )

        if not recent_tts or last_tts is None:
            return

        heard_words = self._social_words(text)
        tts_words = self._social_words(last_tts.text)
        heard_set = set(heard_words)
        phrase_match = None
        for size in (3, 2):
            heard_phrases = {
                tuple(heard_words[i:i + size])
                for i in range(max(0, len(heard_words) - size + 1))
            }
            for i in range(max(0, len(tts_words) - size + 1)):
                phrase = tuple(tts_words[i:i + size])
                if (
                    phrase in heard_phrases
                    and sum(len(word) for word in phrase) >= 8
                ):
                    phrase_match = phrase
                    break
            if phrase_match is not None:
                break

        if phrase_match is not None:
            phrase_text = " ".join(phrase_match)
            amount = float(self.cfg.behavior.phrase_reuse_reward)
            self.language.reinforce_text(phrase_text, amount)
            async with self._brain_lock:
                self.brain.inject("social:phrase-reused", 0.65, 128)
                self.brain.inject("voice:phrase-reused", 0.60, 112)
                self.brain.reward(
                    min(0.20, amount * 0.35),
                    action=last_tts.action,
                    trace=last_tts.learning_trace,
                )
                self.brain.step(1)
            await self._grant_positive_social(
                member,
                "VOICE_PHRASE_REUSE",
                "social:user-reused-phrase",
                self.cfg.behavior.phrase_reuse_affinity_step,
                guild,
                detail=phrase_text,
            )
            self._remember_social_event(
                "VOICE_PHRASE_REUSE",
                phrase_text,
                amount,
                member,
            )
            return

        shared_words = [
            word
            for word in set(tts_words) & heard_set
            if len(word) >= 5
        ]
        if shared_words:
            word = max(shared_words, key=len)
            amount = float(self.cfg.behavior.word_reuse_reward)
            self.language.reinforce_text(word, amount)
            self.language.record_word_feedback(
                word,
                member.id,
                amount,
            )
            async with self._brain_lock:
                self.brain.inject("social:word-reused", 0.65, 128)
                self.brain.inject("voice:word-reused", 0.60, 112)
                self.brain.reward(
                    min(0.15, amount * 0.35),
                    action=last_tts.action,
                    trace=last_tts.learning_trace,
                )
                self.brain.step(1)
            await self._grant_positive_social(
                member,
                "VOICE_WORD_REUSE",
                "social:user-reused-word",
                self.cfg.behavior.word_reuse_affinity_step,
                guild,
                detail=word,
            )
            self._remember_social_event(
                "VOICE_WORD_REUSE",
                word,
                amount,
                member,
            )
            return

        await self._grant_positive_social(
            member,
            "VOICE_CONVERSATION",
            "social:user-continued-conversation",
            self.cfg.behavior.continued_conversation_affinity_step,
            guild,
            detail="odpowiedział głosem po TTS",
            source_trace=last_tts,
            brain_reward=0.025,
        )

    @tasks.loop(seconds=0.25)
    async def stt_segment_loop(self):
        await self.wait_until_ready()
        if self.paused or not self.cfg.voice.stt_enabled:
            return

        now = time.monotonic()
        silence = max(
            0.2,
            float(self.cfg.voice.stt_silence_seconds),
        )
        minimum = max(
            0.2,
            float(self.cfg.voice.stt_min_segment_seconds),
        )
        maximum = max(
            minimum,
            float(self.cfg.voice.stt_max_segment_seconds),
        )
        ready: list[tuple[dict, bytes, float]] = []

        with self._stt_buffer_lock:
            for key, state in list(self._stt_buffers.items()):
                pcm = state["pcm"]
                duration = len(pcm) / 192000.0
                quiet_for = now - float(state["last_packet"])
                should_flush = (
                    duration >= maximum
                    or quiet_for >= silence
                )
                if not should_flush:
                    continue

                self._stt_buffers.pop(key, None)
                if duration >= minimum:
                    ready.append(
                        (dict(state), bytes(pcm), duration)
                    )

        for segment, pcm, duration in ready:
            if self._stt_pending >= 4:
                self._stt_debug.update({
                    "status": "DROP_BUSY",
                    "error": "kolejka STT pełna",
                    "updated_at": time.time(),
                })
                continue
            self._stt_pending += 1
            self._stt_debug["pending"] = self._stt_pending
            asyncio.create_task(
                self._transcribe_voice_segment(
                    segment,
                    pcm,
                    duration,
                )
            )

    @stt_segment_loop.before_loop
    async def before_stt_segment_loop(self):
        await self.wait_until_ready()

    async def _play_chaser_scream(
        self,
        guild: discord.Guild,
        vc: discord.VoiceClient,
    ) -> None:
        if (
            not self.cfg.voice.chaser_scream_enabled
            or not vc.is_connected()
            or vc.channel is None
        ):
            return

        source_path = Path(self.cfg.voice.chaser_scream_file)
        scream_text = self.cfg.voice.chaser_scream_text.strip() or "AAAAAAAA!"
        using_file = source_path.is_file()

        if not using_file:
            source_path = (
                Path("state")
                / "tts"
                / f"chaser_scream_{guild.id}.wav"
            )
            ok = await asyncio.to_thread(
                self._synthesize_tts_file,
                scream_text,
                source_path,
            )
            if not ok:
                self._record_action(
                    "chaser_scream_error",
                    "nie udało się wygenerować fallback TTS",
                    guild,
                )
                return

        try:
            if vc.is_playing():
                self._stop_voice_playback(vc)

            async with self._brain_lock:
                self.brain.inject("internal:panic-scream", 1.6, 128)
                self.brain.inject(
                    f"voice:panic-scream:guild:{guild.id}",
                    1.1,
                    96,
                )
                self.brain.step(1)

            source = self._make_voice_source(
                source_path,
                self.cfg.voice.chaser_scream_volume,
            )
            playback_token = self._play_voice_source(
                vc,
                source,
                "chaser_scream",
            )
            asyncio.create_task(
                self._verify_voice_playback(
                    vc,
                    "chaser_scream",
                    playback_token,
                )
            )

            self._audio_debug.update({
                "status": "PLAYING",
                "stage": "chaser_scream",
                "error": "",
                "guild": guild.name,
                "channel": getattr(vc.channel, "name", "voice"),
                "file": str(source_path),
                "file_size": (
                    source_path.stat().st_size
                    if source_path.is_file()
                    else 0
                ),
                "text": "" if using_file else scream_text,
                "updated_at": time.time(),
            })
            self._record_action(
                "chaser_scream",
                (
                    f"{getattr(vc.channel, 'name', 'voice')} • "
                    + (
                        source_path.name
                        if using_file
                        else f"TTS {scream_text}"
                    )
                ),
                guild,
            )
        except Exception as exc:
            self._audio_debug.update({
                "status": "ERROR",
                "stage": "chaser_scream",
                "error": f"{type(exc).__name__}: {exc}",
                "updated_at": time.time(),
            })
            self._record_action(
                "chaser_scream_error",
                f"{type(exc).__name__}: {exc}",
                guild,
            )
            log.exception(
                "Nie udało się odtworzyć krzyku po ucieczce na serwerze %s",
                guild.id,
            )

    @tasks.loop(seconds=10)
    async def tts_loop(self):
        await self.wait_until_ready()
        if (
            self.paused
            or not self.cfg.voice.tts_enabled
            or not self.language.ready()
        ):
            return

        candidates = [
            vc for vc in self.voice_clients
            if vc.is_connected()
            and vc.channel is not None
            and not vc.is_playing()
            and self._chaser_panic_remaining(vc.guild.id) <= 0.0
            and not (
                self.cfg.behavior.avoid_disliked_users_on_voice
                and self._disliked_members(
                    [m for m in vc.channel.members if not m.bot]
                )
            )
        ]
        if not candidates:
            return

        vc = self.random.choice(candidates)
        guild = vc.guild
        channel_name = getattr(vc.channel, "name", "voice")
        last_author = self.last_text_author.get(guild.id)
        if (
            self.cfg.behavior.ignore_disliked_users_text
            and last_author is not None
            and self._is_disliked_user(last_author)
        ):
            return

        context = self.last_text_context.get(guild.id, "")
        async with self._brain_lock:
            self.brain.inject(
                f"voice:tts-opportunity:guild:{guild.id}",
                0.18,
                64,
            )
            self.brain.step(1)
            scores = self.brain.action_scores()
            if scores["speak"] < self.cfg.behavior.speak_threshold:
                return

            internal = self.brain.internal_state_diagnostics()
            neural_arousal = float(
                internal.get("states", {})
                .get("arousal", {})
                .get("level", 0.0)
            )
            text_out, trigrams = self.language.generate(
                context=context,
                arousal=max(
                    0.0,
                    min(
                        1.0,
                        0.80 * neural_arousal
                        + 0.20 * float(scores["explore"]),
                    ),
                ),
                brain_word_score=self.brain.language_word_score,
                brain_word_feedback=self._brain_word_feedback,
            )
            if text_out:
                text_out = text_out[
                    : max(8, int(self.cfg.voice.tts_max_chars))
                ].strip()
            if text_out:
                self.brain.mark_language_output(text_out)
                learning_trace = self.brain.capture_learning_trace()
            else:
                learning_trace = None

        if not text_out or learning_trace is None:
            return

        wav_path = Path("state") / "tts" / f"{guild.id}.wav"
        self._audio_debug.update({
            "status": "TTS",
            "stage": "synthesize",
            "error": "",
            "guild": guild.name,
            "channel": channel_name,
            "file": str(wav_path),
            "file_size": 0,
            "text": text_out,
            "updated_at": time.time(),
        })
        try:
            ok = await asyncio.to_thread(
                self._synthesize_tts_file,
                text_out,
                wav_path,
            )
            self._audio_debug["file_size"] = (
                wav_path.stat().st_size if wav_path.is_file() else 0
            )
            if not ok:
                self._audio_debug.update({
                    "status": "ERROR",
                    "stage": "synthesize",
                    "error": "TTS nie utworzył poprawnego WAV",
                    "updated_at": time.time(),
                })
                return
            if vc.is_playing() or not vc.is_connected():
                self._audio_debug.update({
                    "status": "SKIP",
                    "stage": "playback",
                    "error": "VC zajęty albo rozłączony",
                    "updated_at": time.time(),
                })
                return

            self._audio_debug.update({
                "status": "TTS",
                "stage": "ffmpeg_source",
                "updated_at": time.time(),
            })
            source = self._make_voice_source(
                wav_path,
                self.cfg.voice.tts_volume,
            )
            playback_token = self._play_voice_source(
                vc,
                source,
                "tts",
            )
            self._last_tts_trace[guild.id] = SentTrace(
                trigrams=trigrams,
                created=time.monotonic(),
                action="speak",
                learning_trace=learning_trace,
                text=text_out,
                guild_id=guild.id,
                channel_id=vc.channel.id,
            )
            self._last_tts_audience[guild.id] = {
                member.id
                for member in vc.channel.members
                if not member.bot
            }
            self._schedule_tts_social_stay(
                guild,
                vc.channel.id,
                [
                    member.id
                    for member in vc.channel.members
                    if not member.bot
                ],
            )
            asyncio.create_task(
                self._verify_voice_playback(
                    vc,
                    "tts",
                    playback_token,
                )
            )
            self._audio_debug.update({
                "status": "PLAYING",
                "stage": "vc.play",
                "error": "",
                "updated_at": time.time(),
            })

            self._last_brain_event = (
                f"TTS • {guild.name} • {channel_name}"
            )
            self._last_brain_action = (
                f"TTS SPEAK → {guild.name}/{channel_name}: "
                f"{text_out[:80]}"
            )
            self._set_reinforceable(
                guild,
                "speak",
                learning_trace,
                f"TTS → {channel_name}: {text_out[:80]}",
            )
            self._record_action(
                "tts_speak",
                f"{channel_name}: {text_out[:120]}",
                guild,
            )
        except Exception as exc:
            self._audio_debug.update({
                "status": "ERROR",
                "stage": self._audio_debug.get("stage", "tts"),
                "error": f"{type(exc).__name__}: {exc}",
                "updated_at": time.time(),
            })
            log.exception(
                "Nie udało się wygenerować lub odtworzyć TTS na serwerze %s",
                guild.id,
            )

    @tts_loop.before_loop
    async def before_tts(self):
        await self.wait_until_ready()

    @tasks.loop(seconds=1)
    async def random_audio_loop(self):
        await self.wait_until_ready()
        if self.paused or not self.cfg.voice.random_audio_enabled:
            return

        candidates = [
            vc for vc in self.voice_clients
            if vc.is_connected()
            and vc.channel is not None
            and not vc.is_playing()
            and self._chaser_panic_remaining(vc.guild.id) <= 0.0
        ]
        if not candidates:
            return

        audio_path = Path(self.cfg.voice.random_audio_file)
        if not audio_path.is_file():
            if not self._random_audio_missing_warned:
                log.warning("Brak pliku losowego audio: %s", audio_path)
                self._random_audio_missing_warned = True
            return

        self._random_audio_missing_warned = False
        denominator = max(
            1,
            int(self.cfg.voice.random_audio_chance_denominator),
        )
        if self.random.randrange(denominator) != 0:
            return

        vc = self.random.choice(candidates)
        guild = vc.guild
        channel_name = getattr(vc.channel, "name", "voice")

        try:
            source = self._make_voice_source(
                audio_path,
                self.cfg.voice.random_audio_volume,
            )
            async with self._brain_lock:
                self.brain.inject("internal:rare-audio", 1.0, 128)
                self.brain.inject(
                    f"voice:rare-audio:guild:{guild.id}",
                    0.65,
                    96,
                )
                self.brain.step(2)

            playback_token = self._play_voice_source(
                vc,
                source,
                "rare_audio",
            )
            asyncio.create_task(
                self._verify_voice_playback(
                    vc,
                    "rare_audio",
                    playback_token,
                )
            )
            self._last_brain_event = (
                f"RARE AUDIO • {guild.name} • {channel_name}"
            )
            self._last_brain_action = (
                f"AUDIO 1/{denominator} → {guild.name}/{channel_name}"
            )
            self._record_action(
                "rare_audio",
                f"1/{denominator} → {channel_name} • {audio_path.name}",
                guild,
            )
        except Exception:
            log.exception(
                "Nie udało się odtworzyć losowego audio na serwerze %s",
                guild.id,
            )

    @random_audio_loop.before_loop
    async def before_random_audio(self):
        await self.wait_until_ready()

    @tasks.loop(seconds=15)
    async def voice_loop(self):
        await self.wait_until_ready()
        if self.paused:
            return
        for guild in self.guilds:
            try:
                await self._voice_decision(guild)
                vc = guild.voice_client
                if vc is not None and vc.is_connected():
                    self._ensure_voice_listener(vc)
                if (
                    vc is not None
                    and vc.is_connected()
                    and vc.channel is not None
                    and self._chaser_panic_remaining(guild.id) <= 0.0
                ):
                    for member in vc.channel.members:
                        if not member.bot:
                            self._schedule_voice_social_stay(
                                guild,
                                member,
                                vc.channel.id,
                            )
            except Exception:
                log.exception("Błąd autonomii voice na serwerze %s", guild.id)

    @voice_loop.before_loop
    async def before_voice(self):
        await self.wait_until_ready()

    async def _voice_decision(self, guild: discord.Guild):
        me = guild.me
        if not me:
            self._voice_debug[guild.id] = {
                "guild": guild.name,
                "guild_id": guild.id,
                "decision": "BRAK BOT MEMBER",
                "reason": "Discord nie zwrócił guild.me",
                "channels": [],
            }
            return

        now = time.monotonic()
        vc = guild.voice_client
        current = (
            self._current_voice_channel(guild)
            if vc is not None and vc.is_connected()
            else None
        )
        connectome_voice_control = bool(
            self.cfg.voice.connectome_voice_control_enabled
        )

        if self._is_voice_guild_blocked(guild):
            if vc is not None and vc.is_connected():
                try:
                    await vc.disconnect(force=False)
                    self._record_action(
                        "voice_guild_block",
                        (
                            f"opuszczono {getattr(current, 'name', 'voice')} • "
                            "serwer ma zakaz voice"
                        ),
                        guild,
                    )
                except (discord.Forbidden, discord.HTTPException):
                    log.exception(
                        "Nie udało się opuścić zablokowanego serwera voice %s",
                        guild.id,
                    )
            self._voice_debug[guild.id] = {
                "guild": guild.name,
                "guild_id": guild.id,
                "enabled": False,
                "current": None,
                "decision": "⛔ ZAKAZ VOICE NA SERWERZE",
                "reason": (
                    f"guild {guild.id} jest na blocked_voice_guild_ids"
                ),
                "channels": [],
                "checked_at": time.time(),
            }
            return
        chaser_remaining = self._chaser_panic_remaining(guild.id, now)
        chaser_id = self._chaser_confirmed.get(guild.id)
        chaser_active = chaser_remaining > 0.0
        arrived = self.voice_arrived.setdefault(
            guild.id,
            now,
        )
        if current is not None:
            self._voice_last_visit.setdefault((guild.id, current.id), arrived)
        dwell_elapsed = max(0.0, now - arrived)
        dwell_remaining = max(0.0, self.cfg.voice.minimum_dwell_seconds - dwell_elapsed)

        max_dwell = max(
            float(self.cfg.voice.minimum_dwell_seconds),
            float(self.cfg.voice.maximum_dwell_seconds),
        )
        overstay_seconds = max(0.0, dwell_elapsed - max_dwell)
        threat_active = bool(current and dwell_elapsed >= max_dwell)
        threat_level = 0.0
        if threat_active:
            ramp = max(1.0, float(self.cfg.voice.threat_ramp_seconds))
            threat_level = min(1.0, 0.25 + 0.75 * (overstay_seconds / ramp))

        episodic_diag = self.voice_episodes.diagnostics()

        debug = {
            "guild": guild.name,
            "guild_id": guild.id,
            "enabled": self.cfg.voice.enabled,
            "connectome_voice_control": connectome_voice_control,
            "decision_source": (
                "connectome-readout-competition"
                if connectome_voice_control
                else "legacy-thresholds"
            ),
            "brain_decision": None,
            "current": current.name if current else None,
            "decision": "ANALIZA",
            "reason": "oczekiwanie na wynik",
            "join_threshold": self.cfg.voice.join_threshold,
            "move_threshold": self.cfg.voice.move_threshold,
            "leave_threshold": self.cfg.voice.leave_threshold,
            "move_margin": self.cfg.voice.move_margin,
            "minimum_dwell_seconds": self.cfg.voice.minimum_dwell_seconds,
            "maximum_dwell_seconds": self.cfg.voice.maximum_dwell_seconds,
            "dwell_elapsed": (
                dwell_elapsed
                if current is not None
                else 0.0
            ),
            "dwell_remaining": (
                dwell_remaining
                if current is not None
                else 0.0
            ),
            "overstay_seconds": (
                overstay_seconds
                if current is not None
                else 0.0
            ),
            "overstay_punished": False,
            "outside_seconds": (
                dwell_elapsed
                if current is None
                else 0.0
            ),
            "available_humans": 0,
            "social_drive_active": False,
            "social_drive_level": 0.0,
            "homeostasis_enabled": bool(
                self.cfg.voice.homeostasis_enabled
            ),
            "voice_scene_age": 0.0,
            "social_fatigue_level": 0.0,
            "habituation_level": 0.0,
            "habituation_suppression": 0.0,
            "exploration_drive_level": 0.0,
            "homeostasis_guided": {},
            "internal_states": {},
            "internal_state_cues": {},
            "prediction_context": None,
            "prediction_scene_key": None,
            "prediction_expected": {},
            "episodic_recall": {},
            "episodic_persistent": bool(
                self.voice_episodes.db is not None
            ),
            "episodic_database": str(
                self.voice_episodes.path or ""
            ),
            "episodic_prediction_count": (
                self.voice_episodes.prediction_count()
            ),
            "episodic_recent": self.voice_episodes.recent(6),
            "episodic_memory_scenes": int(
                episodic_diag.get("memory_scenes", 0)
            ),
            "episodic_consolidated_scenes": int(
                episodic_diag.get("consolidated_scenes", 0)
            ),
            "episodic_top_memories": list(
                episodic_diag.get("top_memories", [])
            )[:6],
            "episodic_forgetting": dict(
                episodic_diag.get("forgetting", {})
            ),
            "predicted_reward": 0.0,
            "prediction_error": None,
            "prediction_correction_applied": 0.0,
            "episodic_memory_size": self.voice_episodes.size(),
            "social_drive_stay_punished": False,
            "social_drive_stay_punish_amount": 0.0,
            "social_join_reward": 0.0,
            "reward_opportunity_channel": None,
            "reward_opportunity_strength": 0.0,
            "reward_opportunity_effective_strength": 0.0,
            "reward_opportunity_remaining": 0.0,
            "reward_opportunity_found": False,
            "reward_opportunity_reward": 0.0,
            "reward_opportunity_guided_mode": None,
            "reward_opportunity_guided_neurons": 0,
            "reward_opportunity_guided_reach_mean": 0.0,
            "reward_opportunity_guided_reach_max": 0.0,
            "reward_opportunity_stay_punished": False,
            "reward_opportunity_stay_punish_amount": 0.0,
            "motivation_propagation_steps": 0,
            "social_drive_guided_mode": None,
            "social_drive_guided_neurons": 0,
            "social_drive_guided_reach_mean": 0.0,
            "social_drive_guided_reach_max": 0.0,
            "threat_active": threat_active,
            "threat_level": threat_level,
            "chaser_active": chaser_active,
            "chaser_remaining": chaser_remaining,
            "chaser_id": chaser_id,
            "threat_magnitude": 0.0,
            "effective_move_score": None,
            "effective_move_margin": None,
            "escape_target": None,
            "scores": {},
            "channels": [],
            "checked_at": time.time(),
        }

        channels = []
        for ch in guild.voice_channels:
            is_afk = bool(
                self.cfg.voice.exclude_afk_channel
                and guild.afk_channel
                and ch.id == guild.afk_channel.id
            )
            perms = ch.permissions_for(me)
            humans = [m for m in ch.members if not m.bot]
            disliked_members = self._disliked_members(humans)
            social_blocked = bool(
                not connectome_voice_control
                and self.cfg.behavior.avoid_disliked_users_on_voice
                and disliked_members
                and not chaser_active
            )
            view_ok = bool(perms.view_channel)
            connect_ok = bool(perms.connect)
            include_ok = bool(humans or self.cfg.voice.include_empty_channels)
            deadly_remaining = self._deadly_voice_remaining(
                guild.id,
                ch.id,
                now,
            )
            deadly = deadly_remaining > 0.0
            blocked_voice = self._is_voice_channel_blocked(ch)
            chaser_here = bool(
                chaser_active
                and chaser_id
                and any(m.id == chaser_id for m in ch.members)
            )
            eligible = bool(
                not is_afk
                and not deadly
                and not blocked_voice
                and not chaser_here
                and not social_blocked
                and view_ok
                and connect_ok
                and include_ok
            )

            row = {
                "id": ch.id,
                "name": ch.name,
                "humans": len(humans),
                "view": view_ok,
                "connect": connect_ok,
                "afk": is_afk,
                "eligible": eligible,
                "deadly": deadly,
                "deadly_remaining": deadly_remaining,
                "blocked_voice": blocked_voice,
                "chaser_here": chaser_here,
                "social_blocked": social_blocked,
                "disliked_users": [
                    {
                        "id": member.id,
                        "name": member.display_name,
                        "affinity": affinity,
                    }
                    for member, affinity in disliked_members
                ],
                "user_affinity_min": (
                    min((affinity for _, affinity in disliked_members), default=None)
                ),
                "affinity": None,
                "exploration_score": None,
                "visit_age": None,
                "novelty": None,
                "current": bool(current and current.id == ch.id),
                "status": "OK" if eligible else (
                    "⛔ BLOKADA" if blocked_voice else
                    "🕷 CHASER" if chaser_here else
                    (
                        "🙅 NIELUBI " + ", ".join(
                            member.display_name
                            for member, _ in disliked_members[:2]
                        )
                    ) if social_blocked else
                    f"☠ ŚMIERTELNE {deadly_remaining:.0f}s" if deadly else
                    "AFK" if is_afk else
                    "BRAK VIEW" if not view_ok else
                    "BRAK CONNECT" if not connect_ok else
                    "PUSTY WYŁĄCZONY"
                ),
            }
            debug["channels"].append(row)
            if eligible:
                channels.append((ch, humans))

        if (
            chaser_active
            and chaser_id
            and current is not None
            and any(m.id == chaser_id for m in current.members)
        ):
            async with self._brain_lock:
                magnitude = float(self.cfg.voice.chaser_threat_magnitude)
                self.brain.inject("internal:predator-chaser", magnitude, 192)
                self.brain.inject(
                    f"voice:predator:{guild.id}:{chaser_id}",
                    magnitude,
                    160,
                )
                self.brain.step(2)
                chaser_trace = self.brain.capture_learning_trace()
            self._schedule_chaser_escape(
                guild,
                chaser_id,
                chaser_trace,
            )
            debug["decision"] = "CHASE • UCIEKAM"
            debug["reason"] = (
                f"Chaser {chaser_id} jest na obecnym kanale; "
                f"panic {chaser_remaining:.1f}s"
            )
            # Panic has absolute priority over dwell/stay/normal voice logic.
            # Once the chaser catches Mucha, only the chase task may decide
            # what happens next.
            self._voice_debug[guild.id] = debug
            return

        current_disliked = (
            self._disliked_members(
                [m for m in current.members if not m.bot]
            )
            if current is not None and not chaser_active
            else []
        )
        debug["social_avoid_active"] = bool(current_disliked)
        debug["social_avoid_users"] = [
            {
                "id": member.id,
                "name": member.display_name,
                "affinity": affinity,
            }
            for member, affinity in current_disliked
        ]

        if not channels:
            if (
                not connectome_voice_control
                and current is not None
                and current_disliked
                and vc is not None
                and vc.is_connected()
            ):
                try:
                    await vc.disconnect(force=False)
                    names = ", ".join(
                        member.display_name
                        for member, _ in current_disliked
                    )
                    debug["decision"] = "SOCIAL AVOID • LEAVE"
                    debug["reason"] = (
                        f"nielubiany użytkownik na kanale: {names}; "
                        "brak bezpiecznego kanału"
                    )
                    self._record_action(
                        "social_voice_leave",
                        f"{current.name} • {names}",
                        guild,
                    )
                except (discord.Forbidden, discord.HTTPException):
                    log.exception(
                        "Nie udało się opuścić kanału podczas social avoid"
                    )
            else:
                debug["decision"] = "NIE WCHODZĘ"
                debug["reason"] = "brak dostępnych kanałów voice"
            self._voice_debug[guild.id] = debug
            return

        available_human_ids = {
            int(member.id)
            for _, humans in channels
            for member in humans
        }
        available_humans = len(available_human_ids)
        outside_seconds = (
            dwell_elapsed
            if current is None
            else 0.0
        )
        social_drive_level = 0.0
        if (
            connectome_voice_control
            and self.cfg.voice.social_drive_enabled
            and current is None
            and available_humans > 0
            and not chaser_active
        ):
            start_after = max(
                0.0,
                float(
                    self.cfg.voice.social_drive_start_seconds
                ),
            )
            ramp = max(
                1.0,
                float(
                    self.cfg.voice.social_drive_ramp_seconds
                ),
            )
            social_drive_level = max(
                0.0,
                min(
                    1.0,
                    (outside_seconds - start_after) / ramp,
                ),
            )

        debug["outside_seconds"] = outside_seconds
        debug["available_humans"] = available_humans
        debug["social_drive_level"] = social_drive_level
        debug["social_drive_active"] = bool(
            social_drive_level > 0.0
        )

        reward_opportunity = None
        if (
            connectome_voice_control
            and current is None
            and not chaser_active
        ):
            reward_opportunity = (
                self._voice_reward_opportunity_for(
                    guild.id,
                    channels,
                    now,
                )
            )
        elif current is not None:
            self._voice_reward_opportunity.pop(
                guild.id,
                None,
            )

        opportunity_effective_strength = 0.0
        if reward_opportunity is not None:
            debug["reward_opportunity_channel"] = (
                reward_opportunity["channel_name"]
            )
            debug["reward_opportunity_strength"] = float(
                reward_opportunity["strength"]
            )
            opportunity_effective_strength = min(
                4.0,
                float(reward_opportunity["strength"])
                * (
                    1.0
                    + 0.75 * social_drive_level
                ),
            )
            debug["reward_opportunity_effective_strength"] = (
                opportunity_effective_strength
            )
            debug["reward_opportunity_remaining"] = max(
                0.0,
                float(
                    reward_opportunity["expires_at"]
                )
                - now,
            )
            for row in debug["channels"]:
                if int(row["id"]) == int(
                    reward_opportunity["channel_id"]
                ):
                    row["reward_opportunity"] = True
                    row["reward_opportunity_strength"] = float(
                        reward_opportunity["strength"]
                    )

        alternatives_count = sum(
            1
            for ch, _ in channels
            if current is None or ch.id != current.id
        )
        current_humans = (
            [m for m in current.members if not m.bot]
            if current is not None
            else []
        )
        homeostasis = self._voice_homeostasis_levels(
            guild.id,
            current.id if current is not None else None,
            [m.id for m in current_humans],
            now=now,
            dwell_elapsed=dwell_elapsed,
            outside_seconds=outside_seconds,
            alternatives=alternatives_count,
        )
        debug["voice_scene_age"] = float(
            homeostasis["scene_age"]
        )
        debug["social_fatigue_level"] = float(
            homeostasis["social_fatigue"]
        )
        debug["habituation_level"] = float(
            homeostasis["habituation"]
        )
        debug["habituation_suppression"] = float(
            homeostasis["habituation_suppression"]
        )
        debug["exploration_drive_level"] = float(
            homeostasis["exploration"]
        )
        prediction_context = self._voice_prediction_context(
            connected=current is not None,
            social_need=social_drive_level,
            social_fatigue=float(homeostasis["social_fatigue"]),
            habituation=float(homeostasis["habituation"]),
            exploration=float(homeostasis["exploration"]),
            human_count=(
                len(current_humans)
                if current is not None
                else available_humans
            ),
            alternatives=alternatives_count,
        )
        prediction_actions = (
            ("voice_move", "voice_leave", "stay")
            if current is not None
            else ("voice_join", "stay")
        )
        prediction_user_ids = (
            [int(m.id) for m in current_humans]
            if current is not None
            else sorted(available_human_ids)
        )
        prediction_scene_key = (
            self.voice_episodes.make_scene_key(
                current.id if current is not None else None,
                prediction_user_ids,
            )
        )
        prediction_expected = (
            self.voice_episodes.predictions(
                prediction_context,
                prediction_actions,
                scene_key=prediction_scene_key,
            )
            if self.cfg.voice.episodic_prediction_enabled
            else {}
        )
        debug["prediction_context"] = prediction_context
        debug["prediction_scene_key"] = prediction_scene_key
        debug["prediction_expected"] = dict(prediction_expected)
        last_prediction = self._voice_prediction_last.get(
            guild.id
        )
        if last_prediction is not None:
            debug["prediction_error"] = float(
                last_prediction.get("prediction_error", 0.0)
            )
        debug["episodic_memory_size"] = self.voice_episodes.size()
        debug["prediction_credit_queue_depth"] = len(
            self._voice_prediction_pending.get(guild.id, [])
        )
        debug["prediction_correction_queue_depth"] = len(
            self._voice_prediction_corrections.get(guild.id, [])
        )
        debug["memory_replay"] = dict(self._memory_replay_debug)
        disliked_strength = max(
            (
                min(1.0, max(0.0, -float(affinity)))
                for _, affinity in current_disliked
            ),
            default=0.0,
        )
        min_dwell = max(
            1.0,
            float(self.cfg.voice.minimum_dwell_seconds),
        )
        dwell_progress = (
            min(1.5, dwell_elapsed / min_dwell)
            if current is not None
            else 0.0
        )
        decision_trace = None
        brain_decision = None
        reward_opportunity_diag = None
        voice_context_diag = None

        async with self._brain_lock:
            corrections = self._voice_prediction_corrections.pop(
                guild.id,
                [],
            )
            if corrections:
                total_correction = 0.0
                for correction in corrections:
                    self.brain.reward(
                        float(correction["amount"]),
                        action=str(correction["action"]),
                        trace=correction["trace"],
                    )
                    total_correction += float(correction["amount"])
                self.brain.step(
                    max(1, min(3, len(corrections)))
                )
                debug["prediction_correction_applied"] = float(
                    total_correction
                )
                debug["prediction_corrections_applied"] = int(
                    len(corrections)
                )
                self._last_brain_event = (
                    "PREDICTION ERROR • temporal credit • "
                    f"{len(corrections)} traces • "
                    f"{total_correction:+.3f}"
                )

            for ch, humans in channels:
                sensory_scale = 1.0
                if current is not None and ch.id == current.id:
                    sensory_scale = max(
                        0.05,
                        1.0
                        - float(
                            homeostasis[
                                "habituation_suppression"
                            ]
                        ),
                    )
                self.brain.inject_voice_snapshot(
                    guild.id,
                    ch.id,
                    [m.id for m in humans],
                    sensory_scale=sensory_scale,
                )
            deadly_duration = max(
                1.0,
                float(self.cfg.voice.deadly_channel_seconds),
            )
            for row in debug["channels"]:
                remaining = float(row.get("deadly_remaining", 0.0))
                if remaining <= 0.0:
                    continue
                memory_strength = min(1.0, remaining / deadly_duration)
                self.brain.inject(
                    f"voice:deadly-channel:{guild.id}:{row['id']}",
                    float(self.cfg.voice.deadly_threat_magnitude)
                    * (0.5 + 0.5 * memory_strength),
                    128,
                )

            if (
                connectome_voice_control
                and reward_opportunity is not None
            ):
                reward_opportunity_diag = (
                    self.brain.inject_voice_reward_opportunity(
                        guild.id,
                        int(
                            reward_opportunity["channel_id"]
                        ),
                        human_count=int(
                            reward_opportunity["human_count"]
                        ),
                        strength=opportunity_effective_strength,
                    )
                )

            episodic_recall = {}
            if (
                connectome_voice_control
                and self.cfg.voice.episodic_prediction_enabled
            ):
                recall_scale = max(
                    0.0,
                    min(
                        4.0,
                        float(
                            self.cfg.voice.episodic_recall_magnitude
                        ),
                    ),
                )
                for action_name, expected_reward in (
                    prediction_expected.items()
                ):
                    magnitude = (
                        recall_scale
                        * max(0.0, float(expected_reward))
                    )
                    if magnitude <= 0.001:
                        continue
                    episodic_recall[action_name] = (
                        self.brain.inject_action_guided_sensory(
                            action_name,
                            (
                                "episodic-recall:"
                                f"{guild.id}:"
                                f"{prediction_scene_key}"
                            ),
                            magnitude,
                            width=176,
                            hops=3,
                        )
                    )
                if episodic_recall:
                    debug["episodic_recall"] = {
                        key: {
                            "mode": str(value.get("mode", "")),
                            "neurons": int(
                                value.get("neurons", 0)
                            ),
                            "reach_max": float(
                                value.get("reach_max", 0.0)
                            ),
                            "magnitude": float(
                                value.get("magnitude", 0.0)
                            ),
                        }
                        for key, value in episodic_recall.items()
                    }

            if connectome_voice_control:
                if chaser_active:
                    panic_scale = max(
                        0.0,
                        min(
                            1.0,
                            chaser_remaining
                            / max(
                                1.0,
                                float(
                                    self.cfg.voice.chaser_panic_seconds
                                ),
                            ),
                        ),
                    )
                    self.brain.inject_internal_state_cue(
                        "stress",
                        0.85 + 0.75 * panic_scale,
                        key=f"internal-state:stress:chaser:{guild.id}",
                    )
                    self.brain.inject_internal_state_cue(
                        "arousal",
                        0.45 + 0.45 * panic_scale,
                        key=f"internal-state:arousal:chaser:{guild.id}",
                    )
                voice_context_diag = (
                    self.brain.inject_voice_decision_context(
                    guild.id,
                    current.id if current is not None else None,
                    connected=bool(
                        vc is not None and vc.is_connected()
                    ),
                    dwell_progress=dwell_progress,
                    overstay_level=threat_level,
                    human_count=len(current_humans),
                    disliked_strength=disliked_strength,
                    alternatives=alternatives_count,
                    outside_seconds=outside_seconds,
                    available_humans=available_humans,
                    social_drive_level=social_drive_level,
                    social_drive_magnitude=float(
                        self.cfg.voice.social_drive_max_magnitude
                    ),
                    social_fatigue_level=float(
                        homeostasis["social_fatigue"]
                    ),
                    social_fatigue_magnitude=float(
                        self.cfg.voice.social_fatigue_max_magnitude
                    ),
                    habituation_level=float(
                        homeostasis["habituation"]
                    ),
                    habituation_change_magnitude=float(
                        self.cfg.voice.habituation_change_magnitude
                    ),
                    exploration_drive_level=float(
                        homeostasis["exploration"]
                    ),
                    exploration_drive_magnitude=float(
                        self.cfg.voice.exploration_drive_max_magnitude
                    ),
                    )
                )
                motivation_active = bool(
                    (
                        current is None
                        and (
                            reward_opportunity is not None
                            or social_drive_level > 0.0
                        )
                    )
                    or float(homeostasis["social_fatigue"]) > 0.0
                    or float(homeostasis["habituation"]) > 0.0
                    or float(homeostasis["exploration"]) > 0.0
                )
                propagation_steps = max(
                    2,
                    int(self.cfg.voice.threat_steps)
                    if threat_active
                    else 2,
                    int(
                        self.cfg.voice.motivation_propagation_steps
                    )
                    if motivation_active
                    else 2,
                )
                debug["motivation_propagation_steps"] = (
                    propagation_steps
                )
                self.brain.step(propagation_steps)
                brain_decision = self.brain.voice_action_decision(
                    connected=bool(
                        vc is not None and vc.is_connected()
                    ),
                    can_join=bool(channels),
                    can_move=bool(
                        dwell_remaining <= 0.0
                        and alternatives_count > 0
                    ),
                    can_leave=bool(
                        dwell_remaining <= 0.0
                    ),
                )
                scores = dict(brain_decision["scores"])
                debug["internal_states"] = (
                    self.brain.internal_state_diagnostics()
                )
                if voice_context_diag:
                    debug["internal_state_cues"] = dict(
                        voice_context_diag.get(
                            "internal_state_cues",
                            {},
                        )
                    )
                decision_trace = self.brain.capture_learning_trace()
                chosen_action = str(brain_decision["action"])
                predicted_reward = float(
                    prediction_expected.get(chosen_action, 0.0)
                )
                debug["predicted_reward"] = predicted_reward
                if self.cfg.voice.episodic_prediction_enabled:
                    prediction_members = (
                        current_humans
                        if current is not None
                        else [
                            member
                            for _, humans in channels
                            for member in humans
                        ]
                    )
                    seen_members = {}
                    for member in prediction_members:
                        seen_members[int(member.id)] = member
                    self._queue_voice_prediction(guild.id, {
                        "time": now,
                        "context": prediction_context,
                        "scene_key": prediction_scene_key,
                        "action": chosen_action,
                        "predicted_reward": predicted_reward,
                        "trace": decision_trace,
                        "channel_id": (
                            int(current.id)
                            if current is not None
                            else None
                        ),
                        "channel_name": (
                            str(current.name)
                            if current is not None
                            else ""
                        ),
                        "user_ids": sorted(
                            seen_members.keys()
                        ),
                        "user_names": [
                            seen_members[user_id].display_name
                            for user_id in sorted(seen_members)
                        ],
                    })
            else:
                self.brain.step(2)
                scores = self.brain.action_scores()

            affinities = {
                ch.id: self.brain.channel_affinity(guild.id, ch.id)
                for ch, _ in channels
            }

        if reward_opportunity_diag:
            debug["reward_opportunity_guided_mode"] = str(
                reward_opportunity_diag.get("mode", "")
            )
            debug["reward_opportunity_guided_neurons"] = int(
                reward_opportunity_diag.get("neurons", 0)
            )
            debug["reward_opportunity_guided_reach_mean"] = float(
                reward_opportunity_diag.get("reach_mean", 0.0)
            )
            debug["reward_opportunity_guided_reach_max"] = float(
                reward_opportunity_diag.get("reach_max", 0.0)
            )
        social_diag = (
            (voice_context_diag or {}).get("social_drive")
            if voice_context_diag
            else None
        )
        if social_diag:
            debug["social_drive_guided_mode"] = str(
                social_diag.get("mode", "")
            )
            debug["social_drive_guided_neurons"] = int(
                social_diag.get("neurons", 0)
            )
            debug["social_drive_guided_reach_mean"] = float(
                social_diag.get("reach_mean", 0.0)
            )
            debug["social_drive_guided_reach_max"] = float(
                social_diag.get("reach_max", 0.0)
            )

        if voice_context_diag:
            guided = {}
            for key in (
                "social_fatigue_move",
                "social_fatigue_leave",
                "habituation",
                "exploration_drive",
            ):
                row = voice_context_diag.get(key)
                if not row:
                    continue
                guided[key] = {
                    "mode": str(row.get("mode", "")),
                    "neurons": int(row.get("neurons", 0)),
                    "reach_mean": float(
                        row.get("reach_mean", 0.0)
                    ),
                    "reach_max": float(
                        row.get("reach_max", 0.0)
                    ),
                }
            debug["homeostasis_guided"] = guided

        if (
            connectome_voice_control
            and brain_decision is not None
            and current is None
            and social_drive_level > 0.0
            and brain_decision["action"] == "stay"
            and decision_trace is not None
        ):
            interval = max(
                5.0,
                float(
                    self.cfg.voice.social_drive_learning_interval_seconds
                ),
            )
            last_social_punish = (
                self._last_social_drive_punish.get(
                    guild.id,
                    0.0,
                )
            )
            if now - last_social_punish >= interval:
                punish_amount = max(
                    0.0,
                    min(
                        0.5,
                        float(
                            self.cfg.voice.social_drive_stay_punish
                        )
                        * social_drive_level,
                    ),
                )
                if punish_amount > 0.0:
                    async with self._brain_lock:
                        self.brain.reward(
                            -punish_amount,
                            action="stay",
                            trace=decision_trace,
                        )
                        self.brain.step(1)
                        brain_decision = (
                            self.brain.voice_action_decision(
                                connected=False,
                                can_join=bool(channels),
                            )
                        )
                        scores = dict(
                            brain_decision["scores"]
                        )
                        decision_trace = (
                            self.brain.capture_learning_trace()
                        )
                        affinities = {
                            ch.id: self.brain.channel_affinity(
                                guild.id,
                                ch.id,
                            )
                            for ch, _ in channels
                        }
                    self._last_social_drive_punish[
                        guild.id
                    ] = now
                    debug["social_drive_stay_punished"] = True
                    debug[
                        "social_drive_stay_punish_amount"
                    ] = -punish_amount
                    self._record_reward(
                        -punish_amount,
                        "stay",
                        "neural social drive outside voice",
                        guild,
                    )

        if (
            connectome_voice_control
            and brain_decision is not None
            and current is None
            and reward_opportunity is not None
            and brain_decision["action"] == "stay"
            and decision_trace is not None
            and not debug["social_drive_stay_punished"]
        ):
            interval = max(
                5.0,
                float(
                    self.cfg.voice.reward_opportunity_stay_punish_interval_seconds
                ),
            )
            last_missed = (
                self._last_reward_opportunity_stay_punish.get(
                    guild.id,
                    0.0,
                )
            )
            if now - last_missed >= interval:
                cue_scale = max(
                    0.25,
                    min(
                        1.5,
                        opportunity_effective_strength
                        / max(
                            0.25,
                            float(
                                self.cfg.voice.reward_opportunity_max_strength
                            ),
                        ),
                    ),
                )
                punish_amount = max(
                    0.0,
                    min(
                        0.5,
                        float(
                            self.cfg.voice.reward_opportunity_stay_punish
                        )
                        * cue_scale,
                    ),
                )
                if punish_amount > 0.0:
                    async with self._brain_lock:
                        self.brain.reward(
                            -punish_amount,
                            action="stay",
                            trace=decision_trace,
                        )
                        self.brain.step(1)
                        brain_decision = (
                            self.brain.voice_action_decision(
                                connected=False,
                                can_join=bool(channels),
                            )
                        )
                        scores = dict(
                            brain_decision["scores"]
                        )
                        decision_trace = (
                            self.brain.capture_learning_trace()
                        )
                        affinities = {
                            ch.id: self.brain.channel_affinity(
                                guild.id,
                                ch.id,
                            )
                            for ch, _ in channels
                        }
                    self._last_reward_opportunity_stay_punish[
                        guild.id
                    ] = now
                    debug[
                        "reward_opportunity_stay_punished"
                    ] = True
                    debug[
                        "reward_opportunity_stay_punish_amount"
                    ] = -punish_amount
                    self._record_reward(
                        -punish_amount,
                        "stay",
                        "ignored neural reward opportunity",
                        guild,
                    )

        if brain_decision is not None:
            debug["brain_decision"] = {
                "action": brain_decision["action"],
                "score": brain_decision["score"],
                "runner_up": brain_decision["runner_up"],
                "runner_up_score": brain_decision["runner_up_score"],
                "margin": brain_decision["margin"],
                "candidates": dict(
                    brain_decision["candidates"]
                ),
                "tie_band": float(
                    brain_decision.get("tie_band", 0.0)
                ),
                "tie_break": brain_decision.get(
                    "tie_break"
                ),
                "tie_evidence_margin": float(
                    brain_decision.get(
                        "tie_evidence_margin",
                        0.0,
                    )
                ),
                "raw_winner_margin": float(
                    brain_decision.get(
                        "raw_winner_margin",
                        brain_decision.get("margin", 0.0),
                    )
                ),
                "tie_evidence": dict(
                    brain_decision.get(
                        "tie_evidence",
                        {}
                    )
                ),
            }

        debug["scores"] = {
            "voice_join": scores["voice_join"],
            "voice_move": scores["voice_move"],
            "voice_leave": scores["voice_leave"],
            "stay": scores["stay"],
        }
        for row in debug["channels"]:
            if row["id"] in affinities:
                row["affinity"] = affinities[row["id"]]

        if chaser_active and (vc is None or not vc.is_connected()):
            debug["decision"] = "CHASE • UKRYWA SIĘ POZA VOICE"
            debug["reason"] = (
                f"panika jeszcze {chaser_remaining:.1f}s; "
                "Mucha nie wraca na voice podczas pościgu"
            )
            debug["current"] = None
            self._voice_debug[guild.id] = debug
            return

        if vc is None or not vc.is_connected():
            join = scores["voice_join"]
            if connectome_voice_control:
                if (
                    brain_decision is None
                    or brain_decision["action"] != "voice_join"
                ):
                    debug["decision"] = "ZOSTAJĘ POZA VOICE"
                    tie_note = ""
                    if brain_decision is not None and brain_decision.get(
                        "tie_break"
                    ):
                        tie_note = (
                            " • tie-break "
                            f"{brain_decision['tie_break']}"
                        )
                    debug["reason"] = (
                        "connectome winner "
                        f"{(brain_decision or {}).get('action', 'stay')} • "
                        f"join {join:.3f}"
                        f"{tie_note}"
                    )
                    self._voice_debug[guild.id] = debug
                    return
            elif join < self.cfg.voice.join_threshold:
                debug["decision"] = "NIE WCHODZĘ"
                debug["reason"] = (
                    f"voice_join {join:.3f} < próg "
                    f"{self.cfg.voice.join_threshold:.3f}"
                )
                self._voice_debug[guild.id] = debug
                return

            target, exploration = self._choose_voice_target(
                guild,
                channels,
                affinities,
                now,
                current_id=None,
                preferred_channel_id=(
                    int(
                        reward_opportunity["channel_id"]
                    )
                    if (
                        connectome_voice_control
                        and reward_opportunity is not None
                    )
                    else None
                ),
            )
            if target is None:
                debug["decision"] = "NIE WCHODZĘ"
                debug["reason"] = "brak celu po filtrach eksploracji"
                self._voice_debug[guild.id] = debug
                return
            for row in debug["channels"]:
                extra = exploration.get(row["id"])
                if extra:
                    row.update(extra)
            target_aff = affinities[target.id]
            debug["decision"] = f"JOIN → {target.name}"
            if connectome_voice_control and brain_decision is not None:
                tie_note = (
                    f"; tie-break {brain_decision['tie_break']}"
                    if brain_decision.get("tie_break")
                    else ""
                )
                debug["reason"] = (
                    f"CONNECTOME WINNER voice_join {join:.3f}; "
                    f"margin {brain_decision['margin']:.3f}"
                    f"{tie_note}; "
                    f"target neural affinity {target_aff:.3f}"
                )
            else:
                debug["reason"] = (
                    f"voice_join {join:.3f} ≥ "
                    f"{self.cfg.voice.join_threshold:.3f}; "
                    f"affinity {target_aff:.3f}; eksploracja "
                    f"{exploration.get(target.id, {}).get('exploration_score', target_aff):.3f}"
                )
            try:
                connect_kwargs = {
                    "self_deaf": not bool(self.cfg.voice.stt_enabled),
                }
                if (
                    self.cfg.voice.stt_enabled
                    and voice_recv is not None
                ):
                    connect_kwargs["cls"] = voice_recv.VoiceRecvClient
                new_vc = await target.connect(**connect_kwargs)
                self._ensure_voice_listener(new_vc)
                self._update_pending_voice_scene(
                    guild.id,
                    target,
                )
                self.voice_arrived[guild.id] = now
                self._mark_voice_visit(guild.id, target.id, now)
                await self._mark_social_voice_arrival(
                    guild,
                    target,
                    "voice_join",
                    now,
                )
                self._last_overstay_punish.pop(guild.id, None)
                self._last_brain_action = f"VOICE JOIN → {target.name}"
                if connectome_voice_control and decision_trace is not None:
                    learning_trace = decision_trace
                else:
                    async with self._brain_lock:
                        learning_trace = self.brain.capture_learning_trace()
                if (
                    connectome_voice_control
                    and reward_opportunity is not None
                    and int(target.id)
                    == int(
                        reward_opportunity["channel_id"]
                    )
                    and learning_trace is not None
                ):
                    opportunity_success = (
                        self.random.random()
                        < max(
                            0.0,
                            min(
                                1.0,
                                float(
                                    self.cfg.voice.reward_opportunity_success_chance
                                ),
                            ),
                        )
                    )
                    debug["reward_opportunity_found"] = bool(
                        opportunity_success
                    )
                    if opportunity_success:
                        opportunity_reward = max(
                            0.0,
                            min(
                                1.0,
                                float(
                                    self.cfg.voice.reward_opportunity_reward
                                ),
                            ),
                        )
                        if opportunity_reward > 0.0:
                            async with self._brain_lock:
                                self.brain.reward(
                                    opportunity_reward,
                                    action="voice_join",
                                    trace=learning_trace,
                                )
                                self.brain.step(1)
                            debug[
                                "reward_opportunity_reward"
                            ] = opportunity_reward
                            self._record_reward(
                                opportunity_reward,
                                "voice_join",
                                "random voice reward opportunity",
                                guild,
                            )
                    self._voice_reward_opportunity.pop(
                        guild.id,
                        None,
                    )

                if (
                    connectome_voice_control
                    and social_drive_level > 0.0
                    and learning_trace is not None
                ):
                    join_reward = max(
                        0.0,
                        min(
                            1.0,
                            float(
                                self.cfg.voice.social_join_reward
                            )
                            * (
                                0.50
                                + 0.50 * social_drive_level
                            ),
                        ),
                    )
                    if join_reward > 0.0:
                        async with self._brain_lock:
                            self.brain.reward(
                                join_reward,
                                action="voice_join",
                                trace=learning_trace,
                            )
                            self.brain.step(1)
                        debug["social_join_reward"] = (
                            join_reward
                        )
                        self._record_reward(
                            join_reward,
                            "voice_join",
                            "neural social drive join",
                            guild,
                        )
                self._set_reinforceable(
                    guild,
                    "voice_join",
                    learning_trace,
                    f"→ {target.name}",
                )
                self._record_action("voice_join", f"→ {target.name}", guild)
                debug["current"] = target.name
                debug["dwell_remaining"] = self.cfg.voice.minimum_dwell_seconds
                debug["decision"] = f"WESZŁA → {target.name}"
            except (discord.ClientException, discord.Forbidden, discord.HTTPException) as exc:
                debug["decision"] = "BŁĄD JOIN"
                debug["reason"] = f"{type(exc).__name__}: {exc}"
            self._voice_debug[guild.id] = debug
            return

        if current is None:
            debug["decision"] = "NIEZNANY STAN"
            debug["reason"] = "voice client jest połączony, ale kanał jest None"
            self._voice_debug[guild.id] = debug
            return

        if self._is_voice_channel_blocked(current):
            target, exploration = self._choose_voice_target(
                guild,
                channels,
                affinities,
                now,
                current_id=current.id,
            )
            if target is None:
                debug["decision"] = "BLOKADA • BRAK WYJŚCIA"
                debug["reason"] = (
                    f"kanał {current.id} jest zablokowany, "
                    "ale nie ma innego dostępnego kanału"
                )
                self._voice_debug[guild.id] = debug
                return
            try:
                await vc.move_to(target)
                self.voice_arrived[guild.id] = now
                self._mark_voice_visit(guild.id, target.id, now)
                await self._mark_social_voice_arrival(
                    guild,
                    target,
                    "voice_move",
                    now,
                )
                self._last_overstay_punish.pop(guild.id, None)
                self._last_brain_action = (
                    f"VOICE BLOCK ESCAPE → {target.name}"
                )
                self._record_action(
                    "blocked_voice_escape",
                    f"{current.name} → {target.name}",
                    guild,
                )
                debug["current"] = target.name
                debug["decision"] = f"BLOKADA • WYJŚCIE → {target.name}"
                debug["reason"] = (
                    f"kanał {current.id} jest na blocked_voice_channel_ids"
                )
            except (
                discord.Forbidden,
                discord.HTTPException,
                asyncio.TimeoutError,
            ) as exc:
                debug["decision"] = "BŁĄD WYJŚCIA Z BLOKADY"
                debug["reason"] = f"{type(exc).__name__}: {exc}"
            self._voice_debug[guild.id] = debug
            return

        if (
            current_disliked
            and not chaser_active
            and not connectome_voice_control
        ):
            target, exploration = self._choose_voice_target(
                guild,
                channels,
                affinities,
                now,
                current_id=current.id,
            )
            if target is not None:
                names = ", ".join(
                    member.display_name
                    for member, _ in current_disliked
                )
                try:
                    await vc.move_to(target)
                    self.voice_arrived[guild.id] = now
                    self._mark_voice_visit(guild.id, target.id, now)
                    await self._mark_social_voice_arrival(
                        guild,
                        target,
                        "voice_move",
                        now,
                    )
                    self._last_brain_action = (
                        f"SOCIAL AVOID → {target.name}"
                    )
                    self._last_brain_event = (
                        f"SOCIAL AVOID • omija {names}"
                    )
                    self._record_action(
                        "social_voice_avoid",
                        f"{current.name} → {target.name} • {names}",
                        guild,
                    )
                    debug["current"] = target.name
                    debug["decision"] = f"SOCIAL AVOID → {target.name}"
                    debug["reason"] = (
                        f"omija: {names}; affinity <= "
                        f"{self.cfg.behavior.user_avoid_threshold:+.2f}"
                    )
                    self._voice_debug[guild.id] = debug
                    return
                except (
                    discord.Forbidden,
                    discord.HTTPException,
                    asyncio.TimeoutError,
                ) as exc:
                    debug["decision"] = "SOCIAL AVOID • BŁĄD"
                    debug["reason"] = f"{type(exc).__name__}: {exc}"

        if dwell_remaining > 0:
            debug["decision"] = "MOTOR REFRACTORY • STAY"
            debug["reason"] = (
                f"move/leave fizycznie zablokowane jeszcze "
                f"{dwell_remaining:.1f}s; connectome dostał early-dwell"
            )
            self._voice_debug[guild.id] = debug
            return

        threat_trace = None
        if threat_active and not connectome_voice_control:
            threat_magnitude = float(self.cfg.voice.threat_magnitude) * threat_level
            async with self._brain_lock:
                self.brain.inject(
                    "internal:threat:voice-overstay",
                    threat_magnitude,
                    192,
                )
                self.brain.inject(
                    f"voice:threat:guild:{guild.id}",
                    0.75 * threat_magnitude,
                    128,
                )
                self.brain.inject(
                    f"voice:threat:channel:{guild.id}:{current.id}",
                    threat_magnitude,
                    128,
                )
                self.brain.step(max(1, int(self.cfg.voice.threat_steps)))
                scores = self.brain.action_scores()
                affinities = {
                    ch.id: self.brain.channel_affinity(guild.id, ch.id)
                    for ch, _ in channels
                }
                threat_trace = self.brain.capture_learning_trace()

            debug["threat_magnitude"] = threat_magnitude
            debug["scores"].update({
                "voice_join": scores["voice_join"],
                "voice_move": scores["voice_move"],
                "voice_leave": scores["voice_leave"],
                "stay": scores["stay"],
            })
            for row in debug["channels"]:
                if row["id"] in affinities:
                    row["affinity"] = affinities[row["id"]]
            self._last_brain_event = (
                f"VOICE THREAT • {current.name} • "
                f"{threat_level * 100:.0f}% zagrożenia"
            )

            last_punish = self._last_overstay_punish.get(guild.id, 0.0)
            punish_interval = max(
                float(self.cfg.voice.poll_seconds),
                float(self.cfg.voice.overstay_punish_interval_seconds),
            )
            if now - last_punish >= punish_interval:
                punish_amount = max(
                    0.0,
                    min(1.0, float(self.cfg.voice.overstay_punish_amount)),
                )
                async with self._brain_lock:
                    self.brain.reward(
                        -punish_amount,
                        action="stay",
                        trace=threat_trace,
                    )
                    self.brain.step(1)
                    scores = self.brain.action_scores()
                    affinities = {
                        ch.id: self.brain.channel_affinity(guild.id, ch.id)
                        for ch, _ in channels
                    }
                self._last_overstay_punish[guild.id] = now
                self._set_reinforceable(
                    guild,
                    "stay",
                    threat_trace,
                    f"threat overstay • {current.name} • {dwell_elapsed:.0f}s",
                )
                debug["overstay_punished"] = True
                debug["overstay_punish_amount"] = -punish_amount
                debug["scores"].update({
                    "voice_join": scores["voice_join"],
                    "voice_move": scores["voice_move"],
                    "voice_leave": scores["voice_leave"],
                    "stay": scores["stay"],
                })
                for row in debug["channels"]:
                    if row["id"] in affinities:
                        row["affinity"] = affinities[row["id"]]
                self._record_reward(
                    -punish_amount,
                    "stay",
                    "voice threat overstay",
                    guild,
                )
                self._record_action(
                    "threat",
                    f"-{punish_amount:.2f} stay • {current.name} • "
                    f"threat {threat_level * 100:.0f}%",
                    guild,
                )

        if threat_active and connectome_voice_control:
            threat_trace = decision_trace
            self._last_brain_event = (
                f"VOICE THREAT INPUT • {current.name} • "
                f"{threat_level * 100:.0f}%"
            )
            debug["threat_magnitude"] = (
                float(self.cfg.voice.threat_magnitude)
                * threat_level
            )
            last_punish = self._last_overstay_punish.get(
                guild.id,
                0.0,
            )
            punish_interval = max(
                float(self.cfg.voice.poll_seconds),
                float(
                    self.cfg.voice.overstay_punish_interval_seconds
                ),
            )
            if (
                now - last_punish >= punish_interval
                and brain_decision is not None
                and brain_decision["action"] == "stay"
                and threat_trace is not None
            ):
                punish_amount = max(
                    0.0,
                    min(
                        1.0,
                        float(
                            self.cfg.voice.overstay_punish_amount
                        ),
                    ),
                )
                async with self._brain_lock:
                    self.brain.reward(
                        -punish_amount,
                        action="stay",
                        trace=threat_trace,
                    )
                    self.brain.step(1)
                    brain_decision = (
                        self.brain.voice_action_decision(
                            connected=True,
                            can_move=bool(
                                alternatives_count > 0
                            ),
                            can_leave=True,
                        )
                    )
                    scores = dict(
                        brain_decision["scores"]
                    )
                    decision_trace = (
                        self.brain.capture_learning_trace()
                    )
                    affinities = {
                        ch.id: self.brain.channel_affinity(
                            guild.id,
                            ch.id,
                        )
                        for ch, _ in channels
                    }
                self._last_overstay_punish[guild.id] = now
                debug["overstay_punished"] = True
                debug["overstay_punish_amount"] = (
                    -punish_amount
                )
                debug["brain_decision"] = {
                    "action": brain_decision["action"],
                    "score": brain_decision["score"],
                    "runner_up": brain_decision["runner_up"],
                    "runner_up_score": brain_decision[
                        "runner_up_score"
                    ],
                    "margin": brain_decision["margin"],
                    "candidates": dict(
                        brain_decision["candidates"]
                    ),
                }
                debug["scores"].update({
                    "voice_join": scores["voice_join"],
                    "voice_move": scores["voice_move"],
                    "voice_leave": scores["voice_leave"],
                    "stay": scores["stay"],
                })
                self._record_reward(
                    -punish_amount,
                    "stay",
                    "connectome voice overstay",
                    guild,
                )

        current_aff = affinities.get(current.id, 0.5)

        # When the current channel becomes threatening, deliberately search for
        # the best *other* channel instead of allowing current affinity to win.
        alternatives = [
            item for item in channels
            if item[0].id != current.id
        ]
        if (
            threat_active
            and alternatives
            and not connectome_voice_control
        ):
            target, exploration = self._choose_voice_target(
                guild,
                alternatives,
                affinities,
                now,
                current_id=current.id,
            )
            if target is None:
                debug["decision"] = "ZAGROŻONA • BRAK CELU"
                debug["reason"] = "brak dostępnego innego kanału po filtrach"
                self._voice_debug[guild.id] = debug
                return
            for row in debug["channels"]:
                extra = exploration.get(row["id"])
                if extra:
                    row.update(extra)
            target_aff = affinities[target.id]
            target_exploration = float(
                exploration.get(target.id, {}).get(
                    "exploration_score",
                    target_aff,
                )
            )
            current_exploration, _, _, _ = self._voice_exploration_score(
                guild.id,
                current.id,
                current_aff,
                now,
            )
            effective_move_score = min(
                1.0,
                scores["voice_move"]
                + threat_level * float(self.cfg.voice.threat_move_boost),
            )
            effective_margin = (
                float(self.cfg.voice.move_margin)
                - threat_level * float(self.cfg.voice.threat_affinity_relaxation)
            )
            required_exploration = current_exploration + effective_margin

            debug["escape_target"] = target.name
            debug["effective_move_score"] = effective_move_score
            debug["effective_move_margin"] = effective_margin
            debug["target_exploration_score"] = target_exploration
            debug["current_exploration_score"] = current_exploration

            if (
                effective_move_score >= self.cfg.voice.move_threshold
                and target_exploration >= required_exploration
            ):
                debug["decision"] = f"UCIECZKA → {target.name}"
                debug["reason"] = (
                    f"threat {threat_level:.2f}; move "
                    f"{scores['voice_move']:.3f}+"
                    f"{threat_level * float(self.cfg.voice.threat_move_boost):.3f}"
                    f"={effective_move_score:.3f}; explore "
                    f"{target_exploration:.3f} ≥ "
                    f"{required_exploration:.3f}; affinity {target_aff:.3f}"
                )
                try:
                    await vc.move_to(target)
                    self._update_pending_voice_scene(
                        guild.id,
                        target,
                    )
                    self.voice_arrived[guild.id] = now
                    self._mark_voice_visit(guild.id, target.id, now)
                    await self._mark_social_voice_arrival(
                        guild,
                        target,
                        "voice_move",
                        now,
                    )
                    self._last_overstay_punish.pop(guild.id, None)
                    self._mark_deadly_voice_channel(
                        guild.id,
                        current.id,
                        now,
                    )
                    for row in debug["channels"]:
                        if row["id"] == current.id:
                            row["deadly"] = True
                            row["deadly_remaining"] = float(
                                self.cfg.voice.deadly_channel_seconds
                            )
                            row["eligible"] = False
                            row["status"] = (
                                f"☠ ŚMIERTELNE "
                                f"{self.cfg.voice.deadly_channel_seconds}s"
                            )

                    escape_reward = max(
                        0.0,
                        min(1.0, float(self.cfg.voice.threat_escape_reward)),
                    )
                    learning_trace = threat_trace
                    async with self._brain_lock:
                        if learning_trace is None:
                            learning_trace = self.brain.capture_learning_trace()
                        if escape_reward > 0:
                            self.brain.reward(
                                escape_reward,
                                action="voice_move",
                                trace=learning_trace,
                            )
                            self.brain.step(1)

                    self._last_brain_action = (
                        f"VOICE ESCAPE → {target.name} "
                        f"(threat {threat_level * 100:.0f}%)"
                    )
                    self._last_brain_event = (
                        f"VOICE SAFE • uciekła z {current.name} do {target.name}"
                    )
                    self._set_reinforceable(
                        guild,
                        "voice_move",
                        learning_trace,
                        f"escape {current.name} → {target.name}",
                    )
                    if escape_reward > 0:
                        self._record_reward(
                            escape_reward,
                            "voice_move",
                            "voice threat escape",
                            guild,
                        )
                    self._record_action(
                        "escape",
                        f"{current.name} → {target.name} • "
                        f"threat {threat_level * 100:.0f}% • "
                        f"☠ {self.cfg.voice.deadly_channel_seconds}s",
                        guild,
                    )
                    debug["current"] = target.name
                    debug["dwell_remaining"] = self.cfg.voice.minimum_dwell_seconds
                    debug["decision"] = f"UCIEKŁA → {target.name}"
                    debug["reason"] += (
                        f"; reward za ucieczkę +{escape_reward:.2f}"
                    )
                    self._voice_debug[guild.id] = debug
                    return
                except (
                    discord.Forbidden,
                    discord.HTTPException,
                    asyncio.TimeoutError,
                ) as exc:
                    debug["decision"] = "BŁĄD UCIECZKI"
                    debug["reason"] = f"{type(exc).__name__}: {exc}"
                    self._voice_debug[guild.id] = debug
                    return

        should_leave = (
            bool(
                connectome_voice_control
                and brain_decision is not None
                and brain_decision["action"] == "voice_leave"
            )
            or bool(
                not connectome_voice_control
                and scores["voice_leave"]
                >= self.cfg.voice.leave_threshold
            )
        )
        if should_leave:
            old_name = getattr(current, "name", "voice")
            debug["decision"] = f"LEAVE ← {old_name}"
            if connectome_voice_control and brain_decision is not None:
                debug["reason"] = (
                    "CONNECTOME WINNER voice_leave "
                    f"{scores['voice_leave']:.3f}; "
                    f"margin {brain_decision['margin']:.3f}"
                )
            else:
                debug["reason"] = (
                    f"voice_leave {scores['voice_leave']:.3f} ≥ "
                    f"{self.cfg.voice.leave_threshold:.3f}"
                )
            try:
                await vc.disconnect(force=False)
                self.voice_arrived[guild.id] = now
                self._voice_arrival_members.pop(guild.id, None)
                self._voice_arrival_channel.pop(guild.id, None)
                self._voice_arrival_learning.pop(guild.id, None)
                self._last_overstay_punish.pop(guild.id, None)
                self._last_brain_action = f"VOICE LEAVE ← {old_name}"
                if connectome_voice_control and decision_trace is not None:
                    learning_trace = decision_trace
                else:
                    async with self._brain_lock:
                        learning_trace = self.brain.capture_learning_trace()
                if (
                    connectome_voice_control
                    and threat_active
                    and learning_trace is not None
                ):
                    escape_reward = max(
                        0.0,
                        min(
                            1.0,
                            float(
                                self.cfg.voice.threat_escape_reward
                            ),
                        ),
                    )
                    if escape_reward > 0.0:
                        async with self._brain_lock:
                            self.brain.reward(
                                escape_reward,
                                action="voice_leave",
                                trace=learning_trace,
                            )
                            self.brain.step(1)
                        self._record_reward(
                            escape_reward,
                            "voice_leave",
                            "connectome threat leave",
                            guild,
                        )
                self._set_reinforceable(
                    guild,
                    "voice_leave",
                    learning_trace,
                    f"← {old_name}",
                )
                self._record_action("voice_leave", f"← {old_name}", guild)
                debug["current"] = None
                debug["decision"] = f"WYSZŁA ← {old_name}"
            except (discord.Forbidden, discord.HTTPException) as exc:
                debug["decision"] = "BŁĄD LEAVE"
                debug["reason"] = f"{type(exc).__name__}: {exc}"
            self._voice_debug[guild.id] = debug
            return

        # If threat is active but there is no successful escape yet, do not let
        # the current channel win simply because it has the highest affinity.
        if (
            threat_active
            and alternatives
            and not connectome_voice_control
        ):
            debug["decision"] = "ZAGROŻONA • SZUKA UCIECZKI"
            debug["reason"] = (
                f"threat {threat_level:.2f}; move "
                f"{debug['effective_move_score']:.3f} / "
                f"{self.cfg.voice.move_threshold:.3f}; "
                f"cel {debug['escape_target']}"
            )
            self._voice_debug[guild.id] = debug
            return

        if (
            connectome_voice_control
            and (
                brain_decision is None
                or brain_decision["action"] != "voice_move"
            )
        ):
            debug["decision"] = "CONNECTOME • STAY"
            debug["reason"] = (
                f"winner {(brain_decision or {}).get('action', 'stay')} • "
                f"stay {scores['stay']:.3f} / "
                f"move {scores['voice_move']:.3f} / "
                f"leave {scores['voice_leave']:.3f}"
            )
            self._voice_debug[guild.id] = debug
            return

        target, exploration = self._choose_voice_target(
            guild,
            channels,
            affinities,
            now,
            current_id=current.id,
        )
        if target is None:
            debug["decision"] = "ZOSTAJĘ"
            debug["reason"] = "brak innego dostępnego kanału po filtrach"
            self._voice_debug[guild.id] = debug
            return
        for row in debug["channels"]:
            extra = exploration.get(row["id"])
            if extra:
                row.update(extra)
        target_aff = affinities[target.id]

        if target.id == current.id:
            debug["decision"] = "ZOSTAJĘ"
            debug["reason"] = f"obecny kanał ma najwyższe affinity {current_aff:.3f}"
            self._voice_debug[guild.id] = debug
            return

        if (
            not connectome_voice_control
            and scores["voice_move"] < self.cfg.voice.move_threshold
        ):
            debug["decision"] = "ZOSTAJĘ"
            debug["reason"] = (
                f"voice_move {scores['voice_move']:.3f} < próg "
                f"{self.cfg.voice.move_threshold:.3f}"
            )
            self._voice_debug[guild.id] = debug
            return

        target_exploration = float(
            exploration.get(target.id, {}).get(
                "exploration_score",
                target_aff,
            )
        )
        current_exploration, _, _, _ = self._voice_exploration_score(
            guild.id,
            current.id,
            current_aff,
            now,
        )
        required_exploration = (
            current_exploration + float(self.cfg.voice.move_margin)
        )
        if (
            not connectome_voice_control
            and target_exploration < required_exploration
        ):
            debug["decision"] = "ZOSTAJĘ"
            debug["reason"] = (
                f"explore {target.name}={target_exploration:.3f} < wymagane "
                f"{required_exploration:.3f}; affinity "
                f"{target_aff:.3f}/{current_aff:.3f}"
            )
            self._voice_debug[guild.id] = debug
            return

        debug["decision"] = f"MOVE → {target.name}"
        if connectome_voice_control and brain_decision is not None:
            debug["reason"] = (
                f"CONNECTOME WINNER voice_move "
                f"{scores['voice_move']:.3f}; "
                f"margin {brain_decision['margin']:.3f}; "
                f"target neural affinity {target_aff:.3f}"
            )
        else:
            debug["reason"] = (
                f"voice_move {scores['voice_move']:.3f} ≥ "
                f"{self.cfg.voice.move_threshold:.3f}; "
                f"explore {target_exploration:.3f} > "
                f"{current_exploration:.3f}; "
                f"affinity {target_aff:.3f}"
            )
        try:
            await vc.move_to(target)
            self._update_pending_voice_scene(
                guild.id,
                target,
            )
            self.voice_arrived[guild.id] = now
            self._mark_voice_visit(guild.id, target.id, now)
            await self._mark_social_voice_arrival(
                guild,
                target,
                "voice_move",
                now,
            )
            self._last_overstay_punish.pop(guild.id, None)
            self._last_brain_action = f"VOICE MOVE → {target.name}"
            if connectome_voice_control and decision_trace is not None:
                learning_trace = decision_trace
            else:
                async with self._brain_lock:
                    learning_trace = self.brain.capture_learning_trace()
            if (
                connectome_voice_control
                and threat_active
                and learning_trace is not None
            ):
                self._mark_deadly_voice_channel(
                    guild.id,
                    current.id,
                    now,
                )
                escape_reward = max(
                    0.0,
                    min(
                        1.0,
                        float(self.cfg.voice.threat_escape_reward),
                    ),
                )
                if escape_reward > 0.0:
                    async with self._brain_lock:
                        self.brain.reward(
                            escape_reward,
                            action="voice_move",
                            trace=learning_trace,
                        )
                        self.brain.step(1)
                    self._record_reward(
                        escape_reward,
                        "voice_move",
                        "connectome threat escape",
                        guild,
                    )
            self._set_reinforceable(
                guild,
                "voice_move",
                learning_trace,
                f"→ {target.name}",
            )
            self._record_action("voice_move", f"→ {target.name}", guild)
            debug["current"] = target.name
            debug["dwell_remaining"] = self.cfg.voice.minimum_dwell_seconds
            debug["decision"] = f"PRZENIESIONA → {target.name}"
        except (discord.Forbidden, discord.HTTPException, asyncio.TimeoutError) as exc:
            debug["decision"] = "BŁĄD MOVE"
            debug["reason"] = f"{type(exc).__name__}: {exc}"
        self._voice_debug[guild.id] = debug

    async def _admin_command(self, message: discord.Message):
        if not isinstance(message.author, discord.Member) or not message.author.guild_permissions.administrator:
            return
        cmd = message.content[len(self.cfg.discord.command_prefix):].strip().lower()
        text_blocked = self._is_text_channel_blocked(message.channel)
        if cmd == "status":
            if text_blocked:
                await message.add_reaction("🚫")
                return
            d = self.brain.diagnostics()
            total, unique = self.language.stats()
            lang = self.language.diagnostics()
            scores = self.brain.action_scores()
            txt = (
                f"🪰 **Mucha v0.1**\n"
                f"neurony: `{d['neurons']:,}` | połączenia: `{d['connections']:,}`\n"
                f"aktywne >0.1: `{d['active_abs_gt_0_1']:,}` | mean |a|: `{d['mean_abs']:.4f}`\n"
                f"język: `{total:,}` znaków / `{unique:,}` unikalnych | "
                f"słownik: `{lang['word_vocab']:,}` słów | "
                f"przejścia: `{lang['transitions']:,}` | gotowa: `{self.language.ready()}`\n"
                f"connectom→słowa: "
                f"`{'AKTYWNY' if lang['connectome_word_control_ready'] else 'UCZY SŁOWNIK'}` "
                f"(`{lang['word_vocab']:,}/{lang['connectome_word_control_min_vocab']:,}`)\n"
                f"reward trace: `{d['reward_trace']:.3f}` | ticks: `{d['ticks']:,}`\n"
                f"speak `{scores['speak']:.2f}` move `{scores['voice_move']:.2f}` join `{scores['voice_join']:.2f}` leave `{scores['voice_leave']:.2f}`"
            )
            await message.channel.send(txt, allowed_mentions=discord.AllowedMentions.none())
        elif cmd == "save":
            async with self._brain_lock:
                self.brain.save()
            await message.add_reaction("💾")
        elif cmd == "pause":
            self.paused = True
            self._last_brain_action = "ADMIN → pause"
            await message.add_reaction("⏸️")
        elif cmd == "resume":
            self.paused = False
            self._last_brain_action = "ADMIN → resume"
            await message.add_reaction("▶️")
        elif cmd == "reward":
            context = self._last_reinforceable.get(message.guild.id)
            if context is None:
                self._record_action(
                    "reward",
                    "pominięto: brak akcji do nagrodzenia na tym serwerze",
                    message.guild,
                )
                await message.add_reaction("⚠️")
                return
            action, trace = context
            async with self._brain_lock:
                self.brain.reward(1.0, action=action, trace=trace)
                self.brain.step(1)
            self._record_reward(1.0, action, "admin", message.guild)
            self._record_action(
                "reward",
                f"+1 → {action}",
                message.guild,
            )
            await message.add_reaction("👍")
        elif cmd == "punish":
            context = self._last_reinforceable.get(message.guild.id)
            if context is None:
                self._record_action(
                    "reward",
                    "pominięto: brak akcji do ukarania na tym serwerze",
                    message.guild,
                )
                await message.add_reaction("⚠️")
                return
            action, trace = context
            async with self._brain_lock:
                self.brain.reward(-1.0, action=action, trace=trace)
                self.brain.step(1)
            self._record_reward(-1.0, action, "admin", message.guild)
            self._record_action(
                "reward",
                f"-1 → {action}",
                message.guild,
            )
            await message.add_reaction("👎")
        elif cmd == "audiotest":
            vc = message.guild.voice_client
            if (
                vc is None
                or not vc.is_connected()
                or vc.channel is None
            ):
                author_voice = getattr(message.author, "voice", None)
                target_channel = getattr(author_voice, "channel", None)
                if target_channel is None:
                    self._record_action(
                        "audio_test",
                        "pominięto: Mucha nie jest na voice i admin też nie",
                        message.guild,
                    )
                    await message.add_reaction("⚠️")
                    return

                try:
                    connect_kwargs = {
                        "self_deaf": not bool(self.cfg.voice.stt_enabled),
                    }
                    if (
                        self.cfg.voice.stt_enabled
                        and voice_recv is not None
                    ):
                        connect_kwargs["cls"] = voice_recv.VoiceRecvClient
                    vc = await target_channel.connect(**connect_kwargs)
                    self._ensure_voice_listener(vc)
                    self.voice_arrived[message.guild.id] = time.monotonic()
                    self._record_action(
                        "audio_test_join",
                        f"→ {target_channel.name}",
                        message.guild,
                    )
                except (
                    discord.ClientException,
                    discord.Forbidden,
                    discord.HTTPException,
                ) as exc:
                    self._audio_debug.update({
                        "status": "ERROR",
                        "stage": "audiotest_join",
                        "error": f"{type(exc).__name__}: {exc}",
                        "updated_at": time.time(),
                    })
                    log.exception(
                        "Audiotest nie mógł wejść na voice na serwerze %s",
                        message.guild.id,
                    )
                    await message.add_reaction("❌")
                    return

            # Audiotest is an explicit admin action: interrupt any current
            # playback instead of refusing with a warning while TTS/rare audio
            # happens to be active.
            if vc.is_playing():
                self._stop_voice_playback(vc)
                await asyncio.sleep(0.08)

            wav_path = Path("state") / "tts" / f"{message.guild.id}.wav"
            test_text = "Test głosu Muchy."
            voice_state = message.guild.me.voice if message.guild.me else None
            self._audio_debug.update({
                "status": "TEST",
                "stage": "synthesize",
                "server_muted": bool(getattr(voice_state, "mute", False)),
                "server_deafened": bool(getattr(voice_state, "deaf", False)),
                "suppressed": bool(getattr(voice_state, "suppress", False)),
                "error": "",
                "guild": message.guild.name,
                "channel": vc.channel.name,
                "file": str(wav_path),
                "file_size": 0,
                "text": test_text,
                "updated_at": time.time(),
            })
            try:
                ok = await asyncio.to_thread(
                    self._synthesize_tts_file,
                    test_text,
                    wav_path,
                )
                self._audio_debug["file_size"] = (
                    wav_path.stat().st_size if wav_path.is_file() else 0
                )
                if not ok:
                    self._audio_debug.update({
                        "status": "ERROR",
                        "stage": "synthesize",
                        "error": "TTS nie utworzył poprawnego WAV",
                        "updated_at": time.time(),
                    })
                    await message.add_reaction("❌")
                    return
                self._audio_debug.update({
                    "status": "TEST",
                    "stage": "ffmpeg_source",
                    "updated_at": time.time(),
                })
                source = self._make_voice_source(
                    wav_path,
                    self.cfg.voice.tts_volume,
                )
                playback_token = self._play_voice_source(
                    vc,
                    source,
                    "audiotest",
                )
                asyncio.create_task(
                    self._verify_voice_playback(
                        vc,
                        "audiotest",
                        playback_token,
                    )
                )
                self._audio_debug.update({
                    "status": "PLAYING",
                    "stage": "vc.play",
                    "error": "",
                    "updated_at": time.time(),
                })
                self._record_action(
                    "audio_test",
                    f"{vc.channel.name}: {wav_path}",
                    message.guild,
                )
                await message.add_reaction("🔊")
            except Exception as exc:
                self._audio_debug.update({
                    "status": "ERROR",
                    "stage": self._audio_debug.get("stage", "audiotest"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "updated_at": time.time(),
                })
                self._record_action(
                    "audio_error",
                    f"{type(exc).__name__}: {exc}",
                    message.guild,
                )
                log.exception(
                    "Audio test nie powiódł się na serwerze %s",
                    message.guild.id,
                )
                await message.add_reaction("❌")
        elif cmd == "help":
            if text_blocked:
                await message.add_reaction("🚫")
                return
            await message.channel.send("`!mucha status` `save` `pause` `resume` `reward` `punish` `audiotest`", allowed_mentions=discord.AllowedMentions.none())
