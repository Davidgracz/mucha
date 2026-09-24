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
import emoji as emoji_lib
import pyttsx3
import tomli_w
from discord.ext import tasks

try:
    from piper import PiperVoice, SynthesisConfig
except ImportError:
    PiperVoice = None
    SynthesisConfig = None

from .brain import FlyBrain
from .config import Config
from .connectome import Connectome
from .console_ui import ConsoleBrainUI
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
        )
        self.random = random.Random(cfg.brain.seed + 1)
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
        self._deadly_voice_until: dict[tuple[int, int], float] = {}
        self._voice_last_visit: dict[tuple[int, int], float] = {}
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
        behavior_fields = [
            "speak_threshold",
            "reaction_threshold",
            "reaction_cooldown_seconds",
            "social_learning_enabled",
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
            "positive_contact_cooldown_seconds",
            "familiar_affinity_threshold",
            "user_avoid_threshold",
            "ignore_disliked_users_text",
            "avoid_disliked_users_on_voice",
        ]
        voice_fields = [
            "poll_seconds",
            "minimum_dwell_seconds",
            "maximum_dwell_seconds",
            "move_threshold",
            "join_threshold",
            "leave_threshold",
            "include_empty_channels",
            "tts_enabled",
            "tts_interval_seconds",
            "tts_volume",
            "random_audio_enabled",
        ]
        data = {
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
            "channels": {
                "text": [],
                "voice": [],
            },
        }

        blocked_text = set(self.cfg.discord.blocked_text_channel_ids)
        blocked_voice = set(self.cfg.voice.blocked_voice_channel_ids)
        for guild in self.guilds:
            for channel in guild.text_channels:
                data["channels"]["text"].append({
                    "id": channel.id,
                    "name": channel.name,
                    "guild": guild.name,
                    "blocked": channel.id in blocked_text,
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
            ("behavior", "speak_threshold"): (float, 0.0, 1.0),
            ("behavior", "reaction_threshold"): (float, 0.0, 1.0),
            ("behavior", "reaction_cooldown_seconds"): (int, 0, 3600),
            ("behavior", "social_learning_enabled"): (bool, None, None),
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
            ("behavior", "positive_contact_cooldown_seconds"): (int, 1, 3600),
            ("behavior", "familiar_affinity_threshold"): (float, -1.0, 1.0),
            ("behavior", "user_avoid_threshold"): (float, -1.0, 1.0),
            ("behavior", "ignore_disliked_users_text"): (bool, None, None),
            ("behavior", "avoid_disliked_users_on_voice"): (bool, None, None),
            ("voice", "poll_seconds"): (int, 1, 3600),
            ("voice", "minimum_dwell_seconds"): (int, 0, 86400),
            ("voice", "maximum_dwell_seconds"): (int, 1, 86400),
            ("voice", "move_threshold"): (float, 0.0, 1.0),
            ("voice", "join_threshold"): (float, 0.0, 1.0),
            ("voice", "leave_threshold"): (float, 0.0, 1.0),
            ("voice", "include_empty_channels"): (bool, None, None),
            ("voice", "tts_enabled"): (bool, None, None),
            ("voice", "tts_interval_seconds"): (int, 1, 3600),
            ("voice", "tts_volume"): (float, 0.0, 2.0),
            ("voice", "random_audio_enabled"): (bool, None, None),
        }

        config_path = Path("config.toml")
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)

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
            else:
                value = float(value)
            if minimum is not None:
                value = max(minimum, value)
            if maximum is not None:
                value = min(maximum, value)

            raw.setdefault(section, {})[key] = value
            setattr(getattr(self.cfg, section), key, value)
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
            self.cfg.discord.blocked_text_channel_ids = ids
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
            self.cfg.voice.blocked_voice_channel_ids = ids
            changed.append("voice.blocked_voice_channel_ids")

        temp_path = config_path.with_suffix(".toml.tmp")
        temp_path.write_text(
            tomli_w.dumps(raw),
            encoding="utf-8",
        )
        temp_path.replace(config_path)

        self.voice_loop.change_interval(
            seconds=max(1, int(self.cfg.voice.poll_seconds))
        )
        self.tts_loop.change_interval(
            seconds=max(1, int(self.cfg.voice.tts_interval_seconds))
        )

        if self.is_ready():
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

        self._record_action(
            "config",
            ", ".join(changed) if changed else "brak zmian",
        )
        return {
            "ok": True,
            "changed": changed,
            "config": self._dashboard_config_snapshot(),
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

    def _user_affinity(self, user_id: int) -> float:
        return self.language.get_user_affinity(int(user_id))

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
        affinity = (
            self._user_affinity(member.id)
            if member is not None
            else 0.0
        )
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
            "updated_at": time.time(),
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
            affinity_delta = -min(
                0.30,
                max(
                    0.01,
                    float(self.cfg.behavior.user_affinity_negative_step)
                    * (0.45 + 1.05 * severity),
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

            brain_penalty = -min(0.35, 0.06 + 0.24 * severity)
            async with self._brain_lock:
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
                    f"severity {severity:.2f} • affinity {new_affinity:+.2f}"
                ),
                message.guild,
            )
            self._remember_social_event(
                "VERBAL_REJECTION",
                f"{label} • affinity {new_affinity:+.2f}",
                affinity_delta,
                message.author,
            )

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
        return int(channel_id) in self.cfg.discord.blocked_text_channel_ids

    def _is_voice_channel_blocked(self, channel: object) -> bool:
        channel_id = getattr(channel, "id", None)
        if channel_id is None:
            return False
        return int(channel_id) in self.cfg.voice.blocked_voice_channel_ids

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
        delay_min = max(
            0.0,
            float(self.cfg.voice.chaser_escape_delay_min_seconds),
        )
        delay_max = max(
            delay_min,
            float(self.cfg.voice.chaser_escape_delay_max_seconds),
        )
        await asyncio.sleep(self.random.uniform(delay_min, delay_max))

        guild = self.get_guild(guild_id)
        if guild is None:
            return
        vc = guild.voice_client
        me = guild.me
        if vc is None or not vc.is_connected() or vc.channel is None or me is None:
            return

        current = vc.channel
        now = time.monotonic()
        clean: list[tuple[discord.VoiceChannel, list[discord.Member]]] = []
        fallback: list[tuple[discord.VoiceChannel, list[discord.Member]]] = []

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
            if self._deadly_voice_remaining(guild.id, ch.id, now) <= 0.0:
                clean.append(item)

        candidates = clean or fallback
        if not candidates:
            self._record_action(
                "chaser_trapped",
                f"brak kanału ucieczki z {current.name}",
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
            return

        try:
            await vc.move_to(target)
            self.voice_arrived[guild.id] = now
            self._mark_voice_visit(guild.id, target.id, now)

            reward = max(
                0.0,
                min(1.0, float(self.cfg.voice.chaser_escape_reward)),
            )
            async with self._brain_lock:
                if reward > 0.0:
                    self.brain.reward(
                        reward,
                        action="voice_move",
                        trace=learning_trace,
                    )
                    self.brain.step(1)

            self._last_brain_event = (
                f"CHASER • ucieczka {current.name} → {target.name}"
            )
            self._last_brain_action = (
                f"PANIC ESCAPE → {target.name}"
            )
            self._set_reinforceable(
                guild,
                "voice_move",
                learning_trace,
                f"chaser escape {current.name} → {target.name}",
            )
            self._record_action(
                "chaser_escape",
                f"{current.name} → {target.name}",
                guild,
            )
            self._ensure_chaser_scream_loop(guild)
            if reward > 0.0:
                self._record_reward(
                    reward,
                    "voice_move",
                    "Mucha Chaser escape",
                    guild,
                )
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
                await asyncio.sleep(0.1)
                continue

            if not self.cfg.voice.chaser_scream_enabled:
                return

            if vc.is_playing():
                await asyncio.sleep(0.05)
                continue

            await self._play_chaser_scream(guild, vc)

            await asyncio.sleep(0.05)
            while (
                self._chaser_panic_remaining(guild_id) > 0.0
                and vc.is_connected()
                and vc.channel is not None
                and vc.is_playing()
            ):
                await asyncio.sleep(0.05)

    def _mark_voice_visit(
        self,
        guild_id: int,
        channel_id: int,
        now: float | None = None,
    ) -> None:
        now = time.monotonic() if now is None else now
        self._voice_last_visit[(int(guild_id), int(channel_id))] = now

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
        await self._update_presence()

    async def close(self) -> None:
        try:
            await self.web_ui.stop()
            self.console_ui.stop()
            self.brain.save()
            self.language.close()
        finally:
            await super().close()

    async def on_message(self, message: discord.Message):
        if message.guild is None or message.author.id == self.user.id:
            return
        if message.content.startswith(self.cfg.discord.command_prefix):
            await self._admin_command(message)
            return
        if self.paused:
            return
        if message.author.bot and not self.cfg.language.learn_from_bots:
            return

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
            await self._send_learned(message.channel, message.content, scores["explore"])
            self.last_reply[message.guild.id] = now

    async def _send_learned(self, channel: discord.abc.Messageable, context: str, arousal: float):
        if self._is_text_channel_blocked(channel):
            return
        text, trigrams = self.language.generate(context=context, arousal=arousal)
        if not text:
            return
        try:
            async with self._brain_lock:
                learning_trace = self.brain.capture_learning_trace()
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
                    self.language.reinforce(trigrams, -penalty)
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
            if amount > 0:
                affinity_delta = (
                    float(self.cfg.behavior.user_affinity_positive_step)
                    * POSITIVE_REACTION_WEIGHT.get(emoji, 0.6)
                )
                affinity_kind = "positive"
            else:
                affinity_delta = -(
                    float(self.cfg.behavior.user_affinity_negative_step)
                    * NEGATIVE_REACTION_WEIGHT.get(emoji, 0.6)
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
            self.language.reinforce(trace.trigrams, amount)
            async with self._brain_lock:
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
        if self.paused or (self.user and member.id == self.user.id):
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
        my_channel = (
            vc.channel
            if vc and vc.is_connected() and vc.channel is not None
            else None
        )
        changed_channel = getattr(before.channel, "id", None) != getattr(
            after.channel,
            "id",
            None,
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

    @tasks.loop(seconds=5)
    async def idle_loop(self):
        await self.wait_until_ready()
        if self.paused:
            return
        async with self._brain_lock:
            self.brain.inject("internal:time", 0.035, 32)
            self.brain.step(self.cfg.brain.idle_steps)
            scores = self.brain.action_scores()

        now = time.monotonic()
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

    async def _console_snapshot(self) -> dict:
        async with self._brain_lock:
            scores = self.brain.action_scores()
            diag = self.brain.diagnostics()
            top_neurons = self.brain.top_active_neurons(self.cfg.console_ui.top_neurons)

        language_total, language_unique = self.language.stats()
        language_diag = self.language.diagnostics()
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
            "audio_debug": dict(self._audio_debug),
            "reaction_debug": reaction_debug,
            "learning_debug": self.brain.learning_diagnostics(),
            "social_debug": dict(self._social_debug),
            "user_affinities": self.language.user_affinities(50),
            "word_feedback": self.language.top_word_feedback(30),
            "social_settings": {
                "user_avoid_threshold": self.cfg.behavior.user_avoid_threshold,
                "ignore_disliked_users_text": self.cfg.behavior.ignore_disliked_users_text,
                "avoid_disliked_users_on_voice": self.cfg.behavior.avoid_disliked_users_on_voice,
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
            options="-vn -ar 48000 -ac 2",
        )
        return discord.PCMVolumeTransformer(
            pcm,
            volume=volume,
        )

    async def _verify_voice_playback(
        self,
        vc: discord.VoiceClient,
        label: str,
    ) -> None:
        await asyncio.sleep(0.35)
        self._audio_debug.update({
            "playing": bool(vc.is_playing()),
            "connected": bool(vc.is_connected()),
            "stage": "playing_check",
            "status": "PLAYING" if vc.is_playing() else "STOPPED",
            "error": (
                ""
                if vc.is_playing()
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
                vc.stop()

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
            vc.play(source)
            asyncio.create_task(
                self._verify_voice_playback(vc, "chaser_scream")
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

        async with self._brain_lock:
            self.brain.inject(
                f"voice:tts-opportunity:guild:{guild.id}",
                0.18,
                64,
            )
            self.brain.step(1)
            scores = self.brain.action_scores()
            learning_trace = self.brain.capture_learning_trace()

        if scores["speak"] < self.cfg.behavior.speak_threshold:
            return

        context = self.last_text_context.get(guild.id, "")
        text_out, trigrams = self.language.generate(
            context=context,
            arousal=scores["explore"],
        )
        if not text_out:
            return

        text_out = text_out[: max(8, int(self.cfg.voice.tts_max_chars))].strip()
        if not text_out:
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
            vc.play(source)
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
                self._verify_voice_playback(vc, "tts")
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

            vc.play(source)
            asyncio.create_task(
                self._verify_voice_playback(vc, "rare_audio")
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
        current = vc.channel if vc and vc.is_connected() else None
        chaser_remaining = self._chaser_panic_remaining(guild.id, now)
        chaser_id = self._chaser_confirmed.get(guild.id)
        chaser_active = chaser_remaining > 0.0
        arrived = self.voice_arrived.get(guild.id, now)
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

        debug = {
            "guild": guild.name,
            "guild_id": guild.id,
            "enabled": self.cfg.voice.enabled,
            "current": current.name if current else None,
            "decision": "ANALIZA",
            "reason": "oczekiwanie na wynik",
            "join_threshold": self.cfg.voice.join_threshold,
            "move_threshold": self.cfg.voice.move_threshold,
            "leave_threshold": self.cfg.voice.leave_threshold,
            "move_margin": self.cfg.voice.move_margin,
            "minimum_dwell_seconds": self.cfg.voice.minimum_dwell_seconds,
            "maximum_dwell_seconds": self.cfg.voice.maximum_dwell_seconds,
            "dwell_elapsed": dwell_elapsed,
            "dwell_remaining": dwell_remaining,
            "overstay_seconds": overstay_seconds,
            "overstay_punished": False,
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
                self.cfg.behavior.avoid_disliked_users_on_voice
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
                current is not None
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

        async with self._brain_lock:
            for ch, humans in channels:
                self.brain.inject_voice_snapshot(guild.id, ch.id, [m.id for m in humans])
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
            self.brain.step(2)
            scores = self.brain.action_scores()
            affinities = {
                ch.id: self.brain.channel_affinity(guild.id, ch.id)
                for ch, _ in channels
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

        if vc is None or not vc.is_connected():
            join = scores["voice_join"]
            if join < self.cfg.voice.join_threshold:
                debug["decision"] = "NIE WCHODZĘ"
                debug["reason"] = (
                    f"voice_join {join:.3f} < próg {self.cfg.voice.join_threshold:.3f}"
                )
                self._voice_debug[guild.id] = debug
                return

            target, exploration = self._choose_voice_target(
                guild,
                channels,
                affinities,
                now,
                current_id=None,
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
            debug["reason"] = (
                f"voice_join {join:.3f} ≥ {self.cfg.voice.join_threshold:.3f}; "
                f"affinity {target_aff:.3f}; eksploracja "
                f"{exploration.get(target.id, {}).get('exploration_score', target_aff):.3f}"
            )
            try:
                await target.connect(self_deaf=True)
                self.voice_arrived[guild.id] = now
                self._mark_voice_visit(guild.id, target.id, now)
                self._last_overstay_punish.pop(guild.id, None)
                self._last_brain_action = f"VOICE JOIN → {target.name}"
                async with self._brain_lock:
                    learning_trace = self.brain.capture_learning_trace()
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

        if current_disliked and not chaser_active:
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
            debug["decision"] = "ZOSTAJĘ"
            debug["reason"] = f"minimum dwell: jeszcze {dwell_remaining:.1f} s"
            self._voice_debug[guild.id] = debug
            return

        threat_trace = None
        if threat_active:
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

        current_aff = affinities.get(current.id, 0.5)

        # When the current channel becomes threatening, deliberately search for
        # the best *other* channel instead of allowing current affinity to win.
        alternatives = [
            item for item in channels
            if item[0].id != current.id
        ]
        if threat_active and alternatives:
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
                    self.voice_arrived[guild.id] = now
                    self._mark_voice_visit(guild.id, target.id, now)
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

        if scores["voice_leave"] >= self.cfg.voice.leave_threshold:
            old_name = getattr(current, "name", "voice")
            debug["decision"] = f"LEAVE ← {old_name}"
            debug["reason"] = (
                f"voice_leave {scores['voice_leave']:.3f} ≥ "
                f"{self.cfg.voice.leave_threshold:.3f}"
            )
            try:
                await vc.disconnect(force=False)
                self.voice_arrived[guild.id] = now
                self._last_overstay_punish.pop(guild.id, None)
                self._last_brain_action = f"VOICE LEAVE ← {old_name}"
                async with self._brain_lock:
                    learning_trace = self.brain.capture_learning_trace()
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
        if threat_active and alternatives:
            debug["decision"] = "ZAGROŻONA • SZUKA UCIECZKI"
            debug["reason"] = (
                f"threat {threat_level:.2f}; move "
                f"{debug['effective_move_score']:.3f} / "
                f"{self.cfg.voice.move_threshold:.3f}; "
                f"cel {debug['escape_target']}"
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

        if scores["voice_move"] < self.cfg.voice.move_threshold:
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
        if target_exploration < required_exploration:
            debug["decision"] = "ZOSTAJĘ"
            debug["reason"] = (
                f"explore {target.name}={target_exploration:.3f} < wymagane "
                f"{required_exploration:.3f}; affinity "
                f"{target_aff:.3f}/{current_aff:.3f}"
            )
            self._voice_debug[guild.id] = debug
            return

        debug["decision"] = f"MOVE → {target.name}"
        debug["reason"] = (
            f"voice_move {scores['voice_move']:.3f} ≥ {self.cfg.voice.move_threshold:.3f}; "
            f"explore {target_exploration:.3f} > {current_exploration:.3f}; "
            f"affinity {target_aff:.3f}"
        )
        try:
            await vc.move_to(target)
            self.voice_arrived[guild.id] = now
            self._mark_voice_visit(guild.id, target.id, now)
            self._last_overstay_punish.pop(guild.id, None)
            self._last_brain_action = f"VOICE MOVE → {target.name}"
            async with self._brain_lock:
                learning_trace = self.brain.capture_learning_trace()
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
                f"przejścia: `{lang['transitions']:,}` | gotowa: `{self.language.ready()}`\n"
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
                or vc.is_playing()
            ):
                await message.add_reaction("⚠️")
                return

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
                vc.play(source)
                asyncio.create_task(
                    self._verify_voice_playback(vc, "audiotest")
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
