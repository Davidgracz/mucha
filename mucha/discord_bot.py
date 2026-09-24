from __future__ import annotations

import asyncio
import logging
import math
import random
import shutil
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import discord
import emoji as emoji_lib
import pyttsx3
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


@dataclass
class SentTrace:
    trigrams: list[tuple[str,str,str]]
    created: float
    action: str
    learning_trace: tuple


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
        )

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
        self.language.learn(message.content)
        mentioned = self.user in message.mentions if self.user else False
        channel_name = getattr(message.channel, "name", str(message.channel.id))
        self._last_brain_event = f"TEXT • {message.author.display_name} • #{channel_name}" + (" • mention" if mentioned else "")

        async with self._brain_lock:
            self.brain.inject_text(message.content, message.author.id, mentioned)
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
        if scores["react"] >= self.cfg.behavior.reaction_threshold and react_cooldown <= 0.0:
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
            self.sent[sent.id] = SentTrace(
                trigrams=trigrams,
                created=time.monotonic(),
                action="speak",
                learning_trace=learning_trace,
            )
            channel_name = getattr(channel, "name", "kanał")
            self._last_brain_action = f"TEXT → #{channel_name}: {text[:80]}"
            guild = getattr(channel, "guild", None)
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
            self._last_brain_event = f"REACTION • {emoji} • reward {amount:+.0f}"
            self.language.reinforce(trace.trigrams, amount)
            async with self._brain_lock:
                self.brain.reward(
                    amount,
                    action=trace.action,
                    trace=trace.learning_trace,
                )
                self.brain.step(1)
            guild = self.get_guild(payload.guild_id) if payload.guild_id else None
            self._record_reward(amount, trace.action, f"Discord {emoji}", guild)
            self._record_action(
                "reward",
                f"{amount:+.0f} → {trace.action} ({emoji})",
                guild,
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
        ]
        if not candidates:
            return

        vc = self.random.choice(candidates)
        guild = vc.guild
        channel_name = getattr(vc.channel, "name", "voice")

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
                "affinity": None,
                "exploration_score": None,
                "visit_age": None,
                "novelty": None,
                "current": bool(current and current.id == ch.id),
                "status": "OK" if eligible else (
                    "⛔ BLOKADA" if blocked_voice else
                    "🕷 CHASER" if chaser_here else
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

        if not channels:
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
