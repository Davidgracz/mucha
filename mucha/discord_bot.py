from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass

import discord
import emoji as emoji_lib
from discord.ext import tasks

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
        self.console_loop.change_interval(seconds=max(0.25, self.cfg.console_ui.refresh_seconds))
        self.idle_loop.start()
        self.presence_loop.start()
        if self.cfg.voice.enabled:
            self.voice_loop.start()

    async def on_ready(self):
        m = self.connectome.metadata
        log.info("Zalogowano jako %s", self.user)
        log.info("Connectome: %s neuronów, %s połączeń", self.connectome.n_neurons, self.connectome.matrix.nnz)
        log.info("Źródło: %s", m.get("source", "unknown"))
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

        self.last_text_channel[message.guild.id] = message.channel.id
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
            self.language.ready()
            and urge >= self.cfg.behavior.speak_threshold
            and now - last >= self.cfg.language.reply_cooldown_seconds
        ):
            await self._send_learned(message.channel, message.content, scores["explore"])
            self.last_reply[message.guild.id] = now

    async def _send_learned(self, channel: discord.abc.Messageable, context: str, arousal: float):
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
        async with self._brain_lock:
            self.brain.inject(key, 0.65, 64)
            self.brain.inject(f"voice-user:{member.id}", 0.35, 48)
            self.brain.step(1)

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
            if isinstance(channel, discord.TextChannel):
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
        arrived = self.voice_arrived.get(guild.id, now)
        dwell_elapsed = max(0.0, now - arrived)
        dwell_remaining = max(0.0, self.cfg.voice.minimum_dwell_seconds - dwell_elapsed)

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
            "overstay_seconds": max(
                0.0,
                dwell_elapsed - self.cfg.voice.maximum_dwell_seconds,
            ),
            "overstay_punished": False,
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
            eligible = bool(not is_afk and view_ok and connect_ok and include_ok)

            row = {
                "id": ch.id,
                "name": ch.name,
                "humans": len(humans),
                "view": view_ok,
                "connect": connect_ok,
                "afk": is_afk,
                "eligible": eligible,
                "affinity": None,
                "current": bool(current and current.id == ch.id),
                "status": "OK" if eligible else (
                    "AFK" if is_afk else
                    "BRAK VIEW" if not view_ok else
                    "BRAK CONNECT" if not connect_ok else
                    "PUSTY WYŁĄCZONY"
                ),
            }
            debug["channels"].append(row)
            if eligible:
                channels.append((ch, humans))

        if not channels:
            debug["decision"] = "NIE WCHODZĘ"
            debug["reason"] = "brak dostępnych kanałów voice"
            self._voice_debug[guild.id] = debug
            return

        async with self._brain_lock:
            for ch, humans in channels:
                self.brain.inject_voice_snapshot(guild.id, ch.id, [m.id for m in humans])
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

            target = max(channels, key=lambda x: affinities[x[0].id])[0]
            target_aff = affinities[target.id]
            debug["decision"] = f"JOIN → {target.name}"
            debug["reason"] = (
                f"voice_join {join:.3f} ≥ {self.cfg.voice.join_threshold:.3f}; "
                f"najwyższe affinity {target_aff:.3f}"
            )
            try:
                await target.connect(self_deaf=True)
                self.voice_arrived[guild.id] = now
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

        if dwell_remaining > 0:
            debug["decision"] = "ZOSTAJĘ"
            debug["reason"] = f"minimum dwell: jeszcze {dwell_remaining:.1f} s"
            self._voice_debug[guild.id] = debug
            return

        max_dwell = max(
            float(self.cfg.voice.minimum_dwell_seconds),
            float(self.cfg.voice.maximum_dwell_seconds),
        )
        if dwell_elapsed >= max_dwell:
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
                    learning_trace = self.brain.capture_learning_trace()
                    self.brain.reward(
                        -punish_amount,
                        action="stay",
                        trace=learning_trace,
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
                    learning_trace,
                    f"overstay • {current.name} • {dwell_elapsed:.0f}s",
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
                    "voice overstay",
                    guild,
                )
                self._record_action(
                    "punish",
                    f"-{punish_amount:.2f} stay • {current.name} • {dwell_elapsed:.0f}s",
                    guild,
                )
                self._last_brain_event = (
                    f"VOICE OVERSTAY • {current.name} • punish -{punish_amount:.2f}"
                )

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

        current_aff = affinities.get(current.id, 0.5)
        target, _ = max(channels, key=lambda x: affinities[x[0].id])
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

        required_aff = current_aff + self.cfg.voice.move_margin
        if target_aff < required_aff:
            debug["decision"] = "ZOSTAJĘ"
            debug["reason"] = (
                f"affinity {target.name}={target_aff:.3f} < wymagane "
                f"{required_aff:.3f} (current {current_aff:.3f} + margin "
                f"{self.cfg.voice.move_margin:.3f})"
            )
            self._voice_debug[guild.id] = debug
            return

        debug["decision"] = f"MOVE → {target.name}"
        debug["reason"] = (
            f"voice_move {scores['voice_move']:.3f} ≥ {self.cfg.voice.move_threshold:.3f}; "
            f"affinity {target_aff:.3f} > {current_aff:.3f}"
        )
        try:
            await vc.move_to(target)
            self.voice_arrived[guild.id] = now
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
        if cmd == "status":
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
        elif cmd == "help":
            await message.channel.send("`!mucha status` `save` `pause` `resume` `reward` `punish`", allowed_mentions=discord.AllowedMentions.none())
