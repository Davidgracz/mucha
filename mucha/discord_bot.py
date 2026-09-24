from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass

import discord
from discord.ext import tasks

from .brain import FlyBrain
from .config import Config
from .connectome import Connectome
from .language import OnlineLanguage

log = logging.getLogger("mucha")

POSITIVE = {"👍", "❤️", "❤", "😂", "🤣", "🔥", "🪰", "💚", "👏"}
NEGATIVE = {"👎", "😡", "🤮", "💩", "😒"}


@dataclass
class SentTrace:
    trigrams: list[tuple[str,str,str]]
    created: float


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
            cfg.language.min_tokens_before_speaking,
            cfg.language.min_unique_tokens_before_speaking,
            cfg.language.max_generated_tokens,
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

    async def setup_hook(self) -> None:
        self.idle_loop.change_interval(seconds=self.cfg.behavior.idle_tick_seconds)
        self.voice_loop.change_interval(seconds=self.cfg.voice.poll_seconds)
        self.idle_loop.start()
        self.presence_loop.start()
        if self.cfg.voice.enabled:
            self.voice_loop.start()

    async def on_ready(self):
        m = self.connectome.metadata
        log.info("Zalogowano jako %s", self.user)
        log.info("Connectome: %s neuronów, %s połączeń", self.connectome.n_neurons, self.connectome.matrix.nnz)
        log.info("Źródło: %s", m.get("source", "unknown"))
        await self._update_presence()

    async def close(self) -> None:
        try:
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

        async with self._brain_lock:
            self.brain.inject_text(message.content, message.author.id, mentioned)
            self.brain.step(self.cfg.brain.steps_per_event)
            scores = self.brain.action_scores()

        now = time.monotonic()
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
            sent = await channel.send(text, allowed_mentions=discord.AllowedMentions.none())
            self.sent[sent.id] = SentTrace(trigrams=trigrams, created=time.monotonic())
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
            self.language.reinforce(trace.trigrams, amount)
            async with self._brain_lock:
                self.brain.reward(amount)
                self.brain.step(1)

    async def on_voice_state_update(self, member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
        if self.paused or (self.user and member.id == self.user.id):
            return
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
                afk=False,
            )
            self._last_presence_text = text
        except discord.HTTPException:
            log.exception("Nie udało się zaktualizować statusu Discord")

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
            return
        channels = []
        for ch in guild.voice_channels:
            if self.cfg.voice.exclude_afk_channel and guild.afk_channel and ch.id == guild.afk_channel.id:
                continue
            perms = ch.permissions_for(me)
            if not (perms.view_channel and perms.connect):
                continue
            humans = [m for m in ch.members if not m.bot]
            if humans or self.cfg.voice.include_empty_channels:
                channels.append((ch, humans))
        if not channels:
            return

        async with self._brain_lock:
            for ch, humans in channels:
                self.brain.inject_voice_snapshot(guild.id, ch.id, [m.id for m in humans])
            self.brain.step(2)
            scores = self.brain.action_scores()
            affinities = {ch.id: self.brain.channel_affinity(guild.id, ch.id) for ch, _ in channels}

        vc = guild.voice_client
        now = time.monotonic()

        if vc is None or not vc.is_connected():
            if scores["voice_join"] < self.cfg.voice.join_threshold:
                return
            target = max(channels, key=lambda x: affinities[x[0].id])[0]
            try:
                await target.connect(self_deaf=True)
                self.voice_arrived[guild.id] = now
            except (discord.ClientException, discord.Forbidden, discord.HTTPException):
                return
            return

        current = vc.channel
        if current is None:
            return
        arrived = self.voice_arrived.get(guild.id, now)
        if now - arrived < self.cfg.voice.minimum_dwell_seconds:
            return

        if scores["voice_leave"] >= self.cfg.voice.leave_threshold:
            await vc.disconnect(force=False)
            self.voice_arrived[guild.id] = now
            return

        current_aff = affinities.get(current.id, 0.5)
        target, _ = max(channels, key=lambda x: affinities[x[0].id])
        target_aff = affinities[target.id]
        if (
            target.id != current.id
            and scores["voice_move"] >= self.cfg.voice.move_threshold
            and target_aff >= current_aff + self.cfg.voice.move_margin
        ):
            try:
                await vc.move_to(target)
                self.voice_arrived[guild.id] = now
            except (discord.Forbidden, discord.HTTPException, asyncio.TimeoutError):
                pass

    async def _admin_command(self, message: discord.Message):
        if not isinstance(message.author, discord.Member) or not message.author.guild_permissions.administrator:
            return
        cmd = message.content[len(self.cfg.discord.command_prefix):].strip().lower()
        if cmd == "status":
            d = self.brain.diagnostics()
            total, unique = self.language.stats()
            scores = self.brain.action_scores()
            txt = (
                f"🪰 **Mucha v0.1**\n"
                f"neurony: `{d['neurons']:,}` | połączenia: `{d['connections']:,}`\n"
                f"aktywne >0.1: `{d['active_abs_gt_0_1']:,}` | mean |a|: `{d['mean_abs']:.4f}`\n"
                f"język: `{total:,}` tokenów / `{unique:,}` unikalnych | gotowa: `{self.language.ready()}`\n"
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
            await message.add_reaction("⏸️")
        elif cmd == "resume":
            self.paused = False
            await message.add_reaction("▶️")
        elif cmd == "reward":
            async with self._brain_lock:
                self.brain.reward(1.0)
            await message.add_reaction("👍")
        elif cmd == "punish":
            async with self._brain_lock:
                self.brain.reward(-1.0)
            await message.add_reaction("👎")
        elif cmd == "help":
            await message.channel.send("`!mucha status` `save` `pause` `resume` `reward` `punish`", allowed_mentions=discord.AllowedMentions.none())
