from __future__ import annotations

import threading
import time
from collections import deque
from typing import Iterable


class VoiceSensoryBus:
    """Thread-safe live voice-scene telemetry for the connectome.

    Discord voice PCM arrives from a receiver callback, while snapshots are
    consumed from the asyncio event loop. This class only measures the scene:
    it does not choose or score actions.
    """

    def __init__(
        self,
        *,
        speaker_timeout_seconds: float = 0.55,
        recent_window_seconds: float = 60.0,
        reply_window_seconds: float = 15.0,
    ) -> None:
        self.speaker_timeout_seconds = max(
            0.15,
            float(speaker_timeout_seconds),
        )
        self.recent_window_seconds = max(
            5.0,
            float(recent_window_seconds),
        )
        self.reply_window_seconds = max(
            1.0,
            float(reply_window_seconds),
        )
        self._lock = threading.Lock()
        self._guilds: dict[int, dict] = {}

    def _guild(self, guild_id: int, now: float) -> dict:
        guild_id = int(guild_id)
        state = self._guilds.get(guild_id)
        if state is None:
            state = {
                "created": now,
                "scene_channel_id": None,
                "scene_started": now,
                "connected": False,
                "connection_changed": now,
                "last_any_speech": None,
                "users": {},
                "events": deque(maxlen=48),
                "last_reply": None,
                "last_transcript": None,
            }
            self._guilds[guild_id] = state
        return state

    @staticmethod
    def _append_event(
        state: dict,
        kind: str,
        now: float,
        *,
        user_id: int | None = None,
        user_name: str | None = None,
        channel_id: int | None = None,
        duration: float | None = None,
    ) -> None:
        state["events"].append({
            "kind": str(kind),
            "time": float(now),
            "user_id": int(user_id) if user_id is not None else None,
            "user_name": str(user_name or ""),
            "channel_id": int(channel_id) if channel_id is not None else None,
            "duration": (
                max(0.0, float(duration))
                if duration is not None
                else None
            ),
        })

    def _finish_turn_locked(
        self,
        state: dict,
        user: dict,
        now: float,
    ) -> None:
        if not user.get("speaking"):
            return
        started = float(user.get("speaking_since", now))
        last_packet = float(user.get("last_packet", now))
        duration = max(0.0, last_packet - started)
        user["speaking"] = False
        user["last_turn_seconds"] = duration
        user["last_stop"] = now
        self._append_event(
            state,
            "speech_stop",
            now,
            user_id=int(user["user_id"]),
            user_name=str(user.get("name", "")),
            channel_id=int(user.get("channel_id", 0)) or None,
            duration=duration,
        )

    def _close_stale_locked(self, state: dict, now: float) -> None:
        for user in state["users"].values():
            if (
                user.get("speaking")
                and now - float(user.get("last_packet", now))
                > self.speaker_timeout_seconds
            ):
                self._finish_turn_locked(state, user, now)

    def note_pcm(
        self,
        guild_id: int,
        channel_id: int,
        user_id: int,
        user_name: str,
        *,
        now: float | None = None,
    ) -> bool:
        now = time.monotonic() if now is None else float(now)
        guild_id = int(guild_id)
        channel_id = int(channel_id)
        user_id = int(user_id)

        with self._lock:
            state = self._guild(guild_id, now)
            if state.get("scene_channel_id") != channel_id:
                for existing in state["users"].values():
                    if (
                        existing.get("speaking")
                        and int(existing.get("channel_id", 0)) != channel_id
                    ):
                        self._finish_turn_locked(state, existing, now)
                state["scene_channel_id"] = channel_id
                state["scene_started"] = now
                state["last_any_speech"] = None

            user = state["users"].get(user_id)
            if user is None:
                user = {
                    "user_id": user_id,
                    "name": str(user_name or user_id),
                    "channel_id": channel_id,
                    "speaking": False,
                    "speaking_since": now,
                    "last_packet": now,
                    "last_stop": None,
                    "last_turn_seconds": 0.0,
                    "packets": 0,
                    "turn_starts": deque(maxlen=96),
                    "last_transcript": "",
                    "last_transcript_duration": 0.0,
                    "last_transcript_at": None,
                }
                state["users"][user_id] = user

            gap = now - float(user.get("last_packet", now))
            if (
                user.get("speaking")
                and (
                    int(user.get("channel_id", channel_id)) != channel_id
                    or gap > self.speaker_timeout_seconds
                )
            ):
                self._finish_turn_locked(state, user, now)

            user["name"] = str(user_name or user_id)
            user["channel_id"] = channel_id
            started_now = not bool(user.get("speaking"))
            if started_now:
                user["speaking"] = True
                user["speaking_since"] = now
                user["turn_starts"].append(now)
                self._append_event(
                    state,
                    "speech_start",
                    now,
                    user_id=user_id,
                    user_name=user["name"],
                    channel_id=channel_id,
                )

            user["last_packet"] = now
            user["packets"] = int(user.get("packets", 0)) + 1
            state["last_any_speech"] = now
            return started_now

    def note_transcript(
        self,
        guild_id: int,
        channel_id: int,
        user_id: int,
        user_name: str,
        text: str,
        duration: float,
        *,
        now: float | None = None,
    ) -> None:
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            state = self._guild(int(guild_id), now)
            user = state["users"].setdefault(
                int(user_id),
                {
                    "user_id": int(user_id),
                    "name": str(user_name or user_id),
                    "channel_id": int(channel_id),
                    "speaking": False,
                    "speaking_since": now,
                    "last_packet": now,
                    "last_stop": None,
                    "last_turn_seconds": 0.0,
                    "packets": 0,
                    "turn_starts": deque(maxlen=96),
                    "last_transcript": "",
                    "last_transcript_duration": 0.0,
                    "last_transcript_at": None,
                },
            )
            user["name"] = str(user_name or user_id)
            user["channel_id"] = int(channel_id)
            user["last_transcript"] = str(text or "")[:180]
            user["last_transcript_duration"] = max(0.0, float(duration))
            user["last_transcript_at"] = now
            state["last_transcript"] = {
                "user_id": int(user_id),
                "user_name": user["name"],
                "channel_id": int(channel_id),
                "text": str(text or "")[:180],
                "duration": max(0.0, float(duration)),
                "time": now,
            }

    def note_reply_after_tts(
        self,
        guild_id: int,
        channel_id: int,
        user_id: int,
        user_name: str,
        *,
        tts_age_seconds: float,
        now: float | None = None,
    ) -> None:
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            state = self._guild(int(guild_id), now)
            state["last_reply"] = {
                "user_id": int(user_id),
                "user_name": str(user_name or user_id),
                "channel_id": int(channel_id),
                "tts_age_seconds": max(0.0, float(tts_age_seconds)),
                "time": now,
            }
            self._append_event(
                state,
                "reply_after_tts",
                now,
                user_id=int(user_id),
                user_name=str(user_name or user_id),
                channel_id=int(channel_id),
            )

    def clear_guild(self, guild_id: int) -> None:
        with self._lock:
            self._guilds.pop(int(guild_id), None)

    @staticmethod
    def _member_map(members: Iterable[dict]) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for raw in members:
            try:
                uid = int(raw.get("id"))
            except (TypeError, ValueError, AttributeError):
                continue
            out[uid] = {
                "id": uid,
                "name": str(raw.get("name") or uid),
                "affinity": float(raw.get("affinity") or 0.0),
            }
        return out

    def snapshot(
        self,
        guild_id: int,
        *,
        connected: bool,
        channel_id: int | None,
        channel_name: str | None,
        current_members: Iterable[dict] = (),
        other_members: Iterable[dict] = (),
        familiar_threshold: float = 0.10,
        avoid_threshold: float = -0.35,
        now: float | None = None,
    ) -> dict:
        now = time.monotonic() if now is None else float(now)
        guild_id = int(guild_id)
        current = self._member_map(current_members)
        other = self._member_map(other_members)
        familiar_threshold = float(familiar_threshold)
        avoid_threshold = float(avoid_threshold)

        with self._lock:
            state = self._guild(guild_id, now)
            self._close_stale_locked(state, now)

            connected = bool(connected)
            normalized_channel = (
                int(channel_id)
                if channel_id is not None
                else None
            )
            if bool(state.get("connected")) != connected:
                state["connected"] = connected
                state["connection_changed"] = now
            if state.get("scene_channel_id") != normalized_channel:
                state["scene_channel_id"] = normalized_channel
                state["scene_started"] = now
                state["last_any_speech"] = None
                for user in state["users"].values():
                    if user.get("speaking"):
                        self._finish_turn_locked(state, user, now)

            cutoff = now - self.recent_window_seconds
            speakers: list[dict] = []
            recent_turns = 0
            for uid, member in current.items():
                user = state["users"].get(uid)
                if user is None:
                    continue
                turns = user.get("turn_starts")
                if isinstance(turns, deque):
                    while turns and float(turns[0]) < cutoff:
                        turns.popleft()
                    turns_60s = len(turns)
                else:
                    turns_60s = 0
                recent_turns += turns_60s

                if (
                    user.get("speaking")
                    and int(user.get("channel_id", 0))
                    == (normalized_channel or 0)
                ):
                    speakers.append({
                        "id": uid,
                        "name": member["name"],
                        "affinity": float(member["affinity"]),
                        "speaking_for": max(
                            0.0,
                            now - float(user.get("speaking_since", now)),
                        ),
                        "packet_age": max(
                            0.0,
                            now - float(user.get("last_packet", now)),
                        ),
                        "turns_60s": turns_60s,
                        "last_turn_seconds": float(
                            user.get("last_turn_seconds", 0.0)
                        ),
                    })

            speaker_count = len(speakers)
            overlap_count = max(0, speaker_count - 1)
            if connected and speaker_count == 0:
                reference = state.get("last_any_speech")
                if reference is None:
                    reference = state.get("scene_started", now)
                silence_seconds = max(0.0, now - float(reference))
            else:
                silence_seconds = 0.0

            current_affinities = [
                float(row["affinity"]) for row in current.values()
            ]
            other_affinities = [
                float(row["affinity"]) for row in other.values()
            ]
            all_affinities = current_affinities + other_affinities

            reply = state.get("last_reply")
            reply_age = (
                max(0.0, now - float(reply["time"]))
                if reply
                else None
            )
            reply_active = bool(
                reply is not None
                and reply_age is not None
                and reply_age <= self.reply_window_seconds
            )

            events = []
            for event in list(state["events"])[-12:]:
                row = dict(event)
                row["age_seconds"] = max(
                    0.0,
                    now - float(event.get("time", now)),
                )
                row.pop("time", None)
                events.append(row)

            last_transcript = state.get("last_transcript")
            transcript_row = None
            if last_transcript is not None:
                transcript_row = dict(last_transcript)
                transcript_row["age_seconds"] = max(
                    0.0,
                    now - float(last_transcript.get("time", now)),
                )
                transcript_row.pop("time", None)

            status = (
                "SPEAKING"
                if speaker_count
                else "SILENCE"
                if connected
                else "OUTSIDE"
            )
            return {
                "enabled": True,
                "status": status,
                "connected": connected,
                "guild_id": guild_id,
                "channel_id": normalized_channel,
                "channel_name": str(channel_name or ""),
                "human_count": len(current),
                "speaker_count": speaker_count,
                "overlap_count": overlap_count,
                "silence_seconds": silence_seconds,
                "turns_per_minute": int(recent_turns),
                "speakers": speakers,
                "familiar_here": sum(
                    1
                    for row in current.values()
                    if float(row["affinity"]) >= familiar_threshold
                ),
                "liked_here": sum(
                    1
                    for row in current.values()
                    if float(row["affinity"]) >= 0.35
                ),
                "disliked_here": sum(
                    1
                    for row in current.values()
                    if float(row["affinity"]) <= avoid_threshold
                ),
                "other_voice_humans": len(other),
                "other_familiar_humans": sum(
                    1
                    for row in other.values()
                    if float(row["affinity"]) >= familiar_threshold
                ),
                "affinity_mean": (
                    sum(all_affinities) / len(all_affinities)
                    if all_affinities
                    else 0.0
                ),
                "affinity_here_mean": (
                    sum(current_affinities) / len(current_affinities)
                    if current_affinities
                    else 0.0
                ),
                "reply_after_tts": reply_active,
                "reply_user_id": (
                    int(reply["user_id"])
                    if reply_active and reply is not None
                    else None
                ),
                "reply_user_name": (
                    str(reply.get("user_name", ""))
                    if reply_active and reply is not None
                    else ""
                ),
                "reply_age_seconds": (
                    reply_age if reply_active else None
                ),
                "reply_tts_age_seconds": (
                    float(reply.get("tts_age_seconds", 0.0))
                    if reply_active and reply is not None
                    else None
                ),
                "scene_age_seconds": (
                    max(0.0, now - float(state.get("scene_started", now)))
                    if connected
                    else 0.0
                ),
                "outside_seconds": (
                    max(
                        0.0,
                        now - float(state.get("connection_changed", now)),
                    )
                    if not connected
                    else 0.0
                ),
                "last_transcript": transcript_row,
                "events": events,
                "updated_at": time.time(),
            }
