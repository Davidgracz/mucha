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
                "events": deque(maxlen=96),
                "turn_history": deque(maxlen=256),
                "speech_intervals": deque(maxlen=256),
                "overlap_starts": deque(maxlen=128),
                "handoffs": deque(maxlen=128),
                "last_turn_user_id": None,
                "last_turn_stop": None,
                "last_reply": None,
                "last_tts": None,
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
        user["last_stop"] = last_packet
        if duration > 0.0:
            state["speech_intervals"].append({
                "start": started,
                "end": last_packet,
                "user_id": int(user["user_id"]),
            })
        state["last_turn_user_id"] = int(user["user_id"])
        state["last_turn_stop"] = last_packet
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
                active_others = sum(
                    1
                    for other_id, other in state["users"].items()
                    if int(other_id) != user_id
                    and other.get("speaking")
                    and int(other.get("channel_id", 0)) == channel_id
                )
                previous_user = state.get("last_turn_user_id")
                previous_stop = state.get("last_turn_stop")
                user["speaking"] = True
                user["speaking_since"] = now
                user["turn_starts"].append(now)
                state["turn_history"].append((now, user_id))
                self._append_event(
                    state,
                    "speech_start",
                    now,
                    user_id=user_id,
                    user_name=user["name"],
                    channel_id=channel_id,
                )
                if active_others > 0:
                    state["overlap_starts"].append(now)
                    self._append_event(
                        state,
                        "overlap_start",
                        now,
                        user_id=user_id,
                        user_name=user["name"],
                        channel_id=channel_id,
                    )
                elif (
                    previous_user is not None
                    and int(previous_user) != user_id
                    and previous_stop is not None
                ):
                    latency = max(
                        0.0,
                        now - float(previous_stop),
                    )
                    if latency <= 5.0:
                        state["handoffs"].append({
                            "time": now,
                            "from_user_id": int(previous_user),
                            "to_user_id": user_id,
                            "latency": latency,
                        })
                        self._append_event(
                            state,
                            "turn_handoff",
                            now,
                            user_id=user_id,
                            user_name=user["name"],
                            channel_id=channel_id,
                            duration=latency,
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
            transcript_text = str(text or "")[:180]
            transcript_duration = max(0.0, float(duration))
            word_count = len(
                [part for part in transcript_text.split() if part]
            )
            words_per_second = (
                word_count / transcript_duration
                if transcript_duration > 0.05
                else 0.0
            )
            user["last_transcript"] = transcript_text
            user["last_transcript_duration"] = transcript_duration
            user["last_transcript_at"] = now
            state["last_transcript"] = {
                "user_id": int(user_id),
                "user_name": user["name"],
                "channel_id": int(channel_id),
                "text": transcript_text,
                "duration": transcript_duration,
                "word_count": word_count,
                "words_per_second": words_per_second,
                "time": now,
            }

    def note_tts(
        self,
        guild_id: int,
        channel_id: int,
        text: str,
        *,
        now: float | None = None,
    ) -> None:
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            state = self._guild(int(guild_id), now)
            state["last_tts"] = {
                "channel_id": int(channel_id),
                "text": str(text or "")[:180],
                "time": now,
                "replied": False,
                "reply_user_id": None,
            }
            self._append_event(
                state,
                "mucha_tts",
                now,
                channel_id=int(channel_id),
            )

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
            last_tts = state.get("last_tts")
            if (
                last_tts is not None
                and int(last_tts.get("channel_id", 0)) == int(channel_id)
            ):
                last_tts["replied"] = True
                last_tts["reply_user_id"] = int(user_id)
            self._append_event(
                state,
                "reply_after_tts",
                now,
                user_id=int(user_id),
                user_name=str(user_name or user_id),
                channel_id=int(channel_id),
            )

    def _recent_dynamics_locked(
        self,
        state: dict,
        now: float,
        current: dict[int, dict],
        channel_id: int | None,
    ) -> dict:
        window = float(self.recent_window_seconds)
        cutoff = now - window

        intervals = state["speech_intervals"]
        while (
            intervals
            and float(intervals[0].get("end", 0.0)) < cutoff
        ):
            intervals.popleft()

        turn_history = state["turn_history"]
        while turn_history and float(turn_history[0][0]) < cutoff:
            turn_history.popleft()

        overlap_starts = state["overlap_starts"]
        while overlap_starts and float(overlap_starts[0]) < cutoff:
            overlap_starts.popleft()

        handoffs = state["handoffs"]
        while (
            handoffs
            and float(handoffs[0].get("time", 0.0)) < cutoff
        ):
            handoffs.popleft()

        clipped: list[tuple[float, float, int]] = []
        per_user: dict[int, float] = {}
        finished_turn_lengths: list[float] = []
        for row in intervals:
            start = max(cutoff, float(row.get("start", now)))
            end = min(now, float(row.get("end", now)))
            if end <= start:
                continue
            user_id = int(row.get("user_id", 0))
            clipped.append((start, end, user_id))
            duration = end - start
            per_user[user_id] = per_user.get(user_id, 0.0) + duration
            finished_turn_lengths.append(duration)

        normalized_channel = int(channel_id) if channel_id is not None else None
        active_current = 0
        for user_id, user in state["users"].items():
            if (
                not user.get("speaking")
                or normalized_channel is None
                or int(user.get("channel_id", 0)) != normalized_channel
            ):
                continue
            start = max(
                cutoff,
                float(user.get("speaking_since", now)),
            )
            if now <= start:
                continue
            uid = int(user_id)
            active_current += 1
            clipped.append((start, now, uid))
            per_user[uid] = per_user.get(uid, 0.0) + (now - start)

        merged: list[list[float]] = []
        for start, end, _ in sorted(clipped):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        speech_seconds = sum(end - start for start, end in merged)
        speech_ratio = max(0.0, min(1.0, speech_seconds / window))

        turns_recent = list(turn_history)
        unique_recent = len({int(row[1]) for row in turns_recent})
        switches = sum(
            1
            for left, right in zip(
                turns_recent,
                turns_recent[1:],
            )
            if int(left[1]) != int(right[1])
        )

        aggregate_speaker_seconds = sum(per_user.values())
        top_speaker_id = None
        top_speaker_seconds = 0.0
        if per_user:
            top_speaker_id, top_speaker_seconds = max(
                per_user.items(),
                key=lambda item: item[1],
            )
        dominance = (
            top_speaker_seconds / aggregate_speaker_seconds
            if aggregate_speaker_seconds > 1e-9
            else 0.0
        )

        handoff_rows = list(handoffs)
        mean_handoff = (
            sum(float(row.get("latency", 0.0)) for row in handoff_rows)
            / len(handoff_rows)
            if handoff_rows
            else None
        )
        mean_turn = (
            sum(finished_turn_lengths) / len(finished_turn_lengths)
            if finished_turn_lengths
            else 0.0
        )
        longest_turn = (
            max(finished_turn_lengths)
            if finished_turn_lengths
            else 0.0
        )
        overlap_events = len(overlap_starts)
        turns_count = len(turns_recent)

        intensity = max(
            0.0,
            min(
                1.0,
                0.40 * min(1.0, turns_count / 18.0)
                + 0.25 * min(1.0, switches / 10.0)
                + 0.25 * speech_ratio
                + 0.10 * min(1.0, unique_recent / 4.0),
            ),
        )

        if overlap_events >= 2 or active_current >= 2:
            mode = "CROSSTALK"
        elif (
            aggregate_speaker_seconds >= 5.0
            and dominance >= 0.72
        ):
            mode = "MONOLOGUE"
        elif switches >= 2 and unique_recent >= 2:
            mode = "DIALOGUE"
        elif active_current >= 1:
            mode = "CONVERSATION"
        elif speech_ratio <= 0.03:
            mode = "QUIET"
        else:
            mode = "CONVERSATION"

        top_member = current.get(int(top_speaker_id)) if top_speaker_id else None
        return {
            "window_seconds": window,
            "speech_seconds": speech_seconds,
            "speech_ratio": speech_ratio,
            "unique_speakers": unique_recent,
            "speaker_switches": switches,
            "overlap_events": overlap_events,
            "handoff_count": len(handoff_rows),
            "mean_handoff_seconds": mean_handoff,
            "mean_turn_seconds": mean_turn,
            "longest_turn_seconds": longest_turn,
            "top_speaker_id": top_speaker_id,
            "top_speaker_name": (
                str(top_member.get("name", top_speaker_id))
                if top_member is not None
                else (
                    str(top_speaker_id)
                    if top_speaker_id is not None
                    else ""
                )
            ),
            "top_speaker_seconds": top_speaker_seconds,
            "dominance": dominance,
            "conversation_intensity": intensity,
            "conversation_mode": mode,
        }

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

            dynamics = self._recent_dynamics_locked(
                state,
                now,
                current,
                normalized_channel,
            )

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
            last_tts = state.get("last_tts")
            tts_age = (
                max(0.0, now - float(last_tts["time"]))
                if last_tts is not None
                else None
            )
            tts_pending_reply = bool(
                last_tts is not None
                and not bool(last_tts.get("replied"))
                and tts_age is not None
                and tts_age <= self.reply_window_seconds
                and (
                    normalized_channel is None
                    or int(last_tts.get("channel_id", 0))
                    == normalized_channel
                )
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
                "speech_seconds_60s": float(
                    dynamics["speech_seconds"]
                ),
                "speech_ratio_60s": float(
                    dynamics["speech_ratio"]
                ),
                "unique_speakers_60s": int(
                    dynamics["unique_speakers"]
                ),
                "speaker_switches_60s": int(
                    dynamics["speaker_switches"]
                ),
                "overlap_events_60s": int(
                    dynamics["overlap_events"]
                ),
                "handoff_count_60s": int(
                    dynamics["handoff_count"]
                ),
                "mean_handoff_seconds": (
                    float(dynamics["mean_handoff_seconds"])
                    if dynamics["mean_handoff_seconds"] is not None
                    else None
                ),
                "mean_turn_seconds": float(
                    dynamics["mean_turn_seconds"]
                ),
                "longest_turn_seconds": float(
                    dynamics["longest_turn_seconds"]
                ),
                "top_speaker_id": dynamics["top_speaker_id"],
                "top_speaker_name": str(
                    dynamics["top_speaker_name"]
                ),
                "speaker_dominance": float(
                    dynamics["dominance"]
                ),
                "conversation_intensity": float(
                    dynamics["conversation_intensity"]
                ),
                "conversation_mode": str(
                    dynamics["conversation_mode"]
                ),
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
                "tts_pending_reply": tts_pending_reply,
                "tts_age_seconds": (
                    tts_age if tts_pending_reply else None
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
