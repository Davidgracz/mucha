from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sqlite3
import time


@dataclass(slots=True)
class VoiceEpisode:
    time: float
    guild_id: int
    channel_id: int | None
    channel_name: str
    user_ids: tuple[int, ...]
    user_names: tuple[str, ...]
    context: str
    scene_key: str
    action: str
    predicted_reward: float
    actual_reward: float
    prediction_error: float
    source: str


class VoiceEpisodicMemory:
    """Persistent episodic memory and reward predictor for voice behavior."""

    def __init__(
        self,
        *,
        max_events: int = 256,
        learning_rate: float = 0.20,
        database: str | Path | None = None,
        max_persisted_events: int = 10000,
        consolidation_gain: float = 0.08,
        forgetting_half_life_days: float = 14.0,
        forgetting_interval_seconds: int = 300,
        consolidated_threshold: float = 0.35,
        semantic_memory_enabled: bool = True,
        semantic_recall_min_observations: int = 2,
    ):
        self.max_events = max(16, int(max_events))
        self.max_persisted_events = max(
            self.max_events,
            int(max_persisted_events),
        )
        self.learning_rate = max(
            0.001,
            min(1.0, float(learning_rate)),
        )
        self.consolidation_gain = max(
            0.0,
            min(1.0, float(consolidation_gain)),
        )
        self.forgetting_half_life_days = max(
            0.25,
            float(forgetting_half_life_days),
        )
        self.forgetting_interval_seconds = max(
            30,
            int(forgetting_interval_seconds),
        )
        self.consolidated_threshold = max(
            0.0,
            min(1.0, float(consolidated_threshold)),
        )
        self.semantic_memory_enabled = bool(
            semantic_memory_enabled
        )
        self.semantic_recall_min_observations = max(
            1,
            int(semantic_recall_min_observations),
        )
        self._semantic: dict[
            tuple[str, str, str],
            dict,
        ] = {}
        self._person_profile_cache: dict[int, dict] = {}
        self._channel_profile_cache: dict[int, dict] = {}
        self._social_scene_profile_cache: dict[str, dict] = {}
        self._voice_dynamics_profile_cache: dict[str, dict] = {}
        self._last_forgetting_at = time.time()
        self._last_forgetting_diag: dict = {
            "ran": False,
            "factor": 1.0,
            "elapsed_seconds": 0.0,
        }
        self._consolidation: dict[
            tuple[str, str],
            dict,
        ] = {}
        self._values: dict[
            tuple[str, str, str],
            float,
        ] = {}
        self._counts: dict[
            tuple[str, str, str],
            int,
        ] = {}
        self._episodes: deque[VoiceEpisode] = deque(
            maxlen=self.max_events
        )
        self.path = (
            Path(database)
            if database is not None
            else None
        )
        self.db: sqlite3.Connection | None = None
        if self.path is not None:
            self.path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            self.db = sqlite3.connect(
                self.path,
                timeout=5.0,
            )
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=NORMAL")
            self.db.execute("PRAGMA busy_timeout=5000")
            self._create_schema()
            self._load_persistent_state()

    @staticmethod
    def make_scene_key(
        channel_id: int | None,
        user_ids: list[int] | tuple[int, ...],
    ) -> str:
        users = ",".join(
            str(int(x))
            for x in sorted(set(int(x) for x in user_ids))
        )
        channel = (
            str(int(channel_id))
            if channel_id is not None
            else "outside"
        )
        return f"channel={channel}|users={users or '-'}"

    @staticmethod
    def semantic_concepts(
        context: str,
        channel_id: int | None,
        user_ids: list[int] | tuple[int, ...],
    ) -> list[tuple[str, str]]:
        concepts: list[tuple[str, str]] = []
        context = str(context or "").strip()
        if context:
            concepts.append(("state", context))

        channel_key = (
            str(int(channel_id))
            if channel_id is not None
            else ""
        )
        if channel_key:
            concepts.append(("channel", channel_key))

        users = sorted(set(int(x) for x in user_ids))
        for user_id in users:
            user_key = str(user_id)
            concepts.append(("user", user_key))
            if channel_key:
                concepts.append(
                    (
                        "user_channel",
                        f"{user_key}@{channel_key}",
                    )
                )
            if context:
                concepts.append(
                    (
                        "user_state",
                        f"{user_key}@{context}",
                    )
                )
        return concepts

    @staticmethod
    def _semantic_confidence(entry: dict) -> float:
        observations = max(
            0,
            int(entry.get("observations", 0)),
        )
        if observations <= 0:
            return 0.0
        positive = max(
            0,
            int(entry.get("positive_count", 0)),
        )
        negative = max(
            0,
            int(entry.get("negative_count", 0)),
        )
        signed = positive + negative
        agreement = (
            abs(positive - negative) / signed
            if signed > 0
            else 0.0
        )
        reward_abs_mean = min(
            1.0,
            max(
                0.0,
                float(entry.get("reward_abs_sum", 0.0))
                / max(1, observations),
            ),
        )
        maturity = observations / (observations + 3.0)
        return max(
            0.0,
            min(
                1.0,
                maturity
                * (
                    0.45
                    + 0.35 * agreement
                    + 0.20 * reward_abs_mean
                ),
            ),
        )

    def _persist_semantic_entry(
        self,
        concept_type: str,
        concept_key: str,
        action: str,
        entry: dict,
    ) -> None:
        if self.db is None:
            return
        self.db.execute(
            """
            INSERT INTO voice_semantic_memory(
                concept_type, concept_key, action,
                expected_reward, observations,
                positive_count, negative_count,
                reward_abs_sum, last_reward, updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(concept_type, concept_key, action)
            DO UPDATE SET
                expected_reward=excluded.expected_reward,
                observations=excluded.observations,
                positive_count=excluded.positive_count,
                negative_count=excluded.negative_count,
                reward_abs_sum=excluded.reward_abs_sum,
                last_reward=excluded.last_reward,
                updated_at=excluded.updated_at
            """,
            (
                concept_type,
                concept_key,
                action,
                float(entry["expected_reward"]),
                int(entry["observations"]),
                int(entry["positive_count"]),
                int(entry["negative_count"]),
                float(entry["reward_abs_sum"]),
                float(entry["last_reward"]),
                float(entry["updated_at"]),
            ),
        )

    def _update_semantic_entry(
        self,
        *,
        concept_type: str,
        concept_key: str,
        action: str,
        actual_reward: float,
        now: float,
        persist: bool = True,
    ) -> dict:
        key = (
            str(concept_type),
            str(concept_key),
            str(action),
        )
        entry = self._semantic.get(key)
        if entry is None:
            entry = {
                "expected_reward": 0.0,
                "observations": 0,
                "positive_count": 0,
                "negative_count": 0,
                "reward_abs_sum": 0.0,
                "last_reward": 0.0,
                "updated_at": 0.0,
            }
            self._semantic[key] = entry

        actual = max(
            -1.0,
            min(1.0, float(actual_reward)),
        )
        previous_n = max(
            0,
            int(entry["observations"]),
        )
        new_n = previous_n + 1
        before = float(entry["expected_reward"])
        entry["expected_reward"] = max(
            -1.0,
            min(
                1.0,
                before + (actual - before) / new_n,
            ),
        )
        entry["observations"] = new_n
        if actual > 1e-9:
            entry["positive_count"] = (
                int(entry["positive_count"]) + 1
            )
        elif actual < -1e-9:
            entry["negative_count"] = (
                int(entry["negative_count"]) + 1
            )
        entry["reward_abs_sum"] = (
            float(entry["reward_abs_sum"])
            + abs(actual)
        )
        entry["last_reward"] = actual
        entry["updated_at"] = float(now)
        self._person_profile_cache.clear()
        self._channel_profile_cache.clear()
        self._social_scene_profile_cache.clear()
        self._voice_dynamics_profile_cache.clear()
        if persist:
            self._persist_semantic_entry(
                key[0],
                key[1],
                key[2],
                entry,
            )
        result = dict(entry)
        result.update({
            "concept_type": key[0],
            "concept_key": key[1],
            "action": key[2],
            "confidence": self._semantic_confidence(entry),
        })
        result["strength"] = (
            result["confidence"]
            * abs(float(result["expected_reward"]))
        )
        return result

    def _update_semantics(
        self,
        *,
        context: str,
        channel_id: int | None,
        user_ids: list[int] | tuple[int, ...],
        action: str,
        actual_reward: float,
        now: float,
        persist: bool = True,
    ) -> list[dict]:
        if not self.semantic_memory_enabled:
            return []
        return [
            self._update_semantic_entry(
                concept_type=concept_type,
                concept_key=concept_key,
                action=action,
                actual_reward=actual_reward,
                now=now,
                persist=persist,
            )
            for concept_type, concept_key in self.semantic_concepts(
                context,
                channel_id,
                user_ids,
            )
        ]

    def _bootstrap_semantics_from_db(self) -> None:
        if self.db is None or not self.semantic_memory_enabled:
            return
        rows = self.db.execute(
            """
            SELECT created_at, guild_id, channel_id, channel_name,
                   user_ids_json, user_names_json, context,
                   scene_key, action, predicted_reward,
                   actual_reward, prediction_error, source
            FROM voice_episodes
            ORDER BY id DESC
            LIMIT 2000
            """
        ).fetchall()
        for row in reversed(rows):
            episode = self._episode_from_row(row)
            self._update_semantics(
                context=episode.context,
                channel_id=episode.channel_id,
                user_ids=episode.user_ids,
                action=episode.action,
                actual_reward=episode.actual_reward,
                now=episode.time,
                persist=True,
            )
        if rows:
            self.db.commit()

    def _backfill_person_semantics_from_db(self) -> None:
        """Create user+state concepts for older episode databases once."""
        if self.db is None or not self.semantic_memory_enabled:
            return
        if any(
            concept_type == "user_state"
            for concept_type, _, _ in self._semantic.keys()
        ):
            return

        rows = self.db.execute(
            """
            SELECT created_at, guild_id, channel_id, channel_name,
                   user_ids_json, user_names_json, context,
                   scene_key, action, predicted_reward,
                   actual_reward, prediction_error, source
            FROM voice_episodes
            ORDER BY id DESC
            LIMIT 5000
            """
        ).fetchall()
        for row in reversed(rows):
            episode = self._episode_from_row(row)
            context = str(episode.context or "").strip()
            if not context:
                continue
            for user_id in sorted(set(episode.user_ids)):
                self._update_semantic_entry(
                    concept_type="user_state",
                    concept_key=f"{int(user_id)}@{context}",
                    action=episode.action,
                    actual_reward=episode.actual_reward,
                    now=episode.time,
                    persist=True,
                )
        if rows:
            self.db.commit()

    def _backfill_channel_semantics_from_db(self) -> None:
        """Create durable channel/place concepts for older episode DBs once."""
        if self.db is None or not self.semantic_memory_enabled:
            return
        if any(
            concept_type == "channel_visit"
            for concept_type, _, _ in self._semantic.keys()
        ):
            return

        rows = self.db.execute(
            """
            SELECT created_at, guild_id, channel_id, channel_name,
                   user_ids_json, user_names_json, context,
                   scene_key, action, predicted_reward,
                   actual_reward, prediction_error, source
            FROM voice_episodes
            WHERE channel_id IS NOT NULL
            ORDER BY id DESC
            LIMIT 5000
            """
        ).fetchall()
        for row in reversed(rows):
            episode = self._episode_from_row(row)
            if episode.channel_id is None:
                continue
            channel_key = str(int(episode.channel_id))
            self._update_semantic_entry(
                concept_type="channel_visit",
                concept_key=channel_key,
                action="historical_episode",
                actual_reward=0.0,
                now=episode.time,
                persist=True,
            )
            if episode.channel_name:
                self._update_semantic_entry(
                    concept_type="channel_meta",
                    concept_key=channel_key,
                    action=f"name:{episode.channel_name}",
                    actual_reward=0.0,
                    now=episode.time,
                    persist=True,
                )
            for user_id in sorted(set(episode.user_ids)):
                self._update_semantic_entry(
                    concept_type="channel_people",
                    concept_key=channel_key,
                    action=str(int(user_id)),
                    actual_reward=0.0,
                    now=episode.time,
                    persist=True,
                )
        if rows:
            self.db.commit()

    @staticmethod
    def make_voice_dynamics_key(snapshot: dict) -> str:
        """Build a channel/person-independent signature of conversation dynamics."""
        snapshot = dict(snapshot or {})
        mode = str(
            snapshot.get("conversation_mode") or "UNKNOWN"
        ).upper()

        def bucket01(value, maximum: int = 3) -> int:
            return min(
                maximum,
                max(
                    0,
                    int(
                        round(
                            max(0.0, min(1.0, float(value or 0.0)))
                            * maximum
                        )
                    ),
                ),
            )

        switches = max(
            0,
            int(snapshot.get("speaker_switches_60s", 0) or 0),
        )
        if switches <= 0:
            switch_bucket = 0
        elif switches <= 2:
            switch_bucket = 1
        elif switches <= 6:
            switch_bucket = 2
        else:
            switch_bucket = 3

        overlaps = max(
            0,
            int(snapshot.get("overlap_events_60s", 0) or 0),
        )
        overlap_bucket = (
            0
            if overlaps <= 0
            else 1
            if overlaps == 1
            else 2
            if overlaps <= 3
            else 3
        )

        handoff = snapshot.get("mean_handoff_seconds")
        if handoff is None:
            handoff_bucket = "none"
        else:
            handoff_value = max(0.0, float(handoff))
            handoff_bucket = (
                "rapid"
                if handoff_value < 0.35
                else "normal"
                if handoff_value < 1.0
                else "slow"
                if handoff_value < 2.5
                else "gap"
            )

        turn = max(
            0.0,
            float(snapshot.get("mean_turn_seconds", 0.0) or 0.0),
        )
        turn_bucket = (
            "none"
            if turn <= 0.0
            else "short"
            if turn < 1.5
            else "medium"
            if turn < 4.0
            else "long"
            if turn < 9.0
            else "verylong"
        )

        silence = max(
            0.0,
            float(snapshot.get("silence_seconds", 0.0) or 0.0),
        )
        silence_bucket = (
            "active"
            if silence < 1.0
            else "pause"
            if silence < 3.0
            else "quiet"
            if silence < 10.0
            else "long"
        )

        transcript = dict(snapshot.get("last_transcript") or {})
        rate = max(
            0.0,
            float(transcript.get("words_per_second", 0.0) or 0.0),
        )
        rate_bucket = (
            "none"
            if rate <= 0.0
            else "slow"
            if rate < 1.4
            else "normal"
            if rate < 2.6
            else "fast"
            if rate < 4.0
            else "veryfast"
        )

        return (
            f"mode={mode}"
            f"|int={bucket01(snapshot.get('conversation_intensity', 0.0))}"
            f"|speech={bucket01(snapshot.get('speech_ratio_60s', 0.0))}"
            f"|switch={switch_bucket}"
            f"|overlap={overlap_bucket}"
            f"|handoff={handoff_bucket}"
            f"|turn={turn_bucket}"
            f"|dom={bucket01(snapshot.get('speaker_dominance', 0.0))}"
            f"|silence={silence_bucket}"
            f"|rate={rate_bucket}"
        )

    @staticmethod
    def parse_voice_dynamics_key(dynamics_key: str) -> dict:
        parts: dict[str, str] = {}
        for part in str(dynamics_key or "").split("|"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            parts[key] = value

        def as_int(name: str) -> int:
            try:
                return int(parts.get(name, "0"))
            except (TypeError, ValueError):
                return 0

        return {
            "dynamics_key": str(dynamics_key or ""),
            "conversation_mode": str(
                parts.get("mode", "UNKNOWN")
            ),
            "intensity_bucket": max(0, min(3, as_int("int"))),
            "speech_bucket": max(0, min(3, as_int("speech"))),
            "switch_bucket": max(0, min(3, as_int("switch"))),
            "overlap_bucket": max(0, min(3, as_int("overlap"))),
            "handoff_bucket": str(parts.get("handoff", "none")),
            "turn_bucket": str(parts.get("turn", "none")),
            "dominance_bucket": max(0, min(3, as_int("dom"))),
            "silence_bucket": str(parts.get("silence", "active")),
            "speech_rate_bucket": str(parts.get("rate", "none")),
        }

    def observe_voice_dynamics_contact(
        self,
        dynamics_key: str,
        *,
        now: float | None = None,
    ) -> dict:
        dynamics_key = str(dynamics_key or "")
        if not dynamics_key or not self.semantic_memory_enabled:
            return {}
        result = self._update_semantic_entry(
            concept_type="voice_dynamics_seen",
            concept_key=dynamics_key,
            action="seen",
            actual_reward=0.0,
            now=float(time.time() if now is None else now),
            persist=True,
        )
        if self.db is not None:
            self.db.commit()
        return result

    def observe_voice_dynamics_outcome(
        self,
        dynamics_key: str,
        action: str,
        actual_reward: float,
        *,
        now: float | None = None,
    ) -> dict:
        dynamics_key = str(dynamics_key or "")
        if not dynamics_key or not self.semantic_memory_enabled:
            return {}
        result = self._update_semantic_entry(
            concept_type="voice_dynamics",
            concept_key=dynamics_key,
            action=str(action or "stay"),
            actual_reward=max(-1.0, min(1.0, float(actual_reward))),
            now=float(time.time() if now is None else now),
            persist=True,
        )
        if self.db is not None:
            self.db.commit()
        return result

    def voice_dynamics_profile(self, dynamics_key: str) -> dict:
        dynamics_key = str(dynamics_key or "")
        cached = self._voice_dynamics_profile_cache.get(dynamics_key)
        if cached is not None:
            return dict(cached)

        seen_rows: list[tuple[str, dict]] = []
        action_rows_raw: list[tuple[str, dict]] = []
        for (
            concept_type,
            concept_key,
            action,
        ), entry in self._semantic.items():
            if concept_key != dynamics_key:
                continue
            if concept_type == "voice_dynamics_seen":
                seen_rows.append((str(action), entry))
            elif concept_type == "voice_dynamics":
                action_rows_raw.append((str(action), entry))

        seen = self._aggregate_person_entries(seen_rows)
        outcomes = self._aggregate_person_entries(action_rows_raw)
        seen_observations = int(seen["observations"])
        outcome_observations = int(outcomes["observations"])
        observations = seen_observations + outcome_observations
        familiarity = (
            1.0 - math.exp(-observations / 8.0)
            if observations > 0
            else 0.0
        )
        confidence = max(
            0.0,
            min(
                1.0,
                0.62 * float(outcomes["confidence"])
                + 0.38 * familiarity,
            ),
        )
        valence = float(outcomes["expected_reward"])
        if valence >= 0.08:
            valence_label = "positive"
        elif valence <= -0.08:
            valence_label = "negative"
        elif (
            int(outcomes["positive_count"]) > 0
            and int(outcomes["negative_count"]) > 0
        ):
            valence_label = "mixed"
        else:
            valence_label = "neutral"

        actions: list[dict] = []
        for action, entry in action_rows_raw:
            row_conf = self._semantic_confidence(entry)
            expected = float(entry.get("expected_reward", 0.0))
            actions.append({
                "action": str(action),
                "expected_reward": expected,
                "confidence": row_conf,
                "signal": expected * row_conf,
                "observations": int(entry.get("observations", 0)),
                "positive_count": int(entry.get("positive_count", 0)),
                "negative_count": int(entry.get("negative_count", 0)),
                "updated_at": float(entry.get("updated_at", 0.0)),
            })
        actions.sort(
            key=lambda row: (
                abs(float(row["signal"])),
                int(row["observations"]),
            ),
            reverse=True,
        )
        preferred = max(
            actions,
            key=lambda row: float(row["signal"]),
            default=None,
        )
        avoided = min(
            actions,
            key=lambda row: float(row["signal"]),
            default=None,
        )
        if preferred is not None and float(preferred["signal"]) <= 0.0:
            preferred = None
        if avoided is not None and float(avoided["signal"]) >= 0.0:
            avoided = None

        profile = {
            **self.parse_voice_dynamics_key(dynamics_key),
            "observations": observations,
            "seen_observations": seen_observations,
            "outcome_observations": outcome_observations,
            "familiarity": max(0.0, min(1.0, familiarity)),
            "confidence": confidence,
            "valence": max(-1.0, min(1.0, valence)),
            "valence_label": valence_label,
            "positive_count": int(outcomes["positive_count"]),
            "negative_count": int(outcomes["negative_count"]),
            "preferred_action": (
                dict(preferred)
                if preferred is not None
                else None
            ),
            "avoided_action": (
                dict(avoided)
                if avoided is not None
                else None
            ),
            "actions": actions[:10],
            "updated_at": max(
                float(seen["updated_at"]),
                float(outcomes["updated_at"]),
            ),
        }
        self._voice_dynamics_profile_cache[dynamics_key] = dict(profile)
        return dict(profile)

    def voice_dynamics_profiles(
        self,
        limit: int = 30,
    ) -> list[dict]:
        limit = max(1, min(100, int(limit)))
        keys = {
            concept_key
            for concept_type, concept_key, _ in self._semantic.keys()
            if concept_type in {"voice_dynamics_seen", "voice_dynamics"}
        }
        rows = [
            self.voice_dynamics_profile(dynamics_key)
            for dynamics_key in keys
        ]
        rows.sort(
            key=lambda row: (
                int(row.get("observations", 0)),
                float(row.get("confidence", 0.0)),
                float(row.get("updated_at", 0.0)),
            ),
            reverse=True,
        )
        return rows[:limit]

    @staticmethod
    def make_social_scene_key(
        *,
        channel_id: int | None,
        user_ids: list[int] | tuple[int, ...],
        context: str,
        conversation_mode: str = "UNKNOWN",
        intensity: float = 0.0,
        speech_ratio: float = 0.0,
        human_count: int | None = None,
        dominant_state: str = "",
        dominant_state_level: float = 0.0,
    ) -> str:
        """Build a stable, bucketed identity for a recurring social situation."""
        users = sorted(set(int(x) for x in user_ids if int(x) > 0))[:6]
        people = ",".join(str(x) for x in users) or "-"
        channel = (
            str(int(channel_id))
            if channel_id is not None
            else "outside"
        )
        mode = str(conversation_mode or "UNKNOWN").upper()
        intensity_bucket = min(
            3,
            max(0, int(round(max(0.0, min(1.0, float(intensity))) * 3.0))),
        )
        speech_bucket = min(
            3,
            max(0, int(round(max(0.0, min(1.0, float(speech_ratio))) * 3.0))),
        )
        humans = (
            len(users)
            if human_count is None
            else max(0, min(6, int(human_count)))
        )
        state = str(dominant_state or "none").lower()
        state_bucket = min(
            3,
            max(
                0,
                int(
                    round(
                        max(
                            0.0,
                            min(1.0, float(dominant_state_level)),
                        )
                        * 3.0
                    )
                ),
            ),
        )
        context = str(context or "")
        context_parts = {}
        for part in context.split("|"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            if key in {"need", "fatigue", "hab", "explore"}:
                context_parts[key] = value
        state_context = ",".join(
            f"{key}:{context_parts.get(key, '0')}"
            for key in ("need", "fatigue", "hab", "explore")
        )
        return (
            f"channel={channel}"
            f"|users={people}"
            f"|mode={mode}"
            f"|int={intensity_bucket}"
            f"|speech={speech_bucket}"
            f"|humans={humans}"
            f"|state={state}:{state_bucket}"
            f"|ctx={state_context}"
        )

    @staticmethod
    def parse_social_scene_key(scene_key: str) -> dict:
        parts: dict[str, str] = {}
        for part in str(scene_key or "").split("|"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            parts[key] = value

        users = []
        for raw in parts.get("users", "-").split(","):
            if raw.isdigit():
                users.append(int(raw))
        state_raw = parts.get("state", "none:0")
        state_name, _, state_bucket_raw = state_raw.partition(":")
        try:
            state_bucket = int(state_bucket_raw or 0)
        except ValueError:
            state_bucket = 0

        def as_int(name: str, default: int = 0) -> int:
            try:
                return int(parts.get(name, default))
            except (TypeError, ValueError):
                return default

        channel_raw = parts.get("channel", "outside")
        return {
            "scene_key": str(scene_key or ""),
            "channel_id": (
                int(channel_raw)
                if channel_raw.isdigit()
                else None
            ),
            "user_ids": users,
            "conversation_mode": str(
                parts.get("mode", "UNKNOWN")
            ),
            "intensity_bucket": max(0, min(3, as_int("int"))),
            "speech_bucket": max(0, min(3, as_int("speech"))),
            "human_count": max(0, min(6, as_int("humans"))),
            "dominant_state": state_name or "none",
            "dominant_state_bucket": max(
                0,
                min(3, state_bucket),
            ),
            "context_signature": str(parts.get("ctx", "")),
        }

    def observe_social_scene_contact(
        self,
        scene_key: str,
        *,
        now: float | None = None,
    ) -> dict:
        """Store one neutral perception of a recurring social situation."""
        scene_key = str(scene_key or "")
        if not scene_key or not self.semantic_memory_enabled:
            return {}
        result = self._update_semantic_entry(
            concept_type="social_scene_seen",
            concept_key=scene_key,
            action="seen",
            actual_reward=0.0,
            now=float(time.time() if now is None else now),
            persist=True,
        )
        if self.db is not None:
            self.db.commit()
        return result

    def observe_social_scene_outcome(
        self,
        scene_key: str,
        action: str,
        actual_reward: float,
        *,
        now: float | None = None,
    ) -> dict:
        """Store the real outcome of an action taken in a social situation."""
        scene_key = str(scene_key or "")
        if not scene_key or not self.semantic_memory_enabled:
            return {}
        result = self._update_semantic_entry(
            concept_type="social_scene",
            concept_key=scene_key,
            action=str(action or "stay"),
            actual_reward=max(-1.0, min(1.0, float(actual_reward))),
            now=float(time.time() if now is None else now),
            persist=True,
        )
        if self.db is not None:
            self.db.commit()
        return result

    def social_scene_profile(self, scene_key: str) -> dict:
        scene_key = str(scene_key or "")
        cached = self._social_scene_profile_cache.get(scene_key)
        if cached is not None:
            return dict(cached)

        seen_rows: list[tuple[str, dict]] = []
        action_rows_raw: list[tuple[str, dict]] = []
        for (
            concept_type,
            concept_key,
            action,
        ), entry in self._semantic.items():
            if concept_key != scene_key:
                continue
            if concept_type == "social_scene_seen":
                seen_rows.append((str(action), entry))
            elif concept_type == "social_scene":
                action_rows_raw.append((str(action), entry))

        seen = self._aggregate_person_entries(seen_rows)
        outcomes = self._aggregate_person_entries(action_rows_raw)
        seen_observations = int(seen["observations"])
        outcome_observations = int(outcomes["observations"])
        observations = seen_observations + outcome_observations
        familiarity = (
            1.0 - math.exp(-observations / 8.0)
            if observations > 0
            else 0.0
        )
        confidence = max(
            0.0,
            min(
                1.0,
                0.60 * float(outcomes["confidence"])
                + 0.40 * familiarity,
            ),
        )
        valence = float(outcomes["expected_reward"])
        if valence >= 0.08:
            valence_label = "positive"
        elif valence <= -0.08:
            valence_label = "negative"
        elif (
            int(outcomes["positive_count"]) > 0
            and int(outcomes["negative_count"]) > 0
        ):
            valence_label = "mixed"
        else:
            valence_label = "neutral"

        actions: list[dict] = []
        for action, entry in action_rows_raw:
            row_conf = self._semantic_confidence(entry)
            expected = float(entry.get("expected_reward", 0.0))
            actions.append({
                "action": str(action),
                "expected_reward": expected,
                "confidence": row_conf,
                "signal": expected * row_conf,
                "observations": int(entry.get("observations", 0)),
                "positive_count": int(entry.get("positive_count", 0)),
                "negative_count": int(entry.get("negative_count", 0)),
                "updated_at": float(entry.get("updated_at", 0.0)),
            })
        actions.sort(
            key=lambda row: (
                abs(float(row["signal"])),
                int(row["observations"]),
            ),
            reverse=True,
        )
        preferred = max(
            actions,
            key=lambda row: float(row["signal"]),
            default=None,
        )
        avoided = min(
            actions,
            key=lambda row: float(row["signal"]),
            default=None,
        )
        if preferred is not None and float(preferred["signal"]) <= 0.0:
            preferred = None
        if avoided is not None and float(avoided["signal"]) >= 0.0:
            avoided = None

        parsed = self.parse_social_scene_key(scene_key)
        profile = {
            **parsed,
            "observations": observations,
            "seen_observations": seen_observations,
            "outcome_observations": outcome_observations,
            "familiarity": max(0.0, min(1.0, familiarity)),
            "confidence": confidence,
            "valence": max(-1.0, min(1.0, valence)),
            "valence_label": valence_label,
            "positive_count": int(outcomes["positive_count"]),
            "negative_count": int(outcomes["negative_count"]),
            "preferred_action": (
                dict(preferred)
                if preferred is not None
                else None
            ),
            "avoided_action": (
                dict(avoided)
                if avoided is not None
                else None
            ),
            "actions": actions[:10],
            "updated_at": max(
                float(seen["updated_at"]),
                float(outcomes["updated_at"]),
            ),
        }
        self._social_scene_profile_cache[scene_key] = dict(profile)
        return dict(profile)

    def social_scene_profiles(
        self,
        limit: int = 30,
    ) -> list[dict]:
        limit = max(1, min(100, int(limit)))
        keys = {
            concept_key
            for concept_type, concept_key, _ in self._semantic.keys()
            if concept_type in {"social_scene_seen", "social_scene"}
        }
        rows = [
            self.social_scene_profile(scene_key)
            for scene_key in keys
        ]
        rows.sort(
            key=lambda row: (
                int(row.get("observations", 0)),
                float(row.get("confidence", 0.0)),
                float(row.get("updated_at", 0.0)),
            ),
            reverse=True,
        )
        return rows[:limit]

    def _backfill_social_scene_semantics_from_db(self) -> None:
        """Create coarse social-scene memories for older episode databases."""
        if self.db is None or not self.semantic_memory_enabled:
            return
        if any(
            concept_type == "social_scene_seen"
            for concept_type, _, _ in self._semantic.keys()
        ):
            return

        rows = self.db.execute(
            """
            SELECT created_at, guild_id, channel_id, channel_name,
                   user_ids_json, user_names_json, context,
                   scene_key, action, predicted_reward,
                   actual_reward, prediction_error, source
            FROM voice_episodes
            ORDER BY id DESC
            LIMIT 5000
            """
        ).fetchall()
        for row in reversed(rows):
            episode = self._episode_from_row(row)
            social_key = self.make_social_scene_key(
                channel_id=episode.channel_id,
                user_ids=episode.user_ids,
                context=episode.context,
                conversation_mode="UNKNOWN",
                intensity=0.0,
                speech_ratio=0.0,
                human_count=len(episode.user_ids),
            )
            self._update_semantic_entry(
                concept_type="social_scene_seen",
                concept_key=social_key,
                action="seen",
                actual_reward=0.0,
                now=episode.time,
                persist=True,
            )
            self._update_semantic_entry(
                concept_type="social_scene",
                concept_key=social_key,
                action=episode.action,
                actual_reward=episode.actual_reward,
                now=episode.time,
                persist=True,
            )
        if rows:
            self.db.commit()

    def semantic_recall(
        self,
        context: str,
        actions: list[str] | tuple[str, ...],
        *,
        channel_id: int | None = None,
        user_ids: list[int] | tuple[int, ...] = (),
        min_observations: int | None = None,
    ) -> dict[str, dict]:
        if not self.semantic_memory_enabled:
            return {}

        minimum = max(
            1,
            int(
                self.semantic_recall_min_observations
                if min_observations is None
                else min_observations
            ),
        )
        concepts = self.semantic_concepts(
            context,
            channel_id,
            user_ids,
        )
        type_weight = {
            "state": 0.70,
            "channel": 0.90,
            "user": 1.00,
            "user_channel": 1.20,
            "user_state": 1.15,
        }
        result: dict[str, dict] = {}
        for action in actions:
            contributors: list[dict] = []
            weighted_total = 0.0
            weight_sum = 0.0
            observations_total = 0
            for concept_type, concept_key in concepts:
                entry = self._semantic.get(
                    (
                        concept_type,
                        concept_key,
                        str(action),
                    )
                )
                if entry is None:
                    continue
                observations = int(
                    entry.get("observations", 0)
                )
                if observations < minimum:
                    continue
                confidence = self._semantic_confidence(entry)
                if confidence <= 1e-6:
                    continue
                specificity = float(
                    type_weight.get(concept_type, 0.60)
                )
                weight = specificity * confidence
                expected = float(
                    entry.get("expected_reward", 0.0)
                )
                weighted_total += weight * expected
                weight_sum += weight
                observations_total += observations
                contributors.append({
                    "concept_type": concept_type,
                    "concept_key": concept_key,
                    "expected_reward": expected,
                    "confidence": confidence,
                    "observations": observations,
                    "weight": weight,
                })

            if weight_sum <= 1e-9:
                continue
            value = max(
                -1.0,
                min(1.0, weighted_total / weight_sum),
            )
            confidence = max(
                0.0,
                min(
                    1.0,
                    1.0 - math.exp(-weight_sum / 1.6),
                ),
            )
            contributors.sort(
                key=lambda row: abs(
                    float(row["weight"])
                    * float(row["expected_reward"])
                ),
                reverse=True,
            )
            result[str(action)] = {
                "expected_reward": value,
                "confidence": confidence,
                "signal": value * confidence,
                "observations": observations_total,
                "contributors": contributors[:8],
            }
        return result

    def semantic_uncertainty(
        self,
        context: str,
        actions: list[str] | tuple[str, ...],
        *,
        channel_id: int | None = None,
        user_ids: list[int] | tuple[int, ...] = (),
    ) -> dict:
        """Estimate how little semantic experience exists for a scene.

        Missing action/concept combinations count as unknown. Existing entries
        become more familiar as observations accumulate and their semantic
        confidence rises. This does not choose an action; it is an information
        signal for curiosity/exploration.
        """
        actions_clean = tuple(
            dict.fromkeys(
                str(action)
                for action in actions
                if str(action)
            )
        )
        concepts = self.semantic_concepts(
            context,
            channel_id,
            user_ids,
        )
        if not actions_clean or not concepts:
            return {
                "uncertainty": 1.0,
                "familiarity": 0.0,
                "observations": 0,
                "known_pairs": 0,
                "possible_pairs": (
                    len(actions_clean) * len(concepts)
                ),
                "concepts": [],
            }

        type_weight = {
            "state": 0.45,
            "channel": 0.90,
            "user": 1.00,
            "user_channel": 1.15,
            "user_state": 1.10,
        }
        rows: list[dict] = []
        weighted_uncertainty = 0.0
        total_weight = 0.0
        observations_total = 0
        known_pairs_total = 0

        for concept_type, concept_key in concepts:
            confidence_sum = 0.0
            observations = 0
            known_pairs = 0
            action_rows: list[dict] = []
            for action in actions_clean:
                entry = self._semantic.get(
                    (
                        concept_type,
                        concept_key,
                        action,
                    )
                )
                if entry is None:
                    action_rows.append({
                        "action": action,
                        "observations": 0,
                        "confidence": 0.0,
                        "expected_reward": 0.0,
                    })
                    continue
                obs = max(
                    0,
                    int(entry.get("observations", 0)),
                )
                confidence = self._semantic_confidence(entry)
                observations += obs
                observations_total += obs
                known_pairs += 1
                known_pairs_total += 1
                confidence_sum += confidence
                action_rows.append({
                    "action": action,
                    "observations": obs,
                    "confidence": confidence,
                    "expected_reward": float(
                        entry.get("expected_reward", 0.0)
                    ),
                })

            count = max(1, len(actions_clean))
            mean_confidence = confidence_sum / count
            maturity = (
                1.0
                - math.exp(
                    -float(observations)
                    / max(1.0, 3.0 * count)
                )
            )
            coverage = known_pairs / count
            familiarity = max(
                0.0,
                min(
                    1.0,
                    0.55 * mean_confidence
                    + 0.25 * maturity
                    + 0.20 * coverage,
                ),
            )
            uncertainty = 1.0 - familiarity
            weight = float(
                type_weight.get(concept_type, 0.60)
            )
            weighted_uncertainty += weight * uncertainty
            total_weight += weight
            rows.append({
                "concept_type": concept_type,
                "concept_key": concept_key,
                "uncertainty": uncertainty,
                "familiarity": familiarity,
                "observations": observations,
                "known_pairs": known_pairs,
                "possible_pairs": count,
                "mean_confidence": mean_confidence,
                "actions": action_rows,
            })

        rows.sort(
            key=lambda row: (
                float(row["uncertainty"]),
                -int(row["observations"]),
            ),
            reverse=True,
        )
        overall = (
            weighted_uncertainty / total_weight
            if total_weight > 1e-9
            else 1.0
        )
        return {
            "uncertainty": max(0.0, min(1.0, overall)),
            "familiarity": max(
                0.0,
                min(1.0, 1.0 - overall),
            ),
            "observations": observations_total,
            "known_pairs": known_pairs_total,
            "possible_pairs": (
                len(actions_clean) * len(concepts)
            ),
            "concepts": rows,
        }

    def _record_channel_history(
        self,
        channel_id: int,
        *,
        channel_name: str = "",
        kind: str,
        source: str = "",
        action: str = "",
        amount: float = 0.0,
        human_count: int = 0,
        user_ids: list[int] | tuple[int, ...] = (),
        conversation_mode: str = "",
        intensity: float = 0.0,
        speech_ratio: float = 0.0,
        context: str = "",
        now: float | None = None,
        prune: bool = True,
    ) -> None:
        if self.db is None:
            return
        channel_id = int(channel_id)
        if channel_id <= 0:
            return
        created_at = float(time.time() if now is None else now)
        clean_users = sorted(set(
            int(x) for x in user_ids if int(x) > 0
        ))
        self.db.execute(
            """
            INSERT INTO channel_interaction_history(
                created_at, channel_id, channel_name, kind, source,
                action, amount, human_count, user_ids_json,
                conversation_mode, intensity, speech_ratio, context
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                created_at,
                channel_id,
                str(channel_name or ""),
                str(kind or "event"),
                str(source or ""),
                str(action or ""),
                max(-1.0, min(1.0, float(amount))),
                max(0, int(human_count)),
                json.dumps(clean_users),
                str(conversation_mode or ""),
                max(0.0, min(1.0, float(intensity))),
                max(0.0, min(1.0, float(speech_ratio))),
                str(context or ""),
            ),
        )
        if prune:
            self.db.execute(
                """
                DELETE FROM channel_interaction_history
                WHERE id NOT IN (
                    SELECT id FROM channel_interaction_history
                    ORDER BY id DESC
                    LIMIT ?
                )
                """,
                (max(1000, self.max_persisted_events * 3),),
            )
        self._channel_profile_cache.pop(channel_id, None)

    def _channel_history_rows(
        self,
        channel_id: int,
        limit: int = 64,
    ) -> list[dict]:
        if self.db is None:
            return []
        limit = max(1, min(200, int(limit)))
        rows = self.db.execute(
            """
            SELECT created_at, channel_name, kind, source, action,
                   amount, human_count, user_ids_json,
                   conversation_mode, intensity, speech_ratio, context
            FROM channel_interaction_history
            WHERE channel_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(channel_id), limit),
        ).fetchall()
        result = []
        for row in rows:
            try:
                user_ids = [
                    int(x)
                    for x in json.loads(row[7] or "[]")
                ]
            except (TypeError, ValueError, json.JSONDecodeError):
                user_ids = []
            result.append({
                "time": float(row[0]),
                "channel_name": str(row[1] or ""),
                "kind": str(row[2] or ""),
                "source": str(row[3] or ""),
                "action": str(row[4] or ""),
                "amount": float(row[5] or 0.0),
                "human_count": int(row[6] or 0),
                "user_ids": user_ids,
                "conversation_mode": str(row[8] or ""),
                "intensity": float(row[9] or 0.0),
                "speech_ratio": float(row[10] or 0.0),
                "context": str(row[11] or ""),
            })
        return result

    def _channel_history_summary(self, channel_id: int) -> dict:
        if self.db is None:
            return {
                "observations": 0,
                "first_seen": 0.0,
                "last_seen": 0.0,
            }
        row = self.db.execute(
            """
            SELECT COUNT(*), MIN(created_at), MAX(created_at)
            FROM channel_interaction_history
            WHERE channel_id = ?
            """,
            (int(channel_id),),
        ).fetchone()
        return {
            "observations": int(row[0] or 0),
            "first_seen": float(row[1] or 0.0),
            "last_seen": float(row[2] or 0.0),
        }

    def _backfill_channel_history_from_db(self) -> None:
        """Seed Stage 29 place chronology from persisted voice episodes."""
        if self.db is None:
            return
        existing = int(
            self.db.execute(
                "SELECT COUNT(*) FROM channel_interaction_history"
            ).fetchone()[0]
        )
        if existing > 0:
            return
        rows = self.db.execute(
            """
            SELECT created_at, channel_id, channel_name,
                   user_ids_json, context, action,
                   actual_reward, source
            FROM voice_episodes
            WHERE channel_id IS NOT NULL
            ORDER BY id ASC
            """
        ).fetchall()
        for (
            created_at,
            channel_id,
            channel_name,
            user_ids_json,
            context,
            action,
            actual_reward,
            source,
        ) in rows:
            try:
                user_ids = [
                    int(x)
                    for x in json.loads(user_ids_json or "[]")
                ]
            except (TypeError, ValueError, json.JSONDecodeError):
                user_ids = []
            self._record_channel_history(
                int(channel_id),
                channel_name=str(channel_name or ""),
                kind="episode",
                source=str(source or ""),
                action=str(action or ""),
                amount=float(actual_reward or 0.0),
                human_count=len(user_ids),
                user_ids=user_ids,
                context=str(context or ""),
                now=float(created_at),
                prune=False,
            )
        self.db.execute(
            """
            DELETE FROM channel_interaction_history
            WHERE id NOT IN (
                SELECT id FROM channel_interaction_history
                ORDER BY id DESC
                LIMIT ?
            )
            """,
            (max(1000, self.max_persisted_events * 3),),
        )
        self.db.commit()

    def observe_channel_visit(
        self,
        channel_id: int,
        channel_name: str,
        user_ids: list[int] | tuple[int, ...] = (),
        *,
        source: str = "visit",
        now: float | None = None,
    ) -> dict:
        """Persist one real encounter with a Discord voice place."""
        channel_id = int(channel_id)
        if channel_id <= 0 or not self.semantic_memory_enabled:
            return {}
        now_value = float(time.time() if now is None else now)
        channel_key = str(channel_id)

        visit = self._update_semantic_entry(
            concept_type="channel_visit",
            concept_key=channel_key,
            action=str(source or "visit"),
            actual_reward=0.0,
            now=now_value,
            persist=True,
        )
        clean_name = str(channel_name or channel_id).strip()
        if clean_name:
            self._update_semantic_entry(
                concept_type="channel_meta",
                concept_key=channel_key,
                action=f"name:{clean_name[:120]}",
                actual_reward=0.0,
                now=now_value,
                persist=True,
            )
        clean_users = sorted(set(
            int(x) for x in user_ids if int(x) > 0
        ))
        for user_id in clean_users:
            self._update_semantic_entry(
                concept_type="channel_people",
                concept_key=channel_key,
                action=str(user_id),
                actual_reward=0.0,
                now=now_value,
                persist=True,
            )
        self._record_channel_history(
            channel_id,
            channel_name=clean_name,
            kind="visit",
            source=str(source or "visit"),
            human_count=len(clean_users),
            user_ids=clean_users,
            now=now_value,
        )
        if self.db is not None:
            self.db.commit()
        return visit

    def observe_channel_dynamics(
        self,
        channel_id: int,
        channel_name: str,
        *,
        conversation_mode: str,
        intensity: float,
        speech_ratio: float,
        human_count: int,
        speaker_user_id: int | None = None,
        now: float | None = None,
    ) -> list[dict]:
        """Persist neutral statistics describing what a place is usually like."""
        channel_id = int(channel_id)
        if channel_id <= 0 or not self.semantic_memory_enabled:
            return []
        now_value = float(time.time() if now is None else now)
        channel_key = str(channel_id)
        mode = str(conversation_mode or "QUIET").upper()
        intensity_bucket = min(
            5,
            max(0, int(round(max(0.0, min(1.0, float(intensity))) * 5.0))),
        )
        speech_bucket = min(
            5,
            max(0, int(round(max(0.0, min(1.0, float(speech_ratio))) * 5.0))),
        )
        humans_bucket = min(6, max(0, int(human_count)))

        rows = [
            self._update_semantic_entry(
                concept_type="channel_mode",
                concept_key=channel_key,
                action=mode,
                actual_reward=0.0,
                now=now_value,
                persist=True,
            ),
            self._update_semantic_entry(
                concept_type="channel_intensity",
                concept_key=channel_key,
                action=str(intensity_bucket),
                actual_reward=0.0,
                now=now_value,
                persist=True,
            ),
            self._update_semantic_entry(
                concept_type="channel_speech",
                concept_key=channel_key,
                action=str(speech_bucket),
                actual_reward=0.0,
                now=now_value,
                persist=True,
            ),
            self._update_semantic_entry(
                concept_type="channel_humans",
                concept_key=channel_key,
                action=str(humans_bucket),
                actual_reward=0.0,
                now=now_value,
                persist=True,
            ),
        ]
        clean_name = str(channel_name or channel_id).strip()
        if clean_name:
            self._update_semantic_entry(
                concept_type="channel_meta",
                concept_key=channel_key,
                action=f"name:{clean_name[:120]}",
                actual_reward=0.0,
                now=now_value,
                persist=True,
            )
        speaker_ids: list[int] = []
        if speaker_user_id is not None and int(speaker_user_id) > 0:
            speaker_ids.append(int(speaker_user_id))
            self._update_semantic_entry(
                concept_type="channel_people",
                concept_key=channel_key,
                action=str(int(speaker_user_id)),
                actual_reward=0.0,
                now=now_value,
                persist=True,
            )
        self._record_channel_history(
            channel_id,
            channel_name=clean_name,
            kind="dynamics",
            source="voice_dynamics",
            human_count=max(0, int(human_count)),
            user_ids=speaker_ids,
            conversation_mode=mode,
            intensity=float(intensity),
            speech_ratio=float(speech_ratio),
            now=now_value,
        )
        if self.db is not None:
            self.db.commit()
        return rows

    @staticmethod
    def _weighted_bucket_mean(
        rows: list[tuple[str, dict]],
        maximum_bucket: int,
    ) -> float:
        total = 0
        weighted = 0.0
        for bucket, entry in rows:
            try:
                value = max(
                    0,
                    min(maximum_bucket, int(bucket)),
                )
            except (TypeError, ValueError):
                continue
            observations = max(
                0,
                int(entry.get("observations", 0)),
            )
            total += observations
            weighted += value * observations
        if total <= 0 or maximum_bucket <= 0:
            return 0.0
        return max(
            0.0,
            min(1.0, weighted / (total * maximum_bucket)),
        )

    def channel_profile(self, channel_id: int) -> dict:
        """Derive a durable model of one Discord voice place."""
        channel_id = int(channel_id)
        cached = self._channel_profile_cache.get(channel_id)
        if cached is not None:
            return dict(cached)

        channel_key = str(channel_id)
        direct_rows: list[tuple[str, dict]] = []
        visit_rows: list[tuple[str, dict]] = []
        meta_rows: list[tuple[str, dict]] = []
        people_rows_raw: list[tuple[str, dict]] = []
        mode_rows_raw: list[tuple[str, dict]] = []
        intensity_rows: list[tuple[str, dict]] = []
        speech_rows: list[tuple[str, dict]] = []
        human_rows: list[tuple[str, dict]] = []

        for (
            concept_type,
            concept_key,
            action,
        ), entry in self._semantic.items():
            if concept_key != channel_key:
                continue
            row = (str(action), entry)
            if concept_type == "channel":
                direct_rows.append(row)
            elif concept_type == "channel_visit":
                visit_rows.append(row)
            elif concept_type == "channel_meta":
                meta_rows.append(row)
            elif concept_type == "channel_people":
                people_rows_raw.append(row)
            elif concept_type == "channel_mode":
                mode_rows_raw.append(row)
            elif concept_type == "channel_intensity":
                intensity_rows.append(row)
            elif concept_type == "channel_speech":
                speech_rows.append(row)
            elif concept_type == "channel_humans":
                human_rows.append(row)

        direct = self._aggregate_person_entries(direct_rows)
        visits = self._aggregate_person_entries(visit_rows)
        reward_observations = int(direct["observations"])
        visit_observations = int(visits["observations"])
        dynamics_observations = sum(
            max(0, int(entry.get("observations", 0)))
            for _, entry in mode_rows_raw
        )
        observations = (
            reward_observations
            + visit_observations
            + dynamics_observations
        )
        familiarity = (
            1.0 - math.exp(-observations / 12.0)
            if observations > 0
            else 0.0
        )
        valence = float(direct["expected_reward"])
        confidence = max(
            0.0,
            min(
                1.0,
                0.60 * float(direct["confidence"])
                + 0.40 * familiarity,
            ),
        )
        if valence >= 0.08:
            valence_label = "positive"
        elif valence <= -0.08:
            valence_label = "negative"
        else:
            valence_label = "neutral"

        action_rows: list[dict] = []
        for action, entry in direct_rows:
            row_conf = self._semantic_confidence(entry)
            expected = float(
                entry.get("expected_reward", 0.0)
            )
            action_rows.append({
                "action": action,
                "expected_reward": expected,
                "confidence": row_conf,
                "signal": expected * row_conf,
                "observations": int(
                    entry.get("observations", 0)
                ),
                "updated_at": float(
                    entry.get("updated_at", 0.0)
                ),
            })
        action_rows.sort(
            key=lambda row: (
                abs(float(row["signal"])),
                int(row["observations"]),
            ),
            reverse=True,
        )
        preferred = max(
            action_rows,
            key=lambda row: float(row["signal"]),
            default=None,
        )
        avoided = min(
            action_rows,
            key=lambda row: float(row["signal"]),
            default=None,
        )
        if preferred is not None and float(preferred["signal"]) <= 0.0:
            preferred = None
        if avoided is not None and float(avoided["signal"]) >= 0.0:
            avoided = None

        channel_name = ""
        if meta_rows:
            latest_meta = max(
                meta_rows,
                key=lambda row: float(
                    row[1].get("updated_at", 0.0)
                ),
            )
            if latest_meta[0].startswith("name:"):
                channel_name = latest_meta[0][5:]

        people = []
        for user_key, entry in people_rows_raw:
            if not str(user_key).isdigit():
                continue
            people.append({
                "user_id": int(user_key),
                "observations": int(
                    entry.get("observations", 0)
                ),
                "updated_at": float(
                    entry.get("updated_at", 0.0)
                ),
            })
        people.sort(
            key=lambda row: (
                int(row["observations"]),
                float(row["updated_at"]),
            ),
            reverse=True,
        )

        modes = []
        for mode, entry in mode_rows_raw:
            modes.append({
                "mode": str(mode),
                "observations": int(
                    entry.get("observations", 0)
                ),
                "updated_at": float(
                    entry.get("updated_at", 0.0)
                ),
            })
        modes.sort(
            key=lambda row: (
                int(row["observations"]),
                float(row["updated_at"]),
            ),
            reverse=True,
        )
        dominant_mode = (
            str(modes[0]["mode"])
            if modes
            else "UNKNOWN"
        )

        recent_episodes = []
        last_seen = max(
            [float(direct["updated_at"]), float(visits["updated_at"])]
            + [
                float(entry.get("updated_at", 0.0))
                for _, entry in mode_rows_raw
            ]
            + [0.0]
        )
        for episode in reversed(self._episodes):
            if episode.channel_id != channel_id:
                continue
            if not channel_name and episode.channel_name:
                channel_name = str(episode.channel_name)
            last_seen = max(last_seen, float(episode.time))
            recent_episodes.append({
                "time": float(episode.time),
                "action": str(episode.action),
                "actual_reward": float(episode.actual_reward),
                "prediction_error": float(
                    episode.prediction_error
                ),
                "human_count": len(episode.user_ids),
                "user_ids": list(episode.user_ids),
                "source": str(episode.source),
            })
            if len(recent_episodes) >= 6:
                break

        profile = {
            "channel_id": channel_id,
            "channel_name": channel_name or str(channel_id),
            "observations": observations,
            "reward_observations": reward_observations,
            "visit_observations": visit_observations,
            "dynamics_observations": dynamics_observations,
            "familiarity": max(0.0, min(1.0, familiarity)),
            "confidence": confidence,
            "valence": max(-1.0, min(1.0, valence)),
            "valence_label": valence_label,
            "dominant_mode": dominant_mode,
            "conversation_modes": modes[:8],
            "mean_intensity": self._weighted_bucket_mean(
                intensity_rows,
                5,
            ),
            "mean_speech_ratio": self._weighted_bucket_mean(
                speech_rows,
                5,
            ),
            "mean_human_density": (
                self._weighted_bucket_mean(human_rows, 6)
                * 6.0
            ),
            "people": people[:12],
            "actions": action_rows[:10],
            "preferred_action": (
                dict(preferred)
                if preferred is not None
                else None
            ),
            "avoided_action": (
                dict(avoided)
                if avoided is not None
                else None
            ),
            "recent_episodes": recent_episodes,
            "updated_at": max(
                float(direct["updated_at"]),
                float(visits["updated_at"]),
                last_seen,
            ),
            "last_seen": last_seen,
        }
        self._channel_profile_cache[channel_id] = dict(profile)
        return dict(profile)

    def channel_profiles(
        self,
        limit: int = 30,
    ) -> list[dict]:
        limit = max(1, min(100, int(limit)))
        channel_ids = sorted({
            int(concept_key)
            for concept_type, concept_key, _ in self._semantic.keys()
            if (
                concept_type
                in {
                    "channel",
                    "channel_visit",
                    "channel_people",
                    "channel_mode",
                    "channel_meta",
                }
                and str(concept_key).isdigit()
            )
        })
        rows = [
            self.channel_profile(channel_id)
            for channel_id in channel_ids
        ]
        rows.sort(
            key=lambda row: (
                int(row.get("observations", 0)),
                float(row.get("confidence", 0.0)),
                float(row.get("updated_at", 0.0)),
            ),
            reverse=True,
        )
        return rows[:limit]

    def _record_person_history(
        self,
        user_id: int,
        *,
        kind: str,
        source: str = "",
        action: str = "",
        amount: float = 0.0,
        user_name: str = "",
        guild_id: int | None = None,
        channel_id: int | None = None,
        channel_name: str = "",
        context: str = "",
        now: float | None = None,
        prune: bool = True,
    ) -> None:
        if self.db is None:
            return
        user_id = int(user_id)
        if user_id <= 0:
            return
        created_at = float(time.time() if now is None else now)
        self.db.execute(
            """
            INSERT INTO person_interaction_history(
                created_at, user_id, user_name, kind, source,
                action, amount, guild_id, channel_id,
                channel_name, context
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                created_at,
                user_id,
                str(user_name or ""),
                str(kind or "event"),
                str(source or ""),
                str(action or ""),
                max(-1.0, min(1.0, float(amount))),
                int(guild_id) if guild_id is not None else None,
                int(channel_id) if channel_id is not None else None,
                str(channel_name or ""),
                str(context or ""),
            ),
        )
        if prune:
            self.db.execute(
                """
                DELETE FROM person_interaction_history
                WHERE id NOT IN (
                    SELECT id FROM person_interaction_history
                    ORDER BY id DESC
                    LIMIT ?
                )
                """,
                (max(1000, self.max_persisted_events * 3),),
            )
        self._person_profile_cache.pop(user_id, None)

    def _person_history_rows(
        self,
        user_id: int,
        limit: int = 48,
    ) -> list[dict]:
        if self.db is None:
            return []
        limit = max(1, min(200, int(limit)))
        rows = self.db.execute(
            """
            SELECT created_at, user_name, kind, source, action,
                   amount, guild_id, channel_id, channel_name, context
            FROM person_interaction_history
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(user_id), limit),
        ).fetchall()
        return [
            {
                "time": float(row[0]),
                "user_name": str(row[1] or ""),
                "kind": str(row[2] or ""),
                "source": str(row[3] or ""),
                "action": str(row[4] or ""),
                "amount": float(row[5] or 0.0),
                "guild_id": (
                    int(row[6]) if row[6] is not None else None
                ),
                "channel_id": (
                    int(row[7]) if row[7] is not None else None
                ),
                "channel_name": str(row[8] or ""),
                "context": str(row[9] or ""),
            }
            for row in rows
        ]

    def _person_history_summary(self, user_id: int) -> dict:
        if self.db is None:
            return {
                "observations": 0,
                "first_seen": 0.0,
                "last_seen": 0.0,
            }
        row = self.db.execute(
            """
            SELECT COUNT(*), MIN(created_at), MAX(created_at)
            FROM person_interaction_history
            WHERE user_id = ?
            """,
            (int(user_id),),
        ).fetchone()
        return {
            "observations": int(row[0] or 0),
            "first_seen": float(row[1] or 0.0),
            "last_seen": float(row[2] or 0.0),
        }

    def _backfill_person_history_from_db(self) -> None:
        """Seed Stage 28 chronology from already persisted voice episodes."""
        if self.db is None:
            return
        existing = int(
            self.db.execute(
                "SELECT COUNT(*) FROM person_interaction_history"
            ).fetchone()[0]
        )
        if existing > 0:
            return
        rows = self.db.execute(
            """
            SELECT created_at, guild_id, channel_id, channel_name,
                   user_ids_json, user_names_json, context,
                   action, actual_reward, source
            FROM voice_episodes
            ORDER BY id ASC
            """
        ).fetchall()
        for row in rows:
            (
                created_at,
                guild_id,
                channel_id,
                channel_name,
                user_ids_json,
                user_names_json,
                context,
                action,
                actual_reward,
                source,
            ) = row
            try:
                user_ids = [
                    int(x)
                    for x in json.loads(user_ids_json or "[]")
                ]
            except (TypeError, ValueError, json.JSONDecodeError):
                user_ids = []
            try:
                user_names = [
                    str(x)
                    for x in json.loads(user_names_json or "[]")
                ]
            except (TypeError, ValueError, json.JSONDecodeError):
                user_names = []
            for pos, user_id in enumerate(user_ids):
                self._record_person_history(
                    user_id,
                    kind="episode",
                    source=str(source or ""),
                    action=str(action or ""),
                    amount=float(actual_reward or 0.0),
                    user_name=(
                        user_names[pos]
                        if pos < len(user_names)
                        else ""
                    ),
                    guild_id=(
                        int(guild_id)
                        if guild_id is not None
                        else None
                    ),
                    channel_id=(
                        int(channel_id)
                        if channel_id is not None
                        else None
                    ),
                    channel_name=str(channel_name or ""),
                    context=str(context or ""),
                    now=float(created_at),
                    prune=False,
                )
        self.db.execute(
            """
            DELETE FROM person_interaction_history
            WHERE id NOT IN (
                SELECT id FROM person_interaction_history
                ORDER BY id DESC
                LIMIT ?
            )
            """,
            (max(1000, self.max_persisted_events * 3),),
        )
        self.db.commit()

    def observe_person_contact(
        self,
        user_id: int,
        source: str,
        *,
        user_name: str = "",
        guild_id: int | None = None,
        channel_id: int | None = None,
        channel_name: str = "",
        context: str = "",
        now: float | None = None,
    ) -> dict:
        """Persist that a known person interacted with Mucha.

        Contact is neutral evidence. It increases familiarity without
        pretending that the interaction itself was rewarding or punishing.
        """
        user_id = int(user_id)
        if user_id <= 0 or not self.semantic_memory_enabled:
            return {}
        event_time = float(time.time() if now is None else now)
        result = self._update_semantic_entry(
            concept_type="person_contact",
            concept_key=str(user_id),
            action=str(source or "unknown"),
            actual_reward=0.0,
            now=event_time,
            persist=True,
        )
        self._record_person_history(
            user_id,
            kind="contact",
            source=str(source or "unknown"),
            amount=0.0,
            user_name=str(user_name or ""),
            guild_id=guild_id,
            channel_id=channel_id,
            channel_name=str(channel_name or ""),
            context=str(context or ""),
            now=event_time,
        )
        if self.db is not None:
            self.db.commit()
        return result

    def observe_person_social_event(
        self,
        user_id: int,
        event: str,
        amount: float,
        *,
        now: float | None = None,
    ) -> dict:
        """Persist signed social evidence for a person."""
        user_id = int(user_id)
        if user_id <= 0 or not self.semantic_memory_enabled:
            return {}
        event_time = float(time.time() if now is None else now)
        signed_amount = max(-1.0, min(1.0, float(amount)))
        result = self._update_semantic_entry(
            concept_type="person_social",
            concept_key=str(user_id),
            action=str(event or "social"),
            actual_reward=signed_amount,
            now=event_time,
            persist=True,
        )
        self._record_person_history(
            user_id,
            kind="social",
            source=str(event or "social"),
            action=str(event or "social"),
            amount=signed_amount,
            now=event_time,
        )
        if self.db is not None:
            self.db.commit()
        return result

    @staticmethod
    def _aggregate_person_entries(
        rows: list[tuple[str, dict]],
    ) -> dict:
        observations = sum(
            max(0, int(entry.get("observations", 0)))
            for _, entry in rows
        )
        positive = sum(
            max(0, int(entry.get("positive_count", 0)))
            for _, entry in rows
        )
        negative = sum(
            max(0, int(entry.get("negative_count", 0)))
            for _, entry in rows
        )
        if observations <= 0:
            return {
                "observations": 0,
                "expected_reward": 0.0,
                "confidence": 0.0,
                "positive_count": positive,
                "negative_count": negative,
                "updated_at": 0.0,
            }
        expected = sum(
            float(entry.get("expected_reward", 0.0))
            * max(0, int(entry.get("observations", 0)))
            for _, entry in rows
        ) / observations
        confidence = sum(
            VoiceEpisodicMemory._semantic_confidence(entry)
            * max(0, int(entry.get("observations", 0)))
            for _, entry in rows
        ) / observations
        updated_at = max(
            (
                float(entry.get("updated_at", 0.0))
                for _, entry in rows
            ),
            default=0.0,
        )
        return {
            "observations": observations,
            "expected_reward": max(-1.0, min(1.0, expected)),
            "confidence": max(0.0, min(1.0, confidence)),
            "positive_count": positive,
            "negative_count": negative,
            "updated_at": updated_at,
        }

    def person_profile(self, user_id: int) -> dict:
        """Derive one durable person model from persistent semantic memory."""
        user_id = int(user_id)
        cached = self._person_profile_cache.get(user_id)
        if cached is not None:
            return dict(cached)

        user_key = str(user_id)
        action_rows: list[dict] = []
        direct_rows: list[tuple[str, dict]] = []
        contact_rows: list[tuple[str, dict]] = []
        social_rows: list[tuple[str, dict]] = []
        channel_groups: dict[str, list[tuple[str, dict]]] = {}
        context_groups: dict[str, list[tuple[str, dict]]] = {}

        for (
            concept_type,
            concept_key,
            action,
        ), entry in self._semantic.items():
            if concept_type == "user" and concept_key == user_key:
                direct_rows.append((action, entry))
                confidence = self._semantic_confidence(entry)
                expected = float(
                    entry.get("expected_reward", 0.0)
                )
                action_rows.append({
                    "action": str(action),
                    "expected_reward": expected,
                    "confidence": confidence,
                    "signal": expected * confidence,
                    "observations": int(
                        entry.get("observations", 0)
                    ),
                    "positive_count": int(
                        entry.get("positive_count", 0)
                    ),
                    "negative_count": int(
                        entry.get("negative_count", 0)
                    ),
                    "updated_at": float(
                        entry.get("updated_at", 0.0)
                    ),
                })
            elif (
                concept_type == "person_contact"
                and concept_key == user_key
            ):
                contact_rows.append((str(action), entry))
            elif (
                concept_type == "person_social"
                and concept_key == user_key
            ):
                social_rows.append((str(action), entry))
            elif (
                concept_type == "user_channel"
                and concept_key.startswith(user_key + "@")
            ):
                channel_key = concept_key.split("@", 1)[1]
                channel_groups.setdefault(
                    channel_key,
                    [],
                ).append((str(action), entry))
            elif (
                concept_type == "user_state"
                and concept_key.startswith(user_key + "@")
            ):
                context_key = concept_key.split("@", 1)[1]
                context_groups.setdefault(
                    context_key,
                    [],
                ).append((str(action), entry))

        direct = self._aggregate_person_entries(direct_rows)
        contacts = self._aggregate_person_entries(contact_rows)
        social = self._aggregate_person_entries(social_rows)

        voice_observations = int(direct["observations"])
        contact_observations = int(contacts["observations"])
        social_observations = int(social["observations"])
        observations = (
            voice_observations
            + contact_observations
            + social_observations
        )
        familiarity = (
            1.0 - math.exp(-observations / 12.0)
            if observations > 0
            else 0.0
        )

        evidence_weight = (
            voice_observations + social_observations
        )
        if evidence_weight > 0:
            valence = (
                float(direct["expected_reward"])
                * voice_observations
                + float(social["expected_reward"])
                * social_observations
            ) / evidence_weight
        else:
            valence = 0.0
        valence = max(-1.0, min(1.0, valence))

        evidence_confidence = (
            (
                float(direct["confidence"]) * voice_observations
                + float(social["confidence"]) * social_observations
            ) / evidence_weight
            if evidence_weight > 0
            else 0.0
        )
        profile_confidence = max(
            0.0,
            min(
                1.0,
                0.55 * evidence_confidence
                + 0.45 * familiarity,
            ),
        )
        positive = (
            int(direct["positive_count"])
            + int(social["positive_count"])
        )
        negative = (
            int(direct["negative_count"])
            + int(social["negative_count"])
        )
        signed_total = positive + negative
        if (
            signed_total >= 4
            and positive >= 1
            and negative >= 1
            and min(positive, negative) / signed_total >= 0.25
        ):
            valence_label = "mixed"
        elif valence >= 0.08:
            valence_label = "positive"
        elif valence <= -0.08:
            valence_label = "negative"
        else:
            valence_label = "neutral"

        action_rows.sort(
            key=lambda row: (
                abs(float(row["signal"])),
                int(row["observations"]),
            ),
            reverse=True,
        )
        preferred = max(
            action_rows,
            key=lambda row: float(row["signal"]),
            default=None,
        )
        avoided = min(
            action_rows,
            key=lambda row: float(row["signal"]),
            default=None,
        )
        if (
            preferred is not None
            and float(preferred["signal"]) <= 0.0
        ):
            preferred = None
        if (
            avoided is not None
            and float(avoided["signal"]) >= 0.0
        ):
            avoided = None

        channel_names: dict[str, str] = {}
        display_name = ""
        last_seen = 0.0
        for episode in reversed(self._episodes):
            if user_id not in episode.user_ids:
                continue
            if last_seen <= 0.0:
                last_seen = float(episode.time)
            try:
                pos = list(episode.user_ids).index(user_id)
                if pos < len(episode.user_names):
                    display_name = str(
                        episode.user_names[pos]
                    )
            except ValueError:
                pass
            if episode.channel_id is not None:
                channel_names[str(int(episode.channel_id))] = str(
                    episode.channel_name or episode.channel_id
                )
            if display_name and len(channel_names) >= 8:
                break

        channel_rows: list[dict] = []
        for channel_key, rows in channel_groups.items():
            agg = self._aggregate_person_entries(rows)
            channel_rows.append({
                "channel_id": int(channel_key)
                if channel_key.isdigit()
                else channel_key,
                "channel_name": channel_names.get(
                    channel_key,
                    channel_key,
                ),
                **agg,
            })
        channel_rows.sort(
            key=lambda row: (
                int(row["observations"]),
                float(row["confidence"]),
                float(row["updated_at"]),
            ),
            reverse=True,
        )

        context_rows: list[dict] = []
        for context_key, rows in context_groups.items():
            agg = self._aggregate_person_entries(rows)
            context_rows.append({
                "context": context_key,
                **agg,
            })
        context_rows.sort(
            key=lambda row: (
                int(row["observations"]),
                float(row["confidence"]),
                float(row["updated_at"]),
            ),
            reverse=True,
        )

        contact_sources = []
        for source, entry in contact_rows:
            contact_sources.append({
                "source": str(source),
                "observations": int(
                    entry.get("observations", 0)
                ),
                "updated_at": float(
                    entry.get("updated_at", 0.0)
                ),
            })
        contact_sources.sort(
            key=lambda row: (
                int(row["observations"]),
                float(row["updated_at"]),
            ),
            reverse=True,
        )

        social_events = []
        for event, entry in social_rows:
            confidence = self._semantic_confidence(entry)
            expected = float(
                entry.get("expected_reward", 0.0)
            )
            social_events.append({
                "event": str(event),
                "expected_reward": expected,
                "confidence": confidence,
                "signal": expected * confidence,
                "observations": int(
                    entry.get("observations", 0)
                ),
                "positive_count": int(
                    entry.get("positive_count", 0)
                ),
                "negative_count": int(
                    entry.get("negative_count", 0)
                ),
                "updated_at": float(
                    entry.get("updated_at", 0.0)
                ),
            })
        social_events.sort(
            key=lambda row: (
                abs(float(row["signal"])),
                int(row["observations"]),
            ),
            reverse=True,
        )

        recent_episodes = []
        for episode in reversed(self._episodes):
            if user_id not in episode.user_ids:
                continue
            recent_episodes.append({
                "time": float(episode.time),
                "channel_id": episode.channel_id,
                "channel_name": str(episode.channel_name),
                "context": str(episode.context),
                "action": str(episode.action),
                "predicted_reward": float(
                    episode.predicted_reward
                ),
                "actual_reward": float(episode.actual_reward),
                "prediction_error": float(
                    episode.prediction_error
                ),
                "source": str(episode.source),
            })
            if len(recent_episodes) >= 6:
                break

        history = self._person_history_rows(user_id, 64)
        history_summary = self._person_history_summary(user_id)
        if history and not display_name:
            display_name = next(
                (
                    str(row.get("user_name") or "")
                    for row in history
                    if row.get("user_name")
                ),
                "",
            )

        signed_history = [
            row
            for row in history
            if (
                str(row.get("kind")) in {"episode", "social"}
                and abs(float(row.get("amount", 0.0))) > 1e-9
            )
        ]
        recent_signed = signed_history[:12]
        recent_valence = (
            sum(float(row["amount"]) for row in recent_signed)
            / len(recent_signed)
            if recent_signed
            else 0.0
        )
        newer = recent_signed[:6]
        older = recent_signed[6:12]
        newer_mean = (
            sum(float(row["amount"]) for row in newer) / len(newer)
            if newer
            else 0.0
        )
        older_mean = (
            sum(float(row["amount"]) for row in older) / len(older)
            if older
            else newer_mean
        )
        trend_delta = newer_mean - older_mean
        if len(recent_signed) < 4 or abs(trend_delta) < 0.06:
            relationship_trend = "stable"
        elif trend_delta > 0.0:
            relationship_trend = "improving"
        else:
            relationship_trend = "worsening"

        if len(recent_signed) >= 2:
            mean_signed = sum(
                float(row["amount"]) for row in recent_signed
            ) / len(recent_signed)
            variance = sum(
                (float(row["amount"]) - mean_signed) ** 2
                for row in recent_signed
            ) / len(recent_signed)
            relationship_stability = max(
                0.0,
                min(1.0, 1.0 - math.sqrt(variance)),
            )
        else:
            relationship_stability = 0.0

        first_seen_history = float(
            history_summary.get("first_seen", 0.0)
        )
        last_seen_history = float(
            history_summary.get("last_seen", 0.0)
        )
        if first_seen_history > 0.0:
            first_seen = first_seen_history
        else:
            first_seen = last_seen
        last_seen = max(last_seen, last_seen_history)
        relationship_age_days = (
            max(0.0, (last_seen - first_seen) / 86400.0)
            if first_seen > 0.0 and last_seen >= first_seen
            else 0.0
        )

        action_history_groups: dict[str, list[float]] = {}
        channel_history_counts: dict[tuple[int | None, str], int] = {}
        for row in history:
            action_name = str(row.get("action") or "")
            if (
                row.get("kind") == "episode"
                and action_name
            ):
                action_history_groups.setdefault(
                    action_name,
                    [],
                ).append(float(row.get("amount", 0.0)))
            channel_key = (
                row.get("channel_id"),
                str(row.get("channel_name") or ""),
            )
            if channel_key[0] is not None or channel_key[1]:
                channel_history_counts[channel_key] = (
                    channel_history_counts.get(channel_key, 0) + 1
                )

        action_outcomes = []
        for action_name, amounts in action_history_groups.items():
            mean_reward = sum(amounts) / max(1, len(amounts))
            positive_n = sum(1 for amount in amounts if amount > 1e-9)
            negative_n = sum(1 for amount in amounts if amount < -1e-9)
            action_outcomes.append({
                "action": action_name,
                "observations": len(amounts),
                "mean_reward": max(-1.0, min(1.0, mean_reward)),
                "positive_count": positive_n,
                "negative_count": negative_n,
            })
        action_outcomes.sort(
            key=lambda row: (
                int(row["observations"]),
                abs(float(row["mean_reward"])),
            ),
            reverse=True,
        )
        dominant_history_channel = None
        if channel_history_counts:
            (channel_id_key, channel_name_key), count = max(
                channel_history_counts.items(),
                key=lambda item: item[1],
            )
            dominant_history_channel = {
                "channel_id": channel_id_key,
                "channel_name": channel_name_key,
                "observations": int(count),
            }

        profile = {
            "user_id": user_id,
            "display_name": display_name,
            "observations": observations,
            "voice_observations": voice_observations,
            "contact_observations": contact_observations,
            "social_observations": social_observations,
            "familiarity": max(0.0, min(1.0, familiarity)),
            "confidence": profile_confidence,
            "expected_reward": valence,
            "valence": valence,
            "valence_label": valence_label,
            "positive_count": positive,
            "negative_count": negative,
            "neutral_count": max(
                0,
                observations - positive - negative,
            ),
            "updated_at": max(
                float(direct["updated_at"]),
                last_seen_history,
            ),
            "first_seen": first_seen,
            "last_seen": last_seen,
            "relationship_age_days": relationship_age_days,
            "history_observations": int(
                history_summary.get("observations", len(history))
            ),
            "recent_valence": max(
                -1.0,
                min(1.0, recent_valence),
            ),
            "relationship_trend": relationship_trend,
            "relationship_trend_delta": float(trend_delta),
            "relationship_stability": relationship_stability,
            "dominant_history_channel": dominant_history_channel,
            "action_outcomes": action_outcomes[:8],
            "interaction_history": history[:16],
            "preferred_action": (
                dict(preferred)
                if preferred is not None
                else None
            ),
            "avoided_action": (
                dict(avoided)
                if avoided is not None
                else None
            ),
            "social_expected_reward": float(
                social["expected_reward"]
            ),
            "contact_sources": contact_sources[:8],
            "social_events": social_events[:10],
            "recent_episodes": recent_episodes,
            "actions": action_rows[:10],
            "channels": channel_rows[:8],
            "contexts": context_rows[:8],
        }
        self._person_profile_cache[user_id] = dict(profile)
        return dict(profile)

    def person_profiles(
        self,
        limit: int = 30,
    ) -> list[dict]:
        limit = max(1, min(100, int(limit)))
        user_ids = sorted({
            int(concept_key)
            for concept_type, concept_key, _ in self._semantic.keys()
            if (
                concept_type
                in {"user", "person_contact", "person_social"}
                and str(concept_key).isdigit()
            )
        })
        rows = [
            self.person_profile(user_id)
            for user_id in user_ids
        ]
        rows.sort(
            key=lambda row: (
                int(row.get("observations", 0)),
                float(row.get("confidence", 0.0)),
                float(row.get("updated_at", 0.0)),
            ),
            reverse=True,
        )
        return rows[:limit]

    def semantic_summary(
        self,
        limit: int = 20,
    ) -> list[dict]:
        limit = max(1, min(200, int(limit)))
        rows: list[dict] = []
        for (
            concept_type,
            concept_key,
            action,
        ), entry in self._semantic.items():
            confidence = self._semantic_confidence(entry)
            expected = float(
                entry.get("expected_reward", 0.0)
            )
            rows.append({
                "concept_type": concept_type,
                "concept_key": concept_key,
                "action": action,
                "expected_reward": expected,
                "confidence": confidence,
                "strength": confidence * abs(expected),
                "observations": int(
                    entry.get("observations", 0)
                ),
                "positive_count": int(
                    entry.get("positive_count", 0)
                ),
                "negative_count": int(
                    entry.get("negative_count", 0)
                ),
                "last_reward": float(
                    entry.get("last_reward", 0.0)
                ),
                "updated_at": float(
                    entry.get("updated_at", 0.0)
                ),
            })
        rows.sort(
            key=lambda row: (
                float(row["strength"]),
                int(row["observations"]),
                float(row["updated_at"]),
            ),
            reverse=True,
        )
        return rows[:limit]

    def _create_schema(self) -> None:
        assert self.db is not None
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS voice_episode_predictions(
                context TEXT NOT NULL,
                scene_key TEXT NOT NULL DEFAULT '',
                action TEXT NOT NULL,
                expected_reward REAL NOT NULL DEFAULT 0,
                observations INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL DEFAULT 0,
                PRIMARY KEY(context, scene_key, action)
            );

            CREATE TABLE IF NOT EXISTS voice_episodes(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                guild_id INTEGER NOT NULL,
                channel_id INTEGER,
                channel_name TEXT NOT NULL DEFAULT '',
                user_ids_json TEXT NOT NULL DEFAULT '[]',
                user_names_json TEXT NOT NULL DEFAULT '[]',
                context TEXT NOT NULL,
                scene_key TEXT NOT NULL DEFAULT '',
                action TEXT NOT NULL,
                predicted_reward REAL NOT NULL,
                actual_reward REAL NOT NULL,
                prediction_error REAL NOT NULL,
                source TEXT NOT NULL DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_voice_episodes_time
            ON voice_episodes(created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_voice_episodes_scene
            ON voice_episodes(scene_key, action, created_at DESC);

            CREATE TABLE IF NOT EXISTS voice_memory_consolidation(
                scene_key TEXT NOT NULL,
                action TEXT NOT NULL,
                strength REAL NOT NULL DEFAULT 0,
                event_count INTEGER NOT NULL DEFAULT 0,
                replay_count INTEGER NOT NULL DEFAULT 0,
                positive_count INTEGER NOT NULL DEFAULT 0,
                negative_count INTEGER NOT NULL DEFAULT 0,
                last_reward REAL NOT NULL DEFAULT 0,
                last_replay REAL NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL DEFAULT 0,
                PRIMARY KEY(scene_key, action)
            );

            CREATE INDEX IF NOT EXISTS idx_voice_memory_strength
            ON voice_memory_consolidation(strength DESC, updated_at DESC);

            CREATE TABLE IF NOT EXISTS voice_semantic_memory(
                concept_type TEXT NOT NULL,
                concept_key TEXT NOT NULL,
                action TEXT NOT NULL,
                expected_reward REAL NOT NULL DEFAULT 0,
                observations INTEGER NOT NULL DEFAULT 0,
                positive_count INTEGER NOT NULL DEFAULT 0,
                negative_count INTEGER NOT NULL DEFAULT 0,
                reward_abs_sum REAL NOT NULL DEFAULT 0,
                last_reward REAL NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL DEFAULT 0,
                PRIMARY KEY(concept_type, concept_key, action)
            );

            CREATE INDEX IF NOT EXISTS idx_voice_semantic_strength
            ON voice_semantic_memory(observations DESC, updated_at DESC);

            CREATE TABLE IF NOT EXISTS person_interaction_history(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                user_id INTEGER NOT NULL,
                user_name TEXT NOT NULL DEFAULT '',
                kind TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                action TEXT NOT NULL DEFAULT '',
                amount REAL NOT NULL DEFAULT 0,
                guild_id INTEGER,
                channel_id INTEGER,
                channel_name TEXT NOT NULL DEFAULT '',
                context TEXT NOT NULL DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_person_history_user_time
            ON person_interaction_history(user_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS channel_interaction_history(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                channel_id INTEGER NOT NULL,
                channel_name TEXT NOT NULL DEFAULT '',
                kind TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                action TEXT NOT NULL DEFAULT '',
                amount REAL NOT NULL DEFAULT 0,
                human_count INTEGER NOT NULL DEFAULT 0,
                user_ids_json TEXT NOT NULL DEFAULT '[]',
                conversation_mode TEXT NOT NULL DEFAULT '',
                intensity REAL NOT NULL DEFAULT 0,
                speech_ratio REAL NOT NULL DEFAULT 0,
                context TEXT NOT NULL DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_channel_history_channel_time
            ON channel_interaction_history(channel_id, created_at DESC);
            """
        )
        self.db.commit()

    def _load_persistent_state(self) -> None:
        assert self.db is not None
        for context, scene_key, action, value, count in self.db.execute(
            """
            SELECT context, scene_key, action,
                   expected_reward, observations
            FROM voice_episode_predictions
            """
        ):
            key = (
                str(context),
                str(scene_key or ""),
                str(action),
            )
            self._values[key] = float(value)
            self._counts[key] = int(count)

        for (
            scene_key,
            action,
            strength,
            event_count,
            replay_count,
            positive_count,
            negative_count,
            last_reward,
            last_replay,
            updated_at,
        ) in self.db.execute(
            """
            SELECT scene_key, action, strength, event_count,
                   replay_count, positive_count, negative_count,
                   last_reward, last_replay, updated_at
            FROM voice_memory_consolidation
            """
        ):
            self._consolidation[
                (str(scene_key or ""), str(action))
            ] = {
                "strength": max(
                    0.0,
                    min(1.0, float(strength)),
                ),
                "event_count": int(event_count),
                "replay_count": int(replay_count),
                "positive_count": int(positive_count),
                "negative_count": int(negative_count),
                "last_reward": float(last_reward),
                "last_replay": float(last_replay),
                "updated_at": float(updated_at),
            }

        for (
            concept_type,
            concept_key,
            action,
            expected_reward,
            observations,
            positive_count,
            negative_count,
            reward_abs_sum,
            last_reward,
            updated_at,
        ) in self.db.execute(
            """
            SELECT concept_type, concept_key, action,
                   expected_reward, observations,
                   positive_count, negative_count,
                   reward_abs_sum, last_reward, updated_at
            FROM voice_semantic_memory
            """
        ):
            self._semantic[
                (
                    str(concept_type),
                    str(concept_key),
                    str(action),
                )
            ] = {
                "expected_reward": float(expected_reward),
                "observations": int(observations),
                "positive_count": int(positive_count),
                "negative_count": int(negative_count),
                "reward_abs_sum": float(reward_abs_sum),
                "last_reward": float(last_reward),
                "updated_at": float(updated_at),
            }

        rows = self.db.execute(
            """
            SELECT created_at, guild_id, channel_id, channel_name,
                   user_ids_json, user_names_json, context,
                   scene_key, action, predicted_reward,
                   actual_reward, prediction_error, source
            FROM voice_episodes
            ORDER BY id DESC
            LIMIT ?
            """,
            (self.max_events,),
        ).fetchall()
        for row in reversed(rows):
            self._episodes.append(
                self._episode_from_row(row)
            )

        self._backfill_person_history_from_db()
        self._backfill_channel_history_from_db()

        if (
            self.semantic_memory_enabled
            and not self._semantic
        ):
            self._bootstrap_semantics_from_db()
        if self.semantic_memory_enabled:
            self._backfill_person_semantics_from_db()
            self._backfill_channel_semantics_from_db()
            self._backfill_social_scene_semantics_from_db()

    @staticmethod
    def _episode_from_row(row) -> VoiceEpisode:
        (
            created_at,
            guild_id,
            channel_id,
            channel_name,
            user_ids_json,
            user_names_json,
            context,
            scene_key,
            action,
            predicted_reward,
            actual_reward,
            prediction_error,
            source,
        ) = row
        try:
            user_ids = tuple(
                int(x)
                for x in json.loads(user_ids_json or "[]")
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            user_ids = ()
        try:
            user_names = tuple(
                str(x)
                for x in json.loads(user_names_json or "[]")
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            user_names = ()
        return VoiceEpisode(
            time=float(created_at),
            guild_id=int(guild_id),
            channel_id=(
                int(channel_id)
                if channel_id is not None
                else None
            ),
            channel_name=str(channel_name or ""),
            user_ids=user_ids,
            user_names=user_names,
            context=str(context),
            scene_key=str(scene_key or ""),
            action=str(action),
            predicted_reward=float(predicted_reward),
            actual_reward=float(actual_reward),
            prediction_error=float(prediction_error),
            source=str(source or ""),
        )

    def _predict_exact(
        self,
        context: str,
        action: str,
        scene_key: str = "",
    ) -> tuple[float, int]:
        key = (
            str(context),
            str(scene_key or ""),
            str(action),
        )
        return (
            float(self._values.get(key, 0.0)),
            int(self._counts.get(key, 0)),
        )

    def predict(
        self,
        context: str,
        action: str,
        *,
        scene_key: str = "",
    ) -> float:
        generic, generic_n = self._predict_exact(
            context,
            action,
            "",
        )
        if not scene_key:
            return generic
        scene, scene_n = self._predict_exact(
            context,
            action,
            scene_key,
        )
        if scene_n <= 0:
            return generic
        # Exact social scene dominates once repeated, while the generic
        # context remains a prior for new or rare combinations.
        scene_weight = min(
            0.80,
            0.45 + 0.10 * min(3, scene_n),
        )
        if generic_n <= 0:
            scene_weight = 1.0
        return (
            scene_weight * scene
            + (1.0 - scene_weight) * generic
        )

    def _update_prediction(
        self,
        *,
        context: str,
        scene_key: str,
        action: str,
        actual: float,
    ) -> tuple[float, int]:
        key = (context, scene_key, action)
        before = float(self._values.get(key, 0.0))
        updated = before + self.learning_rate * (
            actual - before
        )
        updated = max(-1.0, min(1.0, updated))
        count = int(self._counts.get(key, 0)) + 1
        self._values[key] = updated
        self._counts[key] = count

        if self.db is not None:
            self.db.execute(
                """
                INSERT INTO voice_episode_predictions(
                    context, scene_key, action,
                    expected_reward, observations, updated_at
                ) VALUES(?,?,?,?,?,?)
                ON CONFLICT(context, scene_key, action)
                DO UPDATE SET
                    expected_reward=excluded.expected_reward,
                    observations=excluded.observations,
                    updated_at=excluded.updated_at
                """,
                (
                    context,
                    scene_key,
                    action,
                    updated,
                    count,
                    time.time(),
                ),
            )
        return updated, count

    def _memory_entry(
        self,
        scene_key: str,
        action: str,
    ) -> dict:
        key = (str(scene_key or ""), str(action))
        current = self._consolidation.get(key)
        if current is None:
            current = {
                "strength": 0.0,
                "event_count": 0,
                "replay_count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "last_reward": 0.0,
                "last_replay": 0.0,
                "updated_at": 0.0,
            }
            self._consolidation[key] = current
        return current

    def _persist_memory_entry(
        self,
        scene_key: str,
        action: str,
        entry: dict,
    ) -> None:
        if self.db is None:
            return
        self.db.execute(
            """
            INSERT INTO voice_memory_consolidation(
                scene_key, action, strength, event_count,
                replay_count, positive_count, negative_count,
                last_reward, last_replay, updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(scene_key, action)
            DO UPDATE SET
                strength=excluded.strength,
                event_count=excluded.event_count,
                replay_count=excluded.replay_count,
                positive_count=excluded.positive_count,
                negative_count=excluded.negative_count,
                last_reward=excluded.last_reward,
                last_replay=excluded.last_replay,
                updated_at=excluded.updated_at
            """,
            (
                str(scene_key or ""),
                str(action),
                float(entry["strength"]),
                int(entry["event_count"]),
                int(entry["replay_count"]),
                int(entry["positive_count"]),
                int(entry["negative_count"]),
                float(entry["last_reward"]),
                float(entry["last_replay"]),
                float(entry["updated_at"]),
            ),
        )

    def _touch_memory_event(
        self,
        *,
        scene_key: str,
        action: str,
        actual_reward: float,
        prediction_error: float,
        now: float,
    ) -> dict:
        entry = self._memory_entry(scene_key, action)
        strength = max(
            0.0,
            min(1.0, float(entry["strength"])),
        )
        actual = float(actual_reward)
        error = float(prediction_error)
        previous = float(entry["last_reward"])
        evidence = min(
            1.0,
            0.55 * abs(actual) + 0.45 * abs(error),
        )
        consistent = (
            abs(previous) <= 1e-9
            or abs(actual) <= 1e-9
            or previous * actual >= 0.0
        )
        if consistent:
            strength += (
                self.consolidation_gain
                * (0.15 + 0.85 * evidence)
                * (1.0 - strength)
            )
        else:
            strength *= (
                1.0
                - min(
                    0.80,
                    self.consolidation_gain
                    * (0.70 + 0.80 * evidence),
                )
            )
        entry["strength"] = max(0.0, min(1.0, strength))
        entry["event_count"] = int(entry["event_count"]) + 1
        if actual > 0.0:
            entry["positive_count"] = (
                int(entry["positive_count"]) + 1
            )
        elif actual < 0.0:
            entry["negative_count"] = (
                int(entry["negative_count"]) + 1
            )
        entry["last_reward"] = actual
        entry["updated_at"] = float(now)
        self._persist_memory_entry(scene_key, action, entry)
        return dict(entry)

    def consolidate_replay(
        self,
        episode: dict,
        replay_reward: float,
        *,
        now: float | None = None,
    ) -> dict:
        now_value = float(time.time() if now is None else now)
        scene_key = str(episode.get("scene_key") or "")
        action = str(episode.get("action") or "stay")
        entry = self._memory_entry(scene_key, action)
        strength = max(
            0.0,
            min(1.0, float(entry["strength"])),
        )
        original = float(episode.get("actual_reward", 0.0))
        replay_value = float(replay_reward)
        consistent = (
            abs(original) <= 1e-9
            or abs(replay_value) <= 1e-9
            or original * replay_value >= 0.0
        )
        evidence = min(
            1.0,
            0.50 * abs(original)
            + 0.50 * abs(replay_value) * 4.0,
        )
        if consistent:
            strength += (
                self.consolidation_gain
                * (0.35 + 0.65 * evidence)
                * (1.0 - strength)
            )
        else:
            strength *= (
                1.0
                - min(
                    0.90,
                    self.consolidation_gain
                    * (0.80 + evidence),
                )
            )
        entry["strength"] = max(0.0, min(1.0, strength))
        entry["replay_count"] = int(entry["replay_count"]) + 1
        entry["last_replay"] = now_value
        entry["updated_at"] = now_value
        self._persist_memory_entry(scene_key, action, entry)
        if self.db is not None:
            self.db.commit()
        result = dict(entry)
        result["scene_key"] = scene_key
        result["action"] = action
        result["status"] = (
            "consolidated"
            if float(entry["strength"]) >= self.consolidated_threshold
            else "forming"
        )
        return result

    def rehearse_semantic_replay(
        self,
        episode: dict,
        *,
        now: float | None = None,
    ) -> list[dict]:
        """Rehearse existing semantic memories without inventing observations.

        Sleep/replay may stabilize the expected value learned from real
        episodes, but it must not increase the observation count as if a new
        Discord event had happened.
        """
        if not self.semantic_memory_enabled:
            return []

        now_value = float(time.time() if now is None else now)
        actual = max(
            -1.0,
            min(1.0, float(episode.get("actual_reward", 0.0))),
        )
        error = max(
            -1.0,
            min(1.0, float(episode.get("prediction_error", 0.0))),
        )
        target = max(
            -1.0,
            min(1.0, 0.65 * actual + 0.35 * error),
        )
        evidence = min(
            1.0,
            0.60 * abs(actual) + 0.40 * abs(error),
        )
        alpha = max(
            0.0,
            min(
                0.20,
                self.consolidation_gain
                * (0.15 + 0.45 * evidence),
            ),
        )
        if alpha <= 1e-9:
            return []

        updates: list[dict] = []
        action = str(episode.get("action") or "stay")
        for concept_type, concept_key in self.semantic_concepts(
            str(episode.get("context") or ""),
            episode.get("channel_id"),
            episode.get("user_ids") or (),
        ):
            key = (concept_type, concept_key, action)
            entry = self._semantic.get(key)
            if entry is None:
                continue
            before = float(entry.get("expected_reward", 0.0))
            after = max(
                -1.0,
                min(1.0, before + alpha * (target - before)),
            )
            entry["expected_reward"] = after
            entry["updated_at"] = now_value
            self._persist_semantic_entry(
                concept_type,
                concept_key,
                action,
                entry,
            )
            updates.append({
                "concept_type": concept_type,
                "concept_key": concept_key,
                "action": action,
                "before": before,
                "after": after,
                "delta": after - before,
                "observations": int(
                    entry.get("observations", 0)
                ),
                "confidence": self._semantic_confidence(entry),
            })

        if updates:
            self._person_profile_cache.clear()
            self._channel_profile_cache.clear()
            self._social_scene_profile_cache.clear()
            self._voice_dynamics_profile_cache.clear()
        if self.db is not None and updates:
            self.db.commit()
        return updates

    def apply_forgetting(
        self,
        *,
        now: float | None = None,
        force: bool = False,
    ) -> dict:
        now_value = float(time.time() if now is None else now)
        elapsed = max(
            0.0,
            now_value - float(self._last_forgetting_at),
        )
        if (
            not force
            and elapsed < float(self.forgetting_interval_seconds)
        ):
            diag = dict(self._last_forgetting_diag)
            diag.update({
                "ran": False,
                "elapsed_seconds": elapsed,
                "next_in_seconds": max(
                    0.0,
                    float(self.forgetting_interval_seconds) - elapsed,
                ),
            })
            return diag
        if elapsed <= 0.0:
            return dict(self._last_forgetting_diag)

        half_life = max(
            1.0,
            self.forgetting_half_life_days * 86400.0,
        )
        factor = 0.5 ** (elapsed / half_life)
        for key in list(self._values):
            self._values[key] = float(self._values[key]) * factor
        for entry in self._consolidation.values():
            entry["strength"] = max(
                0.0,
                min(
                    1.0,
                    float(entry["strength"]) * factor,
                ),
            )
        for entry in self._semantic.values():
            entry["expected_reward"] = (
                float(entry["expected_reward"]) * factor
            )
            entry["reward_abs_sum"] = (
                float(entry["reward_abs_sum"]) * factor
            )

        self._person_profile_cache.clear()
        self._channel_profile_cache.clear()
        self._social_scene_profile_cache.clear()
        self._voice_dynamics_profile_cache.clear()

        if self.db is not None:
            self.db.execute(
                """
                UPDATE voice_episode_predictions
                SET expected_reward = expected_reward * ?
                """,
                (factor,),
            )
            self.db.execute(
                """
                UPDATE voice_memory_consolidation
                SET strength = strength * ?
                """,
                (factor,),
            )
            self.db.execute(
                """
                UPDATE voice_semantic_memory
                SET expected_reward = expected_reward * ?,
                    reward_abs_sum = reward_abs_sum * ?
                """,
                (factor, factor),
            )
            self.db.commit()

        self._last_forgetting_at = now_value
        self._last_forgetting_diag = {
            "ran": True,
            "time": now_value,
            "elapsed_seconds": elapsed,
            "factor": float(factor),
            "predictions": int(len(self._values)),
            "memory_scenes": int(len(self._consolidation)),
            "semantic_entries": int(len(self._semantic)),
        }
        return dict(self._last_forgetting_diag)

    def consolidation_summary(
        self,
        limit: int = 12,
    ) -> list[dict]:
        limit = max(1, min(100, int(limit)))
        recent_meta: dict[tuple[str, str], VoiceEpisode] = {}
        for episode in reversed(self._episodes):
            key = (episode.scene_key, episode.action)
            if key not in recent_meta:
                recent_meta[key] = episode

        rows = []
        for (scene_key, action), entry in self._consolidation.items():
            episode = recent_meta.get((scene_key, action))
            rows.append({
                "scene_key": scene_key,
                "action": action,
                "strength": float(entry["strength"]),
                "event_count": int(entry["event_count"]),
                "replay_count": int(entry["replay_count"]),
                "positive_count": int(entry["positive_count"]),
                "negative_count": int(entry["negative_count"]),
                "last_reward": float(entry["last_reward"]),
                "last_replay": float(entry["last_replay"]),
                "updated_at": float(entry["updated_at"]),
                "channel_id": (
                    episode.channel_id
                    if episode is not None
                    else None
                ),
                "channel_name": (
                    episode.channel_name
                    if episode is not None
                    else ""
                ),
                "user_ids": (
                    list(episode.user_ids)
                    if episode is not None
                    else []
                ),
                "user_names": (
                    list(episode.user_names)
                    if episode is not None
                    else []
                ),
                "last_episode_time": (
                    float(episode.time)
                    if episode is not None
                    else 0.0
                ),
                "status": (
                    "consolidated"
                    if float(entry["strength"])
                    >= self.consolidated_threshold
                    else "forming"
                ),
            })
        rows.sort(
            key=lambda row: (
                float(row["strength"]),
                int(row["replay_count"]),
                int(row["event_count"]),
                float(row["updated_at"]),
            ),
            reverse=True,
        )
        return rows[:limit]

    def observe(
        self,
        *,
        guild_id: int,
        context: str,
        action: str,
        actual_reward: float,
        source: str = "",
        predicted_reward: float | None = None,
        now: float | None = None,
        channel_id: int | None = None,
        channel_name: str = "",
        user_ids: list[int] | tuple[int, ...] = (),
        user_names: list[str] | tuple[str, ...] = (),
        scene_key: str | None = None,
    ) -> dict:
        context = str(context)
        action = str(action)
        names_in = [str(x) for x in user_names]
        user_map: dict[int, str] = {}
        for index, raw_user_id in enumerate(user_ids):
            uid = int(raw_user_id)
            name = names_in[index] if index < len(names_in) else ""
            if uid not in user_map or (not user_map[uid] and name):
                user_map[uid] = name
        normalized_users = tuple(sorted(user_map))
        normalized_names = tuple(
            user_map[uid] for uid in normalized_users
        )
        scene_key = (
            self.make_scene_key(channel_id, normalized_users)
            if scene_key is None
            else str(scene_key)
        )
        predicted = (
            self.predict(
                context,
                action,
                scene_key=scene_key,
            )
            if predicted_reward is None
            else float(predicted_reward)
        )
        actual = max(-1.0, min(1.0, float(actual_reward)))
        error = actual - predicted

        generic_updated, generic_count = (
            self._update_prediction(
                context=context,
                scene_key="",
                action=action,
                actual=actual,
            )
        )
        scene_updated, scene_count = (
            self._update_prediction(
                context=context,
                scene_key=scene_key,
                action=action,
                actual=actual,
            )
        )

        episode = VoiceEpisode(
            time=float(time.time() if now is None else now),
            guild_id=int(guild_id),
            channel_id=(
                int(channel_id)
                if channel_id is not None
                else None
            ),
            channel_name=str(channel_name or ""),
            user_ids=normalized_users,
            user_names=normalized_names,
            context=context,
            scene_key=scene_key,
            action=action,
            predicted_reward=predicted,
            actual_reward=actual,
            prediction_error=error,
            source=str(source),
        )
        self._episodes.append(episode)
        memory_entry = self._touch_memory_event(
            scene_key=episode.scene_key,
            action=episode.action,
            actual_reward=episode.actual_reward,
            prediction_error=episode.prediction_error,
            now=episode.time,
        )
        semantic_uncertainty_before = self.semantic_uncertainty(
            episode.context,
            [episode.action],
            channel_id=episode.channel_id,
            user_ids=episode.user_ids,
        )
        semantic_updates = self._update_semantics(
            context=episode.context,
            channel_id=episode.channel_id,
            user_ids=episode.user_ids,
            action=episode.action,
            actual_reward=episode.actual_reward,
            now=episode.time,
            persist=True,
        )
        semantic_uncertainty_after = self.semantic_uncertainty(
            episode.context,
            [episode.action],
            channel_id=episode.channel_id,
            user_ids=episode.user_ids,
        )
        information_gain = max(
            0.0,
            float(
                semantic_uncertainty_before.get(
                    "uncertainty",
                    1.0,
                )
            )
            - float(
                semantic_uncertainty_after.get(
                    "uncertainty",
                    1.0,
                )
            ),
        )

        if self.db is not None:
            self.db.execute(
                """
                INSERT INTO voice_episodes(
                    created_at, guild_id, channel_id, channel_name,
                    user_ids_json, user_names_json, context,
                    scene_key, action, predicted_reward,
                    actual_reward, prediction_error, source
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    episode.time,
                    episode.guild_id,
                    episode.channel_id,
                    episode.channel_name,
                    json.dumps(episode.user_ids),
                    json.dumps(
                        episode.user_names,
                        ensure_ascii=False,
                    ),
                    episode.context,
                    episode.scene_key,
                    episode.action,
                    episode.predicted_reward,
                    episode.actual_reward,
                    episode.prediction_error,
                    episode.source,
                ),
            )
            self.db.execute(
                """
                DELETE FROM voice_episodes
                WHERE id NOT IN (
                    SELECT id FROM voice_episodes
                    ORDER BY id DESC
                    LIMIT ?
                )
                """,
                (self.max_persisted_events,),
            )
            for pos, user_id in enumerate(episode.user_ids):
                user_name = (
                    episode.user_names[pos]
                    if pos < len(episode.user_names)
                    else ""
                )
                self._record_person_history(
                    int(user_id),
                    kind="episode",
                    source=episode.source,
                    action=episode.action,
                    amount=episode.actual_reward,
                    user_name=str(user_name),
                    guild_id=episode.guild_id,
                    channel_id=episode.channel_id,
                    channel_name=episode.channel_name,
                    context=episode.context,
                    now=episode.time,
                )
            if episode.channel_id is not None:
                self._record_channel_history(
                    int(episode.channel_id),
                    channel_name=episode.channel_name,
                    kind="episode",
                    source=episode.source,
                    action=episode.action,
                    amount=episode.actual_reward,
                    human_count=len(episode.user_ids),
                    user_ids=episode.user_ids,
                    context=episode.context,
                    now=episode.time,
                )
            self.db.commit()

        return {
            "time": episode.time,
            "guild_id": episode.guild_id,
            "channel_id": episode.channel_id,
            "channel_name": episode.channel_name,
            "user_ids": list(episode.user_ids),
            "user_names": list(episode.user_names),
            "context": episode.context,
            "scene_key": episode.scene_key,
            "action": episode.action,
            "predicted_reward": episode.predicted_reward,
            "actual_reward": episode.actual_reward,
            "prediction_error": episode.prediction_error,
            "updated_prediction": self.predict(
                context,
                action,
                scene_key=scene_key,
            ),
            "generic_prediction": generic_updated,
            "scene_prediction": scene_updated,
            "observations": generic_count,
            "scene_observations": scene_count,
            "memory_strength": float(
                memory_entry["strength"]
            ),
            "memory_status": (
                "consolidated"
                if float(memory_entry["strength"])
                >= self.consolidated_threshold
                else "forming"
            ),
            "memory_replays": int(
                memory_entry["replay_count"]
            ),
            "semantic_updates": semantic_updates,
            "semantic_uncertainty_before": float(
                semantic_uncertainty_before.get(
                    "uncertainty",
                    1.0,
                )
            ),
            "semantic_uncertainty_after": float(
                semantic_uncertainty_after.get(
                    "uncertainty",
                    1.0,
                )
            ),
            "information_gain": float(information_gain),
            "source": episode.source,
        }

    def predictions(
        self,
        context: str,
        actions: list[str] | tuple[str, ...],
        *,
        scene_key: str = "",
    ) -> dict[str, float]:
        return {
            str(action): self.predict(
                context,
                str(action),
                scene_key=scene_key,
            )
            for action in actions
        }

    def prediction_details(
        self,
        context: str,
        action: str,
        *,
        scene_key: str = "",
    ) -> dict:
        """Return the learned reward prediction plus evidence strength."""
        generic, generic_n = self._predict_exact(
            context,
            action,
            "",
        )
        scene, scene_n = (
            self._predict_exact(
                context,
                action,
                scene_key,
            )
            if scene_key
            else (0.0, 0)
        )
        observations = int(generic_n + scene_n)
        confidence = (
            1.0 - math.exp(-float(observations) / 4.0)
            if observations > 0
            else 0.0
        )
        return {
            "expected_reward": float(
                self.predict(
                    context,
                    action,
                    scene_key=scene_key,
                )
            ),
            "observations": observations,
            "confidence": float(confidence),
            "generic_reward": float(generic),
            "generic_observations": int(generic_n),
            "scene_reward": float(scene),
            "scene_observations": int(scene_n),
            "scene_key": str(scene_key or ""),
        }

    def predictions_detailed(
        self,
        context: str,
        actions: list[str] | tuple[str, ...],
        *,
        scene_key: str = "",
    ) -> dict[str, dict]:
        return {
            str(action): self.prediction_details(
                context,
                str(action),
                scene_key=scene_key,
            )
            for action in actions
        }

    @staticmethod
    def _episode_dict(row: VoiceEpisode) -> dict:
        return {
            "time": row.time,
            "guild_id": row.guild_id,
            "channel_id": row.channel_id,
            "channel_name": row.channel_name,
            "user_ids": list(row.user_ids),
            "user_names": list(row.user_names),
            "context": row.context,
            "scene_key": row.scene_key,
            "action": row.action,
            "predicted_reward": row.predicted_reward,
            "actual_reward": row.actual_reward,
            "prediction_error": row.prediction_error,
            "source": row.source,
        }

    def recent(self, limit: int = 20) -> list[dict]:
        limit = max(1, min(self.max_events, int(limit)))
        return [
            self._episode_dict(row)
            for row in list(self._episodes)[-limit:]
        ]

    def replay_candidates(
        self,
        *,
        limit: int = 12,
        max_age_seconds: float = 14 * 86400.0,
        now: float | None = None,
    ) -> list[dict]:
        """Return significant recent episodes suitable for offline replay.

        Ranking favors surprise (prediction error), reward magnitude and
        recency. Selection/replay policy stays in the Discord runtime so this
        class remains deterministic and testable.
        """
        now_value = float(time.time() if now is None else now)
        max_age = max(1.0, float(max_age_seconds))
        limit = max(1, min(256, int(limit)))

        if self.db is not None:
            rows = self.db.execute(
                """
                SELECT created_at, guild_id, channel_id, channel_name,
                       user_ids_json, user_names_json, context,
                       scene_key, action, predicted_reward,
                       actual_reward, prediction_error, source
                FROM voice_episodes
                WHERE created_at >= ?
                ORDER BY id DESC
                LIMIT 512
                """,
                (now_value - max_age,),
            ).fetchall()
            episodes = [
                self._episode_from_row(row)
                for row in rows
            ]
        else:
            episodes = [
                row
                for row in reversed(self._episodes)
                if now_value - float(row.time) <= max_age
            ]

        ranked: list[tuple[float, VoiceEpisode, dict]] = []
        for episode in episodes:
            age = max(0.0, now_value - float(episode.time))
            recency = max(0.0, 1.0 - age / max_age)
            surprise = min(1.0, abs(float(episode.prediction_error)))
            reward = min(1.0, abs(float(episode.actual_reward)))
            memory = dict(
                self._memory_entry(
                    episode.scene_key,
                    episode.action,
                )
            )
            strength = max(
                0.0,
                min(1.0, float(memory["strength"])),
            )
            repetitions = min(
                1.0,
                (
                    int(memory["event_count"])
                    + 2 * int(memory["replay_count"])
                )
                / 12.0,
            )
            score = (
                1.60 * surprise
                + 1.00 * reward
                + 0.35 * recency
                + 1.10 * strength
                + 0.25 * repetitions
            )
            if score <= 0.01:
                continue
            ranked.append((score, episode, memory))

        ranked.sort(
            key=lambda item: (item[0], item[1].time),
            reverse=True,
        )
        result: list[dict] = []
        for score, episode, memory in ranked[:limit]:
            row = self._episode_dict(episode)
            row["replay_score"] = float(score)
            row["age_seconds"] = max(
                0.0,
                now_value - float(episode.time),
            )
            row["memory_strength"] = float(
                memory["strength"]
            )
            row["memory_replays"] = int(
                memory["replay_count"]
            )
            row["memory_events"] = int(
                memory["event_count"]
            )
            row["memory_status"] = (
                "consolidated"
                if float(memory["strength"])
                >= self.consolidated_threshold
                else "forming"
            )
            result.append(row)
        return result

    def size(self) -> int:
        if self.db is None:
            return len(self._episodes)
        row = self.db.execute(
            "SELECT COUNT(*) FROM voice_episodes"
        ).fetchone()
        return int(row[0]) if row else 0

    def prediction_count(self) -> int:
        return len(self._values)

    def diagnostics(self) -> dict:
        consolidated = sum(
            1
            for entry in self._consolidation.values()
            if float(entry["strength"])
            >= self.consolidated_threshold
        )
        person_history_events = 0
        if self.db is not None:
            person_history_events = int(
                self.db.execute(
                    "SELECT COUNT(*) FROM person_interaction_history"
                ).fetchone()[0]
            )
        return {
            "persistent": self.db is not None,
            "database": str(self.path) if self.path else None,
            "episodes": self.size(),
            "cached_episodes": len(self._episodes),
            "predictions": self.prediction_count(),
            "memory_scenes": len(self._consolidation),
            "consolidated_scenes": int(consolidated),
            "consolidated_threshold": float(
                self.consolidated_threshold
            ),
            "semantic_enabled": bool(
                self.semantic_memory_enabled
            ),
            "semantic_entries": int(len(self._semantic)),
            "person_history_events": person_history_events,
            "semantic_recall_min_observations": int(
                self.semantic_recall_min_observations
            ),
            "top_semantics": self.semantic_summary(16),
            "forgetting": dict(self._last_forgetting_diag),
            "top_memories": self.consolidation_summary(12),
            "recent": self.recent(12),
        }

    def close(self) -> None:
        if self.db is not None:
            self.db.commit()
            self.db.close()
            self.db = None
