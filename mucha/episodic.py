from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
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

        ranked: list[tuple[float, VoiceEpisode]] = []
        for episode in episodes:
            age = max(0.0, now_value - float(episode.time))
            recency = max(0.0, 1.0 - age / max_age)
            surprise = min(1.0, abs(float(episode.prediction_error)))
            reward = min(1.0, abs(float(episode.actual_reward)))
            score = (
                1.60 * surprise
                + 1.00 * reward
                + 0.35 * recency
            )
            if score <= 0.01:
                continue
            ranked.append((score, episode))

        ranked.sort(
            key=lambda item: (item[0], item[1].time),
            reverse=True,
        )
        result: list[dict] = []
        for score, episode in ranked[:limit]:
            row = self._episode_dict(episode)
            row["replay_score"] = float(score)
            row["age_seconds"] = max(
                0.0,
                now_value - float(episode.time),
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
        return {
            "persistent": self.db is not None,
            "database": str(self.path) if self.path else None,
            "episodes": self.size(),
            "cached_episodes": len(self._episodes),
            "predictions": self.prediction_count(),
            "recent": self.recent(12),
        }

    def close(self) -> None:
        if self.db is not None:
            self.db.commit()
            self.db.close()
            self.db = None
