from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time


@dataclass(slots=True)
class VoiceEpisode:
    time: float
    guild_id: int
    context: str
    action: str
    predicted_reward: float
    actual_reward: float
    prediction_error: float
    source: str


class VoiceEpisodicMemory:
    """Small online predictor for voice outcomes.

    It intentionally stays simple: context/action pairs keep an EWMA expected
    reward, while recent prediction errors are retained as episodes. The
    connectome still selects the action; this class only supplies an expected
    outcome and an error signal after the real result arrives.
    """

    def __init__(
        self,
        *,
        max_events: int = 256,
        learning_rate: float = 0.20,
    ):
        self.max_events = max(16, int(max_events))
        self.learning_rate = max(
            0.001,
            min(1.0, float(learning_rate)),
        )
        self._values: dict[tuple[str, str], float] = {}
        self._counts: dict[tuple[str, str], int] = {}
        self._episodes: deque[VoiceEpisode] = deque(
            maxlen=self.max_events
        )

    def predict(self, context: str, action: str) -> float:
        return float(
            self._values.get(
                (str(context), str(action)),
                0.0,
            )
        )

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
    ) -> dict:
        context = str(context)
        action = str(action)
        key = (context, action)
        predicted = (
            self.predict(context, action)
            if predicted_reward is None
            else float(predicted_reward)
        )
        actual = max(-1.0, min(1.0, float(actual_reward)))
        error = actual - predicted
        updated = predicted + self.learning_rate * error
        updated = max(-1.0, min(1.0, updated))
        self._values[key] = updated
        self._counts[key] = int(self._counts.get(key, 0)) + 1

        episode = VoiceEpisode(
            time=float(time.time() if now is None else now),
            guild_id=int(guild_id),
            context=context,
            action=action,
            predicted_reward=predicted,
            actual_reward=actual,
            prediction_error=error,
            source=str(source),
        )
        self._episodes.append(episode)
        return {
            "time": episode.time,
            "guild_id": episode.guild_id,
            "context": episode.context,
            "action": episode.action,
            "predicted_reward": episode.predicted_reward,
            "actual_reward": episode.actual_reward,
            "prediction_error": episode.prediction_error,
            "updated_prediction": updated,
            "observations": self._counts[key],
            "source": episode.source,
        }

    def predictions(
        self,
        context: str,
        actions: list[str] | tuple[str, ...],
    ) -> dict[str, float]:
        return {
            str(action): self.predict(context, str(action))
            for action in actions
        }

    def recent(self, limit: int = 20) -> list[dict]:
        limit = max(1, min(self.max_events, int(limit)))
        rows = list(self._episodes)[-limit:]
        return [
            {
                "time": row.time,
                "guild_id": row.guild_id,
                "context": row.context,
                "action": row.action,
                "predicted_reward": row.predicted_reward,
                "actual_reward": row.actual_reward,
                "prediction_error": row.prediction_error,
                "source": row.source,
            }
            for row in rows
        ]

    def size(self) -> int:
        return len(self._episodes)
