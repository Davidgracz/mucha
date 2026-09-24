from __future__ import annotations

import hashlib
import math
from typing import Iterable
import numpy as np

from .config import BrainConfig
from .connectome import Connectome


def _sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


class FlyBrain:
    """Persistent connectome-driven state.

    v0.1 uses the real sparse FlyWire topology. It is intentionally a light,
    continuously-running approximation, not a claim of reproducing a real fly mind.
    """

    ACTIONS = (
        "speak", "react", "voice_join", "voice_move", "voice_leave", "explore", "stay"
    )

    def __init__(self, connectome: Connectome, cfg: BrainConfig):
        self.c = connectome
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        n = self.c.n_neurons
        self.state = np.zeros(n, dtype=np.float32)
        self.eligibility = np.zeros(n, dtype=np.float32)
        self.plastic_bias = np.zeros(n, dtype=np.float32)
        self.reward_trace = 0.0
        self.tick_count = 0
        self._pool_cache: dict[tuple[str, int], np.ndarray] = {}
        self._load_state()

    def _load_state(self) -> None:
        p = self.cfg.state_file
        if not p.exists():
            return
        data = np.load(p, allow_pickle=False)
        if "state" in data and data["state"].shape == self.state.shape:
            self.state[:] = data["state"].astype(np.float32)
        if "eligibility" in data and data["eligibility"].shape == self.eligibility.shape:
            self.eligibility[:] = data["eligibility"].astype(np.float32)
        if "plastic_bias" in data and data["plastic_bias"].shape == self.plastic_bias.shape:
            self.plastic_bias[:] = data["plastic_bias"].astype(np.float32)
        if "reward_trace" in data:
            self.reward_trace = float(data["reward_trace"])
        if "tick_count" in data:
            self.tick_count = int(data["tick_count"])

    def save(self) -> None:
        p = self.cfg.state_file
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            state=self.state.astype(np.float16),
            eligibility=self.eligibility.astype(np.float16),
            plastic_bias=self.plastic_bias.astype(np.float16),
            reward_trace=np.float32(self.reward_trace),
            tick_count=np.int64(self.tick_count),
        )
        tmp.replace(p)

    def _stable_seed(self, key: str) -> int:
        h = hashlib.blake2b(key.encode("utf-8", "ignore"), digest_size=8, person=b"mucha-v01")
        return int.from_bytes(h.digest(), "little") ^ self.cfg.seed

    def _subset(self, key: str, source: np.ndarray, size: int = 96) -> np.ndarray:
        if len(source) == 0:
            source = np.arange(self.c.n_neurons, dtype=np.int32)
        size = min(size, len(source))
        cache_key = (key, size)
        if cache_key in self._pool_cache:
            return self._pool_cache[cache_key]
        rng = np.random.default_rng(self._stable_seed(key))
        idx = rng.choice(source, size=size, replace=False).astype(np.int32)
        self._pool_cache[cache_key] = idx
        return idx

    def inject(self, key: str, magnitude: float = 1.0, width: int = 96) -> None:
        idx = self._subset("sensory:" + key, self.c.sensory, width)
        jitter = self.rng.uniform(0.75, 1.25, size=len(idx)).astype(np.float32)
        self.state[idx] += np.float32(magnitude) * jitter
        np.clip(self.state, -3.0, 3.0, out=self.state)

    def inject_text(self, text: str, author_id: int, mentioned: bool) -> None:
        stripped = text.strip()
        if not stripped:
            return
        mag = min(1.8, 0.25 + len(stripped) / 180.0)
        self.inject("text:any", mag, 128)
        self.inject(f"user:{author_id}", 0.45 + 0.2 * mentioned, 64)
        if mentioned:
            self.inject("text:mention-self", 1.25, 128)
        if "?" in stripped:
            self.inject("text:question", 0.35, 48)
        if "!" in stripped:
            self.inject("text:exclamation", 0.30, 48)
        if stripped.isupper() and len(stripped) > 3:
            self.inject("text:loud", 0.45, 64)

    def inject_voice_snapshot(self, guild_id: int, channel_id: int, user_ids: Iterable[int]) -> None:
        users = list(user_ids)
        self.inject(f"voice:guild:{guild_id}:channel:{channel_id}", 0.18 + 0.12 * min(len(users), 8), 64)
        for uid in users[:12]:
            self.inject(f"voice:user:{uid}", 0.08, 24)

    def step(self, ticks: int = 1) -> None:
        for _ in range(max(1, ticks)):
            propagated = self.c.matrix.dot(self.state).astype(np.float32, copy=False)
            noise = self.rng.normal(0.0, self.cfg.noise, size=self.state.shape).astype(np.float32)
            x = (
                self.cfg.leak * self.state
                + self.cfg.propagation_gain * propagated
                + self.plastic_bias
                + noise
            )
            self.state[:] = np.tanh(x)
            self.eligibility[:] = 0.94 * self.eligibility + 0.06 * np.abs(self.state)
            self.plastic_bias *= np.float32(self.cfg.plasticity_decay)
            self.reward_trace *= 0.96
            self.tick_count += 1

    def reward(self, amount: float) -> None:
        amount = float(max(-1.0, min(1.0, amount)))
        self.reward_trace = max(-2.0, min(2.0, self.reward_trace + amount))
        delta = self.cfg.plasticity_lr * amount * self.eligibility
        self.plastic_bias += delta.astype(np.float32)
        np.clip(self.plastic_bias, -self.cfg.max_bias, self.cfg.max_bias, out=self.plastic_bias)
        if len(self.c.modulatory):
            idx = self._subset("reward:modulatory", self.c.modulatory, min(128, len(self.c.modulatory)))
            self.state[idx] += np.float32(0.35 * amount)

    def readout(self, key: str, width: int = 96) -> float:
        idx = self._subset("output:" + key, self.c.output, width)
        raw = float(np.mean(self.state[idx]))
        return _sigmoid(3.2 * raw + 0.35 * self.reward_trace)

    def action_scores(self) -> dict[str, float]:
        return {name: self.readout("action:" + name, 128) for name in self.ACTIONS}

    def channel_affinity(self, guild_id: int, channel_id: int) -> float:
        return self.readout(f"voice-affinity:{guild_id}:{channel_id}", 96)

    def top_active_neurons(self, count: int = 8) -> list[tuple[int, float]]:
        """Return root_id and signed activation for the strongest neurons."""
        count = max(1, min(int(count), self.c.n_neurons))
        abs_state = np.abs(self.state)
        if count >= self.c.n_neurons:
            idx = np.argsort(abs_state)[::-1]
        else:
            idx = np.argpartition(abs_state, -count)[-count:]
            idx = idx[np.argsort(abs_state[idx])[::-1]]
        return [(int(self.c.root_ids[i]), float(self.state[i])) for i in idx[:count]]

    def diagnostics(self) -> dict[str, float | int]:
        return {
            "neurons": self.c.n_neurons,
            "connections": int(self.c.matrix.nnz),
            "active_abs_gt_0_1": int(np.count_nonzero(np.abs(self.state) > 0.1)),
            "mean_abs": float(np.mean(np.abs(self.state))),
            "max_abs": float(np.max(np.abs(self.state))),
            "reward_trace": float(self.reward_trace),
            "ticks": int(self.tick_count),
        }
