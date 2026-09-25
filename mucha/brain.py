from __future__ import annotations

import hashlib
import math
import re
import time
from typing import Iterable

import numpy as np

from .compute import ComputeBackend
from .config import BrainConfig
from .connectome import Connectome


def _sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


class FlyBrain:
    """Persistent connectome-driven state with optional CUDA acceleration."""

    ACTIONS = (
        "speak", "react", "voice_join", "voice_move", "voice_leave", "explore", "stay"
    )

    def __init__(self, connectome: Connectome, cfg: BrainConfig):
        self.c = connectome
        self.cfg = cfg
        self.compute = ComputeBackend(cfg.backend, cfg.gpu_device)
        self.xp = self.compute.xp
        self.matrix = self.compute.sparse_from_scipy(self.c.matrix)

        # CPU RNG is only used for stable index selection. Dynamic noise lives
        # on the active backend and therefore stays on the GPU in CUDA mode.
        self.rng = np.random.default_rng(cfg.seed)
        if self.compute.is_gpu:
            self.compute.cp.random.seed(cfg.seed)

        n = self.c.n_neurons
        self.state = self.compute.zeros(n, dtype=np.float32)
        self.eligibility = self.compute.zeros(n, dtype=np.float32)
        self.plastic_bias = self.compute.zeros(n, dtype=np.float32)
        self.reward_trace = 0.0
        self.tick_count = 0
        self._pool_cache: dict[tuple[str, int], np.ndarray] = {}
        self.last_learning: dict = {
            "amount": 0.0,
            "action": None,
            "changed_neurons": 0,
            "mean_delta": 0.0,
            "max_delta": 0.0,
            "before": {},
            "after": {},
            "impact": {},
            "top_changed": [],
        }
        self._load_state()
        self._learning_session_started_at = time.time()
        self._startup_plastic_bias = (
            self.compute.to_cpu(self.plastic_bias)
            .astype(np.float32, copy=True)
        )
        self._session_reward_events = 0
        self._session_positive_reward_events = 0
        self._session_negative_reward_events = 0
        self._session_positive_reward_total = 0.0
        self._session_negative_reward_total = 0.0
        self._session_bias_update_operations = 0
        self._session_reward_bias_delta = np.zeros(
            self.c.n_neurons,
            dtype=np.float32,
        )

    @property
    def backend_name(self) -> str:
        return self.compute.info.active

    @property
    def device_name(self) -> str:
        return self.compute.info.device_name

    def _load_state(self) -> None:
        p = self.cfg.state_file
        if not p.exists():
            return
        data = np.load(p, allow_pickle=False)
        if "state" in data and data["state"].shape == self.state.shape:
            self.state[...] = self.compute.asarray(data["state"], dtype=np.float32)
        if "eligibility" in data and data["eligibility"].shape == self.eligibility.shape:
            self.eligibility[...] = self.compute.asarray(data["eligibility"], dtype=np.float32)
        if "plastic_bias" in data and data["plastic_bias"].shape == self.plastic_bias.shape:
            self.plastic_bias[...] = self.compute.asarray(data["plastic_bias"], dtype=np.float32)
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
            state=self.compute.to_cpu(self.state).astype(np.float16),
            eligibility=self.compute.to_cpu(self.eligibility).astype(np.float16),
            plastic_bias=self.compute.to_cpu(self.plastic_bias).astype(np.float16),
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

    def _backend_indices(self, idx: np.ndarray):
        if self.compute.is_gpu:
            return self.compute.asarray(idx, dtype=np.int32)
        return idx

    def inject(self, key: str, magnitude: float = 1.0, width: int = 96) -> None:
        idx_cpu = self._subset("sensory:" + key, self.c.sensory, width)
        idx = self._backend_indices(idx_cpu)
        jitter = self.compute.random_uniform(0.75, 1.25, size=len(idx_cpu))
        self.state[idx] += np.float32(magnitude) * jitter
        self.xp.clip(self.state, -3.0, 3.0, out=self.state)

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

        # Give learned words stable sensory populations. The language model can
        # later read the current connectome state for the same words, so word
        # choice is influenced by what the fly brain is currently processing.
        seen: set[str] = set()
        words = re.findall(
            r"[^\W_]{3,}",
            stripped.lower(),
            flags=re.UNICODE,
        )
        concept_words: list[str] = []
        for word in words:
            if word in seen:
                continue
            seen.add(word)
            concept_words.append(word)
            if len(concept_words) >= 12:
                break

        concept_mag = 0.18 + (0.04 if mentioned else 0.0)
        for word in concept_words:
            self.inject(
                f"text:word:{word}",
                concept_mag,
                32,
            )

        # A few adjacent pairs provide a weak association trace without turning
        # entire sentences into atomic memories.
        for a, b in zip(concept_words, concept_words[1:6]):
            self.inject(
                f"text:pair:{a}|{b}",
                0.10,
                24,
            )

    def inject_voice_snapshot(self, guild_id: int, channel_id: int, user_ids: Iterable[int]) -> None:
        users = list(user_ids)
        self.inject(
            f"voice:guild:{guild_id}:channel:{channel_id}",
            0.18 + 0.12 * min(len(users), 8),
            64,
        )
        for uid in users[:12]:
            self.inject(f"voice:user:{uid}", 0.08, 24)

    def step(self, ticks: int = 1) -> None:
        for _ in range(max(1, ticks)):
            propagated = self.matrix.dot(self.state).astype(self.xp.float32, copy=False)
            noise = self.compute.random_normal(0.0, self.cfg.noise, size=self.state.shape)
            x = (
                self.cfg.leak * self.state
                + self.cfg.propagation_gain * propagated
                + self.plastic_bias
                + noise
            )
            self.state[...] = self.xp.tanh(x)
            self.eligibility[...] = 0.94 * self.eligibility + 0.06 * self.xp.abs(self.state)
            self.plastic_bias *= np.float32(self.cfg.plasticity_decay)
            self.reward_trace *= 0.96
            self.tick_count += 1

    def capture_learning_trace(self, count: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        """Capture a compact CPU copy of the strongest eligibility values."""
        count = max(64, min(int(count), self.c.n_neurons))
        abs_e = self.xp.abs(self.eligibility)
        if count >= self.c.n_neurons:
            idx = self.xp.argsort(abs_e)[::-1]
        else:
            idx = self.xp.argpartition(abs_e, -count)[-count:]
            idx = idx[self.xp.argsort(abs_e[idx])[::-1]]
        idx_cpu = self.compute.to_cpu(idx[:count]).astype(np.int32, copy=False)
        val_cpu = self.compute.to_cpu(self.eligibility[idx[:count]]).astype(np.float32, copy=False)
        return idx_cpu, val_cpu

    def reward(
        self,
        amount: float,
        action: str | None = None,
        trace: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict:
        """Apply reward to the recent neural trace and preferentially to one action."""
        amount = float(max(-1.0, min(1.0, amount)))
        before = self.action_scores()
        self.reward_trace = max(-2.0, min(2.0, self.reward_trace + amount))

        changed_idx_cpu: np.ndarray
        changed_delta_cpu: np.ndarray

        if trace is not None and len(trace[0]):
            idx_cpu, elig_cpu = trace
            idx = self._backend_indices(idx_cpu)
            elig = self.compute.asarray(elig_cpu, dtype=np.float32)
            delta = (self.cfg.plasticity_lr * amount * elig).astype(self.xp.float32)
            self.plastic_bias[idx] += delta
            changed_idx_cpu = idx_cpu.astype(np.int32, copy=False)
            changed_delta_cpu = self.compute.to_cpu(delta).astype(np.float32, copy=False)
        else:
            delta = (self.cfg.plasticity_lr * amount * self.eligibility).astype(self.xp.float32)
            self.plastic_bias += delta
            abs_delta = self.xp.abs(delta)
            k = min(4096, self.c.n_neurons)
            idx = self.xp.argpartition(abs_delta, -k)[-k:]
            idx = idx[self.xp.argsort(abs_delta[idx])[::-1]]
            changed_idx_cpu = self.compute.to_cpu(idx).astype(np.int32, copy=False)
            changed_delta_cpu = self.compute.to_cpu(delta[idx]).astype(np.float32, copy=False)

        # Action-specific reinforcement. This makes feedback about "speak" or
        # "react" preferentially change the corresponding output population.
        if action in self.ACTIONS:
            out_cpu = self._subset("output:action:" + action, self.c.output, 128)
            out = self._backend_indices(out_cpu)
            action_delta = np.float32(self.cfg.plasticity_lr * amount * 8.0)
            self.plastic_bias[out] += action_delta
            self.state[out] += np.float32(0.12 * amount)

            extra_idx = out_cpu.astype(np.int32, copy=False)
            extra_delta = np.full(len(extra_idx), float(action_delta), dtype=np.float32)
            changed_idx_cpu = np.concatenate([changed_idx_cpu, extra_idx])
            changed_delta_cpu = np.concatenate([changed_delta_cpu, extra_delta])

        self.xp.clip(
            self.plastic_bias,
            -self.cfg.max_bias,
            self.cfg.max_bias,
            out=self.plastic_bias,
        )

        if len(self.c.modulatory):
            idx_cpu = self._subset(
                "reward:modulatory", self.c.modulatory, min(128, len(self.c.modulatory))
            )
            idx = self._backend_indices(idx_cpu)
            self.state[idx] += np.float32(0.35 * amount)

        after = self.action_scores()
        impact = {name: after[name] - before[name] for name in self.ACTIONS}

        if len(changed_idx_cpu):
            order = np.argsort(np.abs(changed_delta_cpu))[::-1][:12]
            top_changed = [
                {
                    "root_id": int(self.c.root_ids[int(changed_idx_cpu[i])]),
                    "delta": float(changed_delta_cpu[i]),
                    "activation": float(
                        self.compute.to_cpu(
                            self.state[self._backend_indices(
                                np.asarray([changed_idx_cpu[i]], dtype=np.int32)
                            )]
                        )[0]
                    ),
                }
                for i in order
            ]
            nonzero = int(np.count_nonzero(np.abs(changed_delta_cpu) > 1e-9))
            mean_delta = float(np.mean(changed_delta_cpu))
            max_delta = float(np.max(np.abs(changed_delta_cpu)))
        else:
            top_changed = []
            nonzero = 0
            mean_delta = 0.0
            max_delta = 0.0

        self.last_learning = {
            "amount": amount,
            "action": action,
            "changed_neurons": nonzero,
            "mean_delta": mean_delta,
            "max_delta": max_delta,
            "before": before,
            "after": after,
            "impact": impact,
            "top_changed": top_changed,
        }

        self._session_reward_events += 1
        if amount > 0:
            self._session_positive_reward_events += 1
            self._session_positive_reward_total += amount
        elif amount < 0:
            self._session_negative_reward_events += 1
            self._session_negative_reward_total += amount

        if len(changed_idx_cpu):
            changed_mask = np.abs(changed_delta_cpu) > 1e-9
            if np.any(changed_mask):
                changed_indices = changed_idx_cpu[changed_mask]
                changed_values = changed_delta_cpu[changed_mask]
                self._session_bias_update_operations += int(
                    len(np.unique(changed_indices))
                )
                np.add.at(
                    self._session_reward_bias_delta,
                    changed_indices,
                    changed_values,
                )

        return self.last_learning

    def readout(self, key: str, width: int = 96) -> float:
        idx_cpu = self._subset("output:" + key, self.c.output, width)
        idx = self._backend_indices(idx_cpu)
        raw = self.compute.scalar(self.xp.mean(self.state[idx]))
        # Reward trace is a mild global arousal signal. Most learning is now
        # action-specific in reward(), so feedback no longer lifts every action equally.
        return _sigmoid(3.2 * raw + 0.08 * self.reward_trace)

    def language_word_score(self, token: str, width: int = 32) -> float:
        """Return a connectome-state preference score for one learned word."""
        token = str(token).strip().lower()
        if not token:
            return 0.5

        width = max(8, min(int(width), 64))
        sensory_cpu = self._subset(
            "sensory:text:word:" + token,
            self.c.sensory,
            width,
        )
        output_cpu = self._subset(
            "output:language:word:" + token,
            self.c.output,
            width,
        )
        sensory = self._backend_indices(sensory_cpu)
        output = self._backend_indices(output_cpu)

        sensory_raw = self.compute.scalar(
            self.xp.mean(self.state[sensory])
        )
        output_raw = self.compute.scalar(
            self.xp.mean(self.state[output])
        )
        raw = 0.35 * sensory_raw + 0.65 * output_raw
        return _sigmoid(3.0 * raw + 0.06 * self.reward_trace)

    def action_scores(self) -> dict[str, float]:
        return {name: self.readout("action:" + name, 128) for name in self.ACTIONS}

    def channel_affinity(self, guild_id: int, channel_id: int) -> float:
        return self.readout(f"voice-affinity:{guild_id}:{channel_id}", 96)

    def top_active_neurons(self, count: int = 8) -> list[tuple[int, float]]:
        """Return root_id and signed activation for the strongest neurons."""
        count = max(1, min(int(count), self.c.n_neurons))
        abs_state = self.xp.abs(self.state)
        if count >= self.c.n_neurons:
            idx = self.xp.argsort(abs_state)[::-1]
        else:
            idx = self.xp.argpartition(abs_state, -count)[-count:]
            idx = idx[self.xp.argsort(abs_state[idx])[::-1]]

        idx_cpu = self.compute.to_cpu(idx[:count]).astype(np.int64, copy=False)
        values_cpu = self.compute.to_cpu(self.state[idx[:count]]).astype(np.float32, copy=False)
        return [
            (int(self.c.root_ids[int(i)]), float(v))
            for i, v in zip(idx_cpu, values_cpu)
        ]

    def learning_since_start_diagnostics(self) -> dict:
        current_bias = (
            self.compute.to_cpu(self.plastic_bias)
            .astype(np.float32, copy=False)
        )
        current_delta = current_bias - self._startup_plastic_bias
        current_abs_delta = np.abs(current_delta)

        reward_delta = self._session_reward_bias_delta
        reward_abs_delta = np.abs(reward_delta)
        reward_changed_mask = reward_abs_delta > 1e-9
        changed_unique = int(np.count_nonzero(reward_changed_mask))

        if reward_delta.size and changed_unique:
            top_idx = int(np.argmax(reward_abs_delta))
            top_neuron = {
                "root_id": int(self.c.root_ids[top_idx]),
                "delta": float(reward_delta[top_idx]),
                "current_bias": float(current_bias[top_idx]),
            }
            max_abs_delta = float(reward_abs_delta[top_idx])
        else:
            top_neuron = None
            max_abs_delta = 0.0

        return {
            "started_at": float(self._learning_session_started_at),
            "uptime_seconds": max(
                0.0,
                time.time() - self._learning_session_started_at,
            ),
            "reward_events": int(self._session_reward_events),
            "positive_reward_events": int(
                self._session_positive_reward_events
            ),
            "negative_reward_events": int(
                self._session_negative_reward_events
            ),
            "positive_reward_total": float(
                self._session_positive_reward_total
            ),
            "negative_reward_total": float(
                self._session_negative_reward_total
            ),
            "bias_update_operations": int(
                self._session_bias_update_operations
            ),
            "unique_neurons_changed": changed_unique,
            "bias_mean_abs_delta": (
                float(np.mean(reward_abs_delta))
                if reward_abs_delta.size
                else 0.0
            ),
            "bias_sum_abs_delta": (
                float(np.sum(reward_abs_delta))
                if reward_abs_delta.size
                else 0.0
            ),
            "bias_max_abs_delta": max_abs_delta,
            "top_changed_neuron": top_neuron,
            "current_unique_neurons_changed": int(
                np.count_nonzero(current_abs_delta > 1e-9)
            ),
            "current_bias_mean_abs_delta": (
                float(np.mean(current_abs_delta))
                if current_abs_delta.size
                else 0.0
            ),
            "current_bias_max_abs_delta": (
                float(np.max(current_abs_delta))
                if current_abs_delta.size
                else 0.0
            ),
        }

    def learning_diagnostics(self) -> dict:
        return self.last_learning

    def diagnostics(self) -> dict[str, float | int | str]:
        active = self.compute.int_scalar(
            self.xp.count_nonzero(self.xp.abs(self.state) > 0.1)
        )
        mean_abs = self.compute.scalar(self.xp.mean(self.xp.abs(self.state)))
        max_abs = self.compute.scalar(self.xp.max(self.xp.abs(self.state)))
        bias_abs = self.xp.abs(self.plastic_bias)
        bias_mean = self.compute.scalar(self.xp.mean(self.plastic_bias))
        bias_mean_abs = self.compute.scalar(self.xp.mean(bias_abs))
        bias_max_abs = self.compute.scalar(self.xp.max(bias_abs))
        bias_positive = self.compute.int_scalar(self.xp.count_nonzero(self.plastic_bias > 1e-7))
        bias_negative = self.compute.int_scalar(self.xp.count_nonzero(self.plastic_bias < -1e-7))
        hist_counts, hist_edges = self.xp.histogram(
            self.plastic_bias,
            bins=21,
            range=(-self.cfg.max_bias, self.cfg.max_bias),
        )
        bias_hist = {
            "counts": self.compute.to_cpu(hist_counts).astype(np.int64).tolist(),
            "edges": self.compute.to_cpu(hist_edges).astype(np.float32).tolist(),
        }
        return {
            "neurons": self.c.n_neurons,
            "connections": int(self.c.matrix.nnz),
            "active_abs_gt_0_1": active,
            "mean_abs": mean_abs,
            "max_abs": max_abs,
            "reward_trace": float(self.reward_trace),
            "ticks": int(self.tick_count),
            "backend": self.compute.info.active,
            "device": self.compute.info.device_name,
            "cuda_runtime": self.compute.info.cuda_runtime or "",
            "bias_mean": bias_mean,
            "bias_mean_abs": bias_mean_abs,
            "bias_max_abs": bias_max_abs,
            "bias_positive": bias_positive,
            "bias_negative": bias_negative,
            "bias_hist": bias_hist,
        }
