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
        self._sensory_lookup = set(int(i) for i in self.c.sensory.tolist())
        self._output_lookup = set(int(i) for i in self.c.output.tolist())
        self._modulatory_lookup = set(
            int(i) for i in self.c.modulatory.tolist()
        )
        self._connectome_visual_stable_indices: list[int] = []
        self._connectome_visual_stable_updated = 0.0
        self._neuro_map_coords = np.zeros((n, 3), dtype=np.float32)
        self._neuro_map_real_mask = np.zeros(n, dtype=bool)
        self._neuro_map_reference_indices = np.empty(0, dtype=np.int32)
        self._neuro_map_regions: dict[str, np.ndarray] = {}
        self._action_output_pools: dict[str, np.ndarray] = {}
        self._init_neuro_map_metadata()
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

    def _init_neuro_map_metadata(self) -> None:
        """Prepare compact spatial/annotation caches for the web neuro-map."""
        n = self.c.n_neurons
        meta = self.c.neuron_meta or {}

        def string_meta(key: str) -> np.ndarray:
            arr = meta.get(key)
            if arr is None or len(arr) != n:
                return np.full(n, "", dtype="<U1")
            return np.asarray(arr).astype(str, copy=False)

        x = np.asarray(
            meta.get("x", np.full(n, np.nan, dtype=np.float32)),
            dtype=np.float32,
        )
        y = np.asarray(
            meta.get("y", np.full(n, np.nan, dtype=np.float32)),
            dtype=np.float32,
        )
        z = np.asarray(
            meta.get("z", np.full(n, np.nan, dtype=np.float32)),
            dtype=np.float32,
        )
        finite_positions = (
            np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        )
        coordinate_source = str(
            self.c.metadata.get("coordinate_source", "")
        ).lower()
        coords_are_real = (
            "flywire" in coordinate_source
            and not bool(self.c.metadata.get("demo"))
        )
        real_mask = (
            finite_positions
            if coords_are_real
            else np.zeros(n, dtype=bool)
        )
        self._neuro_map_real_mask = real_mask

        coords = np.zeros((n, 3), dtype=np.float32)
        raw_axes = (x, y, z)
        for axis, raw in enumerate(raw_axes):
            finite = raw[np.isfinite(raw)]
            if len(finite) >= 8:
                low, high = np.percentile(finite, [1.0, 99.0])
                if high <= low:
                    low = float(np.min(finite))
                    high = float(np.max(finite))
                span = max(1e-6, float(high - low))
                coords[:, axis] = np.clip(
                    (raw - low) / span,
                    0.0,
                    1.0,
                )
            else:
                coords[:, axis] = np.nan

        # Deterministic brain-shaped fallback for neurons without coordinates.
        # It is clearly flagged in the UI and never presented as anatomy.
        rng = np.random.default_rng(self.cfg.seed + 90210)
        angle = rng.uniform(0.0, 2.0 * np.pi, size=n)
        radius = np.sqrt(rng.uniform(0.0, 1.0, size=n))
        fallback = np.empty((n, 3), dtype=np.float32)
        fallback[:, 0] = (
            0.5 + 0.46 * radius * np.cos(angle)
        ).astype(np.float32)
        fallback[:, 1] = (
            0.5 + 0.36 * radius * np.sin(angle)
        ).astype(np.float32)
        fallback[:, 2] = np.clip(
            0.5 + rng.normal(0.0, 0.22, size=n),
            0.02,
            0.98,
        ).astype(np.float32)

        side = string_meta("side")
        left = np.char.find(np.char.lower(side), "left") >= 0
        right = np.char.find(np.char.lower(side), "right") >= 0
        fallback[left, 0] = np.minimum(
            fallback[left, 0],
            0.47,
        )
        fallback[right, 0] = np.maximum(
            fallback[right, 0],
            0.53,
        )

        missing = ~real_mask
        coords[missing] = fallback[missing]
        coords[~np.isfinite(coords)] = fallback[~np.isfinite(coords)]
        self._neuro_map_coords = coords

        # Region grouping uses Codex class first, then super_class. These are
        # biological annotations, unlike Mucha's artificial Discord readouts.
        cell_class = string_meta("cell_class")
        super_class = string_meta("super_class")
        region_labels = np.where(
            np.char.str_len(cell_class) > 0,
            cell_class,
            super_class,
        )
        regions: dict[str, list[int]] = {}
        for idx, raw_label in enumerate(region_labels):
            label = str(raw_label).strip()
            if not label:
                continue
            regions.setdefault(label, []).append(idx)
        self._neuro_map_regions = {
            label: np.asarray(indices, dtype=np.int32)
            for label, indices in regions.items()
            if len(indices) >= 2
        }

        self._action_output_pools = {
            action: self._subset(
                "output:action:" + action,
                self.c.output,
                128,
            )
            for action in self.ACTIONS
        }

        # Fixed low-density point cloud gives the eye a stable outline of the
        # whole brain while only active neurons are drawn brightly.
        sample_n = min(1800, n)
        sample_rng = np.random.default_rng(self.cfg.seed + 4242)
        if sample_n >= n:
            ref = np.arange(n, dtype=np.int32)
        else:
            real_idx = np.flatnonzero(real_mask)
            keep_real = min(len(real_idx), int(sample_n * 0.85))
            chosen: list[int] = []
            if keep_real:
                chosen.extend(
                    sample_rng.choice(
                        real_idx,
                        size=keep_real,
                        replace=False,
                    ).astype(np.int32).tolist()
                )
            remaining = sample_n - len(chosen)
            if remaining:
                pool = np.setdiff1d(
                    np.arange(n, dtype=np.int32),
                    np.asarray(chosen, dtype=np.int32),
                    assume_unique=False,
                )
                chosen.extend(
                    sample_rng.choice(
                        pool,
                        size=min(remaining, len(pool)),
                        replace=False,
                    ).astype(np.int32).tolist()
                )
            ref = np.asarray(chosen, dtype=np.int32)
        self._neuro_map_reference_indices = ref

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

    def connectome_visual_snapshot(
        self,
        count: int = 64,
        edge_limit: int = 180,
        follow_activity: bool = False,
    ) -> dict:
        """Compact graph for the neural dashboard.

        Stable mode keeps the same functional window for roughly 45 seconds
        and only replaces neurons when a newcomer is clearly more active.
        Follow mode deliberately tracks the current strongest populations.
        """
        count = max(12, min(int(count), 80, self.c.n_neurons))
        edge_limit = max(16, min(int(edge_limit), 260))
        follow_activity = bool(follow_activity)

        state_cpu = self.compute.to_cpu(self.state).astype(
            np.float32,
            copy=False,
        )
        eligibility_cpu = self.compute.to_cpu(self.eligibility).astype(
            np.float32,
            copy=False,
        )
        bias_cpu = self.compute.to_cpu(self.plastic_bias).astype(
            np.float32,
            copy=False,
        )
        abs_state = np.abs(state_cpu)

        sensory_set = self._sensory_lookup
        output_set = self._output_lookup
        modulatory_set = self._modulatory_lookup

        def role_of(idx: int) -> str:
            if idx in sensory_set:
                return "sensory"
            if idx in output_set:
                return "output"
            if idx in modulatory_set:
                return "modulatory"
            return "internal"

        def strongest(pool: np.ndarray, limit: int) -> list[int]:
            if limit <= 0 or len(pool) == 0:
                return []
            pool = np.asarray(pool, dtype=np.int32)
            values = abs_state[pool]
            k = min(limit, len(pool))
            if k >= len(pool):
                order = np.argsort(values)[::-1]
            else:
                part = np.argpartition(values, -k)[-k:]
                order = part[np.argsort(values[part])[::-1]]
            return [int(pool[i]) for i in order[:k]]

        def active_selection(limit: int) -> list[int]:
            limit = max(12, min(int(limit), 80, self.c.n_neurons))
            sensory_n = max(4, limit // 6)
            output_n = max(5, limit // 5)
            modulatory_n = max(3, limit // 10)

            selected: list[int] = []
            selected.extend(strongest(self.c.sensory, sensory_n))
            selected.extend(strongest(self.c.output, output_n))
            selected.extend(strongest(self.c.modulatory, modulatory_n))

            remaining = max(0, limit - len(set(selected)))
            if remaining:
                k = min(
                    self.c.n_neurons,
                    max(limit * 4, remaining),
                )
                if k >= self.c.n_neurons:
                    candidates = np.argsort(abs_state)[::-1]
                else:
                    part = np.argpartition(abs_state, -k)[-k:]
                    candidates = part[
                        np.argsort(abs_state[part])[::-1]
                    ]
                selected.extend(int(i) for i in candidates)

            unique: list[int] = []
            seen: set[int] = set()
            for idx in selected:
                if idx in seen:
                    continue
                seen.add(idx)
                unique.append(idx)
                if len(unique) >= limit:
                    break
            return unique

        dynamic = active_selection(count)
        now = time.monotonic()
        replacements = 0
        stable_window_seconds = 45.0

        if follow_activity:
            unique = dynamic
            stable_age = 0.0
            refresh_in = 0.0
            mode = "follow_activity"
        else:
            cached = self._connectome_visual_stable_indices
            cache_valid = (
                len(cached) == count
                and all(0 <= idx < self.c.n_neurons for idx in cached)
            )
            if not cache_valid:
                self._connectome_visual_stable_indices = list(dynamic)
                self._connectome_visual_stable_updated = now
                cached = self._connectome_visual_stable_indices

            age = max(
                0.0,
                now - float(self._connectome_visual_stable_updated),
            )

            # Normal replacements happen once per stable window. A single very
            # strong newcomer may break in earlier, but only after a short
            # settling period and only if it is dramatically stronger.
            scheduled = age >= stable_window_seconds
            emergency = age >= 10.0

            if scheduled or emergency:
                stable = list(cached)
                stable_set = set(stable)
                candidate_pool = active_selection(min(80, count + 16))
                candidate_pool = [
                    idx for idx in candidate_pool
                    if idx not in stable_set
                ]

                max_replacements = 6 if scheduled else 1
                ratio_required = 1.35 if scheduled else 2.50
                margin_required = 0.018 if scheduled else 0.055

                for candidate in candidate_pool:
                    if replacements >= max_replacements:
                        break
                    candidate_role = role_of(candidate)
                    same_role_positions = [
                        pos for pos, idx in enumerate(stable)
                        if role_of(idx) == candidate_role
                    ]
                    if not same_role_positions:
                        continue

                    weakest_pos = min(
                        same_role_positions,
                        key=lambda pos: float(abs_state[stable[pos]]),
                    )
                    weakest_idx = stable[weakest_pos]
                    weak = float(abs_state[weakest_idx])
                    strong = float(abs_state[candidate])

                    if strong < max(
                        weak * ratio_required,
                        weak + margin_required,
                    ):
                        continue

                    stable_set.discard(weakest_idx)
                    stable[weakest_pos] = candidate
                    stable_set.add(candidate)
                    replacements += 1

                if replacements or scheduled:
                    self._connectome_visual_stable_indices = stable
                    self._connectome_visual_stable_updated = now
                    cached = stable

            unique = list(self._connectome_visual_stable_indices)
            stable_age = max(
                0.0,
                now - float(self._connectome_visual_stable_updated),
            )
            refresh_in = max(
                0.0,
                stable_window_seconds - stable_age,
            )
            mode = "stable"

        selected_arr = np.asarray(unique, dtype=np.int32)

        nodes = []
        for idx in unique:
            roles = []
            if idx in sensory_set:
                roles.append("sensory")
            if idx in output_set:
                roles.append("output")
            if idx in modulatory_set:
                roles.append("modulatory")
            if not roles:
                roles.append("internal")
            nodes.append({
                "id": str(int(self.c.root_ids[idx])),
                "index": int(idx),
                "activation": float(state_cpu[idx]),
                "eligibility": float(eligibility_cpu[idx]),
                "bias": float(bias_cpu[idx]),
                "role": roles[0],
                "roles": roles,
            })

        edges: list[dict] = []
        if len(selected_arr) >= 2:
            sub = self.c.matrix[selected_arr][:, selected_arr].tocoo()
            ranked = []
            for row, col, weight in zip(
                sub.row,
                sub.col,
                sub.data,
            ):
                if row == col:
                    continue
                source_idx = int(selected_arr[int(col)])
                target_idx = int(selected_arr[int(row)])
                weight_f = float(weight)
                activity = (
                    0.20
                    + abs(float(state_cpu[source_idx]))
                    + 0.35 * abs(float(state_cpu[target_idx]))
                )
                importance = abs(weight_f) * activity
                ranked.append(
                    (
                        importance,
                        source_idx,
                        target_idx,
                        weight_f,
                    )
                )
            ranked.sort(key=lambda item: item[0], reverse=True)
            for importance, source_idx, target_idx, weight in ranked[:edge_limit]:
                edges.append({
                    "source": str(int(self.c.root_ids[source_idx])),
                    "target": str(int(self.c.root_ids[target_idx])),
                    "weight": float(weight),
                    "importance": float(importance),
                })

        role_counts = {
            "sensory": sum(1 for n in nodes if "sensory" in n["roles"]),
            "internal": sum(1 for n in nodes if n["role"] == "internal"),
            "modulatory": sum(
                1 for n in nodes if "modulatory" in n["roles"]
            ),
            "output": sum(1 for n in nodes if "output" in n["roles"]),
        }

        return {
            "nodes": nodes,
            "edges": edges,
            "role_counts": role_counts,
            "selected_neurons": len(nodes),
            "selected_edges": len(edges),
            "total_neurons": int(self.c.n_neurons),
            "total_connections": int(self.c.matrix.nnz),
            "mean_abs_activation": float(
                np.mean(abs_state) if len(state_cpu) else 0.0
            ),
            "max_abs_activation": float(
                np.max(abs_state) if len(state_cpu) else 0.0
            ),
            "mode": mode,
            "stable_window_seconds": stable_window_seconds,
            "stable_age_seconds": float(stable_age),
            "refresh_in_seconds": float(refresh_in),
            "replacements": int(replacements),
        }

    def neuro_map_snapshot(
        self,
        count: int = 220,
        projection: str = "xy",
    ) -> dict:
        """Return a spatial activity map with biological annotations.

        Coordinates come from FlyWire Codex marked-neuron coordinates when
        neuron_meta.npz is present. Missing positions use an explicit synthetic
        fallback so the dashboard remains usable on older caches.
        """
        projection = str(projection or "xy").lower()
        if projection not in {"xy", "xz", "yz"}:
            projection = "xy"

        count = max(40, min(int(count), 360, self.c.n_neurons))
        state_cpu = self.compute.to_cpu(self.state).astype(
            np.float32,
            copy=False,
        )
        eligibility_cpu = self.compute.to_cpu(self.eligibility).astype(
            np.float32,
            copy=False,
        )
        bias_cpu = self.compute.to_cpu(self.plastic_bias).astype(
            np.float32,
            copy=False,
        )
        abs_state = np.abs(state_cpu)

        if count >= self.c.n_neurons:
            selected = np.argsort(abs_state)[::-1].astype(np.int32)
        else:
            part = np.argpartition(abs_state, -count)[-count:]
            selected = part[
                np.argsort(abs_state[part])[::-1]
            ].astype(np.int32)

        meta = self.c.neuron_meta or {}

        def text_at(key: str, idx: int) -> str:
            arr = meta.get(key)
            if arr is None or len(arr) != self.c.n_neurons:
                return ""
            return str(arr[idx]).strip()

        # Vectorized direct connectivity into Mucha's artificial action
        # readout populations. This is system influence, not a biological claim.
        action_influence: dict[str, np.ndarray] = {}
        for action, targets in self._action_output_pools.items():
            if len(targets) == 0 or len(selected) == 0:
                action_influence[action] = np.zeros(
                    len(selected),
                    dtype=np.float32,
                )
                continue
            sub = abs(self.c.matrix[targets][:, selected])
            action_influence[action] = np.asarray(
                sub.sum(axis=0)
            ).ravel().astype(np.float32, copy=False)

        nodes: list[dict] = []
        sensory_set = self._sensory_lookup
        output_set = self._output_lookup
        modulatory_set = self._modulatory_lookup
        for pos, idx_raw in enumerate(selected):
            idx = int(idx_raw)
            roles = []
            if idx in sensory_set:
                roles.append("sensory")
            if idx in output_set:
                roles.append("output")
            if idx in modulatory_set:
                roles.append("modulatory")
            if not roles:
                roles.append("internal")

            actions = sorted(
                (
                    {
                        "name": action,
                        "strength": float(values[pos]),
                    }
                    for action, values in action_influence.items()
                    if float(values[pos]) > 0.0
                ),
                key=lambda item: item["strength"],
                reverse=True,
            )[:3]

            nodes.append({
                "id": str(int(self.c.root_ids[idx])),
                "index": idx,
                "x": float(self._neuro_map_coords[idx, 0]),
                "y": float(self._neuro_map_coords[idx, 1]),
                "z": float(self._neuro_map_coords[idx, 2]),
                "real_position": bool(self._neuro_map_real_mask[idx]),
                "activation": float(state_cpu[idx]),
                "eligibility": float(eligibility_cpu[idx]),
                "bias": float(bias_cpu[idx]),
                "role": roles[0],
                "roles": roles,
                "super_class": text_at("super_class", idx),
                "cell_class": text_at("cell_class", idx),
                "sub_class": text_at("sub_class", idx),
                "side": text_at("side", idx),
                "flow": text_at("flow", idx),
                "nerve": text_at("nerve", idx),
                "primary_type": text_at("primary_type", idx),
                "nt_type": text_at("nt_type", idx),
                "system_actions": actions,
            })

        regions: list[dict] = []
        for label, indices in self._neuro_map_regions.items():
            if len(indices) == 0:
                continue
            values = abs_state[indices]
            mean_abs = float(np.mean(values))
            max_abs = float(np.max(values))
            active_count = int(np.count_nonzero(values > 0.1))
            # Prefer regions that are both strongly and broadly active.
            score = mean_abs * (
                1.0 + math.log1p(active_count) * 0.35
            )
            coords = self._neuro_map_coords[indices]
            regions.append({
                "name": label,
                "mean_abs": mean_abs,
                "max_abs": max_abs,
                "active_count": active_count,
                "total": int(len(indices)),
                "score": float(score),
                "x": float(np.mean(coords[:, 0])),
                "y": float(np.mean(coords[:, 1])),
                "z": float(np.mean(coords[:, 2])),
            })
        regions.sort(key=lambda item: item["score"], reverse=True)
        regions = regions[:24]

        reference = [
            {
                "x": float(self._neuro_map_coords[idx, 0]),
                "y": float(self._neuro_map_coords[idx, 1]),
                "z": float(self._neuro_map_coords[idx, 2]),
                "real": bool(self._neuro_map_real_mask[idx]),
            }
            for idx in self._neuro_map_reference_indices
        ]

        real_count = int(np.count_nonzero(self._neuro_map_real_mask))
        coverage = real_count / max(1, self.c.n_neurons)
        if coverage >= 0.70:
            coordinate_mode = "real"
        elif coverage > 0.0:
            coordinate_mode = "hybrid"
        else:
            coordinate_mode = "synthetic"

        return {
            "projection": projection,
            "coordinate_mode": coordinate_mode,
            "coordinate_coverage": float(coverage),
            "coordinate_neurons": real_count,
            "nodes": nodes,
            "regions": regions,
            "reference": reference,
            "total_neurons": int(self.c.n_neurons),
            "active_abs_gt_0_1": int(
                np.count_nonzero(abs_state > 0.1)
            ),
            "mean_abs_activation": float(
                np.mean(abs_state) if len(abs_state) else 0.0
            ),
            "max_abs_activation": float(
                np.max(abs_state) if len(abs_state) else 0.0
            ),
        }

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
