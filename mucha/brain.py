from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter, deque
from typing import Iterable

import numpy as np
from scipy import sparse as scipy_sparse

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

    # These are seed labels for actions that have a defensible broad
    # motor/behavioral analogue in the fly. "speak" is intentionally empty:
    # Discord communication has no literal FAFB output neuron and is learned
    # adaptively from text/social sensory cues plus the real connectome.
    ACTION_BIOLOGICAL_SEEDS = {
        "speak": (),
        "react": (
            "adn1", "adn2", "groom", "dnp07", "dnp10",
            "landing", "jump", "giant fiber",
        ),
        "voice_join": (
            "dnp09", "bdn2", "odn1", "forward", "walking",
            "locomot",
        ),
        "voice_move": (
            "dna01", "dna02", "dng13", "steer", "turn",
            "walking", "locomot",
        ),
        "voice_leave": (
            "mdn", "moonwalker", "dnp06", "escape", "evasive",
            "backward", "takeoff", "giant fiber",
        ),
        "explore": (
            "dna01", "dna02", "dng13", "walking", "steer",
            "turn", "locomot", "navigation",
        ),
        "stay": (
            "dnp09", "freeze", "freezing", "stop", "brake",
        ),
    }

    def __init__(self, connectome: Connectome, cfg: BrainConfig):
        self.c = connectome
        self.cfg = cfg
        self.compute = ComputeBackend(cfg.backend, cfg.gpu_device)
        self.xp = self.compute.xp
        self._modulator_pools = self._build_modulator_pools()
        self._neuromodulator_state = {
            "dopamine": 0.0,
            "serotonin": 0.0,
            "octopamine": 0.0,
        }
        runtime_matrix = self._build_runtime_connectome_matrix()
        self.matrix = self.compute.sparse_from_scipy(runtime_matrix)
        self._runtime_matrix_cpu = runtime_matrix
        self._neuromodulator_effects = {
            "effective_leak": float(cfg.leak),
            "effective_gain": float(cfg.propagation_gain),
            "effective_noise": float(cfg.noise),
            "plasticity_gain": 1.0,
        }

        # CPU RNG is only used for stable index selection. Dynamic noise lives
        # on the active backend and therefore stays on the GPU in CUDA mode.
        self.rng = np.random.default_rng(cfg.seed)
        if self.compute.is_gpu:
            self.compute.cp.random.seed(cfg.seed)

        n = self.c.n_neurons
        self.state = self.compute.zeros(n, dtype=np.float32)
        self.eligibility = self.compute.zeros(n, dtype=np.float32)
        self.plastic_bias = self.compute.zeros(n, dtype=np.float32)
        self._synaptic_delta_map: dict[int, float] = {}
        self._synaptic_matrix_cpu = scipy_sparse.csr_matrix(
            self.c.matrix.shape,
            dtype=np.float32,
        )
        self.synaptic_matrix = self.compute.sparse_from_scipy(
            self._synaptic_matrix_cpu
        )
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
        self._neuro_map_region_types: dict[str, list[dict]] = {}
        self._neuro_map_region_source = "cell_class"
        self._neuro_map_history = deque(maxlen=180)
        self._action_output_pools: dict[str, np.ndarray] = {}
        self._action_output_seeds: dict[str, np.ndarray] = {}
        self._action_output_info: dict[str, dict] = {}
        self._adaptive_speak_cues = np.empty(0, dtype=np.int32)
        self._adaptive_speak_structural_score = np.empty(
            0,
            dtype=np.float32,
        )
        self._action_structural_cache: tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ] | None = None
        self._word_association_signature_cache: dict[
            str, tuple[np.ndarray, np.ndarray]
        ] = {}
        self._absolute_connectome_cpu: scipy_sparse.csr_matrix | None = None
        self._guided_sensory_cache: dict[
            tuple[str, str, int, int],
            tuple[np.ndarray, np.ndarray, dict],
        ] = {}
        self._last_guided_cue: dict = {}
        self._user_identity_cache: dict[int, np.ndarray] = {}
        self._user_memory_cache: dict[int, np.ndarray] = {}
        self._user_memory_last_activation: dict[int, float] = {}
        self._init_neuro_map_metadata()
        self.last_learning: dict = {
            "amount": 0.0,
            "action": None,
            "changed_neurons": 0,
            "mean_delta": 0.0,
            "max_delta": 0.0,
            "changed_synapses": 0,
            "learned_synapses": 0,
            "synaptic_mean_delta": 0.0,
            "synaptic_max_delta": 0.0,
            "before": {},
            "after": {},
            "impact": {},
            "top_changed": [],
        }
        self._load_state()
        self._refresh_adaptive_speak_pool()
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

    def _build_modulator_pools(self) -> dict[str, np.ndarray]:
        """Split annotated modulatory neurons by predicted transmitter."""
        n = self.c.n_neurons
        meta = self.c.neuron_meta or {}
        nt = meta.get("nt_type")
        pools = {
            "dopamine": [],
            "serotonin": [],
            "octopamine": [],
        }
        aliases = {
            "dopamine": {"da", "dopamine"},
            "serotonin": {"ser", "5ht", "5-ht", "serotonin"},
            "octopamine": {"oct", "oa", "octopamine"},
        }
        if nt is not None and len(nt) == n:
            values = np.asarray(nt).astype(str, copy=False)
            for idx, raw in enumerate(values):
                value = str(raw).strip().lower()
                for name, names in aliases.items():
                    if value in names:
                        pools[name].append(idx)
                        break
        return {
            name: np.asarray(indices, dtype=np.int32)
            for name, indices in pools.items()
        }

    def _build_runtime_connectome_matrix(self) -> scipy_sparse.csr_matrix:
        """Reduce direct monoamine columns when global modulation is enabled."""
        matrix = self.c.matrix.tocsr().astype(np.float32, copy=True)
        typed_modulators = np.unique(
            np.concatenate(
                [
                    pool
                    for pool in self._modulator_pools.values()
                    if len(pool)
                ]
            )
        ) if any(
            len(pool)
            for pool in self._modulator_pools.values()
        ) else np.empty(0, dtype=np.int32)
        if (
            not self.cfg.neuromodulation_enabled
            or not len(typed_modulators)
        ):
            return matrix

        residual = max(
            0.0,
            min(
                1.0,
                float(self.cfg.neuromodulatory_direct_residual),
            ),
        )
        scale = np.ones(self.c.n_neurons, dtype=np.float32)
        scale[typed_modulators] = np.float32(residual)
        return matrix.dot(
            scipy_sparse.diags(scale, dtype=np.float32)
        ).tocsr().astype(np.float32)

    def _update_neuromodulator_state(self) -> None:
        if not self.cfg.neuromodulation_enabled:
            for key in self._neuromodulator_state:
                self._neuromodulator_state[key] = 0.0
            return

        smoothing = max(
            0.0,
            min(0.999, float(self.cfg.neuromodulator_smoothing)),
        )
        for name, pool in self._modulator_pools.items():
            target = 0.0
            if len(pool):
                idx = self._backend_indices(pool)
                target = self.compute.scalar(
                    self.xp.mean(self.xp.abs(self.state[idx]))
                )
                target = max(0.0, min(1.0, float(target)))
            previous = float(
                self._neuromodulator_state.get(name, 0.0)
            )
            self._neuromodulator_state[name] = (
                smoothing * previous
                + (1.0 - smoothing) * target
            )

    def _plasticity_gain(self) -> float:
        if not self.cfg.neuromodulation_enabled:
            return 1.0
        dopamine = float(
            self._neuromodulator_state.get("dopamine", 0.0)
        )
        return max(
            0.25,
            min(
                4.0,
                1.0
                + float(self.cfg.dopamine_plasticity_gain)
                * dopamine,
            ),
        )

    def _neuromodulation_tick_effects(
        self,
    ) -> tuple[float, float, float]:
        self._update_neuromodulator_state()
        if not self.cfg.neuromodulation_enabled:
            leak = float(self.cfg.leak)
            gain = float(self.cfg.propagation_gain)
            noise = float(self.cfg.noise)
        else:
            serotonin = float(
                self._neuromodulator_state.get("serotonin", 0.0)
            )
            octopamine = float(
                self._neuromodulator_state.get("octopamine", 0.0)
            )
            leak = min(
                0.995,
                max(
                    0.0,
                    float(self.cfg.leak)
                    + float(self.cfg.serotonin_stability_gain)
                    * serotonin,
                ),
            )
            gain = max(
                0.0,
                float(self.cfg.propagation_gain)
                * (
                    1.0
                    + float(self.cfg.octopamine_arousal_gain)
                    * octopamine
                ),
            )
            noise = max(
                0.0,
                float(self.cfg.noise)
                * (1.0 + 0.70 * octopamine)
                * max(0.40, 1.0 - 0.35 * serotonin),
            )
        self._neuromodulator_effects = {
            "effective_leak": leak,
            "effective_gain": gain,
            "effective_noise": noise,
            "plasticity_gain": self._plasticity_gain(),
        }
        return leak, gain, noise

    def neuromodulator_diagnostics(self) -> dict:
        return {
            "enabled": bool(self.cfg.neuromodulation_enabled),
            "direct_residual": float(
                self.cfg.neuromodulatory_direct_residual
            ),
            "dopamine": {
                "neurons": int(
                    len(self._modulator_pools["dopamine"])
                ),
                "level": float(
                    self._neuromodulator_state["dopamine"]
                ),
            },
            "serotonin": {
                "neurons": int(
                    len(self._modulator_pools["serotonin"])
                ),
                "level": float(
                    self._neuromodulator_state["serotonin"]
                ),
            },
            "octopamine": {
                "neurons": int(
                    len(self._modulator_pools["octopamine"])
                ),
                "level": float(
                    self._neuromodulator_state["octopamine"]
                ),
            },
            **self._neuromodulator_effects,
        }

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

        # Prefer named FlyWire neuropils when the metadata builder could
        # derive them from the connection table. A neuron can span several
        # neuropils; this compact dashboard groups it by its strongest incident
        # synapse-count neuropil and exposes the top three in the inspector.
        primary_neuropil = string_meta("primary_neuropil")
        cell_class = string_meta("cell_class")
        super_class = string_meta("super_class")
        primary_type = string_meta("primary_type")
        neuropil_coverage = float(
            np.count_nonzero(
                np.char.str_len(primary_neuropil) > 0
            )
            / max(1, n)
        )
        if neuropil_coverage >= 0.10:
            region_labels = primary_neuropil
            self._neuro_map_region_source = "neuropil"
        else:
            region_labels = np.where(
                np.char.str_len(cell_class) > 0,
                cell_class,
                super_class,
            )
            self._neuro_map_region_source = "cell_class"

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

        self._neuro_map_region_types = {}
        for label, indices in self._neuro_map_regions.items():
            names = []
            for idx in indices:
                value = str(primary_type[int(idx)]).strip()
                if not value:
                    value = str(cell_class[int(idx)]).strip()
                if value:
                    names.append(value)
            counts = Counter(names)
            self._neuro_map_region_types[label] = [
                {"name": name, "count": int(count)}
                for name, count in counts.most_common(5)
            ]

        self._build_action_output_pools()

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

    def _action_metadata_text(self) -> np.ndarray:
        """Return one normalized searchable annotation string per neuron."""
        n = self.c.n_neurons
        meta = self.c.neuron_meta or {}
        fields = (
            "primary_type",
            "cell_class",
            "sub_class",
            "super_class",
            "flow",
            "nerve",
            "primary_neuropil",
        )
        text = np.full(n, "", dtype=object)
        for key in fields:
            arr = meta.get(key)
            if arr is None or len(arr) != n:
                continue
            values = np.asarray(arr).astype(str, copy=False)
            for i, value in enumerate(values):
                value = str(value).strip().lower()
                if value:
                    text[i] = (str(text[i]) + " " + value).strip()
        return np.asarray(text, dtype=str)

    def _structural_action_fallback(
        self,
        action: str,
        width: int,
    ) -> np.ndarray:
        """Pick a deterministic non-random output pool from graph structure."""
        output = np.asarray(self.c.output, dtype=np.int32)
        if not len(output):
            output = np.arange(self.c.n_neurons, dtype=np.int32)
        width = min(max(1, int(width)), len(output))

        if self._action_structural_cache is None:
            abs_matrix = self.c.matrix.copy()
            abs_matrix.data = np.abs(abs_matrix.data)
            incoming_all = np.asarray(
                abs_matrix.sum(axis=1)
            ).ravel().astype(np.float32)
            outgoing_all = np.asarray(
                abs_matrix.sum(axis=0)
            ).ravel().astype(np.float32)
            signed_out_all = np.asarray(
                self.c.matrix.sum(axis=0)
            ).ravel().astype(np.float32)
            self._action_structural_cache = (
                output.copy(),
                incoming_all[output],
                outgoing_all[output],
                signed_out_all[output],
            )

        cached_output, incoming, outgoing, signed_out = (
            self._action_structural_cache
        )
        if not np.array_equal(cached_output, output):
            self._action_structural_cache = None
            return self._structural_action_fallback(
                action,
                width,
            )

        def norm(values: np.ndarray) -> np.ndarray:
            values = np.asarray(values, dtype=np.float32)
            lo = float(np.min(values)) if len(values) else 0.0
            hi = float(np.max(values)) if len(values) else 0.0
            if hi - lo <= 1e-9:
                return np.zeros_like(values)
            return (values - lo) / (hi - lo)

        inc = norm(incoming)
        out = norm(outgoing)
        signed = norm(signed_out)
        balanced = 1.0 - np.abs(inc - out)

        weights = {
            "speak": (0.20, 0.65, 0.15, 0.10),
            "react": (0.65, 0.20, 0.15, 0.05),
            "voice_join": (0.30, 0.45, 0.25, 0.10),
            "voice_move": (0.25, 0.35, 0.15, 0.40),
            "voice_leave": (0.20, 0.50, 0.30, 0.05),
            "explore": (0.35, 0.30, 0.10, 0.35),
            "stay": (0.50, 0.10, 0.10, 0.25),
        }
        wi, wo, ws, wb = weights.get(
            action,
            (0.25, 0.45, 0.15, 0.15),
        )
        score = wi * inc + wo * out + ws * signed + wb * balanced

        # Different actions use opposite tails where that better matches the
        # adapter semantics, but every choice still comes from graph features.
        if action == "stay":
            order = np.argsort(score)
        else:
            order = np.argsort(score)[::-1]
        return output[order[:width]].astype(np.int32, copy=False)

    def _expand_action_seed_pool(
        self,
        seeds: np.ndarray,
        width: int,
    ) -> np.ndarray:
        """Expand typed seeds through their real one-hop FAFB connectivity."""
        seeds = np.unique(
            np.asarray(seeds, dtype=np.int32)
        )
        output = np.asarray(self.c.output, dtype=np.int32)
        if not len(output):
            output = np.arange(self.c.n_neurons, dtype=np.int32)
        width = min(max(1, int(width)), len(output))
        if not len(seeds):
            return np.empty(0, dtype=np.int32)

        if len(seeds) >= width:
            return seeds[:width].astype(np.int32, copy=False)

        outgoing = abs(self.c.matrix[output][:, seeds])
        incoming = abs(self.c.matrix[seeds][:, output])
        score = (
            np.asarray(outgoing.sum(axis=1)).ravel()
            + 0.55
            * np.asarray(incoming.sum(axis=0)).ravel()
        ).astype(np.float32, copy=False)

        seed_set = set(int(i) for i in seeds.tolist())
        order = np.argsort(score)[::-1]
        expanded = list(int(i) for i in seeds.tolist())
        for pos in order:
            idx = int(output[int(pos)])
            if idx in seed_set or float(score[int(pos)]) <= 0.0:
                continue
            expanded.append(idx)
            seed_set.add(idx)
            if len(expanded) >= width:
                break
        return np.asarray(expanded[:width], dtype=np.int32)

    def _build_adaptive_speak_cues(self) -> np.ndarray:
        """Return the artificial sensory adapter used for text/social input."""
        pools = [
            self._subset(
                "sensory:text:any",
                self.c.sensory,
                128,
            ),
            self._subset(
                "sensory:text:mention-self",
                self.c.sensory,
                96,
            ),
            self._subset(
                "sensory:text:question",
                self.c.sensory,
                48,
            ),
            self._subset(
                "sensory:text:exclamation",
                self.c.sensory,
                48,
            ),
        ]
        return np.unique(
            np.concatenate(pools)
        ).astype(np.int32, copy=False)

    def _adaptive_speak_structure(
        self,
        output: np.ndarray,
        cues: np.ndarray,
    ) -> np.ndarray:
        """Measure real 1-2 hop FAFB reach from Discord sensory cues."""
        if not len(output) or not len(cues):
            return np.zeros(len(output), dtype=np.float32)

        absolute = self.c.matrix.copy()
        absolute.data = np.abs(absolute.data)
        cue_vector = np.zeros(
            self.c.n_neurons,
            dtype=np.float32,
        )
        cue_vector[cues] = 1.0
        hop1 = np.asarray(
            absolute.dot(cue_vector)
        ).ravel().astype(np.float32, copy=False)
        hop2 = np.asarray(
            absolute.dot(hop1)
        ).ravel().astype(np.float32, copy=False)
        return (
            0.35 * hop1[output]
            + hop2[output]
        ).astype(np.float32, copy=False)

    @staticmethod
    def _unit_scale(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if not len(values):
            return values
        lo = float(np.min(values))
        hi = float(np.max(values))
        if hi - lo <= 1e-9:
            return np.zeros_like(values)
        return (values - lo) / (hi - lo)

    def _refresh_adaptive_speak_pool(
        self,
        width: int = 128,
    ) -> None:
        """Build a learned communication circuit from activity and reward.

        Discord text is an artificial sensory modality, so there is no claim
        that one named fly neuron means "speak". Instead the adapter starts
        from stable text/social sensory populations, propagates through the
        real FAFB topology and lets reward/plasticity reshape which output
        neurons become the communication readout.
        """
        output = np.asarray(
            self.c.output,
            dtype=np.int32,
        )
        if not len(output):
            output = np.arange(
                self.c.n_neurons,
                dtype=np.int32,
            )
        width = min(max(1, int(width)), len(output))
        cues = self._build_adaptive_speak_cues()

        if (
            not len(self._adaptive_speak_structural_score)
            or len(self._adaptive_speak_structural_score) != len(output)
        ):
            self._adaptive_speak_structural_score = (
                self._adaptive_speak_structure(
                    output,
                    cues,
                )
            )
        structural = self._unit_scale(
            self._adaptive_speak_structural_score
        )

        state_cpu = self.compute.to_cpu(
            self.state
        ).astype(np.float32, copy=False)
        bias_cpu = self.compute.to_cpu(
            self.plastic_bias
        ).astype(np.float32, copy=False)
        activity = self._unit_scale(
            np.abs(state_cpu[output])
        )
        learned_bias = np.clip(
            bias_cpu[output]
            / max(1e-6, float(self.cfg.max_bias)),
            -1.0,
            1.0,
        ).astype(np.float32, copy=False)

        learned_reach = np.zeros(
            len(output),
            dtype=np.float32,
        )
        if self._synaptic_delta_map and len(cues):
            learned_abs = self._synaptic_matrix_cpu.copy()
            learned_abs.data = np.abs(learned_abs.data)
            cue_vector = np.zeros(
                self.c.n_neurons,
                dtype=np.float32,
            )
            cue_vector[cues] = 1.0
            learned_hop = np.asarray(
                learned_abs.dot(cue_vector)
            ).ravel().astype(np.float32, copy=False)
            learned_reach = self._unit_scale(
                learned_hop[output]
            )

        score = (
            1.00 * structural
            + 0.45 * activity
            + 0.70 * learned_bias
            + 0.85 * learned_reach
        )
        order = np.argsort(score)[::-1]
        pool = output[
            order[:width]
        ].astype(np.int32, copy=False)

        self._adaptive_speak_cues = cues
        self._action_output_seeds["speak"] = cues
        self._action_output_pools["speak"] = pool
        self._action_output_info["speak"] = {
            "mode": "adaptive-learned-connectome",
            "seed_count": int(len(cues)),
            "pool_size": int(len(pool)),
            "seed_terms": [
                "text:any",
                "mention:self",
                "question",
                "exclamation",
            ],
            "matched_terms": [],
            "seed_types": [
                {
                    "name": "Discord sensory adapter",
                    "count": int(len(cues)),
                }
            ],
            "seed_neurons": [],
            "external_seed_count": int(len(cues)),
            "seed_kind": "sensory-cues",
            "structural_mean": float(
                np.mean(structural) if len(structural) else 0.0
            ),
            "learned_reach_mean": float(
                np.mean(learned_reach)
                if len(learned_reach)
                else 0.0
            ),
        }

    def _seed_display_type(
        self,
        idx: int,
    ) -> str:
        """Return the cleanest available biological label for one neuron."""
        meta = self.c.neuron_meta or {}
        for key in (
            "primary_type",
            "cell_class",
            "sub_class",
            "super_class",
        ):
            arr = meta.get(key)
            if arr is None or len(arr) != self.c.n_neurons:
                continue
            value = str(arr[int(idx)]).strip()
            if value:
                return value
        return "untyped"

    def _action_seed_candidates(
        self,
        action: str,
        annotations: np.ndarray,
        output: np.ndarray,
        limit: int = 32,
    ) -> tuple[np.ndarray, list[str]]:
        """Find biological anchors, preferring output cells but scanning all.

        Older code searched only c.output. Some useful descending/courtship
        types can be present in neuron metadata without having landed in that
        broad runtime pool, so the second pass searches the full annotated
        connectome and lets real connectivity project them into output cells.
        """
        terms = tuple(
            str(term).lower()
            for term in self.ACTION_BIOLOGICAL_SEEDS.get(action, ())
        )
        if not terms:
            return np.empty(0, dtype=np.int32), []

        output_set = set(int(i) for i in output.tolist())
        ranked: list[tuple[int, int, int, int]] = []
        seen: set[int] = set()

        def rank_match(
            source_rank: int,
            idx: int,
            label: str,
        ) -> tuple[int, int, int, int] | None:
            hit_positions = [
                pos
                for pos, term in enumerate(terms)
                if term in label
            ]
            if not hit_positions:
                return None
            priority = min(hit_positions)
            specificity = max(
                len(terms[pos])
                for pos in hit_positions
            )
            return (
                source_rank,
                priority,
                -specificity,
                idx,
            )

        for idx_raw in output:
            idx = int(idx_raw)
            label = str(annotations[idx]).lower()
            match = rank_match(0, idx, label)
            if match is None:
                continue
            ranked.append(match)
            seen.add(idx)

        # Only broaden the search when the output pool had no biological
        # anchor at all. Existing working circuits must not be diluted by broad
        # whole-brain text matches such as "walking" or "turn".
        if not ranked:
            for idx in range(self.c.n_neurons):
                if idx in seen:
                    continue
                label = str(annotations[idx]).lower()
                match = rank_match(1, idx, label)
                if match is not None:
                    ranked.append(match)

        ranked.sort()
        chosen = np.asarray(
            [item[3] for item in ranked[: max(1, int(limit))]],
            dtype=np.int32,
        )
        matched_terms: list[str] = []
        for idx in chosen:
            label = str(annotations[int(idx)]).lower()
            for term in terms:
                if term in label and term not in matched_terms:
                    matched_terms.append(term)
        return chosen, matched_terms

    def _build_action_output_pools(self, width: int = 128) -> None:
        """Build action readouts from annotated neurons and real connectivity."""
        annotations = self._action_metadata_text()
        output = np.asarray(self.c.output, dtype=np.int32)
        if not len(output):
            output = np.arange(self.c.n_neurons, dtype=np.int32)

        self._action_output_pools = {}
        self._action_output_seeds = {}
        self._action_output_info = {}

        for action in self.ACTIONS:
            if action == "speak":
                self._refresh_adaptive_speak_pool(width)
                continue

            terms = self.ACTION_BIOLOGICAL_SEEDS.get(action, ())
            seeds, matched_terms = self._action_seed_candidates(
                action,
                annotations,
                output,
            )
            if len(seeds):
                pool = self._expand_action_seed_pool(
                    seeds,
                    width,
                )
                mode = (
                    "annotated+connectome"
                    if len(pool) > len(seeds)
                    else "annotated"
                )
            else:
                pool = self._structural_action_fallback(
                    action,
                    width,
                )
                mode = "structural-fallback"

            type_counts = Counter(
                self._seed_display_type(int(idx))
                for idx in seeds
            )
            seed_types = [
                {"name": name, "count": int(count)}
                for name, count in type_counts.most_common(10)
            ]
            seed_neurons = [
                {
                    "root_id": int(self.c.root_ids[int(idx)]),
                    "type": self._seed_display_type(int(idx)),
                    "in_output_pool": bool(
                        int(idx) in self._output_lookup
                    ),
                }
                for idx in seeds[:12]
            ]

            self._action_output_seeds[action] = seeds
            self._action_output_pools[action] = pool
            self._action_output_info[action] = {
                "mode": mode,
                "seed_count": int(len(seeds)),
                "pool_size": int(len(pool)),
                "seed_terms": list(terms),
                "matched_terms": matched_terms,
                "seed_types": seed_types,
                "seed_neurons": seed_neurons,
                "external_seed_count": int(
                    sum(
                        int(idx) not in self._output_lookup
                        for idx in seeds
                    )
                ),
            }

    def action_pool_diagnostics(self) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for action, info in self._action_output_info.items():
            row = {
                key: (
                    list(value)
                    if isinstance(value, tuple)
                    else value
                )
                for key, value in info.items()
            }
            pool = self._action_output_pools.get(
                action,
                np.empty(0, dtype=np.int32),
            )
            seeds = self._action_output_seeds.get(
                action,
                np.empty(0, dtype=np.int32),
            )
            if len(pool):
                idx = self._backend_indices(pool)
                values = self.compute.to_cpu(
                    self.state[idx]
                ).astype(np.float32, copy=False)
                row["mean_activation"] = float(
                    np.mean(values)
                )
                row["mean_abs_activation"] = float(
                    np.mean(np.abs(values))
                )
                row["max_abs_activation"] = float(
                    np.max(np.abs(values))
                )
            else:
                row["mean_activation"] = 0.0
                row["mean_abs_activation"] = 0.0
                row["max_abs_activation"] = 0.0

            seed_activity = []
            if len(seeds):
                idx = self._backend_indices(seeds)
                values = self.compute.to_cpu(
                    self.state[idx]
                ).astype(np.float32, copy=False)
                order = np.argsort(np.abs(values))[::-1][:8]
                for pos in order:
                    neuron_idx = int(seeds[int(pos)])
                    seed_activity.append({
                        "root_id": int(
                            self.c.root_ids[neuron_idx]
                        ),
                        "type": (
                            "text/social sensory cue"
                            if action == "speak"
                            else self._seed_display_type(
                                neuron_idx
                            )
                        ),
                        "activation": float(values[int(pos)]),
                        "in_output_pool": bool(
                            neuron_idx in self._output_lookup
                        ),
                    })
            row["top_seed_activity"] = seed_activity
            result[action] = row
        return result

    def _rebuild_synaptic_matrix(self) -> None:
        n = self.c.n_neurons
        if not self._synaptic_delta_map:
            cpu = scipy_sparse.csr_matrix(
                self.c.matrix.shape,
                dtype=np.float32,
            )
        else:
            items = list(self._synaptic_delta_map.items())
            keys = np.fromiter(
                (item[0] for item in items),
                dtype=np.int64,
                count=len(items),
            )
            values = np.fromiter(
                (item[1] for item in items),
                dtype=np.float32,
                count=len(items),
            )
            rows = (keys // n).astype(np.int32, copy=False)
            cols = (keys % n).astype(np.int32, copy=False)
            cpu = scipy_sparse.coo_matrix(
                (values, (rows, cols)),
                shape=self.c.matrix.shape,
                dtype=np.float32,
            ).tocsr()
            cpu.eliminate_zeros()

        self._synaptic_matrix_cpu = cpu
        self.synaptic_matrix = self.compute.sparse_from_scipy(cpu)

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
        if (
            "synaptic_post" in data
            and "synaptic_pre" in data
            and "synaptic_delta" in data
        ):
            post = np.asarray(data["synaptic_post"], dtype=np.int64)
            pre = np.asarray(data["synaptic_pre"], dtype=np.int64)
            delta = np.asarray(data["synaptic_delta"], dtype=np.float32)
            valid = (
                (post >= 0)
                & (post < self.c.n_neurons)
                & (pre >= 0)
                & (pre < self.c.n_neurons)
                & np.isfinite(delta)
                & (np.abs(delta) > 1e-9)
            )
            post = post[valid]
            pre = pre[valid]
            delta = delta[valid]
            max_edges = max(
                0,
                int(self.cfg.synaptic_plasticity_max_edges),
            )
            if max_edges and len(delta) > max_edges:
                order = np.argsort(np.abs(delta))[::-1][:max_edges]
                post = post[order]
                pre = pre[order]
                delta = delta[order]
            n = self.c.n_neurons
            self._synaptic_delta_map = {
                int(r) * n + int(col): float(value)
                for r, col, value in zip(post, pre, delta)
            }
            self._rebuild_synaptic_matrix()
        if "reward_trace" in data:
            self.reward_trace = float(data["reward_trace"])
        if "tick_count" in data:
            self.tick_count = int(data["tick_count"])

    def save(self) -> None:
        p = self.cfg.state_file
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.npz")
        syn_items = list(self._synaptic_delta_map.items())
        if syn_items:
            syn_keys = np.fromiter(
                (item[0] for item in syn_items),
                dtype=np.int64,
                count=len(syn_items),
            )
            syn_delta = np.fromiter(
                (item[1] for item in syn_items),
                dtype=np.float32,
                count=len(syn_items),
            )
            syn_post = (
                syn_keys // self.c.n_neurons
            ).astype(np.int32, copy=False)
            syn_pre = (
                syn_keys % self.c.n_neurons
            ).astype(np.int32, copy=False)
        else:
            syn_post = np.empty(0, dtype=np.int32)
            syn_pre = np.empty(0, dtype=np.int32)
            syn_delta = np.empty(0, dtype=np.float32)

        np.savez_compressed(
            tmp,
            state=self.compute.to_cpu(self.state).astype(np.float16),
            eligibility=self.compute.to_cpu(self.eligibility).astype(np.float16),
            plastic_bias=self.compute.to_cpu(self.plastic_bias).astype(np.float16),
            synaptic_post=syn_post,
            synaptic_pre=syn_pre,
            synaptic_delta=syn_delta.astype(np.float32, copy=False),
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

    @staticmethod
    def _normalize_reach(values: np.ndarray) -> np.ndarray:
        values = np.asarray(
            values,
            dtype=np.float32,
        )
        if not len(values):
            return values
        peak = float(np.max(values))
        if peak <= 1e-12:
            return np.zeros_like(values)
        return values / np.float32(peak)

    def _action_guided_sensory_pool(
        self,
        action: str,
        key: str,
        *,
        width: int = 192,
        hops: int = 3,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Choose sensory neurons by real structural reach to an action pool.

        Matrix orientation is [post, pre], so reverse propagation through
        abs(matrix).T estimates which sensory cells can reach the selected
        output population in a small number of real FAFB hops.
        """
        action = str(action or "").strip()
        key = str(key or "").strip()
        width = max(
            8,
            min(int(width), max(8, len(self.c.sensory))),
        )
        hops = max(1, min(4, int(hops)))
        cache_key = (action, key, width, hops)
        cached = self._guided_sensory_cache.get(cache_key)
        if cached is not None:
            return cached

        sensory = np.asarray(
            self.c.sensory,
            dtype=np.int32,
        )
        output = np.asarray(
            self._action_output_pools.get(
                action,
                np.empty(0, dtype=np.int32),
            ),
            dtype=np.int32,
        )
        if not len(sensory) or not len(output):
            fallback = self._subset(
                f"guided-fallback:{action}:{key}",
                sensory,
                width,
            ).astype(np.int32, copy=False)
            weights = np.ones(
                len(fallback),
                dtype=np.float32,
            )
            diag = {
                "action": action,
                "key": key,
                "mode": "fallback-random-sensory",
                "neurons": int(len(fallback)),
                "hops": hops,
                "reach_mean": 0.0,
                "reach_max": 0.0,
            }
            result = (fallback, weights, diag)
            self._guided_sensory_cache[cache_key] = result
            return result

        absolute_reverse = (
            self._absolute_connectome()
            .transpose()
            .tocsr()
        )
        signed_reverse = (
            self.c.matrix
            .transpose()
            .tocsr()
        )
        structural_frontier = np.zeros(
            self.c.n_neurons,
            dtype=np.float32,
        )
        signed_frontier = np.zeros_like(
            structural_frontier
        )
        target_value = np.float32(
            1.0 / max(1, len(output))
        )
        structural_frontier[output] = target_value
        signed_frontier[output] = target_value
        structural_combined = np.zeros_like(
            structural_frontier
        )
        signed_combined = np.zeros_like(
            signed_frontier
        )
        hop_weights = (0.20, 0.45, 1.0, 0.55)
        for hop in range(hops):
            structural_frontier = np.asarray(
                absolute_reverse.dot(
                    structural_frontier
                )
            ).ravel().astype(
                np.float32,
                copy=False,
            )
            structural_frontier = (
                self._normalize_reach(
                    structural_frontier
                )
            )

            signed_frontier = np.asarray(
                signed_reverse.dot(
                    signed_frontier
                )
            ).ravel().astype(
                np.float32,
                copy=False,
            )
            signed_peak = float(
                np.max(
                    np.abs(signed_frontier)
                )
            )
            if signed_peak > 1e-12:
                signed_frontier = (
                    signed_frontier
                    / np.float32(signed_peak)
                )
            else:
                signed_frontier = (
                    np.zeros_like(
                        signed_frontier
                    )
                )

            weight = np.float32(
                hop_weights[hop]
            )
            structural_combined += (
                weight * structural_frontier
            )
            signed_combined += (
                weight * signed_frontier
            )

        structural_scores = (
            structural_combined[sensory]
        )
        signed_scores = signed_combined[sensory]
        positive = np.flatnonzero(
            signed_scores > 1e-6
        )
        mode = "connectome-guided-excitatory-sensory"
        if len(positive):
            sensory_scores = (
                0.72 * signed_scores
                + 0.28 * structural_scores
            ).astype(np.float32, copy=False)
        else:
            # Some prepared matrices may not carry transmitter sign. In that
            # case retain the topologically guided route rather than reverting
            # immediately to random sensory cells.
            positive = np.flatnonzero(
                structural_scores > 1e-9
            )
            sensory_scores = structural_scores
            mode = "connectome-guided-structural-sensory"

        if not len(positive):
            fallback = self._subset(
                f"guided-zero:{action}:{key}",
                sensory,
                width,
            ).astype(np.int32, copy=False)
            weights = np.ones(
                len(fallback),
                dtype=np.float32,
            )
            diag = {
                "action": action,
                "key": key,
                "mode": "fallback-zero-reach",
                "neurons": int(len(fallback)),
                "hops": hops,
                "reach_mean": 0.0,
                "reach_max": 0.0,
            }
            result = (fallback, weights, diag)
            self._guided_sensory_cache[cache_key] = result
            return result

        candidate_scores = sensory_scores[positive].astype(
            np.float32,
            copy=True,
        )
        # Tiny stable jitter keeps channel-specific cues from collapsing onto
        # an identical set when many sensory cells have equal structural reach.
        rng = np.random.default_rng(
            self._stable_seed(
                f"guided:{action}:{key}:{hops}"
            )
        )
        candidate_scores *= rng.uniform(
            0.985,
            1.015,
            size=len(candidate_scores),
        ).astype(np.float32)
        k = min(width, len(positive))
        if k >= len(positive):
            local_order = np.argsort(
                candidate_scores
            )[::-1]
        else:
            part = np.argpartition(
                candidate_scores,
                -k,
            )[-k:]
            local_order = part[
                np.argsort(candidate_scores[part])[::-1]
            ]
        chosen_local = positive[
            local_order[:k]
        ]
        chosen = sensory[
            chosen_local
        ].astype(np.int32, copy=False)
        reach = sensory_scores[
            chosen_local
        ].astype(np.float32, copy=False)
        peak = max(
            1e-9,
            float(np.max(reach)),
        )
        weights = (
            0.55
            + 0.90 * np.clip(
                reach / peak,
                0.0,
                1.0,
            )
        ).astype(np.float32, copy=False)
        diag = {
            "action": action,
            "key": key,
            "mode": mode,
            "neurons": int(len(chosen)),
            "hops": hops,
            "reach_mean": float(
                np.mean(reach)
                if len(reach)
                else 0.0
            ),
            "reach_max": float(
                np.max(reach)
                if len(reach)
                else 0.0
            ),
            "positive_signed_candidates": int(
                np.count_nonzero(
                    signed_scores > 1e-6
                )
            ),
        }
        result = (chosen, weights, diag)
        self._guided_sensory_cache[cache_key] = result
        return result

    def inject_action_guided_sensory(
        self,
        action: str,
        key: str,
        magnitude: float,
        *,
        width: int = 192,
        hops: int = 3,
    ) -> dict:
        """Inject only sensory cells structurally able to reach an action."""
        magnitude = max(
            0.0,
            min(4.0, float(magnitude)),
        )
        pool, reach_weights, diag = (
            self._action_guided_sensory_pool(
                action,
                key,
                width=width,
                hops=hops,
            )
        )
        if magnitude > 0.0 and len(pool):
            idx = self._backend_indices(pool)
            jitter = self.compute.random_uniform(
                0.88,
                1.12,
                size=len(pool),
            )
            if self.compute.is_gpu:
                structural = self.compute.asarray(
                    reach_weights,
                    dtype=np.float32,
                )
            else:
                structural = reach_weights
            self.state[idx] += (
                np.float32(magnitude)
                * structural
                * jitter
            )
            self.xp.clip(
                self.state,
                -3.0,
                3.0,
                out=self.state,
            )
            self.eligibility[idx] = self.xp.maximum(
                self.eligibility[idx],
                np.float32(
                    min(1.0, 0.25 + 0.45 * magnitude)
                ),
            )
        diag = dict(diag)
        diag["magnitude"] = magnitude
        self._last_guided_cue = diag
        return diag

    def _absolute_connectome(self) -> scipy_sparse.csr_matrix:
        if self._absolute_connectome_cpu is None:
            matrix = self.c.matrix.copy().tocsr()
            matrix.data = np.abs(
                matrix.data
            ).astype(np.float32, copy=False)
            self._absolute_connectome_cpu = matrix
        return self._absolute_connectome_cpu

    def user_identity_pool(
        self,
        user_id: int,
        width: int = 96,
    ) -> np.ndarray:
        """Stable neural identity for one Discord user."""
        user_id = int(user_id)
        cached = self._user_identity_cache.get(user_id)
        if cached is not None and len(cached):
            return cached
        pool = self._subset(
            f"social:user-identity:{user_id}",
            self.c.sensory,
            width,
        ).astype(np.int32, copy=False)
        self._user_identity_cache[user_id] = pool
        return pool

    def user_memory_pool(
        self,
        user_id: int,
        width: int = 128,
    ) -> np.ndarray:
        """Build a stable memory assembly from real FAFB connectivity."""
        user_id = int(user_id)
        cached = self._user_memory_cache.get(user_id)
        if cached is not None and len(cached):
            return cached

        identity = self.user_identity_pool(user_id)
        width = max(
            32,
            min(int(width), self.c.n_neurons),
        )
        absolute = self._absolute_connectome()
        cue = np.zeros(
            self.c.n_neurons,
            dtype=np.float32,
        )
        cue[identity] = 1.0
        hop1 = np.asarray(
            absolute.dot(cue)
        ).ravel().astype(np.float32, copy=False)
        hop2 = np.asarray(
            absolute.dot(hop1)
        ).ravel().astype(np.float32, copy=False)
        score = (
            0.55 * hop1
            + hop2
        ).astype(np.float32, copy=False)
        score[identity] = 0.0

        if np.any(score > 0.0):
            k = min(width, int(np.count_nonzero(score > 0.0)))
            part = np.argpartition(score, -k)[-k:]
            order = part[
                np.argsort(score[part])[::-1]
            ]
            pool = order.astype(np.int32, copy=False)
        else:
            # This should be rare; keep the representation deterministic if a
            # demo/sparse connectome has no outgoing path from the identity.
            pool = self._subset(
                f"social:user-memory:{user_id}",
                np.arange(
                    self.c.n_neurons,
                    dtype=np.int32,
                ),
                width,
            ).astype(np.int32, copy=False)

        self._user_memory_cache[user_id] = pool
        return pool

    def activate_user_memory(
        self,
        user_id: int,
        magnitude: float = 0.45,
    ) -> None:
        """Reactivate a user's stable identity and its learned assembly."""
        user_id = int(user_id)
        magnitude = max(0.0, min(2.0, float(magnitude)))
        identity = self.user_identity_pool(user_id)
        memory = self.user_memory_pool(user_id)

        identity_idx = self._backend_indices(identity)
        memory_idx = self._backend_indices(memory)
        self.state[identity_idx] += np.float32(magnitude)
        # A weak recall term lets persistent plasticity reappear immediately;
        # subsequent ticks still propagate through the real connectome.
        self.state[memory_idx] += np.float32(
            0.16 * magnitude
        )
        self.eligibility[identity_idx] = self.xp.maximum(
            self.eligibility[identity_idx],
            np.float32(0.72),
        )
        self.eligibility[memory_idx] = self.xp.maximum(
            self.eligibility[memory_idx],
            np.float32(0.36),
        )
        self.xp.clip(
            self.state,
            -3.0,
            3.0,
            out=self.state,
        )
        self._user_memory_last_activation[user_id] = time.time()

    def reinforce_user_memory(
        self,
        user_id: int,
        amount: float,
    ) -> dict:
        """Write social valence into a user's neural assembly.

        Short-term valence is encoded in neuronal bias. Long-term memory is a
        sparse learned-synapse overlay restricted to the identity/memory
        assembly, so it survives restarts through brain_state.npz.
        """
        user_id = int(user_id)
        amount = max(-1.0, min(1.0, float(amount)))
        if abs(amount) <= 1e-12:
            return self.user_memory_diagnostics(user_id)

        identity = self.user_identity_pool(user_id)
        memory = self.user_memory_pool(user_id)
        self.activate_user_memory(
            user_id,
            0.28 + min(0.72, abs(amount) * 2.0),
        )

        identity_idx = self._backend_indices(identity)
        memory_idx = self._backend_indices(memory)
        bias_delta = np.float32(0.12 * amount)
        identity_delta = np.float32(0.035 * amount)
        self.plastic_bias[memory_idx] += bias_delta
        self.plastic_bias[identity_idx] += identity_delta
        self.xp.clip(
            self.plastic_bias,
            -self.cfg.max_bias,
            self.cfg.max_bias,
            out=self.plastic_bias,
        )

        combined = np.unique(
            np.concatenate([identity, memory])
        ).astype(np.int32, copy=False)
        eligibility = np.full(
            len(combined),
            0.62,
            dtype=np.float32,
        )
        identity_set = set(
            int(i) for i in identity.tolist()
        )
        for pos, idx in enumerate(combined):
            if int(idx) in identity_set:
                eligibility[pos] = 0.90

        # Small positive contacts should still accumulate; strong rejection
        # should be memorable after one event, but remains bounded.
        synaptic_amount = math.copysign(
            min(
                1.0,
                0.22 + abs(amount) * 6.5,
            ),
            amount,
        )
        self._reinforce_synapses(
            synaptic_amount,
            (combined, eligibility),
        )
        return self.user_memory_diagnostics(user_id)

    def user_memory_diagnostics(
        self,
        user_id: int,
    ) -> dict:
        user_id = int(user_id)
        identity = self.user_identity_pool(user_id)
        memory = self.user_memory_pool(user_id)
        identity_idx = self._backend_indices(identity)
        memory_idx = self._backend_indices(memory)

        identity_activity = self.compute.to_cpu(
            self.state[identity_idx]
        ).astype(np.float32, copy=False)
        memory_activity = self.compute.to_cpu(
            self.state[memory_idx]
        ).astype(np.float32, copy=False)
        memory_bias = self.compute.to_cpu(
            self.plastic_bias[memory_idx]
        ).astype(np.float32, copy=False)

        mean_abs_bias = float(
            np.mean(np.abs(memory_bias))
            if len(memory_bias)
            else 0.0
        )
        bias_sum_abs = float(
            np.sum(np.abs(memory_bias))
        )
        bias_valence = (
            float(np.sum(memory_bias)) / bias_sum_abs
            if bias_sum_abs > 1e-9
            else 0.0
        )
        bias_strength = min(
            1.0,
            mean_abs_bias / 0.040,
        )

        combined = np.unique(
            np.concatenate([identity, memory])
        ).astype(np.int32, copy=False)
        learned = self._synaptic_matrix_cpu[
            combined
        ][:, combined].tocsr()
        base = self.c.matrix[
            combined
        ][:, combined].tocsr()
        sign_base = base.copy()
        if sign_base.nnz:
            sign_base.data = np.sign(
                sign_base.data
            ).astype(np.float32, copy=False)
        aligned = learned.multiply(sign_base).tocoo()
        learned_count = int(learned.nnz)

        if aligned.nnz:
            aligned_values = np.asarray(
                aligned.data,
                dtype=np.float32,
            )
            aligned_abs = float(
                np.sum(np.abs(aligned_values))
            )
            synaptic_valence = (
                float(np.sum(aligned_values))
                / aligned_abs
                if aligned_abs > 1e-12
                else 0.0
            )
            mean_abs_delta = float(
                np.mean(np.abs(aligned_values))
            )
        else:
            synaptic_valence = 0.0
            mean_abs_delta = 0.0

        synaptic_strength = min(
            1.0,
            learned_count / 512.0
            + mean_abs_delta / 0.010,
        )
        evidence = (
            bias_strength + synaptic_strength
        )
        if evidence > 1e-9:
            neural_affinity = (
                bias_valence * bias_strength
                + synaptic_valence * synaptic_strength
            ) / evidence
        else:
            neural_affinity = 0.0
        neural_affinity = max(
            -1.0,
            min(1.0, float(neural_affinity)),
        )
        maturity = max(
            0.0,
            min(
                1.0,
                0.35 * bias_strength
                + 0.65 * synaptic_strength,
            ),
        )

        top_edges: list[dict] = []
        if learned.nnz:
            coo = learned.tocoo()
            order = np.argsort(
                np.abs(coo.data)
            )[::-1][:5]
            for pos in order:
                local_target = int(coo.row[int(pos)])
                local_source = int(coo.col[int(pos)])
                target_idx = int(combined[local_target])
                source_idx = int(combined[local_source])
                top_edges.append({
                    "source": int(
                        self.c.root_ids[source_idx]
                    ),
                    "target": int(
                        self.c.root_ids[target_idx]
                    ),
                    "delta": float(
                        coo.data[int(pos)]
                    ),
                })

        last = self._user_memory_last_activation.get(
            user_id
        )
        return {
            "user_id": user_id,
            "neural_affinity": neural_affinity,
            "maturity": maturity,
            "identity_neurons": int(len(identity)),
            "memory_neurons": int(len(memory)),
            "learned_synapses": learned_count,
            "mean_abs_synaptic_delta": mean_abs_delta,
            "mean_abs_bias": mean_abs_bias,
            "identity_activity": float(
                np.mean(np.abs(identity_activity))
                if len(identity_activity)
                else 0.0
            ),
            "memory_activity": float(
                np.mean(np.abs(memory_activity))
                if len(memory_activity)
                else 0.0
            ),
            "last_activation_age": (
                max(0.0, time.time() - last)
                if last is not None
                else None
            ),
            "top_edges": top_edges,
        }

    def inject_text(self, text: str, author_id: int, mentioned: bool) -> None:
        stripped = text.strip()
        if not stripped:
            return
        mag = min(1.8, 0.25 + len(stripped) / 180.0)
        self.inject("text:any", mag, 128)
        self.activate_user_memory(
            author_id,
            0.45 + 0.20 * mentioned,
        )
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
            self.activate_user_memory(uid, 0.08)

    def inject_voice_reward_opportunity(
        self,
        guild_id: int,
        channel_id: int,
        *,
        human_count: int,
        strength: float,
    ) -> dict:
        """Present a possible reward through sensory cells that reach JOIN."""
        strength = max(0.0, min(4.0, float(strength)))
        human_count = max(0, int(human_count))
        if strength <= 0.0:
            return {
                "mode": "inactive",
                "neurons": 0,
                "reach_mean": 0.0,
                "reach_max": 0.0,
                "magnitude": 0.0,
            }

        guided = self.inject_action_guided_sensory(
            "voice_join",
            (
                f"reward-opportunity:{guild_id}:"
                f"{channel_id}"
            ),
            strength,
            width=224,
            hops=3,
        )
        # Generic context still carries "people/reward exists" information,
        # but the strongest motivational component is selected by actual
        # structural reach to the voice_join readout.
        self.inject(
            (
                "voice:reward-opportunity:people:"
                f"{min(8, human_count)}"
            ),
            0.30 + 0.08 * min(8, human_count),
            72,
        )
        return guided

    def step(self, ticks: int = 1) -> None:
        for _ in range(max(1, ticks)):
            leak, propagation_gain, noise_sigma = (
                self._neuromodulation_tick_effects()
            )
            propagated = self.matrix.dot(self.state).astype(
                self.xp.float32,
                copy=False,
            )
            if self._synaptic_delta_map:
                propagated = propagated + self.synaptic_matrix.dot(
                    self.state
                ).astype(self.xp.float32, copy=False)
            noise = self.compute.random_normal(
                0.0,
                noise_sigma,
                size=self.state.shape,
            )
            x = (
                np.float32(leak) * self.state
                + np.float32(propagation_gain) * propagated
                + self.plastic_bias
                + noise
            )
            self.state[...] = self.xp.tanh(x)
            self.eligibility[...] = (
                0.94 * self.eligibility
                + 0.06 * self.xp.abs(self.state)
            )
            self.plastic_bias *= np.float32(
                self.cfg.plasticity_decay
            )
            self.reward_trace *= 0.96
            self.tick_count += 1

    def mark_language_output(self, text: str) -> None:
        """Leave an eligibility trace for the words Mucha is about to send.

        This is an internal efference copy, not a new sensory message. It makes
        later Discord reward target the actual generated words and adjacent
        word pairs as well as the context that produced the reply.
        """
        words = re.findall(
            r"[^\W_]{3,}(?:['’][^\W_]+)?",
            str(text or "").lower(),
            flags=re.UNICODE,
        )[:18]
        if not words:
            return

        seen: set[str] = set()
        for word in words:
            if word in seen:
                continue
            seen.add(word)
            idx_cpu = self._subset(
                "output:language:word:" + word,
                self.c.output,
                32,
            )
            idx = self._backend_indices(idx_cpu)
            self.state[idx] += np.float32(0.10)
            self.eligibility[idx] = self.xp.maximum(
                self.eligibility[idx],
                np.float32(0.55),
            )

        for a, b in zip(words, words[1:]):
            idx_cpu = self._subset(
                f"sensory:text:pair:{a}|{b}",
                self.c.sensory,
                24,
            )
            idx = self._backend_indices(idx_cpu)
            self.state[idx] += np.float32(0.06)
            self.eligibility[idx] = self.xp.maximum(
                self.eligibility[idx],
                np.float32(0.42),
            )

        self.xp.clip(self.state, -3.0, 3.0, out=self.state)

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

    def _reinforce_synapses(
        self,
        amount: float,
        trace: tuple[np.ndarray, np.ndarray] | None,
    ) -> dict:
        """Reinforce real FAFB edges that participated in the recent trace.

        The static connectome is never overwritten. Learning is stored as a
        sparse overlay on existing edges, so positive reward strengthens the
        original excitatory/inhibitory direction and punishment weakens it
        without flipping the biological sign of the connection.
        """
        empty = {
            "changed": 0,
            "total": len(self._synaptic_delta_map),
            "mean_delta": 0.0,
            "max_delta": 0.0,
        }
        if (
            not self.cfg.synaptic_plasticity_enabled
            or abs(float(amount)) <= 1e-12
            or self.cfg.synaptic_plasticity_lr <= 0.0
        ):
            return empty

        trace_limit = max(
            32,
            min(
                int(self.cfg.synaptic_plasticity_trace_neurons),
                self.c.n_neurons,
            ),
        )
        if trace is None or not len(trace[0]):
            idx_cpu, elig_cpu = self.capture_learning_trace(trace_limit)
        else:
            idx_cpu = np.asarray(trace[0], dtype=np.int32)
            elig_cpu = np.asarray(trace[1], dtype=np.float32)
            if len(idx_cpu) > trace_limit:
                order = np.argsort(np.abs(elig_cpu))[::-1][:trace_limit]
                idx_cpu = idx_cpu[order]
                elig_cpu = elig_cpu[order]

        if len(idx_cpu) < 2:
            return empty

        sub = self.c.matrix[idx_cpu][:, idx_cpu].tocoo()
        if sub.nnz == 0:
            return empty

        base = np.asarray(sub.data, dtype=np.float32)
        post_elig = np.abs(elig_cpu[sub.row])
        pre_elig = np.abs(elig_cpu[sub.col])
        structural = np.sqrt(np.clip(np.abs(base), 0.0, 1.0))
        edge_elig = np.sqrt(post_elig * pre_elig) * structural

        nonzero = np.flatnonzero(edge_elig > 1e-7)
        if not len(nonzero):
            return empty

        per_event_limit = min(4096, len(nonzero))
        if len(nonzero) > per_event_limit:
            values = edge_elig[nonzero]
            top = np.argpartition(
                values,
                -per_event_limit,
            )[-per_event_limit:]
            chosen = nonzero[top]
        else:
            chosen = nonzero

        lr = max(
            0.0,
            float(self.cfg.synaptic_plasticity_lr)
            * self._plasticity_gain(),
        )
        max_delta = max(
            1e-6,
            float(self.cfg.synaptic_plasticity_max_delta),
        )
        n = self.c.n_neurons
        applied: list[float] = []

        for pos in chosen:
            post = int(idx_cpu[int(sub.row[pos])])
            pre = int(idx_cpu[int(sub.col[pos])])
            base_weight = float(base[pos])
            if abs(base_weight) <= 1e-12:
                continue

            key = post * n + pre
            old = float(self._synaptic_delta_map.get(key, 0.0))
            step_delta = (
                lr
                * float(amount)
                * float(edge_elig[pos])
                * (1.0 if base_weight > 0.0 else -1.0)
            )
            new = old + step_delta

            # Punishment may weaken an edge, but never reverse its biological
            # excitatory/inhibitory sign.
            if base_weight > 0.0:
                new = min(
                    max_delta,
                    max(-0.90 * base_weight, new),
                )
            else:
                new = max(
                    -max_delta,
                    min(0.90 * abs(base_weight), new),
                )

            if abs(new) <= 1e-9:
                self._synaptic_delta_map.pop(key, None)
            else:
                self._synaptic_delta_map[key] = float(new)
            applied.append(float(new - old))

        max_edges = max(
            0,
            int(self.cfg.synaptic_plasticity_max_edges),
        )
        if max_edges and len(self._synaptic_delta_map) > max_edges:
            strongest = sorted(
                self._synaptic_delta_map.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:max_edges]
            self._synaptic_delta_map = dict(strongest)

        if applied:
            self._rebuild_synaptic_matrix()

        if not applied:
            return empty

        arr = np.asarray(applied, dtype=np.float32)
        return {
            "changed": int(np.count_nonzero(np.abs(arr) > 1e-12)),
            "total": len(self._synaptic_delta_map),
            "mean_delta": float(np.mean(arr)),
            "max_delta": float(np.max(np.abs(arr))),
        }

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

        if self.cfg.neuromodulation_enabled:
            # Reward valence still comes from signed reward_trace. Monoamine
            # pools control learning/arousal globally; this is an engineering
            # adapter over biological transmitter classes, not a claim that a
            # single fly monoamine universally encodes one valence.
            magnitude = np.float32(0.35 * abs(amount))
            da_pool = self._modulator_pools["dopamine"]
            if len(da_pool):
                da = self._backend_indices(da_pool)
                self.state[da] += magnitude
            if amount < 0.0:
                oct_pool = self._modulator_pools["octopamine"]
                if len(oct_pool):
                    oct_idx = self._backend_indices(oct_pool)
                    self.state[oct_idx] += np.float32(
                        0.25 * abs(amount)
                    )
            ser_pool = self._modulator_pools["serotonin"]
            if len(ser_pool):
                ser = self._backend_indices(ser_pool)
                self.state[ser] += np.float32(
                    0.08 * abs(amount)
                )
            # Update the slow global state before applying this reward so
            # dopamine can modulate the same learning episode.
            self._update_neuromodulator_state()
            self._neuromodulator_effects["plasticity_gain"] = (
                self._plasticity_gain()
            )
        elif len(self.c.modulatory):
            idx_cpu = self._subset(
                "reward:modulatory",
                self.c.modulatory,
                min(128, len(self.c.modulatory)),
            )
            idx = self._backend_indices(idx_cpu)
            self.state[idx] += np.float32(0.35 * amount)

        changed_idx_cpu: np.ndarray
        changed_delta_cpu: np.ndarray

        if trace is not None and len(trace[0]):
            idx_cpu, elig_cpu = trace
            idx = self._backend_indices(idx_cpu)
            elig = self.compute.asarray(elig_cpu, dtype=np.float32)
            delta = (
                self.cfg.plasticity_lr
                * self._plasticity_gain()
                * amount
                * elig
            ).astype(self.xp.float32)
            self.plastic_bias[idx] += delta
            changed_idx_cpu = idx_cpu.astype(np.int32, copy=False)
            changed_delta_cpu = self.compute.to_cpu(delta).astype(np.float32, copy=False)
        else:
            delta = (
                self.cfg.plasticity_lr
                * self._plasticity_gain()
                * amount
                * self.eligibility
            ).astype(self.xp.float32)
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
            out_cpu = self._action_output_pools.get(action)
            if out_cpu is None or not len(out_cpu):
                out_cpu = self._structural_action_fallback(action, 128)
            out = self._backend_indices(out_cpu)
            action_delta = np.float32(
                self.cfg.plasticity_lr
                * self._plasticity_gain()
                * amount
                * 8.0
            )
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

        synaptic_learning = self._reinforce_synapses(
            amount,
            trace,
        )
        if action == "speak":
            self._refresh_adaptive_speak_pool()

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
            "changed_synapses": synaptic_learning["changed"],
            "learned_synapses": synaptic_learning["total"],
            "synaptic_mean_delta": synaptic_learning["mean_delta"],
            "synaptic_max_delta": synaptic_learning["max_delta"],
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
        """Return the live connectome preference for one learned word.

        Current neuronal activity remains the main signal. Learned plastic bias
        is included as a smaller term so reward can immediately alter future
        word choice instead of waiting for many unrelated ticks.
        """
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
        sensory_bias = self.compute.scalar(
            self.xp.mean(self.plastic_bias[sensory])
        )
        output_bias = self.compute.scalar(
            self.xp.mean(self.plastic_bias[output])
        )
        activity_raw = 0.35 * sensory_raw + 0.65 * output_raw
        learned_raw = 0.35 * sensory_bias + 0.65 * output_bias
        return _sigmoid(
            3.0 * activity_raw
            + 2.0 * learned_raw
            + 0.06 * self.reward_trace
        )

    def advance_language_word(
        self,
        token: str,
        previous_token: str | None = None,
        magnitude: float = 0.18,
        steps: int = 2,
    ) -> None:
        """Feed one chosen word back through the live connectome.

        The selected output population becomes an efference copy, then the
        actual FAFB connection matrix is stepped before the next word is
        scored. This makes sentence generation recurrent: every chosen word
        changes the brain state used to choose the following word.
        """
        token = str(token or "").strip().lower()
        if not token:
            return

        magnitude = max(0.0, min(1.0, float(magnitude)))
        steps = max(1, min(8, int(steps)))
        output_cpu = self._subset(
            "output:language:word:" + token,
            self.c.output,
            32,
        )
        sensory_cpu = self._subset(
            "sensory:text:word:" + token,
            self.c.sensory,
            24,
        )
        output = self._backend_indices(output_cpu)
        sensory = self._backend_indices(sensory_cpu)

        self.state[output] += np.float32(magnitude)
        self.state[sensory] += np.float32(magnitude * 0.30)
        self.eligibility[output] = self.xp.maximum(
            self.eligibility[output],
            np.float32(0.62),
        )
        self.eligibility[sensory] = self.xp.maximum(
            self.eligibility[sensory],
            np.float32(0.36),
        )

        previous = str(previous_token or "").strip().lower()
        if previous:
            pair_cpu = self._subset(
                f"sensory:text:pair:{previous}|{token}",
                self.c.sensory,
                24,
            )
            pair = self._backend_indices(pair_cpu)
            self.state[pair] += np.float32(magnitude * 0.45)
            self.eligibility[pair] = self.xp.maximum(
                self.eligibility[pair],
                np.float32(0.48),
            )

        self.xp.clip(self.state, -3.0, 3.0, out=self.state)
        self.step(steps)

    def _word_association_signature(
        self,
        token: str,
        width: int = 32,
        target_count: int = 512,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a cached one-hop connectome signature for a learned word."""
        token = str(token).strip().lower()
        cached = self._word_association_signature_cache.get(token)
        if cached is not None:
            return cached

        seed_cpu = self._subset(
            "sensory:text:word:" + token,
            self.c.sensory,
            width,
        )
        if not len(seed_cpu):
            empty = (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
            )
            self._word_association_signature_cache[token] = empty
            return empty

        spread = self.c.matrix[:, seed_cpu]
        strength = np.asarray(
            np.abs(spread).sum(axis=1)
        ).reshape(-1).astype(np.float32, copy=False)
        nonzero = np.flatnonzero(strength > 0)
        if not len(nonzero):
            empty = (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
            )
            self._word_association_signature_cache[token] = empty
            return empty

        k = min(max(32, int(target_count)), len(nonzero))
        values = strength[nonzero]
        if k >= len(nonzero):
            order = np.argsort(values)[::-1]
        else:
            part = np.argpartition(values, -k)[-k:]
            order = part[np.argsort(values[part])[::-1]]

        idx = nonzero[order[:k]].astype(np.int32, copy=False)
        val = strength[idx].astype(np.float32, copy=False)
        norm = float(np.linalg.norm(val))
        if norm > 1e-12:
            val = (val / norm).astype(np.float32, copy=False)

        if len(self._word_association_signature_cache) >= 512:
            self._word_association_signature_cache.clear()
        result = (idx, val)
        self._word_association_signature_cache[token] = result
        return result

    def word_association_snapshot(
        self,
        words: Iterable[str],
        max_nodes: int = 28,
        edge_limit: int = 70,
    ) -> dict:
        """Build a word-association graph directly from the live connectome.

        Words use their deterministic sensory populations. Structural affinity
        comes from overlap between their one-hop propagation signatures.
        Pair-specific sensory populations add live activity and learned
        plastic-bias terms, so Discord reward can strengthen or weaken edges
        without maintaining a separate association database.
        """
        max_nodes = max(4, min(40, int(max_nodes)))
        edge_limit = max(8, min(160, int(edge_limit)))
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in words:
            token = str(raw or "").strip().lower()
            if (
                len(token) < 3
                or token in seen
                or re.fullmatch(
                    r"[^\W_]+(?:['’][^\W_]+)?",
                    token,
                    flags=re.UNICODE,
                )
                is None
            ):
                continue
            seen.add(token)
            cleaned.append(token)
            if len(cleaned) >= max_nodes:
                break

        state_cpu = self.compute.to_cpu(self.state).astype(
            np.float32,
            copy=False,
        )
        bias_cpu = self.compute.to_cpu(self.plastic_bias).astype(
            np.float32,
            copy=False,
        )
        max_bias = max(1e-6, float(self.cfg.max_bias))

        nodes: list[dict] = []
        signatures: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for token in cleaned:
            sensory_idx = self._subset(
                "sensory:text:word:" + token,
                self.c.sensory,
                32,
            )
            output_idx = self._subset(
                "output:language:word:" + token,
                self.c.output,
                32,
            )
            idx = np.unique(
                np.concatenate([sensory_idx, output_idx])
            ).astype(np.int32, copy=False)
            activation = (
                float(np.mean(np.abs(state_cpu[idx])))
                if len(idx)
                else 0.0
            )
            signed_activation = (
                float(np.mean(state_cpu[idx]))
                if len(idx)
                else 0.0
            )
            bias = (
                float(np.mean(bias_cpu[idx]))
                if len(idx)
                else 0.0
            )
            brain_score = float(self.language_word_score(token))
            salience = max(
                0.0,
                min(
                    1.0,
                    0.55 * brain_score
                    + 0.30 * min(1.0, activation / 0.22)
                    + 0.15 * min(1.0, abs(bias) / max_bias),
                ),
            )
            signatures[token] = self._word_association_signature(token)
            nodes.append({
                "id": token,
                "brain_score": brain_score,
                "salience": salience,
                "activation": activation,
                "signed_activation": signed_activation,
                "plastic_bias": bias,
                "sensory_neurons": int(len(sensory_idx)),
                "output_neurons": int(len(output_idx)),
            })

        edges: list[dict] = []
        for left_i, left in enumerate(cleaned):
            idx_a, val_a = signatures[left]
            norm_a = float(np.linalg.norm(val_a))
            for right in cleaned[left_i + 1:]:
                idx_b, val_b = signatures[right]
                norm_b = float(np.linalg.norm(val_b))
                structural = 0.0
                if len(idx_a) and len(idx_b) and norm_a > 0 and norm_b > 0:
                    _, pos_a, pos_b = np.intersect1d(
                        idx_a,
                        idx_b,
                        assume_unique=True,
                        return_indices=True,
                    )
                    if len(pos_a):
                        structural = float(
                            np.dot(val_a[pos_a], val_b[pos_b])
                            / max(1e-12, norm_a * norm_b)
                        )

                pair_forward = self._subset(
                    f"sensory:text:pair:{left}|{right}",
                    self.c.sensory,
                    24,
                )
                pair_reverse = self._subset(
                    f"sensory:text:pair:{right}|{left}",
                    self.c.sensory,
                    24,
                )
                pair_idx = np.unique(
                    np.concatenate([pair_forward, pair_reverse])
                ).astype(np.int32, copy=False)
                pair_activation = (
                    float(np.mean(np.abs(state_cpu[pair_idx])))
                    if len(pair_idx)
                    else 0.0
                )
                pair_bias = (
                    float(np.mean(bias_cpu[pair_idx]))
                    if len(pair_idx)
                    else 0.0
                )
                learned = math.tanh(
                    pair_bias / max(1e-6, max_bias * 0.22)
                )
                activity = min(1.0, pair_activation / 0.22)
                weight = max(
                    0.0,
                    min(
                        1.0,
                        0.76 * structural
                        + 0.16 * learned
                        + 0.08 * activity,
                    ),
                )
                if weight <= 0.002:
                    continue
                edges.append({
                    "source": left,
                    "target": right,
                    "weight": weight,
                    "structural": structural,
                    "learned": float(learned),
                    "pair_activation": pair_activation,
                    "pair_bias": pair_bias,
                })

        edges.sort(key=lambda item: float(item["weight"]), reverse=True)
        edges = edges[:edge_limit]
        strongest = edges[0] if edges else None
        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "strongest": strongest,
            "reward_trace": float(self.reward_trace),
            "ticks": int(self.tick_count),
            "source": self.c.metadata.get("source", "unknown"),
            "method": (
                "word sensory pools + one-hop connectome overlap + "
                "pair plastic bias/activity"
            ),
        }

    def action_scores(self) -> dict[str, float]:
        scores: dict[str, float] = {}
        for name in self.ACTIONS:
            idx_cpu = self._action_output_pools.get(name)
            if idx_cpu is None or not len(idx_cpu):
                scores[name] = self.readout("action:" + name, 128)
                continue
            idx = self._backend_indices(idx_cpu)
            raw = self.compute.scalar(
                self.xp.mean(self.state[idx])
            )
            scores[name] = _sigmoid(
                3.2 * raw + 0.08 * self.reward_trace
            )
        return scores

    def inject_voice_decision_context(
        self,
        guild_id: int,
        current_channel_id: int | None,
        *,
        connected: bool,
        dwell_progress: float,
        overstay_level: float,
        human_count: int,
        disliked_strength: float = 0.0,
        alternatives: int = 0,
        outside_seconds: float = 0.0,
        available_humans: int = 0,
        social_drive_level: float = 0.0,
        social_drive_magnitude: float = 1.0,
    ) -> dict:
        """Encode voice context as sensory input before motor competition.

        Operational constraints remain outside the brain, but ordinary reasons
        to stay/move/leave are represented as stimuli so reward can reshape the
        real paths from context to the action readouts.
        """
        guided_social_drive = None
        self.inject(
            "voice:context:connected"
            if connected
            else "voice:context:disconnected",
            0.48,
            96,
        )
        if current_channel_id is not None:
            self.inject(
                f"voice:context:current:{guild_id}:{current_channel_id}",
                0.34,
                72,
            )

        dwell_progress = max(0.0, min(1.5, float(dwell_progress)))
        overstay_level = max(0.0, min(1.0, float(overstay_level)))
        disliked_strength = max(
            0.0,
            min(1.0, float(disliked_strength)),
        )
        human_count = max(0, int(human_count))
        alternatives = max(0, int(alternatives))
        outside_seconds = max(0.0, float(outside_seconds))
        available_humans = max(0, int(available_humans))
        social_drive_level = max(
            0.0,
            min(1.0, float(social_drive_level)),
        )
        social_drive_magnitude = max(
            0.0,
            min(4.0, float(social_drive_magnitude)),
        )

        if connected:
            if dwell_progress < 1.0:
                self.inject(
                    "voice:context:early-dwell",
                    0.20 + 0.50 * (1.0 - dwell_progress),
                    96,
                )
            else:
                self.inject(
                    "voice:context:dwell-complete",
                    0.28 + 0.18 * min(1.0, dwell_progress - 1.0),
                    72,
                )
        else:
            outside_bucket = min(
                8,
                int(outside_seconds // 60.0),
            )
            self.inject(
                f"voice:context:outside-minutes:{outside_bucket}",
                0.18 + 0.06 * outside_bucket,
                72,
            )
            if available_humans > 0:
                people_bucket = min(8, available_humans)
                self.inject(
                    f"voice:context:people-available:{people_bucket}",
                    0.22 + 0.07 * people_bucket,
                    96,
                )
            if social_drive_level > 0.0:
                magnitude = (
                    social_drive_magnitude
                    * social_drive_level
                )
                guided_social_drive = (
                    self.inject_action_guided_sensory(
                        "voice_join",
                        f"social-drive:{guild_id}",
                        magnitude,
                        width=224,
                        hops=3,
                    )
                )
                # Keep a weaker untargeted state cue so the network still gets
                # context that this is prolonged isolation, not a direct motor
                # command.
                self.inject(
                    f"voice:social-drive-context:{guild_id}",
                    0.24 * magnitude,
                    64,
                )
        if overstay_level > 0.0:
            self.inject(
                "internal:threat:voice-overstay",
                0.55 + 1.10 * overstay_level,
                192,
            )
            self.inject(
                f"voice:threat:guild:{guild_id}",
                0.42 + 0.70 * overstay_level,
                128,
            )
        if disliked_strength > 0.0:
            self.inject(
                "social:voice-disliked-present",
                0.42 + 0.85 * disliked_strength,
                144,
            )
            self.inject(
                f"social:voice-disliked-present:guild:{guild_id}",
                0.32 + 0.62 * disliked_strength,
                96,
            )

        decision_humans = (
            human_count
            if connected
            else available_humans
        )
        crowd_bucket = min(6, decision_humans)
        self.inject(
            f"voice:context:humans:{crowd_bucket}",
            0.16 + 0.07 * min(decision_humans, 8),
            64,
        )
        option_bucket = min(6, alternatives)
        self.inject(
            f"voice:context:alternatives:{option_bucket}",
            0.14 + 0.06 * min(alternatives, 8),
            56,
        )
        return {
            "social_drive": guided_social_drive,
            "outside_seconds": outside_seconds,
            "available_humans": available_humans,
        }

    def _action_tie_evidence(
        self,
        action: str,
        competitors: tuple[str, ...],
    ) -> dict:
        """Measure neural evidence beyond the scalar action score.

        For close readouts, prefer neurons unique to the competing action and
        inspect what the real base+learned connectome would feed into them on
        the next tick. This avoids Python insertion order deciding behavior.
        """
        pool = np.asarray(
            self._action_output_pools.get(
                action,
                np.empty(0, dtype=np.int32),
            ),
            dtype=np.int32,
        )
        if not len(pool):
            return {
                "support": -1e9,
                "neurons": 0,
                "unique_neurons": 0,
                "mean_activation": 0.0,
                "mean_abs_activation": 0.0,
                "projected_input": 0.0,
                "mean_bias": 0.0,
                "mean_eligibility": 0.0,
            }

        other_parts = [
            np.asarray(
                self._action_output_pools.get(
                    name,
                    np.empty(0, dtype=np.int32),
                ),
                dtype=np.int32,
            )
            for name in competitors
            if name != action
        ]
        if other_parts:
            other = np.unique(
                np.concatenate(other_parts)
            ).astype(np.int32, copy=False)
            unique = np.setdiff1d(
                pool,
                other,
                assume_unique=False,
            ).astype(np.int32, copy=False)
        else:
            unique = pool

        # A tiny unique set is noisy; fall back to the whole biological pool.
        evidence_pool = (
            unique
            if len(unique) >= 8
            else pool
        )
        state_cpu = self.compute.to_cpu(
            self.state
        ).astype(np.float32, copy=False)
        bias_cpu = self.compute.to_cpu(
            self.plastic_bias
        ).astype(np.float32, copy=False)
        eligibility_cpu = self.compute.to_cpu(
            self.eligibility
        ).astype(np.float32, copy=False)

        activation = state_cpu[evidence_pool]
        bias = bias_cpu[evidence_pool]
        eligibility = eligibility_cpu[evidence_pool]

        projected = np.asarray(
            self.c.matrix[
                evidence_pool,
                :,
            ].dot(state_cpu)
        ).ravel().astype(np.float32, copy=False)
        if self._synaptic_delta_map:
            projected += np.asarray(
                self._synaptic_matrix_cpu[
                    evidence_pool,
                    :,
                ].dot(state_cpu)
            ).ravel().astype(np.float32, copy=False)

        mean_activation = float(
            np.mean(activation)
        )
        mean_abs_activation = float(
            np.mean(np.abs(activation))
        )
        projected_input = float(
            np.mean(projected)
        )
        mean_bias = float(
            np.mean(bias)
        )
        mean_eligibility = float(
            np.mean(eligibility)
        )

        max_bias = max(
            1e-6,
            float(self.cfg.max_bias),
        )
        # All terms come from current neural state/topology. tanh only brings
        # very different numeric scales into a comparable range.
        support = (
            0.34 * math.tanh(5.0 * mean_activation)
            + 0.38 * math.tanh(2.5 * projected_input)
            + 0.16 * math.tanh(mean_bias / max_bias)
            + 0.08 * min(1.0, mean_eligibility)
            + 0.04 * math.tanh(4.0 * mean_abs_activation)
        )
        return {
            "support": float(support),
            "neurons": int(len(evidence_pool)),
            "unique_neurons": int(len(unique)),
            "mean_activation": mean_activation,
            "mean_abs_activation": mean_abs_activation,
            "projected_input": projected_input,
            "mean_bias": mean_bias,
            "mean_eligibility": mean_eligibility,
        }

    def voice_action_decision(
        self,
        *,
        connected: bool,
        can_join: bool = True,
        can_move: bool = True,
        can_leave: bool = True,
    ) -> dict:
        """Let action readouts compete; the adapter only masks impossible acts."""
        scores = self.action_scores()
        if connected:
            candidates = {
                "stay": float(scores["stay"]),
            }
            if can_move:
                candidates["voice_move"] = float(
                    scores["voice_move"]
                )
            if can_leave:
                candidates["voice_leave"] = float(
                    scores["voice_leave"]
                )
        else:
            candidates = {
                "stay": float(scores["stay"]),
            }
            if can_join:
                candidates["voice_join"] = float(
                    scores["voice_join"]
                )

        ranked = sorted(
            candidates.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        winner, winner_score = ranked[0]
        runner_name, runner_score = (
            ranked[1]
            if len(ranked) > 1
            else ("none", 0.0)
        )

        tie_band = 0.0025
        tie_break = None
        tie_evidence: dict[str, dict] = {}
        close_actions = tuple(
            name
            for name, score in ranked
            if float(winner_score) - float(score)
            <= tie_band
        )
        if len(close_actions) > 1:
            tie_evidence = {
                name: self._action_tie_evidence(
                    name,
                    close_actions,
                )
                for name in close_actions
            }
            evidence_ranked = sorted(
                close_actions,
                key=lambda name: float(
                    tie_evidence[name]["support"]
                ),
                reverse=True,
            )
            best = evidence_ranked[0]
            second = evidence_ranked[1]
            evidence_margin = (
                float(tie_evidence[best]["support"])
                - float(tie_evidence[second]["support"])
            )
            if abs(evidence_margin) > 1e-9:
                winner = best
                runner_name = second
                winner_score = candidates[winner]
                runner_score = candidates[runner_name]
                tie_break = "neural-evidence"
            else:
                # A completely flat network should not silently bias STAY just
                # because it was inserted first into a Python dict.
                pick = int(
                    self.compute.scalar(
                        self.compute.random_uniform(
                            0.0,
                            float(len(evidence_ranked)),
                            size=1,
                        )[0]
                    )
                )
                pick = min(
                    len(evidence_ranked) - 1,
                    max(0, pick),
                )
                winner = evidence_ranked[pick]
                remaining = [
                    name
                    for name in evidence_ranked
                    if name != winner
                ]
                runner_name = (
                    remaining[0]
                    if remaining
                    else "none"
                )
                winner_score = candidates[winner]
                runner_score = (
                    candidates[runner_name]
                    if runner_name in candidates
                    else 0.0
                )
                tie_break = "unbiased-flat-neural-fallback"

        return {
            "action": winner,
            "score": float(winner_score),
            "runner_up": runner_name,
            "runner_up_score": float(runner_score),
            "margin": float(winner_score - runner_score),
            "candidates": candidates,
            "scores": scores,
            "source": "connectome-readout-competition",
            "tie_band": tie_band,
            "tie_break": tie_break,
            "tie_evidence": tie_evidence,
        }

    def action_path_snapshot(
        self,
        action: str,
        *,
        max_depth: int = 5,
        max_paths: int = 5,
        beam_width: int = 24,
    ) -> dict:
        """Trace strong live presynaptic routes into one action readout."""
        action = str(action or "").strip()
        if action not in self.ACTIONS:
            return {
                "action": action,
                "error": "unknown action",
                "paths": [],
            }

        max_depth = max(1, min(7, int(max_depth)))
        max_paths = max(1, min(8, int(max_paths)))
        beam_width = max(8, min(64, int(beam_width)))
        pool = np.asarray(
            self._action_output_pools.get(
                action,
                np.empty(0, dtype=np.int32),
            ),
            dtype=np.int32,
        )
        if not len(pool):
            return {
                "action": action,
                "error": "empty action pool",
                "paths": [],
            }

        state_cpu = self.compute.to_cpu(
            self.state
        ).astype(np.float32, copy=False)
        bias_cpu = self.compute.to_cpu(
            self.plastic_bias
        ).astype(np.float32, copy=False)
        eligibility_cpu = self.compute.to_cpu(
            self.eligibility
        ).astype(np.float32, copy=False)

        output_rank = (
            np.abs(state_cpu[pool])
            + 0.30 * np.abs(bias_cpu[pool])
            + 0.12 * eligibility_cpu[pool]
        )
        seed_count = min(8, len(pool))
        if seed_count >= len(pool):
            seed_order = np.argsort(output_rank)[::-1]
        else:
            part = np.argpartition(
                output_rank,
                -seed_count,
            )[-seed_count:]
            seed_order = part[
                np.argsort(output_rank[part])[::-1]
            ]
        targets = [
            int(pool[int(pos)])
            for pos in seed_order[:seed_count]
        ]

        n = self.c.n_neurons
        sensory = self._sensory_lookup
        output = self._output_lookup
        modulatory = self._modulatory_lookup

        def role(idx: int) -> str:
            if idx in sensory:
                return "sensory"
            if idx in output:
                return "output"
            if idx in modulatory:
                return "modulatory"
            return "internal"

        def node_row(idx: int) -> dict:
            meta = self.c.neuron_meta or {}
            primary_type = ""
            neuropil = ""
            arr = meta.get("primary_type")
            if arr is not None and len(arr) == n:
                primary_type = str(arr[idx]).strip()
            arr = meta.get("primary_neuropil")
            if arr is not None and len(arr) == n:
                neuropil = str(arr[idx]).strip()
            return {
                "id": str(int(self.c.root_ids[idx])),
                "index": idx,
                "role": role(idx),
                "type": primary_type,
                "neuropil": neuropil,
                "activation": float(state_cpu[idx]),
                "bias": float(bias_cpu[idx]),
                "eligibility": float(eligibility_cpu[idx]),
            }

        # Beam items move backwards from output toward sensory.
        beam: list[tuple[float, list[int], list[dict]]] = [
            (
                float(
                    abs(state_cpu[target])
                    + 0.25 * abs(bias_cpu[target])
                ),
                [target],
                [],
            )
            for target in targets
        ]
        complete: list[tuple[float, list[int], list[dict]]] = []
        partial: list[tuple[float, list[int], list[dict]]] = []

        for _depth in range(max_depth):
            expanded: list[
                tuple[float, list[int], list[dict]]
            ] = []
            for path_score, nodes_rev, edges_rev in beam:
                target = int(nodes_rev[-1])
                row = self.c.matrix.getrow(target)
                if row.nnz == 0:
                    partial.append(
                        (path_score, nodes_rev, edges_rev)
                    )
                    continue

                sources = row.indices.astype(
                    np.int32,
                    copy=False,
                )
                base_weights = row.data.astype(
                    np.float32,
                    copy=False,
                )
                candidate_rows: list[
                    tuple[float, int, float, float, float]
                ] = []
                for source_raw, base_raw in zip(
                    sources,
                    base_weights,
                ):
                    source = int(source_raw)
                    if source in nodes_rev:
                        continue
                    base_weight = float(base_raw)
                    delta = float(
                        self._synaptic_delta_map.get(
                            target * n + source,
                            0.0,
                        )
                    )
                    effective = base_weight + delta
                    importance = (
                        abs(effective)
                        * (
                            0.18
                            + abs(float(state_cpu[source]))
                            + 0.35 * float(
                                eligibility_cpu[source]
                            )
                        )
                    )
                    candidate_rows.append(
                        (
                            importance,
                            source,
                            base_weight,
                            delta,
                            effective,
                        )
                    )

                candidate_rows.sort(
                    key=lambda item: item[0],
                    reverse=True,
                )
                for (
                    importance,
                    source,
                    base_weight,
                    delta,
                    effective,
                ) in candidate_rows[:10]:
                    edge = {
                        "source": str(
                            int(self.c.root_ids[source])
                        ),
                        "target": str(
                            int(self.c.root_ids[target])
                        ),
                        "source_index": source,
                        "target_index": target,
                        "base_weight": base_weight,
                        "learned_delta": delta,
                        "effective_weight": effective,
                        "importance": float(importance),
                    }
                    next_nodes = nodes_rev + [source]
                    next_edges = edges_rev + [edge]
                    next_score = (
                        path_score
                        + math.log1p(
                            max(0.0, importance) * 8.0
                        )
                    )
                    item = (
                        next_score,
                        next_nodes,
                        next_edges,
                    )
                    if source in sensory:
                        complete.append(item)
                    else:
                        expanded.append(item)

            complete.sort(
                key=lambda item: item[0],
                reverse=True,
            )
            if len(complete) >= max_paths:
                break
            expanded.sort(
                key=lambda item: item[0],
                reverse=True,
            )
            beam = expanded[:beam_width]
            partial.extend(beam[:4])
            if not beam:
                break

        chosen = (
            complete[:max_paths]
            if complete
            else sorted(
                partial,
                key=lambda item: item[0],
                reverse=True,
            )[:max_paths]
        )
        paths: list[dict] = []
        for rank, (
            path_score,
            nodes_rev,
            edges_rev,
        ) in enumerate(chosen, start=1):
            nodes_forward = list(reversed(nodes_rev))
            edges_forward = list(reversed(edges_rev))
            paths.append({
                "rank": rank,
                "complete": bool(
                    nodes_forward
                    and nodes_forward[0] in sensory
                ),
                "strength": float(path_score),
                "nodes": [
                    node_row(idx)
                    for idx in nodes_forward
                ],
                "edges": edges_forward,
            })

        info = self._action_output_info.get(
            action,
            {},
        )
        return {
            "action": action,
            "score": float(
                self.action_scores().get(action, 0.0)
            ),
            "mode": str(info.get("mode", "unknown")),
            "pool_size": int(len(pool)),
            "max_depth": max_depth,
            "paths": paths,
            "complete_paths": int(
                sum(
                    1
                    for path in paths
                    if path["complete"]
                )
            ),
            "method": (
                "live backward beam trace over real FAFB edges; "
                "edge weight = base + learned delta"
            ),
        }

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
            """Choose active neurons plus real neighbors that connect them.

            The old graph selected every visible dot independently, then drew
            only edges whose two endpoints happened to be selected. In a
            139k-node sparse graph this made many output dots look isolated.
            Here ~60% of the view is activity-driven and the rest is filled by
            actual presynaptic/postsynaptic neighbors of those core neurons.
            """
            limit = max(12, min(int(limit), 80, self.c.n_neurons))
            core_limit = max(12, min(limit, int(round(limit * 0.60))))
            sensory_n = max(4, core_limit // 6)
            output_n = max(5, core_limit // 5)
            modulatory_n = max(3, core_limit // 10)

            core: list[int] = []
            core.extend(strongest(self.c.sensory, sensory_n))
            core.extend(strongest(self.c.output, output_n))
            core.extend(strongest(self.c.modulatory, modulatory_n))

            core_remaining = max(0, core_limit - len(set(core)))
            if core_remaining:
                k = min(
                    self.c.n_neurons,
                    max(core_limit * 4, core_remaining),
                )
                if k >= self.c.n_neurons:
                    candidates = np.argsort(abs_state)[::-1]
                else:
                    part = np.argpartition(abs_state, -k)[-k:]
                    candidates = part[
                        np.argsort(abs_state[part])[::-1]
                    ]
                core.extend(int(i) for i in candidates)

            selected: list[int] = []
            seen: set[int] = set()
            for idx in core:
                if idx in seen:
                    continue
                seen.add(idx)
                selected.append(idx)
                if len(selected) >= core_limit:
                    break

            neighbor_rank: dict[int, float] = {}

            # Strong presynaptic partners make the path INTO pink/output
            # neurons visible instead of showing an isolated output dot.
            output_core = np.asarray(
                [idx for idx in selected if idx in output_set],
                dtype=np.int32,
            )
            if len(output_core):
                incoming = self.c.matrix[output_core, :].tocoo()
                for local_row, pre_idx, weight in zip(
                    incoming.row,
                    incoming.col,
                    incoming.data,
                ):
                    source_idx = int(pre_idx)
                    target_idx = int(output_core[int(local_row)])
                    if source_idx == target_idx:
                        continue
                    importance = abs(float(weight)) * (
                        0.30
                        + abs(float(state_cpu[source_idx]))
                        + 0.60 * abs(float(state_cpu[target_idx]))
                    )
                    neighbor_rank[source_idx] = max(
                        neighbor_rank.get(source_idx, 0.0),
                        importance * 1.45,
                    )

            # Also keep the first real postsynaptic partners of visible
            # sensory neurons so input -> internal propagation can be seen.
            sensory_core = np.asarray(
                [idx for idx in selected if idx in sensory_set],
                dtype=np.int32,
            )
            if len(sensory_core):
                outgoing = self.c.matrix[:, sensory_core].tocoo()
                for target_idx, local_col, weight in zip(
                    outgoing.row,
                    outgoing.col,
                    outgoing.data,
                ):
                    source_idx = int(sensory_core[int(local_col)])
                    target_idx = int(target_idx)
                    if source_idx == target_idx:
                        continue
                    importance = abs(float(weight)) * (
                        0.30
                        + abs(float(state_cpu[source_idx]))
                        + 0.50 * abs(float(state_cpu[target_idx]))
                    )
                    neighbor_rank[target_idx] = max(
                        neighbor_rank.get(target_idx, 0.0),
                        importance * 1.25,
                    )

            for neighbor_idx, _importance in sorted(
                neighbor_rank.items(),
                key=lambda item: item[1],
                reverse=True,
            ):
                if neighbor_idx in seen:
                    continue
                seen.add(neighbor_idx)
                selected.append(int(neighbor_idx))
                if len(selected) >= limit:
                    break

            # Sparse corners can still have too few local neighbors. Fill the
            # remaining slots by activity, preserving the requested count.
            if len(selected) < limit:
                k = min(
                    self.c.n_neurons,
                    max(limit * 5, limit - len(selected)),
                )
                if k >= self.c.n_neurons:
                    candidates = np.argsort(abs_state)[::-1]
                else:
                    part = np.argpartition(abs_state, -k)[-k:]
                    candidates = part[
                        np.argsort(abs_state[part])[::-1]
                    ]
                for idx_raw in candidates:
                    idx = int(idx_raw)
                    if idx in seen:
                        continue
                    seen.add(idx)
                    selected.append(idx)
                    if len(selected) >= limit:
                        break
            return selected[:limit]

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

        connected_ids: set[str] = set()
        for edge in edges:
            connected_ids.add(str(edge["source"]))
            connected_ids.add(str(edge["target"]))
        output_ids = {
            str(node["id"])
            for node in nodes
            if "output" in node["roles"]
        }
        connected_output_ids = connected_ids & output_ids

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
            "connected_neurons": len(connected_ids),
            "isolated_neurons": max(
                0,
                len(nodes) - len(connected_ids),
            ),
            "connected_output_neurons": len(
                connected_output_ids
            ),
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

        Regions are named FlyWire neuropils when connection-derived metadata is
        available. Otherwise the dashboard explicitly falls back to biological
        classification groups.
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

        def float_at(key: str, idx: int) -> float:
            arr = meta.get(key)
            if arr is None or len(arr) != self.c.n_neurons:
                return 0.0
            try:
                return float(arr[idx])
            except (TypeError, ValueError):
                return 0.0

        # Vectorized direct connectivity into Mucha's action readout
        # populations. Typed descending neurons are biological anchors; the
        # mapping from those broad behaviors to Discord actions is still an
        # explicit adapter and must not be read as a literal biological claim.
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

            neuropils = []
            total_np_mass = max(
                0.0,
                float_at("neuropil_synapse_mass", idx),
            )
            for rank in (1, 2, 3):
                name = text_at(f"neuropil_{rank}", idx)
                mass = max(
                    0.0,
                    float_at(f"neuropil_{rank}_mass", idx),
                )
                if name and mass > 0.0:
                    neuropils.append({
                        "name": name,
                        "mass": mass,
                        "share": (
                            mass / total_np_mass
                            if total_np_mass > 0.0
                            else 0.0
                        ),
                    })

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
                "primary_neuropil": text_at(
                    "primary_neuropil",
                    idx,
                ),
                "neuropils": neuropils,
                "system_actions": actions,
            })

        region_rows: list[dict] = []
        region_values: dict[str, float] = {}
        for label, indices in self._neuro_map_regions.items():
            if len(indices) == 0:
                continue
            values = abs_state[indices]
            mean_abs = float(np.mean(values))
            max_abs = float(np.max(values))
            active_count = int(np.count_nonzero(values > 0.1))
            score = mean_abs * (
                1.0 + math.log1p(active_count) * 0.35
            )
            coords = self._neuro_map_coords[indices]
            region_values[label] = mean_abs

            top_k = min(8, len(indices))
            if top_k >= len(indices):
                local_order = np.argsort(values)[::-1]
            else:
                local_part = np.argpartition(
                    values,
                    -top_k,
                )[-top_k:]
                local_order = local_part[
                    np.argsort(values[local_part])[::-1]
                ]
            top_neurons = []
            for local_idx in local_order[:top_k]:
                neuron_idx = int(indices[int(local_idx)])
                top_neurons.append({
                    "id": str(
                        int(self.c.root_ids[neuron_idx])
                    ),
                    "activation": float(
                        state_cpu[neuron_idx]
                    ),
                    "primary_type": (
                        text_at("primary_type", neuron_idx)
                        or text_at("cell_class", neuron_idx)
                        or text_at("super_class", neuron_idx)
                    ),
                    "nt_type": text_at(
                        "nt_type",
                        neuron_idx,
                    ),
                })

            region_rows.append({
                "name": label,
                "mean_abs": mean_abs,
                "max_abs": max_abs,
                "active_count": active_count,
                "total": int(len(indices)),
                "score": float(score),
                "x": float(np.mean(coords[:, 0])),
                "y": float(np.mean(coords[:, 1])),
                "z": float(np.mean(coords[:, 2])),
                "dominant_types": list(
                    self._neuro_map_region_types.get(
                        label,
                        [],
                    )
                ),
                "top_neurons": top_neurons,
            })

        scores_now = self.action_scores()
        self._neuro_map_history.append({
            "time": float(time.time()),
            "regions": region_values,
            "scores": {
                key: float(value)
                for key, value in scores_now.items()
            },
        })

        def correlation(
            xs: list[float],
            ys: list[float],
        ) -> float | None:
            if len(xs) < 8 or len(ys) != len(xs):
                return None
            xa = np.asarray(xs, dtype=np.float64)
            ya = np.asarray(ys, dtype=np.float64)
            if (
                float(np.std(xa)) < 1e-8
                or float(np.std(ya)) < 1e-8
            ):
                return None
            value = float(np.corrcoef(xa, ya)[0, 1])
            if not math.isfinite(value):
                return None
            return max(-1.0, min(1.0, value))

        history = list(self._neuro_map_history)
        for row in region_rows:
            label = row["name"]
            activity_series = [
                float(sample["regions"].get(label, 0.0))
                for sample in history
            ]
            row["history"] = activity_series[-90:]
            correlations = []
            for action in self.ACTIONS:
                action_series = [
                    float(sample["scores"].get(action, 0.0))
                    for sample in history
                ]
                corr = correlation(
                    activity_series,
                    action_series,
                )
                if corr is None:
                    continue
                correlations.append({
                    "action": action,
                    "correlation": corr,
                })
            correlations.sort(
                key=lambda item: abs(
                    item["correlation"]
                ),
                reverse=True,
            )
            row["readout_correlations"] = correlations[:4]
            row["correlation_samples"] = len(
                activity_series
            )

        region_rows.sort(
            key=lambda item: item["score"],
            reverse=True,
        )
        regions = region_rows[:32]

        reference = [
            {
                "x": float(self._neuro_map_coords[idx, 0]),
                "y": float(self._neuro_map_coords[idx, 1]),
                "z": float(self._neuro_map_coords[idx, 2]),
                "real": bool(self._neuro_map_real_mask[idx]),
            }
            for idx in self._neuro_map_reference_indices
        ]

        real_count = int(
            np.count_nonzero(
                self._neuro_map_real_mask
            )
        )
        coverage = real_count / max(1, self.c.n_neurons)
        if coverage >= 0.70:
            coordinate_mode = "real"
        elif coverage > 0.0:
            coordinate_mode = "hybrid"
        else:
            coordinate_mode = "synthetic"

        neuropil_coverage = float(
            self.c.metadata.get(
                "neuropil_coverage",
                0.0,
            )
            or 0.0
        )
        if self._neuro_map_region_source == "neuropil":
            region_source_detail = (
                "FlyWire neuropil z największą sumą syn_count "
                "na wejściach i wyjściach neuronu"
            )
        else:
            region_source_detail = (
                "Fallback: FlyWire class/super_class; "
                "brak mapy neuropili w cache"
            )

        return {
            "projection": projection,
            "coordinate_mode": coordinate_mode,
            "coordinate_coverage": float(coverage),
            "coordinate_neurons": real_count,
            "region_source": self._neuro_map_region_source,
            "region_source_detail": region_source_detail,
            "neuropil_coverage": neuropil_coverage,
            "neuropil_labels": int(
                self.c.metadata.get(
                    "neuropil_labels",
                    0,
                )
                or 0
            ),
            "nodes": nodes,
            "regions": regions,
            "reference": reference,
            "total_neurons": int(self.c.n_neurons),
            "active_abs_gt_0_1": int(
                np.count_nonzero(abs_state > 0.1)
            ),
            "mean_abs_activation": float(
                np.mean(abs_state)
                if len(abs_state)
                else 0.0
            ),
            "max_abs_activation": float(
                np.max(abs_state)
                if len(abs_state)
                else 0.0
            ),
            "history_samples": len(history),
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
        synaptic_values = np.asarray(
            list(self._synaptic_delta_map.values()),
            dtype=np.float32,
        )
        synaptic_mean_abs = (
            float(np.mean(np.abs(synaptic_values)))
            if synaptic_values.size
            else 0.0
        )
        synaptic_max_abs = (
            float(np.max(np.abs(synaptic_values)))
            if synaptic_values.size
            else 0.0
        )
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
            "learned_synapses": len(self._synaptic_delta_map),
            "synaptic_mean_abs": synaptic_mean_abs,
            "synaptic_max_abs": synaptic_max_abs,
            "action_pools": self.action_pool_diagnostics(),
            "neuromodulation": self.neuromodulator_diagnostics(),
        }
