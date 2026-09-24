from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from scipy import sparse


@dataclass(slots=True)
class Connectome:
    matrix: sparse.csr_matrix
    root_ids: np.ndarray
    sensory: np.ndarray
    output: np.ndarray
    modulatory: np.ndarray
    metadata: dict

    @property
    def n_neurons(self) -> int:
        return int(self.matrix.shape[0])

    @classmethod
    def load(cls, directory: str | Path) -> "Connectome":
        d = Path(directory)
        matrix_path = d / "matrix.npz"
        roots_path = d / "root_ids.npy"
        pools_path = d / "pools.npz"
        meta_path = d / "manifest.json"

        missing = [p.name for p in (matrix_path, roots_path, pools_path, meta_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Brakuje przygotowanych danych connectome: " + ", ".join(missing) +
                ". Uruchom tools/prepare_connectome.py albo tools/make_demo_connectome.py."
            )

        matrix = sparse.load_npz(matrix_path).tocsr().astype(np.float32)
        root_ids = np.load(roots_path, allow_pickle=False)
        pools = np.load(pools_path, allow_pickle=False)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(
            matrix=matrix,
            root_ids=root_ids,
            sensory=pools["sensory"].astype(np.int32),
            output=pools["output"].astype(np.int32),
            modulatory=pools["modulatory"].astype(np.int32),
            metadata=metadata,
        )
