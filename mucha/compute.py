from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
from scipy import sparse as scipy_sparse

log = logging.getLogger("mucha.compute")


@dataclass(slots=True)
class ComputeInfo:
    requested: str
    active: str
    device_name: str
    cuda_runtime: str | None = None


class ComputeBackend:
    """Small NumPy/CuPy compatibility layer for Mucha's connectome runtime."""

    def __init__(self, requested: str = "auto", device: int = 0):
        requested = (requested or "auto").strip().lower()
        if requested not in {"auto", "cpu", "cuda", "gpu"}:
            raise ValueError("brain.backend must be one of: auto, cpu, cuda")

        self.requested = requested
        self.device_index = int(device)
        self.cp = None
        self.cupy_sparse = None
        self.xp = np
        self.is_gpu = False
        self.info = ComputeInfo(requested=requested, active="cpu", device_name="CPU")

        if requested in {"auto", "cuda", "gpu"}:
            self._try_cuda(required=requested in {"cuda", "gpu"})

    def _try_cuda(self, required: bool) -> None:
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cupy_sparse

            count = int(cp.cuda.runtime.getDeviceCount())
            if count < 1:
                raise RuntimeError("CUDA device not found")
            if self.device_index < 0 or self.device_index >= count:
                raise RuntimeError(f"CUDA device {self.device_index} does not exist; detected {count}")

            cp.cuda.Device(self.device_index).use()
            props = cp.cuda.runtime.getDeviceProperties(self.device_index)
            raw_name = props.get("name", b"CUDA GPU")
            if isinstance(raw_name, bytes):
                name = raw_name.decode("utf-8", "replace")
            else:
                name = str(raw_name)

            runtime = cp.cuda.runtime.runtimeGetVersion()
            runtime_text = f"{runtime // 1000}.{(runtime % 1000) // 10}"

            self.cp = cp
            self.cupy_sparse = cupy_sparse
            self.xp = cp
            self.is_gpu = True
            self.info = ComputeInfo(
                requested=self.requested,
                active="cuda",
                device_name=name,
                cuda_runtime=runtime_text,
            )
            log.info("Compute backend: CUDA on %s (runtime %s)", name, runtime_text)
        except Exception as exc:
            if required:
                raise RuntimeError(
                    "GPU backend requested but CUDA/CuPy is unavailable. "
                    "Install a matching CuPy wheel or set brain.backend='auto'/'cpu'."
                ) from exc
            log.warning("CUDA unavailable, falling back to CPU: %s", exc)

    def asarray(self, value: Any, dtype=None):
        return self.xp.asarray(value, dtype=dtype)

    def zeros(self, shape, dtype=np.float32):
        return self.xp.zeros(shape, dtype=dtype)

    def to_cpu(self, value):
        if self.is_gpu:
            return self.cp.asnumpy(value)
        return np.asarray(value)

    def scalar(self, value) -> float:
        if self.is_gpu:
            return float(value.item())
        return float(value)

    def int_scalar(self, value) -> int:
        if self.is_gpu:
            return int(value.item())
        return int(value)

    def sparse_from_scipy(self, matrix: scipy_sparse.csr_matrix):
        if self.is_gpu:
            # Explicit arrays avoid ambiguous implicit SciPy/CuPy conversions.
            data = self.cp.asarray(matrix.data)
            indices = self.cp.asarray(matrix.indices)
            indptr = self.cp.asarray(matrix.indptr)
            return self.cupy_sparse.csr_matrix(
                (data, indices, indptr), shape=matrix.shape, dtype=self.cp.float32
            )
        return matrix.tocsr().astype(np.float32, copy=False)

    def random_normal(self, mean: float, std: float, size):
        if self.is_gpu:
            return self.cp.random.normal(mean, std, size=size).astype(self.cp.float32)
        return np.random.normal(mean, std, size=size).astype(np.float32)

    def random_uniform(self, low: float, high: float, size):
        if self.is_gpu:
            return self.cp.random.uniform(low, high, size=size).astype(self.cp.float32)
        return np.random.uniform(low, high, size=size).astype(np.float32)
