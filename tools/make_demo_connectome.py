from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from scipy import sparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/connectome")
    ap.add_argument("--neurons", type=int, default=2500)
    ap.add_argument("--connections", type=int, default=45000)
    ap.add_argument("--seed", type=int, default=67)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n = args.neurons
    pre = rng.integers(0, n, size=args.connections, dtype=np.int32)
    post = rng.integers(0, n, size=args.connections, dtype=np.int32)
    weights = rng.lognormal(mean=-1.1, sigma=0.7, size=args.connections).astype(np.float32)
    inhibitory = rng.random(args.connections) < 0.30
    weights[inhibitory] *= -1
    m = sparse.coo_matrix((weights, (post, pre)), shape=(n,n), dtype=np.float32).tocsr()
    denom = np.asarray(abs(m).sum(axis=1)).ravel().astype(np.float32)
    denom[denom < 1.0] = 1.0
    m = sparse.diags(1.0 / denom).dot(m).tocsr().astype(np.float32)
    sparse.save_npz(out / "matrix.npz", m, compressed=True)
    np.save(out / "root_ids.npy", np.arange(1, n+1, dtype=np.uint64), allow_pickle=False)
    sensory = np.arange(0, max(200, n//5), dtype=np.int32)
    output = np.arange(max(0,n-n//5), n, dtype=np.int32)
    modulatory = rng.choice(np.arange(n, dtype=np.int32), size=min(100, n), replace=False)
    np.savez_compressed(out / "pools.npz", sensory=sensory, output=output, modulatory=modulatory)
    (out / "manifest.json").write_text(json.dumps({
        "source": "synthetic demo graph - NOT FlyWire",
        "neurons": n,
        "connections": int(m.nnz),
        "demo": True,
    }, indent=2), encoding="utf-8")
    print(f"Demo connectome zapisany: {n} neuronów, {m.nnz} połączeń -> {out}")


if __name__ == "__main__":
    main()
