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

    # Synthetic anatomical metadata lets the dashboard exercise the same
    # neuro-map path as a real FlyWire cache without pretending to be real.
    angle = rng.uniform(0.0, 2.0 * np.pi, size=n)
    radius = np.sqrt(rng.uniform(0.0, 1.0, size=n))
    x = (radius * np.cos(angle) * 0.95).astype(np.float32)
    y = (radius * np.sin(angle) * 0.72).astype(np.float32)
    z = (rng.normal(0.0, 0.35, size=n)).astype(np.float32)
    side = np.where(x < -0.08, "left", np.where(x > 0.08, "right", "midline"))
    super_class = np.full(n, "central", dtype="<U20")
    super_class[sensory] = "sensory"
    super_class[output] = "descending"
    cell_class = np.full(n, "central", dtype="<U24")
    cell_class[sensory] = "sensory-demo"
    cell_class[output] = "motor-demo"
    sub_class = np.full(n, "", dtype="<U32")
    flow = np.full(n, "central", dtype="<U16")
    flow[sensory] = "sensory"
    flow[output] = "descending"
    nt_type = rng.choice(
        np.array(["ACH", "GABA", "GLUT", "DA"], dtype="<U8"),
        size=n,
        p=[0.55, 0.20, 0.20, 0.05],
    )
    primary_type = np.full(n, "demo-neuron", dtype="<U32")
    np.savez_compressed(
        out / "neuron_meta.npz",
        x=x,
        y=y,
        z=z,
        super_class=super_class,
        cell_class=cell_class,
        sub_class=sub_class,
        side=side.astype("<U12"),
        flow=flow,
        nerve=np.full(n, "", dtype="<U24"),
        primary_type=primary_type,
        nt_type=nt_type,
    )
    (out / "manifest.json").write_text(json.dumps({
        "source": "synthetic demo graph - NOT FlyWire",
        "neurons": n,
        "connections": int(m.nnz),
        "demo": True,
        "neuron_meta": True,
        "coordinate_source": "synthetic demo coordinates",
    }, indent=2), encoding="utf-8")
    print(f"Demo connectome zapisany: {n} neuronów, {m.nnz} połączeń -> {out}")


if __name__ == "__main__":
    main()
