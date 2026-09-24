from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

NT_SIGN = {
    "ACH": 1.0, "ACETYLCHOLINE": 1.0,
    "GABA": -1.0,
    "GLUT": -1.0, "GLUTAMATE": -1.0,
    "DA": 0.35, "DOPAMINE": 0.35,
    "SER": 0.30, "SEROTONIN": 0.30,
    "OCT": 0.30, "OCTOPAMINE": 0.30,
}


def find_file(d: Path, names: list[str], required: bool = True) -> Path | None:
    for name in names:
        p = d / name
        if p.exists():
            return p
    if required:
        raise FileNotFoundError("Nie znaleziono żadnego z: " + ", ".join(names))
    return None


def norm_nt(v: object) -> str:
    return str(v or "").strip().upper()


def main():
    ap = argparse.ArgumentParser(description="Buduje lekką macierz runtime z FlyWire FAFB v783.")
    ap.add_argument("--input", required=True, help="Folder z plikami .csv.gz z Codex")
    ap.add_argument("--output", default="data/connectome")
    ap.add_argument("--chunksize", type=int, default=600_000)
    args = ap.parse_args()

    raw = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    cls_path = find_file(raw, ["classification.csv.gz", "classification.csv"])
    conn_path = find_file(raw, [
        "connections_princeton.csv.gz", "connections.csv.gz",
        "connections_princeton_no_threshold.csv.gz", "connections_princeton.csv", "connections.csv"
    ])
    neurons_path = find_file(raw, ["neurons.csv.gz", "neurons.csv"], required=False)

    print("[1/5] Wczytuję klasyfikację neuronów...")
    cls = pd.read_csv(cls_path, dtype={"root_id": "string"})
    if "root_id" not in cls.columns:
        raise SystemExit("classification nie ma kolumny root_id")
    cls = cls.drop_duplicates("root_id").copy()
    root_str = cls["root_id"].astype(str).to_numpy()
    root_py = np.array([int(x) for x in root_str], dtype=object)
    order = np.argsort(root_py)
    cls = cls.iloc[order].reset_index(drop=True)
    root_ids = np.array([int(x) for x in cls["root_id"].astype(str)], dtype=np.uint64)
    n = len(root_ids)
    id_to_idx = {str(int(rid)): i for i, rid in enumerate(root_ids)}
    print(f"      neurony: {n:,}")

    nt_by_id: dict[str, str] = {}
    if neurons_path is not None:
        print("[2/5] Wczytuję przewidywane neuroprzekaźniki...")
        neu = pd.read_csv(neurons_path, dtype={"root_id": "string"})
        nt_col = next((c for c in ("nt_type", "predicted_nt", "neurotransmitter") if c in neu.columns), None)
        if nt_col:
            nt_by_id = dict(zip(neu["root_id"].astype(str), neu[nt_col].map(norm_nt)))
    else:
        print("[2/5] Brak neurons.csv.gz — użyję nt_type z tabeli połączeń, jeśli istnieje.")

    print("[3/5] Przetwarzam połączenia chunkami...")
    rows_all: list[np.ndarray] = []
    cols_all: list[np.ndarray] = []
    data_all: list[np.ndarray] = []
    total_rows = 0
    kept = 0

    for chunk in pd.read_csv(conn_path, chunksize=args.chunksize, dtype={"pre_root_id": "string", "post_root_id": "string"}):
        pre_col = next((c for c in ("pre_root_id", "pre_pt_root_id", "pre") if c in chunk.columns), None)
        post_col = next((c for c in ("post_root_id", "post_pt_root_id", "post") if c in chunk.columns), None)
        syn_col = next((c for c in ("syn_count", "weight", "synapses") if c in chunk.columns), None)
        if not all((pre_col, post_col, syn_col)):
            raise SystemExit(f"Nie rozpoznaję kolumn połączeń: {list(chunk.columns)}")
        total_rows += len(chunk)
        pre_s = chunk[pre_col].astype(str)
        post_s = chunk[post_col].astype(str)
        pre_idx = pre_s.map(id_to_idx)
        post_idx = post_s.map(id_to_idx)
        good = pre_idx.notna() & post_idx.notna()
        if not good.any():
            continue
        pre_idx = pre_idx[good].astype(np.int32).to_numpy()
        post_idx = post_idx[good].astype(np.int32).to_numpy()
        syn = pd.to_numeric(chunk.loc[good, syn_col], errors="coerce").fillna(0).to_numpy(np.float32)
        strength = np.log1p(np.maximum(syn, 0.0)).astype(np.float32)
        if "nt_type" in chunk.columns:
            nts = chunk.loc[good, "nt_type"].map(norm_nt).to_numpy()
            signs = np.array([NT_SIGN.get(x, 0.15) for x in nts], dtype=np.float32)
        elif nt_by_id:
            ids = pre_s[good].to_numpy()
            signs = np.array([NT_SIGN.get(nt_by_id.get(x, ""), 0.15) for x in ids], dtype=np.float32)
        else:
            signs = np.ones_like(strength, dtype=np.float32)
        weights = strength * signs
        rows_all.append(post_idx)
        cols_all.append(pre_idx)
        data_all.append(weights)
        kept += len(weights)
        print(f"      {total_rows:,} wierszy, zachowano {kept:,}", end="\r")
    print()

    if not rows_all:
        raise SystemExit("Brak połączeń pasujących do listy neuronów.")

    row = np.concatenate(rows_all)
    col = np.concatenate(cols_all)
    dat = np.concatenate(data_all)
    del rows_all, cols_all, data_all

    print("[4/5] Buduję i normalizuję macierz sparse...")
    mat = sparse.coo_matrix((dat, (row, col)), shape=(n, n), dtype=np.float32).tocsr()
    mat.sum_duplicates()
    denom = np.asarray(abs(mat).sum(axis=1)).ravel().astype(np.float32)
    denom[denom < 1.0] = 1.0
    mat = sparse.diags((1.0 / denom).astype(np.float32)).dot(mat).tocsr().astype(np.float32)

    sc = cls.get("super_class", pd.Series([""] * n)).fillna("").astype(str).str.lower()
    flow = cls.get("flow", pd.Series([""] * n)).fillna("").astype(str).str.lower()
    sensory = np.flatnonzero(sc.str.contains("sensory") | flow.str.contains("sensory")).astype(np.int32)
    output = np.flatnonzero(sc.str.contains("descending|motor") | flow.str.contains("descending|motor")).astype(np.int32)

    modulatory_mask = np.zeros(n, dtype=bool)
    if nt_by_id:
        for i, rid in enumerate(root_ids):
            if nt_by_id.get(str(int(rid)), "") in {"DA", "DOPAMINE", "SER", "SEROTONIN", "OCT", "OCTOPAMINE"}:
                modulatory_mask[i] = True
    modulatory = np.flatnonzero(modulatory_mask).astype(np.int32)

    if len(sensory) < 50:
        sensory = np.arange(min(n, max(1000, n // 8)), dtype=np.int32)
    if len(output) < 50:
        output = np.arange(max(0, n - max(1000, n // 8)), n, dtype=np.int32)
    if len(modulatory) < 10:
        modulatory = np.arange(0, n, max(1, n // 128), dtype=np.int32)[:128]

    print("[5/5] Zapisuję runtime...")
    sparse.save_npz(out / "matrix.npz", mat, compressed=True)
    np.save(out / "root_ids.npy", root_ids, allow_pickle=False)
    np.savez_compressed(out / "pools.npz", sensory=sensory, output=output, modulatory=modulatory)
    manifest = {
        "source": "FlyWire Codex FAFB v783 (user-provided static downloads)",
        "neurons": int(n),
        "connections_runtime": int(mat.nnz),
        "source_rows": int(total_rows),
        "sensory_pool": int(len(sensory)),
        "output_pool": int(len(output)),
        "modulatory_pool": int(len(modulatory)),
        "weighting": "log1p(syn_count) * neurotransmitter sign; incoming L1 normalization",
        "note": "Continuous lightweight runtime, not a biophysical reproduction of a real fly brain.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
