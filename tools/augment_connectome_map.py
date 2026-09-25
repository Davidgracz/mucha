from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


POSITION_RE = re.compile(
    r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
)


def find_file(
    directory: Path,
    names: list[str],
) -> Path | None:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    return None


def string_array(
    values,
    size: int,
    max_chars: int,
) -> np.ndarray:
    out = np.full(size, "", dtype=f"<U{max_chars}")
    if values is None:
        return out
    for idx, value in values.items():
        if 0 <= int(idx) < size:
            text = str(value or "").strip()
            out[int(idx)] = text[:max_chars]
    return out


def aligned_series(
    frame: pd.DataFrame | None,
    id_to_idx: dict[str, int],
    column: str,
) -> dict[int, str]:
    if frame is None or column not in frame.columns:
        return {}
    result: dict[int, str] = {}
    roots = frame["root_id"].astype(str)
    for root_id, value in zip(roots, frame[column]):
        idx = id_to_idx.get(root_id)
        if idx is None:
            continue
        text = "" if pd.isna(value) else str(value).strip()
        if text:
            result[idx] = text
    return result


def parse_position(value: object) -> tuple[float, float, float] | None:
    if value is None or pd.isna(value):
        return None
    parts = POSITION_RE.findall(str(value))
    if len(parts) < 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Dodaje do istniejącego cache Muchy współrzędne i adnotacje "
            "neuronów do widoku Neuro-map."
        )
    )
    ap.add_argument(
        "--raw",
        required=True,
        help="Folder z plikami Codex FAFB v783 (.csv lub .csv.gz).",
    )
    ap.add_argument(
        "--connectome",
        default="data/connectome",
        help="Istniejący folder runtime connectome.",
    )
    args = ap.parse_args()

    raw = Path(args.raw)
    connectome = Path(args.connectome)
    roots_path = connectome / "root_ids.npy"
    manifest_path = connectome / "manifest.json"
    if not roots_path.exists() or not manifest_path.exists():
        raise SystemExit(
            "Brak root_ids.npy/manifest.json. Najpierw przygotuj connectome."
        )

    root_ids = np.load(roots_path, allow_pickle=False)
    n = len(root_ids)
    id_to_idx = {
        str(int(root_id)): idx
        for idx, root_id in enumerate(root_ids)
    }

    cls_path = find_file(
        raw,
        ["classification.csv.gz", "classification.csv"],
    )
    coords_path = find_file(
        raw,
        ["coordinates.csv.gz", "coordinates.csv"],
    )
    neurons_path = find_file(
        raw,
        ["neurons.csv.gz", "neurons.csv"],
    )
    types_path = find_file(
        raw,
        [
            "consolidated_cell_types.csv.gz",
            "consolidated_cell_types.csv",
            "cell_types.csv.gz",
            "cell_types.csv",
        ],
    )

    print(f"Runtime neurons: {n:,}")
    print(f"classification: {cls_path or 'brak'}")
    print(f"coordinates:    {coords_path or 'brak'}")
    print(f"neurons:        {neurons_path or 'brak'}")
    print(f"cell types:     {types_path or 'brak'}")

    cls = (
        pd.read_csv(cls_path, dtype={"root_id": "string"})
        if cls_path else None
    )
    neurons = (
        pd.read_csv(neurons_path, dtype={"root_id": "string"})
        if neurons_path else None
    )
    types = (
        pd.read_csv(types_path, dtype={"root_id": "string"})
        if types_path else None
    )

    x = np.full(n, np.nan, dtype=np.float32)
    y = np.full(n, np.nan, dtype=np.float32)
    z = np.full(n, np.nan, dtype=np.float32)

    coordinate_rows = 0
    coordinate_neurons = 0
    if coords_path is not None:
        coords = pd.read_csv(
            coords_path,
            dtype={"root_id": "string"},
            usecols=lambda col: col in {"root_id", "position"},
        )
        if "root_id" not in coords.columns or "position" not in coords.columns:
            raise SystemExit(
                "coordinates.csv musi zawierać root_id i position"
            )

        buckets: dict[int, list[tuple[float, float, float]]] = {}
        for root_id, position in zip(
            coords["root_id"].astype(str),
            coords["position"],
        ):
            idx = id_to_idx.get(root_id)
            if idx is None:
                continue
            parsed = parse_position(position)
            if parsed is None:
                continue
            buckets.setdefault(idx, []).append(parsed)
            coordinate_rows += 1

        for idx, points in buckets.items():
            arr = np.asarray(points, dtype=np.float64)
            # Median is robust when a neuron has several marked coordinates.
            med = np.median(arr, axis=0)
            x[idx], y[idx], z[idx] = med.astype(np.float32)
        coordinate_neurons = len(buckets)

    primary_col = None
    if types is not None:
        primary_col = next(
            (
                col for col in (
                    "primary_type",
                    "cell_type",
                    "type",
                )
                if col in types.columns
            ),
            None,
        )

    nt_col = None
    if neurons is not None:
        nt_col = next(
            (
                col for col in (
                    "nt_type",
                    "predicted_nt",
                    "neurotransmitter",
                )
                if col in neurons.columns
            ),
            None,
        )

    arrays = {
        "x": x,
        "y": y,
        "z": z,
        "super_class": string_array(
            aligned_series(cls, id_to_idx, "super_class"),
            n,
            32,
        ),
        "cell_class": string_array(
            aligned_series(cls, id_to_idx, "class"),
            n,
            40,
        ),
        "sub_class": string_array(
            aligned_series(cls, id_to_idx, "sub_class"),
            n,
            48,
        ),
        "side": string_array(
            aligned_series(cls, id_to_idx, "side"),
            n,
            16,
        ),
        "flow": string_array(
            aligned_series(cls, id_to_idx, "flow"),
            n,
            20,
        ),
        "nerve": string_array(
            aligned_series(cls, id_to_idx, "nerve"),
            n,
            40,
        ),
        "primary_type": string_array(
            aligned_series(
                types,
                id_to_idx,
                primary_col or "__missing__",
            ),
            n,
            64,
        ),
        "nt_type": string_array(
            aligned_series(
                neurons,
                id_to_idx,
                nt_col or "__missing__",
            ),
            n,
            16,
        ),
    }

    out_path = connectome / "neuron_meta.npz"
    np.savez_compressed(out_path, **arrays)

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    manifest.update({
        "neuron_meta": True,
        "coordinate_source": (
            "FlyWire Codex FAFB v783 marked neuron coordinates"
            if coordinate_neurons
            else "unavailable"
        ),
        "coordinate_rows_used": int(coordinate_rows),
        "coordinate_neurons": int(np.count_nonzero(finite)),
        "coordinate_coverage": float(
            np.count_nonzero(finite) / max(1, n)
        ),
        "neuron_meta_fields": sorted(arrays.keys()),
    })
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Zapisano: {out_path}")
    print(
        "Współrzędne: "
        f"{np.count_nonzero(finite):,}/{n:,} "
        f"({manifest['coordinate_coverage'] * 100:.1f}%)"
    )
    print("Neuro-map metadata OK")


if __name__ == "__main__":
    main()
