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


def parse_position(
    value: object,
) -> tuple[float, float, float] | None:
    if value is None or pd.isna(value):
        return None
    parts = POSITION_RE.findall(str(value))
    if len(parts) < 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


def build_neuropil_summary(
    path: Path,
    id_to_idx: dict[str, int],
    n: int,
    chunksize: int,
) -> tuple[dict[str, np.ndarray], dict]:
    """Summarise each neuron by the neuropils carrying its incident synapses.

    The filtered Codex connection table contains one row per
    pre/post/neuropil combination. We add syn_count to both the presynaptic and
    postsynaptic neuron so the resulting primary neuropil is a compact spatial
    summary of where that neuron's strongest annotated connectivity occurs.
    """
    label_to_col: dict[str, int] = {}
    labels: list[str] = []
    capacity = 32
    mass = np.zeros((n, capacity), dtype=np.float32)
    rows_seen = 0
    rows_used = 0

    print("Buduję mapę neuropili z tabeli połączeń...")
    reader = pd.read_csv(
        path,
        chunksize=max(50_000, int(chunksize)),
        dtype={
            "pre_root_id": "string",
            "post_root_id": "string",
            "pre_pt_root_id": "string",
            "post_pt_root_id": "string",
        },
    )

    for chunk in reader:
        rows_seen += len(chunk)
        pre_col = next(
            (
                c for c in (
                    "pre_root_id",
                    "pre_pt_root_id",
                    "pre",
                )
                if c in chunk.columns
            ),
            None,
        )
        post_col = next(
            (
                c for c in (
                    "post_root_id",
                    "post_pt_root_id",
                    "post",
                )
                if c in chunk.columns
            ),
            None,
        )
        neuropil_col = next(
            (
                c for c in (
                    "neuropil",
                    "region",
                    "roi",
                )
                if c in chunk.columns
            ),
            None,
        )
        syn_col = next(
            (
                c for c in (
                    "syn_count",
                    "weight",
                    "synapses",
                )
                if c in chunk.columns
            ),
            None,
        )
        if not all((pre_col, post_col, neuropil_col, syn_col)):
            raise SystemExit(
                "Tabela połączeń nie ma wymaganych kolumn "
                "pre/post/neuropil/syn_count: "
                + ", ".join(chunk.columns)
            )

        neuropils = (
            chunk[neuropil_col]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        syn = pd.to_numeric(
            chunk[syn_col],
            errors="coerce",
        ).fillna(0.0)

        good = neuropils.ne("") & syn.gt(0.0)
        if not good.any():
            continue

        for label in neuropils[good].unique():
            if label not in label_to_col:
                label_to_col[label] = len(labels)
                labels.append(label)

        while len(labels) > capacity:
            new_capacity = capacity * 2
            expanded = np.zeros(
                (n, new_capacity),
                dtype=np.float32,
            )
            expanded[:, :capacity] = mass
            mass = expanded
            capacity = new_capacity

        pre_idx = (
            chunk.loc[good, pre_col]
            .astype(str)
            .map(id_to_idx)
        )
        post_idx = (
            chunk.loc[good, post_col]
            .astype(str)
            .map(id_to_idx)
        )
        np_labels = (
            neuropils[good]
            .map(label_to_col)
            .to_numpy(np.int32)
        )
        np_syn = syn[good].to_numpy(np.float32)

        pre_good = pre_idx.notna().to_numpy()
        if pre_good.any():
            np.add.at(
                mass,
                (
                    pre_idx[pre_good].astype(np.int32).to_numpy(),
                    np_labels[pre_good],
                ),
                np_syn[pre_good],
            )
            rows_used += int(np.count_nonzero(pre_good))

        post_good = post_idx.notna().to_numpy()
        if post_good.any():
            np.add.at(
                mass,
                (
                    post_idx[post_good].astype(np.int32).to_numpy(),
                    np_labels[post_good],
                ),
                np_syn[post_good],
            )
            rows_used += int(np.count_nonzero(post_good))

        print(
            f"  {rows_seen:,} rows • neuropils {len(labels)}",
            end="\r",
        )
    print()

    label_count = len(labels)
    empty_names = {
        "primary_neuropil": np.full(n, "", dtype="<U32"),
        "neuropil_1": np.full(n, "", dtype="<U32"),
        "neuropil_2": np.full(n, "", dtype="<U32"),
        "neuropil_3": np.full(n, "", dtype="<U32"),
        "neuropil_1_mass": np.zeros(n, dtype=np.float32),
        "neuropil_2_mass": np.zeros(n, dtype=np.float32),
        "neuropil_3_mass": np.zeros(n, dtype=np.float32),
        "neuropil_synapse_mass": np.zeros(n, dtype=np.float32),
    }
    if label_count == 0:
        return empty_names, {
            "neuropil_source": str(path),
            "neuropil_labels": 0,
            "neuropil_rows_seen": rows_seen,
            "neuropil_rows_used": rows_used,
            "neuropil_neurons": 0,
        }

    used = mass[:, :label_count]
    top_k = min(3, label_count)
    if top_k == label_count:
        top_idx = np.argsort(
            used,
            axis=1,
        )[:, ::-1][:, :top_k]
    else:
        part = np.argpartition(
            used,
            kth=label_count - top_k,
            axis=1,
        )[:, -top_k:]
        part_values = np.take_along_axis(
            used,
            part,
            axis=1,
        )
        order = np.argsort(
            part_values,
            axis=1,
        )[:, ::-1]
        top_idx = np.take_along_axis(
            part,
            order,
            axis=1,
        )

    top_values = np.take_along_axis(
        used,
        top_idx,
        axis=1,
    )
    label_array = np.asarray(labels, dtype="<U32")

    result = dict(empty_names)
    for rank in range(top_k):
        values = top_values[:, rank].astype(
            np.float32,
            copy=False,
        )
        names = np.full(n, "", dtype="<U32")
        present = values > 0.0
        names[present] = label_array[
            top_idx[present, rank]
        ]
        result[f"neuropil_{rank + 1}"] = names
        result[f"neuropil_{rank + 1}_mass"] = values

    result["primary_neuropil"] = result["neuropil_1"].copy()
    result["neuropil_synapse_mass"] = np.sum(
        used,
        axis=1,
        dtype=np.float32,
    )

    neurons_with_region = int(
        np.count_nonzero(
            result["primary_neuropil"] != ""
        )
    )
    return result, {
        "neuropil_source": (
            "FlyWire Codex FAFB v783 filtered connections "
            "(incident synapse-count summary)"
        ),
        "neuropil_file": path.name,
        "neuropil_labels": int(label_count),
        "neuropil_rows_seen": int(rows_seen),
        "neuropil_rows_used": int(rows_used),
        "neuropil_neurons": neurons_with_region,
        "neuropil_coverage": float(
            neurons_with_region / max(1, n)
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Dodaje do istniejącego cache Muchy współrzędne, adnotacje "
            "i neuropile do widoku Neuro-map."
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
    ap.add_argument(
        "--chunksize",
        type=int,
        default=450_000,
        help="Chunk tabeli połączeń przy liczeniu neuropili.",
    )
    args = ap.parse_args()

    raw = Path(args.raw)
    connectome = Path(args.connectome)
    roots_path = connectome / "root_ids.npy"
    manifest_path = connectome / "manifest.json"
    if not roots_path.exists() or not manifest_path.exists():
        raise SystemExit(
            "Brak root_ids.npy/manifest.json. "
            "Najpierw przygotuj connectome."
        )

    root_ids = np.load(
        roots_path,
        allow_pickle=False,
    )
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
    connections_path = find_file(
        raw,
        [
            "connections_princeton.csv.gz",
            "connections.csv.gz",
            "connections_princeton.csv",
            "connections.csv",
            "connections_princeton_no_threshold.csv.gz",
            "connections_no_threshold.csv.gz",
        ],
    )

    print(f"Runtime neurons: {n:,}")
    print(f"classification: {cls_path or 'brak'}")
    print(f"coordinates:    {coords_path or 'brak'}")
    print(f"neurons:        {neurons_path or 'brak'}")
    print(f"cell types:     {types_path or 'brak'}")
    print(f"connections:    {connections_path or 'brak'}")

    cls = (
        pd.read_csv(
            cls_path,
            dtype={"root_id": "string"},
        )
        if cls_path else None
    )
    neurons = (
        pd.read_csv(
            neurons_path,
            dtype={"root_id": "string"},
        )
        if neurons_path else None
    )
    types = (
        pd.read_csv(
            types_path,
            dtype={"root_id": "string"},
        )
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
            usecols=lambda col: col in {
                "root_id",
                "position",
            },
        )
        if (
            "root_id" not in coords.columns
            or "position" not in coords.columns
        ):
            raise SystemExit(
                "coordinates.csv musi zawierać root_id i position"
            )

        buckets: dict[
            int,
            list[tuple[float, float, float]],
        ] = {}
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
            arr = np.asarray(
                points,
                dtype=np.float64,
            )
            med = np.median(arr, axis=0)
            x[idx], y[idx], z[idx] = med.astype(
                np.float32
            )
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

    arrays: dict[str, np.ndarray] = {
        "x": x,
        "y": y,
        "z": z,
        "super_class": string_array(
            aligned_series(
                cls,
                id_to_idx,
                "super_class",
            ),
            n,
            32,
        ),
        "cell_class": string_array(
            aligned_series(
                cls,
                id_to_idx,
                "class",
            ),
            n,
            40,
        ),
        "sub_class": string_array(
            aligned_series(
                cls,
                id_to_idx,
                "sub_class",
            ),
            n,
            48,
        ),
        "side": string_array(
            aligned_series(
                cls,
                id_to_idx,
                "side",
            ),
            n,
            16,
        ),
        "flow": string_array(
            aligned_series(
                cls,
                id_to_idx,
                "flow",
            ),
            n,
            20,
        ),
        "nerve": string_array(
            aligned_series(
                cls,
                id_to_idx,
                "nerve",
            ),
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

    neuropil_manifest: dict = {
        "neuropil_source": "unavailable",
        "neuropil_labels": 0,
        "neuropil_neurons": 0,
        "neuropil_coverage": 0.0,
    }
    if connections_path is not None:
        neuropil_arrays, neuropil_manifest = build_neuropil_summary(
            connections_path,
            id_to_idx,
            n,
            args.chunksize,
        )
        arrays.update(neuropil_arrays)
    else:
        arrays.update({
            "primary_neuropil": np.full(
                n,
                "",
                dtype="<U32",
            ),
            "neuropil_1": np.full(
                n,
                "",
                dtype="<U32",
            ),
            "neuropil_2": np.full(
                n,
                "",
                dtype="<U32",
            ),
            "neuropil_3": np.full(
                n,
                "",
                dtype="<U32",
            ),
            "neuropil_1_mass": np.zeros(
                n,
                dtype=np.float32,
            ),
            "neuropil_2_mass": np.zeros(
                n,
                dtype=np.float32,
            ),
            "neuropil_3_mass": np.zeros(
                n,
                dtype=np.float32,
            ),
            "neuropil_synapse_mass": np.zeros(
                n,
                dtype=np.float32,
            ),
        })

    out_path = connectome / "neuron_meta.npz"
    np.savez_compressed(
        out_path,
        **arrays,
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8",
        )
    )
    finite = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
    )
    manifest.update({
        "neuron_meta": True,
        "coordinate_source": (
            "FlyWire Codex FAFB v783 marked neuron coordinates"
            if coordinate_neurons
            else "unavailable"
        ),
        "coordinate_rows_used": int(
            coordinate_rows
        ),
        "coordinate_neurons": int(
            np.count_nonzero(finite)
        ),
        "coordinate_coverage": float(
            np.count_nonzero(finite)
            / max(1, n)
        ),
        "neuron_meta_fields": sorted(
            arrays.keys()
        ),
        **neuropil_manifest,
    })
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Zapisano: {out_path}")
    print(
        "Współrzędne: "
        f"{np.count_nonzero(finite):,}/{n:,} "
        f"({manifest['coordinate_coverage'] * 100:.1f}%)"
    )
    print(
        "Neuropile: "
        f"{manifest.get('neuropil_neurons', 0):,}/{n:,} "
        f"({manifest.get('neuropil_coverage', 0.0) * 100:.1f}%) • "
        f"{manifest.get('neuropil_labels', 0)} regionów"
    )
    print("Neuro-map metadata OK")


if __name__ == "__main__":
    main()
