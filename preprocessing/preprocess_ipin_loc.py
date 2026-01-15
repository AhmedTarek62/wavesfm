"""Precompute IPIN localization CIR tensors into an HDF5 cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def _coerce_cir(values: Sequence, cir_len: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if cir_len is not None and arr.shape[0] != cir_len:
        raise ValueError(f"Unexpected CIR length: {arr.shape[0]} (expected {cir_len})")
    return arr


def _reduce_anchor_rows(
    rows: pd.DataFrame,
    cir_real_col: str,
    cir_imag_col: str,
    cir_len: int,
    policy: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if rows.empty:
        raise ValueError("No CIR rows available for anchor.")
    if policy == "mean" and len(rows) > 1:
        reals = []
        imags = []
        for real_vals, imag_vals in zip(rows[cir_real_col], rows[cir_imag_col]):
            reals.append(_coerce_cir(real_vals, cir_len))
            imags.append(_coerce_cir(imag_vals, cir_len))
        real = np.stack(reals, axis=0).mean(axis=0)
        imag = np.stack(imags, axis=0).mean(axis=0)
        return real, imag

    row = rows.iloc[0]
    return _coerce_cir(row[cir_real_col], cir_len), _coerce_cir(row[cir_imag_col], cir_len)


def preprocess_ipin_loc(
    data_path: Path,
    output: Path,
    anchors: Iterable[str] | None = None,
    anchor_col: str = "anch_id",
    cir_real_col: str = "cir_real",
    cir_imag_col: str = "cir_imag",
    ref_x_col: str = "ref_x",
    ref_y_col: str = "ref_y",
    rec_time_col: str = "rec_time",
    burst_id_col: str = "burst_id",
    min_anchors: int = 4,
    drop_incomplete: bool = False,
    missing_value: str = "zero",
    anchor_policy: str = "first",
    position_tol: float = 1e-3,
    strict_positions: bool = False,
    batch_size: int = 1024,
    compression: str | None = None,
    overwrite: bool = False,
) -> Path:
    data_path = Path(data_path).expanduser()
    output = Path(output).expanduser()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(data_path)
    if not hasattr(df, "columns"):
        raise TypeError(f"Expected a pandas DataFrame in {data_path}")

    required_cols = [anchor_col, cir_real_col, cir_imag_col, ref_x_col, ref_y_col]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in dataframe")

    df[anchor_col] = df[anchor_col].astype(str)
    anchor_candidates = sorted(df[anchor_col].dropna().unique().tolist())
    if anchors is None:
        anchors_list = anchor_candidates
    else:
        anchors_list = [str(a) for a in anchors]
        missing = sorted(set(anchors_list) - set(anchor_candidates))
        if missing:
            print(f"Warning: requested anchors missing from data: {missing}")

    if not anchors_list:
        raise RuntimeError("No anchors available to build samples.")

    if burst_id_col not in df.columns:
        raise KeyError(f"Missing burst id column '{burst_id_col}' in dataframe")

    first_real = df[cir_real_col].iloc[0]
    first_imag = df[cir_imag_col].iloc[0]
    cir_len = len(_coerce_cir(first_real))
    if len(_coerce_cir(first_imag)) != cir_len:
        raise ValueError("First CIR real/imag lengths do not match.")

    if drop_incomplete:
        min_anchor_count = len(anchors_list)
    else:
        min_anchor_count = int(min_anchors)
    if min_anchor_count <= 0:
        raise ValueError("min_anchors must be positive.")
    if min_anchor_count > len(anchors_list):
        raise ValueError("min_anchors cannot exceed the number of anchors.")

    if missing_value not in ("zero", "nan"):
        raise ValueError("missing_value must be 'zero' or 'nan'.")
    if anchor_policy not in ("first", "mean"):
        raise ValueError("anchor_policy must be 'first' or 'mean'.")

    anchor_set = set(anchors_list)
    grouped = df.groupby(burst_id_col, sort=False)
    total_groups = grouped.ngroups

    valid_keys = []
    dup_groups = 0
    pos_mismatch = 0

    for key, group in tqdm(grouped, total=total_groups, desc="Indexing IPIN groups"):
        present = [a for a in group[anchor_col].unique().tolist() if a in anchor_set]
        present_count = len(present)
        if drop_incomplete:
            if present_count != len(anchors_list):
                continue
        else:
            if present_count < min_anchor_count:
                continue

        loc_vals = group[[ref_x_col, ref_y_col]].to_numpy(dtype=np.float32, copy=False)
        if loc_vals.size == 0:
            continue
        base = loc_vals[0]
        if loc_vals.shape[0] > 1:
            delta = np.max(np.abs(loc_vals - base), axis=0)
            if np.any(delta > position_tol):
                pos_mismatch += 1
                if strict_positions:
                    raise ValueError(f"Position mismatch in group {key}")

        if group[anchor_col].duplicated().any():
            dup_groups += 1
        valid_keys.append(key)

    if not valid_keys:
        raise RuntimeError("No valid samples found. Check burst_id coverage and anchor coverage.")

    dropped = total_groups - len(valid_keys)
    print(f"Total groups: {total_groups} | kept: {len(valid_keys)} | dropped: {dropped}")
    if dup_groups:
        print(f"Groups with duplicate anchors: {dup_groups}")
    if pos_mismatch:
        print(f"Groups with position mismatch > {position_tol}: {pos_mismatch}")

    n = len(valid_keys)
    batch = max(1, int(batch_size))
    chunk = min(batch, n)
    fill_value = np.nan if missing_value == "nan" else 0.0

    has_rec_time = rec_time_col in df.columns
    has_burst_id = burst_id_col in df.columns

    sum_real = 0.0
    sum_imag = 0.0
    sumsq_real = 0.0
    sumsq_imag = 0.0
    total_vals = 0
    loc_min = np.array([np.inf, np.inf], dtype=np.float64)
    loc_max = np.array([-np.inf, -np.inf], dtype=np.float64)

    grouped = df.groupby(burst_id_col, sort=False)
    valid_set = set(valid_keys)

    with h5py.File(output, "w") as h5:
        h5.create_dataset(
            "cir",
            shape=(n, 2, len(anchors_list), cir_len),
            dtype="float32",
            chunks=(chunk, 2, len(anchors_list), cir_len),
            compression=compression,
        )
        h5.create_dataset(
            "location",
            shape=(n, 2),
            dtype="float32",
            chunks=(chunk, 2),
            compression=compression,
        )
        h5.create_dataset(
            "anchor_mask",
            shape=(n, len(anchors_list)),
            dtype="uint8",
            chunks=(chunk, len(anchors_list)),
            compression=compression,
        )
        if has_rec_time:
            h5.create_dataset(
                "rec_time",
                shape=(n, len(anchors_list)),
                dtype="int64",
                chunks=(chunk, len(anchors_list)),
                compression=compression,
            )
        if has_burst_id:
            h5.create_dataset(
                "burst_id",
                shape=(n,),
                dtype="int64",
                chunks=(chunk,),
                compression=compression,
            )

        h5.attrs["anchors"] = json.dumps(list(anchors_list))
        h5.attrs["group_keys"] = json.dumps([burst_id_col])
        h5.attrs["fill_missing"] = missing_value
        h5.attrs["anchor_policy"] = anchor_policy
        h5.attrs["min_anchors"] = int(min_anchor_count)
        h5.attrs["drop_incomplete"] = bool(drop_incomplete)
        h5.attrs["position_tol"] = float(position_tol)
        h5.attrs["source_path"] = str(data_path)
        h5.attrs["cir_len"] = int(cir_len)
        h5.attrs["version"] = "v1"

        write_idx = 0
        batch_idx = 0
        cir_batch = np.empty((batch, 2, len(anchors_list), cir_len), dtype=np.float32)
        loc_batch = np.empty((batch, 2), dtype=np.float32)
        mask_batch = np.zeros((batch, len(anchors_list)), dtype=np.uint8)
        rec_time_batch = (
            np.full((batch, len(anchors_list)), -1, dtype=np.int64) if has_rec_time else None
        )
        burst_batch = np.empty((batch,), dtype=np.int64) if has_burst_id else None

        for key, group in tqdm(grouped, total=total_groups, desc="Writing IPIN samples"):
            if key not in valid_set:
                continue

            cir = np.full((2, len(anchors_list), cir_len), fill_value, dtype=np.float32)
            mask = np.zeros((len(anchors_list),), dtype=np.uint8)
            rec_times = np.full((len(anchors_list),), -1, dtype=np.int64)

            by_anchor = {str(a): rows for a, rows in group.groupby(anchor_col, sort=False)}
            for a_idx, anchor_id in enumerate(anchors_list):
                rows = by_anchor.get(anchor_id)
                if rows is None:
                    continue
                real, imag = _reduce_anchor_rows(rows, cir_real_col, cir_imag_col, cir_len, anchor_policy)
                cir[0, a_idx, :] = real
                cir[1, a_idx, :] = imag
                mask[a_idx] = 1
                if has_rec_time:
                    rec_times[a_idx] = int(rows[rec_time_col].iloc[0])

            loc_vals = group[[ref_x_col, ref_y_col]].to_numpy(dtype=np.float32, copy=False)
            loc = loc_vals[0]

            if has_burst_id:
                burst_vals = group[burst_id_col].to_numpy()
                burst_id = int(burst_vals[0])

            cir_batch[batch_idx] = cir
            loc_batch[batch_idx] = loc
            mask_batch[batch_idx] = mask
            if has_rec_time and rec_time_batch is not None:
                rec_time_batch[batch_idx] = rec_times
            if has_burst_id and burst_batch is not None:
                burst_batch[batch_idx] = burst_id

            if mask.any():
                present = mask.astype(bool)
                real_vals = cir[0, present, :]
                imag_vals = cir[1, present, :]
                sum_real += real_vals.sum()
                sum_imag += imag_vals.sum()
                sumsq_real += np.square(real_vals, dtype=np.float64).sum()
                sumsq_imag += np.square(imag_vals, dtype=np.float64).sum()
                total_vals += real_vals.size

            loc_min = np.minimum(loc_min, loc)
            loc_max = np.maximum(loc_max, loc)

            batch_idx += 1
            if batch_idx == batch:
                sl = slice(write_idx, write_idx + batch_idx)
                h5["cir"][sl] = cir_batch[:batch_idx]
                h5["location"][sl] = loc_batch[:batch_idx]
                h5["anchor_mask"][sl] = mask_batch[:batch_idx]
                if has_rec_time and rec_time_batch is not None:
                    h5["rec_time"][sl] = rec_time_batch[:batch_idx]
                if has_burst_id and burst_batch is not None:
                    h5["burst_id"][sl] = burst_batch[:batch_idx]
                write_idx += batch_idx
                batch_idx = 0

        if batch_idx:
            sl = slice(write_idx, write_idx + batch_idx)
            h5["cir"][sl] = cir_batch[:batch_idx]
            h5["location"][sl] = loc_batch[:batch_idx]
            h5["anchor_mask"][sl] = mask_batch[:batch_idx]
            if has_rec_time and rec_time_batch is not None:
                h5["rec_time"][sl] = rec_time_batch[:batch_idx]
            if has_burst_id and burst_batch is not None:
                h5["burst_id"][sl] = burst_batch[:batch_idx]
            write_idx += batch_idx

        if write_idx != n:
            raise RuntimeError(f"Wrote {write_idx} samples, expected {n}.")

        if total_vals == 0:
            raise RuntimeError("No CIR values available for statistics.")
        mean_real = sum_real / total_vals
        mean_imag = sum_imag / total_vals
        var_real = sumsq_real / total_vals - mean_real ** 2
        var_imag = sumsq_imag / total_vals - mean_imag ** 2

        h5.attrs["mean_real"] = float(mean_real)
        h5.attrs["mean_imag"] = float(mean_imag)
        h5.attrs["std_real"] = float(np.sqrt(max(var_real, 0.0)))
        h5.attrs["std_imag"] = float(np.sqrt(max(var_imag, 0.0)))
        h5.attrs["loc_min"] = loc_min.astype(np.float32)
        h5.attrs["loc_max"] = loc_max.astype(np.float32)

    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess IPIN localization dataset into a cache.")
    p.add_argument("--data-path", required=True, help="Path to pandas pickle file.")
    p.add_argument("--output", default=None, help="Output HDF5 path (default: <input>_clean.h5).")
    p.add_argument("--anchors", nargs="*", default=None, help="Anchor IDs to include (default: all).")
    p.add_argument("--min-anchors", type=int, default=4, help="Minimum anchors per sample (default: 4).")
    p.add_argument("--drop-incomplete", action="store_true", help="Drop samples missing any anchors.")
    p.add_argument(
        "--missing-value",
        default="zero",
        choices=["zero", "nan"],
        help="Fill value for missing anchors (default: zero).",
    )
    p.add_argument(
        "--anchor-policy",
        default="first",
        choices=["first", "mean"],
        help="How to handle duplicate anchor captures (default: first).",
    )
    p.add_argument("--position-tol", type=float, default=1e-3, help="Tolerance for ref_x/ref_y mismatch.")
    p.add_argument("--strict-positions", action="store_true", help="Error on position mismatches.")
    p.add_argument("--batch-size", type=int, default=1024, help="Samples per write batch.")
    p.add_argument(
        "--compression",
        default="none",
        choices=["gzip", "lzf", "none"],
        help="h5 dataset compression (default: none).",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    comp = None if args.compression == "none" else args.compression
    src = Path(args.data_path).expanduser()
    output = Path(args.output).expanduser() if args.output else src.with_name(f"{src.stem}_clean.h5")
    out = preprocess_ipin_loc(
        data_path=src,
        output=output,
        anchors=args.anchors,
        min_anchors=args.min_anchors,
        drop_incomplete=args.drop_incomplete,
        missing_value=args.missing_value,
        anchor_policy=args.anchor_policy,
        position_tol=args.position_tol,
        strict_positions=args.strict_positions,
        batch_size=args.batch_size,
        compression=comp,
        overwrite=args.overwrite,
    )
    print(f"Wrote IPIN cache to {out}")


if __name__ == "__main__":
    main()
