"""Precompute Icarus Powder interference detection tensors with normalization and center-crop."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


MU = np.asarray([0.0, 0.0], dtype=np.float32)
STD = np.asarray([0.00275, 0.00260], dtype=np.float32)


def _infer_iq_dtype(nbytes: int, expected_n_complex: int):
    # (name, dtype, scale, bytes-per-complex)
    cands = [
        ("float32", np.float32, None, 8),  # 2 * float32
        ("int16", np.int16, 32768.0, 4),   # 2 * int16
        ("int8", np.int8, 128.0, 2),       # 2 * int8
    ]
    best = min(
        ((abs(nbytes - expected_n_complex * bpc), name, dt, scale, bpc) for name, dt, scale, bpc in cands),
        key=lambda x: x[0],
    )
    _, name, dt, scale, bpc = best
    return name, dt, scale, bpc


def _expected_n_complex(row: pd.Series) -> int:
    if "No_of_Samples" in row:
        return int(row["No_of_Samples"])
    fs = row["Sampling_Rate"]
    dur = row["Frameduration"]
    assert fs is not None and dur is not None, "CSV must contain either No_of_Samples or both Sampling_Rate and Frameduration"
    return int(round(float(fs) * float(dur)))


def _load_interleaved_iq(bin_file: Path, meta_row: pd.Series):
    """Load interleaved I,Q -> complex64 using CSV's Sampling_Rate & Frameduration."""
    n_exp = _expected_n_complex(meta_row)
    nbytes = bin_file.stat().st_size

    _, dtype, scale, _ = _infer_iq_dtype(nbytes, n_exp)
    raw = np.fromfile(bin_file, dtype=dtype)
    if raw.size % 2:
        raw = raw[:-1]

    I = raw[0::2].astype(np.float32, copy=False)
    Q = raw[1::2].astype(np.float32, copy=False)
    if scale is not None:
        I /= scale
        Q /= scale

    z = I + 1j * Q
    if abs(z.size - n_exp) <= 2:
        z = z[:n_exp]
    return z.astype(np.complex64)


def _encode_label(signal_type: str, dsss_mod):
    """0 for LTE only; else the integer DSSS_Mod (>=1)."""
    if str(signal_type).strip().lower() == "lte":
        return 0
    try:
        k = int(np.log2(float(dsss_mod)))
        return k
    except Exception:
        return 0


def _band_from_batchdir(batch_dir_name: str) -> int:
    s = batch_dir_name.lower()
    if "10mhz" in s or "10_mhz" in s or "10-mhz" in s:
        return 10
    if "5mhz" in s or "5_mhz" in s or "5-mhz" in s:
        return 5
    raise ValueError(f"Cannot infer band from dir name: {batch_dir_name}")


def _collect_items(root: Path, csv_subdir: str, iq_subdir: str) -> List[dict]:
    items: List[dict] = []
    for batch in sorted(root.iterdir()):
        if not batch.is_dir():
            continue

        band = _band_from_batchdir(batch.name)
        csv_dir = batch / csv_subdir
        iq_dir = batch / iq_subdir
        if not (csv_dir.exists() and iq_dir.exists()):
            continue

        for csv_path in sorted(csv_dir.glob("*.csv")):
            stem = csv_path.stem
            cand1 = iq_dir / stem
            cand2 = iq_dir / f"{stem}.bin"
            if cand1.exists():
                bin_path = cand1
            elif cand2.exists():
                bin_path = cand2
            else:
                continue

            df = pd.read_csv(csv_path)
            if len(df) == 0:
                continue
            row = df.iloc[0]
            fs = float(row["Sampling_Rate"])
            items.append({"csv": csv_path, "bin": bin_path, "row": row, "fs": fs, "band": band})
    return items


def preprocess_icarus(
    root: Path,
    output: Path,
    max_len: int = 4096,
    csv_subdir: str = "Metadata",
    iq_subdir: str = "IQ",
    compression: str | None = None,
    overwrite: bool = False,
) -> Path:
    root = Path(root)
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    items = _collect_items(root, csv_subdir, iq_subdir)
    if not items:
        raise RuntimeError(f"No (csv,bin) pairs found under {root}")

    n = len(items)
    chunk = min(256, n)
    counts = np.zeros(3, dtype=np.int64)  # LTE=0, DSSS_mod log2 variants
    with h5py.File(output, "w") as h5:
        h5.create_dataset(
            "sample",
            shape=(n, 2, 1, max_len),
            dtype="float32",
            chunks=(chunk, 2, 1, max_len),
            compression=compression,
        )
        h5.create_dataset("label", shape=(n,), dtype="int64", chunks=(chunk,), compression=compression)
        h5.create_dataset("band", shape=(n,), dtype="int8", chunks=(chunk,), compression=compression)
        h5.create_dataset("fs", shape=(n,), dtype="float32", chunks=(chunk,), compression=compression)
        h5.create_dataset(
            "source_file",
            shape=(n,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            compression=compression,
        )
        h5.attrs["root"] = str(root)
        h5.attrs["version"] = "v1"
        h5.attrs["mu"] = json.dumps([float(x) for x in MU])
        h5.attrs["std"] = json.dumps([float(x) for x in STD])
        h5.attrs["max_len"] = int(max_len)

        for idx, it in enumerate(tqdm(items, desc="Caching Icarus")):
            z = _load_interleaved_iq(it["bin"], it["row"])
            x = np.stack([z.real, z.imag], axis=0).astype(np.float32)
            midpoint = x.shape[1] // 2
            if x.shape[1] < max_len:
                raise ValueError(f"Signal shorter than max_len at {it['bin']}")
            x = x[:, midpoint - max_len // 2: midpoint + max_len // 2]
            x = (x - MU[:, None]) / STD[:, None]

            h5["sample"][idx] = x[:, None, :]
            lbl = _encode_label(it["row"].get("Signal_Type", ""), it["row"].get("DSSS_Mod", None))
            counts[min(lbl, 2)] += 1  # clamp unexpected to last bin
            h5["label"][idx] = lbl
            h5["band"][idx] = int(it["band"])
            h5["fs"][idx] = float(it["fs"])
            h5["source_file"][idx] = str(it["bin"])

        freq = counts.astype(np.float64) / max(1, counts.sum())
        weights = np.where(freq > 0, 1.0 / freq, 0.0)
        weights = weights / weights.sum().clip(min=1e-8)
        h5.attrs["class_weights"] = weights.astype(np.float32)

    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute Icarus Powder interference detection cache.")
    p.add_argument("--root", required=True, help="Root directory containing batch folders.")
    p.add_argument("--output", required=True, help="Output HDF5 path.")
    p.add_argument("--max-len", type=int, default=4096, help="Center-cropped complex length (default: 4096).")
    p.add_argument("--csv-subdir", default="Metadata", help="Metadata subdirectory name (default: Metadata).")
    p.add_argument("--iq-subdir", default="IQ", help="IQ subdirectory name (default: IQ).")
    p.add_argument(
        "--compression",
        default="none",
        choices=["gzip", "lzf", "none"],
        help="Dataset compression (default: none).",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    comp = None if args.compression == "none" else args.compression
    out = preprocess_icarus(
        root=Path(args.root),
        output=Path(args.output),
        max_len=args.max_len,
        csv_subdir=args.csv_subdir,
        iq_subdir=args.iq_subdir,
        compression=comp,
        overwrite=args.overwrite,
    )
    print(f"Wrote Icarus cache to {out}")


if __name__ == "__main__":
    main()
