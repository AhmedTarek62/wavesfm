"""
Flatten RML 2016/2022 datasets into an HDF5 cache with normalized IQ samples.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


LABELS = (
    "8PSK",
    "AM-DSB",
    "AM-SSB",
    "BPSK",
    "CPFSK",
    "GFSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "QPSK",
    "WBFM",
)

STATS = {
    "2016": {"mu": np.array((0.0, 0.0), dtype=np.float32).reshape(2, 1), "std": np.array((0.0058, 0.0062), dtype=np.float32).reshape(2, 1)},
    "2022": {"mu": np.array((0.0, 0.0), dtype=np.float32).reshape(2, 1), "std": np.array((2.925, 2.924), dtype=np.float32).reshape(2, 1)},
}


def _load_data(root: Path, version: str):
    if version == "2022":
        return np.load(root / "RML22.01A", allow_pickle=True)
    with open(root / "RML2016.10a_dict.pkl", "rb") as f:
        return pickle.load(f, encoding="latin1")


def preprocess_rml(
    root: Path,
    version: str,
    output: Path,
    batch_size: int = 1024,
    compression: str | None = None,
    overwrite: bool = False,
) -> Path:
    root = Path(root)
    output = Path(output)
    if version not in STATS:
        raise ValueError("version must be '2016' or '2022'")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    data = _load_data(root, version)
    keys = list(data.keys())  # (mod, snr)
    if not keys:
        raise RuntimeError(f"No samples found in {root}")

    total = sum(data[k].shape[0] for k in keys)
    # Inspect first sample for length
    first_arr = data[keys[0]]
    sample_len = first_arr.shape[-1]
    mu = STATS[version]["mu"]
    std = STATS[version]["std"]
    chunk = min(batch_size, total)

    with h5py.File(output, "w") as h5:
        h5.create_dataset(
            "sample",
            shape=(total, 2, 1, sample_len),
            dtype="float32",
            chunks=(chunk, 2, 1, sample_len),
            compression=compression,
        )
        h5.create_dataset("label", shape=(total,), dtype="int64", chunks=(chunk,), compression=compression)
        h5.create_dataset("snr", shape=(total,), dtype="int16", chunks=(chunk,), compression=compression)
        h5.create_dataset(
            "modulation",
            shape=(total,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            chunks=(chunk,),
            compression=compression,
        )
        h5.attrs["root"] = str(root)
        h5.attrs["version"] = version
        h5.attrs["labels"] = json.dumps(list(LABELS))
        h5.attrs["mu"] = json.dumps([float(x) for x in mu.flatten()])
        h5.attrs["std"] = json.dumps([float(x) for x in std.flatten()])
        h5.attrs["sample_len"] = int(sample_len)

        idx = 0
        for mod, snr in tqdm(keys, desc="Caching RML"):
            arr = data[(mod, snr)].astype(np.float32, copy=False)
            n = arr.shape[0]
            arr = (arr - mu) / std  # broadcast over channel axis
            labels = np.full((n,), LABELS.index(mod), dtype=np.int64)
            snr_vals = np.full((n,), int(snr), dtype=np.int16)

            h5["sample"][idx : idx + n] = arr[:, :, None, :]
            h5["label"][idx : idx + n] = labels
            h5["snr"][idx : idx + n] = snr_vals
            h5["modulation"][idx : idx + n] = np.asarray([mod] * n, dtype=object)
            idx += n

    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute RML dataset into an HDF5 cache.")
    p.add_argument("--root", required=True, help="Directory containing RML files (RML2016.10a_dict.pkl or RML22.01A).")
    p.add_argument("--version", required=True, choices=["2016", "2022"], help="Dataset version to process.")
    p.add_argument("--output", required=True, help="Output HDF5 path.")
    p.add_argument("--batch-size", type=int, default=1024, help="Chunk size for HDF5 writes (default: 1024).")
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
    out = preprocess_rml(
        root=Path(args.root),
        version=args.version,
        output=Path(args.output),
        batch_size=args.batch_size,
        compression=comp,
        overwrite=args.overwrite,
    )
    print(f"Wrote RML cache to {out}")


if __name__ == "__main__":
    main()
