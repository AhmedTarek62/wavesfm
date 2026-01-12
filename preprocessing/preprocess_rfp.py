"""
Chunk RF fingerprinting recordings into standardized IQ tensors and store in HDF5.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import h5py
import numpy as np
from tqdm import tqdm


DEFAULT_LABELS = ["bes", "browning", "honors", "meb"]


def _read_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _list_records(root: Path, allowed_labels: List[str]) -> List[Dict]:
    records: List[Dict] = []
    jsons = sorted(root.glob("*.json"))
    stems_with_bin = {p.stem for p in root.glob("*.bin")}
    label_to_idx = {lbl: i for i, lbl in enumerate(allowed_labels)}

    for jpath in jsons:
        stem = jpath.stem
        if stem not in stems_with_bin:
            continue
        bpath = root / f"{stem}.bin"
        meta = _read_json(jpath)

        dtype_str = meta.get("global", {}).get("core:datatype", "").lower()
        if dtype_str not in ("cf32", "complex64", "float32_complex"):
            continue

        tx = meta.get("annotations", {}).get("transmitter", {}).get("core:location", "").strip().lower()
        if tx not in label_to_idx:
            continue
        y = label_to_idx[tx]

        sr = _safe_float(meta.get("global", {}).get("core:sample_rate"))
        cf = _safe_float(meta.get("captures", {}).get("core:center_frequency"))

        nbytes = os.path.getsize(bpath)
        n_complex = nbytes // np.dtype(np.complex64).itemsize

        records.append(
            dict(
                stem=stem,
                bin_path=str(bpath),
                json_path=str(jpath),
                label_idx=y,
                label_name=tx,
                sample_rate=sr,
                center_freq=cf,
                n_complex=n_complex,
            )
        )
    return records


def preprocess_rfp(
    root: Path,
    output: Path,
    chunk_len: int = 512,
    hop_len: Optional[int] = None,
    allowed_labels: Optional[Iterable[str]] = None,
    compression: str | None = None,
    overwrite: bool = False,
) -> Path:
    root = Path(root)
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = list(allowed_labels) if allowed_labels is not None else list(DEFAULT_LABELS)
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    hop = int(hop_len) if hop_len is not None else int(chunk_len)

    records = _list_records(root, labels)
    if not records:
        raise RuntimeError(f"No usable recordings found under {root}")

    index: List[Dict] = []
    for rec_id, rec in enumerate(records):
        n_chunks = 1 + (rec["n_complex"] - chunk_len) // hop if rec["n_complex"] >= chunk_len else 0
        for c in range(n_chunks):
            index.append(
                dict(
                    rec_id=rec_id,
                    start=c * hop,
                    length=chunk_len,
                )
            )

    if not index:
        raise RuntimeError("No chunks indexed; check chunk_len/hop_len and data length.")

    n = len(index)
    chunk = min(512, n)
    with h5py.File(output, "w") as h5:
        h5.create_dataset(
            "sample",
            shape=(n, 2, 1, chunk_len),
            dtype="float32",
            chunks=(chunk, 2, 1, chunk_len),
            compression=compression,
        )
        h5.create_dataset("label", shape=(n,), dtype="int64", chunks=(chunk,), compression=compression)
        h5.create_dataset(
            "source_file",
            shape=(n,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            compression=compression,
        )
        h5.create_dataset("start", shape=(n,), dtype="int64", chunks=(chunk,), compression=compression)
        h5.create_dataset("sample_rate", shape=(n,), dtype="float32", chunks=(chunk,), compression=compression)
        h5.create_dataset("center_freq", shape=(n,), dtype="float32", chunks=(chunk,), compression=compression)

        h5.attrs["labels"] = json.dumps(labels)
        h5.attrs["chunk_len"] = int(chunk_len)
        h5.attrs["hop_len"] = int(hop)
        h5.attrs["root"] = str(root)
        h5.attrs["version"] = "v1"

        for idx, entry in enumerate(tqdm(index, desc="Caching RF fingerprinting")):
            rec = records[entry["rec_id"]]
            mm = np.memmap(rec["bin_path"], mode="r", dtype=np.complex64)
            seg_c = mm[entry["start"] : entry["start"] + entry["length"]]
            x = np.empty((2, entry["length"]), dtype=np.float32)
            x[0] = seg_c.real.astype(np.float32, copy=False)
            x[1] = seg_c.imag.astype(np.float32, copy=False)
            for ch in range(2):
                mu = x[ch].mean()
                sigma = x[ch].std()
                x[ch] = (x[ch] - mu) / (sigma + 1e-7)

            h5["sample"][idx] = x[:, None, :]
            h5["label"][idx] = rec["label_idx"]
            h5["source_file"][idx] = rec["stem"]
            h5["start"][idx] = entry["start"]
            h5["sample_rate"][idx] = rec["sample_rate"] if rec["sample_rate"] is not None else np.nan
            h5["center_freq"][idx] = rec["center_freq"] if rec["center_freq"] is not None else np.nan

    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute RF fingerprinting cache from .bin/.json pairs.")
    p.add_argument("--root", required=True, help="Directory containing matching *.bin/*.json recordings.")
    p.add_argument("--output", required=True, help="Output HDF5 path.")
    p.add_argument("--chunk-len", type=int, default=512, help="Complex samples per chunk (default: 512).")
    p.add_argument("--hop-len", type=int, default=None, help="Hop between chunks (default: chunk-len).")
    p.add_argument("--labels", type=str, nargs="*", default=None, help="Allowed transmitter labels (default presets).")
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
    out = preprocess_rfp(
        root=Path(args.root),
        output=Path(args.output),
        chunk_len=args.chunk_len,
        hop_len=args.hop_len,
        allowed_labels=args.labels,
        compression=comp,
        overwrite=args.overwrite,
    )
    print(f"Wrote RF fingerprinting cache to {out}")


if __name__ == "__main__":
    main()
