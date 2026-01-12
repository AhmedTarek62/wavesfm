"""Repack RadCom OTA keyed by tuple strings into normalized, column-style datasets."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import h5py
import numpy as np
from tqdm import tqdm


def _parse_key(key: str) -> Tuple[str, str, str, str]:
    parsed = ast.literal_eval(key)
    modulation, signal_type, snr, sample_idx = parsed
    return str(modulation), str(signal_type), str(snr), str(sample_idx)


def _to_2ct(sample: np.ndarray) -> np.ndarray:
    arr = np.asarray(sample, dtype=np.float32)
    i_part = arr[:128]
    q_part = arr[128:]
    return np.stack([i_part, q_part], axis=0)[:, None, :]


def _maybe_parse_snr(values: Sequence[str]) -> Tuple[bool, List[float] | List[str]]:
    parsed: List[float] = []
    for val in values:
        try:
            parsed.append(float(val))
        except (TypeError, ValueError):
            return False, list(values)
    return True, parsed


def _compute_stats(src: h5py.File, keys: list[str], batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    sum_ch = np.zeros(2, dtype=np.float64)
    sumsq_ch = np.zeros_like(sum_ch)
    total = 0
    for start in tqdm(range(0, len(keys), batch_size), desc="Pass 1: stats", unit="sample"):
        end = min(start + batch_size, len(keys))
        batch_keys = keys[start:end]
        samples = np.stack([_to_2ct(src[k][:]) for k in batch_keys], axis=0)
        sum_ch += samples.sum(axis=(0, 2, 3))
        sumsq_ch += np.square(samples, dtype=np.float64).sum(axis=(0, 2, 3))
        total += samples.shape[0] * samples.shape[2] * samples.shape[3]
    mean = sum_ch / float(total)
    var = sumsq_ch / float(total) - np.square(mean)
    std = np.sqrt(np.clip(var, 1e-12, None))
    return mean.astype(np.float32), std.astype(np.float32)


def preprocess_radcom(
    input_path: Path,
    output: Path,
    batch_size: int = 256,
    compression: str | None = None,
    sort_keys: bool = True,
    overwrite: bool = False,
) -> Path:
    input_path = Path(input_path)
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as src:
        raw_keys = sorted(src.keys()) if sort_keys else list(src.keys())
        parsed_keys = [_parse_key(k) for k in raw_keys]
        label_pairs = sorted({(m, s) for m, s, _, _ in parsed_keys})
        label_map = {pair: i for i, pair in enumerate(label_pairs)}
        sample_shape_raw = src[raw_keys[0]].shape

        mean, std = _compute_stats(src, raw_keys, batch_size)
        snr_tokens = [snr for _, _, snr, _ in parsed_keys]
        snr_is_float, snr_values = _maybe_parse_snr(snr_tokens)

    n = len(raw_keys)
    sample_example = _to_2ct(np.zeros(sample_shape_raw, dtype=np.float32))
    sample_shape = sample_example.shape
    chunk = min(batch_size, n)
    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output, "w") as dst, h5py.File(input_path, "r") as src:
        dst.create_dataset(
            "sample",
            shape=(n, *sample_shape),
            dtype="float32",
            chunks=(chunk, *sample_shape),
            compression=compression,
        )
        dst.create_dataset("label", shape=(n,), dtype="int64", chunks=(chunk,), compression=compression)
        dst.create_dataset("modulation", shape=(n,), dtype=str_dtype, chunks=(chunk,), compression=compression)
        dst.create_dataset("signal_type", shape=(n,), dtype=str_dtype, chunks=(chunk,), compression=compression)
        snr_dtype = "float32" if snr_is_float else str_dtype
        dst.create_dataset("snr", shape=(n,), dtype=snr_dtype, chunks=(chunk,), compression=compression)

        dst.attrs["source"] = str(input_path)
        dst.attrs["version"] = "v2"
        dst.attrs["sorted_keys"] = bool(sort_keys)
        dst.attrs["sample_shape_raw"] = json.dumps(list(sample_shape_raw))
        dst.attrs["sample_shape"] = json.dumps(list(sample_shape))
        dst.attrs["sample_dtype"] = "float32"
        dst.attrs["label_pairs"] = json.dumps(label_pairs)
        dst.attrs["mean"] = mean
        dst.attrs["std"] = std

        counts = np.zeros(len(label_pairs), dtype=np.int64)
        idx = 0
        for start in tqdm(range(0, n, batch_size), desc="Pass 2: write", unit="sample"):
            end = min(start + batch_size, n)
            keys_batch = raw_keys[start:end]
            mods = []
            sigs = []
            snrs = []
            samples = []
            labels = []
            for key in keys_batch:
                mod, sig, snr, _ = _parse_key(key)
                sample = _to_2ct(src[key][:])
                sample = (sample - mean[:, None, None]) / std[:, None, None]
                samples.append(sample)
                lbl = label_map[(mod, sig)]
                labels.append(lbl)
                counts[lbl] += 1
                mods.append(mod)
                sigs.append(sig)
                snrs.append(snr)

            batch_samples = np.stack(samples, axis=0)
            dst["sample"][idx:idx + len(batch_samples)] = batch_samples
            dst["label"][idx:idx + len(batch_samples)] = np.asarray(labels, dtype=np.int64)
            dst["modulation"][idx:idx + len(batch_samples)] = np.asarray(mods, dtype=object)
            if snr_is_float:
                dst["snr"][idx:idx + len(batch_samples)] = np.asarray(snr_values[idx:idx + len(batch_samples)], dtype=np.float32)
            else:
                dst["snr"][idx:idx + len(batch_samples)] = np.asarray(snrs, dtype=object)
            dst["signal_type"][idx:idx + len(batch_samples)] = np.asarray(sigs, dtype=object)
            idx += len(batch_samples)

        freq = counts.astype(np.float64) / max(1, counts.sum())
        weights = np.where(freq > 0, 1.0 / freq, 0.0)
        weights = weights / weights.sum().clip(min=1e-8)
        dst.attrs["class_weights"] = weights.astype(np.float32)

    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repack and normalize RadCom OTA keyed by tuple labels.")
    p.add_argument("--input", required=True, help="Source file with tuple keys (modulation, signal_type, snr, sample_idx).")
    p.add_argument("--output", required=True, help="Destination path for reorganized arrays.")
    p.add_argument("--batch-size", type=int, default=256, help="Chunk size for output datasets (default: 256).")
    p.add_argument(
        "--compression",
        default="none",
        choices=["gzip", "lzf", "none"],
        help="Dataset compression (default: none).",
    )
    p.add_argument("--no-sort", action="store_true", help="Keep original key order instead of sorting.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    comp = None if args.compression == "none" else args.compression
    out = preprocess_radcom(
        input_path=Path(args.input),
        output=Path(args.output),
        batch_size=args.batch_size,
        compression=comp,
        sort_keys=not args.no_sort,
        overwrite=args.overwrite,
    )
    print(f"Wrote reorganized RadCom cache to {out}")


if __name__ == "__main__":
    main()
