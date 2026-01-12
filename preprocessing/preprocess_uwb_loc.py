"""Convert UWB localization JSON dumps into CIR tensors with normalized stats."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import h5py
from tqdm import tqdm


ANCHORS = ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8")
CHANNELS = ("ch1", "ch2", "ch3", "ch4", "ch5", "ch7")


def _parse_location_name(name: str) -> Tuple[float, float, float]:
    x, y, z = name.split("_")
    return float(x), float(y), float(z)


def _cir_to_np(raw_cir: Sequence) -> np.ndarray:
    vals = [complex(v) for v in raw_cir]
    return np.asarray(vals, dtype=np.complex64)


def preprocess_uwb(
    root: Path,
    environment: str,
    output: Path,
    batch_size: int = 1024,
    anchors: Iterable[str] = ANCHORS,
    channels: Iterable[str] = CHANNELS,
) -> Path:
    env_dir = root / environment
    with (env_dir / "data.json").open("r") as f:
        data = json.load(f)
    measurements = data["measurements"]  # location -> anchor_id -> channel -> list[dict]
    selected_anchors = tuple(anchors)
    selected_channels = tuple(channels)

    index: List[Tuple[str, str, int]] = []
    for loc_name in measurements.keys():
        for ch_name in selected_channels:
            if any(
                anchor_id not in measurements[loc_name]
                or ch_name not in measurements[loc_name][anchor_id]
                for anchor_id in selected_anchors
            ):
                continue

            min_count = min(len(measurements[loc_name][a][ch_name]) for a in selected_anchors)
            for m_idx in range(min_count):
                index.append((loc_name, ch_name, m_idx))

    if not index:
        raise RuntimeError("No samples found. Check anchors/channels selections.")

    first_loc, first_ch, first_m = index[0]
    first_entry = measurements[first_loc][selected_anchors[0]][first_ch][first_m]
    cir_len = len(_cir_to_np(first_entry["cir"]))

    output.parent.mkdir(parents=True, exist_ok=True)

    sum_real = 0.0
    sum_imag = 0.0
    sumsq_real = 0.0
    sumsq_imag = 0.0
    total_vals_per_component = 0
    loc_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    loc_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    with h5py.File(output, "w") as h5:
        h5.create_dataset("cir", shape=(len(index), 2, len(selected_anchors), cir_len), dtype="float32", compression=None)
        h5.create_dataset("location", shape=(len(index), 3), dtype="float32")
        h5.create_dataset("channel", shape=(len(index),), dtype="int16")
        h5.attrs["environment"] = environment
        h5.attrs["anchors"] = json.dumps(list(selected_anchors))
        h5.attrs["channels"] = json.dumps(list(selected_channels))

        total = len(index)
        for start in tqdm(range(0, total, batch_size), desc="Writing UWB samples"):
            end = min(start + batch_size, total)
            chunk = index[start:end]

            cir_batch = np.empty((len(chunk), 2, len(selected_anchors), cir_len), dtype=np.float32)
            loc_batch = np.empty((len(chunk), 3), dtype=np.float32)
            ch_batch = np.empty((len(chunk),), dtype=np.int16)

            for j, (loc_name, ch_name, meas_idx) in enumerate(chunk):
                cirs: List[np.ndarray] = []
                for anchor_id in selected_anchors:
                    entry = measurements[loc_name][anchor_id][ch_name][meas_idx]
                    cir_raw = entry["cir"]
                    loc_from_entry = (float(entry["x_tag"]), float(entry["y_tag"]), float(entry["z_tag"]))
                    loc_from_path = _parse_location_name(loc_name)
                    if not np.allclose(loc_from_entry, loc_from_path, atol=1e-6):
                        raise ValueError(f"Location mismatch at {loc_name}, anchor {anchor_id}, ch {ch_name}")
                    cirs.append(_cir_to_np(cir_raw))

                cir_arr = np.stack(cirs, axis=0)  # (A, L)
                real = cir_arr.real.astype(np.float32, copy=False)
                imag = cir_arr.imag.astype(np.float32, copy=False)
                cir_batch[j] = np.stack((real, imag), axis=0)  # (2, A, L)
                loc_batch[j] = np.asarray(_parse_location_name(loc_name), dtype=np.float32)
                ch_batch[j] = int(ch_name.replace("ch", "")) if ch_name.startswith("ch") else -1

            h5["cir"][start:end] = cir_batch
            h5["location"][start:end] = loc_batch
            h5["channel"][start:end] = ch_batch

            sum_real += cir_batch[:, 0].sum()
            sum_imag += cir_batch[:, 1].sum()
            sumsq_real += np.square(cir_batch[:, 0], dtype=np.float64).sum()
            sumsq_imag += np.square(cir_batch[:, 1], dtype=np.float64).sum()
            total_vals_per_component += cir_batch[:, 0].size
            loc_min = np.minimum(loc_min, loc_batch.min(axis=0))
            loc_max = np.maximum(loc_max, loc_batch.max(axis=0))

        mean_real = sum_real / total_vals_per_component
        mean_imag = sum_imag / total_vals_per_component
        var_real = sumsq_real / total_vals_per_component - mean_real**2
        var_imag = sumsq_imag / total_vals_per_component - mean_imag**2
        h5.attrs["mean_real"] = float(mean_real)
        h5.attrs["mean_imag"] = float(mean_imag)
        h5.attrs["std_real"] = float(np.sqrt(max(var_real, 0.0)))
        h5.attrs["std_imag"] = float(np.sqrt(max(var_imag, 0.0)))
        h5.attrs["loc_min"] = loc_min.astype(np.float32)
        h5.attrs["loc_max"] = loc_max.astype(np.float32)

    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess UWB dataset JSON into a cache.")
    p.add_argument("--root", default="~/data/uwb_loc", help="Dataset root containing environment folders.")
    p.add_argument("--environment", default="environment0", help="Environment folder name (e.g., environment0).")
    p.add_argument("--output", default=None, help="Output path (default: <root>/<environment>_clean.h5).")
    p.add_argument("--batch-size", type=int, default=1024, help="Number of samples per write batch.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser()
    env = args.environment
    output = Path(args.output).expanduser() if args.output else root / f"{env}_clean.h5"

    out = preprocess_uwb(root, env, output, args.batch_size)
    print(f"Wrote cleaned dataset to {out}")


if __name__ == "__main__":
    main()
