"""Precompute positioning tensors: load, normalize, resize features and positions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torchvision.transforms import Compose, Lambda, Normalize, Resize, InterpolationMode
from tqdm import tqdm


SCENE_STATS = {
    "outdoor": {
        "min_val": -0.096,
        "max_val": 1.136,
        "mu": [0.4638, 0.4631, 0.4703, 0.4620],
        "std": [0.1154, 0.1176, 0.0979, 0.1281],
        "coord_min": [0.0, 0.0, 0.0],
        "coord_max": [80.0, 60.0, 40.0],
    },
    "indoor": {
        "min_val": -0.123,
        "max_val": 1.415,
        "mu": [0.3824, 0.3853, 0.3841, 0.3931, 0.3909],
        "std": [0.1168, 0.1112, 0.1182, 0.0988, 0.0972],
        "coord_min": [0.0, 0.0, 0.0],
        "coord_max": [60.0, 20.0, 4.0],
    },
}


def _build_transform(img_size: int, stats: dict) -> Compose:
    return Compose(
        [
            Lambda(lambda x: torch.as_tensor(x, dtype=torch.float32)),
            Resize((img_size, img_size), antialias=True, interpolation=InterpolationMode.BICUBIC),
            Lambda(lambda x: (x - stats["min_val"]) / (stats["max_val"] - stats["min_val"])),
            Normalize(stats["mu"], stats["std"]),
        ]
    )


def _norm_position(pos: np.ndarray, coord_min: np.ndarray, coord_max: np.ndarray) -> np.ndarray:
    denom = np.clip(coord_max - coord_min, 1e-6, None)
    return 2.0 * (pos - coord_min) / denom - 1.0


def preprocess_positioning(
    datapath: Path,
    output: Path,
    img_size: int = 224,
    scene: str = "outdoor",
    batch_size: int = 256,
    compression: str | None = None,
    overwrite: bool = False,
) -> Path:
    datapath = Path(datapath)
    output = Path(output)
    if scene not in SCENE_STATS:
        raise ValueError(f"Unknown scene '{scene}'. Expected one of {tuple(SCENE_STATS)}.")
    stats = SCENE_STATS[scene]

    if output.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    data_files = tuple(sorted(os.listdir(datapath)))
    if not data_files:
        raise RuntimeError(f"No positioning files found in {datapath}")

    transform = _build_transform(img_size, stats)
    coord_min = np.asarray(stats["coord_min"], dtype=np.float32)
    coord_max = np.asarray(stats["coord_max"], dtype=np.float32)

    # Inspect first sample to size datasets.
    with h5py.File(datapath / data_files[0], "r") as f0:
        feat0 = f0["features"][:]
        lbl0 = np.asarray(f0["position"], dtype=np.float32)
    feat0_t = transform(feat0)
    feature_shape = tuple(feat0_t.shape)
    label_shape = tuple(_norm_position(lbl0, coord_min, coord_max).shape)

    n = len(data_files)
    batch = max(1, int(batch_size))
    chunk = min(batch, n)
    with h5py.File(output, "w") as h5:
        h5.create_dataset(
            "features",
            shape=(n, *feature_shape),
            dtype="float32",
            chunks=(chunk, *feature_shape),
            compression=compression,
        )
        h5.create_dataset(
            "label",
            shape=(n, *label_shape) if label_shape else (n,),
            dtype="float32",
            chunks=(chunk, *label_shape) if label_shape else (chunk,),
            compression=compression,
        )
        h5.create_dataset(
            "source_file",
            shape=(n,),
            dtype=h5py.special_dtype(vlen=str),
            compression=compression,
        )
        h5.attrs["scene"] = scene
        h5.attrs["img_size"] = img_size
        h5.attrs["root"] = str(datapath)
        h5.attrs["version"] = "v1"
        h5.attrs["min_val"] = float(stats["min_val"])
        h5.attrs["max_val"] = float(stats["max_val"])
        h5.attrs["mu"] = json.dumps([float(x) for x in stats["mu"]])
        h5.attrs["std"] = json.dumps([float(x) for x in stats["std"]])
        h5.attrs["coord_nominal_min"] = json.dumps([float(x) for x in stats["coord_min"]])
        h5.attrs["coord_nominal_max"] = json.dumps([float(x) for x in stats["coord_max"]])
        
        for start in tqdm(range(0, n, batch), desc="Caching positioning", unit="batch"):
            end = min(start + batch, n)
            batch_files = data_files[start:end]
            batch_len = len(batch_files)
            feats = np.empty((batch_len, *feature_shape), dtype=np.float32)
            labels = np.empty((batch_len, *label_shape), dtype=np.float32)
            src_files = [""] * batch_len

            for j, fname in enumerate(batch_files):
                with h5py.File(datapath / fname, "r") as f:
                    feat = f["features"][:]
                    pos = np.asarray(f["position"], dtype=np.float32)
                feat_t = transform(feat).numpy()
                feats[j] = feat_t
                labels[j] = _norm_position(pos, coord_min, coord_max)
                src_files[j] = fname

            h5["features"][start:end] = feats
            h5["label"][start:end] = labels
            h5["source_file"][start:end] = src_files

    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute positioning tensors into a single cache.")
    p.add_argument("--data-path", required=True, help="Directory containing positioning files.")
    p.add_argument("--output", required=True, help="Output path (e.g., data/positioning_cache.h5).")
    p.add_argument("--scene", default="outdoor", choices=list(SCENE_STATS.keys()), help="Which scene stats to use.")
    p.add_argument("--img-size", type=int, default=224, help="Resize target (default: 224).")
    p.add_argument("--batch-size", type=int, default=256, help="Chunk size for writes (default: 256).")
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
    out = preprocess_positioning(
        datapath=Path(args.data_path),
        output=Path(args.output),
        img_size=args.img_size,
        scene=args.scene,
        batch_size=args.batch_size,
        compression=comp,
        overwrite=args.overwrite,
    )
    print(f"Wrote positioning cache to {out}")


if __name__ == "__main__":
    main()
