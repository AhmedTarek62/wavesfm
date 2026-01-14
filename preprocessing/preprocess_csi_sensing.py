"""Precompute CSI sensing tensors with deterministic load/normalize/resize."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import h5py
from scipy.io import loadmat
import torch
from torchvision.transforms import Compose, Lambda, Resize, InterpolationMode, Normalize
from tqdm import tqdm


LABELS = ("run", "walk", "fall", "box", "circle", "clean")
MIN_VAL = 2.44
MAX_VAL = 54.72
MU = [0.7396, 0.7722, 0.7758]
STD = [0.0960, 0.0764, 0.0888]


def _build_transform(img_size: int) -> Compose:
    return Compose(
        [
            Lambda(lambda x: torch.as_tensor(x, dtype=torch.float32)),
            Resize((img_size, img_size), antialias=True, interpolation=InterpolationMode.BICUBIC),
            Lambda(lambda x: (x - MIN_VAL) / (MAX_VAL - MIN_VAL)),
            Normalize(MU, STD),
        ]
    )


def _label_from_name(name: str) -> int:
    match = re.match(r"([a-zA-Z]+)(\d+)", name)
    if not match:
        raise ValueError(f"Unexpected filename format (expected <label><index>): {name}")
    label_name = match.group(1)
    return LABELS.index(label_name)


def preprocess_csi_sensing(
    data_path: Path,
    output: Path,
    img_size: int = 224,
    batch_size: int = 256,
    compression: str | None = None,
    overwrite: bool = False,
) -> Path:
    root_dir = Path(data_path)
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    file_list = tuple(sorted(os.listdir(root_dir)))
    if not file_list:
        raise RuntimeError(f"No CSI sensing files found in {root_dir}")

    transform = _build_transform(img_size)
    n = len(file_list)
    batch = max(1, int(batch_size))
    chunk = min(batch, n)

    with h5py.File(output, "w") as h5:
        dset = h5.create_dataset(
            "csi",
            shape=(n, 3, img_size, img_size),
            dtype="float32",
            chunks=(chunk, 3, img_size, img_size),
            compression=compression,
        )
        labels = h5.create_dataset(
            "label",
            shape=(n,),
            dtype="int64",
            chunks=(chunk,),
            compression=compression,
        )
        src = h5.create_dataset(
            "source_file",
            shape=(n,),
            dtype=h5py.special_dtype(vlen=str),
            compression=compression,
        )
        h5.attrs["img_size"] = img_size
        h5.attrs["labels"] = json.dumps(list(LABELS))
        h5.attrs["min_val"] = float(MIN_VAL)
        h5.attrs["max_val"] = float(MAX_VAL)
        h5.attrs["mu"] = json.dumps([float(x) for x in MU])
        h5.attrs["std"] = json.dumps([float(x) for x in STD])
        h5.attrs["root"] = str(root_dir)
        h5.attrs["version"] = "v1"

        for start in tqdm(range(0, n, batch), desc="Caching CSI sensing", unit="batch"):
            end = min(start + batch, n)
            batch_names = file_list[start:end]
            csi_batch = []
            label_batch = []
            for sample_name in batch_names:
                csi = loadmat(root_dir / sample_name)["CSIamp"].reshape(3, 114, -1)
                csi = transform(csi)
                label_index = _label_from_name(sample_name)
                csi_batch.append(csi)
                label_batch.append(label_index)

            dset[start:end] = torch.stack(csi_batch, dim=0).cpu().numpy()
            labels[start:end] = label_batch
            src[start:end] = batch_names

    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute CSI sensing tensors for fine-tuning/eval.")
    p.add_argument("--data-path", required=True, help="Directory containing sensing CSI .mat files.")
    p.add_argument("--output", required=True, help="Output path (e.g., data/csi_sensing_cache.h5).")
    p.add_argument("--img-size", type=int, default=224, help="Resize target (default: 224).")
    p.add_argument("--batch-size", type=int, default=256, help="Chunk size for writes (default: 256).")
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
    out = preprocess_csi_sensing(
        data_path=Path(args.data_path),
        output=Path(args.output),
        img_size=args.img_size,
        batch_size=args.batch_size,
        compression=comp,
        overwrite=args.overwrite,
    )
    print(f"Wrote CSI sensing cache to {out}")


if __name__ == "__main__":
    main()
