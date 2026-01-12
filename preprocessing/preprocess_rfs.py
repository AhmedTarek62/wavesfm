"""Precompute radio signal classification tensors (spectrogram images) into a cache."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import torch
from PIL import Image
from torchvision.transforms import Compose, Grayscale, InterpolationMode, Normalize, Resize, ToTensor
from tqdm import tqdm


LABELS = [
    "ads-b", "airband", "ais", "automatic-picture-transmission", "bluetooth", "cellular",
    "digital-audio-broadcasting", "digital-speech-decoder", "fm", "lora", "morse", "on-off-keying",
    "packet", "pocsag", "Radioteletype", "remote-keyless-entry", "RS41-Radiosonde", "sstv", "vor", "wifi",
]


def preprocess_rfs(
    root_dir: Path,
    output: Path,
    img_size: int = 224,
    batch_size: int = 512,
    num_workers: int = 0,
    compression: str | None = None,
    overwrite: bool = False,
) -> Path:
    root_dir = Path(root_dir)
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    samples = tuple(sorted(f for f in os.listdir(root_dir) if (root_dir / f).is_file()))
    n = len(samples)
    if n == 0:
        raise RuntimeError(f"No radio signal images found in {root_dir}")

    transform = Compose([
        ToTensor(),
        Grayscale(),
        Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        Normalize(mean=[0.5], std=[0.5]),
    ])

    chunk = min(batch_size, n)
    with h5py.File(output, "w") as h5:
        dset = h5.create_dataset(
            "image",
            shape=(n, 1, img_size, img_size),
            dtype="float32",
            chunks=(chunk, 1, img_size, img_size),
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
        h5.attrs["labels"] = json.dumps(LABELS)
        h5.attrs["mean"] = json.dumps([0.5])
        h5.attrs["std"] = json.dumps([0.5])
        h5.attrs["root"] = str(root_dir)
        h5.attrs["version"] = "v1"

        idx = 0
        for name in tqdm(samples, desc="Caching radio signals"):
            img = Image.open(root_dir / name)
            img = img.transpose(Image.ROTATE_90)
            if img.mode != "RGB":
                img = img.convert("RGB")
            tensor = transform(img).float()
            label = name.split("_")[0]

            dset[idx] = tensor
            labels[idx] = LABELS.index(label)
            src[idx] = name
            idx += 1

    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute radio signal ID tensors.")
    p.add_argument("--data-path", required=True, help="Directory containing radio signal image files.")
    p.add_argument("--output", required=True, help="Output path (e.g., data/rfs_cache.h5).")
    p.add_argument("--img-size", type=int, default=224, help="Resize target (default: 224).")
    p.add_argument("--batch-size", type=int, default=512, help="Write batch size (default: 512).")
    p.add_argument("--num-workers", type=int, default=0, help="Dataloader workers (default: 0).")
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
    out = preprocess_rfs(
        root_dir=Path(args.data_path),
        output=Path(args.output),
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        compression=comp,
        overwrite=args.overwrite,
    )
    print(f"Wrote radio signal cache to {out}")


if __name__ == "__main__":
    main()
