import argparse

from wavesfm.preprocessing import preprocess_sensing


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess CSI sensing data into HDF5 tensors.")
    p.add_argument("--data-path", required=True, help="Root directory with raw sensing samples.")
    p.add_argument("--output", required=True, help="Destination HDF5 path (e.g., data/sensing_train.h5).")
    p.add_argument("--img-size", type=int, default=224, help="Resize target (default: 224).")
    p.add_argument("--batch-size", type=int, default=256, help="Write batch size.")
    p.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    p.add_argument("--compression", default="none", choices=["gzip", "lzf", "none"], help="h5 compression.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output file.")
    return p.parse_args()


def main():
    args = parse_args()
    comp = None if args.compression == "none" else args.compression
    preprocess_sensing(
        data_path=args.data_path,
        output=args.output,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        compression=comp,
        overwrite=args.overwrite,
    )
    print(f"Wrote sensing HDF5 to {args.output}")


if __name__ == "__main__":
    main()
