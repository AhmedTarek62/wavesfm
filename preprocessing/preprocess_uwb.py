import argparse

from wavesfm.preprocessing import preprocess_uwb


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess UWB positioning data into HDF5 tensors.")
    p.add_argument("--root", required=True, help="Root directory containing UWB raw data (with environment subfolders).")
    p.add_argument("--environment", required=True, help="Environment name under root (e.g., indoor/outdoor).")
    p.add_argument("--output", required=True, help="Destination HDF5 path (e.g., data/uwb_train.h5).")
    p.add_argument("--batch-size", type=int, default=64, help="Write batch size.")
    p.add_argument("--compression", default="none", choices=["gzip", "lzf", "none"], help="h5 compression.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output file.")
    return p.parse_args()


def main():
    args = parse_args()
    comp = None if args.compression == "none" else args.compression
    preprocess_uwb(
        root=args.root,
        environment=args.environment,
        output=args.output,
        batch_size=args.batch_size,
        compression=comp,
        overwrite=args.overwrite,
    )
    print(f"Wrote UWB HDF5 to {args.output}")


if __name__ == "__main__":
    main()
