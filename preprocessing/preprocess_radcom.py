import argparse

from wavesfm.preprocessing import preprocess_radcom_ota


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess RadCom OTA data into HDF5 tensors.")
    p.add_argument("--input", required=True, help="Input RadCom OTA .h5 file (raw).")
    p.add_argument("--output", required=True, help="Destination HDF5 path (e.g., data/radcom.h5).")
    p.add_argument("--batch-size", type=int, default=256, help="Write batch size.")
    p.add_argument("--compression", default="none", choices=["gzip", "lzf", "none"], help="h5 compression.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output file.")
    p.add_argument("--no-sort", action="store_true", help="Skip sorting input keys before writing.")
    return p.parse_args()


def main():
    args = parse_args()
    comp = None if args.compression == "none" else args.compression
    preprocess_radcom_ota(
        input_path=args.input,
        output=args.output,
        batch_size=args.batch_size,
        compression=comp,
        sort_keys=not args.no_sort,
        overwrite=args.overwrite,
    )
    print(f"Wrote RadCom HDF5 to {args.output}")


if __name__ == "__main__":
    main()
