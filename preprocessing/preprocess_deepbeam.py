import argparse

from wavesfm.preprocessing import preprocess_iq_task


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess DeepBeam dataset into HDF5.")
    p.add_argument("--data-path", required=True, help="Root of DeepBeam dataset.")
    p.add_argument("--output", required=True, help="Destination HDF5 path (e.g., data/deepbeam_train.h5).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--compression", default="none", choices=["gzip", "lzf", "none"])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    preprocess_iq_task(
        task="deepbeam",
        data_path=args.data_path,
        output=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        compression=args.compression,
        overwrite=args.overwrite,
    )
    print(f"Wrote DeepBeam HDF5 to {args.output}")


if __name__ == "__main__":
    main()
