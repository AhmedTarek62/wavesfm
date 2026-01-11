import argparse
from wavesfm.preprocessing import preprocess_iq_task

def parse_args():
    p = argparse.ArgumentParser(description="Preprocess AMC IQ shards into a single HDF5 file.")
    p.add_argument("--data-path", required=True, help="Input shard directory or glob (H5 files with iq_data/modulation/angles).")
    p.add_argument("--output", required=True, help="Destination HDF5 path (e.g., data/amc_train.h5).")
    p.add_argument("--batch-size", type=int, default=256, help="Read/write batch size.")
    p.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    p.add_argument("--compression", default="none", choices=["gzip", "lzf", "none"], help="h5 dataset compression.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output file.")
    return p.parse_args()

def main():
    args = parse_args()
    comp = args.compression
    preprocess_iq_task(
        task="amc",
        data_path=args.data_path,
        output=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        compression=comp,
        overwrite=args.overwrite,
    )
    print(f"Wrote AMC HDF5 to {args.output}")

if __name__ == "__main__":
    main()
