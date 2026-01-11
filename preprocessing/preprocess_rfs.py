import argparse

from wavesfm.preprocessing import preprocess_radio_signals


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess radio spectrogram images into HDF5 tensors.")
    p.add_argument("--data-path", required=True, help="Directory with raw spectrogram images.")
    p.add_argument("--output", required=True, help="Destination HDF5 path (e.g., data/rfs_train.h5).")
    p.add_argument("--img-size", type=int, default=224, help="Resize target (default: 224).")
    p.add_argument("--batch-size", type=int, default=512, help="Write batch size.")
    p.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    p.add_argument("--compression", default="none", choices=["gzip", "lzf", "none"], help="h5 compression.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output file.")
    return p.parse_args()


def main():
    args = parse_args()
    comp = None if args.compression == "none" else args.compression
    preprocess_radio_signals(
        root_dir=args.data_path,
        output=args.output,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        compression=comp,
        overwrite=args.overwrite,
    )
    print(f"Wrote radio-spectrogram HDF5 to {args.output}")


if __name__ == "__main__":
    main()
