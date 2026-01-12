# WavesFM fine-tuning (HDF5-first)

A lightweight, public-friendly entrypoint for reproducing our multimodal fine-tuning runs. Training only touches preprocessed HDF5 files; a small preprocessing package is included for users who need to generate those files from the original datasets.

## Supported tasks
- IQ tasks: `amc`, `aoa`, `rml`, `rfp`, `deepbeam`, `interf`
- Vision tasks: `sensing`, `rfs`
- Positioning: `pos` (5G), `uwb`
- RadCom OTA: `radcom`

Dataset loaders are HDF5-only and live under `wavesfm/dataset_classes/`; `wavesfm/data.py` wires them up per task.

## Preprocessing
Use the per-dataset scripts under `wavesfm/preprocessing/` (mirroring the original mae layout) to turn raw data into a single HDF5 file per split.

Examples:
```bash
# CSI sensing
python wavesfm/preprocessing/preprocess_csi_sensing.py --csi-path data/sensing/raw/train --output data/sensing_train.h5

# 5G positioning
python wavesfm/preprocessing/preprocess_nr_positioning.py --data-path data/pos/raw/outdoor/train --output data/pos_train.h5 --scene outdoor

# RadCom OTA
python wavesfm/preprocessing/preprocess_radcom.py --input data/radcom/raw.h5 --output data/radcom_all.h5

# UWB positioning
python wavesfm/preprocessing/preprocess_uwb_loc.py --root data/uwb/raw --environment indoor --output data/uwb_train.h5 --batch-size 64

# RML (amc) / RF fingerprinting / interference (Icarus)
python wavesfm/preprocessing/preprocess_rml.py --root data/rml --version 2016 --output data/rml_train.h5
python wavesfm/preprocessing/preprocess_rfp.py --root data/rfp --output data/rfp_train.h5
python wavesfm/preprocessing/preprocess_icarus.py --root data/icarus --output data/icarus_train.h5
```

The IQ helper writes:
- `iq`: float32 `(N, 2, C, T)`
- `label` (classification) or `target` (regression)
- attrs: `mean`, `std`, `labels` (if available), `task`

Other preprocessors reuse the original scripts under `preprocessing/` to keep parity with our pipelines.

## Fine-tuning
Run the unified entrypoint:
```bash
python wavesfm/main_finetune_multi.py \
  --task amc \
  --train-data data/amc_train.h5 \
  --val-data data/amc_val.h5 \
  --model vit_multi_small \
  --finetune checkpoints/pretrained.pth \
  --batch-size 64 --epochs 50 \
  --output-dir runs/amc-small
```

Notes:
- Training/eval only use the HDF5 files; no raw preprocessing happens in the dataloaders.
- Logging is JSONL to `output_dir/log.txt` plus checkpoints (`best.pth` and periodic `checkpoint_*.pth`).
- `--eval-only` runs validation without training.

## Extending
- To add a new task, register it in `wavesfm/data.py` (dataset factory + task metadata) and provide a preprocessing helper that writes an HDF5 with an `iq` or `image/features` dataset and a target.
- The training loop in `wavesfm/engine.py` is intentionally small and easy to tweak for extra metrics or behaviors.
