# WavesFM fine-tuning (HDF5-first)

A lightweight, public-friendly entrypoint for reproducing our multimodal fine-tuning runs. Training only touches preprocessed HDF5 files; a small preprocessing package is included for users who need to generate those files from the original datasets.

## Supported tasks
- IQ tasks: `amc`, `aoa`, `rml`, `rfp`, `deepbeam`, `interf`
- Vision tasks: `sensing`, `rfs`
- Positioning: `pos` (5G), `uwb`
- RadCom OTA: `radcom`

Dataset loaders live in `wavesfm/datasets.py` with clear names (`IqDataset`, `SensingDataset`, `PositioningDataset`, `RadioSignalDataset`, `RadComDataset`, `UwbDataset`).

## Preprocessing
Use `wavesfm.preprocessing` to turn raw data into a single HDF5 file per split.

Examples:
```bash
# IQ tasks (amc/aoa/rml/rfp/deepbeam/interf)
python - <<'PY'
from wavesfm.preprocessing import preprocess_iq_task
preprocess_iq_task("amc", data_path="data/raw_iq/train_shards", output="data/amc_train.h5")
preprocess_iq_task("amc", data_path="data/raw_iq/val_shards", output="data/amc_val.h5")
PY

# CSI sensing
from wavesfm.preprocessing import preprocess_sensing
preprocess_sensing(data_path="data/sensing/raw/train", output="data/sensing_train.h5", overwrite=True)

# Radio signals (spectrograms)
from wavesfm.preprocessing import preprocess_radio_signals
preprocess_radio_signals(root_dir="data/rfs/raw", output="data/rfs_train.h5", overwrite=True)

# Positioning
from wavesfm.preprocessing import preprocess_positioning
preprocess_positioning(datapath="data/pos/raw/outdoor/train", output="data/pos_train.h5", scene="outdoor")

# RadCom OTA
from wavesfm.preprocessing import preprocess_radcom_ota
preprocess_radcom_ota(input_path="data/radcom/raw.h5", output="data/radcom_all.h5")

# UWB positioning
from wavesfm.preprocessing import preprocess_uwb
preprocess_uwb(root="data/uwb/raw", environment="indoor", output="data/uwb_train.h5", batch_size=64)
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
