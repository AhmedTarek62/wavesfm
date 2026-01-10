# WavesFM cache-first fine-tuning

A minimal, public-friendly entrypoint for reproducing our multimodal fine-tuning runs. This version only depends on cached datasets (HDF5 written by the preprocessing scripts) and keeps logging to simple JSONL lines in `log.txt`.

## Supported cached tasks
- `sensing` — `CSISensingCache` from `preprocessing/preprocess_csi_sensing_cache.py`
- `rfs` — `RadioSignalCache` from `preprocessing/preprocess_radio_sig_cache.py`
- `pos` — `Positioning5GCache` from `preprocessing/preprocess_positioning_cache.py`
- `radcom` — `RadComOtaCache` from `preprocessing/preprocess_radcom_ota.py`
- `uwb` — `UWBLoc` from `preprocessing/preprocess_uwb_loc.py`

Each task expects `--train-cache` (and optionally `--val-cache`) to point to the HDF5 file produced by the matching preprocessing script. If `--val-cache` is omitted, the script will split the train cache using `--val-split`.

## Quickstart
```
python main_finetune_multi.py \
  --task sensing \
  --train-cache /path/to/csi_sensing_train.h5 \
  --val-cache /path/to/csi_sensing_val.h5 \
  --model vit_multi_small \
  --finetune /path/to/pretrained.pth \
  --batch-size 64 --epochs 50 \
  --output-dir runs/sensing-small
```

Key behaviors:
- Uses cached datasets only; no raw preprocessing in the dataloader.
- Cosine LR with warmup, AdamW, optional gradient accumulation.
- Writes metrics per epoch to `log.txt` (JSONL) and checkpoints to `output_dir`.
- `--eval-only` skips training and reports validation metrics.

## Extending
- To add a new cached task, register it in `wavesfm/data.py` with a factory pointing to your dataset class and fill out `TaskInfo`.
- The training loop in `wavesfm/engine.py` is intentionally small; extend it if you need extra metrics.
