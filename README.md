# WavesFM

![WavesFM](wavesfm.png)

Lightweight fine-tuning package for WavesFM, a multimodal wireless foundation model, on various downstream tasks. The workflow is:
1) preprocess raw dataset → 2) downlaod pretrained checkpoint → 3) train/eval from a single cache file.

## Package overview
- `main_finetune.py`: CLI entrypoint for training and evaluation.
- `data.py`: task registry + dataset wiring + task metadata.
- `models_vit.py`: ViT-based model used across tasks.
- `engine.py`: training loop, evaluation, and metrics.
- `dataset_classes/`: minimal dataset loaders used by `data.py`.
- `preprocessing/`: scripts that turn raw datasets into .h5 caches.
- `lora.py`: optional LoRA adapters for attention projections.

## Supported tasks
Use the short name with `--task`:
- `sensing` — human activity sensing from WiFi CSI (vision)
- `rfs` — radio signal classification from spectrograms (vision).
- `pos` — 5G NR positioning (vision).
- `uwb` — UWB localization from CIR (IQ).
- `radcom` — radar signal classification (IQ).
- `rml` — automatic modulation classification (IQ).
- `rfp` — RF fingerprinting (IQ).
- `interf` — interference detection (IQ).

## Data caches (what the code expects)
Each task trains from a single `.h5` file that stores samples + targets (and optional metadata like label names or class weights). The exact dataset keys depend on the task, but conceptually:
- vision-style tasks store `(C,H,W)` tensors plus an integer label
- IQ-style tasks store `(2,C,T)` tensors plus either an integer label (classification) or a float target (regression)

The training pipeline reads these caches and does no raw parsing.

## Preprocessing
Preprocessing scripts live under `preprocessing/`. They are dataset-specific, so use `--help` to see the expected raw layout and the required arguments.

Generic patterns:
- CSI sensing: `python preprocessing/preprocess_csi_sensing.py --csi-path <raw_dir> --output <cache.h5>`
- Radio spectrograms: `python preprocessing/preprocess_rfs.py --data-path <raw_dir> --output <cache.h5>`
- 5G positioning: `python preprocessing/preprocess_nr_positioning.py --data-path <raw_dir> --scene {outdoor|indoor} --output <cache.h5>`
- RadCom OTA: `python preprocessing/preprocess_radcom.py --input <raw_file> --output <cache.h5>`
- UWB: `python preprocessing/preprocess_uwb_loc.py --root <raw_root> --environment <env> --output <cache.h5>`

## Training & evaluation
Train:
```bash
python main_finetune.py \
  --task <task> \
  --train-data <train.h5> \
  --val-data <val.h5> \
  --output-dir <run_dir>
```

Evaluate only:
```bash
python main_finetune.py \
  --task <task> \
  --train-data <train.h5> \
  --val-data <val.h5> \
  --eval-only
```

Common flags:
- `--model`: model name from `models_vit.py`.
- `--finetune`: initialize from a pretrained checkpoint (loads model weights).
- `--resume`: resume training from a WavesFM checkpoint (model + optimizer + scheduler).
- `--lora`: enable LoRA adapters (`--lora-rank`, `--lora-alpha`).
- `--val-split`: auto-split if you don’t provide `--val-data`.

Outputs:
- logs: `output_dir/log.txt` (JSONL)
- checkpoints: `output_dir/best.pth` and periodic `output_dir/checkpoint_*.pth`

## Adding a new task/dataset
1) Write a preprocessing script that produces a cache file (samples + targets).
2) Register a new task in `data.py` so `build_datasets()` knows which loader/keys to use and what output shape to expect.
3) Train with `--task <your_task>`.

## Citation
If you use this code, please cite:
```
@article{aboulfotouh2025multimodal,
  title = {Multimodal Wireless Foundation Models},
  author = {Aboulfotouh, Ahmed and Abou-Zeid, Hatem},
  journal = {arXiv preprint arXiv:2511.15162},
  year = {2025},
  url = {https://arxiv.org/abs/2511.15162}
}
```

Please also credit the owners of datasets.

## Credits
Some code is adapted from:
- MAE: https://github.com/facebookresearch/mae
- timm-vit-lora: https://github.com/mnikitin/timm-vit-lora
- DeiT: https://github.com/facebookresearch/deit
- Transformer utils: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
- MoCo v3: https://github.com/facebookresearch/moco-v3
