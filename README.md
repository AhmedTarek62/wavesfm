# WavesFM fine-tuning

Lightweight utilities for fine-tuning multimodal wireless models on cached `.h5` datasets. Preprocessing scripts build the caches; training never touches raw data.

## Tasks
- Vision: `sensing`, `rfs`, `positioning`
- IQ: `rml`, `rfp`, `interf`
- CIR: `pos` (5G), `uwb`
- Radar: `radcom`

Loaders live in `dataset_classes/`; `data.py` wires them to tasks.

## Preprocessing
Per-dataset scripts under `preprocessing/` write one cache per split. Examples:
```bash
# CSI sensing
python wavesfm/preprocessing/preprocess_csi_sensing.py --csi-path data/sensing/raw/train --output data/sensing_train.h5
# Radio signals (spectrogram images)
python wavesfm/preprocessing/preprocess_rfs.py --data-path data/rfs/raw --output data/rfs_train.h5
# 5G positioning
python wavesfm/preprocessing/preprocess_nr_positioning.py --data-path data/pos/raw/outdoor/train --output data/pos_train.h5 --scene outdoor
# RadCom OTA
python wavesfm/preprocessing/preprocess_radcom.py --input data/radcom/raw.h5 --output data/radcom_all.h5
# UWB positioning
python wavesfm/preprocessing/preprocess_uwb_loc.py --root data/uwb/raw --environment indoor --output data/uwb_train.h5
# RML / RF fingerprinting (Powder) / interference (Icarus)
python wavesfm/preprocessing/preprocess_rml.py --root data/rml --version 2016 --output data/rml_train.h5
python wavesfm/preprocessing/preprocess_rfp.py --root data/rfp --output data/rfp_train.h5
python wavesfm/preprocessing/preprocess_icarus.py --root data/icarus --output data/icarus_train.h5
```

## Training
Unified entrypoint:
```bash
python wavesfm/main_finetune.py \
  --task amc \
  --train-data data/amc_train.h5 \
  --val-data data/amc_val.h5 \
  --model vit_multi_small \
  --batch-size 64 --epochs 50 \
  --output-dir runs/amc-small
```
Add `--lora --lora-rank 8 --lora-alpha 1.0` to enable LoRA adapters.

Notes:
- Training/eval only use cached files; no raw preprocessing happens in dataloaders.
- Logging is JSONL to `output_dir/log.txt` plus checkpoints (`best.pth` and periodic `checkpoint_*.pth`).
- `--eval-only` runs validation without training.

## Extending
- Register new tasks in `data.py` (factory + metadata) and add a preprocessing helper that writes an `iq` or `image/features` dataset plus targets.
- The training loop in `engine.py` is small and easy to adapt for extra metrics/behaviors.

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

@article{aboulfotouh2025image,
  author  = {Aboulfotouh, Ahmed and Mohammed, Elsayed and Abou-Zeid, Hatem},
  title   = {{6G} {WavesFM}: A Foundation Model for Sensing, Communication, and Localization},
  journal = {IEEE Open J. Commun. Soc.},
  year    = {2025},
  volume  = {6},
  doi     = {10.1109/OJCOMS.2025.3600616}
}
```

## Credits
### Some of the code is adapted from these amazing repos.
- timm-vit-lora: https://github.com/mnikitin/timm-vit-lora
- MAE: https://github.com/facebookresearch/mae
- DeiT: https://github.com/facebookresearch/deit
- Transformer utils: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
- MoCo v3: https://github.com/facebookresearch/moco-v3
