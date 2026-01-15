# WavesFM

![WavesFM](wavesfm.png)

Lightweight fine-tuning utilities for multimodal wireless models across a range of downstream tasks.

At a high level:
1) preprocess raw datasets into a single cache file (per split) → 2) optionally download a pretrained checkpoint → 3) fine-tune/evaluate on a selected task.

## Package overview
- `main_finetune.py`: CLI entrypoint for training and evaluation.
- `data.py`: task registry + dataset wiring + task metadata.
- `models_vit.py`: ViT-based model used across tasks.
- `engine.py`: training loop, evaluation, and metrics.
- `dataset_classes/`: minimal dataset loaders used by `data.py`.
- `preprocessing/`: scripts that turn raw datasets into cache files (stored as `.h5`).
- `lora.py`: optional LoRA adapters for attention projections.

## Installation
`requirements.txt` pins the core dependencies for reproducibility. Install PyTorch/torchvision separately to match your CUDA stack.

```bash
git clone https://github.com/AhmedTarek62/wavesfm.git
cd wavesfm
python -m venv .venv
source .venv/bin/activate
pip install -U pip
# Install PyTorch for your CUDA version (see pytorch.org), then:
pip install torch torchvision
pip install -r requirements.txt
```

## Supported tasks
Use the short name with `--task`:
- `sensing` — CSI sensing human activity classification (image-like CSI tensors).
- `rfs` — radio signal classification from spectrograms (image-like).
- `pos` — 5G NR positioning (regression on image-like CSI features).
- `uwb-indoor` — UWB indoor localization from CIR (regression on IQ-like CIR tensors).
- `uwb-industrial` — UWB industrial localization from CIR (regression on IQ-like CIR tensors, official train/test split).
- `radcom` — RadCom OTA classification (IQ).
- `rml` — RadioML modulation classification (IQ).
- `rfp` — Powder RF fingerprinting (IQ).
- `interf` — Icarus interference detection (IQ).

## Data caches (what the code expects)
Each task trains from a single cache file that stores samples + targets (and optional metadata like label names or class weights).

Conceptually:
- vision-style tasks store `(C,H,W)` tensors plus an integer label (classification) or a vector target (regression)
- IQ-style tasks store `(2,C,T)` tensors plus either an integer label (classification) or a vector target (regression)

The training pipeline reads these caches and does no raw parsing.

## Preprocessing
Preprocessing scripts live under `preprocessing/`. They are dataset-specific; use `--help` to see the expected raw layout and required arguments. Most directory-based scripts take `--data-path <raw_dir>` (and `--output <cache.h5>`), while file-based ones use `--input <raw_file>`.

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
- `--model`: model name from `models_vit.py` (shared across tasks).
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

## Dataset citations
- CSI sensing (WiFi): `@ARTICLE{9667414, author={Yang, Jianfei and Chen, Xinyan and Zou, Han and Wang, Dazhuo and Xu, Qianwen and Xie, Lihua}, journal={IEEE Internet of Things Journal}, title={EfficientFi: Toward Large-Scale Lightweight WiFi Sensing via CSI Compression}, year={2022}, volume={9}, number={15}, pages={13086-13095}, doi={10.1109/JIOT.2021.3139958}}`
- Radio spectrograms (RFS): `@dataset{zahid2024commrad, title = {CommRad RF: A dataset of communication radio signals for detection, identification and classification}, author = {Zahid, M.}, publisher = {Zenodo}, year = {2024}, doi = {10.5281/zenodo.14192970}}`
- 5G positioning (pos): `@data{jsat-pb50-21, doi = {10.21227/jsat-pb50}, url = {https://dx.doi.org/10.21227/jsat-pb50}, author = {Kaixuan Gao and Huiqiang Wang and Hongwu Lv}, publisher = {IEEE Dataport}, title = {CSI Dataset towards 5G NR High-Precision Positioning}, year = {2021}}`
- RadCom OTA (radcom): `@article{AjagannathCOMNET21, title = {Dataset for modulation classification and signal type classification for multi-task and single task learning}, journal = {Computer Networks}, volume = {199}, pages = {108441}, year = {2021}, doi = {10.1016/j.comnet.2021.108441}, author = {Anu Jagannath and Jithin Jagannath}}`
- RML (modulation): `@article{10.1109/TWC.2023.3254490, author = {Sathyanarayanan, Venkatesh and Gerstoft, Peter and Gamal, Aly El}, title = {RML22: Realistic Dataset Generation for Wireless Modulation Classification}, journal = {Trans. Wireless. Comm.}, year = {2023}, volume = {22}, number = {11}, pages = {7663--7675}, doi = {10.1109/TWC.2023.3254490}}`
- RF fingerprinting (Powder/RFP): `@inproceedings{reusmuns2019trust, title={Trust in 5G Open RANs through Machine Learning: RF Fingerprinting on the POWDER PAWR Platform}, author={Reus-Muns, Guillem and Jaisinghani, Dhertya and Sankhe, Kunal and Chowdhury, Kaushik}, booktitle={IEEE Globecom 2020-IEEE Global Communications Conference}, year={2020}, organization={IEEE}}`
- Interference (Icarus): `@INPROCEEDINGS{10228929, author={Roy, Debashri and Chaudhury, Vini and Tassie, Chinenye and Spooner, Chad and Chowdhury, Kaushik}, booktitle={IEEE INFOCOM 2023 - IEEE Conference on Computer Communications}, title={ICARUS: Learning on IQ and Cycle Frequencies for Detecting Anomalous RF Underlay Signals}, year={2023}, pages={1-10}, doi={10.1109/INFOCOM53939.2023.10228929}}`
- UWB indoor: `@dataset{bregar2023uwb, title = {UWB Positioning and Tracking Data Set}, author = {Bregar, Klemen}, publisher = {Zenodo}, year = {2023}, doi = {10.5281/zenodo.7629141}}`
- UWB industrial: https://cmutschler.de/datasets/channel-impulse-responses