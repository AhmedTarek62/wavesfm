from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler, Subset

from dataset_classes.base import IQDataset


LABELS = (
    "8PSK",
    "AM-DSB",
    "AM-SSB",
    "BPSK",
    "CPFSK",
    "GFSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "QPSK",
    "WBFM",
)


class RML(IQDataset):
    """RML cache produced by preprocess_rml.py (h5)."""

    def __init__(self, h5_path: str | Path, return_snr: bool = False):
        self.return_snr = bool(return_snr)
        super().__init__(h5_path, sample_key="sample", label_key="label", meta_keys=("snr",))
        if self.labels is None:
            self.labels = LABELS
        self.label_to_idx = {lb: i for i, lb in enumerate(self.labels)}

    def __getitem__(self, idx: int):
        sample, label, meta = super().__getitem__(idx)
        if not self.return_snr:
            return sample, label
        return sample, label, torch.as_tensor(meta["snr"], dtype=torch.long)

    @property
    def snr_by_index(self):
        with h5py.File(self.h5_path, "r") as h5:
            return np.asarray(h5["snr"], dtype=np.int16)


def _unwrap_subset(ds):
    if not isinstance(ds, Subset):
        return ds, None
    all_idx = ds.indices
    base = ds.dataset
    while isinstance(base, Subset):
        all_idx = [base.indices[i] for i in all_idx]
        base = base.dataset
    return base, np.asarray(all_idx, dtype=np.int64)


def make_snr_sampler(
    dataset,
    policy: str = "custom",
    *,
    snr_weights: dict | None = None,
    mu: float = 12.0,
    sigma: float = 5.0,
    floor: float = 0.1,
    low_cut: int = 0,
    high_cut: int = 18,
    tail_weight: float = 0.2,
    temperature: float = 1.0,
    normalize: bool = True,
    num_samples: int | None = None,
    replacement: bool = True,
    generator: torch.Generator = None,
) -> WeightedRandomSampler:
    base_ds, idx_map = _unwrap_subset(dataset)
    snrs_full = base_ds.snr_by_index.astype(np.float32)
    snrs = snrs_full[idx_map] if idx_map is not None else snrs_full

    if policy == "custom":
        w = np.vectorize(lambda s: snr_weights.get(int(s), 0.0), otypes=[np.float32])(snrs) if snr_weights else np.ones_like(snrs, dtype=np.float32)
    elif policy == "gaussian":
        w = np.exp(-0.5 * ((snrs - mu) / max(1e-6, sigma)) ** 2).astype(np.float32)
        w = floor + (1.0 - floor) * w
    elif policy == "clip":
        mid = (snrs >= low_cut) & (snrs <= high_cut)
        w = np.where(mid, 1.0, tail_weight).astype(np.float32)
    else:
        raise ValueError("policy must be 'custom', 'gaussian', or 'clip'")

    if temperature != 1.0:
        w = np.power(w, temperature, dtype=np.float32)

    w = np.clip(w, 1e-6, None)
    if normalize:
        w = w / w.sum()

    weights_tensor = torch.as_tensor(w, dtype=torch.double)
    if num_samples is None:
        num_samples = len(dataset)

    return WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=num_samples,
        replacement=replacement,
        generator=generator,
    )
