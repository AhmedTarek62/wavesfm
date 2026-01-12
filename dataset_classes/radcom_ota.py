from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import h5py

from dataset_classes.base import IQDataset

RADCOM_OTA_LABELS: Tuple[Tuple[str, str], ...] = (
    ("AM-DSB", "AM radio"),
    ("AM-SSB", "AM radio"),
    ("ASK", "short-range"),
    ("BPSK", "SATCOM"),
    ("FMCW", "Radar Altimeter"),
    ("PULSED", "Air-Ground-MTI"),
    ("PULSED", "Airborne-detection"),
    ("PULSED", "Airborne-range"),
    ("PULSED", "Ground mapping"),
)


class RadComOta(IQDataset):
    """Load normalized RadCom OTA cache saved by preprocess_radcom.py (h5)."""

    def __init__(self, h5_path: str | Path, return_snr: bool = False):
        self.return_snr = bool(return_snr)
        super().__init__(h5_path, sample_key="sample", label_key="label", meta_keys=("modulation", "signal_type", "snr"))
        with h5py.File(self.h5_path, "r") as h5:  # type: ignore[name-defined]
            lp = h5.attrs.get("label_pairs", "[]")
        self.label_pairs: Tuple[Tuple[str, str], ...] = tuple(tuple(p) for p in json.loads(lp)) or RADCOM_OTA_LABELS

    def __getitem__(self, idx: int):
        sample, label, meta = super().__getitem__(idx)
        if not self.return_snr:
            return sample, label
        snr_val = meta.get("snr", np.nan)
        try:
            snr_val = float(snr_val)
        except Exception:
            pass
        return sample, label, snr_val
