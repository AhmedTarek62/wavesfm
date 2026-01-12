from __future__ import annotations

from pathlib import Path

import torch

from dataset_classes.base import H5IQDataset


class IcarusCache(H5IQDataset):
    """HDF5-backed Icarus interference detection dataset produced by preprocess_icarus.py."""

    def __init__(self, h5_path: str | Path, return_meta: bool = False):
        self.return_meta = bool(return_meta)
        super().__init__(h5_path, sample_key="sample", label_key="label", meta_keys=("band", "fs", "source_file"))

    def __getitem__(self, idx: int):
        if not self.return_meta:
            return super().__getitem__(idx)
        sample, label, meta = super().__getitem__(idx)
        meta["band"] = int(meta["band"])
        meta["fs"] = float(meta["fs"])
        return sample, label, meta
