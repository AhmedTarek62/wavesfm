from __future__ import annotations

from pathlib import Path

from dataset_classes.base import IQDataset


class Powder(IQDataset):
    """Powder RF fingerprinting dataset produced by preprocess_rfp.py (h5)."""

    def __init__(self, h5_path: str | Path, return_meta: bool = False):
        self.return_meta = bool(return_meta)
        super().__init__(
            h5_path,
            sample_key="sample",
            label_key="label",
            meta_keys=("source_file", "start", "sample_rate", "center_freq"),
        )

    def __getitem__(self, idx: int):
        if not self.return_meta:
            return super().__getitem__(idx)
        sample, label, meta = super().__getitem__(idx)
        meta["start"] = int(meta["start"])
        meta["sample_rate"] = float(meta["sample_rate"])
        meta["center_freq"] = float(meta["center_freq"])
        return sample, label, meta
