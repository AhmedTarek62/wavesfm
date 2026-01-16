from __future__ import annotations

from pathlib import Path

import h5py
import torch

from dataset_classes.base import ImageDataset


class DeepMIMO(ImageDataset):
    """Dataset wrapper for DeepMIMO caches with LoS and beam labels."""

    def __init__(
        self,
        h5_path: str | Path,
        *,
        label_key: str = "label_los",
        label_dtype: torch.dtype | None = torch.long,
    ):
        super().__init__(
            h5_path,
            sample_key="sample",
            label_key=label_key,
            label_dtype=label_dtype,
            meta_keys=("scenario",),
        )
        self.n_beams = None
        with h5py.File(self.h5_path, "r") as h5:
            n_beams = h5.attrs.get("n_beams", None)
            if n_beams is not None:
                self.n_beams = int(n_beams)
