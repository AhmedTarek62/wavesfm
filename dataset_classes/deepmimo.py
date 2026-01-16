from __future__ import annotations

from pathlib import Path
import json

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
        n_beams: int | None = None,
        label_dtype: torch.dtype | None = torch.long,
    ):
        self.selected_n_beams = int(n_beams) if n_beams is not None else None
        if label_key.startswith("label_beam") and n_beams is None:
            raise ValueError("DeepMIMO beam labels require n_beams to select label_beam_{n}.")
        if n_beams is not None:
            label_key = f"label_beam_{int(n_beams)}"
        super().__init__(
            h5_path,
            sample_key="sample",
            label_key=label_key,
            label_dtype=label_dtype,
            meta_keys=("scenario",),
        )
        self.n_beams = None
        self.beam_options = None
        self.effective_n_beams = None
        with h5py.File(self.h5_path, "r") as h5:
            n_beams = h5.attrs.get("n_beams", None)
            if n_beams is not None:
                self.n_beams = int(n_beams)
            beam_options = h5.attrs.get("beam_options", None)
            if beam_options:
                try:
                    self.beam_options = tuple(json.loads(beam_options))
                except Exception:
                    self.beam_options = None
            if self.selected_n_beams is not None:
                eff_key = f"effective_n_beams_{self.selected_n_beams}"
                eff = h5.attrs.get(eff_key, None)
                if eff is not None:
                    self.effective_n_beams = int(eff)
