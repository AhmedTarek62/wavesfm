from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import h5py
import torch
from torch.utils.data import Dataset


def _decode(val):
    if isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8")
    return val


class IQDataset(Dataset):
    """Minimal loader for IQ-style caches (pre-normalized, stored in h5)."""

    def __init__(
        self,
        h5_path: str | Path,
        sample_key: str = "sample",
        label_key: str = "label",
        *,
        label_dtype: torch.dtype | None = torch.long,
        meta_keys: Iterable[str] | None = None,
    ):
        self.h5_path = Path(h5_path)
        self.sample_key = sample_key
        self.label_key = label_key
        self.label_dtype = label_dtype
        self.meta_keys = tuple(meta_keys) if meta_keys is not None else ()
        self._h5: Optional[h5py.File] = None

        with h5py.File(self.h5_path, "r") as h5:
            if sample_key not in h5 or label_key not in h5:
                raise KeyError(f"Expected datasets '{sample_key}' and '{label_key}' in {self.h5_path}")
            self.length = h5[sample_key].shape[0]
            self.sample_shape = tuple(h5[sample_key].shape[1:])
            labels_attr = h5.attrs.get("labels", None)
            self.labels = tuple(json.loads(labels_attr)) if labels_attr else None

    def __len__(self) -> int:
        return self.length

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, idx: int):
        h5 = self._file()
        sample = torch.as_tensor(h5[self.sample_key][idx])
        if self.label_dtype is None:
            label = torch.as_tensor(h5[self.label_key][idx])
        else:
            label = torch.as_tensor(h5[self.label_key][idx], dtype=self.label_dtype)

        if not self.meta_keys:
            return sample, label

        meta = {}
        for key in self.meta_keys:
            if key not in h5:
                continue
            val = h5[key][idx]
            meta[key] = _decode(val)
        return sample, label, meta

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


class ImageDataset(Dataset):
    """Minimal loader for image-like caches (C,H,W), pre-normalized and stored in h5."""

    def __init__(
        self,
        h5_path: str | Path,
        sample_key: str = "image",
        label_key: str = "label",
        *,
        label_dtype: torch.dtype | None = torch.long,
        meta_keys: Iterable[str] | None = None,
    ):
        self.h5_path = Path(h5_path)
        self.sample_key = sample_key
        self.label_key = label_key
        self.label_dtype = label_dtype
        self.meta_keys = tuple(meta_keys) if meta_keys is not None else ()
        self._h5: Optional[h5py.File] = None

        with h5py.File(self.h5_path, "r") as h5:
            if sample_key not in h5 or label_key not in h5:
                raise KeyError(f"Expected datasets '{sample_key}' and '{label_key}' in {self.h5_path}")
            self.length = h5[sample_key].shape[0]
            self.sample_shape = tuple(h5[sample_key].shape[1:])
            labels_attr = h5.attrs.get("labels", None)
            self.labels = tuple(json.loads(labels_attr)) if labels_attr else None

    def __len__(self) -> int:
        return self.length

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, idx: int):
        h5 = self._file()
        sample = torch.as_tensor(h5[self.sample_key][idx])
        if self.label_dtype is None:
            label = torch.as_tensor(h5[self.label_key][idx])
        else:
            label = torch.as_tensor(h5[self.label_key][idx], dtype=self.label_dtype)

        if not self.meta_keys:
            return sample, label

        meta = {}
        for key in self.meta_keys:
            if key not in h5:
                continue
            val = h5[key][idx]
            meta[key] = _decode(val)
        return sample, label, meta

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
