from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _decode_attr(val):
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8")
    return val


def _parse_attr_list(val) -> tuple:
    if val is None:
        return ()
    val = _decode_attr(val)
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
        except json.JSONDecodeError:
            return (val,)
        if isinstance(parsed, list):
            return tuple(parsed)
        return (parsed,)
    if isinstance(val, (list, tuple, np.ndarray)):
        out = []
        for item in val:
            item = _decode_attr(item)
            out.append(str(item))
        return tuple(out)
    return (str(val),)


def _pad_or_slice(arr: np.ndarray, target_len: int, pad_val: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size >= target_len:
        return arr[:target_len]
    pad = np.full((target_len - arr.size,), pad_val, dtype=np.float32)
    return np.concatenate([arr, pad], axis=0)


class CIRLocDataset(Dataset):
    """
    Generic HDF5-backed CIR localization dataset.

    Expected HDF5 layout:
      - /cir: float32, shape (N, 2, A, L) storing real/imag parts
      - /location: float32, shape (N, D)
      - attrs: mean_real, mean_imag, std_real, std_imag (float)
      - attrs: loc_min (D,), loc_max (D,)
      - optional datasets: /channel, /anchor_mask, /rec_time, /burst_id
    """

    def __init__(
        self,
        h5_path: str | Path,
        *,
        return_mask: bool = False,
        return_meta: bool = False,
        return_channel: bool = False,
        as_complex: bool = False,
        location_dims: int | None = None,
    ) -> None:
        self.h5_path = str(Path(h5_path).expanduser())
        self.return_mask = bool(return_mask)
        self.return_meta = bool(return_meta)
        self.return_channel = bool(return_channel)
        self.as_complex = bool(as_complex)

        with h5py.File(self.h5_path, "r") as f:
            if "cir" not in f or "location" not in f:
                raise KeyError(f"Expected 'cir' and 'location' datasets in {self.h5_path}")
            cir_shape = f["cir"].shape
            loc_shape = f["location"].shape
            loc_dims = loc_shape[1] if len(loc_shape) > 1 else 1

            self.N = cir_shape[0]
            self.num_anchors = cir_shape[2]
            self.cir_len = cir_shape[3]

            self.location_dims = loc_dims if location_dims is None else int(location_dims)
            if self.location_dims <= 0:
                raise ValueError("location_dims must be positive.")
            if self.location_dims > loc_dims:
                raise ValueError("location_dims cannot exceed location dimension.")

            self.stats = {
                "mean": (
                    float(f.attrs.get("mean_real", 0.0)),
                    float(f.attrs.get("mean_imag", 0.0)),
                ),
                "std": (
                    float(f.attrs.get("std_real", 1.0)),
                    float(f.attrs.get("std_imag", 1.0)),
                ),
            }

            loc_min = f.attrs.get("loc_min", np.zeros(loc_dims, dtype=np.float32))
            loc_max = f.attrs.get("loc_max", np.ones(loc_dims, dtype=np.float32))
            loc_min = _pad_or_slice(loc_min, self.location_dims, 0.0)
            loc_max = _pad_or_slice(loc_max, self.location_dims, 1.0)
            self.loc_min = torch.tensor(loc_min, dtype=torch.float32)
            self.loc_max = torch.tensor(loc_max, dtype=torch.float32)

            anchors_attr = f.attrs.get("anchors", None)
            self.anchors = _parse_attr_list(anchors_attr)

            env_attr = f.attrs.get("environment", "")
            self.environment = str(_decode_attr(env_attr)) if env_attr is not None else ""

            fill_missing = f.attrs.get("fill_missing", "zero")
            self.fill_missing = str(_decode_attr(fill_missing)) if fill_missing is not None else "zero"

            self.has_mask = "anchor_mask" in f
            self.has_rec_time = "rec_time" in f
            self.has_burst_id = "burst_id" in f
            self.has_channel = "channel" in f

            if self.return_channel and not self.has_channel:
                raise KeyError(f"Requested return_channel, but 'channel' not found in {self.h5_path}")

        self._h5: Optional[h5py.File] = None

    def __len__(self) -> int:
        return self.N

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, idx: int):
        f = self._file()
        cir_np = f["cir"][idx]  # (2, A, L)
        loc_np = f["location"][idx]
        loc_np = loc_np[: self.location_dims]

        cir = torch.from_numpy(cir_np).float()
        mean_real, mean_imag = self.stats["mean"]
        std_real, std_imag = self.stats["std"]
        std_real = std_real if std_real > 0 else 1.0
        std_imag = std_imag if std_imag > 0 else 1.0
        cir[0] = (cir[0] - mean_real) / std_real
        cir[1] = (cir[1] - mean_imag) / std_imag

        if self.fill_missing == "nan":
            nan_mask = torch.isnan(cir)
            if nan_mask.any():
                cir = cir.clone()
                cir[nan_mask] = 0.0

        if self.as_complex:
            cir = torch.complex(cir[0], cir[1])

        # pad L dimension to next power of two
        L = cir.shape[-1]
        target_L = 1 << (L - 1).bit_length()
        if target_L > L:
            pad_width = target_L - L
            if self.as_complex:
                cir = torch.nn.functional.pad(cir, (0, pad_width), value=0.0)
            else:
                cir = torch.nn.functional.pad(cir, (0, pad_width))

        loc_raw = torch.from_numpy(np.asarray(loc_np, dtype=np.float32))
        denom = (self.loc_max - self.loc_min).clamp_min(1e-6)
        loc = 2.0 * (loc_raw - self.loc_min) / denom - 1.0

        if self.return_meta:
            meta = {}
            if self.has_burst_id:
                meta["burst_id"] = int(f["burst_id"][idx])
            if self.has_rec_time:
                meta["rec_time"] = torch.from_numpy(f["rec_time"][idx]).long()
            if self.has_mask:
                meta["anchor_mask"] = torch.from_numpy(f["anchor_mask"][idx].astype(np.bool_))
            if self.has_channel:
                meta["channel"] = int(f["channel"][idx])
            return cir, loc, meta

        if self.return_mask:
            if self.has_mask:
                mask = torch.from_numpy(f["anchor_mask"][idx].astype(np.bool_))
            else:
                mask = torch.ones(self.num_anchors, dtype=torch.bool)
            return cir, loc, mask

        if self.return_channel:
            ch_val = int(f["channel"][idx])
            ch = torch.tensor(ch_val, dtype=torch.int16)
            return cir, loc, ch

        return cir, loc

    def __del__(self) -> None:
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


class UWBIndoor(CIRLocDataset):
    """HDF5-backed UWB indoor localization dataset produced by preprocess_uwb_loc.py."""

    def __init__(self, h5_path: str | Path, return_channel: bool = False, as_complex: bool = False) -> None:
        super().__init__(
            h5_path,
            return_channel=return_channel,
            as_complex=as_complex,
            location_dims=2,
        )


class UWBIndustrial(CIRLocDataset):
    """HDF5-backed UWB industrial localization dataset produced by preprocess_ipin_loc.py."""

    def __init__(
        self,
        h5_path: str | Path,
        return_mask: bool = False,
        return_meta: bool = False,
        as_complex: bool = False,
    ) -> None:
        super().__init__(
            h5_path,
            return_mask=return_mask,
            return_meta=return_meta,
            as_complex=as_complex,
            location_dims=2,
        )
