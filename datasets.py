from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class CSISensingCache(Dataset):
    """HDF5-backed CSI sensing tensors.

    Expected datasets: `csi` (N, 3, H, W) and `label` (N,).
    Optional attr `labels` stores label names.
    """

    def __init__(self, h5_path: str | Path):
        self.h5_path = Path(h5_path)
        self._h5: h5py.File | None = None
        with h5py.File(self.h5_path, "r") as h5:
            self.length = h5["csi"].shape[0]
            labels_attr = h5.attrs.get("labels", None)
            self.labels = json.loads(labels_attr) if labels_attr else []

    def _require_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, index: int):
        h5 = self._require_h5()
        csi = torch.as_tensor(h5["csi"][index])
        label = torch.as_tensor(h5["label"][index], dtype=torch.long)
        return csi, label

    def __len__(self) -> int:
        return self.length

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass


class RadioSignalCache(Dataset):
    """HDF5-backed radio spectrogram tensors.

    Expected datasets: `image` (N, 1, H, W) and `label` (N,).
    Attr `labels` optionally lists class names.
    """

    def __init__(self, h5_path: str | Path):
        self.h5_path = Path(h5_path)
        self._h5: h5py.File | None = None
        with h5py.File(self.h5_path, "r") as h5:
            self.length = h5["image"].shape[0]
            labels_attr = h5.attrs.get("labels", None)
            self.labels = json.loads(labels_attr) if labels_attr else []

    def _require_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, index: int):
        h5 = self._require_h5()
        image = torch.as_tensor(h5["image"][index])
        label = torch.as_tensor(h5["label"][index], dtype=torch.long)
        return image, label

    def __len__(self) -> int:
        return self.length

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass


class Positioning5GCache(Dataset):
    """HDF5-backed 5G positioning dataset.

    Expects datasets `features` and `label`, plus attrs with normalization metadata.
    """

    def __init__(self, h5_path: str | Path):
        self.h5_path = Path(h5_path)
        self._h5: h5py.File | None = None
        with h5py.File(self.h5_path, "r") as h5:
            self.length = h5["features"].shape[0]
            self.scene = h5.attrs.get("scene", "")
            self.img_size = h5.attrs.get("img_size", None)
            self.pretrain = bool(h5.attrs.get("pretrain", False))
            self.norm = {
                "min_val": h5.attrs.get("min_val", None),
                "max_val": h5.attrs.get("max_val", None),
                "mu": json.loads(h5.attrs.get("mu", "[]")),
                "std": json.loads(h5.attrs.get("std", "[]")),
                "coord_nominal_min": json.loads(h5.attrs.get("coord_nominal_min", "[]")),
                "coord_nominal_max": json.loads(h5.attrs.get("coord_nominal_max", "[]")),
            }

    def _require_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, index: int):
        h5 = self._require_h5()
        features = torch.as_tensor(h5["features"][index])
        label = torch.as_tensor(h5["label"][index])
        return features, label

    def __len__(self) -> int:
        return self.length

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass


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


class RadComOtaCache(Dataset):
    """Normalized RadCom OTA cache saved by `preprocess_radcom_ota.py`."""

    def __init__(
        self,
        h5_path: str | Path,
        normalize: bool = True,
        stats: Dict[str, Iterable[float]] | None = None,
        return_snr: bool = False,
    ):
        self.h5_path = str(h5_path)
        self.normalize = bool(normalize)
        self.return_snr = bool(return_snr)
        self.labels = RADCOM_OTA_LABELS
        self.label_to_idx = {pair: i for i, pair in enumerate(self.labels)}

        with h5py.File(self.h5_path, "r") as f:
            for req in ("sample", "modulation", "signal_type", "snr"):
                if req not in f:
                    raise KeyError(f"Expected dataset '{req}' in {self.h5_path}")
            self.n = f["sample"].shape[0]
            self.sample_shape = tuple(f["sample"].shape[1:])
            if len(self.sample_shape) != 3 or self.sample_shape[0] != 2:
                raise ValueError(f"Expected samples shaped (2, C, T), got {self.sample_shape}")
            if stats is None:
                mean_attr = f.attrs.get("mean")
                std_attr = f.attrs.get("std")
                if mean_attr is None or std_attr is None:
                    raise KeyError("Missing mean/std attributes in cache; provide stats explicitly.")
                self.stats = {
                    "mean": np.asarray(mean_attr, dtype=np.float32),
                    "std": np.asarray(std_attr, dtype=np.float32),
                }
            else:
                if "mean" not in stats or "std" not in stats:
                    raise KeyError("stats must contain 'mean' and 'std'.")
                self.stats = {
                    "mean": np.asarray(stats["mean"], dtype=np.float32),
                    "std": np.asarray(stats["std"], dtype=np.float32),
                }

    def __len__(self) -> int:
        return self.n

    @staticmethod
    def _decode_str(val) -> str:
        if isinstance(val, bytes):
            return val.decode("utf-8")
        return str(val)

    def __getitem__(self, idx: int):
        with h5py.File(self.h5_path, "r") as f:
            sample = f["sample"][idx]
            mod = self._decode_str(f["modulation"][idx])
            sig = self._decode_str(f["signal_type"][idx])
            snr_val = f["snr"][idx]

        x = torch.from_numpy(sample.astype(np.float32, copy=False))
        if self.normalize:
            mu = torch.tensor(self.stats["mean"], dtype=torch.float32).view(2, 1, 1)
            sd = torch.tensor(self.stats["std"], dtype=torch.float32).clamp_min(1e-6).view(2, 1, 1)
            x = (x - mu) / sd

        label_pair = (mod, sig)
        if label_pair not in self.label_to_idx:
            raise KeyError(f"Unexpected label pair {label_pair} (known: {self.labels})")
        label = torch.tensor(self.label_to_idx[label_pair], dtype=torch.long)

        if not self.return_snr:
            return x, label

        if isinstance(snr_val, bytes):
            snr_val = snr_val.decode("utf-8")
        try:
            snr_val = float(snr_val)
        except (TypeError, ValueError):
            snr_val = str(snr_val)
        return x, label, snr_val


class UWBLoc(Dataset):
    """UWB positioning cache produced by `preprocess_uwb_loc.py`."""

    def __init__(self, h5_path: str | Path, return_channel: bool = False, as_complex: bool = False) -> None:
        self.h5_path = str(Path(h5_path).expanduser())
        self.return_channel = return_channel
        self.as_complex = as_complex

        with h5py.File(self.h5_path, "r") as f:
            cir_shape = f["cir"].shape  # (N, 2, A, L)
            self.N, self.num_anchors, self.cir_len = cir_shape[0], cir_shape[2], cir_shape[3]
            env_attr = f.attrs.get("environment", "")
            self.environment = env_attr.decode("utf-8") if isinstance(env_attr, (bytes, np.bytes_)) else str(env_attr)
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
            loc_min = f.attrs.get("loc_min", np.zeros(3, dtype=np.float32))[:-1]
            loc_max = f.attrs.get("loc_max", np.ones(3, dtype=np.float32))[:-1]
            self.loc_min = torch.tensor(loc_min, dtype=torch.float32)
            self.loc_max = torch.tensor(loc_max, dtype=torch.float32)

        self._h5: h5py.File | None = None

    def __len__(self) -> int:
        return self.N

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, idx: int):
        f = self._file()
        cir_np = f["cir"][idx]  # (2, A, L)
        loc_np = f["location"][idx][:-1]
        ch_np = f["channel"][idx]

        cir = torch.from_numpy(cir_np).float()
        mean_real, mean_imag = self.stats["mean"]
        std_real, std_imag = self.stats["std"]
        cir[0] = (cir[0] - mean_real) / std_real
        cir[1] = (cir[1] - mean_imag) / std_imag
        if self.as_complex:
            cir = torch.complex(cir[0], cir[1])
        # else: keep (2, A, L)

        L = cir.shape[-1]
        target_L = 1 << (L - 1).bit_length()
        if target_L > L:
            pad_width = target_L - L
            pad_tuple = (0, pad_width)
            if self.as_complex:
                cir = F.pad(cir, pad_tuple, value=0.0)
            else:
                cir = F.pad(cir, pad_tuple)

        loc_raw = torch.from_numpy(loc_np).float()
        denom = (self.loc_max - self.loc_min).clamp_min(1e-6)
        loc = 2.0 * (loc_raw - self.loc_min) / denom - 1.0
        ch = torch.tensor(int(ch_np), dtype=torch.int16)

        if self.return_channel:
            return cir, loc, ch
        return cir, loc

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
