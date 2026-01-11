from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SensingDataset(Dataset):
    """CSI sensing tensors written by preprocessing scripts."""

    def __init__(self, h5_path: str | Path):
        self.h5_path = Path(h5_path)
        self._h5: h5py.File | None = None
        with h5py.File(self.h5_path, "r") as h5:
            self.length = h5["csi"].shape[0]
            labels_attr = h5.attrs.get("labels", None)
            self.labels = json.loads(labels_attr) if labels_attr else []

    def _require(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, index: int):
        h5 = self._require()
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


class RadioSignalDataset(Dataset):
    """Spectrogram-like radio signal tensors from preprocessing scripts."""

    def __init__(self, h5_path: str | Path):
        self.h5_path = Path(h5_path)
        self._h5: h5py.File | None = None
        with h5py.File(self.h5_path, "r") as h5:
            self.length = h5["image"].shape[0]
            labels_attr = h5.attrs.get("labels", None)
            self.labels = json.loads(labels_attr) if labels_attr else []

    def _require(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, index: int):
        h5 = self._require()
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


class PositioningDataset(Dataset):
    """5G positioning dataset with normalized targets."""

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

    def _require(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, index: int):
        h5 = self._require()
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


class RadComDataset(Dataset):
    """Normalized RadCom OTA dataset created by preprocessing/preprocess_radcom_ota.py."""

    def __init__(
        self,
        h5_path: str | Path,
        normalize: bool = True,
        stats: dict | None = None,
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
                    raise KeyError("Missing mean/std attributes in RadCom cache; provide stats explicitly.")
                self.stats = {
                    "mean": np.asarray(mean_attr, dtype=np.float32),
                    "std": np.asarray(std_attr, dtype=np.float32),
                }
            else:
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


class UwbDataset(Dataset):
    """UWB positioning dataset produced by preprocessing/preprocess_uwb_loc.py."""

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

        L = cir.shape[-1]
        target_L = 1 << (L - 1).bit_length()
        if target_L > L:
            pad_width = target_L - L
            pad_tuple = (0, pad_width)
            cir = F.pad(cir, pad_tuple, value=0.0 if self.as_complex else 0.0)

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


class IqDataset(Dataset):
    """Generic IQ tensor reader (2, C, T) for classification or regression targets."""

    def __init__(self, h5_path: str | Path, target_key: str | None = None, normalize: bool = True):
        self.h5_path = Path(h5_path)
        self.normalize = normalize
        self._h5: h5py.File | None = None
        with h5py.File(self.h5_path, "r") as h5:
            if "iq" not in h5:
                raise KeyError(f"Expected dataset 'iq' in {self.h5_path}")
            self.length = h5["iq"].shape[0]
            self.sample_shape = tuple(h5["iq"].shape[1:])
            if target_key is None:
                target_key = "label" if "label" in h5 else "target"
            if target_key not in h5:
                raise KeyError(f"Target key '{target_key}' not found in {self.h5_path}")
            self.target_key = target_key
            self.labels = json.loads(h5.attrs.get("labels", "[]"))
            mean_attr = h5.attrs.get("mean")
            std_attr = h5.attrs.get("std")
            self.stats = None
            if mean_attr is not None and std_attr is not None:
                self.stats = {
                    "mean": np.asarray(mean_attr, dtype=np.float32),
                    "std": np.asarray(std_attr, dtype=np.float32),
                }

    def _require(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, index: int):
        h5 = self._require()
        x = torch.as_tensor(h5["iq"][index])
        if self.normalize and self.stats is not None:
            mu = torch.tensor(self.stats["mean"], dtype=torch.float32).view(2, 1, 1)
            sd = torch.tensor(self.stats["std"], dtype=torch.float32).clamp_min(1e-6).view(2, 1, 1)
            x = (x - mu) / sd
        y = torch.as_tensor(h5[self.target_key][index])
        # cast to long for classification if dtype is integer
        if torch.is_floating_point(y):
            pass
        else:
            y = y.to(torch.long)
        return x, y

    def __len__(self) -> int:
        return self.length

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass
