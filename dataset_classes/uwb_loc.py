from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import h5py


class UWBLoc(Dataset):
    """
    HDF5-backed UWB positioning dataset.

    Expected HDF5 layout (produced by preprocess_uwb_loc.py):
      - /cir: float32, shape (N, 2, A, L) storing real/imag parts
      - /location: float32, shape (N, 3) storing (x, y, z)
      - /channel: int, shape (N,) storing channel number (e.g., 1 for ch1, -1 if unknown)
      - attrs: environment (str)
      - attrs: mean_real, mean_imag, std_real, std_imag (float)
      - attrs: loc_min (3,), loc_max (3,)
    """

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

        self._h5: Optional[h5py.File] = None

    def __len__(self) -> int:
        return self.N

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self._file()
        cir_np = f["cir"][idx]  # (2, A, L)
        loc_np = f["location"][idx][:-1]
        ch_np = f["channel"][idx]

        cir = torch.from_numpy(cir_np).float()
        # standardize real/imag separately
        mean_real, mean_imag = self.stats["mean"]
        std_real, std_imag = self.stats["std"]
        cir[0] = (cir[0] - mean_real) / std_real
        cir[1] = (cir[1] - mean_imag) / std_imag
        if self.as_complex:
            cir = torch.complex(cir[0], cir[1])  # (A, L)
        else:
            # already (2, A, L)
            pass

        # pad L dimension to next power of two
        L = cir.shape[-1]
        target_L = 1 << (L - 1).bit_length()
        if target_L > L:
            pad_width = target_L - L
            if self.as_complex:
                cir = torch.nn.functional.pad(cir, (0, pad_width), value=0.0)
            else:
                cir = torch.nn.functional.pad(cir, (0, pad_width))

        loc_raw = torch.from_numpy(loc_np).float()
        denom = (self.loc_max - self.loc_min).clamp_min(1e-6)
        loc = 2.0 * (loc_raw - self.loc_min) / denom - 1.0
        ch = torch.tensor(int(ch_np), dtype=torch.int16)

        if self.return_channel:
            return cir, loc, ch
        return cir, loc

    def __del__(self) -> None:
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
