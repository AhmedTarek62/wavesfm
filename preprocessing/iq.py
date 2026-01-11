from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import h5py
import torch
from torch.utils.data import DataLoader

from dataset_classes import IQDatasetH5Sharded, RML, RFPrintDataset
from dataset_classes.deep_beam import DeepBeam
from dataset_classes.interference_detection import IcarusPowder

SupportedIqTask = Literal["amc", "aoa", "rml", "rfp", "deepbeam", "interf"]


def _make_dataset(task: SupportedIqTask, data_path: str | Path):
    if task in {"amc", "aoa"}:
        return IQDatasetH5Sharded(data_path, mode=task)
    if task == "rml":
        return RML(data_path, version="2022")
    if task == "rfp":
        return RFPrintDataset(data_path)
    if task == "deepbeam":
        return DeepBeam(data_path)
    if task == "interf":
        return IcarusPowder(data_path)
    raise ValueError(f"Unsupported IQ task: {task}")


def preprocess_iq_task(
    task: SupportedIqTask,
    data_path: str | Path,
    output: str | Path,
    batch_size: int = 256,
    num_workers: int = 0,
    compression: str = "none",
    overwrite: bool = False,
) -> Path:
    """
    Convert raw IQ datasets into a single HDF5 file with keys:
      - iq:     float32 (N, 2, C, T)
      - label:  int64   (N,)         for classification tasks
      - target: float32 (N, D)       for regression (aoa)
    """
    data_path = Path(data_path)
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; use overwrite=True to replace.")
    output.parent.mkdir(parents=True, exist_ok=True)
    comp = None if compression == "none" else compression

    ds = _make_dataset(task, data_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    first_x, first_y = ds[0]
    iq_shape = tuple(first_x.shape)
    target_key = "label" if task != "aoa" else "target"
    target_shape = tuple(first_y.shape) if hasattr(first_y, "shape") else ()
    target_dtype = "int64" if target_key == "label" else "float32"

    sum_iq = torch.zeros(2, dtype=torch.float64)
    sumsq_iq = torch.zeros(2, dtype=torch.float64)
    count_iq = torch.zeros(2, dtype=torch.float64)

    chunk = min(batch_size, len(ds))
    with h5py.File(output, "w") as h5:
        h5.create_dataset(
            "iq",
            shape=(len(ds), *iq_shape),
            dtype="float32",
            chunks=(chunk, *iq_shape),
            compression=comp,
        )
        h5.create_dataset(
            target_key,
            shape=(len(ds), *target_shape) if target_shape else (len(ds),),
            dtype=target_dtype,
            chunks=(chunk, *target_shape) if target_shape else (chunk,),
            compression=comp,
        )
        if hasattr(ds, "labels"):
            h5.attrs["labels"] = json.dumps(getattr(ds, "labels", []))
        idx = 0
        for batch in loader:
            x, y = batch
            bsz = x.shape[0]
            h5["iq"][idx : idx + bsz] = x.numpy()
            h5[target_key][idx : idx + bsz] = y.numpy()
            idx += bsz
            sum_iq += x.sum(dim=(0, 2, 3), dtype=torch.float64)
            sumsq_iq += (x * x).sum(dim=(0, 2, 3), dtype=torch.float64)
            count_iq += torch.tensor([x.shape[0] * x.shape[2] * x.shape[3]] * 2, dtype=torch.float64)

        mean = (sum_iq / count_iq).clamp_min(0.0)
        var = (sumsq_iq / count_iq - mean ** 2).clamp_min(0.0)
        std = torch.sqrt(var + 1e-12)
        h5.attrs["mean"] = mean.to(torch.float32).numpy()
        h5.attrs["std"] = std.to(torch.float32).numpy()
        h5.attrs["task"] = task
        h5.attrs["version"] = "v1"

    return output
