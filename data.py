from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import torch
from torch.utils.data import Dataset, random_split

from wavesfm.datasets import (
    IqDataset,
    PositioningDataset,
    RadComDataset,
    RADCOM_OTA_LABELS,
    RadioSignalDataset,
    SensingDataset,
    UwbDataset,
)

SUPPORTED_TASKS = (
    "sensing",
    "rfs",
    "pos",
    "radcom",
    "uwb",
    "amc",
    "aoa",
    "rml",
    "rfp",
    "deepbeam",
    "interf",
)


@dataclass
class TaskInfo:
    name: str
    modality: str  # 'vision' | 'iq'
    target_type: str  # 'classification' | 'position' | 'regression'
    num_classes: int
    in_chans: int | None = None
    coord_min: torch.Tensor | None = None
    coord_max: torch.Tensor | None = None


def _dataset_factory(task: str) -> Callable[[str | Path], Dataset]:
    if task == "sensing":
        return lambda p: SensingDataset(p)
    if task == "rfs":
        return lambda p: RadioSignalDataset(p)
    if task == "pos":
        return lambda p: PositioningDataset(p)
    if task == "radcom":
        return lambda p: RadComDataset(p)
    if task == "uwb":
        return lambda p: UwbDataset(p, as_complex=False)
    if task in {"amc", "rml", "rfp", "deepbeam", "interf"}:
        return lambda p: IqDataset(p, target_key="label")
    if task == "aoa":
        return lambda p: IqDataset(p, target_key="target")
    raise ValueError(f"Unsupported task: {task}")


def _infer_task_info(task: str, dataset: Dataset) -> TaskInfo:
    if task == "sensing":
        num_classes = len(getattr(dataset, "labels", [])) or 6
        return TaskInfo(
            name=task, modality="vision", target_type="classification",
            num_classes=num_classes, in_chans=3,
        )
    if task == "rfs":
        num_classes = len(getattr(dataset, "labels", [])) or 20
        return TaskInfo(
            name=task, modality="vision", target_type="classification",
            num_classes=num_classes, in_chans=1,
        )
    if task == "radcom":
        return TaskInfo(
            name=task, modality="iq", target_type="classification",
            num_classes=len(RADCOM_OTA_LABELS), in_chans=None,
        )
    if task == "pos":
        coord_min = torch.tensor(dataset.norm.get("coord_nominal_min", []), dtype=torch.float32)
        coord_max = torch.tensor(dataset.norm.get("coord_nominal_max", []), dtype=torch.float32)
        sample, label = dataset[0]
        target_dim = int(label.numel()) if torch.is_tensor(label) else len(label)
        return TaskInfo(
            name=task, modality="vision", target_type="position",
            num_classes=target_dim, in_chans=sample.shape[0],
            coord_min=coord_min, coord_max=coord_max,
        )
    if task == "uwb":
        sample, label = dataset[0]
        target_dim = int(label.numel()) if torch.is_tensor(label) else len(label)
        return TaskInfo(
            name=task, modality="iq", target_type="position",
            num_classes=target_dim, in_chans=None,
            coord_min=dataset.loc_min, coord_max=dataset.loc_max,
        )
    if task == "amc":
        num_classes = len(getattr(dataset, "labels", [])) or 7
        return TaskInfo(name=task, modality="iq", target_type="classification", num_classes=num_classes)
    if task == "aoa":
        sample, target = dataset[0]
        target_dim = int(target.numel()) if torch.is_tensor(target) else len(target)
        return TaskInfo(name=task, modality="iq", target_type="regression", num_classes=target_dim)
    if task == "rml":
        num_classes = len(getattr(dataset, "labels", [])) or 11
        return TaskInfo(name=task, modality="iq", target_type="classification", num_classes=num_classes)
    if task == "rfp":
        num_classes = len(getattr(dataset, "labels", [])) or 4
        return TaskInfo(name=task, modality="iq", target_type="classification", num_classes=num_classes)
    if task == "deepbeam":
        num_classes = len(getattr(dataset, "labels", [])) or 5
        return TaskInfo(name=task, modality="iq", target_type="classification", num_classes=num_classes)
    if task == "interf":
        num_classes = len(getattr(dataset, "labels", [])) or 3
        return TaskInfo(name=task, modality="iq", target_type="classification", num_classes=num_classes)
    raise ValueError(f"Unsupported task: {task}")


def build_datasets(
    task: str,
    train_path: str | Path,
    val_path: str | Path | None = None,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, TaskInfo]:
    """
    Create train/val datasets for a given task using preprocessed HDF5 files only.
    """
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Task must be one of {SUPPORTED_TASKS}")

    factory = _dataset_factory(task)
    train_ds = factory(train_path)
    info = _infer_task_info(task, train_ds)

    if val_path:
        val_ds = factory(val_path)
    else:
        if not 0 < val_split < 1:
            raise ValueError("val_split must be in (0, 1) when val_path is omitted.")
        val_size = max(1, int(len(train_ds) * val_split))
        train_size = max(1, len(train_ds) - val_size)
        gen = torch.Generator().manual_seed(seed + 1)
        train_ds, val_ds = random_split(train_ds, [train_size, val_size], generator=gen)

    return train_ds, val_ds, info
