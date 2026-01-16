from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple
import json

import h5py
import torch
from torch.utils.data import Dataset, Subset, random_split

from dataset_classes import (
    ImageDataset,
    RadComOta,
    RML,
    Powder,
    Icarus,
    UWBIndoor,
    UWBIndustrial,
    DeepMIMO,
)

SUPPORTED_TASKS = (
    "sensing",
    "rfs",
    "pos",
    "radcom",
    "uwb-indoor",
    "uwb-industrial",
    "rml",
    "rfp",
    "interf",
    "deepmimo-los",
    "deepmimo-beam",
)


@dataclass
class TaskInfo:
    name: str
    modality: str  # 'vision' | 'iq'
    target_type: str  # 'classification' | 'position' | 'regression'
    num_outputs: int
    in_chans: int | None = None
    coord_min: torch.Tensor | None = None
    coord_max: torch.Tensor | None = None


def _dataset_factory(task: str) -> Callable[[str | Path], Dataset]:
    if task == "sensing":
        return lambda p: ImageDataset(p, sample_key="csi", label_key="label")
    if task == "rfs":
        return lambda p: ImageDataset(p, sample_key="image", label_key="label")
    if task == "pos":
        return lambda p: ImageDataset(p, sample_key="features", label_key="label", label_dtype=torch.float32)
    if task == "radcom":
        return lambda p: RadComOta(p)
    if task == "uwb-indoor":
        return lambda p: UWBIndoor(p, as_complex=False)
    if task == "uwb-industrial":
        return lambda p: UWBIndustrial(p, as_complex=False)
    if task == "rml":
        return lambda p: RML(p)
    if task == "rfp":
        return lambda p: Powder(p)
    if task == "interf":
        return lambda p: Icarus(p)
    raise ValueError(f"Unsupported task: {task}")


def _load_class_weights(h5_path: Path, attr_key: str = "class_weights") -> torch.Tensor | None:
    with h5py.File(h5_path, "r") as h5:
        cw = h5.attrs.get(attr_key, None)
        if cw is None:
            return None
        return torch.as_tensor(cw, dtype=torch.float32)


def _infer_task_info(task: str, dataset: Dataset) -> TaskInfo:
    if task == "sensing":
        num_classes = 6
        return TaskInfo(
            name=task, modality="vision", target_type="classification",
            num_outputs=num_classes, in_chans=3,
        )
    if task == "rfs":
        num_classes = 20
        return TaskInfo(
            name=task, modality="vision", target_type="classification",
            num_outputs=num_classes, in_chans=1,
        )
    if task == "radcom":
        return TaskInfo(
            name=task, modality="iq", target_type="classification",
            num_outputs=9, in_chans=None,
        )
    if task == "pos":
        coord_min = coord_max = torch.tensor([], dtype=torch.float32)
        if hasattr(dataset, "h5_path"):
            with h5py.File(dataset.h5_path, "r") as h5:
                coord_min = torch.tensor(json.loads(h5.attrs.get("coord_nominal_min", "[]")), dtype=torch.float32)
                coord_max = torch.tensor(json.loads(h5.attrs.get("coord_nominal_max", "[]")), dtype=torch.float32)
        sample, label = dataset[0]
        target_dim = int(label.numel()) if torch.is_tensor(label) else len(label)
        return TaskInfo(
            name=task, modality="vision", target_type="position",
            num_outputs=target_dim, in_chans=sample.shape[0],
            coord_min=coord_min, coord_max=coord_max,
        )
    if task == "uwb-indoor":
        sample, label = dataset[0]
        target_dim = int(label.numel()) if torch.is_tensor(label) else len(label)
        return TaskInfo(
            name=task, modality="iq", target_type="position",
            num_outputs=target_dim, in_chans=None,
            coord_min=dataset.loc_min, coord_max=dataset.loc_max,
        )
    if task == "uwb-industrial":
        sample, label = dataset[0]
        target_dim = int(label.numel()) if torch.is_tensor(label) else len(label)
        return TaskInfo(
            name=task, modality="iq", target_type="position",
            num_outputs=target_dim, in_chans=None,
            coord_min=dataset.loc_min, coord_max=dataset.loc_max,
        )
    if task == "rml":
        num_classes = 11
        return TaskInfo(name=task, modality="iq", target_type="classification", num_outputs=num_classes)
    if task == "rfp":
        num_classes = 4
        return TaskInfo(name=task, modality="iq", target_type="classification", num_outputs=num_classes)
    if task == "interf":
        num_classes = 3
        return TaskInfo(name=task, modality="iq", target_type="classification", num_outputs=num_classes)
    if task == "deepmimo-los":
        return TaskInfo(
            name=task,
            modality="vision",
            target_type="classification",
            num_outputs=2,
            in_chans=2,
        )
    if task == "deepmimo-beam":
        n_beams = getattr(dataset, "effective_n_beams", None) or getattr(dataset, "selected_n_beams", None)
        if not n_beams:
            raise ValueError("DeepMIMO beam dataset missing selected n_beams.")
        return TaskInfo(
            name=task,
            modality="vision",
            target_type="classification",
            num_outputs=int(n_beams),
            in_chans=2,
        )
    raise ValueError(f"Unsupported task: {task}")


def _label_to_int(label) -> int:
    if torch.is_tensor(label):
        if label.numel() != 1:
            raise ValueError("Stratified split requires scalar class labels.")
        return int(label.item())
    if isinstance(label, (list, tuple)):
        if len(label) != 1:
            raise ValueError("Stratified split requires scalar class labels.")
        return int(label[0])
    return int(label)


def _extract_labels(dataset: Dataset) -> list[int]:
    if hasattr(dataset, "h5_path") and hasattr(dataset, "label_key"):
        with h5py.File(Path(dataset.h5_path), "r") as h5:
            labels = h5[dataset.label_key][:]
        labels_tensor = torch.as_tensor(labels).reshape(-1)
        if labels_tensor.numel() != len(dataset):
            raise ValueError("Stratified split requires scalar class labels.")
        return [int(v) for v in labels_tensor.tolist()]

    labels = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise ValueError("Dataset __getitem__ must return (sample, label, ...) for stratified split.")
        labels.append(_label_to_int(item[1]))
    return labels


def _stratified_split(dataset: Dataset, val_split: float, seed: int) -> Tuple[Dataset, Dataset]:
    labels = _extract_labels(dataset)
    if len(labels) != len(dataset):
        raise ValueError("Label count mismatch for stratified split.")

    class_to_indices: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        class_to_indices.setdefault(int(label), []).append(idx)

    gen = torch.Generator().manual_seed(seed + 1)
    for indices in class_to_indices.values():
        if len(indices) > 1:
            perm = torch.randperm(len(indices), generator=gen).tolist()
            indices[:] = [indices[i] for i in perm]

    total = len(dataset)
    desired_val = max(1, int(total * val_split))
    val_counts: dict[int, int] = {}
    remainders = []
    base_total = 0
    for label, indices in class_to_indices.items():
        raw = len(indices) * val_split
        base = int(raw)
        base_total += base
        val_counts[label] = base
        remainders.append((raw - base, label))

    remainder = desired_val - base_total
    if remainder > 0:
        remainders.sort(key=lambda item: (-item[0], item[1]))
        for frac, label in remainders[:remainder]:
            val_counts[label] += 1
    elif remainder < 0:
        remove = -remainder
        remainders.sort(key=lambda item: (item[0], item[1]))
        for frac, label in remainders:
            if remove <= 0:
                break
            if val_counts[label] > 0:
                val_counts[label] -= 1
                remove -= 1

    train_indices = []
    val_indices = []
    for label, indices in class_to_indices.items():
        val_count = val_counts[label]
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    if len(train_indices) > 1:
        perm = torch.randperm(len(train_indices), generator=gen).tolist()
        train_indices = [train_indices[i] for i in perm]
    if len(val_indices) > 1:
        perm = torch.randperm(len(val_indices), generator=gen).tolist()
        val_indices = [val_indices[i] for i in perm]

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def build_datasets(
    task: str,
    train_path: str | Path,
    val_path: str | Path | None = None,
    val_split: float = 0.2,
    stratified_split: bool = False,
    seed: int = 42,
    deepmimo_n_beams: int | None = None,
) -> Tuple[Dataset, Dataset, TaskInfo]:
    """
    Create train/val datasets for a given task using preprocessed HDF5 files only.
    """
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Task must be one of {SUPPORTED_TASKS}")

    factory = _dataset_factory(task)
    if task.startswith("deepmimo-"):
        return _build_deepmimo_datasets(
            task=task,
            train_path=train_path,
            val_path=val_path,
            val_split=val_split,
            stratified_split=stratified_split,
            seed=seed,
            deepmimo_n_beams=deepmimo_n_beams,
        )

    train_ds = factory(train_path)
    cw = _load_class_weights(Path(train_path))
    if cw is not None:
        train_ds.class_weights = cw

    info = _infer_task_info(task, train_ds)

    if val_path:
        val_ds = factory(val_path)
        if cw is None:
            cw_val = _load_class_weights(Path(val_path))
            if cw_val is not None:
                train_ds.class_weights = cw_val
    else:
        if not 0 < val_split < 1:
            raise ValueError("val_split must be in (0, 1) when val_path is omitted.")
        if stratified_split and info.target_type == "classification":
            train_ds, val_ds = _stratified_split(train_ds, val_split, seed)
        else:
            val_size = max(1, int(len(train_ds) * val_split))
            train_size = max(1, len(train_ds) - val_size)
            gen = torch.Generator().manual_seed(seed + 1)
            train_ds, val_ds = random_split(train_ds, [train_size, val_size], generator=gen)

    return train_ds, val_ds, info


def _build_deepmimo_datasets(
    task: str,
    train_path: str | Path,
    val_path: str | Path | None,
    val_split: float,
    stratified_split: bool,
    seed: int,
    deepmimo_n_beams: int | None,
) -> Tuple[Dataset, Dataset, TaskInfo]:
    if val_path:
        raise ValueError("DeepMIMO uses val_split only; do not pass val_path.")
    if task == "deepmimo-beam" and deepmimo_n_beams is None:
        raise ValueError("deepmimo-beam requires --deepmimo-n-beams to select label_beam_{n}.")

    if task == "deepmimo-beam":
        train_ds = DeepMIMO(train_path, n_beams=deepmimo_n_beams, label_key="label_beam")
        cw_key = f"class_weights_beam_{int(deepmimo_n_beams)}"
    else:
        train_ds = DeepMIMO(train_path, label_key="label_los")
        cw_key = "class_weights_los"

    cw = _load_class_weights(Path(train_path), attr_key=cw_key)
    if cw is not None:
        train_ds.class_weights = cw

    info = _infer_task_info(task, train_ds)

    if not 0 < val_split < 1:
        raise ValueError("val_split must be in (0, 1) when val_path is omitted.")
    if stratified_split and info.target_type == "classification":
        train_ds, val_ds = _stratified_split(train_ds, val_split, seed)
    else:
        val_size = max(1, int(len(train_ds) * val_split))
        train_size = max(1, len(train_ds) - val_size)
        gen = torch.Generator().manual_seed(seed + 1)
        train_ds, val_ds = random_split(train_ds, [train_size, val_size], generator=gen)

    return train_ds, val_ds, info
