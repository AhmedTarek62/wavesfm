import datetime
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class AverageMeter:
    """Tiny running-average helper."""

    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


def cosine_schedule(base_lr: float, min_lr: float, steps: int, warmup_steps: int) -> list[float]:
    if steps <= 0:
        return []
    warmup_steps = min(warmup_steps, steps)
    sched = []
    for step in range(steps):
        if warmup_steps > 0 and step < warmup_steps:
            lr = base_lr * step / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, steps - warmup_steps)
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
        sched.append(lr)
    return sched


def apply_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        scale = param_group.get("lr_scale", 1.0)
        param_group["lr"] = lr * scale


def get_layer_id_for_vit(name: str, num_layers: int) -> int:
    if name == "cls_token" or "pos_embed" in name:
        return 0
    if name.startswith((
        "patch_embed",
        "vis_patch_embed",
        "iq_segment_embed",
        "iq_ant_embed",
        "iq_downsampler",
        "channel_adapter",
    )):
        return 0
    if name.startswith("blocks"):
        return int(name.split(".")[1]) + 1
    return num_layers


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=None, layer_decay=0.75):
    if no_weight_decay_list is None:
        no_weight_decay_list = []

    param_groups = {}
    if hasattr(model, "blocks"):
        num_layers = len(model.blocks) + 1
    else:
        num_layers = 0

    layer_scales = (
        [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]
        if num_layers > 0 else [1.0]
    )

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(name, num_layers) if num_layers > 0 else 0
        group_name = f"layer_{layer_id}_{g_decay}"

        if group_name not in param_groups:
            lr_scale = layer_scales[layer_id] if num_layers > 0 else 1.0
            param_groups[group_name] = {
                "lr_scale": lr_scale,
                "weight_decay": this_decay,
                "params": [],
            }
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def trim_blocks(model: torch.nn.Module, num_blocks: int) -> torch.nn.Module:
    if not hasattr(model, "blocks"):
        raise ValueError("Model does not expose transformer blocks.")
    total_blocks = len(model.blocks)
    if num_blocks < 1 or num_blocks > total_blocks:
        raise ValueError(f"num_blocks must be in [1, {total_blocks}], got {num_blocks}.")
    if num_blocks == total_blocks:
        return model
    model.blocks = torch.nn.ModuleList(list(model.blocks[:num_blocks]))
    print(f"[model] using first {num_blocks}/{total_blocks} blocks")
    return model


def _format_param_count(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1e6:.2f}M"
    if count >= 1_000:
        return f"{count / 1e3:.2f}K"
    return str(count)


def _group_param_name(name: str) -> str:
    top = name.split(".", 1)[0]
    if top in {"vis_patch_embed", "channel_adapter", "iq_segment_embed", "iq_ant_embed", "iq_downsampler"}:
        return "tokenizer"
    if top in {"mod_ln_scale", "mod_ln_bias"}:
        return "conditional_ln"
    return top


def summarize_finetune_params(
    model: torch.nn.Module,
    total_params: int,
    trainable_params: int,
) -> list[str]:
    groups: dict[str, dict[str, int]] = {}
    for name, param in model.named_parameters():
        group = _group_param_name(name)
        stats = groups.setdefault(group, {"total": 0, "trainable": 0})
        numel = int(param.numel())
        stats["total"] += numel
        if param.requires_grad:
            stats["trainable"] += numel

    frozen_params = total_params - trainable_params
    lines = [
        (
            f"[params] total={_format_param_count(total_params)} "
            f"trainable={_format_param_count(trainable_params)} "
            f"frozen={_format_param_count(frozen_params)}"
        )
    ]

    trainable_groups = []
    frozen_groups = []
    partial_groups = []
    for group, stats in groups.items():
        if stats["trainable"] == 0:
            frozen_groups.append((group, stats))
        elif stats["trainable"] == stats["total"]:
            trainable_groups.append((group, stats))
        else:
            partial_groups.append((group, stats))

    def _sort(items):
        return sorted(items, key=lambda item: item[1]["total"], reverse=True)

    if trainable_groups:
        lines.append("[params] trainable groups:")
        for group, stats in _sort(trainable_groups):
            lines.append(f"  - {group}: {_format_param_count(stats['total'])}")
    if frozen_groups:
        lines.append("[params] frozen groups:")
        for group, stats in _sort(frozen_groups):
            lines.append(f"  - {group}: {_format_param_count(stats['total'])}")
    if partial_groups:
        lines.append("[params] partial groups (trainable/total):")
        for group, stats in _sort(partial_groups):
            lines.append(
                f"  - {group}: {_format_param_count(stats['trainable'])} / {_format_param_count(stats['total'])}"
            )

    return lines


class JsonlLogger:
    """Append metrics as JSON lines and mirror key stats to stdout."""

    def __init__(self, output_dir: Path):
        self.log_path = Path(output_dir) / "log.txt"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict) -> None:
        stamp = datetime.datetime.now().isoformat()
        payload = {"timestamp": stamp, **payload}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")


def pretty_dict(d: dict, prefix: str = "") -> str:
    return prefix + " ".join(f"{k}={v:.4f}" if isinstance(v, (float, int)) else f"{k}={v}" for k, v in d.items())

