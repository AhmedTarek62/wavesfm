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
        if step < warmup_steps:
            lr = base_lr * (step + 1) / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, steps - warmup_steps)
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
        sched.append(lr)
    return sched


def apply_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


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

