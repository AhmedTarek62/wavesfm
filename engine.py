from __future__ import annotations

from typing import Dict

import torch
from timm.utils import accuracy

from dataset_classes import RADCOM_OTA_LABELS
from utils import AverageMeter, apply_lr, pretty_dict


def _unpack_batch(batch):
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError("Expected batch to be (inputs, targets[, ...]).")


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    *,
    accum_steps: int = 1,
    max_norm: float | None = None,
    lr_schedule: list[float] | None = None,
    start_step: int = 0,
    task_type: str = "classification",
    print_freq: int = 20,
) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    mae_meter = AverageMeter()

    autocast_enabled = device.type == "cuda"
    total_steps = len(data_loader)
    global_step = start_step

    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(data_loader):
        if lr_schedule and step % accum_steps == 0:
            apply_lr(optimizer, lr_schedule[global_step])

        samples, targets = _unpack_batch(batch)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            outputs = model(samples)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        if (step + 1) % accum_steps == 0:
            if max_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        loss_meter.update(loss.item(), n=samples.size(0))
        if task_type == "classification":
            acc1, _ = accuracy(outputs, targets, topk=(1, min(3, outputs.shape[1])))
            acc_meter.update(acc1.item(), n=samples.size(0))
        elif task_type == "regression":
            mae = (outputs.squeeze() - targets.squeeze()).abs().mean().item()
            mae_meter.update(mae, n=samples.size(0))

        if step % print_freq == 0 or step + 1 == total_steps:
            lr_now = optimizer.param_groups[0]["lr"]
            msg = {
                "epoch": epoch,
                "step": f"{step + 1}/{total_steps}",
                "loss": loss_meter.avg,
                "lr": lr_now,
            }
            if task_type == "classification":
                msg["acc1"] = acc_meter.avg
            if task_type == "regression":
                msg["mae"] = mae_meter.avg
            print(pretty_dict(msg, prefix="[train] "))

        global_step += 1

    stats = {"loss": loss_meter.avg}
    if task_type == "classification":
        stats["acc1"] = acc_meter.avg
    if task_type == "regression":
        stats["mae"] = mae_meter.avg
    return stats


@torch.no_grad()
def evaluate_classification(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    criterion: torch.nn.Module,
    num_outputs: int,
) -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc3_meter = AverageMeter()
    per_class_correct = torch.zeros(num_outputs, device=device)
    per_class_total = torch.zeros(num_outputs, device=device)

    autocast_enabled = device.type == "cuda"
    for batch in data_loader:
        samples, targets = _unpack_batch(batch)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_meter.update(loss.item(), n=samples.size(0))
        acc1, acc3 = accuracy(outputs, targets, topk=(1, min(3, outputs.shape[1])))
        acc1_meter.update(acc1.item(), n=samples.size(0))
        acc3_meter.update(acc3.item(), n=samples.size(0))

        _, pred = outputs.max(1)
        for i in range(samples.size(0)):
            lbl = targets[i]
            per_class_total[lbl] += 1
            if pred[i] == lbl:
                per_class_correct[lbl] += 1

    pca = torch.where(
        per_class_total > 0,
        per_class_correct / per_class_total * 100.0,
        torch.zeros_like(per_class_total),
    ).mean().item()

    stats = {
        "loss": loss_meter.avg,
        "acc1": acc1_meter.avg,
        "acc3": acc3_meter.avg,
        "pca": pca,
    }
    print(pretty_dict(stats, prefix="[val] "))
    return stats


@torch.no_grad()
def evaluate_radcom(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    criterion: torch.nn.Module,
) -> Dict[str, float]:
    model.eval()
    num_classes = len(RADCOM_OTA_LABELS)
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc3_meter = AverageMeter()
    per_class_correct = torch.zeros(num_classes, device=device)
    per_class_total = torch.zeros(num_classes, device=device)
    mod_correct = torch.zeros(1, device=device)
    mod_total = torch.zeros(1, device=device)
    sig_correct = torch.zeros(1, device=device)
    sig_total = torch.zeros(1, device=device)

    mod_to_idx = {m: i for i, m in enumerate(sorted({m for m, _ in RADCOM_OTA_LABELS}))}
    sig_to_idx = {s: i for i, s in enumerate(sorted({s for _, s in RADCOM_OTA_LABELS}))}

    autocast_enabled = device.type == "cuda"
    for batch in data_loader:
        samples, targets = _unpack_batch(batch)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_meter.update(loss.item(), n=samples.size(0))
        acc1, acc3 = accuracy(outputs, targets, topk=(1, min(3, outputs.shape[1])))
        acc1_meter.update(acc1.item(), n=samples.size(0))
        acc3_meter.update(acc3.item(), n=samples.size(0))

        _, pred = outputs.max(1)
        for i in range(samples.size(0)):
            lbl = targets[i]
            per_class_total[lbl] += 1
            if pred[i] == lbl:
                per_class_correct[lbl] += 1

            true_mod, true_sig = RADCOM_OTA_LABELS[int(lbl)]
            pred_mod, pred_sig = RADCOM_OTA_LABELS[int(pred[i])]
            if mod_to_idx[true_mod] == mod_to_idx[pred_mod]:
                mod_correct += 1
            mod_total += 1
            if sig_to_idx[true_sig] == sig_to_idx[pred_sig]:
                sig_correct += 1
            sig_total += 1

    pca = torch.where(
        per_class_total > 0,
        per_class_correct / per_class_total * 100.0,
        torch.zeros_like(per_class_total),
    ).mean().item()
    mod_acc = 100.0 * float(mod_correct.item()) / max(1, int(mod_total.item()))
    sig_acc = 100.0 * float(sig_correct.item()) / max(1, int(sig_total.item()))

    stats = {
        "loss": loss_meter.avg,
        "acc1": acc1_meter.avg,
        "acc3": acc3_meter.avg,
        "pca": pca,
        "mod_acc": mod_acc,
        "sig_acc": sig_acc,
    }
    print(pretty_dict(stats, prefix="[val] "))
    return stats


@torch.no_grad()
def evaluate_positioning(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    criterion: torch.nn.Module,
    coord_min: torch.Tensor,
    coord_max: torch.Tensor,
) -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()

    autocast_enabled = device.type == "cuda"
    coord_min = coord_min.to(device)
    coord_max = coord_max.to(device)

    def _denorm(x: torch.Tensor) -> torch.Tensor:
        return (x + 1) * 0.5 * (coord_max - coord_min) + coord_min

    all_errors = []
    for batch in data_loader:
        samples, targets = _unpack_batch(batch)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_meter.update(loss.item(), n=samples.size(0))
        pred = _denorm(outputs)
        true = _denorm(targets)
        all_errors.append(torch.linalg.norm(pred - true, dim=-1))

    dist = torch.cat(all_errors, dim=0)
    mean_dist = dist.mean().item()
    std_dist = dist.std(unbiased=False).item()
    med_dist = dist.median().item()
    p75 = torch.quantile(dist, 0.75).item()
    p90 = torch.quantile(dist, 0.90).item()

    stats = {
        "loss": loss_meter.avg,
        "mean_distance_error": mean_dist,
        "stdev_distance_error": std_dist,
        "median_distance_error": med_dist,
        "p75_distance_error": p75,
        "p90_distance_error": p90,
    }
    print(pretty_dict(stats, prefix="[val] "))
    return stats


@torch.no_grad()
def evaluate_regression(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    criterion: torch.nn.Module,
) -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    mae_meter = AverageMeter()
    rmse_meter = AverageMeter()

    autocast_enabled = device.type == "cuda"
    for batch in data_loader:
        samples, targets = _unpack_batch(batch)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_meter.update(loss.item(), n=samples.size(0))
        err = (outputs.squeeze() - targets.squeeze())
        mae = err.abs().mean().item()
        rmse = (err.pow(2).mean().sqrt().item())
        mae_meter.update(mae, n=samples.size(0))
        rmse_meter.update(rmse, n=samples.size(0))

    stats = {"loss": loss_meter.avg, "mae": mae_meter.avg, "rmse": rmse_meter.avg}
    print(pretty_dict(stats, prefix="[val] "))
    return stats


def evaluate(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    criterion: torch.nn.Module,
    task_name: str,
    task_type: str,
    num_outputs: int,
    coord_min: torch.Tensor | None = None,
    coord_max: torch.Tensor | None = None,
) -> Dict[str, float]:
    if task_type == "classification" and task_name == "radcom":
        return evaluate_radcom(model, data_loader, device, criterion)
    if task_type == "classification":
        return evaluate_classification(model, data_loader, device, criterion, num_outputs)
    if task_type == "position":
        assert coord_min is not None and coord_max is not None, "coord_min/max required for position tasks"
        return evaluate_positioning(model, data_loader, device, criterion, coord_min, coord_max)
    if task_type == "regression":
        return evaluate_regression(model, data_loader, device, criterion)
    raise ValueError(f"Unknown task type: {task_type}")
