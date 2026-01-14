from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from timm.layers import trunc_normal_

import models_vit
from data import SUPPORTED_TASKS, build_datasets
from lora import create_lora_model
from engine import evaluate, train_one_epoch
from utils import (
    JsonlLogger,
    count_parameters,
    cosine_schedule,
    param_groups_lrd,
    pretty_dict,
    set_seed,
    trim_blocks,
    summarize_finetune_params,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multimodal fine-tuning entrypoint.")
    p.add_argument("--task", required=True, choices=SUPPORTED_TASKS, help="Which dataset to use.")
    p.add_argument("--train-data", dest="train_path", required=True, help="Path to training data.")
    p.add_argument("--val-data", dest="val_path", help="Optional validation data.")
    p.add_argument("--val-split", type=float, default=0.2, help="Val fraction if validation data is not provided.")

    # Model
    p.add_argument("--model", default="vit_multi_small", help="Model name from models_vit.")
    p.add_argument("--lora", action="store_true", help="Enable LoRA adapters on q,v projections.")
    p.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (default: 8).")
    p.add_argument("--lora-alpha", type=float, default=1.0, help="LoRA alpha scaling (default: 1.0).")
    p.add_argument("--global-pool", default="token", choices=["token", "avg"])
    p.add_argument("--vis-patch", type=int, default=16, help="Vision patch size.")
    p.add_argument("--iq-segment-len", type=int, default=16, help="Hop/segment length for IQ tokenization.")
    p.add_argument("--iq-downsample", type=str, default="none", choices=["none", "avg", "conv"])
    p.add_argument("--iq-target-len", type=int, default=256, help="Target IQ length after downsample.")
    p.add_argument("--freeze-encoder", action="store_true", help="Freeze the transformer encoder blocks.")
    p.add_argument("--frozen-blocks", type=int, default=None, help="Freeze only the first N blocks.")
    p.add_argument("--num-blocks", type=int, default=None, help="Use only the first N transformer blocks in the forward pass.")
    p.add_argument("--use-conditional-ln", action="store_true", help="Enable modality-specific conditional LN.")
    p.add_argument("--strict-probe", action="store_true", help="Freeze tokenizer & conditional LN when encoder is frozen.")
    p.add_argument("--sl-baseline", action="store_true", help="Disable encoder freezing (train full model).")

    # Optimization
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps.")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=None, help="Absolute learning rate. If None, use blr scaling.")
    p.add_argument("--blr", type=float, default=1e-3, help="Base LR: lr = blr * batch_size * accum / 256.")
    p.add_argument("--layer-decay", type=float, default=0.75, help="Layer-wise LR decay (1.0 disables).")
    p.add_argument("--min-lr", type=float, default=1e-6, help="Cosine schedule floor.")
    p.add_argument("--warmup-epochs", type=float, default=5.0, help="Linear warmup duration (in epochs).")
    p.add_argument("--smoothing", type=float, default=0.0, help="Label smoothing for classification.")
    p.add_argument("--max-grad-norm", type=float, default=None, help="Gradient clipping (L2 norm).")
    p.add_argument("--class-weights", action="store_true", help="Use class weights for classification loss.")

    # IO
    p.add_argument("--output-dir", default="wavesfm_runs", help="Where to store checkpoints and logs.")
    p.add_argument("--save-every", type=int, default=10, help="Checkpoint frequency in epochs.")
    p.add_argument("--finetune", default="", help="Pretrained checkpoint to initialize from (loads model only).")
    p.add_argument("--resume", default="", help="Resume from checkpoint (model+optim+scheduler).")
    p.add_argument("--eval-only", action="store_true", help="Skip training and run a single validation pass.")

    # Runtime
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print-freq", type=int, default=20)
    p.add_argument("--pin-mem", action="store_true", default=True)
    p.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
    args = p.parse_args()
    if args.iq_downsample == "none":
        args.iq_downsample = None
    return args


def build_model(args: argparse.Namespace, task_info) -> torch.nn.Module:
    model = models_vit.__dict__[args.model](
        modality=task_info.modality,
        global_pool=args.global_pool,
        num_outputs=task_info.num_outputs,
        vis_patch=args.vis_patch,
        vis_in_chans_actual=task_info.in_chans,
        iq_segment_len=args.iq_segment_len,
        iq_downsample=args.iq_downsample,
        iq_target_len=args.iq_target_len,
        use_conditional_ln=args.use_conditional_ln,
    )

    if args.lora:
        model = create_lora_model(model, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)

    return model


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_metric,
    args: argparse.Namespace,
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "args": vars(args),
    }
    torch.save(state, path)
    print(f"[ckpt] saved to {path}")


def _apply_strict_probe(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name == "cls_token" or name.startswith("head.")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(output_dir)

    set_seed(args.seed)
    cudnn.benchmark = True
    device = torch.device(args.device)

    train_ds, val_ds, task_info = build_datasets(
        args.task,
        args.train_path,
        val_path=args.val_path,
        val_split=args.val_split,
        seed=args.seed,
    )
    print(f"[data] task={args.task} train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    model = build_model(args, task_info)
    if args.finetune:
        ckpt = torch.load(args.finetune, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        msg = model.load_state_dict(state, strict=False)
        print(f"[init] loaded finetune checkpoint {args.finetune}")
        print(msg)
        if hasattr(model, "head") and isinstance(model.head, torch.nn.Linear):
            trunc_normal_(model.head.weight, std=2e-5)

    if args.num_blocks is not None:
        model = trim_blocks(model, args.num_blocks)

    freeze_encoder = args.freeze_encoder or not args.sl_baseline
    if freeze_encoder:
        if args.lora and hasattr(model, "freeze_encoder_lora"):
            model.freeze_encoder_lora()
        elif args.frozen_blocks is not None:
            model.freeze_encoder(args.frozen_blocks)
        else:
            model.freeze_encoder()
        if args.strict_probe:
            _apply_strict_probe(model)
        else:
            model.unfreeze_tokenizer()
            model.unfreeze_conditional_ln()

    model = model.to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"[model] {args.model} total={total_params/1e6:.2f}M trainable={trainable_params/1e6:.2f}M")
    for line in summarize_finetune_params(model, total_params, trainable_params):
        print(line)

    if task_info.target_type == "classification":
        ce_kwargs = {}
        if args.smoothing and args.smoothing > 0.0:
            ce_kwargs["label_smoothing"] = float(args.smoothing)
        if args.class_weights and hasattr(train_ds, "class_weights"):
            ce_kwargs["weight"] = train_ds.class_weights.to(device)
        criterion = torch.nn.CrossEntropyLoss(**ce_kwargs)
    else:
        criterion = torch.nn.MSELoss()

    eff_batch = args.batch_size * args.accum_steps
    if args.lr is None:
        args.lr = args.blr * eff_batch / 256
    param_groups = param_groups_lrd(
        model,
        args.weight_decay,
        layer_decay=args.layer_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    scaler = torch.amp.GradScaler(device == "cuda")

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    lr_schedule = cosine_schedule(args.lr, args.min_lr, total_steps, warmup_steps)

    start_epoch = 0
    if task_info.target_type == "classification":
        best_metric = float("-inf")
        best_key = "pca"
        better = lambda cur, best: cur > best
    elif task_info.target_type == "position":
        best_metric = float("inf")
        best_key = "mean_distance_error"
        better = lambda cur, best: cur < best
    else:
        best_metric = float("inf")
        best_key = "mae"
        better = lambda cur, best: cur < best

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_metric = ckpt.get("best_metric", best_metric)
        print(f"[resume] loaded {args.resume} (epoch {start_epoch})")

    if args.eval_only:
        val_stats = evaluate(
            model,
            val_loader,
            device,
            criterion,
            args.task,
            task_info.target_type,
            task_info.num_outputs,
            coord_min=task_info.coord_min,
            coord_max=task_info.coord_max,
        )
        print("[eval-only]", pretty_dict(val_stats))
        return

    print(f"[train] epochs={args.epochs} base_lr={args.lr:.3e} accum_steps={args.accum_steps}")
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        step_offset = epoch * steps_per_epoch
        train_stats = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            device,
            scaler,
            epoch,
            accum_steps=args.accum_steps,
            max_norm=args.max_grad_norm,
            lr_schedule=lr_schedule,
            start_step=step_offset,
            task_type=task_info.target_type,
            print_freq=args.print_freq,
        )

        val_stats = evaluate(
            model,
            val_loader,
            device,
            criterion,
            args.task,
            task_info.target_type,
            task_info.num_outputs,
            coord_min=task_info.coord_min,
            coord_max=task_info.coord_max,
        )

        current = val_stats.get(best_key)
        if current is not None and better(current, best_metric):
            best_metric = float(current)
            save_checkpoint(output_dir / "best.pth", model, optimizer, scaler, epoch, best_metric, args)

        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            save_checkpoint(output_dir / f"checkpoint_{epoch:03d}.pth", model, optimizer, scaler, epoch, best_metric, args)

        log_payload = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_stats,
            "val": val_stats,
            "best_metric": best_metric,
            "best_key": best_key,
        }
        logger.write(log_payload)

    total_time = time.time() - start_time
    time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"[done] training time {time_str} | best {best_key}={best_metric:.4f}")


if __name__ == "__main__":
    main()
