"""
Runner to sweep WavesFM finetuning across tasks / modes / seeds.

Modes:
  - lp: linear probe (encoder frozen)
  - ft2: partially finetune (freeze first N blocks)
  - lora: LoRA adapters
  - strict: strict probe (head + cls token only)
  - sl: supervised baseline (train full model)

Use CLI args to set dataset root, output root, and checkpoint path so this works
across machines without editing the file. Defaults assume preprocessed .h5 caches.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys



REPO_ROOT = Path(__file__).resolve().parent

# Defaults (override via CLI)
DEFAULT_DATA_ROOT = Path("/home/ahmed/data/finetuning")
DEFAULT_OUTPUT_ROOT = Path("/home/ahmed/runs/wavesfm-finetune")
DEFAULT_CKPT = None
DEFAULT_MODEL_NAME = "sm"

DEFAULT_TASKS = (
    "sensing",
    "pos",
    "rfs",
    "interf",
    "rfp",
    "rml",
    "uwb-indoor",
    "uwb-industrial",
    "radcom",
    "deepmimo-los",
    "deepmimo-beam",
)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_MODES = ("lp", "ft2", "lora", "strict")

# Epochs (fallback to DEFAULT_EPOCHS if task not listed)
TASK_EPOCHS = {
    "rfp": 10,
    "interf": 35,
    "uwb-indoor": 50,
    "uwb-industrial": 50,
    "rml": 50,
    "radcom": 50,
}
DEFAULT_EPOCHS = 100

# Common args
MODEL_ARCH = "vit_multi_small"
BATCH_SIZE = 256
DEFAULT_NUM_WORKERS = 2
WARMUP_EPOCHS = 5
USE_CONDITIONAL_LN = True
COMMON_FLAGS = [
    "--model",
    MODEL_ARCH,
    "--warmup-epochs",
    str(WARMUP_EPOCHS),
]
if USE_CONDITIONAL_LN:
    COMMON_FLAGS.append("--use-conditional-ln")

SMOOTH_TASKS = {"sensing": 0.1, "rfp": 0.1, "interf": 0.02, "rfs": 0.05}
STRATIFIED_TASKS = {"rfs", "interf", "deepmimo-los", "deepmimo-beam"}
TASK_BATCH_SIZE = {
    "rml": 2048,
    "uwb-indoor": 256,
    "uwb-industrial": 512,
    "radcom": 2048,
}
LORA_RANK = 32
LORA_ALPHA = 64
FT2_FROZEN_BLOCKS = 6
INTERF_ACCUM = 2


def _load_log_entries(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    entries = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _pick_best_entry(entries: list[dict]) -> dict | None:
    best_entry = None
    last_best = None
    for entry in entries:
        cur = entry.get("best_metric")
        if cur is None:
            continue
        if last_best is None or cur != last_best:
            last_best = cur
            best_entry = entry
    return best_entry


def _load_summary(summary_path: Path) -> list[dict]:
    if not summary_path.exists():
        return []
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(payload, list):
        return payload
    return []

def _upsert_summary(entries: list[dict], summary: dict) -> list[dict]:
    run_name = summary.get("run_name")
    if not run_name:
        entries.append(summary)
        return entries
    for idx, entry in enumerate(entries):
        if entry.get("run_name") == run_name:
            entries[idx] = summary
            return entries
    entries.append(summary)
    return entries

def _write_summary(summary_path: Path, entries: list[dict]) -> None:
    summary_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")

def parse_args():
    p = argparse.ArgumentParser(description="Sweep WavesFM finetuning runs.")
    p.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Base directory containing finetune caches.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Base directory to save finetune checkpoints/logs.",
    )
    p.add_argument(
        "--ckpt-path",
        type=Path,
        default=DEFAULT_CKPT,
        help="Optional pretrained checkpoint to finetune from.",
    )
    p.add_argument(
        "--ckpt-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Name tag used in output folder/run names.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="DataLoader workers.",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        choices=list(DEFAULT_TASKS),
        help="Tasks to run.",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Seeds to run.",
    )
    p.add_argument(
        "--modes",
        nargs="+",
        default=list(DEFAULT_MODES),
        choices=["lp", "ft2", "lora", "strict", "sl"],
        help="Finetune modes to run.",
    )
    p.add_argument(
        "--trim-blocks",
        type=int,
        default=None,
        help="Use only the first N transformer blocks in the forward pass.",
    )
    p.add_argument(
        "--deepmimo-n-beams",
        type=int,
        default=64,
        help="Select DeepMIMO beam label variant (uses label_beam_{n}); default 64 for beam runs.",
    )
    p.add_argument(
        "--path-override",
        action="append",
        default=[],
        help="Override a task path, format task=/abs/path (can repeat).",
    )
    p.add_argument(
        "--val-split",
        type=float,
        default=None,
        help="Override val split fraction when val data is not provided.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands only.")
    p.add_argument(
        "--skip-if-done",
        action="store_true",
        default=True,
        help="Skip runs if final checkpoint exists.",
    )
    return p.parse_args()


def _build_data_paths(root: Path) -> dict:
    return {
        "pos": root / "nrpos-outdoor.h5",
        "rfs": root / "rfs.h5",
        "sensing": root / "has.h5",
        "rfp": root / "rfp.h5",
        "interf": root / "icarus.h5",
        "rml": root / "rml22.h5",
        "uwb-indoor": root / "environment0.h5",
        "uwb-industrial": root / "ipin-train.h5",
        "radcom": root / "radcom.h5",
        "deepmimo-los": root / "deepmimo.h5",
        "deepmimo-beam": root / "deepmimo.h5",
    }


def _apply_overrides(data_paths: dict, overrides: list):
    for entry in overrides:
        if "=" not in entry:
            raise ValueError(f"Invalid override '{entry}'. Use task=/abs/path")
        task, path = entry.split("=", 1)
        task = task.strip()
        if task not in data_paths:
            raise ValueError(f"Unknown task in override: {task}")
        data_paths[task] = Path(path).expanduser().resolve()


def _validate_paths(data_paths: dict, tasks: list, ckpt: Path | None):
    missing = [data_paths[t] for t in tasks if not data_paths[t].exists()]
    if missing:
        raise FileNotFoundError(f"Missing data paths: {missing}")

    if ckpt is not None and not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")


def main():
    args = parse_args()
    data_paths = _build_data_paths(args.data_root)
    _apply_overrides(data_paths, args.path_override)
    _validate_paths(data_paths, args.tasks, args.ckpt_path)
    args.output_root.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        for mode in args.modes:
            for task in args.tasks:
                data_path = data_paths[task]
                epochs = TASK_EPOCHS.get(task, DEFAULT_EPOCHS)

                mode_tag = mode
                out_dir = args.output_root / f"{args.ckpt_name}_{mode_tag}" / task / f"s{seed}"
                out_dir.mkdir(parents=True, exist_ok=True)
                log_file = out_dir / "train.log"
                run_name = f"{args.ckpt_name}_{task}_{mode_tag}_s{seed}"

                batch_size = TASK_BATCH_SIZE.get(task, BATCH_SIZE)

                cmd = [
                    sys.executable,
                    str(REPO_ROOT / "main_finetune.py"),
                    "--task",
                    task,
                    "--train-data",
                    str(data_path),
                    "--output-dir",
                    str(out_dir),
                    "--batch-size",
                    str(batch_size),
                    "--num-workers",
                    str(args.num_workers),
                    "--epochs",
                    str(epochs),
                    "--seed",
                    str(seed),
                    *COMMON_FLAGS,
                ]

                if args.val_split is not None:
                    cmd += ["--val-split", str(args.val_split)]

                if task in STRATIFIED_TASKS:
                    cmd += ["--stratified-split", "--class-weights"]

                if args.trim_blocks is not None:
                    cmd += ["--trim-blocks", str(args.trim_blocks)]

                if mode == "sl":
                    cmd.append("--sl-baseline")
                elif args.ckpt_path is not None:
                    cmd += ["--finetune", str(args.ckpt_path)]

                if mode == "lora":
                    cmd += ["--lora", "--lora-rank", str(LORA_RANK), "--lora-alpha", str(LORA_ALPHA)]
                elif mode == "ft2":
                    cmd += ["--frozen-blocks", str(FT2_FROZEN_BLOCKS)]
                elif mode == "strict":
                    cmd.append("--strict-probe")

                if task == "interf":
                    cmd += ["--accum-steps", str(INTERF_ACCUM)]

                if task == "deepmimo-beam" and args.deepmimo_n_beams is not None:
                    cmd += ["--deepmimo-n-beams", str(args.deepmimo_n_beams)]

                if task in SMOOTH_TASKS:
                    cmd += ["--smoothing", str(SMOOTH_TASKS[task])]

                if task.startswith("deepmimo"):
                    cmd += ["--vis-img-size", "32"]                    


                pretty = " ".join(cmd)
                print(f"[{mode.upper()}] MODEL={args.ckpt_name} TASK={task} SEED={seed}")
                print(f"  RUN={run_name}")
                print(f"  CMD: {pretty}\n")

                final_ckpt = out_dir / f"checkpoint_{epochs-1:03d}.pth" 
                best_ckpt = out_dir / "best.pth"
                skip_train = args.skip_if_done and (final_ckpt.exists() or best_ckpt.exists())
                if skip_train:
                    print("  SKIP (final checkpoint exists)\n")

                if args.dry_run:
                    continue

                if not skip_train:
                    with open(log_file, "a", encoding="utf-8") as lf:
                        lf.write(pretty + "\n")
                        lf.flush()
                        subprocess.run(cmd, stdout=lf, stderr=lf, check=True)
                    print("  DONE\n")


                summary_path = args.output_root / f"{args.ckpt_name}_{mode_tag}" / "summary.json"
                log_path = out_dir / "log.txt"
                best_ckpt = out_dir / "best.pth"
                entries = _load_log_entries(log_path)
                best_entry = _pick_best_entry(entries)
                if not entries or best_entry is None:
                    print("  WARN (log missing or empty; no summary)\n")
                    continue

                if not best_ckpt.exists():
                    print("  WARN (best checkpoint missing; summary from log)\n")

                summary = {
                    "run_name": run_name,
                    "task": task,
                    "seed": seed,
                    "mode": mode_tag,
                    "ckpt_name": args.ckpt_name,
                    "best_ckpt": str(best_ckpt),
                    "best_epoch": best_entry.get("epoch"),
                    "best_key": best_entry.get("best_key"),
                    "best_metric": best_entry.get("best_metric"),
                    "metrics": best_entry.get("val"),
                    "train": best_entry.get("train"),
                }
                summary_entries = _load_summary(summary_path)
                summary_entries = _upsert_summary(summary_entries, summary)
                _write_summary(summary_path, summary_entries)
                print(f"  SUMMARY: {summary_path}\n")


if __name__ == "__main__":
    main()
