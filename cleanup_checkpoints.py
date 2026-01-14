import argparse
import json
import os
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple


Checkpoint = Tuple[int, Path]

METRIC_PREFERENCES: List[Tuple[str, Callable[[float, float], bool]]] = [
    ("pca", lambda cur, best: cur > best),
    ("mean_distance_error", lambda cur, best: cur < best),
    ("mae", lambda cur, best: cur < best),
    ("acc1", lambda cur, best: cur > best),
    ("acc3", lambda cur, best: cur > best),
    ("mod_acc", lambda cur, best: cur > best),
    ("sig_acc", lambda cur, best: cur > best),
    ("rmse", lambda cur, best: cur < best),
    ("loss", lambda cur, best: cur < best),
]


def parse_log_lines(log_path: Path) -> List[Dict]:
    records: List[Dict] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[WARN] Skipping invalid JSON in {log_path} (line {line_no})")
    return records


def detect_run_type(records: Iterable[Dict]) -> str:
    for record in records:
        if any(key in record for key in ("val", "best_metric", "best_key")):
            return "finetune"
    return "pretrain"


def extract_metric(record: Dict, key: str) -> Optional[float]:
    if key in record:
        try:
            return float(record[key])
        except (TypeError, ValueError):
            return None
    val = record.get("val")
    if isinstance(val, dict) and key in val:
        try:
            return float(val[key])
        except (TypeError, ValueError):
            return None
    return None


def better_for_key(key: str) -> Optional[Callable[[float, float], bool]]:
    for metric_key, better in METRIC_PREFERENCES:
        if key == metric_key:
            return better
    if "acc" in key or key == "pca":
        return lambda cur, best: cur > best
    if "error" in key or "loss" in key or key in {"mae", "rmse"}:
        return lambda cur, best: cur < best
    return None


def pick_metric(records: Iterable[Dict]) -> Tuple[Optional[str], Optional[Callable[[float, float], bool]]]:
    records_list = list(records)
    for record in reversed(records_list):
        key = record.get("best_key")
        if key:
            better = better_for_key(key)
            if better is not None:
                return key, better

    available = set()
    for record in records_list:
        val = record.get("val")
        if isinstance(val, dict):
            available.update(val.keys())

    for key, better in METRIC_PREFERENCES:
        if key in available:
            return key, better
    return None, None


def find_best_epoch_from_best_metric(records: Iterable[Dict]) -> Tuple[Optional[int], Optional[str]]:
    best_entry = None
    last_best = None
    for record in records:
        if "epoch" not in record or "best_metric" not in record:
            continue
        cur = record.get("best_metric")
        if last_best is None or cur != last_best:
            last_best = cur
            best_entry = record
    if best_entry is None:
        return None, None
    try:
        return int(best_entry["epoch"]), best_entry.get("best_key")
    except (TypeError, ValueError):
        return None, None


def find_best_epoch_by_val(records: Iterable[Dict]) -> Tuple[Optional[int], Optional[str]]:
    key, better = pick_metric(records)
    if key is None or better is None:
        return None, None

    best_val = None
    best_epoch = None
    for record in records:
        if "epoch" not in record:
            continue
        current = extract_metric(record, key)
        if current is None:
            continue
        try:
            epoch = int(record["epoch"])
        except (TypeError, ValueError):
            continue
        if best_val is None or better(current, best_val):
            best_val = current
            best_epoch = epoch

    return best_epoch, key


def closest_checkpoint_epoch(target_epoch: int, checkpoints: List[Checkpoint]) -> int:
    return min(checkpoints, key=lambda item: (abs(item[0] - target_epoch), item[0]))[0]


def collect_checkpoints(run_dir: Path) -> List[Checkpoint]:
    ckpts: List[Checkpoint] = []
    pattern = re.compile(r"checkpoint_(\d+)\.pth$")
    for path in run_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match:
            ckpts.append((int(match.group(1)), path))
    ckpts.sort(key=lambda item: item[0])
    return ckpts


def pick_keep_checkpoint(
    run_type: str,
    records: List[Dict],
    checkpoints: List[Checkpoint],
) -> Tuple[Optional[int], str]:
    if not checkpoints:
        return None, "no checkpoints found"

    if run_type == "pretrain":
        return checkpoints[-1][0], "latest checkpoint"

    best_epoch, metric_key = find_best_epoch_from_best_metric(records)
    if best_epoch is None:
        best_epoch, metric_key = find_best_epoch_by_val(records)

    if best_epoch is None:
        return None, "no suitable metric found in log"

    epochs = {epoch for epoch, _ in checkpoints}
    metric_label = metric_key or "best_metric"
    if best_epoch in epochs:
        return best_epoch, f"best by {metric_label}"

    nearest = closest_checkpoint_epoch(best_epoch, checkpoints)
    return nearest, f"best epoch {best_epoch} not saved; keeping closest checkpoint {nearest} instead"


def delete_unwanted(checkpoints: List[Checkpoint], keep_epoch: int, dry_run: bool) -> None:
    for epoch, path in checkpoints:
        if epoch == keep_epoch:
            continue
        if dry_run:
            print(f"[DRY-RUN] Would delete {path}")
        else:
            print(f"[DELETE] Removing {path}")
            path.unlink()


def process_run_dir(run_dir: Path, dry_run: bool) -> None:
    log_path = run_dir / "log.txt"
    if not log_path.exists():
        return

    records = parse_log_lines(log_path)
    if not records:
        print(f"[SKIP] {run_dir} has log.txt but no valid entries")
        return

    run_type = detect_run_type(records)
    checkpoints = collect_checkpoints(run_dir)
    keep_epoch, reason = pick_keep_checkpoint(run_type, records, checkpoints)

    print(
        f"[INFO] {run_dir} | type={run_type} | checkpoints={len(checkpoints)} | reason={reason}"
    )

    if keep_epoch is None:
        return

    delete_unwanted(checkpoints, keep_epoch, dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean up WavesFM checkpoints.")
    parser.add_argument("top_dir", type=Path, help="Root directory to traverse.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. If omitted, a dry run is performed.",
    )
    args = parser.parse_args()

    if not args.top_dir.exists() or not args.top_dir.is_dir():
        raise SystemExit(f"Invalid directory: {args.top_dir}")

    dry_run = not args.apply
    for root, _dirs, _files in os.walk(args.top_dir):
        process_run_dir(Path(root), dry_run=dry_run)


if __name__ == "__main__":
    main()
