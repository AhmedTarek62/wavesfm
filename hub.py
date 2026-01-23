from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple


def _load_config(
    repo_id: str,
    *,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Download config.json from the HF Hub (counts as a tracked download)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("huggingface_hub is required to download config.json") from exc

    config_path = hf_hub_download(
        repo_id,
        "config.json",
        revision=revision,
        token=token,
        cache_dir=cache_dir,
    )
    return json.loads(Path(config_path).read_text())


def _build_from_config(
    config: Dict[str, Any],
    *,
    task_info: Optional[Dict[str, Any]] = None,
    model_args: Optional[Dict[str, Any]] = None,
):
    try:
        from main_finetune import build_model
    except ImportError as exc:
        raise ImportError("wavesfm main_finetune.build_model not available") from exc

    task_info = task_info or config.get("task_info")
    model_args = model_args or config.get("model_args")
    if not task_info or not model_args:
        raise ValueError(
            "Missing task_info/model_args. Add them to config.json or pass explicitly."
        )

    task_ns = SimpleNamespace(**task_info)
    args_ns = SimpleNamespace(**model_args)
    return build_model(args_ns, task_ns)


def download_pretrained(
    repo_id: str,
    *,
    filename: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
) -> str:
    """Download config.json (counts) and then the requested weights file."""
    _load_config(repo_id, revision=revision, token=token, cache_dir=cache_dir)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("huggingface_hub is required to download weights") from exc

    return hf_hub_download(
        repo_id,
        filename,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
    )


def from_pretrained(
    repo_id: str,
    *,
    filename: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
    map_location: str = "cpu",
    strict: bool = True,
    task_info: Optional[Dict[str, Any]] = None,
    model_args: Optional[Dict[str, Any]] = None,
) -> Tuple["torch.nn.Module", str]:
    """Download config + weights from HF Hub and load into a WavesFM model.

    Returns (model, checkpoint_path). The config.json download ensures HF download
    attribution is tracked before fetching the weights file.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required to load WavesFM checkpoints") from exc

    config = _load_config(repo_id, revision=revision, token=token, cache_dir=cache_dir)
    if task_info is None and model_args is None:
        task_info = config.get("task_info")
        model_args = config.get("model_args")
    if task_info is None or model_args is None:
        raise ValueError(
            "Missing task_info/model_args. Pass them explicitly or add to config.json."
        )
    model = _build_from_config(config, task_info=task_info, model_args=model_args)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("huggingface_hub is required to download weights") from exc

    ckpt_path = hf_hub_download(
        repo_id,
        filename,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
    )
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=strict)
    return model, ckpt_path
