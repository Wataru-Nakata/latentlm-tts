"""Plain-PyTorch checkpointing for the no-container (`train_lite`) path.

A single `torch.save` file per checkpoint (model + optimizer + step), with a
`latest.pt` symlink and last-k pruning. The Megatron distributed-checkpoint
format lives in `checkpoint.py` (used by the container `train.py`); the two are
not interchangeable.
"""

from __future__ import annotations

import glob
import os

import torch


def save(output_dir: str, step: int, model, optimizer, keep_last_k: int = 3) -> str:
    """Save `{step, model, optim}` to `<output_dir>/checkpoints/step_XXXXXXXX.pt`.

    Pass the *unwrapped* model (i.e. `model.module` under DDP).
    """
    d = os.path.join(output_dir, "checkpoints")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"step_{step:08d}.pt")
    torch.save(
        {"step": int(step),
         "model": model.state_dict(),
         "optim": optimizer.state_dict()},
        path,
    )
    latest = os.path.join(d, "latest.pt")
    if os.path.islink(latest) or os.path.exists(latest):
        os.remove(latest)
    os.symlink(os.path.basename(path), latest)
    # last-k pruning
    for old in sorted(glob.glob(os.path.join(d, "step_*.pt")))[:-keep_last_k]:
        os.remove(old)
    return path


def load(path: str, model, optimizer=None, map_location="cpu") -> int:
    """Load a checkpoint into `model` (+ optionally `optimizer`). Returns step.

    `path` may point at the `latest.pt` symlink or a concrete `step_*.pt`.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optim" in payload:
        optimizer.load_state_dict(payload["optim"])
    return int(payload.get("step", 0))
