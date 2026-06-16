"""Distributed checkpointing for LatentLM — Megatron-Bridge style.

Layout of one checkpoint dir (``<output_dir>/checkpoints/ckpt-NNNNNNN/``):

    common.pt           plain torch tensors + scalars (rank-0)
    *.distcp            TP-sharded weights + optimizer state (one per rank)
    metadata.json       dist_checkpointing manifest
    <output_dir>/checkpoints/latest -> ckpt-NNNNNNN

Why distributed format: the legacy single-file ``torch.save(model.state_dict())``
path corrupts FP8 amax/scale (TE#982, Megatron-LM#1350). On resume, the
saved amax history was numerically inconsistent with the live activations;
``dist_checkpointing.load`` handles this correctly.

This module mirrors the shape of ``bridge/training/checkpointing.py``:

* Explicit ``TorchDistSaveShardedStrategy`` with `thread_count` from config.
* Force ``async_strategy="mcore"`` to dodge the upstream nvrx hang
  (Megatron-LM#3899, mcore commit 704c7ee5a — also documented in Bridge's
  ``checkpointing.py`` WAR comment).
* Model state dict via ``backbone.gpt.sharded_state_dict(prefix="backbone.gpt.")``
  + plain CPU tensors for ``latent_in_proj`` / ``diffusion_head``.
* Optimizer state via ``optimizer.sharded_state_dict(state)`` when available
  (MegatronOptimizer subclasses, including our Bridge-setup_optimizer wrap).
  Falls back to a separate ``optimizer.pt`` torch.save when not supported.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import torch


def _ckpt_root(output_dir: str) -> Path:
    p = Path(output_dir) / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _delete_extra_state(state: dict) -> int:
    """Strip TE FP8 ``_extra_state`` entries. Bridge does the same on load
    (see bridge/training/checkpointing.py: ``delete_extra_state``). FP8 amax
    history isn't safe to round-trip, so we drop it and let the train loop's
    amax-warmup forwards re-derive it. Returns the number of keys removed.
    """
    n = 0
    for k in list(state.keys()):
        if "_extra_state" in k:
            del state[k]
            n += 1
    return n


def _build_model_sd(model) -> dict[str, Any]:
    """Sharded model state dict — Bridge-style.

    GPT trunk via ``backbone.gpt.sharded_state_dict(...)`` (TP-aware
    ``ShardedTensor`` entries). Custom modules (``latent_in_proj``,
    ``diffusion_head``) added as plain CPU tensors — ``dist_checkpointing``
    routes those into ``common.pt`` automatically.
    """
    sd: dict[str, Any] = {}
    sd.update(model.backbone.gpt.sharded_state_dict(prefix="backbone.gpt."))
    for name, t in model.latent_in_proj.state_dict().items():
        sd[f"latent_in_proj.{name}"] = t.detach().cpu()
    for name, t in model.diffusion_head.state_dict().items():
        sd[f"diffusion_head.{name}"] = t.detach().cpu()
    return sd


def save(
    *,
    output_dir: str,
    step: int,
    model,
    optimizer=None,
    extras: dict | None = None,
    keep_last_k: int = 3,
    thread_count: int = 2,
) -> str:
    """Save a distributed checkpoint. All ranks participate (collective)."""
    import torch.distributed as dist
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.strategies.torch import (
        TorchDistSaveShardedStrategy,
    )

    rank = dist.get_rank() if dist.is_initialized() else 0
    ckpt_dir = _ckpt_root(output_dir) / f"ckpt-{step:07d}"

    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    # Sync all CUDA work before the collective. Belt-and-suspenders.
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Build state dict in Bridge's shape.
    state: dict[str, Any] = {
        "checkpoint_version": 3.0,
        "iteration": step,
        "model": _build_model_sd(model),
    }
    if extras:
        state["extras"] = extras

    # Optimizer: prefer sharded_state_dict() (MegatronOptimizer subclasses,
    # including our Bridge-setup_optimizer wrap) — keeps optim inside the same
    # ckpt dir and reloadable across TP. Fall back to a separate torch.save
    # for non-Megatron optimizers (e.g. bare AdamW).
    fallback_opt_path: Path | None = None
    if optimizer is not None:
        if hasattr(optimizer, "sharded_state_dict"):
            try:
                state["optimizer"] = optimizer.sharded_state_dict(state)
            except Exception as e:
                print(f"[ckpt save] optimizer.sharded_state_dict failed ({e}); "
                      f"falling back to torch.save(optimizer.pt)", flush=True)
                fallback_opt_path = ckpt_dir / "optimizer.pt"
        else:
            fallback_opt_path = ckpt_dir / "optimizer.pt"

    # Bridge-style strategy: explicit TorchDistSaveShardedStrategy with
    # parallel writer threads. async_strategy="mcore" because the default
    # "nvrx" path hits a hang/fault inside save_state_dict_async_plan
    # (Megatron-LM#3899). Same WAR Bridge uses.
    #
    # IMPORTANT: async_sharded_save=True is required for `async_strategy` to
    # be respected — when False, `strategy.save()` hardcodes the nvrx path
    # via `HAVE_NVRX`. We then execute the returned AsyncRequest synchronously
    # ourselves (matches Bridge's behaviour when `ckpt_cfg.async_save=False`
    # in spirit — the actual write blocks until done).
    save_strategy = TorchDistSaveShardedStrategy(
        "torch_dist", 1, thread_count=thread_count,
    )
    async_request = dist_checkpointing.save(
        state,
        str(ckpt_dir),
        sharded_strategy=save_strategy,
        async_sharded_save=True,
        async_strategy="mcore",
    )
    if async_request is not None:
        async_request.execute_sync()

    # Optimizer fallback (only when sharded_state_dict isn't available).
    if rank == 0 and fallback_opt_path is not None and optimizer is not None:
        tmp = fallback_opt_path.with_suffix(".pt.tmp")
        torch.save(optimizer.state_dict(), tmp)
        tmp.replace(fallback_opt_path)

    if rank == 0:
        latest = ckpt_dir.parent / "latest"
        tmp_link = latest.with_suffix(".tmp")
        if tmp_link.is_symlink() or tmp_link.exists():
            tmp_link.unlink()
        tmp_link.symlink_to(ckpt_dir.name)
        os.replace(tmp_link, latest)
        if keep_last_k > 0:
            all_ckpts = sorted(p for p in ckpt_dir.parent.glob("ckpt-*") if p.is_dir())
            for old in all_ckpts[:-keep_last_k]:
                shutil.rmtree(old, ignore_errors=True)

    if dist.is_initialized():
        dist.barrier()
    return str(ckpt_dir)


def load(
    *,
    path: str,
    model,
    optimizer=None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load a distributed checkpoint. Returns ``{step, extras}``.

    ``path`` may be either a concrete ``ckpt-NNNNNNN`` directory or the
    parent ``checkpoints/`` directory (in which case ``latest`` is followed).
    """
    from megatron.core import dist_checkpointing

    p = Path(path)
    if p.is_dir() and (p / "latest").is_symlink():
        p = (p / "latest").resolve()

    # Build template that matches the saved structure.
    template: dict[str, Any] = {
        "checkpoint_version": 3.0,
        "model": _build_model_sd(model),
    }
    if optimizer is not None and hasattr(optimizer, "sharded_state_dict"):
        try:
            template["optimizer"] = optimizer.sharded_state_dict(template)
        except Exception:
            pass

    loaded = dist_checkpointing.load(template, str(p))

    # Model
    n_extra = _delete_extra_state(loaded["model"])
    if n_extra:
        print(f"[ckpt load] stripped {n_extra} _extra_state keys "
              f"(force FP8 re-derivation via amax warmup)", flush=True)

    missing, unexpected = model.load_state_dict(loaded["model"], strict=False)
    print(f"[ckpt load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print(f"  missing[:8]: {missing[:8]}", flush=True)
    if unexpected:
        print(f"  unexpected[:8]: {unexpected[:8]}", flush=True)

    # Optimizer: try sharded payload first, then separate optimizer.pt fallback.
    if optimizer is not None:
        if "optimizer" in loaded:
            try:
                optimizer.load_state_dict(loaded["optimizer"])
                print(f"[ckpt load] loaded sharded optimizer state from {p}",
                      flush=True)
            except Exception as e:
                import warnings
                warnings.warn(f"sharded optimizer state not loaded: {e}")
        else:
            opt_path = p / "optimizer.pt"
            if opt_path.is_file():
                try:
                    opt_state = torch.load(
                        opt_path, map_location=map_location, weights_only=False,
                    )
                    optimizer.load_state_dict(opt_state)
                    print(f"[ckpt load] loaded optimizer state from {opt_path}",
                          flush=True)
                except Exception as e:
                    import warnings
                    warnings.warn(f"optimizer state not loaded: {e}")

    return {
        "step": int(loaded.get("iteration", 0)),
        "extras": loaded.get("extras", {}),
    }
