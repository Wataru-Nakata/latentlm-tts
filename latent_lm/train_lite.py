"""Lightweight, no-container training entrypoint (plain PyTorch).

Trains LatentLM TTS with the plain-PyTorch `TorchBackbone` (RoPE) — no NeMo /
Megatron / transformer-engine, just `pip install`. Single- or multi-GPU via
torchrun (env `RANK`/`WORLD_SIZE`/`LOCAL_RANK`):

    torchrun --nproc_per_node=1 -m latent_lm.train_lite \
        --config configs/tts_lite.yaml --train-steps 100

Use this to train small models or develop without the container. For large
multi-node FP8/TP runs use the Megatron path in `latent_lm.train`.

Reused container-free pieces from `latent_lm.train`: `build_loss`,
`_pack_batch_from_audio` (streaming VibeVoice encode + pack), `_pack_batch_from_cache`.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch


def _cosine_lr(step: int, *, warmup: int, total: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    if step >= total:
        return min_lr
    prog = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * prog))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-steps", type=int, default=None,
                    help="Override optim.max_steps (number of optimizer steps).")
    ap.add_argument("--resume-from", default=None, help="Path to a *.pt checkpoint.")
    args = ap.parse_args()

    from omegaconf import OmegaConf
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    from . import checkpoint_lite as ckpt
    from .data.collate import CollateConfig
    from .data.emilia import EmiliaConfig
    from .data.pipeline import PipelineConfig, make_dataloader
    from .data.text_tokenizer import TextTokenizer, TextTokenizerConfig
    from .models.latent_lm import LatentLM, LatentLMConfig
    from .models.tokenizer import ENCODER_STRIDE, VibeVoiceTokenizer
    from .models.torch_backbone import TorchBackbone, TorchBackboneConfig
    from .train import build_loss, _pack_batch_from_audio

    cfg = OmegaConf.load(args.config)
    out_dir = cfg.paths.output_dir
    steps = int(args.train_steps if args.train_steps is not None else cfg.optim.max_steps)

    # --- distributed (torchrun sets these; defaults = single process) ---
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ddp = world > 1
    if ddp:
        dist.init_process_group("nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    def log(m: str) -> None:
        if rank == 0:
            print(m, flush=True)

    dt = torch.bfloat16 if cfg.precision.params_dtype == "bfloat16" else torch.float32

    # --- tokenizer / vocab + specials ---
    tok = TextTokenizer(TextTokenizerConfig(model_id=cfg.tokenizer.text, tp=1))

    # --- model: plain-torch backbone (lite) ---
    m = cfg.model.lite
    packed = int(cfg.runtime.get("packed_total_length", 0) or 0)
    backbone = TorchBackbone(TorchBackboneConfig(
        hidden_dim=int(m.hidden_dim), n_layers=int(m.n_layers),
        n_heads=int(m.n_heads), ffn_mult=int(m.get("ffn_mult", 4)),
        max_seq_len=max(packed, int(cfg.data.max_text_tokens) + 4 * int(cfg.data.max_latent_frames)),
    ))
    model = LatentLM(
        LatentLMConfig(
            vocab_size=tok.vocab_size, hidden_dim=int(m.hidden_dim),
            latent_dim=int(cfg.model.latent_dim),
            diff_head_layers=int(cfg.model.diff_head_layers),
            diff_head_ffn_mult=int(cfg.model.diff_head_ffn_mult),
        ),
        backbone=backbone,
    ).to(device=device, dtype=torch.float32)
    inner = model
    log(f"[train_lite] params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M  "
        f"vocab={tok.vocab_size}  device={device}")

    loss_fn = build_loss(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.optim.lr),
        betas=tuple(cfg.optim.get("betas", [0.9, 0.95])),
        weight_decay=float(cfg.optim.get("weight_decay", 0.1)))
    base_lr = float(cfg.optim.lr)
    min_lr = base_lr * 0.1
    warmup = int(cfg.optim.get("warmup_steps", 0))
    grad_clip = float(cfg.optim.get("grad_clip", 1.0))
    grad_accum = int(cfg.runtime.get("gradient_accumulation_steps", 1))

    start_step = 0
    if args.resume_from:
        start_step = ckpt.load(args.resume_from, inner, optimizer, map_location=device)
        log(f"[train_lite] resumed from {args.resume_from} at step={start_step}")

    if ddp:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    # --- data ---
    cache_dir = cfg.data.get("cache_dir", None)
    use_cache = bool(cache_dir)
    pack_cfg = CollateConfig(max_text_tokens=int(cfg.data.max_text_tokens),
                             max_latent_frames=int(cfg.data.max_latent_frames),
                             latent_dim=int(cfg.model.latent_dim))
    emilia_cfg = EmiliaConfig(
        dataset_id=cfg.data.dataset_id, config_name=cfg.data.get("config_name", None),
        split=cfg.data.get("split", "train"), languages=tuple(cfg.data.get("languages", [])),
        min_audio_seconds=float(cfg.data.get("min_audio_seconds", 1.0)),
        max_audio_seconds=float(cfg.data.get("max_audio_seconds", 20.0)),
        shuffle_buffer=int(cfg.data.get("shuffle_buffer", 1000)))
    pipe_cfg = PipelineConfig(
        emilia=emilia_cfg, max_text_tokens=int(cfg.data.max_text_tokens),
        audio_encoder_stride=ENCODER_STRIDE, num_workers=int(cfg.data.get("num_workers", 2)),
        batch_size=int(cfg.runtime.micro_batch_size), cache_dir=cache_dir,
        specials=tok.specials if use_cache else None,
        pack_cfg=pack_cfg if use_cache else None,
        packed_total_length=packed if use_cache else 0)
    dl = make_dataloader(pipe_cfg, tok, rank=rank, world_size=world)

    vv = None if use_cache else VibeVoiceTokenizer().to(device)

    log_every = int(cfg.runtime.get("log_every_n_steps", 20))
    ckpt_every = int(cfg.runtime.get("checkpoint_every_n_steps", 0))
    keep_last_k = int(cfg.runtime.get("keep_last_k_checkpoints", 3))

    log(f"[train_lite] training to {steps} steps  (cache={'on' if use_cache else 'streaming'})")
    model.train()
    step = start_step
    t0 = time.time()
    accum = 0
    optimizer.zero_grad(set_to_none=True)
    for raw in dl:
        if use_cache:
            batch = {k: (v.to(device, non_blocking=True) if hasattr(v, "to") else v)
                     for k, v in raw.items()}
        else:
            batch = _pack_batch_from_audio(
                text_ids_list=raw["text_ids"], audio=raw["audio"].to(device),
                audio_lens=raw["audio_lens"], vv=vv, specials=tok.specials,
                pack_cfg=pack_cfg, encoder_stride=ENCODER_STRIDE, device=device,
                packed_total_length=packed)

        with torch.autocast(device_type=device.type, dtype=dt, enabled=(device.type == "cuda")):
            out = model(
                input_ids=batch["input_ids"], input_latents=batch["input_latents"],
                is_audio_input=batch["is_audio_input"],
                attention_mask=batch.get("attention_mask"),
                cu_seqlens=batch.get("cu_seqlens"))
            # Keep the loss (incl. the fp32 diffusion-head matmuls) inside autocast
            # so the head's fp32 weights and the bf16 activations are reconciled by
            # autocast instead of raising a dtype mismatch.
            losses = loss_fn(
                text_logits=out["text_logits"].float(), text_targets=batch["text_targets"],
                audio_targets=batch["audio_targets"], audio_mask=batch["audio_mask"],
                hidden_states=out["hidden"], diff_head=inner.diffusion_head)
        (losses["loss"] / grad_accum).backward()
        accum += 1
        if accum < grad_accum:
            continue
        accum = 0

        lr = _cosine_lr(step, warmup=warmup, total=steps, base_lr=base_lr, min_lr=min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % log_every == 0:
            ms = (time.time() - t0) * 1000 / max(step - start_step, 1)
            log(f"[train_lite] step={step:>6}  loss={float(losses['loss']):.4f}  "
                f"lm={float(losses['lm_loss']):.4f}  diff={float(losses['diff_loss']):.4f}  "
                f"lr={lr:.2e}  ms/step={ms:.0f}")
        if ckpt_every > 0 and step % ckpt_every == 0 and rank == 0:
            p = ckpt.save(out_dir, step, inner, optimizer, keep_last_k=keep_last_k)
            log(f"[train_lite] checkpoint -> {p}")
        if step >= steps:
            break

    if rank == 0:
        p = ckpt.save(out_dir, step, inner, optimizer, keep_last_k=keep_last_k)
        log(f"[train_lite] final checkpoint -> {p}")
    if ddp:
        dist.destroy_process_group()
    log("[train_lite] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
