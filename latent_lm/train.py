"""Training entrypoint — hand-rolled loop with FP8 + 3D parallelism.

Step shape:
    1. Fetch {text_ids, audio, audio_lens} from the Emilia streaming DataLoader.
    2. VibeVoice encode audio on the training GPU (frozen σ-VAE).
    3. Pack each example (text + latents) and collate into a batched sequence.
    4. Forward under `fp8_autocast` (hybrid E4M3/E5M2 DelayedScaling).
    5. Combined LM-CE + DDPM loss, backward, optimizer step.

Flags:
    --dry-run            Build model, print param count, exit. No data.
    --train-steps N      Run N training steps on Emilia, log loss. Requires
                         a VibeVoice download at first launch.

Invoked inside the NeMo 26.04 SIF on a compute node:
    torchrun --standalone --nproc_per_node=4 -m latent_lm.train \
        --config configs/tts_smoke_train.yaml --train-steps 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch


def _load_config(path: str):
    from omegaconf import OmegaConf

    return OmegaConf.load(path)


def _is_main_rank() -> bool:
    import torch.distributed as dist

    return not dist.is_initialized() or dist.get_rank() == 0


def _log(msg: str) -> None:
    if _is_main_rank():
        print(msg, flush=True)


def build_model(cfg, *, vocab_size: int):
    """Build LatentLM via Megatron-Bridge (HF model id → GPTModel + HF weight
    load). Architecture comes from the HF config; YAML `model.*` keys
    (hidden_dim/n_layers/etc.) are no longer consulted — only `latent_dim`,
    `diff_head_layers`, `diff_head_ffn_mult`, and the HF model id matter.
    """
    from .models.bridge_backbone import build_bridge_backbone
    from .models.latent_lm import LatentLM, LatentLMConfig

    p = cfg.parallelism
    dt = torch.bfloat16 if cfg.precision.params_dtype == "bfloat16" else torch.float32
    recompute = cfg.get("recompute", None) if hasattr(cfg, "get") else None
    overrides = {
        "fp8": cfg.precision.fp8,
        "fp8_amax_history_len": cfg.precision.fp8_amax_history_len,
        "fp8_amax_compute_algo": cfg.precision.fp8_amax_compute_algo,
        "params_dtype": dt,
        "bf16": (dt is torch.bfloat16),
        "fp16": (dt is torch.float16),
        # We use torch DDP, not Megatron DDP — apex-style grad-accum fusion
        # writes into a Megatron-DDP buffer that doesn't exist on our path.
        "gradient_accumulation_fusion": False,
    }
    # vocab_size: keep Bridge's HF-native value (else weight load shape
    # mismatch). TextTokenizer places our specials inside the HF-native
    # range so this is safe; tok.vocab_size (padded) just adds unused rows.
    _ = vocab_size  # informational; not propagated to provider
    if recompute is not None:
        overrides["recompute_granularity"] = recompute.granularity
        overrides["recompute_method"] = recompute.method
        overrides["recompute_num_layers"] = recompute.num_layers

    res = build_bridge_backbone(
        cfg.model.hf_model_id,
        load_weights=bool(cfg.model.get("load_hf_weights", True)),
        tp=p.tensor_model_parallel_size,
        pp=p.pipeline_model_parallel_size,
        cp=p.context_parallel_size,
        provider_overrides=overrides,
    )
    backbone = res.backbone
    # HF config wins for hidden / vocab dims; YAML values are informational.
    cfg.model.hidden_dim = res.hidden_dim
    model = LatentLM(
        cfg=LatentLMConfig(
            vocab_size=res.vocab_size,
            hidden_dim=res.hidden_dim,
            latent_dim=cfg.model.latent_dim,
            diff_head_layers=cfg.model.diff_head_layers,
            diff_head_ffn_mult=cfg.model.diff_head_ffn_mult,
            tie_embeddings=False,
        ),
        backbone=backbone,
        token_embed=backbone.word_embeddings,
        lm_head=backbone.output_layer,
        # Pass the GPT trunk's TransformerConfig so LatentLM (via MegatronModule
        # base) exposes self.config.{num_attention_heads,num_query_groups,
        # kv_channels} — required by Bridge's setup_optimizer + Muon QKV split.
        transformer_config=backbone.gpt.config,
    )
    return model, backbone.gpt.config


def _setup_optimizer_via_bridge(cfg, inner_model, *, train_steps: int):
    """Build optimizer + LR scheduler through Bridge's setup_optimizer.

    Returns (megatron_optimizer, scheduler, samples_per_opt_step).

    `samples_per_opt_step` is the increment we pass to `scheduler.step()` each
    opt step — Bridge's scheduler counts in samples (see
    bridge/training/train.py:809).

    Param partitioning happens inside `get_megatron_muon_optimizer`: 2D params
    that aren't flagged `is_embedding_or_output_parameter` go to Muon; the rest
    (biases, norms, embeddings, lm_head) go to AdamW. That matches what our
    old `_split_params_for_muon` heuristic did via shape, since Megatron's
    `LanguageModule` already tags embedding/output weights with that flag.
    """
    from megatron.core.optimizer import OptimizerConfig
    from megatron.bridge.training.config import SchedulerConfig
    from megatron.bridge.training.optim import setup_optimizer
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core import parallel_state as _ps

    use_muon = bool(cfg.optim.get("use_muon", False)
                    if hasattr(cfg.optim, "get") else False)
    optimizer_name = "muon" if use_muon else "adam"

    dt = torch.bfloat16 if cfg.precision.params_dtype == "bfloat16" else torch.float32
    grad_clip = float(cfg.optim.get("grad_clip", 1.0)
                      if hasattr(cfg.optim, "get") else 1.0)
    # Muon uses muon_lr for matrix params; the wrapped scalar AdamW uses
    # the same `lr` from OptimizerConfig (Megatron's pattern — they share lr
    # but mup overrides can scale them differently).
    base_lr = float(cfg.optim.get("muon_lr", cfg.optim.lr)
                    if use_muon else cfg.optim.lr)
    weight_decay = float(cfg.optim.weight_decay)
    # Distributed optimizer shards Adam state across DP ranks (~ DP× memory
    # reduction). Requires Megatron-DDP wrap (handled in train()) — torch DDP
    # doesn't expose a grad buffer the dist optim can hook into.
    use_distributed_optimizer = bool(
        cfg.optim.get("use_distributed_optimizer", False)
        if hasattr(cfg.optim, "get") else False)

    opt_cfg = OptimizerConfig(
        optimizer=optimizer_name,
        lr=base_lr,
        min_lr=0.0,
        weight_decay=weight_decay,
        adam_beta1=float(cfg.optim.betas[0]),
        adam_beta2=float(cfg.optim.betas[1]),
        bf16=(dt is torch.bfloat16),
        fp16=(dt is torch.float16),
        params_dtype=dt,
        clip_grad=grad_clip,
        muon_momentum=float(cfg.optim.get("muon_momentum", 0.95)
                            if hasattr(cfg.optim, "get") else 0.95),
        muon_use_nesterov=True,
        muon_num_ns_steps=5,
        muon_scalar_optimizer="adam",
        use_distributed_optimizer=use_distributed_optimizer,
        use_precision_aware_optimizer=False,
    )

    # Convert opt-step counts to samples for the scheduler. Bridge's
    # `scheduler.step(increment=samples_per_opt_step)` matches global_batch.
    micro = int(cfg.runtime.micro_batch_size)
    accum = int(cfg.runtime.get("gradient_accumulation_steps", 1)
                if hasattr(cfg.runtime, "get") else 1)
    try:
        dp = max(1, _ps.get_data_parallel_world_size())
    except Exception:
        dp = 1
    samples_per_opt_step = micro * accum * dp

    warmup_iters = int(cfg.optim.get("warmup_steps", 0)
                       if hasattr(cfg.optim, "get") else 0)
    decay_iters = int(cfg.optim.get("max_steps", train_steps)
                      if hasattr(cfg.optim, "get") else train_steps)
    sched_cfg = SchedulerConfig(
        lr_decay_style=str(cfg.optim.get("lr_schedule", "cosine")
                           if hasattr(cfg.optim, "get") else "cosine"),
        lr_warmup_init=0.0,
        # Constant weight decay throughout — start_wd == end_wd mirrors what
        # our old LambdaLR-only setup did (no wd schedule).
        start_weight_decay=weight_decay,
        end_weight_decay=weight_decay,
        weight_decay_incr_style="constant",
    )
    sched_cfg.lr_warmup_steps = warmup_iters * samples_per_opt_step
    sched_cfg.lr_decay_steps = max(decay_iters * samples_per_opt_step, 1)
    sched_cfg.wd_incr_steps = max(decay_iters * samples_per_opt_step, 1)

    # Build the full ProcessGroupCollection from Megatron's parallel_state.
    # The classmethod pulls all required groups (mp, ep, expt_dp, dist_opt,
    # etc.) — the AdamW path requires many of them as explicit (None for our
    # non-MoE/non-distopt setup), whereas the Muon path only checked a few.
    pg = ProcessGroupCollection.use_mpu_process_groups()
    # Megatron's `get_megatron_optimizer` reads `model_chunks[0].ddp_config`
    # to decide between the FSDP and dense paths. Plain torch DDP doesn't
    # set this attr, so attach a default DDPConfig (use_megatron_fsdp=False)
    # to take the dense path. Idempotent — only set on the first call.
    if not hasattr(inner_model, "ddp_config"):
        from megatron.core.distributed.distributed_data_parallel_config import (
            DistributedDataParallelConfig,
        )
        inner_model.ddp_config = DistributedDataParallelConfig()
    # Pre-wrap the model in a list: bridge/training/optim.py forwards `model`
    # (not its local list-normalised `model_chunks`) to
    # `get_megatron_muon_optimizer`, which iterates expecting a list.
    optimizer, scheduler = setup_optimizer(
        optimizer_config=opt_cfg, scheduler_config=sched_cfg,
        model=[inner_model], pg_collection=pg,
    )
    return optimizer, scheduler, samples_per_opt_step


def build_loss(cfg):
    from .losses import DDPMLoss, DDPMScheduleConfig, ModalityLoss

    sched = DDPMScheduleConfig(
        num_timesteps=cfg.loss.ddpm.num_timesteps,
        schedule=cfg.loss.ddpm.schedule,
        timesteps_per_forward=cfg.loss.ddpm.timesteps_per_forward,
        prediction_type=cfg.loss.ddpm.get("prediction_type", "v_prediction"),
    )
    ce_chunk_size = int(cfg.loss.get("ce_chunk_size", 4096)
                        if hasattr(cfg.loss, "get") else 4096)
    return ModalityLoss(DDPMLoss(sched), alpha=cfg.loss.alpha_diff,
                        ce_chunk_size=ce_chunk_size)


# Note: we no longer gather the LM head output — `losses._lm_cross_entropy`
# uses Megatron's `vocab_parallel_cross_entropy` when logits are TP-sharded.


def _finalise_batch(packed_examples, *, specials, packed_total_length, device):
    """Collate + move-to-GPU. Uses the packed collator when `packed_total_length`
    is > 0, otherwise the padded collator. Single place to switch modes."""
    from .data.collate import collate_batch, collate_packed

    if packed_total_length and packed_total_length > 0:
        batch = collate_packed(packed_examples, specials=specials,
                               total_length=packed_total_length)
    else:
        batch = collate_batch(packed_examples, specials=specials)
    for k, v in batch.items():
        if hasattr(v, "to"):
            batch[k] = v.to(device, non_blocking=True)
    return batch


def _pack_batch_from_audio(*, text_ids_list, audio, audio_lens, vv, specials,
                           pack_cfg, encoder_stride, device, packed_total_length=0):
    """Streaming mode: encode audio with VibeVoice, then collate."""
    import torch

    from .data.collate import pack_example
    from .models.tokenizer import LATENT_SCALE

    with torch.no_grad():
        latents_batch = vv.encode(audio.to(device))   # (B, T_lat, 64)
    lat_lens = (audio_lens + encoder_stride - 1) // encoder_stride
    packed_examples = [
        pack_example(
            text_ids=text_ids_list[i],
            # Scale to ~unit variance — DDPM ε-prediction needs x0 ~ N(0,1).
            # VibeVoice latents have native std≈5.12; LATENT_SCALE ≈ 0.195.
            latents=(latents_batch[i, : int(lat_lens[i].item())].float().cpu()
                     * LATENT_SCALE),
            specials=specials,
            cfg=pack_cfg,
        )
        for i in range(audio.shape[0])
    ]
    return _finalise_batch(packed_examples, specials=specials,
                           packed_total_length=packed_total_length, device=device)


def _pack_batch_from_cache(*, text_ids_list, latents_list, specials, pack_cfg,
                           device, packed_total_length=0):
    """Cached mode: latents are already pre-encoded; just collate."""
    from .data.collate import pack_example

    packed_examples = [
        pack_example(
            text_ids=text_ids_list[i],
            latents=latents_list[i].float(),
            specials=specials,
            cfg=pack_cfg,
        )
        for i in range(len(text_ids_list))
    ]
    return _finalise_batch(packed_examples, specials=specials,
                           packed_total_length=packed_total_length, device=device)


def train(args, cfg) -> int:
    import torch
    import torch.distributed as dist

    from .data.pipeline import PipelineConfig, make_dataloader
    from .data.text_tokenizer import TextTokenizer, TextTokenizerConfig
    from .data.collate import CollateConfig
    from .data.emilia import EmiliaConfig
    from .models.bridge_backbone import fp8_autocast_context
    from .models.tokenizer import ENCODER_STRIDE, VibeVoiceTokenizer

    p = cfg.parallelism
    tok = TextTokenizer(TextTokenizerConfig(
        model_id=cfg.tokenizer.text,
        tp=p.tensor_model_parallel_size,
    ))
    _log(f"[train] text tokenizer: {cfg.tokenizer.text}  base={tok.base_vocab}  "
         f"padded_vocab={tok.vocab_size}")

    inner_model, tfm_cfg = build_model(cfg, vocab_size=tok.vocab_size)
    # Tokenizer vocab and HF vocab must agree — both pad to multiples of 128.
    # If they differ, the LM head and tokenizer indices won't line up.
    bridge_vocab = int(getattr(inner_model.backbone, "vocab_size", 0))
    if tok.vocab_size != bridge_vocab:
        _log(f"[train] WARNING: tokenizer vocab={tok.vocab_size} != "
             f"bridge vocab={bridge_vocab}. Set tokenizer.text = "
             f"{cfg.model.hf_model_id} so they match.")
    inner_model = inner_model.to("cuda")
    # Megatron modules are already in params_dtype (bf16); our plain-PyTorch
    # extras (latent_in_proj, diffusion_head) default to fp32. Cast them so
    # their weights match the backbone's activation dtype.
    dt = torch.bfloat16 if cfg.precision.params_dtype == "bfloat16" else torch.float32
    inner_model.latent_in_proj.to(dtype=dt)
    inner_model.diffusion_head.to(dtype=dt)
    _log(f"[train] model built. params/rank: {sum(p.numel() for p in inner_model.parameters())/1e6:.1f}M "
         f"  fp8={tfm_cfg.fp8}  tp={tfm_cfg.tensor_model_parallel_size}")

    if args.dry_run:
        return 0

    # Wrap with the DDP that matches our distributed-optimizer choice. Must
    # happen AFTER all parameters live on GPU and have their final dtype
    # (DDP buckets are built once at construction).
    from megatron.core import parallel_state
    from megatron.core.distributed.distributed_data_parallel_config import (
        DistributedDataParallelConfig,
    )
    use_distrib_opt = bool(cfg.optim.get("use_distributed_optimizer", False)
                           if hasattr(cfg.optim, "get") else False)
    if dist.is_initialized() and dist.get_world_size() > 1 and use_distrib_opt:
        # Megatron DDP — exposes a grad buffer that DistributedOptimizer can
        # shard across DP ranks. ~DP× reduction in optimizer-state memory.
        from megatron.core.distributed import DistributedDataParallel as McoreDDP
        from megatron.core.process_groups_config import ProcessGroupCollection

        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            overlap_grad_reduce=False,     # disable bucketing — coalesced collective
                                           # was failing on the bucket-group PG even
                                           # though all parallel_state sub-groups
                                           # supported it in isolation (verified
                                           # via scripts/check_nccl_coalesced_mp.py).
            overlap_param_gather=False,
            bucket_size=None,
            check_for_nan_in_grad=False,
        )
        inner_model.ddp_config = ddp_config
        # Pass an explicit pg_collection. Without it McoreDDP falls back to
        # `setup_process_groups_for_ddp(None, ...)` which builds the
        # intra_distributed_optimizer_instance_group from scratch — and that
        # PG ended up without the coalesced NCCL ops Megatron's param-sync
        # needs (we saw "Backend nccl does not support
        # allgather_into_tensor_coalesced"). use_mpu_process_groups pulls all
        # the right NCCL-backed sub-groups from parallel_state.
        pg = ProcessGroupCollection.use_mpu_process_groups()
        model = McoreDDP(
            config=tfm_cfg,
            ddp_config=ddp_config,
            module=inner_model,
            pg_collection=pg,
        )
        _log(f"[train] wrapped model in Megatron DDP  use_distrib_opt=True "
             f"dp_world={parallel_state.get_data_parallel_world_size()}")
    elif dist.is_initialized() and dist.get_world_size() > 1:
        from torch.nn.parallel import DistributedDataParallel as TorchDDP

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dp_group = parallel_state.get_data_parallel_group()
        model = TorchDDP(
            inner_model,
            device_ids=[local_rank],
            output_device=local_rank,
            process_group=dp_group,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        _log(f"[train] wrapped model in torch DDP  dp_world={dist.get_world_size(group=dp_group)}")
    else:
        model = inner_model

    cache_dir = cfg.data.get("cache_dir", None) if hasattr(cfg.data, "get") else None
    use_cache = bool(cache_dir)
    vv = None
    if not use_cache:
        vv = VibeVoiceTokenizer().to("cuda")
        _log("[train] VibeVoice tokenizer loaded")
    else:
        _log(f"[train] cached mode — reading pre-encoded latents from {cache_dir}")

    loss_fn = build_loss(cfg).to("cuda")

    # Optimizer + LR scheduler via Megatron-Bridge's setup_optimizer.
    # Bridge wraps TensorParallelMuon (or plain AdamW) in
    # Float16OptimizerWithFloat16Params, which keeps fp32 master copies of bf16
    # params and copies bf16 grads → fp32 main_grads before the inner step.
    # That removes the manual `p.grad.float()` cast Newton-Schulz needs, plus
    # the explicit `clip_grad_norm_` call (the wrap clips via config.clip_grad)
    # and the Muon-state fp32 cast on resume (states are fp32 by construction).
    # With distributed optimizer, the optimizer needs the Megatron-DDP
    # wrapper (it owns the grad buffer the optim shards). Otherwise we can
    # use either; pass inner_model to keep the existing behaviour for the
    # torch-DDP path.
    optim_target = model if use_distrib_opt else inner_model
    optimizer, scheduler, samples_per_opt_step = _setup_optimizer_via_bridge(
        cfg, optim_target, train_steps=int(args.train_steps),
    )
    use_muon = "muon" in optimizer.config.optimizer

    # Resume from checkpoint if asked. Restores model + optimizer + step counter.
    # Always load into the inner module (DDP wrapper has no own state).
    start_step = 0
    if args.resume_from:
        from . import checkpoint as ckpt_mod

        payload = ckpt_mod.load(path=args.resume_from, model=inner_model, optimizer=optimizer,
                                map_location=f"cuda:{torch.cuda.current_device()}")
        start_step = int(payload.get("step", 0))
        _log(f"[train] resumed from {args.resume_from} at step={start_step}")
        # Release the large, transient checkpoint-load allocations so they don't
        # leave the CUDA allocator fragmented (otherwise the first big training
        # alloc can't find a contiguous block → OOM a few hundred steps in, even
        # with tens of GB "reserved but unallocated"). Pair with
        # PYTORCH_ALLOC_CONF=expandable_segments:True in the launcher.
        del payload
        import gc as _gc
        _gc.collect()
        torch.cuda.empty_cache()
        # Fast-forward the scheduler to the resumed step (samples-based).
        if start_step > 0:
            scheduler.step(increment=start_step * samples_per_opt_step)

    # TensorBoard writer — rank-0 only, no-op if tensorboard isn't importable.
    tb_writer = None
    output_dir = str(cfg.paths.output_dir) if hasattr(cfg, "paths") else "outputs"
    if _is_main_rank():
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb"))
        except Exception as e:
            _log(f"[train] tensorboard not available: {e}")

    # wandb — rank-0 only, enabled when `runtime.wandb.project` is set.
    # Auth via WANDB_API_KEY env var (passed through from PBS); falls back to
    # offline mode if the API key is missing so training never blocks on it.
    wandb_run = None
    wb_cfg = (cfg.runtime.get("wandb", None) if hasattr(cfg.runtime, "get") else None)
    if _is_main_rank() and wb_cfg and wb_cfg.get("project"):
        try:
            import wandb as _wandb
            from omegaconf import OmegaConf

            mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
            wandb_run = _wandb.init(
                project=wb_cfg.get("project"),
                entity=wb_cfg.get("entity"),
                name=wb_cfg.get("name") or os.path.basename(output_dir.rstrip("/")),
                config=OmegaConf.to_container(cfg, resolve=True),
                dir=output_dir,
                mode=mode,
                resume="allow",
            )
            _log(f"[train] wandb {mode} → {wandb_run.url or wandb_run.dir}")
        except Exception as e:
            _log(f"[train] wandb init failed: {e}")

    pack_cfg = CollateConfig(
        max_text_tokens=cfg.data.max_text_tokens,
        max_latent_frames=cfg.data.max_latent_frames,
        latent_dim=cfg.model.latent_dim,
    )
    packed_total_length = int(cfg.runtime.get("packed_total_length", 0)
                              if hasattr(cfg.runtime, "get") else 0)

    eval_holdout_frac = float(cfg.data.get("eval_holdout_frac", 0.01)
                              if hasattr(cfg.data, "get") else 0.01)

    emilia_cfg = EmiliaConfig(
        dataset_id=cfg.data.dataset_id,
        config_name=cfg.data.get("config_name", None),
        split=cfg.data.get("split", "train"),
        languages=tuple(cfg.data.languages),
        min_audio_seconds=cfg.data.min_audio_seconds,
        max_audio_seconds=cfg.data.max_audio_seconds,
        shuffle_buffer=cfg.data.shuffle_buffer,
    )
    pipe_cfg = PipelineConfig(
        emilia=emilia_cfg,
        max_text_tokens=cfg.data.max_text_tokens,
        audio_encoder_stride=ENCODER_STRIDE,
        num_workers=int(cfg.data.num_workers),
        batch_size=int(cfg.runtime.micro_batch_size),
        cache_dir=cache_dir,
        # Push pack_example + collate_packed into the worker. Main loop then
        # only does H2D + forward — no per-step Python pack cost between
        # forwards (which was leaving the GPU idle).
        specials=tok.specials if use_cache else None,
        pack_cfg=pack_cfg if use_cache else None,
        packed_total_length=packed_total_length if use_cache else 0,
        eval_holdout_frac=eval_holdout_frac if use_cache else 0.0,
        is_eval=False,
    )
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    # Shard the stream by DP rank (each TP group reads the same data).
    from megatron.core import parallel_state
    dp_rank = parallel_state.get_data_parallel_rank()
    dp_world = parallel_state.get_data_parallel_world_size()
    dl = make_dataloader(pipe_cfg, tok, rank=dp_rank, world_size=dp_world)

    # Separate eval dataloader on the held-out shard slice. Disjoint shards =
    # honest eval loss; separate DataLoader = eval doesn't reset the train
    # iterator's worker state (which was causing train/eval to share a small
    # data prefix and the model to memorise it).
    dl_eval = None
    valid_cache_dir = (cfg.data.get("valid_cache_dir", None)
                       if hasattr(cfg.data, "get") else None)
    if use_cache and valid_cache_dir:
        # Dedicated, fixed validation set in its own dir (shared across runs for
        # an apples-to-apples data-scaling comparison). Read the WHOLE dir
        # (holdout_frac=0, is_eval=False) — it contains only held-out shards.
        from dataclasses import replace as _replace
        eval_pipe_cfg = _replace(pipe_cfg, cache_dir=valid_cache_dir,
                                 eval_holdout_frac=0.0, is_eval=False,
                                 num_workers=max(1, int(cfg.data.num_workers) // 2))
        dl_eval = make_dataloader(eval_pipe_cfg, tok, rank=dp_rank, world_size=dp_world)
        _log(f"[train] eval dataloader: dedicated valid dir {valid_cache_dir}")
    elif use_cache and eval_holdout_frac > 0:
        from dataclasses import replace as _replace
        eval_pipe_cfg = _replace(pipe_cfg, is_eval=True,
                                 num_workers=max(1, int(cfg.data.num_workers) // 2))
        dl_eval = make_dataloader(eval_pipe_cfg, tok, rank=dp_rank, world_size=dp_world)
        _log(f"[train] eval dataloader: held-out shards (frac={eval_holdout_frac})")

    # TE variable-length FlashAttention toggle. When true AND packing is on,
    # the backbone routes to TE's packed-attention kernel via PackedSeqParams.
    model.use_packed_attn = bool(
        cfg.runtime.get("use_packed_attn", False) if hasattr(cfg.runtime, "get") else False
    )

    # Optional torch.compile — gated behind `runtime.compile: true`. With
    # sequence packing `packed_total_length` is fixed so dynamic-shape recompiles
    # won't fire. Wait for the training loop to be stable before enabling.
    if bool(cfg.runtime.get("compile", False) if hasattr(cfg.runtime, "get") else False):
        # mode='default' avoids cudagraph capture (which broke on variable-shape
        # packed inputs in 'reduce-overhead' mode). dynamic=True is the safe
        # choice when L can vary across batches.
        _log("[train] torch.compile(model, mode='default', dynamic=True)")
        model = torch.compile(model, mode="default", dynamic=True)

    _log(f"[train] starting training loop for {args.train_steps} steps (from step {start_step})")
    model.train()
    device = torch.device("cuda")
    step = start_step
    t0 = time.time()
    ckpt_every = int(cfg.runtime.get("checkpoint_every_n_steps", 0)
                     if hasattr(cfg.runtime, "get") else 0)
    eval_every = int(cfg.runtime.get("eval_every_n_steps", 0)
                     if hasattr(cfg.runtime, "get") else 0)
    eval_batches = int(cfg.runtime.get("eval_batches", 10)
                       if hasattr(cfg.runtime, "get") else 10)
    keep_last_k = int(cfg.runtime.get("keep_last_k_checkpoints", 3)
                      if hasattr(cfg.runtime, "get") else 3)
    grad_accum = int(cfg.runtime.get("gradient_accumulation_steps", 1)
                     if hasattr(cfg.runtime, "get") else 1)
    log_every = int(cfg.runtime.get("log_every_n_steps", 20)
                    if hasattr(cfg.runtime, "get") else 20)
    if packed_total_length:
        _log(f"[train] sequence packing ON: packed_total_length={packed_total_length}")
    if grad_accum > 1:
        _log(f"[train] gradient accumulation: {grad_accum} micro-steps per optimizer step")

    # --- MFU counter setup ---------------------------------------------------
    # Approximate training FLOPs/token (fwd+bwd):
    #   dense  = 6 * N_matmul  (transformer projections + MLP + LM-head GEMM;
    #            the input token-embedding *lookup* is excluded — it's a gather,
    #            not a matmul — but a tied LM head still counts once as a matmul)
    #   diff   = 6 * N_diffhead * timesteps_per_forward, applied to AUDIO
    #            positions only (per-batch audio fraction, measured live)
    # The non-parametric attention term (12*L*H*T_doc) is omitted, so MFU is a
    # slight (~few %) UNDER-estimate — i.e. conservative. Peak is H100 dense:
    # 989 TFLOP/s bf16, 1979 TFLOP/s fp8 (override via runtime.gpu_peak_tflops).
    _n_total = sum(p.numel() for p in inner_model.parameters())
    _n_diff = sum(p.numel() for p in inner_model.diffusion_head.parameters())
    _tied = inner_model.lm_head.weight is inner_model.token_embed.weight
    _n_embed_lookup = 0 if _tied else inner_model.token_embed.weight.numel()
    flops_dense_per_token = 6.0 * (_n_total - _n_diff - _n_embed_lookup)
    _tpf = 4
    try:
        _tpf = int(cfg.loss.ddpm.timesteps_per_forward)
    except Exception:
        pass
    flops_diff_coeff = 6.0 * _n_diff * _tpf
    _peak_tflops = (cfg.runtime.get("gpu_peak_tflops", None)
                    if hasattr(cfg.runtime, "get") else None)
    if _peak_tflops is None:
        _peak_tflops = 1979.0 if tfm_cfg.fp8 is not None else 989.0
    peak_flops = float(_peak_tflops) * 1e12
    _log(f"[train] MFU counter: dense={flops_dense_per_token/6:.3e} matmul-params  "
         f"diff_head={_n_diff/1e6:.1f}M×{_tpf}  peak={_peak_tflops:.0f} TFLOP/s/GPU  "
         f"gpus={world_size}")

    # FP8 amax warmup on resume. TE's DelayedScaling keeps an `amax_history`
    # buffer per FP8 module; it's NOT round-tripped reliably through PyTorch's
    # state_dict (TE uses _extra_state, which is silently dropped on load if
    # any key shape mismatches). After resume, the first dozen-or-so forward
    # passes have stale FP8 scale → garbage outputs (lm jumps from ~5 → ~60).
    # Run a short eval-mode forward sweep so amax stabilises before we touch
    # gradients. No-op when starting fresh.
    if start_step > 0 and tfm_cfg.fp8 is not None:
        n_warmup = int(cfg.runtime.get("fp8_amax_warmup_steps", 1024)
                       if hasattr(cfg.runtime, "get") else 1024)
        _log(f"[train] FP8 amax warmup: running {n_warmup} forward passes (eval mode, no_grad)")
        model.eval()
        warmup_iter = iter(dl)
        with torch.no_grad():
            for _wi in range(n_warmup):
                try:
                    raw = next(warmup_iter)
                except StopIteration:
                    break
                if use_cache:
                    batch = {k: (v.to(device, non_blocking=True) if hasattr(v, "to") else v)
                             for k, v in raw.items()}
                else:
                    batch = _pack_batch_from_audio(
                        text_ids_list=raw["text_ids"],
                        audio=raw["audio"].to(device),
                        audio_lens=raw["audio_lens"],
                        vv=vv, specials=tok.specials, pack_cfg=pack_cfg,
                        encoder_stride=ENCODER_STRIDE, device=device,
                        packed_total_length=packed_total_length,
                    )
                with fp8_autocast_context(tfm_cfg):
                    model(
                        input_ids=batch["input_ids"],
                        input_latents=batch["input_latents"],
                        is_audio_input=batch["is_audio_input"],
                        attention_mask=batch.get("attention_mask"),
                        cu_seqlens=batch.get("cu_seqlens"),
                    )
        model.train()
        _log(f"[train] FP8 amax warmup done")

    accum_idx = 0
    for raw in dl:
        if use_cache:
            # Worker already ran pack_example + collate_packed — `raw` IS the
            # batch dict. Just move tensors to GPU.
            batch = {k: (v.to(device, non_blocking=True) if hasattr(v, "to") else v)
                     for k, v in raw.items()}
        else:
            batch = _pack_batch_from_audio(
                text_ids_list=raw["text_ids"],
                audio=raw["audio"].to(device),
                audio_lens=raw["audio_lens"],
                vv=vv, specials=tok.specials, pack_cfg=pack_cfg,
                encoder_stride=ENCODER_STRIDE, device=device,
                packed_total_length=packed_total_length,
            )

        if accum_idx == 0:
            optimizer.zero_grad(set_to_none=True)
        with fp8_autocast_context(tfm_cfg):
            out = model(
                input_ids=batch["input_ids"],
                input_latents=batch["input_latents"],
                is_audio_input=batch["is_audio_input"],
                attention_mask=batch.get("attention_mask"),
                cu_seqlens=batch.get("cu_seqlens"),
            )
        # Loss runs in fp32 for CE (via Megatron's vocab_parallel CE when TP>1,
        # else F.cross_entropy) and dt (bf16) for the diffusion head.
        losses = loss_fn(
            text_logits=out["text_logits"].float(),
            text_targets=batch["text_targets"],
            audio_targets=batch["audio_targets"].to(dt),
            audio_mask=batch["audio_mask"],
            hidden_states=out["hidden"].to(dt),
            diff_head=inner_model.diffusion_head,
        )
        (losses["loss"] / grad_accum).backward()
        accum_idx += 1
        if accum_idx >= grad_accum:
            # Float16OptimizerWithFloat16Params handles bf16→fp32 main_grad copy,
            # `clip_grad`-driven grad clipping, and NaN/inf detection inside
            # step(). It returns (success, grad_norm, num_zeros). When NaN/inf
            # is found `success=False` and we skip the scheduler advance.
            success, _grad_norm, _num_zeros = optimizer.step()
            if success:
                scheduler.step(increment=samples_per_opt_step)
            accum_idx = 0
            loss_val = float(losses["loss"].item())
            lm_val = float(losses["lm_loss"].item())
            diff_val = float(losses["diff_loss"].item())
            elapsed = time.time() - t0
            ms_per = elapsed * 1000 / max(step - start_step + 1, 1)
            dp_world = parallel_state.get_data_parallel_world_size()
            tokens_per_step = (int(cfg.runtime.micro_batch_size) *
                               max(packed_total_length, 1) *
                               grad_accum * dp_world)
            tps = tokens_per_step * 1000 / max(ms_per, 1.0)
            # MFU = achieved model-FLOPs/s ÷ aggregate HW peak. `tps` is global
            # (all DP ranks); `world_size` is total GPUs (DP×TP). Audio fraction
            # is measured from this batch so the diffusion-head term tracks the
            # real audio/text mix.
            audio_frac = (float(batch["is_audio_input"].float().mean())
                          if "is_audio_input" in batch else 0.0)
            flops_per_token = flops_dense_per_token + flops_diff_coeff * audio_frac
            mfu = flops_per_token * tps / max(world_size * peak_flops, 1.0)
            # Stdout log honors `runtime.log_every_n_steps`. Wandb / TB still
            # log every step so the curves are smooth (cheap).
            if log_every <= 1 or (step % log_every == 0):
                _log(f"[train] step={step:>4}  loss={loss_val:.4f}  "
                     f"lm={lm_val:.4f}  diff={diff_val:.4f}  "
                     f"ms/step={ms_per:.0f}  toks/s={tps:.0f}  mfu={mfu*100:.1f}%")
            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", loss_val, step)
                tb_writer.add_scalar("train/lm_loss", lm_val, step)
                tb_writer.add_scalar("train/diff_loss", diff_val, step)
                tb_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
                tb_writer.add_scalar("train/mfu", mfu, step)
                tb_writer.add_scalar("train/tokens_per_sec", tps, step)
            if wandb_run is not None:
                wandb_run.log({
                    "train/loss": loss_val,
                    "train/lm_loss": lm_val,
                    "train/diff_loss": diff_val,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/mfu": mfu,
                    "train/tokens_per_sec": tps,
                }, step=step)

            step += 1

            if ckpt_every > 0 and step > 0 and step % ckpt_every == 0:
                from . import checkpoint as ckpt_mod

                ckpt_path = ckpt_mod.save(
                    output_dir=output_dir, step=step,
                    model=inner_model, optimizer=optimizer, keep_last_k=keep_last_k,
                )
                _log(f"[train] checkpoint → {ckpt_path}")

            if eval_every > 0 and step > 0 and step % eval_every == 0:
                eval_metrics = _run_eval(
                    model=model, inner_model=inner_model,
                    dl=(dl_eval if dl_eval is not None else dl),
                    loss_fn=loss_fn,
                    tok=tok, vv=vv, use_cache=use_cache, pack_cfg=pack_cfg,
                    device=device, dt=dt, tfm_cfg=tfm_cfg,
                    packed_total_length=packed_total_length, n_batches=eval_batches,
                )
                _log(f"[train] eval step={step}  loss={eval_metrics['loss']:.4f}  "
                     f"lm={eval_metrics['lm']:.4f}  diff={eval_metrics['diff']:.4f}  "
                     f"n={eval_metrics['n']}")
                if tb_writer is not None:
                    tb_writer.add_scalar("eval/loss", eval_metrics["loss"], step)
                    tb_writer.add_scalar("eval/lm", eval_metrics["lm"], step)
                    tb_writer.add_scalar("eval/diff", eval_metrics["diff"], step)
                if wandb_run is not None:
                    wandb_run.log({
                        "eval/loss": eval_metrics["loss"],
                        "eval/lm": eval_metrics["lm"],
                        "eval/diff": eval_metrics["diff"],
                    }, step=step)
                model.train()

            if step >= args.train_steps:
                break

    # Final save — collective: rank-0 writes, all ranks hit the barrier inside
    # `checkpoint.save`. Don't gate this on `_is_main_rank()`.
    if ckpt_every > 0:
        from . import checkpoint as ckpt_mod

        ckpt_mod.save(output_dir=output_dir, step=step, model=inner_model,
                      optimizer=optimizer, keep_last_k=keep_last_k)
    if tb_writer is not None:
        tb_writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    _log("[train] done")
    return 0


@torch.no_grad()
def _run_eval(*, model, inner_model, dl, loss_fn, tok, vv, use_cache, pack_cfg,
              device, dt, tfm_cfg, packed_total_length, n_batches: int):
    """Lightweight eval: pull `n_batches` from the same iterator (cheap and
    code-path-covering; a real held-out set would need a separate dataloader —
    add when we need proper validation numbers).

    Returns a dict with mean loss, lm, and diff so we can spot training/eval
    divergence (the resume-anomaly investigation needed lm+diff separately).
    """
    import torch

    from .models.bridge_backbone import fp8_autocast_context

    model.eval()
    total_loss = 0.0
    total_lm = 0.0
    total_diff = 0.0
    n = 0
    for raw in dl:
        if n >= n_batches:
            break
        if use_cache:
            batch = {k: (v.to(device, non_blocking=True) if hasattr(v, "to") else v)
                     for k, v in raw.items()}
        else:
            batch = _pack_batch_from_audio(
                text_ids_list=raw["text_ids"], audio=raw["audio"].to(device),
                audio_lens=raw["audio_lens"], vv=vv, specials=tok.specials,
                pack_cfg=pack_cfg, encoder_stride=__import__(
                    "latent_lm.models.tokenizer", fromlist=["ENCODER_STRIDE"]).ENCODER_STRIDE,
                device=device, packed_total_length=packed_total_length,
            )
        with fp8_autocast_context(tfm_cfg):
            out = model(
                input_ids=batch["input_ids"],
                input_latents=batch["input_latents"],
                is_audio_input=batch["is_audio_input"],
                attention_mask=batch.get("attention_mask"),
                cu_seqlens=batch.get("cu_seqlens"),
            )
        losses = loss_fn(
            text_logits=out["text_logits"].float(),
            text_targets=batch["text_targets"],
            audio_targets=batch["audio_targets"].to(dt),
            audio_mask=batch["audio_mask"],
            hidden_states=out["hidden"].to(dt),
            diff_head=inner_model.diffusion_head,
        )
        total_loss += float(losses["loss"].item())
        total_lm += float(losses["lm_loss"].item())
        total_diff += float(losses["diff_loss"].item())
        n += 1
    d = max(n, 1)
    return {
        "loss": total_loss / d,
        "lm": total_lm / d,
        "diff": total_diff / d,
        "n": n,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train-steps", type=int, default=0,
                        help="Number of training iterations to run. 0 = dry-run build only.")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a checkpoint dir (or one containing `latest`).")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    _log(f"[train] config: {args.config}")

    try:
        if args.train_steps > 0:
            return train(args, cfg)
        # Otherwise: just build the model and report.
        from .data.text_tokenizer import TextTokenizer, TextTokenizerConfig
        tok = TextTokenizer(TextTokenizerConfig(
            model_id=getattr(cfg, "tokenizer", {}).get("text", "gpt2")
            if hasattr(cfg, "tokenizer") else "gpt2",
            tp=cfg.parallelism.tensor_model_parallel_size,
        )) if getattr(cfg, "tokenizer", None) else None
        vocab = tok.vocab_size if tok is not None else cfg.model.vocab_size
        model, tfm_cfg = build_model(cfg, vocab_size=vocab)
        n_params = sum(p.numel() for p in model.parameters())
        _log(f"[train] model built. params (pre-TP-shard): {n_params/1e6:.1f}M "
             f"  fp8={tfm_cfg.fp8}  tp={tfm_cfg.tensor_model_parallel_size}  "
             f"pp={tfm_cfg.pipeline_model_parallel_size}  cp={tfm_cfg.context_parallel_size}")
        return 0 if args.dry_run else 2
    finally:
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
