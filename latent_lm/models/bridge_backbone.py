"""Megatron-Bridge backbone for LatentLM.

Wraps a `GPTModel` produced by `megatron.bridge.AutoBridge.from_hf_pretrained`
into our `Backbone` protocol (callable that maps `[B, L, H] → [B, L, H]`).

We bypass `GPTModel.forward` and call `model.decoder` directly so that:
  * we feed our own pre-mixed (text+latent) embedding,
  * the output is hidden states (not logits) — the LM head and diffusion head
    consume them separately in `LatentLM`.

The HF token embedding (`model.embedding.word_embeddings`) and output layer
(`model.output_layer`) are re-exposed so `LatentLM` can tie its `token_embed`
and `lm_head` to them — that's how Qwen's pretrained vocab survives the wrap.

Sequence parallel: not supported on the bypass path. The standard GPTModel
embedding scatters to the SP region before the decoder; we'd need to replicate
that. Asserted off; flip when SP becomes a hard requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
from torch import nn


def _to_packed_seq_params(cu_seqlens: Optional[torch.Tensor], device: torch.device):
    if cu_seqlens is None:
        return None
    from megatron.core.packed_seq_params import PackedSeqParams

    cu = cu_seqlens.to(device=device, dtype=torch.int32)
    max_s = int((cu[1:] - cu[:-1]).max().item())
    return PackedSeqParams(
        cu_seqlens_q=cu, cu_seqlens_kv=cu,
        max_seqlen_q=max_s, max_seqlen_kv=max_s,
        qkv_format="thd",
    )


def _build_dense_mask(L: int, B: int, attention_mask, cu_seqlens, device):
    """Block-diagonal causal mask for non-packed-attn path. True = masked."""
    causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
    if cu_seqlens is not None:
        doc_ids = torch.zeros(L, dtype=torch.long, device=device)
        for d in range(cu_seqlens.numel() - 1):
            s, e = int(cu_seqlens[d].item()), int(cu_seqlens[d + 1].item())
            doc_ids[s:e] = d
        cross_doc = doc_ids[:, None] != doc_ids[None, :]
        return (causal | cross_doc)[None, None]
    if attention_mask is not None:
        pad = ~attention_mask
        return causal[None, None].expand(B, 1, L, L) | pad[:, None, None, :]
    return causal[None, None].expand(B, 1, L, L)


class _OutputLayerBSHAdapter(nn.Module):
    """Wrap ColumnParallelLinear (SBH→SBV) so it can be called on BSH tensors.

    LatentLM passes the backbone hidden as `[B, L, H]` to `lm_head`. The
    Bridge `output_layer` (ColumnParallelLinear) is shape-agnostic over the
    leading dims but returns `(logits, bias)` — we keep that tuple shape so
    `LatentLM.forward` (which already handles tuple returns from TP linears)
    is unchanged.

    When the underlying GPTModel uses tied embeddings, output_layer is built
    with skip_weight_param_allocation=True and the weight is fetched from
    the embedding at forward time. We replicate that here.
    """

    def __init__(self, gpt_model: nn.Module) -> None:
        super().__init__()
        # Store as plain attribute (not nn.Module child) to avoid duplicating
        # the entire GPTModel in the module tree — it's already registered
        # via BridgeBackbone.gpt.
        object.__setattr__(self, "gpt_model", gpt_model)
        self.output_layer = gpt_model.output_layer

    def forward(self, hidden_bsh: torch.Tensor):
        h_sbh = hidden_bsh.transpose(0, 1).contiguous()
        weight = None
        if getattr(self.gpt_model, "share_embeddings_and_output_weights", False):
            weight = self.gpt_model.shared_embedding_or_output_weight()
        out_sbv, bias = self.output_layer(h_sbh, weight=weight)
        return out_sbv.transpose(0, 1).contiguous(), bias

    @property
    def weight(self):
        if getattr(self.gpt_model, "share_embeddings_and_output_weights", False):
            return self.gpt_model.shared_embedding_or_output_weight()
        return self.output_layer.weight


class BridgeBackbone(nn.Module):
    """Backbone wrapping a Bridge-built `GPTModel`.

    Expects the caller (LatentLM) to:
      * use `.word_embeddings` for token embedding (so HF weights are tied),
      * use `.output_layer_adapter` for the LM head (TP-sharded, BSH-friendly),
      * pass mixed [B, L, H] embeddings to `forward`.
    """

    def __init__(self, gpt_model: nn.Module, hidden_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.gpt = gpt_model
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.output_layer_adapter = _OutputLayerBSHAdapter(gpt_model)

    @property
    def word_embeddings(self) -> nn.Module:
        return self.gpt.embedding.word_embeddings

    @property
    def output_layer(self) -> nn.Module:
        return self.output_layer_adapter

    def __call__(
        self,
        x: torch.Tensor,                          # [B, L, H]
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        use_packed_attn: bool = False,
    ) -> torch.Tensor:
        cfg = self.gpt.config
        assert not getattr(cfg, "sequence_parallel", False), (
            "BridgeBackbone bypass does not handle SP scatter; set sequence_parallel=False."
        )

        B, L, H = x.shape
        h_sbh = x.transpose(0, 1).contiguous()

        if use_packed_attn and cu_seqlens is not None:
            packed = _to_packed_seq_params(cu_seqlens, device=x.device)
            mask = None
        else:
            packed = None
            mask = _build_dense_mask(L, B, attention_mask, cu_seqlens, x.device)

        rotary_pos_emb = None
        if getattr(self.gpt, "position_embedding_type", None) == "rope":
            rotary_seq_len = self.gpt.rotary_pos_emb.get_rotary_seq_len(
                None, self.gpt.decoder, h_sbh, cfg, packed,
            )
            rotary_pos_emb = self.gpt.rotary_pos_emb(
                rotary_seq_len,
                packed_seq=(packed is not None and packed.qkv_format == "thd"),
                cp_group=None,
            )

        out_sbh = self.gpt.decoder(
            hidden_states=h_sbh,
            attention_mask=mask,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed,
        )
        return out_sbh.transpose(0, 1).contiguous()


@dataclass
class BridgeBuildResult:
    backbone: BridgeBackbone
    hidden_dim: int
    vocab_size: int
    hf_config: object


def build_bridge_backbone(
    hf_model_id: str,
    *,
    load_weights: bool,
    tp: int = 1,
    pp: int = 1,
    cp: int = 1,
    seed: int = 1234,
    provider_overrides: Optional[dict] = None,
) -> BridgeBuildResult:
    """Build a BridgeBackbone from an HF model id.

    Initialises Megatron parallel state (idempotent), constructs a Bridge
    provider, fills the recipe-side init methods Bridge leaves None, and
    materialises the GPTModel.
    """
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
    from megatron.bridge import AutoBridge

    if not dist.is_initialized():
        import os
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        # Honour LATENT_LM_NCCL_TIMEOUT_S so a hung collective fails fast
        # instead of waiting the 30-min default. Short timeout + the
        # TORCH_NCCL_DUMP_ON_TIMEOUT env var give a stack dump on hang.
        from datetime import timedelta
        nccl_timeout_s = os.environ.get("LATENT_LM_NCCL_TIMEOUT_S")
        init_kwargs = {
            "backend": "nccl",
            "device_id": torch.device(f"cuda:{local_rank}"),
        }
        if nccl_timeout_s:
            init_kwargs["timeout"] = timedelta(seconds=int(nccl_timeout_s))
        dist.init_process_group(**init_kwargs)
    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            context_parallel_size=cp,
        )
    model_parallel_cuda_manual_seed(seed)

    bridge = AutoBridge.from_hf_pretrained(hf_model_id)
    provider = bridge.to_megatron_provider(load_weights=load_weights)

    # Bridge leaves these None — recipes normally fill them. Crashes inside
    # MLP construction if missing (verified in Qwen3.5 PoC).
    init_std = float(getattr(provider, "init_method_std", None) or 0.02)
    init = partial(torch.nn.init.normal_, mean=0.0, std=init_std)
    if getattr(provider, "init_method", None) is None:
        provider.init_method = init
    if getattr(provider, "output_layer_init_method", None) is None:
        provider.output_layer_init_method = init

    provider.tensor_model_parallel_size = tp
    provider.pipeline_model_parallel_size = pp
    provider.context_parallel_size = cp
    provider.sequence_parallel = False  # bypass path requires this

    for k, v in (provider_overrides or {}).items():
        if hasattr(provider, k):
            setattr(provider, k, v)

    # Bridge defers MCore's TransformerConfig.__post_init__ — it's a no-op
    # at __init__, and `finalize()` is what actually derives kv_channels,
    # ffn_hidden_size, num_query_groups, fp8 spec selection, etc. Call it
    # after our overrides so the derived fields reflect our settings.
    # (Without this, kv_channels stays None and RotaryEmbedding crashes
    # with arange(0, None, 2, ...).)
    if hasattr(provider, "finalize"):
        provider.finalize()

    # `provide()` is a lower-level entry than `provide_distributed_model()`;
    # the latter populates `_pg_collection` from MPU. We want to skip the
    # full DDP wrap (we use torch DDP later), so populate it here.
    provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    gpt = provider.provide()

    # `provide_distributed_model` would invoke any registered pre_wrap_hooks
    # (Bridge registers `load_weights_hf_to_megatron` there when load_weights=
    # True). We skipped that path, so trigger the hook ourselves.
    if load_weights and provider.pre_wrap_hook is not None:
        provider.pre_wrap_hook([gpt])

    backbone = BridgeBackbone(
        gpt, hidden_dim=int(provider.hidden_size),
        vocab_size=int(provider.vocab_size),
    )
    return BridgeBuildResult(
        backbone=backbone,
        hidden_dim=int(provider.hidden_size),
        vocab_size=int(provider.vocab_size),
        hf_config=getattr(bridge, "hf_config", None),
    )


def fp8_autocast_context(tfm_cfg, enabled: bool = True):
    """Wrap a forward in TE's `fp8_autocast` when FP8 is enabled in the
    TransformerConfig; otherwise return a no-op context.
    """
    if not enabled or getattr(tfm_cfg, "fp8", None) is None:
        import contextlib

        return contextlib.nullcontext()

    from transformer_engine.common.recipe import DelayedScaling, Format
    from transformer_engine.pytorch import fp8_autocast

    fmt = {"hybrid": Format.HYBRID, "e4m3": Format.E4M3}[tfm_cfg.fp8]
    recipe = DelayedScaling(
        margin=tfm_cfg.fp8_margin,
        fp8_format=fmt,
        amax_history_len=tfm_cfg.fp8_amax_history_len,
        amax_compute_algo=tfm_cfg.fp8_amax_compute_algo,
        reduce_amax=True,
    )
    return fp8_autocast(enabled=True, fp8_recipe=recipe)
