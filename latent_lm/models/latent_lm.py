"""LatentLM model: mixed discrete/continuous AR Transformer.

This module is backbone-agnostic. The backbone is any callable:
    backbone(x: (B, L, H), attention_mask: (B, L) bool) -> (B, L, H)
that applies causal self-attention. At training time we plug in a
Megatron-core / NeMo GPT stack; this file avoids importing NeMo so it can be
unit-tested on the login node.

Input mixing:
    * positions where `is_audio_input[b, l]` is True use `input_latents[b, l]`
      projected via `latent_in_proj`.
    * other positions use `token_embed(input_ids[b, l])`.

Outputs:
    * `hidden`: (B, L, H) — used by diffusion head at audio-predicting positions.
    * `text_logits`: (B, L, V) — used by text CE loss.

Inheriting from MegatronModule: required so `bridge.training.optim.setup_optimizer`
can read `model_chunk.config.{num_attention_heads,num_query_groups,kv_channels}`
when wrapping `TensorParallelMuon` in `Float16OptimizerWithFloat16Params`. The
TransformerConfig comes from the GPT trunk; our diffusion head and latent_in_proj
have no qkv structure, so the QKV-split path in Muon is gated on per-param
`is_qkv` flags (set by Megatron only on `linear_qkv.weight`) and skipped for them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from .diffusion_head import DiffusionHead


class Backbone(Protocol):
    def __call__(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...


@dataclass
class LatentLMConfig:
    vocab_size: int = 32_000          # includes 5 specials appended at the end
    hidden_dim: int = 1024
    latent_dim: int = 64
    diff_head_layers: int = 3
    diff_head_ffn_mult: int = 4
    tie_embeddings: bool = True


def _megatron_module_base():
    """Return MegatronModule when available (Megatron container), else nn.Module.

    Lets this module import cleanly anywhere Megatron isn't usable — e.g. a CPU
    test env where it's not installed, or only partially present. We catch broadly
    (not just ImportError) so a broken/half-installed Megatron degrades to the
    plain nn.Module base rather than crashing the import.
    """
    try:
        from megatron.core.transformer.module import MegatronModule
        return MegatronModule
    except Exception:
        return nn.Module


class LatentLM(_megatron_module_base()):
    """Mixed discrete/continuous AR model.

    `token_embed` and `lm_head` default to vanilla `nn.Embedding` /
    `nn.Linear`, which is correct when TP=1. For TP>1 pass in Megatron's
    `VocabParallelEmbedding` and `ColumnParallelLinear` (set up by
    `latent_lm.models.bridge_backbone`).

    Forward outputs include hidden states (needed by the diffusion head).
    `text_logits` is only computed when the lm_head is a plain `nn.Linear` —
    for TP-parallel column-sharded heads, callers should compute the loss
    in a TP-aware way (see Megatron's `VocabParallelCrossEntropy`).
    """

    def __init__(
        self,
        cfg: LatentLMConfig,
        backbone: Backbone,
        token_embed: nn.Module | None = None,
        lm_head: nn.Module | None = None,
        transformer_config=None,
    ) -> None:
        # MegatronModule.__init__(config=...) sets self.config; setup_optimizer
        # reads num_attention_heads/num_query_groups/kv_channels from it.
        # When MegatronModule isn't importable (login-node tests) we fall back
        # to nn.Module which ignores the config kwarg.
        try:
            super().__init__(config=transformer_config)
        except TypeError:
            super().__init__()
        self.cfg = cfg
        self.backbone = backbone

        self.token_embed = token_embed or nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.latent_in_proj = nn.Linear(cfg.latent_dim, cfg.hidden_dim, bias=False)

        self.lm_head = lm_head or nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings and lm_head is None and token_embed is None:
            self.lm_head.weight = self.token_embed.weight

        self.diffusion_head = DiffusionHead(
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            cond_dim=cfg.hidden_dim,
            n_layers=cfg.diff_head_layers,
            ffn_mult=cfg.diff_head_ffn_mult,
        )

    def embed_inputs(
        self,
        input_ids: torch.Tensor,          # (B, L)
        input_latents: torch.Tensor,      # (B, L, d)
        is_audio_input: torch.Tensor,     # (B, L) bool
    ) -> torch.Tensor:
        text_e = self.token_embed(input_ids)
        lat_e = self.latent_in_proj(input_latents.to(text_e.dtype))
        mask = is_audio_input.unsqueeze(-1).to(text_e.dtype)
        return (1 - mask) * text_e + mask * lat_e

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        input_latents: torch.Tensor,
        is_audio_input: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x = self.embed_inputs(input_ids, input_latents, is_audio_input)
        # Only pass cu_seqlens if the backbone understands it (Megatron path);
        # TinyTransformer ignores it.
        try:
            hidden = self.backbone(
                x, attention_mask=attention_mask, cu_seqlens=cu_seqlens,
                use_packed_attn=getattr(self, "use_packed_attn", False),
            )
        except TypeError:
            hidden = self.backbone(x, attention_mask=attention_mask)
        # Megatron's ColumnParallelLinear returns (output, bias). Handle both.
        out = self.lm_head(hidden)
        text_logits = out[0] if isinstance(out, tuple) else out
        return {"hidden": hidden, "text_logits": text_logits}


# ---------------------------------------------------------------------------
# Minimal PyTorch backbone for unit tests and smoke runs.
# ---------------------------------------------------------------------------


class _CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        key_padding = None
        if attention_mask is not None:
            # attention_mask: (B, L) bool, True = keep
            key_padding = ~attention_mask[:, None, None, :]
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None if key_padding is None else key_padding.to(q.dtype) * -1e4,
            is_causal=True,
        )
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out(out)


class _Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_mult: int = 4) -> None:
        super().__init__()
        self.n1 = nn.RMSNorm(dim)
        self.attn = _CausalSelfAttention(dim, n_heads)
        self.n2 = nn.RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * ffn_mult, dim, bias=False),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.attn(self.n1(x), attention_mask)
        x = x + self.ffn(self.n2(x))
        return x


class TinyTransformer(nn.Module):
    """A minimal reference backbone — NOT for production training.

    Swap for `nemo.collections.llm.GPTModel` (or its Megatron-core backing)
    when wiring into NeMo. The NeMo backbone should expose a forward with the
    same `(x, attention_mask)` signature (write a thin adapter if needed).
    """

    def __init__(
        self, hidden_dim: int = 256, n_layers: int = 4, n_heads: int = 4
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_Block(hidden_dim, n_heads) for _ in range(n_layers)]
        )
        self.norm = nn.RMSNorm(hidden_dim)

    def __call__(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, attention_mask)
        return self.norm(x)


