"""Plain-PyTorch transformer backbone for the no-container training path.

`LatentLM` is backbone-agnostic: the backbone is any callable
`(x: (B, L, H), attention_mask: (B, L) bool) -> (B, L, H)` applying causal
self-attention. The container path uses a Megatron-core GPT stack
(`bridge_backbone.py`); this module provides a self-contained alternative with
**RoPE** so you can train small models with nothing but PyTorch + `pip install`
(no NeMo/Megatron/transformer-engine).

It is the same architecture family as the toy `TinyTransformer` in
`latent_lm.py`, but configurable in depth/width and with rotary position
embeddings (the toy one has no positional encoding — it relied on Megatron's
RoPE). Use `TorchBackbone` for the lite trainer; `TinyTransformer` stays as the
minimal unit-test reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TorchBackboneConfig:
    hidden_dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    ffn_mult: int = 4
    max_seq_len: int = 8192
    rope_theta: float = 10_000.0


def _rotary_freqs(head_dim: int, max_seq_len: int, theta: float) -> torch.Tensor:
    """Precompute (max_seq_len, head_dim/2) rotation angles for RoPE."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    return torch.outer(t, inv_freq)  # (L, head_dim/2)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to x: (B, n_heads, L, head_dim)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    # cos/sin: (L, head_dim/2) -> broadcast over (B, n_heads, L, head_dim/2)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rx1
    out[..., 1::2] = rx2
    return out


class _Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attention_mask: torch.Tensor | None) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        q = _apply_rope(q, cos[:L], sin[:L])
        k = _apply_rope(k, cos[:L], sin[:L])
        attn_mask = None
        if attention_mask is not None:
            # (B, L) bool, True = keep -> additive mask over key dim.
            attn_mask = torch.zeros(B, 1, 1, L, dtype=q.dtype, device=q.device)
            attn_mask = attn_mask.masked_fill(~attention_mask[:, None, None, :], float("-inf"))
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=True)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out(out)


class _Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_mult: int) -> None:
        super().__init__()
        self.n1 = nn.RMSNorm(dim)
        self.attn = _Attention(dim, n_heads)
        self.n2 = nn.RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * ffn_mult, dim, bias=False),
        )

    def forward(self, x, cos, sin, attention_mask):
        x = x + self.attn(self.n1(x), cos, sin, attention_mask)
        x = x + self.ffn(self.n2(x))
        return x


class TorchBackbone(nn.Module):
    """Configurable causal Transformer with RoPE — the no-container backbone.

    Signature matches the `Backbone` protocol in `latent_lm.py`:
        backbone(x, attention_mask=None) -> x
    `cu_seqlens`/`use_packed_attn` are not supported here (that is the Megatron
    packed-attention path); `LatentLM.forward` falls back to this signature.
    """

    def __init__(self, cfg: TorchBackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        head_dim = cfg.hidden_dim // cfg.n_heads
        freqs = _rotary_freqs(head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", freqs.cos(), persistent=False)
        self.register_buffer("rope_sin", freqs.sin(), persistent=False)
        self.blocks = nn.ModuleList(
            [_Block(cfg.hidden_dim, cfg.n_heads, cfg.ffn_mult) for _ in range(cfg.n_layers)]
        )
        self.norm = nn.RMSNorm(cfg.hidden_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        cos = self.rope_cos.to(x.dtype)
        sin = self.rope_sin.to(x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin, attention_mask)
        return self.norm(x)

    # allow positional call like the toy TinyTransformer
    __call__ = nn.Module.__call__
