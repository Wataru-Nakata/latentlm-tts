"""Next-token diffusion head (LatentLM paper §3.2).

  * Residual stack of `n_layers` FFN blocks (3 for TTS in the paper).
  * Each block: pre-RMSNorm → FFN → residual, modulated by AdaLN-Zero that
    conditions on (diffusion timestep embedding, LM backbone hidden `h_i`).
  * Target: ε-prediction (standard DDPM).

Shapes (B = batch, d = latent dim = 64, H = backbone hidden, D_t = timestep dim):
    x_t : (B, d)           noised latent at timestep t
    t   : (B,)             integer diffusion timesteps
    h   : (B, H)           LM hidden state at the predicting position
    out : (B, d)           predicted noise ε̂
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    """Sinusoidal embedding of integer timesteps (DDPM convention)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class AdaLNZeroBlock(nn.Module):
    """Pre-RMSNorm FFN with AdaLN-Zero modulation.

    The modulation MLP starts zero-initialised so the block is the identity
    at init — matches the paper's AdaLN-Zero recipe.
    """

    def __init__(self, hidden_dim: int, cond_dim: int, ffn_mult: int = 4) -> None:
        super().__init__()
        inner = hidden_dim * ffn_mult
        self.norm = RMSNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, inner, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(inner, hidden_dim, bias=False),
        )
        # shift, scale, gate
        self.mod = nn.Linear(cond_dim, hidden_dim * 3, bias=True)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.mod(cond).chunk(3, dim=-1)
        h = self.norm(x) * (1 + scale) + shift
        return x + gate * self.ffn(h)


class DiffusionHead(nn.Module):
    """LatentLM next-token diffusion head.

    Args:
        latent_dim: dimensionality of the continuous latent (64 for VibeVoice).
        hidden_dim: internal width of the head (typically == backbone hidden).
        cond_dim: width of the LM hidden state fed as condition (backbone hidden).
        n_layers: number of AdaLN-Zero FFN blocks. Paper uses 3 for TTS.
        timestep_dim: dimensionality of the sinusoidal timestep embedding.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 1024,
        cond_dim: int = 1024,
        n_layers: int = 3,
        timestep_dim: int = 256,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.timestep_dim = timestep_dim

        self.in_proj = nn.Linear(latent_dim, hidden_dim, bias=False)

        # Combine timestep embedding with LM hidden state into a single
        # condition vector fed to every block's AdaLN modulator.
        self.t_embed = nn.Sequential(
            nn.Linear(timestep_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.h_proj = nn.Linear(cond_dim, hidden_dim, bias=False)

        self.blocks = nn.ModuleList(
            [AdaLNZeroBlock(hidden_dim, hidden_dim, ffn_mult=ffn_mult) for _ in range(n_layers)]
        )

        self.out_norm = RMSNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise ε̂ for a batch of (noised latent, timestep, LM hidden)."""
        t_emb = _timestep_embedding(t, self.timestep_dim).to(x_t.dtype)
        cond = self.t_embed(t_emb) + self.h_proj(h)

        x = self.in_proj(x_t)
        for block in self.blocks:
            x = block(x, cond)
        x = self.out_norm(x)
        return self.out_proj(x)
