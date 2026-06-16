"""Training losses for LatentLM.

Two heads share one backbone:
    * text positions  → cross-entropy on logits (L_LM)
    * audio positions → DDPM v-prediction (L_Diff), or ε-prediction (legacy)

Combined as L = L_LM + α · L_Diff. α is a hyperparameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def cosine_alpha_bar(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule; returns ᾱ_t for t = 0..T."""
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps, dtype=torch.float64) / num_timesteps
    f_t = torch.cos(((t + s) / (1 + s)) * torch.pi / 2).pow(2)
    alpha_bar = f_t / f_t[0]
    return alpha_bar.float()


@dataclass
class DDPMScheduleConfig:
    num_timesteps: int = 1000
    schedule: str = "cosine"  # only cosine implemented
    # K timesteps per forward — variance reduction; cost is on the diff head only.
    timesteps_per_forward: int = 1
    # "v_prediction" (Salimans & Ho 2022) — target v_t = √ᾱ·ε − √(1−ᾱ)·x₀.
    # Per-t loss magnitude is uniform; avoids the low-t degeneracy of
    # ε-prediction where the model can converge to ε̂ ≈ x_t (gives MSE→0 at
    # high t but MSE→2 at low t). "epsilon" kept for ablation.
    prediction_type: str = "v_prediction"


class DDPMLoss(nn.Module):
    """DDPM loss with K timesteps sampled per forward; predicts v̂ or ε̂.

    Diffusion head signature is unchanged (`head(x_t, t, h) → (N, d)`); the
    output is just *interpreted* as v̂ vs ε̂ per `cfg.prediction_type`.
    """

    def __init__(self, cfg: DDPMScheduleConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or DDPMScheduleConfig()
        alpha_bar = cosine_alpha_bar(self.cfg.num_timesteps)
        self.register_buffer("alpha_bar", alpha_bar, persistent=False)

    def sample_timesteps(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.cfg.num_timesteps, (n,), device=device, dtype=torch.long)

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        ab = self.alpha_bar[t].to(x0.dtype).view(-1, *([1] * (x0.ndim - 1)))
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def forward(
        self,
        head,
        x0: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            head: DiffusionHead instance (callable(x_t, t, h) -> v̂ or ε̂).
            x0: (N, d) target latents (flattened over audio positions).
            h:  (N, H) LM hidden states at those positions.
        """
        K = self.cfg.timesteps_per_forward
        n = x0.shape[0]
        if n == 0:
            return x0.new_zeros(())
        x0_rep = x0.repeat_interleave(K, dim=0)
        h_rep = h.repeat_interleave(K, dim=0)
        noise = torch.randn_like(x0_rep)
        t = self.sample_timesteps(n * K, device=x0.device)

        ab = self.alpha_bar[t].to(x0_rep.dtype).view(-1, *([1] * (x0_rep.ndim - 1)))
        sqrt_ab, sqrt_1mab = ab.sqrt(), (1 - ab).sqrt()
        x_t = sqrt_ab * x0_rep + sqrt_1mab * noise
        pred = head(x_t, t, h_rep)

        if self.cfg.prediction_type == "v_prediction":
            target = sqrt_ab * noise - sqrt_1mab * x0_rep
        elif self.cfg.prediction_type == "epsilon":
            target = noise
        else:
            raise ValueError(
                f"unknown prediction_type: {self.cfg.prediction_type!r} "
                f"(expected 'v_prediction' or 'epsilon')")
        loss = F.mse_loss(pred, target)
        # DEBUG: instrument what's actually being computed
        import os as _os
        if _os.environ.get("LATENTLM_DEBUG_DIFF") == "1":
            print(f"[DDPMLoss DEBUG] n={n} K={K}  pred_std={pred.float().std().item():.4f}  "
                  f"target_std={target.float().std().item():.4f}  "
                  f"x0_std={x0.float().std().item():.4f}  "
                  f"h_std={h.float().std().item():.4f}  "
                  f"loss={loss.item():.4f}", flush=True)
        return loss


def _chunked_ce_sum(
    logits_flat: torch.Tensor,
    targets_flat: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Cross-entropy with reduction='sum', computed in chunks under activation
    checkpointing. Returns the unnormalised sum-of-losses over valid tokens
    (`ignore_index=-100` honoured per-token, so chunking the token axis is
    correct). Caller divides by `n_valid` for the mean.

    Why: at L=32K with vocab=152K the dense F.cross_entropy keeps an L×V
    intermediate (~10 GB bf16) that doubles in backward → OOM at micro_batch>2.
    Chunking caps peak memory at chunk_size×V (~1.2 GB at chunk=4096) by
    recomputing each chunk's softmax in backward via `torch.utils.checkpoint`.
    """
    from torch.utils.checkpoint import checkpoint

    total = logits_flat.new_zeros((), dtype=torch.float32)

    def _chunk(lg: torch.Tensor, tg: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(lg, tg, ignore_index=-100, reduction="sum").float()

    n = logits_flat.size(0)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        # use_reentrant=False is the modern, autocast-friendly path; reentrant
        # checkpoint has known issues with mixed-precision and FP8.
        total = total + checkpoint(
            _chunk, logits_flat[start:end], targets_flat[start:end],
            use_reentrant=False,
        )
    return total


def _lm_cross_entropy(
    text_logits: torch.Tensor,
    text_targets: torch.Tensor,
    *,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """TP-aware text CE. Uses Megatron's `vocab_parallel_cross_entropy` when the
    LM head returned a TP-sharded logits tensor (skips the full-vocab all-gather);
    falls back to standard `F.cross_entropy` (chunked when long) otherwise.

    `chunk_size > 0` activates the chunked path on the non-TP branch when
    `N > chunk_size`; set to 0 to disable for ablation.
    """
    try:
        from megatron.core import parallel_state
        from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
        tp_world = parallel_state.get_tensor_model_parallel_world_size() \
            if parallel_state.is_initialized() else 1
    except Exception:
        # Megatron absent or only partially installed (e.g. CPU tests) → TP=1.
        tp_world = 1

    if tp_world > 1:
        # Sharded path. vocab_parallel_cross_entropy doesn't honour ignore_index,
        # so we mask manually.
        mask = (text_targets != -100)
        safe_target = text_targets.masked_fill(~mask, 0)
        per_token = vocab_parallel_cross_entropy(text_logits, safe_target)
        n_valid = mask.sum().clamp(min=1).to(per_token.dtype)
        return (per_token * mask.to(per_token.dtype)).sum() / n_valid

    logits_flat = text_logits.reshape(-1, text_logits.size(-1))
    targets_flat = text_targets.reshape(-1)
    if chunk_size <= 0 or logits_flat.size(0) <= chunk_size:
        return F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)
    n_valid = (targets_flat != -100).sum().clamp(min=1).to(torch.float32)
    sum_loss = _chunked_ce_sum(logits_flat, targets_flat, chunk_size)
    return (sum_loss / n_valid).to(text_logits.dtype)


class ModalityLoss(nn.Module):
    """Combined text-CE + diffusion loss with α weighting.

    Expects a per-position modality mask where True == text and False == audio.
    """

    def __init__(
        self,
        diff_loss: DDPMLoss,
        alpha: float = 1.0,
        ce_chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        self.diff_loss = diff_loss
        self.alpha = alpha
        self.ce_chunk_size = ce_chunk_size

    def forward(
        self,
        *,
        text_logits: torch.Tensor,        # (B, L, V) or (B, L, V/TP) when TP-sharded
        text_targets: torch.Tensor,       # (B, L)  -100 where not text
        audio_targets: torch.Tensor,      # (B, L, d)  zeros where not audio
        audio_mask: torch.Tensor,         # (B, L) bool
        hidden_states: torch.Tensor,      # (B, L, H) — backbone output (pre-heads)
        diff_head,
    ) -> dict[str, torch.Tensor]:
        lm_loss = _lm_cross_entropy(text_logits, text_targets,
                                    chunk_size=self.ce_chunk_size)
        idx = audio_mask.nonzero(as_tuple=False)
        if idx.numel() == 0:
            diff_loss = text_logits.new_zeros(())
        else:
            b, l = idx[:, 0], idx[:, 1]
            x0 = audio_targets[b, l]
            h = hidden_states[b, l]
            diff_loss = self.diff_loss(diff_head, x0, h)
        return {
            "loss": lm_loss + self.alpha * diff_loss,
            "lm_loss": lm_loss.detach(),
            "diff_loss": diff_loss.detach(),
        }
