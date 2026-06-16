"""Autoregressive sampling for LatentLM (TTS direction).

Pipeline:
    1. text tokens → `[BOS] text [BOD]`
    2. AR generate latents: at each step, run the backbone once to get `h_last`,
       then denoise the next 64-d latent via DPM-Solver (3–5 steps) conditioned
       on `h_last`. Apply classifier-free guidance with scale=4.0 (paper).
    3. stop when the LM head emits `<EOD>` OR a max-frame cap is hit.
    4. decode latents via the VibeVoice decoder.

This file is a SCAFFOLD — sampling calls into the backbone are sketched but
need KV-cache integration to be fast; we'll wire that when the NeMo backbone
lands. For now the code is structured so tests can exercise the math on the
`TinyTransformer` reference backbone.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .models.diffusion_head import DiffusionHead
from .models.latent_lm import LatentLM
from .data.collate import SpecialTokens


@dataclass
class SampleConfig:
    max_latent_frames: int = 256
    min_latent_frames: int = 0     # floor — VibeVoice decoder's stem conv needs
                                    # >= ~7 frames or it errors. Bump for
                                    # under-trained models that emit <eod> early.
    cfg_scale: float = 4.0         # paper's optimum for TTS
    dpm_steps: int = 5             # 3–5 in the paper's TTS ablation
    num_train_timesteps: int = 1000
    # "v_prediction" matches the new training default; "epsilon" is for
    # ablation only. Must match the trained checkpoint's loss formulation.
    prediction_type: str = "v_prediction"


def dpm_solver_sample_one(
    head: DiffusionHead,
    h_cond: torch.Tensor,            # (B, H) conditional LM hidden
    h_uncond: torch.Tensor,          # (B, H) unconditional LM hidden (for CFG)
    *,
    cfg_scale: float,
    steps: int,
    num_train_timesteps: int,
    latent_dim: int = 64,
    prediction_type: str = "v_prediction",
) -> torch.Tensor:
    """Run DPM-Solver for `steps` steps and return a predicted x0 latent.

    Uses HuggingFace diffusers' `DPMSolverMultistepScheduler`. The CFG line
    `pred_u + cfg_scale*(pred_c - pred_u)` is correct under both v- and
    ε-prediction (diffusers handles the reverse-process arithmetic internally).
    """
    from diffusers import DPMSolverMultistepScheduler

    B = h_cond.shape[0]
    device = h_cond.device
    dtype = h_cond.dtype

    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type=prediction_type,
        solver_order=2,
    )
    scheduler.set_timesteps(steps, device=device)

    x = torch.randn(B, latent_dim, device=device, dtype=dtype)
    for t in scheduler.timesteps:
        t_batch = t.expand(B).to(device=device)
        pred_c = head(x, t_batch, h_cond)
        pred_u = head(x, t_batch, h_uncond)
        pred = pred_u + cfg_scale * (pred_c - pred_u)
        x = scheduler.step(pred, t, x).prev_sample
    return x


@torch.no_grad()
def sample_tts(
    model: LatentLM,
    text_ids: torch.Tensor,          # (B, T_text) with BOS prepended, BOD appended by caller
    specials: SpecialTokens,
    cfg: SampleConfig,
) -> torch.Tensor:
    """Generate a (B, T_lat, 64) latent tensor.

    Simple reference implementation — re-runs the backbone over the full prefix
    every step. Replace with KV-cached stepping when integrating NeMo.
    """
    B = text_ids.shape[0]
    device = text_ids.device
    H = model.cfg.hidden_dim
    d = model.cfg.latent_dim

    # Build the unconditional context (empty text: just BOS+BOD) for CFG.
    uncond_ids = torch.tensor(
        [specials.bos, specials.bod], device=device, dtype=torch.long
    ).expand(B, -1)

    def _step_hidden(token_seq: torch.Tensor, latent_seq: torch.Tensor, is_audio: torch.Tensor):
        out = model(
            input_ids=token_seq,
            input_latents=latent_seq,
            is_audio_input=is_audio,
        )
        return out["hidden"][:, -1], out["text_logits"][:, -1]

    # State: separate cond / uncond sequences.
    def _init_state(prefix_ids: torch.Tensor):
        L = prefix_ids.shape[1]
        latents = torch.zeros(B, L, d, device=device)
        is_audio = torch.zeros(B, L, dtype=torch.bool, device=device)
        return prefix_ids.clone(), latents, is_audio

    cond_ids, cond_lat, cond_is_aud = _init_state(text_ids)
    unc_ids, unc_lat, unc_is_aud = _init_state(uncond_ids)

    generated = []
    for step_idx in range(cfg.max_latent_frames):
        h_cond, logits_cond = _step_hidden(cond_ids, cond_lat, cond_is_aud)
        h_unc, _ = _step_hidden(unc_ids, unc_lat, unc_is_aud)

        # EOD early-stop: if the LM head (on cond stream) strongly prefers EOD.
        # `min_latent_frames` floor prevents zero-frame outputs from
        # under-trained models (VibeVoice's stem conv needs ≥ kernel_size
        # frames; otherwise decode errors with kernel-too-large).
        if (step_idx >= cfg.min_latent_frames
                and logits_cond.argmax(-1).eq(specials.eod).all()):
            break

        # Keep h in the same dtype as the diffusion head's params (bf16 in our
        # standard setup) — `dpm_solver_sample_one` will sample x in the same
        # dtype so all matmuls inside the head are dtype-consistent.
        x0 = dpm_solver_sample_one(
            model.diffusion_head,
            h_cond,
            h_unc,
            cfg_scale=cfg.cfg_scale,
            steps=cfg.dpm_steps,
            num_train_timesteps=cfg.num_train_timesteps,
            latent_dim=d,
            prediction_type=cfg.prediction_type,
        )
        generated.append(x0)

        # Append the sampled latent to both streams.
        def _append(ids, lat, is_aud, x):
            new_id = torch.full((B, 1), specials.pad, dtype=torch.long, device=device)
            new_lat = x.unsqueeze(1).to(lat.dtype)
            new_aud = torch.ones(B, 1, dtype=torch.bool, device=device)
            return (
                torch.cat([ids, new_id], dim=1),
                torch.cat([lat, new_lat], dim=1),
                torch.cat([is_aud, new_aud], dim=1),
            )

        cond_ids, cond_lat, cond_is_aud = _append(cond_ids, cond_lat, cond_is_aud, x0)
        unc_ids, unc_lat, unc_is_aud = _append(unc_ids, unc_lat, unc_is_aud, x0)

    return torch.stack(generated, dim=1) if generated else torch.zeros(B, 0, d, device=device)

