"""Frozen VibeVoice acoustic tokenizer wrapper.

The VibeVoice σ-VAE maps 16 kHz mono waveforms to a 64-dim continuous latent
at 7.5 Hz and back. We only use it as an encoder/decoder — its weights stay
frozen throughout LatentLM training.

Key facts (confirmed from the bundled VibeVoiceAcousticTokenizerFeatureExtractor
at runtime — the earlier HF model-card summary said 16 kHz but the actual
preprocessor reports 24 kHz):
  * sample rate: 24 000 Hz, mono, amplitude-normalised to -25 dB FS
  * encoder stride: 3200 samples → 7.5 Hz frame rate (24000 / 7.5 = 3200)
  * latent dim: 64
  * supports streaming via padding_cache / use_cache
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


LATENT_DIM: int = 64
LATENT_HZ: float = 7.5
SAMPLE_RATE: int = 24_000
ENCODER_STRIDE: int = 3200  # samples per latent frame
# VibeVoice latents have native std≈5.12 (measured on 10K Emilia samples — see
# `scripts/check_latent_stats.py`). DDPM ε-prediction expects x0 ~ N(0, ~1),
# so we multiply by LATENT_SCALE before training and divide by it before
# decoding back to audio. Same pattern as Stable Diffusion's 0.18215.
LATENT_SCALE: float = 0.195


@dataclass
class VibeVoiceConfig:
    model_id: str = "microsoft/VibeVoice-AcousticTokenizer"
    dtype: torch.dtype = torch.bfloat16
    device: str | torch.device = "cuda"


class VibeVoiceTokenizer(nn.Module):
    """Thin wrapper around `VibeVoiceAcousticTokenizerModel`.

    The module is frozen (`requires_grad_(False)`) and runs in `eval()` mode.
    All public methods operate on batched tensors; padding is handled via the
    underlying feature extractor's `pad_to_multiple_of=ENCODER_STRIDE`.
    """

    def __init__(self, cfg: VibeVoiceConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or VibeVoiceConfig()

        # Late import so the login node (no transformers installed) can still
        # parse this file.
        from transformers import AutoFeatureExtractor
        from transformers.models.auto.modeling_auto import AutoModel

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.cfg.model_id)
        self.model = AutoModel.from_pretrained(
            self.cfg.model_id,
            torch_dtype=self.cfg.dtype,
            trust_remote_code=True,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, waveforms: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Encode a batch of 16 kHz mono waveforms to (B, T_latent, 64).

        Args:
            waveforms: (B, N_samples) float tensor, peak-normalised.
            lengths: optional (B,) int tensor of valid sample counts.

        Returns:
            latents: (B, T_latent, 64) in the module's dtype.
        """
        inputs = self.feature_extractor(
            [w.detach().cpu().numpy() for w in waveforms],
            sampling_rate=SAMPLE_RATE,
            pad_to_multiple_of=ENCODER_STRIDE,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(self.cfg.device, dtype=self.cfg.dtype)
        encoded = self.model.encode(input_values, sample=False)
        # Normalise return shape across VibeVoice versions.
        latents = encoded["latents"] if isinstance(encoded, dict) else encoded.latents
        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode (B, T_latent, 64) latents back to (B, N_samples) waveform."""
        latents = latents.to(self.cfg.device, dtype=self.cfg.dtype)
        decoded = self.model.decode(latents=latents)
        audio = decoded["audio"] if isinstance(decoded, dict) else decoded.audio
        return audio.float()

    @staticmethod
    def latent_frames_for_samples(n_samples: int) -> int:
        return (n_samples + ENCODER_STRIDE - 1) // ENCODER_STRIDE


def vv_encode_chunked(vv, wav, *, max_chunk_samples: int, device) -> torch.Tensor:
    """Encode a single (N,) mono waveform of arbitrary length via VibeVoice
    by splitting into ≤max_chunk_samples pieces, encoding each, and
    concatenating latents along time.

    VV's early FFN holds activations at full audio length × hidden, so a
    monolithic encode of long clips (minutes) OOMs. The encoder is
    convolutional + causal → chunked encode introduces only sub-frame boundary
    artifacts, negligible for training.
    """
    n = int(wav.shape[0])
    out = []
    with torch.no_grad():
        for s in range(0, n, max_chunk_samples):
            chunk = wav[s : s + max_chunk_samples]
            pad = (-int(chunk.shape[0])) % ENCODER_STRIDE
            if pad:
                chunk = torch.nn.functional.pad(chunk, (0, pad))
            gpu_chunk = chunk.unsqueeze(0).to(device)   # (1, N_chunk)
            lat = vv.encode(gpu_chunk)                  # (1, T_chunk, 64)
            out.append(lat[0])
    return torch.cat(out, dim=0)   # (T_total, 64)
