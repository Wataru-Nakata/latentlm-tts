"""Sequence packing for TTS training.

Per example we build:
    [BOS] text_tokens… [BOD] latent_frames… [EOD] [EOS]

* Text embeddings come from a token lookup table in the model.
* Latent embeddings are a `nn.Linear(64 -> H)` projection applied in the model;
  the collator only produces a (L, 64) tensor aligned with the sequence.
* `modality_mask`: 1 for text positions (incl. BOS/BOD/EOD/EOS), 0 for audio.
* `audio_mask`: True on positions that *predict* a latent (the positions whose
  hidden state feeds the diffusion head). For a sequence
  `[BOS, t0, t1, BOD, a0, a1, a2, EOD, EOS]`, the positions predicting audio
  are `[BOD, a0, a1]` — i.e. the last-but-two through BOD. We implement this
  as "positions i such that token_at_(i+1) is an audio latent".
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


SPECIAL_TOKEN_NAMES = ("<pad>", "<bos>", "<eos>", "<BOD>", "<EOD>", "<AOC>")


@dataclass
class SpecialTokens:
    pad: int
    bos: int
    eos: int
    bod: int
    eod: int
    aoc: int  # audio-continue: the LM target at middle audio frames so the
              # head learns "stay in audio mode, don't fire <eod> yet"

    @classmethod
    def from_vocab(cls, vocab_size: int) -> "SpecialTokens":
        """Assign the last 6 IDs of the vocab to our specials."""
        return cls(
            pad=vocab_size - 6,
            bos=vocab_size - 5,
            eos=vocab_size - 4,
            bod=vocab_size - 3,
            eod=vocab_size - 2,
            aoc=vocab_size - 1,
        )


@dataclass
class CollateConfig:
    max_text_tokens: int = 256
    max_latent_frames: int = 256   # 256 / 7.5 Hz ≈ 34 s cap (train on ≤20 s for now)
    latent_dim: int = 64


def pack_example(
    text_ids: torch.Tensor,           # (T_text,) long
    latents: torch.Tensor,            # (T_lat, D) float
    specials: SpecialTokens,
    cfg: CollateConfig,
) -> dict[str, torch.Tensor]:
    """Produce a single packed example (no padding)."""
    text_ids = text_ids[: cfg.max_text_tokens]
    latents = latents[: cfg.max_latent_frames]

    T_text = text_ids.numel()
    T_lat = latents.shape[0]
    L = 1 + T_text + 1 + T_lat + 1 + 1  # BOS text BOD latents EOD EOS

    input_ids = torch.full((L,), specials.pad, dtype=torch.long)
    input_latents = torch.zeros((L, cfg.latent_dim), dtype=torch.float32)
    is_audio_input = torch.zeros(L, dtype=torch.bool)  # whether THIS position's embedding is a latent

    input_ids[0] = specials.bos
    input_ids[1 : 1 + T_text] = text_ids
    input_ids[1 + T_text] = specials.bod
    # positions [2+T_text ... 2+T_text+T_lat-1] are audio latents (embedded via linear)
    aud_start = 2 + T_text
    aud_end = aud_start + T_lat
    is_audio_input[aud_start:aud_end] = True
    input_latents[aud_start:aud_end] = latents
    input_ids[aud_end] = specials.eod
    input_ids[aud_end + 1] = specials.eos

    # Targets: text CE target at position i predicts input[i+1] (shifted).
    # For audio positions, we instead use a separate audio_target tensor —
    # but we DO supervise the LM head at middle audio positions with the
    # `<AOC>` (audio-continue) special token, so the head explicitly learns
    # "stay in audio mode, don't fire <eod> here". Without this, the LM
    # output at middle audio positions is unsupervised and an under-trained
    # model emits <eod> at step 0 of inference, terminating with zero frames.
    text_targets = input_ids.roll(-1).clone()
    text_targets[-1] = -100
    # Positions where the *next* token is an audio latent (diff head owns the
    # frame prediction; LM head learns to predict <AOC> here, NOT <eod>).
    next_is_audio = torch.zeros(L, dtype=torch.bool)
    next_is_audio[aud_start - 1 : aud_end - 1] = True  # BOD and latents except last
    text_targets[next_is_audio] = specials.aoc

    audio_targets = torch.zeros((L, cfg.latent_dim), dtype=torch.float32)
    audio_targets[aud_start - 1 : aud_end - 1] = latents  # shifted by one
    audio_mask = next_is_audio.clone()

    return {
        "input_ids": input_ids,
        "input_latents": input_latents,
        "is_audio_input": is_audio_input,
        "text_targets": text_targets,
        "audio_targets": audio_targets,
        "audio_mask": audio_mask,
        "length": torch.tensor(L, dtype=torch.long),
    }

def _round_up(n: int, k: int) -> int:
    return ((n + k - 1) // k) * k


def collate_packed(
    examples: list[dict[str, torch.Tensor]],
    specials: SpecialTokens,
    *,
    total_length: int,
) -> dict[str, torch.Tensor]:
    """Concatenate examples into a single (1, total_length) packed sequence.

    Greedy pack: add examples until the next would overflow, then pad the
    remainder with a single PAD "document" (masked out of both heads).
    Returns the usual batch keys plus:

        cu_seqlens:  (N+1,) int32 — cumulative per-document boundaries, final == total_length.
        max_seqlen:  () int     — longest document length in the pack.

    `cu_seqlens` is in FlashAttention-compatible form so we can later feed
    Megatron-core's `PackedSeqParams` for variable-length attention kernels.
    For now the backbone uses a block-diagonal attention mask derived from
    `cu_seqlens`, which is correct but not as fast as the packed-attention
    kernel.
    """
    pieces: dict[str, list[torch.Tensor]] = {
        "input_ids": [], "input_latents": [], "is_audio_input": [],
        "text_targets": [], "audio_targets": [], "audio_mask": [],
    }
    seqlens: list[int] = []
    acc = 0
    for e in examples:
        L = int(e["length"])
        if acc + L > total_length:
            break
        for k in pieces:
            pieces[k].append(e[k])
        seqlens.append(L)
        acc += L

    d = examples[0]["input_latents"].shape[-1]
    pad_len = total_length - acc
    if pad_len > 0:
        pieces["input_ids"].append(torch.full((pad_len,), specials.pad, dtype=torch.long))
        pieces["input_latents"].append(torch.zeros((pad_len, d)))
        pieces["is_audio_input"].append(torch.zeros(pad_len, dtype=torch.bool))
        pieces["text_targets"].append(torch.full((pad_len,), -100, dtype=torch.long))
        pieces["audio_targets"].append(torch.zeros((pad_len, d)))
        pieces["audio_mask"].append(torch.zeros(pad_len, dtype=torch.bool))
        seqlens.append(pad_len)

    out = {k: torch.cat(v, dim=0).unsqueeze(0) for k, v in pieces.items()}
    cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(seqlens, dtype=torch.int32), dim=0)
    out["cu_seqlens"] = cu
    out["max_seqlen"] = torch.tensor(max(seqlens), dtype=torch.int32)
    # In packed mode the attention mask is derived from `cu_seqlens` inside
    # MegatronBackbone (block-diagonal causal). We do NOT ship a per-position
    # mask here because (a) it would shadow the cu_seqlens mask and (b) an
    # (L, L) tensor per worker is wasteful.
    return out


def collate_batch(
    examples: list[dict[str, torch.Tensor]],
    specials: SpecialTokens,
    *,
    pad_multiple: int = 16,
) -> dict[str, torch.Tensor]:
    """Right-pad a list of packed examples to a uniform length.

    `pad_multiple=16` covers FP8's 8-divisibility constraint on leading dims
    *and* keeps length divisible by typical TP sizes (4, 8) — important for
    sequence-parallel and FP8 GEMMs.
    """
    L_max = _round_up(max(int(e["length"]) for e in examples), pad_multiple)
    B = len(examples)
    d = examples[0]["input_latents"].shape[-1]

    input_ids = torch.full((B, L_max), specials.pad, dtype=torch.long)
    input_latents = torch.zeros((B, L_max, d))
    is_audio_input = torch.zeros(B, L_max, dtype=torch.bool)
    text_targets = torch.full((B, L_max), -100, dtype=torch.long)
    audio_targets = torch.zeros((B, L_max, d))
    audio_mask = torch.zeros(B, L_max, dtype=torch.bool)
    attention_mask = torch.zeros(B, L_max, dtype=torch.bool)

    for i, e in enumerate(examples):
        L = int(e["length"])
        input_ids[i, :L] = e["input_ids"]
        input_latents[i, :L] = e["input_latents"]
        is_audio_input[i, :L] = e["is_audio_input"]
        text_targets[i, :L] = e["text_targets"]
        audio_targets[i, :L] = e["audio_targets"]
        audio_mask[i, :L] = e["audio_mask"]
        attention_mask[i, :L] = True

    return {
        "input_ids": input_ids,
        "input_latents": input_latents,
        "is_audio_input": is_audio_input,
        "text_targets": text_targets,
        "audio_targets": audio_targets,
        "audio_mask": audio_mask,
        "attention_mask": attention_mask,
    }
