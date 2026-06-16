"""Tests for sequence packing in `latent_lm.data.collate`.

CPU-only — no Megatron, no NeMo container needed.
"""

from __future__ import annotations

import pytest
import torch

from latent_lm.data.collate import (
    CollateConfig,
    SpecialTokens,
    collate_batch,
    collate_packed,
    pack_example,
)


@pytest.fixture
def specials() -> SpecialTokens:
    # Vocab of 100; specials at IDs 95-99.
    return SpecialTokens.from_vocab(100)


@pytest.fixture
def cfg() -> CollateConfig:
    return CollateConfig(max_text_tokens=32, max_latent_frames=32, latent_dim=64)


def _make_example(specials, cfg, n_text: int = 5, n_lat: int = 7):
    text_ids = torch.arange(n_text, dtype=torch.long)
    latents = torch.randn(n_lat, cfg.latent_dim)
    return pack_example(text_ids=text_ids, latents=latents, specials=specials, cfg=cfg)


def test_pack_example_layout(specials, cfg):
    """Sequence is `<bos> text <BOD> latents <EOD> <eos>`; lengths add up."""
    n_text, n_lat = 5, 7
    ex = _make_example(specials, cfg, n_text, n_lat)
    L = 1 + n_text + 1 + n_lat + 1 + 1
    assert int(ex["length"]) == L
    assert ex["input_ids"][0] == specials.bos
    assert ex["input_ids"][1 + n_text] == specials.bod
    assert ex["input_ids"][2 + n_text + n_lat] == specials.eod
    assert ex["input_ids"][L - 1] == specials.eos
    # Text token IDs sit between BOS and BOD.
    assert torch.equal(ex["input_ids"][1 : 1 + n_text],
                       torch.arange(n_text, dtype=torch.long))


def test_audio_input_positions(specials, cfg):
    """`is_audio_input` is True exactly on the latent positions."""
    n_text, n_lat = 5, 7
    ex = _make_example(specials, cfg, n_text, n_lat)
    aud_start = 2 + n_text
    aud_end = aud_start + n_lat
    assert ex["is_audio_input"][aud_start:aud_end].all()
    # Everything else is False.
    pre = ex["is_audio_input"][:aud_start]
    post = ex["is_audio_input"][aud_end:]
    assert not pre.any() and not post.any()


def test_aoc_supervises_audio_positions(specials, cfg):
    """Audio-predicting positions are supervised on BOTH heads: the diffusion
    head predicts the next latent, and the LM head is trained to emit <AOC>
    ("stay in audio mode, don't fire <eod> yet"). So every audio_mask position
    carries an <AOC> text target — they are not disjoint by design."""
    ex = _make_example(specials, cfg, n_text=5, n_lat=7)
    audio_pred_pos = ex["audio_mask"]
    assert audio_pred_pos.any()
    assert (ex["text_targets"][audio_pred_pos] == specials.aoc).all()
    # Conversely, no NON-audio position carries an <AOC> target.
    assert not (ex["text_targets"][~audio_pred_pos] == specials.aoc).any()


def test_audio_target_shifted_by_one(specials, cfg):
    """Position i predicts the latent at i+1 — verify the shift."""
    n_text, n_lat = 3, 4
    ex = _make_example(specials, cfg, n_text, n_lat)
    aud_start = 2 + n_text
    # audio_targets at position [aud_start - 1 : aud_end - 1] equals input_latents at [aud_start : aud_end]
    src = ex["input_latents"][aud_start : aud_start + n_lat]
    tgt = ex["audio_targets"][aud_start - 1 : aud_start - 1 + n_lat]
    assert torch.equal(src, tgt)


def test_collate_batch_padding_to_multiple_of_16(specials, cfg):
    """Padded-batch collator rounds L_max up to a multiple of `pad_multiple`."""
    examples = [_make_example(specials, cfg, n_text=3, n_lat=5),
                _make_example(specials, cfg, n_text=7, n_lat=11)]
    batch = collate_batch(examples, specials=specials, pad_multiple=16)
    L = batch["input_ids"].shape[1]
    assert L % 16 == 0
    assert batch["input_ids"].shape[0] == 2
    # Pad positions have target == -100.
    assert (batch["text_targets"] == -100).any()


def test_collate_packed_cu_seqlens_sum_equals_total(specials, cfg):
    """cu_seqlens last entry equals total_length; differences sum to total_length."""
    examples = [_make_example(specials, cfg, n_text=3, n_lat=5),
                _make_example(specials, cfg, n_text=7, n_lat=11)]
    total = 256
    batch = collate_packed(examples, specials=specials, total_length=total)
    cu = batch["cu_seqlens"]
    assert int(cu[-1]) == total
    # All examples that fit are accounted for + a pad doc.
    diffs = (cu[1:] - cu[:-1]).tolist()
    assert sum(diffs) == total
    # Packed batch has shape (1, total).
    assert batch["input_ids"].shape == (1, total)


def test_collate_packed_no_attention_mask(specials, cfg):
    """In packed mode the block-diagonal mask is built from cu_seqlens by
    the backbone — collator must NOT ship a redundant attention_mask."""
    examples = [_make_example(specials, cfg)]
    batch = collate_packed(examples, specials=specials, total_length=256)
    assert "attention_mask" not in batch
    assert "cu_seqlens" in batch
