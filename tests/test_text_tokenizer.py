"""Tests for the text tokenizer wrapper (vocab alignment, specials placement)."""

from __future__ import annotations

import pytest

from latent_lm.data.collate import SpecialTokens


@pytest.mark.parametrize("vocab_size", [100, 50304, 152064])
def test_specials_at_end_of_vocab(vocab_size):
    """Specials always occupy the last 6 IDs (pad/bos/eos/bod/eod/aoc)."""
    s = SpecialTokens.from_vocab(vocab_size)
    assert s.pad == vocab_size - 6
    assert s.bos == vocab_size - 5
    assert s.eos == vocab_size - 4
    assert s.bod == vocab_size - 3
    assert s.eod == vocab_size - 2
    assert s.aoc == vocab_size - 1
    # All distinct.
    ids = {s.pad, s.bos, s.eos, s.bod, s.eod, s.aoc}
    assert len(ids) == 6


def test_text_tokenizer_vocab_padding():
    """Padded vocab is a multiple of `align_to` and ≥ base + 5."""
    pytest.importorskip("transformers")
    from latent_lm.data.text_tokenizer import TextTokenizer, TextTokenizerConfig

    tok = TextTokenizer(TextTokenizerConfig(model_id="gpt2", tp=4, align_to=128))
    assert tok.vocab_size % 128 == 0
    assert tok.vocab_size >= tok.base_vocab + 5
    # GPT-2 base vocab is 50257; padded to next multiple of 128 = 50304.
    assert tok.vocab_size == 50304


def test_text_tokenizer_encode_decode_roundtrip():
    """Encode → decode preserves the input text (modulo tokenizer-specific normalisation)."""
    pytest.importorskip("transformers")
    from latent_lm.data.text_tokenizer import TextTokenizer, TextTokenizerConfig

    tok = TextTokenizer(TextTokenizerConfig(model_id="gpt2"))
    text = "Hello, world."
    ids = tok.encode(text)
    assert ids.dtype.is_signed and ids.numel() > 0
    decoded = tok.decode(ids)
    assert decoded.strip() == text.strip()
