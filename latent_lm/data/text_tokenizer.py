"""Text tokenizer wrapper that appends our 6 specials.

We take any HuggingFace BPE (GPT-2, Qwen, Llama, …), ask for its `vocab_size`,
reserve IDs `[vocab_size-6, vocab_size)` for
`<pad> <bos> <eos> <BOD> <EOD> <AOC>`, then round the total up to a multiple
of `tp` so `VocabParallelEmbedding` can shard it evenly.

The text tokenizer itself only emits IDs in `[0, vocab_size-6)` — our specials
live outside its emitted range and are inserted at packing time by
`pack_example`. `<AOC>` is the LM target at middle audio frames (see
`collate.py: pack_example`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .collate import SpecialTokens


def _round_up(n: int, k: int) -> int:
    return ((n + k - 1) // k) * k


@dataclass
class TextTokenizerConfig:
    model_id: str = "gpt2"         # swap to multilingual (Qwen, XLM-R) for real runs
    tp: int = 1
    align_to: int = 128            # vocab padding granularity (must be multiple of tp)


class TextTokenizer:
    """Minimal wrapper: encode str -> LongTensor, expose total vocab + specials."""

    def __init__(self, cfg: TextTokenizerConfig | None = None) -> None:
        from transformers import AutoConfig, AutoTokenizer

        self.cfg = cfg or TextTokenizerConfig()
        self._tok = AutoTokenizer.from_pretrained(self.cfg.model_id)
        if self._tok.pad_token is None:
            # GPT-2 style tokenizers have no pad token; we add none either — padding
            # is handled with our own <pad> special at packing time.
            pass
        # Some models (Qwen2.x) pad their embed table beyond the tokenizer's
        # vocab size. Take the max so our embedding is at least as wide as the
        # source model's, which is required if we want to load HF weights.
        tok_size = len(self._tok)
        try:
            model_vocab = int(AutoConfig.from_pretrained(self.cfg.model_id).vocab_size)
        except Exception:
            model_vocab = tok_size
        self.base_vocab = max(tok_size, model_vocab)
        n_specials = 6
        align = max(self.cfg.align_to, self.cfg.tp)
        self.vocab_size = _round_up(self.base_vocab, align)
        # Place specials at the END of the HF-native vocab range so they fit
        # inside HF's LM head (Bridge keeps vocab=base; the last 5 base slots
        # are usually reserved-but-unused HF specials, safe to repurpose).
        # Avoids the shape-mismatch we'd hit if the LM head were widened past
        # HF's `vocab_size`.
        self.specials = SpecialTokens.from_vocab(self.base_vocab)

    def encode(self, text: str, max_length: int | None = None) -> torch.Tensor:
        ids = self._tok(text, add_special_tokens=False, truncation=bool(max_length),
                        max_length=max_length).input_ids
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        ids = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
        # strip any specials that slipped in (IDs ≥ base_vocab)
        ids = [i for i in ids if i < self.base_vocab]
        return self._tok.decode(ids, skip_special_tokens=True)
