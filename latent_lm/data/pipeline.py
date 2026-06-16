"""End-to-end pipeline for TTS training. Two modes:

  * **streaming**: Emilia HF stream → audio + text. Training step must run
    VibeVoice encode on GPU each step. Suitable for first exploration.
  * **cached**: pre-encoded latents from `scripts/cache_latents.py` (shard
    files under `cache_dir`). Training step skips VibeVoice encode entirely.

`make_dataloader()` auto-selects based on whether `PipelineConfig.cache_dir`
is set. Batch dicts differ between modes:

    streaming: {text_ids: list, audio: (B, N), audio_lens: (B,)}
    cached:    {text_ids: list, latents: list[(T_lat, 64)]}

Both are handled in `train._pack_batch`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset

from .cache import CachedDataset
from .collate import (
    CollateConfig, SpecialTokens, collate_packed, pack_example,
)
from .emilia import EmiliaConfig, build_emilia_stream, iterate_examples
from .text_tokenizer import TextTokenizer
from ..models.tokenizer import LATENT_SCALE


@dataclass
class PipelineConfig:
    emilia: EmiliaConfig = field(default_factory=EmiliaConfig)
    max_text_tokens: int = 256
    # Audio is padded per-batch to a multiple of `encoder_stride` to satisfy
    # VibeVoice's input requirement (see models/tokenizer.py:ENCODER_STRIDE).
    audio_encoder_stride: int = 3200
    # DataLoader wrappers — use num_workers>=1 so network I/O overlaps.
    num_workers: int = 2
    prefetch_factor: int = 4
    batch_size: int = 4
    # Enables cached mode when set to a directory with shard-*.pt files.
    cache_dir: str | None = None
    # Optional dedicated validation cache dir (read in full for eval). When set,
    # the eval dataloader uses this instead of an eval_holdout_frac tail slice.
    valid_cache_dir: str | None = None
    # When set (cached mode), the DataLoader workers run pack_example +
    # collate_packed too — the main process receives a fully-collated batch
    # ready for H2D, eliminating per-step Python work between forwards.
    specials: SpecialTokens | None = None
    pack_cfg: CollateConfig | None = None
    packed_total_length: int = 0
    # Hold out the last `eval_holdout_frac` of shards as a fixed eval set
    # (cached mode only). is_eval=True picks the held-out slice; False excludes it.
    eval_holdout_frac: float = 0.0
    is_eval: bool = False
    min_audio_seconds: float = 2.0
    max_audio_seconds: float = 30.0


class _StreamingDataset(IterableDataset):
    def __init__(self, cfg: PipelineConfig, tokenizer: TextTokenizer, *, rank: int, world_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[dict]:
        stream = build_emilia_stream(self.cfg.emilia, rank=self.rank, world_size=self.world_size)
        for ex in iterate_examples(stream, self.cfg.emilia,
                                   rank=self.rank, world_size=self.world_size):
            text = ex["text"]
            if not text:
                continue
            text_ids = self.tokenizer.encode(text, max_length=self.cfg.max_text_tokens)
            if text_ids.numel() == 0:
                continue
            yield {
                "text_ids": text_ids,
                "audio": ex["audio"],
                "audio_len": ex["audio"].shape[0],
            }


class _CachedTokenizedDataset(IterableDataset):
    """Wrap `CachedDataset` so it also tokenises text on the fly.

    When `cfg.pack_cfg` and `cfg.specials` are set, also runs `pack_example`
    in the worker — the main process then only sees per-example *packed* dicts
    (input_ids, input_latents, masks, …) ready for `collate_packed`. This is
    the fast path; without it the main process pays per-step Python cost
    between forwards and the GPU goes idle.
    """

    def __init__(self, cfg: PipelineConfig, tokenizer: TextTokenizer, *, rank: int, world_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.inner = CachedDataset(
            cache_dir=cfg.cache_dir, rank=rank, world_size=world_size,
            eval_holdout_frac=cfg.eval_holdout_frac, is_eval=cfg.is_eval)

    def __iter__(self) -> Iterator[dict]:
        do_pack = self.cfg.pack_cfg is not None and self.cfg.specials is not None
        if not do_pack:
            for ex in self.inner:
                pre = ex.get("text_ids")
                if pre is not None:
                    text_ids = pre[: self.cfg.max_text_tokens] if pre.numel() else pre
                else:
                    text_ids = self.tokenizer.encode(
                        ex["text"], max_length=self.cfg.max_text_tokens)
                if text_ids.numel() == 0:
                    continue
                # Scale to ~unit variance for DDPM (cached file holds raw VV
                # latents at native std≈5.12; train on x0 ≈ N(0,1)).
                yield {"text_ids": text_ids, "latents": ex["latents"] * LATENT_SCALE}
            return

        # Pack-into-shards path. We accumulate per-example pack_example dicts
        # until they fill `packed_total_length` tokens, then yield a single
        # `collate_packed` output. This way the downstream collate_fn just
        # CONCATENATES yielded packs side-by-side instead of dropping examples
        # past the first pack (the old behaviour, which made micro_batch_size a
        # no-op and capped real batch at 1×packed_total_length tokens).
        specials = self.cfg.specials
        pack_cfg = self.cfg.pack_cfg
        total_length = self.cfg.packed_total_length
        buf: list[dict] = []
        buf_acc = 0
        for ex in self.inner:
            pre = ex.get("text_ids")
            if pre is not None:
                text_ids = pre[: self.cfg.max_text_tokens] if pre.numel() else pre
            else:
                text_ids = self.tokenizer.encode(
                    ex["text"], max_length=self.cfg.max_text_tokens)
            if text_ids.numel() == 0:
                continue
            packed = pack_example(
                text_ids=text_ids,
                # Scale to ~unit variance — DDPM ε-prediction needs x0 ~ N(0,1).
                latents=ex["latents"].float() * LATENT_SCALE,
                specials=specials,
                cfg=pack_cfg,
            )
            L = int(packed["length"])
            if L > total_length:
                continue  # single doc longer than a whole pack — drop
            if buf_acc + L > total_length and buf:
                yield collate_packed(buf, specials=specials, total_length=total_length)
                buf = []
                buf_acc = 0
            buf.append(packed)
            buf_acc += L
        if buf:
            yield collate_packed(buf, specials=specials, total_length=total_length)


def _collate_streaming(batch: list[dict], *, audio_encoder_stride: int) -> dict:
    lens = torch.tensor([b["audio_len"] for b in batch], dtype=torch.long)
    max_len = int(lens.max().item())
    pad_to = ((max_len + audio_encoder_stride - 1) // audio_encoder_stride) * audio_encoder_stride
    B = len(batch)
    audio = torch.zeros((B, pad_to), dtype=torch.float32)
    for i, b in enumerate(batch):
        a = b["audio"]
        audio[i, : a.shape[0]] = a
    return {
        "text_ids": [b["text_ids"] for b in batch],
        "audio": audio,
        "audio_lens": lens,
    }


def _collate_cached(batch: list[dict]) -> dict:
    return {
        "text_ids": [b["text_ids"] for b in batch],
        "latents": [b["latents"] for b in batch],
    }


def _collate_cached_packed(batch: list[dict], *, specials: SpecialTokens,
                           total_length: int) -> dict:
    """Concatenate N already-packed shards (each (1, total_length, ...)) into
    a single (1, N*total_length, ...) packed sequence with combined cu_seqlens.

    Each item in `batch` is the output of a per-worker `collate_packed` call,
    so its tensors carry a leading batch dim of 1 and span exactly
    `total_length` tokens. We cat along the seq dim and shift each pack's
    cu_seqlens by the cumulative offset — backbone sees one big packed
    sequence with block-diagonal mask, no model-side change required.
    """
    import torch as _torch
    if len(batch) == 1:
        return batch[0]
    out: dict = {}
    for k in ("input_ids", "input_latents", "is_audio_input",
              "text_targets", "audio_targets", "audio_mask"):
        out[k] = _torch.cat([b[k] for b in batch], dim=1)
    cu_pieces = [_torch.tensor([0], dtype=_torch.int32)]
    for i, b in enumerate(batch):
        cu = b["cu_seqlens"]
        cu_pieces.append(cu[1:] + i * total_length)
    out["cu_seqlens"] = _torch.cat(cu_pieces)
    out["max_seqlen"] = _torch.tensor(
        max(int(b["max_seqlen"]) for b in batch), dtype=_torch.int32)
    return out


def make_dataloader(cfg: PipelineConfig, tokenizer: TextTokenizer, *, rank: int = 0, world_size: int = 1) -> DataLoader:
    if cfg.cache_dir:
        ds = _CachedTokenizedDataset(cfg, tokenizer, rank=rank, world_size=world_size)
        if cfg.pack_cfg is not None and cfg.specials is not None and cfg.packed_total_length:
            specials = cfg.specials
            total_length = cfg.packed_total_length

            def collate(batch):
                return _collate_cached_packed(batch, specials=specials,
                                              total_length=total_length)
        else:
            collate = _collate_cached
    else:
        ds = _StreamingDataset(cfg, tokenizer, rank=rank, world_size=world_size)

        def collate(batch):
            return _collate_streaming(batch, audio_encoder_stride=cfg.audio_encoder_stride)

    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        collate_fn=collate,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=(cfg.num_workers > 0),
        pin_memory=True,
    )
