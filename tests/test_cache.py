"""Tests for the WebDataset-tar cache writer + reader round-trip."""

from __future__ import annotations

import pytest
import torch

from latent_lm.data.cache import CacheShardWriter, CachedDataset


def test_tar_roundtrip(tmp_path):
    """Write 5 examples, read them back, assert tensors match (within bf16 prec)."""
    writer = CacheShardWriter(
        cache_dir=str(tmp_path), shard_size=10, name_prefix="shard-r00",
        latent_dtype="bfloat16",
    )
    refs = []
    for i in range(5):
        latents = torch.randn(7 + i, 64)
        text = f"example {i} text"
        text_ids = torch.tensor([10, 20, 30 + i], dtype=torch.long)
        writer.add(latents=latents, text=text, text_ids=text_ids,
                   language="en", speaker=f"spk{i}")
        refs.append((text, text_ids, latents))
    writer.flush()

    ds = CachedDataset(cache_dir=str(tmp_path), shuffle_shards=False, shuffle_buffer=0, cycle=False)
    out = list(ds)
    assert len(out) == 5

    # Order is deterministic (no shuffle), so refs[i] matches out[i].
    for ref, ex in zip(refs, out):
        text, text_ids, lat = ref
        assert ex["text"] == text
        assert torch.equal(ex["text_ids"], text_ids)
        # bf16 round-trip introduces small quantisation error.
        assert torch.allclose(ex["latents"], lat.float(), atol=1e-2)


def test_tar_no_text_ids(tmp_path):
    """`text_ids` is optional — reader returns None when not stored."""
    writer = CacheShardWriter(cache_dir=str(tmp_path), shard_size=2,
                              name_prefix="shard-r00")
    writer.add(latents=torch.randn(3, 64), text="no ids", text_ids=None)
    writer.flush()

    ds = CachedDataset(cache_dir=str(tmp_path), shuffle_shards=False, shuffle_buffer=0, cycle=False)
    ex = next(iter(ds))
    assert ex["text"] == "no ids"
    assert ex["text_ids"] is None


def test_async_writer(tmp_path):
    """AsyncShardWriter delivers same output as sync writer."""
    from latent_lm.data.cache import AsyncShardWriter

    inner = CacheShardWriter(cache_dir=str(tmp_path), shard_size=10,
                             name_prefix="shard-r00")
    writer = AsyncShardWriter(inner, max_queue=4)
    for i in range(3):
        writer.add(latents=torch.randn(5, 64), text=f"async {i}",
                   text_ids=torch.tensor([i], dtype=torch.long))
    writer.close()

    ds = CachedDataset(cache_dir=str(tmp_path), shuffle_shards=False, shuffle_buffer=0, cycle=False)
    out = list(ds)
    assert len(out) == 3
    assert [ex["text"] for ex in out] == ["async 0", "async 1", "async 2"]
