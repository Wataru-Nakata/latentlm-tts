"""Streaming Emilia dataloader.

Yields `{"text": str, "audio": FloatTensor[N_samples], "language": str, ...}`.

We stream the Amphion Emilia release directly from HF and resample on the fly
to 16 kHz mono (VibeVoice's required rate). No offline pre-tokenization step —
VibeVoice encoding runs in the training loop on GPU.

Because Emilia is an `IterableDataset` in streaming mode, we shuffle with a
shuffle buffer and shard across DDP ranks via `datasets.distributed.split_dataset_by_node`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import torch


@dataclass
class EmiliaConfig:
    dataset_id: str = "amphion/Emilia-Dataset"
    # HF config name. `amphion/Emilia-Dataset` has only 'default'; leave None
    # for that. Other Emilia forks expose per-language configs like 'EN'/'ZH'.
    config_name: str | None = None
    split: str = "train"
    languages: tuple[str, ...] = ()  # empty = don't filter by language column
    max_audio_seconds: float = 20.0
    min_audio_seconds: float = 1.0
    # Streaming-only diversity knob. HF `IterableDataset.shuffle` shuffles shard
    # ORDER *and* keeps a `shuffle_buffer` of records; cross-shard mixing only
    # reaches as far as the buffer spans. The cached path interleaves 16 shards
    # + a buffer for near-i.i.d. batches (see CachedDataset.N_PARALLEL); to
    # approximate that here we use a larger buffer so it spans several shards.
    # It carries raw (lazily-decoded) audio records, so bigger = more worker RAM.
    # NOTE: this only aligns shuffle *diversity*, not throughput — streaming can
    # still starve the GPU and overfit a narrow slice; caching is the real fix.
    shuffle_buffer: int = 5_000
    seed: int = 0
    # HF streaming doesn't cache raw audio; this points at a writable scratch
    # dir on the compute node for metadata caches.
    hf_cache_dir: str | None = None
    # If set, load Emilia from a local snapshot (downloaded via
    # huggingface-cli or `scripts/pbs/download_emilia.pbs`). Eliminates the
    # streaming HF download bandwidth cost per worker.
    local_data_dir: str | None = None
    extra_load_kwargs: dict = field(default_factory=dict)


def build_emilia_stream(cfg: EmiliaConfig, *, rank: int = 0, world_size: int = 1):
    """Return a streaming, per-rank sharded Emilia IterableDataset.

    Two source modes:
      * **HF stream** (default) — `load_dataset("amphion/Emilia-Dataset", streaming=True)`.
        Network-bound on HF's CDN.
      * **Local snapshot** (when `cfg.local_data_dir` is set) — point HF's
        `webdataset` builder at the tars on disk. Layout is
        `<local_data_dir>/Emilia/<LANG>/*.tar`, mirroring the HF repo. We glob
        per requested language so the language filter is applied at file-list
        time (cheap) instead of per-record (expensive).

    Schema yielded in both modes (confirmed 2026-04-25):
        __key__: str
        __url__: str
        json:    {_id, dnsmos, duration, language, phone_count, speaker, text}
        mp3:     Audio(array, sampling_rate, path)  — decoded by Audio cast
    """
    import glob as _glob
    import os as _os

    from datasets import Audio, load_dataset
    from datasets.distributed import split_dataset_by_node

    load_kwargs = dict(cfg.extra_load_kwargs)
    if cfg.config_name:
        load_kwargs.setdefault("name", cfg.config_name)

    if cfg.local_data_dir:
        # Local mode — webdataset builder over our snapshot tars. The
        # `amphion/Emilia-Dataset` repo bundles BOTH the original Emilia
        # corpus AND Emilia-YODAS as sibling subdirs (each laid out as
        # <SUBSET>/<LANG>/*.tar). Glob both so neither is silently dropped.
        langs = [l.upper() for l in cfg.languages] if cfg.languages else \
            ["EN", "ZH", "JA", "KO", "DE", "FR"]
        subsets = ["Emilia", "Emilia-YODAS"]
        files: list[str] = []
        for subset in subsets:
            for lang in langs:
                files.extend(sorted(_glob.glob(
                    _os.path.join(cfg.local_data_dir, subset, lang, "*.tar"))))
        if not files:
            raise FileNotFoundError(
                f"no tars under {cfg.local_data_dir}/{{{','.join(subsets)}}}/"
                f"{{{','.join(langs)}}}")
        ds = load_dataset(
            "webdataset",
            data_files={cfg.split: files},
            split=cfg.split,
            streaming=True,
            cache_dir=cfg.hf_cache_dir,
        )
    else:
        # HF stream mode.
        ds = load_dataset(
            cfg.dataset_id,
            split=cfg.split,
            streaming=True,
            cache_dir=cfg.hf_cache_dir,
            **load_kwargs,
        )
    # Resample audio to 24 kHz mono (VibeVoice's feature extractor expects 24 kHz).
    # Column name is "mp3", not "audio".
    ds = ds.cast_column("mp3", Audio(sampling_rate=24_000, mono=True))

    ds = ds.shuffle(buffer_size=cfg.shuffle_buffer, seed=cfg.seed)
    ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)
    return ds


def iterate_examples(ds, cfg: EmiliaConfig, *, rank: int = 0, world_size: int = 1,
                     max_retries: int = 10) -> Iterator[dict]:
    """Flatten Emilia records to `{text, audio, language, speaker}` dicts.

    Cheap filters (language, duration) run BEFORE touching `mp3.array` so
    rejected records don't pay audio-decode cost. Malformed records are
    silently skipped.

    Resilient to transient HF stream failures: on `httpx.ReadTimeout` or other
    iterator errors we rebuild `ds` and continue. Some records may be re-seen
    after a retry — that's fine for our use case (cached latents are not
    expected to be a unique-record set; downstream training reshuffles).
    """
    import time as _time

    allowed = {l.lower() for l in cfg.languages} if cfg.languages else None

    def _emit(ex):
        meta = ex.get("json") or {}
        lang = (meta.get("language") or "").lower()
        if allowed and lang not in allowed:
            return None
        duration = meta.get("duration")
        if duration is None or duration < cfg.min_audio_seconds or duration > cfg.max_audio_seconds:
            return None
        text = meta.get("text") or ""
        if not text:
            return None
        mp3 = ex.get("mp3")
        if not isinstance(mp3, dict):
            return None
        arr = mp3.get("array")
        if arr is None:
            return None
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        return {
            "text": text,
            "audio": torch.from_numpy(arr).float(),
            "language": lang,
            "speaker": meta.get("speaker"),
        }

    retries = 0
    current = ds
    iterator = iter(current)
    while True:
        try:
            ex = next(iterator)
        except StopIteration:
            return
        except Exception as e:
            retries += 1
            if retries > max_retries:
                raise
            sleep_s = min(60, 2 ** retries)
            print(f"[emilia] stream error (retry {retries}/{max_retries}): "
                  f"{type(e).__name__}: {e}; sleeping {sleep_s}s and rebuilding stream",
                  flush=True)
            _time.sleep(sleep_s)
            current = build_emilia_stream(cfg, rank=rank, world_size=world_size)
            iterator = iter(current)
            continue
        out = _emit(ex)
        if out is None:
            continue
        # _emit doesn't have access to ex.__url__ — re-attach here so the cache
        # writer can record provenance per example.
        out.setdefault("source_url", ex.get("__url__") or ex.get("__key__") or "")
        yield out
