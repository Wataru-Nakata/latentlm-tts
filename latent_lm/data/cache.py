"""Pre-encoded Emilia latent cache.

**Shard format: WebDataset tar.** Each sample is two files in the tar:

    <key>.npy   — VibeVoice latents (bf16 numpy array, shape (T_lat, 64))
    <key>.json  — {"text", "text_ids", "language", "speaker"}

Tar shards are named `<name_prefix>-<idx:05d>.tar` so multiple parallel
cache jobs can write disjoint files into the same dir without collision.

Reader uses `webdataset.WebDataset` for sequential streaming with
shard-level shuffle + per-rank sharding. A pickle-format reader is kept as
a backward-compat path for shards built before the migration.
"""

from __future__ import annotations

import glob
import io
import json
import os
import queue as _queue
import random
import tarfile
import threading
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


@dataclass
class CacheShardWriter:
    """Streaming WebDataset tar writer.

    Each call to `add(...)` appends one sample (two tar entries: latents.npy +
    metadata.json). After `shard_size` samples, the current tar is closed and
    a new one is opened.
    """

    cache_dir: str
    shard_size: int = 1000
    start_index: int = 0
    name_prefix: str = "shard"
    # bf16 halves cache size with no perceptible quality impact (VibeVoice
    # encodes in bf16 internally; latents are normalised to ~unit variance).
    latent_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        self._shard_idx = self.start_index
        self._sample_in_shard = 0
        self._tar: tarfile.TarFile | None = None

    def _open_shard(self) -> None:
        path = os.path.join(self.cache_dir,
                            f"{self.name_prefix}-{self._shard_idx:05d}.tar")
        self._tar = tarfile.open(path, "w")
        self._sample_in_shard = 0

    def _add_member(self, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        self._tar.addfile(info, io.BytesIO(data))

    def add(
        self,
        *,
        latents: torch.Tensor,
        text: str,
        text_ids: torch.Tensor | None = None,
        language: str = "",
        speaker: str | None = None,
        source_url: str = "",
    ) -> None:
        if self._tar is None:
            self._open_shard()
        key = f"{self._shard_idx:05d}_{self._sample_in_shard:06d}"

        # Latents → bf16 numpy → npy bytes. We store as bf16 view through a
        # uint16 dtype because numpy <2.0 doesn't have native bf16; we record
        # `dtype: bfloat16` in the JSON so the reader can reinterpret.
        lat = latents.detach().to(torch.bfloat16 if self.latent_dtype == "bfloat16"
                                  else torch.float32).cpu().contiguous()
        if self.latent_dtype == "bfloat16":
            arr = lat.view(torch.uint16).numpy()
        else:
            arr = lat.numpy()
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        self._add_member(f"{key}.npy", buf.getvalue())

        meta: dict = {
            "text": text,
            "language": language,
            "speaker": speaker,
            "shape": list(lat.shape),
            "dtype": self.latent_dtype,
            "source_url": source_url,
        }
        if text_ids is not None:
            meta["text_ids"] = text_ids.detach().cpu().tolist()
        self._add_member(f"{key}.json",
                         json.dumps(meta, ensure_ascii=False).encode("utf-8"))

        self._sample_in_shard += 1
        if self._sample_in_shard >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None
            self._shard_idx += 1
            self._sample_in_shard = 0

    @property
    def n_shards_written(self) -> int:
        return self._shard_idx - self.start_index


class AsyncShardWriter:
    """Background-thread wrapper around `CacheShardWriter`.

    The main thread calls `add()` which enqueues a sample. A worker thread
    drains the queue and calls the underlying writer's `add()` (which may
    trigger a tar flush on shard boundaries). Bounded queue (`max_queue`)
    backpressures the producer so memory doesn't grow unbounded.

    Use this when the writer's I/O cost is non-negligible — e.g., 3 GB shards
    where closing a tar means flushing GBs of buffered data.
    """

    _STOP = object()

    def __init__(self, inner: "CacheShardWriter", max_queue: int = 16) -> None:
        self.inner = inner
        self._q: _queue.Queue = _queue.Queue(maxsize=max_queue)
        self._exc: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=False, name="cache-writer")
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                item = self._q.get()
                try:
                    if item is self._STOP:
                        return
                    # Items are (method_name, kwargs) tuples — lets the same
                    # queue adds on the inner writer from a background thread.
                    method_name, kwargs = item
                    getattr(self.inner, method_name)(**kwargs)
                finally:
                    self._q.task_done()
        except Exception as e:
            # Capture so the main thread sees it on the next add() call.
            # Deliberately Exception (not BaseException) so KeyboardInterrupt /
            # SystemExit propagate naturally and don't get silently swallowed.
            self._exc = e

    def _check(self) -> None:
        if self._exc is not None:
            exc, self._exc = self._exc, None
            raise exc

    def add(self, **kwargs) -> None:
        self._check()
        self._q.put(("add", kwargs))

    def flush(self) -> None:
        """Block until all queued adds are written, then close current tar."""
        self._q.join()
        self._check()
        self.inner.flush()

    @property
    def n_shards_written(self) -> int:
        return self.inner.n_shards_written

    def close(self) -> None:
        """Drain queue, signal stop, join the writer thread."""
        self._q.put(self._STOP)
        self._q.join()
        self._thread.join()
        self._check()
        self.inner.flush()


def list_shards(cache_dir: str) -> list[str]:
    """Return sorted list of shard files in `cache_dir`. Includes both .tar
    (current) and .pt (legacy) so a mixed directory works. Also recurses
    one level into `rank*/` subdirs
    one rank per subdir to parallelise the encode)."""
    direct = (glob.glob(os.path.join(cache_dir, "*.tar"))
              + glob.glob(os.path.join(cache_dir, "*.pt")))
    rank_subdir = (glob.glob(os.path.join(cache_dir, "rank*", "*.tar"))
                   + glob.glob(os.path.join(cache_dir, "rank*", "*.pt")))
    return sorted(direct + rank_subdir)


def _decode_npy_to_tensor(data: bytes, dtype: str) -> torch.Tensor:
    arr = np.load(io.BytesIO(data), allow_pickle=False)
    if dtype == "bfloat16":
        # arr was stored as uint16; reinterpret as bf16.
        return torch.from_numpy(arr).view(torch.bfloat16)
    return torch.from_numpy(arr)


class CachedDataset(IterableDataset):
    """WebDataset-backed iterable. Falls back to legacy pickle shards when
    encountered (so a directory with mixed `.tar` / `.pt` files works).

    Shards are split per DDP rank deterministically; within a rank the shard
    order is shuffled per epoch and within-shard examples flow in tar order
    (sequential disk reads — no random seeks).
    """

    def __init__(
        self,
        cache_dir: str,
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        shuffle_shards: bool = True,
        shuffle_buffer: int = 1000,
        eval_holdout_frac: float = 0.0,
        is_eval: bool = False,
        cycle: bool = True,
    ) -> None:
        super().__init__()
        self.cache_dir = cache_dir
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle_shards = shuffle_shards
        self.shuffle_buffer = shuffle_buffer
        # cycle=True (default) loops epochs forever — required for DDP training so
        # the iterator never exhausts mid-step. cycle=False does a single pass and
        # stops (use for eval / inspection / `list(ds)`).
        self.cycle = cycle
        all_shards = list_shards(cache_dir)
        if not all_shards:
            raise FileNotFoundError(f"no shards under {cache_dir}")
        # Hold out the last `eval_holdout_frac` of shards as a fixed eval set.
        # Train and eval use disjoint shards so eval loss is never the same data
        # as training, and they don't share dataloader iterator state.
        if eval_holdout_frac > 0:
            n_eval = max(1, int(len(all_shards) * eval_holdout_frac))
            all_shards = all_shards[-n_eval:] if is_eval else all_shards[:-n_eval]
        self._shards = all_shards[rank::world_size]

    # ------------------------------------------------------------------
    # WebDataset (.tar) path
    # ------------------------------------------------------------------

    def _iter_tar_shard(self, path: str, rng: random.Random) -> Iterator[dict]:
        # Sequential read of tar; pair `.npy` and `.json` by basename.
        with tarfile.open(path, "r|") as tf:
            buffered: dict[str, dict] = {}
            for member in tf:
                if not member.isfile():
                    continue
                base, _, ext = member.name.rpartition(".")
                if not base or ext not in ("npy", "json"):
                    continue
                f = tf.extractfile(member)
                if f is None:
                    continue
                data = f.read()
                bucket = buffered.setdefault(base, {})
                bucket[ext] = data
                if "npy" in bucket and "json" in bucket:
                    meta = json.loads(bucket["json"].decode("utf-8"))
                    latents = _decode_npy_to_tensor(bucket["npy"],
                                                    dtype=meta.get("dtype", "float32"))
                    yield {
                        "text": meta.get("text", ""),
                        "text_ids": (torch.tensor(meta["text_ids"], dtype=torch.long)
                                     if "text_ids" in meta else None),
                        "latents": latents.float(),
                        "language": meta.get("language", ""),
                        "speaker": meta.get("speaker"),
                        "source_url": meta.get("source_url", ""),
                    }
                    buffered.pop(base, None)

    # ------------------------------------------------------------------
    # Legacy pickle (.pt) path — backward compat with pre-migration shards
    # ------------------------------------------------------------------

    def _iter_pickle_shard(self, path: str, rng: random.Random) -> Iterator[dict]:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        n = len(blob["texts"])
        text_ids_all = blob.get("text_ids")
        order = list(range(n))
        rng.shuffle(order)
        for i in order:
            yield {
                "text": blob["texts"][i],
                "text_ids": text_ids_all[i] if text_ids_all is not None else None,
                "latents": blob["latents"][i].float(),
                "language": blob["languages"][i] if "languages" in blob else "",
                "speaker": blob["speakers"][i] if "speakers" in blob else None,
            }

    def _open_shard(self, path: str, rng: random.Random) -> Iterator[dict]:
        """Subclass hook — subclasses may override this to use its own
        tar parser. Default routes to legacy paths."""
        inner = (self._iter_tar_shard if path.endswith(".tar")
                 else self._iter_pickle_shard)
        return inner(path, rng)

    def __iter__(self) -> Iterator[dict]:
        # Per-rank shards must be sliced again per DataLoader worker — otherwise
        # all `num_workers` worker processes iterate the same shard list with
        # the same RNG and yield identical batches in round-robin, producing a
        # `num_workers`-period sawtooth in the loss as the model overfits each
        # repeated batch.
        wi = get_worker_info()
        local_w = wi.id if wi is not None else 0
        local_n = wi.num_workers if wi is not None else 1
        my_shards = self._shards[local_w::local_n]
        # If this worker got no shards (e.g., eval with few shards & many
        # workers), exit cleanly instead of spinning the infinite cycle below
        # with an empty list — which would block DataLoader's batching forever.
        if not my_shards:
            return
        # Independent RNG per (rank, worker) so the within-rank shuffles diverge.
        rng = random.Random(self.seed + self.rank * 10_000 + local_w)

        def _open(path: str):
            return self._open_shard(path, rng)

        # Read N shards interleaved within a single worker. With huge shards
        # (e.g. v4's ~28K examples each) reading sequentially means consecutive
        # yields all come from the same shard — content correlations inside a
        # shard then leak into batch-to-batch loss as a periodic sawtooth (the
        # period equals the number of workers in DataLoader rotation). Round-
        # robin across N open shards breaks the correlation: adjacent yields
        # come from N different shards. N=4 halved the spike on v4 but didn't
        # kill it; N=16 makes per-worker yields nearly i.i.d. across shards.
        # Cost is N open file handles per worker.
        N_PARALLEL = 16

        # Infinite cycle over the worker's shard slice. Without this the iterator
        # exhausts after one epoch, the DataLoader stops yielding for that
        # worker, the DDP all-reduce on the next step hangs because some ranks
        # have a batch and others don't, and NCCL eventually aborts the job.
        # Each epoch the shard order is re-shuffled with a fresh seed.
        epoch = 0
        while True:
            shards = list(my_shards)
            if self.shuffle_shards:
                epoch_rng = random.Random(self.seed + self.rank * 10_000 + local_w + epoch * 1_000_003)
                epoch_rng.shuffle(shards)

            shard_queue = list(shards)
            active = []
            for _ in range(min(N_PARALLEL, len(shard_queue))):
                active.append(_open(shard_queue.pop(0)))

            buf: list[dict] = []
            while active:
                # Random shard per yield — avoids any periodic structure that
                # round-robin would leave (period = N_PARALLEL within a worker).
                idx = rng.randrange(len(active))
                try:
                    ex = next(active[idx])
                    buf.append(ex)
                    if len(buf) >= self.shuffle_buffer:
                        j = rng.randrange(len(buf))
                        yield buf[j]
                        buf[j] = buf[-1]
                        buf.pop()
                except StopIteration:
                    if shard_queue:
                        active[idx] = _open(shard_queue.pop(0))
                    else:
                        active.pop(idx)

            rng.shuffle(buf)
            yield from buf
            epoch += 1
            if not self.cycle:
                return


