"""Pre-encode Emilia audio into VibeVoice latent shards.

Speed-ups vs. the v0 single-process loop:
  * `--world-size`/`--rank`     — N parallel cache jobs cover disjoint stream
                                  slices via `split_dataset_by_node`. Each job
                                  writes shards under its own filename prefix
                                  so they can't collide.
  * `--num-workers`             — torch DataLoader workers, each opens its
                                  own HF stream (so N parallel TCP downloads
                                  per job) and decodes mp3 → 24 kHz mono on
                                  CPU. Decoded audio is queued; the main
                                  process batches it and runs VibeVoice on GPU.
  * `--prefetch-factor`         — overlaps download/decode with GPU encode.

Run inside the SIF on a single GPU:
    python -m scripts.cache_latents \\
        --cache-dir /groups/.../latentLM/cache/emilia_en \\
        --rank 0 --world-size 8 --num-workers 8 \\
        --languages en --max-shards 1000 --shard-size 1000

For N parallel jobs: submit `cache_emilia.pbs` N times with different RANK and
matching WORLD_SIZE; each job streams 1/N of Emilia.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--languages", nargs="*", default=["en"])
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--max-shards", type=int, default=200,
                        help="Per-rank cap on shards.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="VibeVoice encode batch size.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--text-tokenizer", default="Qwen/Qwen2.5-0.5B",
                        help="Must match tokenizer.text in your train config.")
    parser.add_argument("--rank", type=int, default=0,
                        help="Job rank for sharded parallel cache builds.")
    parser.add_argument("--world-size", type=int, default=1,
                        help="Total number of parallel cache jobs.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers per job (parallel HF streams).")
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--bucket-buffer", type=int, default=256,
                        help="Buffer N samples then sort by audio length and chunk "
                             "into encode batches of similar length. Reduces padding "
                             "waste from VibeVoice encode (O(L²) attention). 0 = off.")
    parser.add_argument("--local-data-dir", default=None,
                        help="Path to a local Emilia snapshot (skips HF download).")
    args = parser.parse_args()

    from latent_lm.data.cache import AsyncShardWriter, CacheShardWriter
    from latent_lm.data.emilia import EmiliaConfig, build_emilia_stream, iterate_examples
    from latent_lm.models.tokenizer import ENCODER_STRIDE, VibeVoiceTokenizer

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    em_cfg = EmiliaConfig(
        languages=tuple(args.languages),
        min_audio_seconds=args.min_seconds,
        max_audio_seconds=args.max_seconds,
        shuffle_buffer=100,
        local_data_dir=args.local_data_dir,
    )

    class _EmiliaStream(IterableDataset):
        """Per-job Emilia stream slice. We pass JOB-LEVEL rank/world only; HF
        datasets' IterableDataset reads `worker_info` itself and further splits
        across DataLoader workers internally. Subdividing again here would
        double-shrink each worker's slice (observed: rank gets 1/(N*W) instead
        of 1/N — so most of the corpus is silently dropped)."""

        def __iter__(self) -> Iterator[dict]:
            stream = build_emilia_stream(em_cfg, rank=args.rank, world_size=args.world_size)
            for ex in iterate_examples(stream, em_cfg, rank=args.rank,
                                       world_size=args.world_size):
                yield ex

    def _collate(batch: list[dict]) -> dict:
        return {
            "audio": [b["audio"] for b in batch],
            "text": [b["text"] for b in batch],
            "language": [b.get("language", "") for b in batch],
            "speaker": [b.get("speaker") for b in batch],
            "source_url": [b.get("source_url", "") for b in batch],
        }

    loader = DataLoader(
        _EmiliaStream(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=_collate,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=(args.num_workers > 0),
        pin_memory=True,
    )

    vv = VibeVoiceTokenizer().to("cuda")
    print(f"[cache] VibeVoice loaded  rank={args.rank}/{args.world_size}  "
          f"workers={args.num_workers}", flush=True)
    print(f"[cache] storing raw text only (no pre-tokenisation — backbone tokenizer "
          f"is a training-time choice)", flush=True)

    writer = AsyncShardWriter(
        CacheShardWriter(
            cache_dir=args.cache_dir,
            shard_size=args.shard_size,
            start_index=args.start_index,
            name_prefix=f"shard-r{args.rank:02d}",
        ),
        max_queue=16,
    )

    n_examples = 0
    t0 = time.time()
    last_log = time.time()

    # Per-stage timers. Each entry accumulated_ms / n_batches => avg ms per batch.
    stage_total = {"read": 0.0, "prep": 0.0, "encode": 0.0, "d2h": 0.0, "write": 0.0}
    n_batches = 0

    # Background NVML sampler — polls GPU + memory util every 100 ms so we can
    # report avg / max between log lines. Daemon thread; dies with process.
    _nvml_samples_gpu: list[int] = []
    _nvml_samples_mem: list[int] = []
    _nvml_lock = __import__("threading").Lock()
    try:
        import pynvml as _pynvml
        _pynvml.nvmlInit()
        _nvml_handle = _pynvml.nvmlDeviceGetHandleByIndex(
            int(os.environ.get("LOCAL_RANK", 0)))

        def _nvml_loop() -> None:
            while True:
                try:
                    u = _pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle)
                    with _nvml_lock:
                        _nvml_samples_gpu.append(u.gpu)
                        _nvml_samples_mem.append(u.memory)
                except Exception:
                    return
                time.sleep(0.1)

        __import__("threading").Thread(target=_nvml_loop, daemon=True).start()
        _nvml_ok = True
    except Exception as _e:
        print(f"[cache] NVML unavailable ({_e}); skipping GPU util logging", flush=True)
        _nvml_ok = False

    # ---- length-bucketed iteration --------------------------------------
    # Buffer `bucket_buffer` samples, sort by audio length, then emit
    # `batch_size` chunks of similar-length samples. This shrinks the longest
    # sample in each encode batch → much less zero-padding → much less wasted
    # O(L²) attention compute inside VibeVoice. Refill when buffer empties.

    def _bucketed_batches():
        loader_iter = iter(loader)
        buf: list[dict] = []

        while True:
            # Refill to bucket_buffer (or to source exhaustion).
            while args.bucket_buffer > 0 and len(buf) < args.bucket_buffer:
                try:
                    raw = next(loader_iter)
                except StopIteration:
                    break
                n = len(raw["audio"])
                for i in range(n):
                    buf.append({
                        "audio": raw["audio"][i],
                        "text": raw["text"][i],
                        "language": raw["language"][i],
                        "speaker": raw["speaker"][i],
                        "source_url": raw["source_url"][i],
                    })
            # Bucket-off path: just yield raw batches.
            if args.bucket_buffer <= 0:
                for raw in loader_iter:
                    yield {
                        "audio": list(raw["audio"]),
                        "text": list(raw["text"]),
                        "language": list(raw["language"]),
                        "speaker": list(raw["speaker"]),
                        "source_url": list(raw["source_url"]),
                    }
                return
            if not buf:
                return
            # Sort once and drain in encode-sized chunks. Chunks are
            # length-similar by construction.
            buf.sort(key=lambda x: x["audio"].shape[0])
            while buf:
                chunk = buf[: args.batch_size]
                buf = buf[args.batch_size :]
                yield {
                    "audio": [x["audio"] for x in chunk],
                    "text": [x["text"] for x in chunk],
                    "language": [x["language"] for x in chunk],
                    "speaker": [x["speaker"] for x in chunk],
                    "source_url": [x["source_url"] for x in chunk],
                }

    # Use perf_counter for stage boundaries; cuda.synchronize() before timing
    # any GPU stage so wall time is honest (default async hides cost).
    _t_read_start = time.perf_counter()
    for batch in _bucketed_batches():
        # `read` = time from end-of-prev-iteration to receipt of next batch.
        stage_total["read"] += (time.perf_counter() - _t_read_start) * 1000
        audios = batch["audio"]
        texts = batch["text"]
        if not audios:
            _t_read_start = time.perf_counter()
            continue

        # prep: pad batch + h2d copy (waits implicitly when we encode)
        _t = time.perf_counter()
        lens = [a.shape[0] for a in audios]
        max_len = max(lens)
        pad_to = ((max_len + ENCODER_STRIDE - 1) // ENCODER_STRIDE) * ENCODER_STRIDE
        buf = torch.zeros(len(audios), pad_to, dtype=torch.float32, pin_memory=True)
        for i, a in enumerate(audios):
            buf[i, : a.shape[0]] = a
        gpu_buf = buf.to("cuda", non_blocking=True)
        torch.cuda.synchronize()
        stage_total["prep"] += (time.perf_counter() - _t) * 1000

        # encode: GPU forward through VibeVoice
        _t = time.perf_counter()
        latents_gpu = vv.encode(gpu_buf)
        torch.cuda.synchronize()
        stage_total["encode"] += (time.perf_counter() - _t) * 1000

        # d2h: GPU → CPU + dtype cast for write
        _t = time.perf_counter()
        latents_cpu = latents_gpu.float().cpu()
        stage_total["d2h"] += (time.perf_counter() - _t) * 1000

        # write: enqueue per-example to AsyncShardWriter (non-blocking unless queue full)
        _t = time.perf_counter()
        lat_lens = [(l + ENCODER_STRIDE - 1) // ENCODER_STRIDE for l in lens]
        for i in range(len(audios)):
            writer.add(
                latents=latents_cpu[i, : lat_lens[i]],
                text=texts[i],
                language=batch["language"][i],
                speaker=batch["speaker"][i],
                source_url=batch["source_url"][i],
            )
        stage_total["write"] += (time.perf_counter() - _t) * 1000

        n_examples += len(audios)
        n_batches += 1
        _t_read_start = time.perf_counter()

        if writer.n_shards_written >= args.max_shards:
            break
        if time.time() - last_log > 15:
            elapsed = time.time() - t0
            stages = "  ".join(
                f"{k}={stage_total[k]/max(n_batches,1):.1f}ms"
                for k in ("read", "prep", "encode", "d2h", "write")
            )
            gpu_str = ""
            if _nvml_ok:
                with _nvml_lock:
                    gs = list(_nvml_samples_gpu)
                    ms = list(_nvml_samples_mem)
                    _nvml_samples_gpu.clear()
                    _nvml_samples_mem.clear()
                if gs:
                    gpu_str = (f"  gpu_util_avg={sum(gs)/len(gs):.0f}%  "
                               f"gpu_util_max={max(gs)}%  "
                               f"mem_util_avg={sum(ms)/len(ms):.0f}%")
            print(f"[cache] rank={args.rank}  shards={writer.n_shards_written}/{args.max_shards}  "
                  f"examples={n_examples}  ex/s={n_examples/max(elapsed, 1e-3):.1f}  "
                  f"per-batch: {stages}{gpu_str}",
                  flush=True)
            last_log = time.time()

    writer.close()  # drains queue + joins thread + flushes final tar
    elapsed = time.time() - t0
    print(f"[cache] done. rank={args.rank}  shards={writer.n_shards_written} "
          f"examples={n_examples}  elapsed={elapsed:.1f}s  "
          f"ex/s={n_examples/max(elapsed, 1e-3):.1f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
