# latentlm-tts

Text-to-speech with **LatentLM** (next-token diffusion) on the
[Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset) dataset.

An autoregressive Transformer emits a single sequence that mixes **discrete text
tokens** and **continuous 64-d acoustic latents**:

```
<bos>  text tokens …  <bod>  a0 a1 a2 … aT  <eod> <eos>
```

Text positions are predicted with a softmax LM head (cross-entropy); each audio
frame `aᵢ` is produced by a **next-token diffusion head** conditioned on the
Transformer's hidden state. Acoustic latents come from a frozen
[VibeVoice](https://huggingface.co/microsoft/VibeVoice-AcousticTokenizer) σ-VAE
(24 kHz ↔ 64-d @ 7.5 Hz). At inference the model autoregresses latents with
DPM-Solver + classifier-free guidance, then the VibeVoice decoder renders audio.

This implements the speech-TTS instance of *"Multimodal Latent Language Modeling
with Next-Token Diffusion"* (LatentLM, arXiv 2412.08635).

## Install

```bash
git clone <repo-url> && cd latentlm-tts
pip install -e .          # core: inference + data pipeline (add [dev] for tests)
```

Training (`latent_lm.train`) uses a Qwen backbone via **Megatron-Bridge** with
transformer-engine (TP / sequence-parallel / FP8, multi-node). That stack
(`megatron-core`, `megatron-bridge`, `transformer-engine`) is pip-installable
against a matching CUDA toolchain (`pip install -e ".[train]"`), but the NVIDIA
NeMo container (`containers/nemo_latentlm.def`) bundles it and is the path of
least resistance — see `examples/pbs/` for reference job scripts.

### Container setup (one-time)

If you use the NeMo container, run these once before training:

```bash
qsub examples/pbs/build_sif.pbs        # build nemo_latentlm.sif from the def
qsub examples/pbs/install_extras.pbs   # populate ./extras  (see below)
```

`install_extras.pbs` exists because the NeMo image ships `megatron-core` /
`megatron-bridge` as *editable* installs under `/opt/Megatron-Bridge/...`, a
`root:root drwxrwx---` path that a non-root container user cannot read — so
`import megatron.core` fails with `PermissionError`. The script pip-installs a
readable copy of those packages into a user-writable prefix (`./extras`, which
is `.gitignore`d), and `examples/pbs/train_tts.pbs` puts it first on
`PYTHONPATH` (`PYTHONPATH=$EXTRAS:$REPO_ROOT`) so it shadows the broken editable
install. If your environment has a *readable* megatron install (e.g. a plain
`pip install` venv rather than this image), you can skip this step and leave
`EXTRAS` empty.

## Quickstart — training

**Cache the latents first, then train on the cache.** This is the recommended
(and only practical) path for real runs:

```bash
# 1) pre-encode Emilia → 64-d latent shards (once)
python scripts/cache_latents.py --cache-dir ./cache/emilia_en --languages en
#    (or examples/pbs/cache_emilia.pbs; run several shards in parallel)

# 2) train, pointing data.cache_dir at that directory
torchrun --standalone --nproc_per_node=4 -m latent_lm.train \
    --config configs/tts_qwen_500m.yaml --train-steps 200000
# or: ./scripts/launch.sh 4 configs/tts_qwen_500m.yaml --train-steps 200000
# multi-node: examples/pbs/train_tts.pbs (mpirun rendezvous, adapt to your scheduler)
```

Set `data.cache_dir` (in the YAML) to your cache. `configs/tts_smoke.yaml` is a
tiny config for a quick end-to-end check.

> **Why cache, not stream?** With `data.cache_dir: null` the trainer streams
> Emilia and runs the (frozen) VibeVoice encoder on the GPU *every step*. That
> works but is **~30× slower** and you re-encode the same audio every epoch.
> Streaming is fine for a smoke/exploration run; for anything real, encode once
> with `cache_latents.py` and train from the cache. (The trainer densely packs
> either way — an earlier streaming bug that left sequences ~99% padding, which
> collapsed training, is fixed; see `latent_lm/train.py:_pack_batch_from_audio`.)

## Inference

```bash
python scripts/sample_tts.py --config configs/tts_qwen_500m.yaml \
    --resume-from <checkpoint-dir> \
    --prompt "Hello, this is a test." --out hello.wav
```

Or from Python:

```python
from latent_lm import LatentLM, sample_tts, SampleConfig, TextTokenizer
```

## Config reference (key fields)

```yaml
model:   { hf_model_id: <HF id>, load_hf_weights, latent_dim, diff_head_layers, diff_head_ffn_mult }
tokenizer: { text: <HF id>, audio: microsoft/VibeVoice-AcousticTokenizer }
data:    { dataset_id, languages, max_text_tokens, max_latent_frames, cache_dir }
loss:    { alpha_diff, ddpm: {num_timesteps, schedule, prediction_type} }
optim:   { lr, warmup_steps, max_steps, grad_clip }
parallelism: { tensor_model_parallel_size, sequence_parallel }
precision: { params_dtype, fp8 }
runtime: { micro_batch_size, packed_total_length, checkpoint_every_n_steps }
```

## Layout

```
latent_lm/
  models/  latent_lm.py  diffusion_head.py  tokenizer.py(VibeVoice)
           bridge_backbone.py(Megatron-Bridge backbone)
  data/    emilia.py  text_tokenizer.py  collate.py  pipeline.py  cache.py
  losses.py  inference.py  checkpoint.py  train.py
configs/  scripts/  examples/pbs/  containers/  tests/
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/        # CPU-only: collate, losses, tokenizer, cache
```

## Notes & limitations

- VibeVoice encode/decode and training/inference need a GPU; the CPU tests cover
  the pure-PyTorch logic only.
- `examples/pbs/` are cluster-specific (ABCI-Q) references, not portable scripts.

## License

MIT — see [LICENSE](LICENSE).
