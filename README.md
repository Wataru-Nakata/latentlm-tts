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

## Two training paths

| | Lite (`train_lite`) | Scale (`train`) |
|---|---|---|
| Backbone | plain-PyTorch `TorchBackbone` (RoPE) | Qwen via Megatron-Bridge |
| Deps | `pip install` only | NVIDIA NeMo container (Megatron-core, transformer-engine) |
| Parallelism | DDP (torchrun) | TP / SP / FP8, multi-node |
| Use for | small models, dev, getting started | large models, production |

Both share the same model, losses, data pipeline, and inference.

## Install

```bash
git clone <repo-url> && cd latentlm-tts
pip install -e .          # add [dev] for tests/lint
```

## Quickstart A — train without a container (lite path)

Streams Emilia, VibeVoice-encodes on the GPU, trains a small model. Needs one
GPU (or CPU for a tiny smoke).

```bash
# 1 GPU
torchrun --standalone --nproc_per_node=1 -m latent_lm.train_lite \
    --config configs/tts_lite.yaml --train-steps 1000
# or: ./scripts/launch.sh 1 configs/tts_lite.yaml --train-steps 1000
# multi-GPU: ./scripts/launch.sh 4 configs/tts_lite.yaml
```

Pre-encode latents first (optional, faster steady-state) and point `data.cache_dir`
at the output:

```bash
python scripts/cache_latents.py --cache-dir ./cache/emilia_en --languages en
```

## Quickstart B — large-scale training (container path)

`latent_lm.train` uses Megatron-Bridge (TP/FP8/multi-node) and runs inside the
NVIDIA NeMo container. See `containers/nemo_latentlm.def` and `examples/pbs/`
(ABCI-Q reference job scripts — adapt to your scheduler):

```bash
torchrun --standalone --nproc_per_node=4 -m latent_lm.train \
    --config configs/tts_qwen_500m.yaml --train-steps 50000
```

## Inference

```bash
python scripts/sample_tts.py --config configs/tts_lite.yaml \
    --resume-from outputs/tts_lite/checkpoints/latest.pt \
    --prompt "Hello, this is a test." --out hello.wav
```

Or from Python:

```python
from latent_lm import LatentLM, sample_tts, SampleConfig, TextTokenizer
```

## Config reference (key fields)

```yaml
model:   { lite: {hidden_dim, n_layers, n_heads}, latent_dim, diff_head_layers }
tokenizer: { text: <HF id>, audio: microsoft/VibeVoice-AcousticTokenizer }
data:    { dataset_id, languages, max_text_tokens, max_latent_frames, cache_dir }
loss:    { alpha_diff, ddpm: {num_timesteps, schedule, prediction_type} }
optim:   { lr, warmup_steps, max_steps, grad_clip }
runtime: { micro_batch_size, packed_total_length, checkpoint_every_n_steps }
```

## Layout

```
latent_lm/
  models/  latent_lm.py  diffusion_head.py  tokenizer.py(VibeVoice)
           torch_backbone.py(lite)  bridge_backbone.py(container)
  data/    emilia.py  text_tokenizer.py  collate.py  pipeline.py  cache.py
  losses.py  inference.py  checkpoint.py(container)  checkpoint_lite.py
  train.py(container)  train_lite.py(no-container)
configs/  scripts/  examples/pbs/  containers/  tests/
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/        # CPU-only: collate, losses, tokenizer, cache, lite trainer
```

## Notes & limitations

- VibeVoice encode/decode and training/inference need a GPU; the CPU tests cover
  the pure-PyTorch logic only.
- The container path's checkpoints (`checkpoint.py`, Megatron distributed format)
  and the lite path's checkpoints (`checkpoint_lite.py`, plain `torch.save`) are
  **not** interchangeable.
- `examples/pbs/` are cluster-specific (ABCI-Q) references, not portable scripts.

## License

MIT — see [LICENSE](LICENSE).
