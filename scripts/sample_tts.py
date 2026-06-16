"""End-to-end inference sanity: text → latents → waveform.

Exercises the full generation path on a single GPU. Works with a freshly
random-initialised model (audio quality will be garbage — we're validating
the *pipeline*, not the quality). Once a real checkpoint exists, pass
`--resume-from` to sample from it.

Run inside the SIF on rt_QG:
    python -m scripts.sample_tts \
        --config configs/tts_smoke_train.yaml \
        --prompt "Hello, this is LatentLM speaking." \
        --out /tmp/sample.wav
"""

from __future__ import annotations

import argparse
import os
import sys

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", required=True, help="Text to synthesise.")
    parser.add_argument("--out", required=True, help="Output .wav path.")
    parser.add_argument("--resume-from", default=None,
                        help="Checkpoint path (dir with `latest` or a specific .pt).")
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--min-frames", type=int, default=0,
                        help="Force the AR loop to emit at least this many "
                             "latent frames before respecting <eod>. Useful "
                             "for under-trained models that stop too early.")
    parser.add_argument("--dpm-steps", type=int, default=5)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    args = parser.parse_args()

    from omegaconf import OmegaConf

    from latent_lm.data.text_tokenizer import TextTokenizer, TextTokenizerConfig
    from latent_lm.inference import SampleConfig, sample_tts
    from latent_lm.models.tokenizer import VibeVoiceTokenizer

    cfg = OmegaConf.load(args.config)
    # Force-disable FP8 at inference: AR generation grows the seq len one token
    # at a time, which violates TE's "leading dims divisible by 8" rule when
    # short. bf16 is fine for inference (it's the only place FP8 was the win).
    if "precision" in cfg and "fp8" in cfg.precision:
        cfg.precision.fp8 = None
    # When resuming, skip the HF weight load — checkpoint will overwrite
    # everything anyway, and the HF load takes ~30s for 4B models.
    if args.resume_from and "model" in cfg and "use_bridge" in cfg.model:
        cfg.model.load_hf_weights = False
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    tp = int(cfg.parallelism.get("tensor_model_parallel_size", 1)) if "parallelism" in cfg else 1
    tok = TextTokenizer(TextTokenizerConfig(model_id=cfg.tokenizer.text, tp=tp))
    print(f"[sample] tokenizer: {cfg.tokenizer.text}  vocab={tok.vocab_size}", flush=True)

    dt = torch.bfloat16 if cfg.precision.params_dtype == "bfloat16" else torch.float32
    is_lite = bool(cfg.model.get("lite") if hasattr(cfg.model, "get") else None)
    if is_lite:
        # The lite model is built/trained/saved uniformly in fp32 (the bf16 was
        # only ever an autocast at train time). Sample in fp32 end-to-end so the
        # embedding, heads, and the VibeVoice decode all share one dtype — the
        # selective bf16 head cast below is a container-path (TE/Megatron) device.
        dt = torch.float32
        # No-container path: plain-torch TorchBackbone + lite (.pt) checkpoint.
        from latent_lm.models.latent_lm import LatentLM, LatentLMConfig
        from latent_lm.models.torch_backbone import TorchBackbone, TorchBackboneConfig
        m = cfg.model.lite
        packed = int(cfg.runtime.get("packed_total_length", 0) or 0)
        backbone = TorchBackbone(TorchBackboneConfig(
            hidden_dim=int(m.hidden_dim), n_layers=int(m.n_layers), n_heads=int(m.n_heads),
            ffn_mult=int(m.get("ffn_mult", 4)),
            max_seq_len=max(packed, int(cfg.data.max_text_tokens) + 4 * int(cfg.data.max_latent_frames))))
        model = LatentLM(LatentLMConfig(
            vocab_size=tok.vocab_size, hidden_dim=int(m.hidden_dim),
            latent_dim=int(cfg.model.latent_dim), diff_head_layers=int(cfg.model.diff_head_layers),
            diff_head_ffn_mult=int(cfg.model.diff_head_ffn_mult)), backbone=backbone).to("cuda")
    else:
        from latent_lm.train import build_model
        model, _ = build_model(cfg, vocab_size=tok.vocab_size)
        model = model.to("cuda")
    model.latent_in_proj.to(dtype=dt)
    model.diffusion_head.to(dtype=dt)
    model.eval()
    print(f"[sample] model built ({sum(p.numel() for p in model.parameters())/1e6:.1f}M)", flush=True)

    if args.resume_from:
        if is_lite:
            from latent_lm import checkpoint_lite as ckpt_mod
            ckpt_mod.load(args.resume_from, model, map_location="cuda")
        else:
            from latent_lm import checkpoint as ckpt_mod
            ckpt_mod.load(path=args.resume_from, model=model,
                          map_location=f"cuda:{torch.cuda.current_device()}")
        print(f"[sample] loaded {args.resume_from}", flush=True)

    vv = VibeVoiceTokenizer().to("cuda")
    print(f"[sample] VibeVoice loaded", flush=True)

    text_ids_core = tok.encode(args.prompt)
    text_ids = torch.cat([
        torch.tensor([tok.specials.bos], dtype=torch.long),
        text_ids_core,
        torch.tensor([tok.specials.bod], dtype=torch.long),
    ]).unsqueeze(0).to("cuda")  # (1, T)
    print(f"[sample] prompt text_ids shape: {tuple(text_ids.shape)}", flush=True)

    pred_type = (cfg.loss.ddpm.get("prediction_type", "v_prediction")
                 if "loss" in cfg and "ddpm" in cfg.loss else "v_prediction")
    sample_cfg = SampleConfig(
        max_latent_frames=args.max_frames,
        min_latent_frames=args.min_frames,
        dpm_steps=args.dpm_steps,
        cfg_scale=args.cfg_scale,
        prediction_type=pred_type,
    )
    print(f"[sample] prediction_type={pred_type}", flush=True)
    latents = sample_tts(model, text_ids, tok.specials, sample_cfg)
    print(f"[sample] generated latents shape: {tuple(latents.shape)}", flush=True)

    # Train side scales latents by LATENT_SCALE (≈0.195) so DDPM sees x0 ~ N(0,1).
    # VibeVoice's decoder expects native scale (std≈5.12) — undo before decoding.
    from latent_lm.models.tokenizer import LATENT_SCALE
    latents = latents / LATENT_SCALE

    audio = vv.decode(latents.to(dt)).float().cpu()   # (B, channels, N_samples)
    print(f"[sample] decoded audio shape: {tuple(audio.shape)}", flush=True)

    import soundfile as sf

    from latent_lm.models.tokenizer import SAMPLE_RATE

    # soundfile expects (samples,) for mono; VibeVoice returns (B, 1, N).
    wav = audio[0].squeeze(0).numpy()
    sf.write(args.out, wav, SAMPLE_RATE)
    print(f"[sample] wrote {args.out}  ({wav.shape[0]/SAMPLE_RATE:.2f}s @ {SAMPLE_RATE} Hz)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
