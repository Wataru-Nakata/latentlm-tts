"""CPU smoke for the no-container lite path: TorchBackbone + LatentLM + loss +
checkpoint. No network, no GPU — overfit one synthetic batch and assert loss
drops, then round-trip a checkpoint.
"""

from __future__ import annotations

import torch

from latent_lm.checkpoint_lite import load as ckpt_load
from latent_lm.checkpoint_lite import save as ckpt_save
from latent_lm.data.collate import CollateConfig, SpecialTokens, collate_batch, pack_example
from latent_lm.losses import DDPMLoss, DDPMScheduleConfig, ModalityLoss
from latent_lm.models.latent_lm import LatentLM, LatentLMConfig
from latent_lm.models.torch_backbone import TorchBackbone, TorchBackboneConfig


def _build():
    vocab = 100
    specials = SpecialTokens.from_vocab(vocab)
    cfg = CollateConfig(max_text_tokens=16, max_latent_frames=16, latent_dim=8)
    backbone = TorchBackbone(TorchBackboneConfig(
        hidden_dim=64, n_layers=2, n_heads=4, ffn_mult=2, max_seq_len=128))
    model = LatentLM(LatentLMConfig(vocab_size=vocab, hidden_dim=64, latent_dim=8,
                                    diff_head_layers=2, diff_head_ffn_mult=2),
                     backbone=backbone)
    loss_fn = ModalityLoss(DDPMLoss(DDPMScheduleConfig(num_timesteps=100)),
                           alpha=1.0, ce_chunk_size=0)
    return model, loss_fn, specials, cfg


def _batch(specials, cfg):
    exs = []
    for _ in range(2):
        text_ids = torch.randint(0, 90, (5,))
        latents = torch.randn(7, cfg.latent_dim)
        exs.append(pack_example(text_ids=text_ids, latents=latents,
                                specials=specials, cfg=cfg))
    return collate_batch(exs, specials=specials)


def test_torch_backbone_shapes():
    bb = TorchBackbone(TorchBackboneConfig(hidden_dim=32, n_layers=2, n_heads=4, max_seq_len=64))
    x = torch.randn(3, 10, 32)
    y = bb(x)
    assert y.shape == x.shape and torch.isfinite(y).all()


def test_lite_overfits_one_batch():
    torch.manual_seed(0)
    model, loss_fn, specials, cfg = _build()
    batch = _batch(specials, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def step():
        out = model(input_ids=batch["input_ids"], input_latents=batch["input_latents"],
                    is_audio_input=batch["is_audio_input"],
                    attention_mask=batch.get("attention_mask"))
        return loss_fn(text_logits=out["text_logits"].float(),
                       text_targets=batch["text_targets"],
                       audio_targets=batch["audio_targets"], audio_mask=batch["audio_mask"],
                       hidden_states=out["hidden"], diff_head=model.diffusion_head)["loss"]

    first = float(step())
    for _ in range(40):
        opt.zero_grad()
        loss = step()
        loss.backward()
        opt.step()
    last = float(step())
    assert torch.isfinite(torch.tensor(last))
    assert last < first, f"loss did not drop: {first:.3f} -> {last:.3f}"


def test_checkpoint_lite_roundtrip(tmp_path):
    model, _, _, _ = _build()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    p = ckpt_save(str(tmp_path), step=7, model=model, optimizer=opt, keep_last_k=2)
    model2, _, _, _ = _build()
    step = ckpt_load(p, model2, map_location="cpu")
    assert step == 7
    for (k, v), (k2, v2) in zip(model.state_dict().items(), model2.state_dict().items()):
        assert k == k2 and torch.equal(v, v2)
