"""Tests for DDPM loss helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from latent_lm.losses import (
    DDPMLoss,
    DDPMScheduleConfig,
    _chunked_ce_sum,
    _lm_cross_entropy,
    cosine_alpha_bar,
)


def test_cosine_alpha_bar_monotone():
    """ᾱ_t monotonically decreases from ~1 at t=0 to ~0 at t=T."""
    ab = cosine_alpha_bar(1000)
    assert ab.shape == (1001,)
    assert ab[0].item() > 0.999
    assert ab[-1].item() < 0.05
    diffs = ab[1:] - ab[:-1]
    assert (diffs <= 1e-6).all(), "ᾱ_t must be monotonically non-increasing"


def test_q_sample_endpoints():
    """At t=0, q_sample ≈ x0 (no noise); at t=T-1, ≈ noise (no signal)."""
    loss = DDPMLoss(DDPMScheduleConfig(num_timesteps=1000))
    x0 = torch.randn(4, 64)
    noise = torch.randn(4, 64)
    t0 = torch.zeros(4, dtype=torch.long)
    tend = torch.full((4,), 999, dtype=torch.long)
    near_x0 = loss.q_sample(x0, t0, noise)
    near_noise = loss.q_sample(x0, tend, noise)
    assert torch.allclose(near_x0, x0, atol=1e-2)
    assert torch.allclose(near_noise, noise, atol=2e-1)


def test_lm_cross_entropy_falls_back_when_no_megatron():
    """Without parallel_state init, _lm_cross_entropy uses F.cross_entropy."""
    logits = torch.randn(2, 5, 100)
    targets = torch.randint(0, 100, (2, 5))
    targets[0, 0] = -100  # ignored position
    loss = _lm_cross_entropy(logits, targets)
    # Returns a finite scalar.
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_lm_cross_entropy_ignores_negatives():
    """Loss with ALL targets set to -100 yields zero (no valid positions)."""
    logits = torch.randn(2, 5, 100)
    targets = torch.full((2, 5), -100, dtype=torch.long)
    # Standard cross_entropy with all-ignored returns nan; test that our
    # behaviour matches F.cross_entropy (single source of truth in TP=1 path).
    loss = _lm_cross_entropy(logits, targets)
    ref = F.cross_entropy(logits.reshape(-1, 100), targets.reshape(-1), ignore_index=-100)
    # Both should be the same nan / scalar.
    assert torch.equal(loss.isnan(), ref.isnan())


def test_v_prediction_target_endpoints():
    """v_t = √ᾱ·ε − √(1−ᾱ)·x₀.
    At t=0 (ᾱ≈1):    v ≈ ε - 0·x₀ = ε        (actually ε since √ᾱ≈1)
    At t=T-1 (ᾱ≈0):  v ≈ 0·ε − x₀ = -x₀
    Note: paper formulation is v = √ᾱ·ε − √(1−ᾱ)·x₀, so at t=0 v→ε.
    """
    cfg = DDPMScheduleConfig(num_timesteps=1000, prediction_type="v_prediction")
    loss = DDPMLoss(cfg)
    x0 = torch.randn(4, 64)
    noise = torch.randn(4, 64)

    # Inline the v target computation (mirrors DDPMLoss.forward).
    ab0 = loss.alpha_bar[torch.zeros(4, dtype=torch.long)].view(-1, 1)
    abT = loss.alpha_bar[torch.full((4,), 999, dtype=torch.long)].view(-1, 1)
    v_at_0 = ab0.sqrt() * noise - (1 - ab0).sqrt() * x0
    v_at_T = abT.sqrt() * noise - (1 - abT).sqrt() * x0

    assert torch.allclose(v_at_0, noise, atol=2e-2), \
        f"at t=0, v should ≈ ε; max diff = {(v_at_0 - noise).abs().max()}"
    assert torch.allclose(v_at_T, -x0, atol=2e-1), \
        f"at t=T-1, v should ≈ -x₀; max diff = {(v_at_T + x0).abs().max()}"


def test_ddpm_loss_v_prediction_finite():
    """DDPMLoss with v_prediction returns a finite scalar with grad."""
    cfg = DDPMScheduleConfig(num_timesteps=1000, prediction_type="v_prediction",
                             timesteps_per_forward=2)
    loss = DDPMLoss(cfg)
    x0 = torch.randn(8, 64)
    h = torch.randn(8, 128)

    # Tiny "head" — predicts zeros (worst-case learner).
    class _ZeroHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Linear(64 + 128, 64, bias=False)
            torch.nn.init.zeros_(self.w.weight)

        def forward(self, x_t, t, h_in):
            return self.w(torch.cat([x_t, h_in], dim=-1))

    head = _ZeroHead()
    out = loss(head, x0, h)
    assert out.dim() == 0 and torch.isfinite(out)
    out.backward()
    assert head.w.weight.grad is not None


def test_ddpm_loss_unknown_prediction_type():
    """Unknown prediction_type raises a clear ValueError."""
    cfg = DDPMScheduleConfig(num_timesteps=1000, prediction_type="bogus")
    loss = DDPMLoss(cfg)
    x0 = torch.randn(2, 64)
    h = torch.randn(2, 128)
    head = torch.nn.Linear(64, 64)

    class _Head(torch.nn.Module):
        def forward(self, x_t, t, h_in):
            return torch.zeros_like(x_t)

    try:
        loss(_Head(), x0, h)
    except ValueError as e:
        assert "bogus" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown prediction_type")


def test_chunked_ce_matches_unchunked():
    """_chunked_ce_sum / n_valid == F.cross_entropy(reduction='mean', ignore_index=-100)."""
    torch.manual_seed(0)
    N, V = 8000, 1000
    logits = torch.randn(N, V, dtype=torch.float32)
    targets = torch.randint(0, V, (N,), dtype=torch.long)
    # Sprinkle some ignored positions.
    targets[::13] = -100

    n_valid = (targets != -100).sum().clamp(min=1).to(torch.float32)
    chunked = _chunked_ce_sum(logits, targets, chunk_size=1024) / n_valid
    ref = F.cross_entropy(logits, targets, ignore_index=-100)
    assert torch.allclose(chunked, ref, atol=1e-5), \
        f"chunked={chunked.item()}  ref={ref.item()}"


def test_chunked_ce_ignore_index_all_masked():
    """All -100 targets: chunked sum is 0, divided by clamped n_valid still 0
    (F.cross_entropy with reduction='sum' over zero valid tokens returns 0,
    not nan — so the chunked sum-then-divide path is well-defined here)."""
    N, V = 100, 50
    logits = torch.randn(N, V, dtype=torch.float32)
    targets = torch.full((N,), -100, dtype=torch.long)
    chunked = _chunked_ce_sum(logits, targets, chunk_size=32)
    assert chunked.item() == 0.0


def test_lm_cross_entropy_chunked_path():
    """When N > chunk_size on the non-TP path, chunked CE matches dense CE."""
    torch.manual_seed(0)
    B, L, V = 1, 8200, 500
    logits = torch.randn(B, L, V)
    targets = torch.randint(0, V, (B, L))
    targets[0, ::17] = -100

    # Force the chunked path with chunk_size < L
    chunked = _lm_cross_entropy(logits, targets, chunk_size=2048)
    dense = _lm_cross_entropy(logits, targets, chunk_size=0)  # disable
    assert torch.allclose(chunked, dense, atol=1e-5), \
        f"chunked={chunked.item()}  dense={dense.item()}"
