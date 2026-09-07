"""Regression tests for the padding-mask leak in FoundationModel.

Historical bug: `FoundationModel` ran self-attention under the flash backend
with ``src_key_padding_mask=None``. Because ``apply_pad_token_replacement``
inserts a learned ``pad_token`` parameter at every padded position, the
learned vector participated as identical keys/values in every softmax. The
effect was a rank-1 leak proportional to the number of padded slots,
contaminating the CLS / mean-pool latent with an ``n_peaks`` axis that
dominated UMAP visualisations.

The fix always passes the key-padding mask through (FlashMHA folds it into
the SDPA attn_mask as an additive ``-inf`` bias; math backend applies it
directly). These tests guard against regression by asserting that padding
slots cannot influence the latent.
"""

from __future__ import annotations

import pytest
import torch

from instanovo_fm.model.encoder import FoundationModel


def _make_cfg(backend: str) -> dict:
    """Minimal FoundationModel config. Regression (not classification) keeps
    the test dependency surface small — no binning setup required."""
    return {
        "dim_model": 64,
        "n_heads": 4,
        "dim_feedforward": 128,
        "n_layers": 2,
        "dropout": 0.0,
        "n_peaks": 20,
        "max_mz": 2500.0,
        "min_mz": 0.0,
        "max_charge": 10,
        "peak_encoder": {"type": "multiscale"},
        "mz_head": {"task": "regression"},
        "meta_token": {"enabled": False},
        "auxiliary": {"enabled": False},
        "masking": {"blur_sigma_da": 0.0},
        "architecture": {
            "positional_encoding": {"type": "sinusoidal"},
            "relative_bias": {"type": "none"},
            "attention": {"backend": backend},
        },
    }


def _make_padded_spectrum(n_real: int, n_total: int = 20, seed: int = 0) -> torch.Tensor:
    """One spectrum with n_real real peaks followed by zero-padded slots.

    Padding is detected by the encoder as ``spectra.sum(-1) == 0``.
    """
    g = torch.Generator().manual_seed(seed)
    spectra = torch.zeros(1, n_total, 2)
    spectra[0, :n_real, 0] = torch.sort(torch.rand(n_real, generator=g) * 0.9 + 0.05)[0]
    spectra[0, :n_real, 1] = torch.rand(n_real, generator=g) * 0.8 + 0.2
    return spectra


@pytest.fixture(params=["flash", "math"])
def backend(request: pytest.FixtureRequest) -> str:
    return request.param


def test_latent_invariant_to_pad_token_perturbation(backend: str) -> None:
    """Under correct padding-mask plumbing, perturbing the learned pad_token
    parameter must not change the mean-pool latent — padded positions are
    masked out of every softmax, so the content of the pad slot is irrelevant.

    This is the direct regression test for the padding leak: before the fix,
    scaling pad_token shifted the latent; after the fix, the latent is
    invariant because padded keys are set to -inf in the softmax.

    For the math backend there is no pad_token parameter, so we assert
    determinism across two forward passes as a baseline sanity check.
    """
    torch.manual_seed(42)
    cfg = _make_cfg(backend)
    model = FoundationModel(cfg=cfg, dim_model=64, n_heads=4, n_peaks=20, n_layers=2)
    model.eval()

    spectra = _make_padded_spectrum(n_real=5, n_total=20, seed=1)

    with torch.no_grad():
        latent_before = model.encode_mean_pooled(spectra)

    if backend == "flash":
        assert model.pad_token is not None, "Flash backend should create a pad_token"
        with torch.no_grad():
            # Perturb pad_token to a very different value. Under correct
            # masking this cannot influence the latent.
            model.pad_token.data = model.pad_token.data * 100.0 + 50.0
    else:
        assert model.pad_token is None, "Math backend should not create a pad_token"

    with torch.no_grad():
        latent_after = model.encode_mean_pooled(spectra)

    diff = (latent_after - latent_before).norm().item()
    assert diff < 1e-5, (
        f"[{backend}] Latent changed by {diff:.2e} under pad_token perturbation. "
        "This indicates padded positions are leaking into attention — "
        "src_key_padding_mask is not being plumbed through."
    )


def test_latent_invariant_to_padding_count(backend: str) -> None:
    """Given the same K real peaks, varying only the number of padded slots
    must not change the mean-pool latent.

    Two batches with the same first K=5 peaks but different total sequence
    lengths (K+3 pads vs K+15 pads) should produce the same latent for the
    real-peak content under correct masking.
    """
    torch.manual_seed(7)
    cfg = _make_cfg(backend)
    model = FoundationModel(cfg=cfg, dim_model=64, n_heads=4, n_peaks=20, n_layers=2)
    model.eval()

    g = torch.Generator().manual_seed(3)
    mz = torch.sort(torch.rand(5, generator=g) * 0.9 + 0.05)[0]
    inten = torch.rand(5, generator=g) * 0.8 + 0.2

    def _make(n_total: int) -> torch.Tensor:
        s = torch.zeros(1, n_total, 2)
        s[0, :5, 0] = mz
        s[0, :5, 1] = inten
        return s

    spec_short = _make(8)   # 5 real + 3 pads
    spec_long = _make(20)   # 5 real + 15 pads

    with torch.no_grad():
        latent_short = model.encode_mean_pooled(spec_short)
        latent_long = model.encode_mean_pooled(spec_long)

    diff = (latent_short - latent_long).norm().item()
    assert diff < 1e-4, (
        f"[{backend}] Latent differs by {diff:.2e} between padding-count 3 vs 15 "
        "for the same 5 real peaks. Padding count must not influence the latent — "
        "this is the n_peaks-axis leak we are guarding against."
    )


# ---------------------------------------------------------------------------
# forward()-level tests — exercise the TRAINING-path site (encoder.py:938–941)
# ---------------------------------------------------------------------------
#
# The tests above call encode_mean_pooled, which exercises only site 1300–1303
# in encoder.py. The most safety-critical site is the one used by forward(),
# because that is the path training uses: bad masking there means every
# gradient step sees contaminated attention. The following tests directly
# exercise forward() so a regression in site 938–941 is caught even if the
# encode_mean_pooled tests still pass.


def test_forward_real_peak_predictions_invariant_to_padding_count(backend: str) -> None:
    """Forward-path regression test (covers encoder.py:938–941).

    Calling ``forward()`` on two batches with identical real peaks but
    different padding counts must produce identical predictions at the
    real-peak positions. The training loss only sums over real peaks, so
    any deviation here means the training gradients were contaminated by
    padded positions.
    """
    torch.manual_seed(11)
    cfg = _make_cfg(backend)
    model = FoundationModel(cfg=cfg, dim_model=64, n_heads=4, n_peaks=20, n_layers=2)
    model.eval()

    g = torch.Generator().manual_seed(5)
    mz = torch.sort(torch.rand(5, generator=g) * 0.9 + 0.05)[0]
    inten = torch.rand(5, generator=g) * 0.8 + 0.2

    def _make(n_total: int) -> torch.Tensor:
        s = torch.zeros(1, n_total, 2)
        s[0, :5, 0] = mz
        s[0, :5, 1] = inten
        return s

    spec_short = _make(8)
    spec_long = _make(20)

    with torch.no_grad():
        pred_short, _ = model(spec_short)
        pred_long, _ = model(spec_long)

    # Regression task returns (B, L, 1). Compare only the first 5 positions
    # (the real peaks shared between both batches). Padded-position outputs
    # are expected to differ — they contain junk because the loss ignores them.
    real_short = pred_short[:, :5]
    real_long = pred_long[:, :5]
    diff = (real_short - real_long).abs().max().item()

    assert diff < 1e-4, (
        f"[{backend}] forward() predictions at real peaks differ by {diff:.2e} "
        "between padding-count 3 vs 15. This is the training-path site of the "
        "padding leak (encoder.py:938–941) — every gradient step of training "
        "would see contaminated attention."
    )


def test_forward_real_peak_predictions_invariant_to_pad_token_perturbation(
    backend: str,
) -> None:
    """Forward-path regression test (covers encoder.py:938–941).

    Scaling the learned ``pad_token`` parameter by 100× + shift must not
    change ``forward()`` predictions at real-peak positions. Under correct
    masking the pad_token is multiplied into -inf-masked softmax weights
    (= zero), so its value cannot influence any output. On the math backend
    there is no pad_token; the test asserts deterministic output as a
    baseline sanity check.
    """
    torch.manual_seed(13)
    cfg = _make_cfg(backend)
    model = FoundationModel(cfg=cfg, dim_model=64, n_heads=4, n_peaks=20, n_layers=2)
    model.eval()

    spec = _make_padded_spectrum(n_real=5, n_total=20, seed=9)

    with torch.no_grad():
        pred_before, _ = model(spec)

    if backend == "flash":
        assert model.pad_token is not None
        with torch.no_grad():
            model.pad_token.data = model.pad_token.data * 100.0 + 50.0
    else:
        assert model.pad_token is None

    with torch.no_grad():
        pred_after, _ = model(spec)

    real_before = pred_before[:, :5]
    real_after = pred_after[:, :5]
    diff = (real_after - real_before).abs().max().item()

    assert diff < 1e-5, (
        f"[{backend}] forward() predictions at real peaks changed by {diff:.2e} "
        "under pad_token perturbation. The training path is leaking pad-token "
        "contributions into every attention softmax — encoder.py:938–941."
    )
