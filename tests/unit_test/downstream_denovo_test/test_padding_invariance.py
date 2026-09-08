"""Tests for the padding-mask in DownstreamDeNovo._encoder.

These tests assert:
    1. Encoder output at non-padded positions is invariant to pad_token.
    2. Encoder output at non-padded positions is invariant to the number
       of padded slots appended to a fixed set of real peaks.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from instanovo_fm.downstream.de_novo_sequencing.model import DownstreamDeNovo


def _make_padded_batch(n_real: int, n_total: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (spectra, spectra_mask) for a single-spectrum batch.

    spectra: (1, n_total, 2) — first n_real rows are real peaks, rest are zero-padded.
    spectra_mask: (1, n_total) bool — True at padded positions.
    """
    g = torch.Generator().manual_seed(seed)
    spectra = torch.zeros(1, n_total, 2)
    # m/z in (0, 1) sorted ascending; intensity in (0.2, 1.0).
    spectra[0, :n_real, 0] = torch.sort(torch.rand(n_real, generator=g) * 0.9 + 0.05)[0]
    spectra[0, :n_real, 1] = torch.rand(n_real, generator=g) * 0.8 + 0.2
    spectra_mask = torch.zeros(1, n_total, dtype=torch.bool)
    spectra_mask[0, n_real:] = True
    return spectra, spectra_mask


@pytest.fixture(params=["math", "flash"])
def backend(request: pytest.FixtureRequest) -> str:
    param: str = request.param
    return param


def test_encoder_output_invariant_to_pad_token_perturbation(
    backend: str,
    matched_denovo_model_factory: Callable[..., DownstreamDeNovo],
) -> None:
    """Perturbing pad_token must not change encoder output at non-padded positions.

    Under correct mask plumbing, padded keys are -inf in every softmax, so
    the pad_token value cannot influence any non-padded position. Under the
    bug (flash + src_key_padding_mask=None), pad_token leaks into every
    attention head as identical k/v, and any perturbation propagates
    everywhere.

    For the math backend there is no pad_token (it's None), so we assert
    determinism across two forward passes as a baseline sanity check.
    """
    model = matched_denovo_model_factory(backend=backend)
    n_real, n_total = 5, 20

    spectra, spectra_mask = _make_padded_batch(n_real=n_real, n_total=n_total, seed=1)

    with torch.no_grad():
        x_before, _ = model._encoder(spectra=spectra, spectra_mask=spectra_mask)

    if backend == "flash":
        assert model.pad_token is not None, "Flash backend should create a pad_token"
        with torch.no_grad():
            # Large multiplicative + additive perturbation: under correct
            # masking this cannot influence any non-padded output.
            model.pad_token.data = model.pad_token.data * 100.0 + 50.0
    else:
        assert model.pad_token is None, "Math backend should not create a pad_token"

    with torch.no_grad():
        x_after, _ = model._encoder(spectra=spectra, spectra_mask=spectra_mask)

    # _encoder prepends 1 special token (latent) when meta is disabled.
    # Real-peak positions in the encoder output are [1 : 1 + n_real].
    real_slice = slice(1, 1 + n_real)
    diff = (x_after[:, real_slice] - x_before[:, real_slice]).norm().item()
    assert diff < 1e-5, (
        f"[{backend}] Encoder output at real-peak positions changed by {diff:.2e} "
        f"under pad_token perturbation. This means padded positions are leaking "
        f"into attention — src_key_padding_mask is not being plumbed through "
        f"(see model.py)."
    )


def test_encoder_output_invariant_to_padding_count(
    backend: str,
    matched_denovo_model_factory: Callable[..., DownstreamDeNovo],
) -> None:
    """Same K real peaks padded to different total lengths → same output at those K positions.

    Two batches: K=5 real peaks padded to L1=8 vs L2=20. Under correct
    masking the encoder output at the 5 real-peak positions must be
    identical between the two batches. Under the bug, padded slots leak
    into attention so the rank-1 contribution scales with padding count
    and the outputs diverge.
    """
    model = matched_denovo_model_factory(backend=backend)
    n_real = 5

    spectra_short, mask_short = _make_padded_batch(n_real=n_real, n_total=8, seed=7)
    spectra_long, mask_long = _make_padded_batch(n_real=n_real, n_total=20, seed=7)
    # Sanity check: same first n_real rows in both.
    assert torch.allclose(spectra_short[0, :n_real], spectra_long[0, :n_real])

    with torch.no_grad():
        x_short, _ = model._encoder(spectra=spectra_short, spectra_mask=mask_short)
        x_long, _ = model._encoder(spectra=spectra_long, spectra_mask=mask_long)

    real_slice = slice(1, 1 + n_real)
    diff = (x_long[:, real_slice] - x_short[:, real_slice]).norm().item()
    assert diff < 1e-5, (
        f"[{backend}] Encoder output at real-peak positions differs by {diff:.2e} "
        f"between L=8 and L=20 padded inputs. This is the rank-1 padding leak: "
        f"the contribution of pad_token scales with the number of padded slots."
    )
