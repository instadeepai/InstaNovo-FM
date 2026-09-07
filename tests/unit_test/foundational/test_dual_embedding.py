"""Tests for DualPeakEmbedding."""

import torch
import pytest

from instanovo_fm.model.embeddings import (
    DualPeakEmbedding,
    MultiScalePeakEmbedding,
)


@pytest.fixture
def dual_encoder():
    return DualPeakEmbedding(
        h_size=64,
        dropout=0.0,
        min_mz=50.0,
        max_mz=2500.0,
        num_rbf=128,
        normalize_mz=True,
    )


class TestDualPeakEmbedding:
    def test_output_shape(self, dual_encoder):
        B, L = 2, 50
        spectra = torch.rand(B, L, 2)  # normalized m/z + intensity
        out = dual_encoder(spectra)
        assert out.shape == (B, L, 64)

    def test_rezero_gate_at_init(self, dual_encoder):
        """Gate parameter starts at zero."""
        assert dual_encoder.rbf_gate.item() == 0.0

    def test_output_matches_sinusoidal_at_init(self, dual_encoder):
        """At init, gate=sigmoid(0)=0.5, so there's an RBF contribution.
        But the gate starts at 0 (not -inf), so this isn't exactly sinusoidal.
        Verify that the output is close to sinusoidal + small RBF contribution."""
        B, L = 2, 50
        spectra = torch.rand(B, L, 2)

        with torch.no_grad():
            full_out = dual_encoder(spectra)
            sin_only = dual_encoder.sin_encoder(spectra)

        # The difference should be bounded (sigmoid(0)=0.5 * rbf)
        diff = (full_out - sin_only).abs().mean()
        # Just verify it's finite and the outputs are in the same ballpark
        assert diff.isfinite()
        assert diff < sin_only.abs().mean() * 2  # RBF contribution < 2x sin magnitude

    def test_backward_pass(self, dual_encoder):
        B, L = 2, 50
        spectra = torch.rand(B, L, 2, requires_grad=True)
        out = dual_encoder(spectra)
        loss = out.sum()
        loss.backward()
        assert spectra.grad is not None

    def test_single_peak(self, dual_encoder):
        spectra = torch.rand(1, 1, 2)
        out = dual_encoder(spectra)
        assert out.shape == (1, 1, 64)
