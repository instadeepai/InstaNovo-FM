"""Tests for PairwiseAttentionBias module."""

import pytest
import torch

from instanovo_fm.model.pairwise_bias import PairwiseAttentionBias


class TestPairwiseAttentionBias:
    """Tests for the refactored PairwiseAttentionBias module."""

    def test_output_shape(self):
        """PairwiseAttentionBias returns (B, L, L, hidden_dim)."""
        B, L, hidden_dim = 2, 10, 32
        module = PairwiseAttentionBias(num_freqs=16, hidden_dim=hidden_dim)
        mz = torch.rand(B, L, 1) * 2000
        out = module(mz)
        assert out.shape == (B, L, L, hidden_dim)

    def test_asymmetry(self):
        """Signed differences produce asymmetric features: feats[i,j] != feats[j,i]."""
        module = PairwiseAttentionBias(num_freqs=16, hidden_dim=32)
        mz = torch.tensor([[[100.0], [200.0], [300.0]]])  # (1, 3, 1)
        out = module(mz)
        # feats[0,1] should differ from feats[1,0] (signed Δm/z flips sign)
        assert not torch.allclose(out[0, 0, 1], out[0, 1, 0]), \
            "Pairwise features should be asymmetric (signed differences)"

    def test_diagonal_is_zero_input(self):
        """Diagonal entries (i==i) should have Δm/z=0, producing identical features."""
        module = PairwiseAttentionBias(num_freqs=16, hidden_dim=32)
        mz = torch.tensor([[[100.0], [200.0], [300.0]]])
        out = module(mz)
        # All diagonal entries should be identical (same Δm/z=0 input)
        assert torch.allclose(out[0, 0, 0], out[0, 1, 1], atol=1e-6)
        assert torch.allclose(out[0, 0, 0], out[0, 2, 2], atol=1e-6)

    def test_fourier_frequency_range(self):
        """Frequencies should span from 2π/λ_max to 2π/λ_min."""
        import math
        lambda_min, lambda_max = 0.01, 1000.0
        module = PairwiseAttentionBias(
            num_freqs=8, hidden_dim=16, lambda_min=lambda_min, lambda_max=lambda_max
        )
        freqs = module.freqs
        assert freqs.shape == (8,)
        assert abs(freqs[0].item() - 2 * math.pi / lambda_max) < 1e-3
        assert abs(freqs[-1].item() - 2 * math.pi / lambda_min) < 1e-2

    def test_gradient_flow(self):
        """Gradients should flow through the module."""
        module = PairwiseAttentionBias(num_freqs=8, hidden_dim=16)
        mz = torch.rand(1, 5, 1) * 1000
        mz = mz.detach().requires_grad_(True)  # Make it a leaf tensor
        out = module(mz)
        loss = out.sum()
        loss.backward()
        assert mz.grad is not None
        # Check f_pw parameters got gradients
        for p in module.f_pw.parameters():
            assert p.grad is not None

    def test_different_batch_sizes(self):
        """Module should handle various batch sizes."""
        module = PairwiseAttentionBias(num_freqs=8, hidden_dim=16)
        for B in [1, 4, 16]:
            mz = torch.rand(B, 10, 1) * 1000
            out = module(mz)
            assert out.shape == (B, 10, 10, 16)

    def test_state_dict_keys(self):
        """State dict keys should match expected f_pw MLP structure."""
        module = PairwiseAttentionBias(num_freqs=32, hidden_dim=32)
        keys = set(module.state_dict().keys())
        expected = {
            'freqs',
            'f_pw.0.weight', 'f_pw.0.bias',
            'f_pw.2.weight', 'f_pw.2.bias',
        }
        assert keys == expected
