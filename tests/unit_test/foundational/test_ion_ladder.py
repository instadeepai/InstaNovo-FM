"""Tests for IonLadderEncoder."""

import torch
import pytest

from instanovo_fm.model.ion_ladder import IonLadderEncoder


@pytest.fixture
def residue_masses():
    """Subset of standard amino acid masses for testing."""
    return [57.021, 71.037, 87.032, 97.053, 99.068, 101.048, 103.009,
            113.084, 114.043, 115.027, 128.059, 128.095, 129.043,
            131.040, 137.059, 147.068, 156.101, 163.063, 186.079]


@pytest.fixture
def ion_ladder(residue_masses):
    return IonLadderEncoder(
        d_model=64,
        residue_masses=residue_masses,
        window_k=10,
        sigma=0.35,
        charge_states=(1, 2),
    )


class TestIonLadderEncoder:
    def test_output_shape(self, ion_ladder):
        B, L, D = 2, 50, 64
        mz = torch.linspace(100, 1000, L).unsqueeze(0).expand(B, -1)
        embeddings = torch.randn(B, L, D)
        out = ion_ladder(mz, embeddings)
        assert out.shape == (B, L, D)

    def test_rezero_gate_at_init(self, ion_ladder):
        """At init, gate_scale=0 → sigmoid(0)=0.5, but gate_scale is zeros."""
        assert ion_ladder.gate_scale.item() == 0.0

    def test_output_close_to_input_at_init(self, ion_ladder):
        """With gate_scale=0, sigmoid(0)=0.5, so output = input + 0.5*projected.
        With Xavier-init'd weights and zero-init gate, the contribution should be small
        relative to random embeddings. Verify the residual doesn't dominate."""
        B, L, D = 2, 50, 64
        mz = torch.linspace(100, 1000, L).unsqueeze(0).expand(B, -1)
        embeddings = torch.randn(B, L, D) * 10  # Large embeddings
        out = ion_ladder(mz, embeddings)
        diff = (out - embeddings).abs().mean()
        # The projected features should be modest relative to input
        assert diff < embeddings.abs().mean()

    def test_padding_handling(self, ion_ladder):
        B, L, D = 2, 50, 64
        mz = torch.linspace(100, 1000, L).unsqueeze(0).expand(B, -1)
        embeddings = torch.randn(B, L, D)

        # Mark last 10 positions as padding
        pad_mask = torch.zeros(B, L, dtype=torch.bool)
        pad_mask[:, 40:] = True

        out = ion_ladder(mz, embeddings, pad_mask=pad_mask)
        assert out.shape == (B, L, D)

    def test_different_window_k(self, residue_masses):
        for k in [5, 20, 40]:
            encoder = IonLadderEncoder(
                d_model=64, residue_masses=residue_masses, window_k=k,
            )
            mz = torch.linspace(100, 1000, 50).unsqueeze(0)
            embeddings = torch.randn(1, 50, 64)
            out = encoder(mz, embeddings)
            assert out.shape == (1, 50, 64)

    def test_backward_pass(self, ion_ladder):
        B, L, D = 2, 50, 64
        mz = torch.linspace(100, 1000, L).unsqueeze(0).expand(B, -1)
        embeddings = torch.randn(B, L, D, requires_grad=True)
        out = ion_ladder(mz, embeddings)
        loss = out.sum()
        loss.backward()
        assert embeddings.grad is not None
        assert embeddings.grad.shape == (B, L, D)

    def test_single_peak(self, ion_ladder):
        """Edge case: single peak per spectrum."""
        mz = torch.tensor([[500.0]])
        embeddings = torch.randn(1, 1, 64)
        out = ion_ladder(mz, embeddings)
        assert out.shape == (1, 1, 64)

    def test_all_padding(self, ion_ladder):
        """Edge case: all positions are padding."""
        B, L, D = 1, 10, 64
        mz = torch.zeros(B, L)
        embeddings = torch.randn(B, L, D)
        pad_mask = torch.ones(B, L, dtype=torch.bool)
        out = ion_ladder(mz, embeddings, pad_mask=pad_mask)
        assert out.shape == (B, L, D)

    def test_mlm_mask_excludes_masked_peaks(self, ion_ladder):
        """Masked peaks must not contribute to unmasked peaks' features.

        If peak B is masked, unmasked peak A should produce identical features
        regardless of B's m/z value — verifying no information leakage.
        """
        B, L, D = 1, 20, 64
        torch.manual_seed(42)
        embeddings = torch.randn(B, L, D)

        # Base m/z values: evenly spaced
        mz = torch.linspace(100, 500, L).unsqueeze(0)

        # Mask peak at index 5
        mlm_mask = torch.zeros(B, L, dtype=torch.bool)
        mlm_mask[:, 5] = True

        # Run with mlm_mask
        out_masked = ion_ladder(mz, embeddings, mlm_mask=mlm_mask)

        # Now change the masked peak's m/z to something very different
        mz_altered = mz.clone()
        mz_altered[:, 5] = 999.0  # Completely different value

        out_altered = ion_ladder(mz_altered, embeddings, mlm_mask=mlm_mask)

        # All UNMASKED peaks should produce identical features regardless
        # of what the masked peak's m/z was
        unmasked = ~mlm_mask.squeeze(0)
        assert torch.allclose(
            out_masked[:, unmasked], out_altered[:, unmasked], atol=1e-6
        ), "Unmasked peaks' features changed when a masked peak's m/z changed — leakage!"

    def test_mlm_mask_no_leakage_vs_removed(self, ion_ladder):
        """Features for unmasked peaks with mlm_mask should equal features
        computed with the masked peak entirely absent (replaced by padding)."""
        B, L, D = 1, 10, 64
        torch.manual_seed(42)
        embeddings = torch.randn(B, L, D)

        # m/z with peak at index 3 that will be masked
        mz = torch.linspace(100, 300, L).unsqueeze(0)

        mlm_mask = torch.zeros(B, L, dtype=torch.bool)
        mlm_mask[:, 3] = True

        # Run with mlm_mask
        out_masked = ion_ladder(mz, embeddings, mlm_mask=mlm_mask)

        # Run with peak 3 zeroed out as if it were padding
        mz_removed = mz.clone()
        mz_removed[:, 3] = 0.0
        pad_mask = torch.zeros(B, L, dtype=torch.bool)
        pad_mask[:, 3] = True

        out_removed = ion_ladder(mz_removed, embeddings, pad_mask=pad_mask)

        # Unmasked peaks should have identical features in both cases
        check_idx = [i for i in range(L) if i != 3]
        assert torch.allclose(
            out_masked[:, check_idx], out_removed[:, check_idx], atol=1e-6
        ), "mlm_mask should be equivalent to treating peak as padding"

    def test_neutral_losses_expand_feature_dim(self, residue_masses):
        """Neutral losses should expand the reference mass table."""
        encoder_bare = IonLadderEncoder(
            d_model=64, residue_masses=residue_masses, window_k=10,
        )
        encoder_losses = IonLadderEncoder(
            d_model=64, residue_masses=residue_masses, window_k=10,
            neutral_losses={"H2O": 18.010565, "NH3": 17.026549},
        )
        # With 2 losses, reference masses = bare + H2O-shifted + NH3-shifted = 3x
        n_bare = len(encoder_bare.residue_masses_buf)
        n_with_losses = len(encoder_losses.residue_masses_buf)
        assert n_with_losses == 3 * n_bare

        # Both should still produce correct output shape
        B, L, D = 2, 50, 64
        mz = torch.linspace(100, 1000, L).unsqueeze(0).expand(B, -1)
        embeddings = torch.randn(B, L, D)
        assert encoder_losses(mz, embeddings).shape == (B, L, D)

    def test_neutral_losses_none_equals_bare(self, residue_masses):
        """Passing neutral_losses=None should be identical to bare encoder."""
        enc_none = IonLadderEncoder(
            d_model=64, residue_masses=residue_masses, window_k=10,
            neutral_losses=None,
        )
        enc_bare = IonLadderEncoder(
            d_model=64, residue_masses=residue_masses, window_k=10,
        )
        assert len(enc_none.residue_masses_buf) == len(enc_bare.residue_masses_buf)

    def test_neutral_losses_detects_shifted_match(self, residue_masses):
        """A peak pair differing by residue_mass + H2O should have a strong
        Gaussian match in the loss-expanded reference masses."""
        gly_mass = 57.021
        h2o_mass = 18.010565
        sigma = 0.35

        encoder = IonLadderEncoder(
            d_model=64, residue_masses=residue_masses, window_k=10,
            neutral_losses={"H2O": h2o_mass},
        )

        # The reference mass table should contain gly_mass + h2o_mass
        ref = encoder.residue_masses_buf
        target = gly_mass + h2o_mass
        # Find closest reference mass to target
        min_dist = (ref - target).abs().min().item()
        # Should be within sigma (strong Gaussian activation)
        assert min_dist < sigma, (
            f"Expected reference mass near {target:.3f} Da (Gly+H2O), "
            f"closest was {min_dist:.4f} Da away (sigma={sigma})"
        )

    def test_without_mlm_mask_sees_all_peaks(self, ion_ladder):
        """Without mlm_mask, changing a peak's m/z SHOULD affect neighbours.

        We place two peaks exactly one Glycine mass (57.021 Da) apart, then
        move the second peak away. The first peak's IonLadder features should
        change because it loses a strong Glycine soft-match.
        """
        B, D = 1, 64
        torch.manual_seed(42)

        # Place peaks such that peak 0 and peak 1 are ~57 Da apart (Glycine match)
        mz = torch.tensor([[200.0, 257.021, 400.0, 500.0, 600.0]])
        L = mz.shape[1]
        embeddings = torch.randn(B, L, D)

        out_original = ion_ladder(mz, embeddings)

        # Move peak 1 far away — peak 0 loses its Glycine-matching neighbour
        mz_altered = mz.clone()
        mz_altered[:, 1] = 999.0
        out_altered = ion_ladder(mz_altered, embeddings)

        # Peak 0's features SHOULD change (it lost a strong residue-mass match)
        assert not torch.allclose(
            out_original[:, 0], out_altered[:, 0], atol=1e-6
        ), "Without mlm_mask, neighbours should see each other's m/z"
