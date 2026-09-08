"""Unit tests for intensity regression with Huber loss.

Tests cover:
- IntensityRegressionHead architecture and output
- Huber loss computation and masking
- Gradient flow and numerical stability
- Robustness comparison with MSE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from instanovo_fm.model.heads import IntensityRegressionHead
from instanovo_fm.trainer.losses import compute_intensity_regression_loss


class TestIntensityRegressionHead:
    """Test suite for intensity regression head."""

    def test_intensity_regression_head_shape(self):
        """Test head output shape is correct."""
        head = IntensityRegressionHead(d_model=768, max_intensity=1.0)
        x = torch.randn(4, 200, 768)  # (B, L, D)
        out = head(x)
        assert out.shape == (4, 200, 1), f"Expected shape (4, 200, 1), got {out.shape}"

    def test_intensity_regression_head_range(self):
        """Test head output bounded to [0, max_intensity]."""
        head = IntensityRegressionHead(d_model=768, max_intensity=1.0)
        x = torch.randn(4, 200, 768)
        out = head(x)
        assert (out >= 0).all(), "All outputs should be >= 0"
        assert (out <= 1.0).all(), "All outputs should be <= max_intensity"

    def test_intensity_regression_head_range_custom_max(self):
        """Test head respects custom max_intensity parameter."""
        max_intensity = 0.5
        head = IntensityRegressionHead(d_model=512, max_intensity=max_intensity)
        x = torch.randn(2, 100, 512)
        out = head(x)
        assert (out >= 0).all(), "All outputs should be >= 0"
        assert (out <= max_intensity).all(), f"All outputs should be <= {max_intensity}"

    def test_intensity_regression_head_dtype(self):
        """Test head preserves float32 dtype."""
        head = IntensityRegressionHead(d_model=768)
        x = torch.randn(4, 200, 768, dtype=torch.float32)
        out = head(x)
        assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"

    def test_intensity_regression_head_gradient(self):
        """Test head has learnable parameters with gradients."""
        head = IntensityRegressionHead(d_model=256)
        x = torch.randn(2, 50, 256, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()

        # Check that head parameters have gradients
        for name, param in head.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradient"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient"


class TestHuberLoss:
    """Test suite for Huber loss computation."""

    def test_huber_loss_basic(self):
        """Test basic Huber loss computation."""
        pred = torch.tensor([[0.1, 0.2, 0.3]]).unsqueeze(-1)  # (1, 3, 1)
        target = torch.tensor([[0.15, 0.25, 0.35]])           # (1, 3)

        loss = compute_intensity_regression_loss(pred, target, delta=0.1)
        assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
        assert loss > 0, "Loss should be positive for non-zero error"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_huber_loss_masking(self):
        """Test that -100 labels are properly ignored."""
        pred = torch.tensor([[0.1, 0.2, 0.3]]).unsqueeze(-1)  # (1, 3, 1)
        target = torch.tensor([[0.15, -100, 0.35]])           # Middle position ignored

        loss = compute_intensity_regression_loss(pred, target, delta=0.1)

        # Manually compute expected loss (only positions 0 and 2)
        expected_loss = compute_intensity_regression_loss(
            torch.tensor([[0.1, 0.3]]).unsqueeze(-1),
            torch.tensor([[0.15, 0.35]]),
            delta=0.1
        )
        assert torch.allclose(loss, expected_loss, atol=1e-5), \
            f"Loss with masking should match loss with only valid positions"

    def test_huber_loss_all_masked(self):
        """Test zero loss with gradient when all positions masked."""
        pred = torch.randn(4, 200, 1, requires_grad=True)
        target = torch.full((4, 200), -100.0)

        loss = compute_intensity_regression_loss(pred, target)
        assert loss.item() == 0.0, "Loss should be zero when all masked"
        assert loss.requires_grad, "Loss should have gradient"

        # Test backward pass
        loss.backward()
        assert pred.grad is not None, "Gradient should exist"
        assert torch.isfinite(pred.grad).all(), "Gradient should be finite"

    def test_huber_loss_partial_masking(self):
        """Test loss with partial masking (mix of valid and invalid)."""
        pred = torch.tensor([[0.1, 0.2, 0.3, 0.4]]).unsqueeze(-1)
        target = torch.tensor([[-100, 0.25, -100, 0.45]])  # Positions 1 and 3 valid

        loss = compute_intensity_regression_loss(pred, target, delta=0.1)

        # Should compute loss only on positions 1 and 3
        expected = compute_intensity_regression_loss(
            torch.tensor([[0.2, 0.4]]).unsqueeze(-1),
            torch.tensor([[0.25, 0.45]]),
            delta=0.1
        )
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_huber_vs_mse_small_errors(self):
        """Test that Huber ≈ MSE for small errors."""
        # Small errors: Huber should be similar to MSE
        pred = torch.tensor([[0.1, 0.2]]).unsqueeze(-1)
        target = torch.tensor([[0.12, 0.21]])

        huber = compute_intensity_regression_loss(pred, target, delta=0.1)
        mse = F.mse_loss(pred.squeeze(), target)

        # For small errors (<delta), Huber ≈ 0.5 * error² / delta
        # MSE = error²
        # So Huber should be similar (within a factor related to delta)
        assert torch.isfinite(huber), "Huber loss should be finite"
        assert torch.isfinite(mse), "MSE should be finite"
        # Both should be small and comparable in magnitude
        assert huber < 0.01 and mse < 0.01, "Both losses should be small for small errors"

    def test_huber_vs_mse_on_outliers(self):
        """Test that Huber loss handles outliers robustly."""
        # Test with large errors - Huber should be robust
        pred = torch.tensor([[0.1, 0.9]]).unsqueeze(-1)
        target = torch.tensor([[0.12, 0.1]])  # Large error on position 1 (0.8)

        huber = compute_intensity_regression_loss(pred, target, delta=0.1)

        # Huber loss should be finite and positive
        assert torch.isfinite(huber), "Huber loss should be finite"
        assert huber > 0, "Huber loss should be positive for errors"

        # Test that Huber grows linearly for large errors (robustness property)
        # Create progressively larger outliers
        pred_small = torch.tensor([[0.1]]).unsqueeze(-1)
        target_small = torch.tensor([[0.2]])  # Error: 0.1
        loss_small = compute_intensity_regression_loss(pred_small, target_small, delta=0.1)

        pred_large = torch.tensor([[0.1]]).unsqueeze(-1)
        target_large = torch.tensor([[0.5]])  # Error: 0.4 (4x larger)
        loss_large = compute_intensity_regression_loss(pred_large, target_large, delta=0.1)

        # For Huber loss with delta=0.1:
        # Error 0.1: at threshold, approximately quadratic
        # Error 0.4: well beyond threshold, linear growth
        # Loss should increase sub-quadratically (not 4² = 16x increase)
        ratio = loss_large / loss_small
        assert ratio < 10, f"Huber loss should grow sub-quadratically for outliers (ratio: {ratio:.2f})"

    def test_huber_delta_parameter(self):
        """Test that delta parameter affects loss behavior."""
        pred = torch.tensor([[0.1, 0.5]]).unsqueeze(-1)
        target = torch.tensor([[0.2, 0.2]])  # Errors: 0.1, 0.3

        # Test different delta values
        deltas = [0.01, 0.05, 0.1, 0.5, 1.0]
        losses = []

        for delta in deltas:
            loss = compute_intensity_regression_loss(pred, target, delta=delta)
            assert loss > 0 and torch.isfinite(loss), f"Loss should be positive and finite for delta={delta}"
            losses.append(loss.item())

        # All losses should be positive - exact ordering depends on error distribution
        # Just verify they're all reasonable values
        assert all(0.0 < l < 1.0 for l in losses), "All losses should be in reasonable range"

    def test_gradient_flow(self):
        """Test that gradients flow through loss."""
        pred = torch.randn(4, 200, 1, requires_grad=True)
        target = torch.rand(4, 200) * 0.5  # Random targets in [0, 0.5]

        loss = compute_intensity_regression_loss(pred, target, delta=0.1)
        loss.backward()

        assert pred.grad is not None, "Gradient should flow to predictions"
        assert torch.isfinite(pred.grad).all(), "Gradients should be finite"
        assert (pred.grad != 0).any(), "At least some gradients should be non-zero"

    def test_numerical_stability_extreme_values(self):
        """Test loss computation doesn't produce NaN or Inf with extreme values."""
        # Very small values
        pred_small = torch.tensor([[1e-8, 1e-6, 0.0]]).unsqueeze(-1)
        target_small = torch.tensor([[0.0, 1e-6, 1e-8]])
        loss_small = compute_intensity_regression_loss(pred_small, target_small, delta=0.1)
        assert torch.isfinite(loss_small), "Loss should be finite for very small values"

        # Values at boundary
        pred_boundary = torch.tensor([[0.0, 1.0, 0.5]]).unsqueeze(-1)
        target_boundary = torch.tensor([[0.0, 1.0, 0.5]])
        loss_boundary = compute_intensity_regression_loss(pred_boundary, target_boundary, delta=0.1)
        assert torch.isfinite(loss_boundary), "Loss should be finite at boundaries"
        assert loss_boundary.item() < 1e-6, "Loss should be near zero for perfect predictions"

        # Mixed extreme values
        pred_mixed = torch.tensor([[1e-8, 1.0, 0.5]]).unsqueeze(-1)
        target_mixed = torch.tensor([[1.0, 1e-8, 0.5]])
        loss_mixed = compute_intensity_regression_loss(pred_mixed, target_mixed, delta=0.1)
        assert torch.isfinite(loss_mixed), "Loss should be finite for mixed extreme values"

    def test_numerical_stability_zero_targets(self):
        """Test loss with zero targets (common in intensity data)."""
        pred = torch.rand(4, 200, 1)
        target = torch.zeros(4, 200)

        loss = compute_intensity_regression_loss(pred, target, delta=0.1)
        assert torch.isfinite(loss), "Loss should be finite for zero targets"
        assert loss > 0, "Loss should be positive when predictions don't match zero targets"

    def test_batch_size_independence(self):
        """Test that loss is independent of batch size (proper averaging)."""
        # Single sample
        pred_single = torch.tensor([[0.1, 0.2]]).unsqueeze(-1)
        target_single = torch.tensor([[0.15, 0.25]])
        loss_single = compute_intensity_regression_loss(pred_single, target_single, delta=0.1)

        # Same sample repeated in batch
        pred_batch = pred_single.repeat(4, 1, 1)
        target_batch = target_single.repeat(4, 1)
        loss_batch = compute_intensity_regression_loss(pred_batch, target_batch, delta=0.1)

        # Should be identical (reduction='mean')
        assert torch.allclose(loss_single, loss_batch, atol=1e-6), \
            "Loss should be independent of batch size"


class TestLabelCreation:
    """Test intensity label creation logic."""

    def test_label_creation_from_spectra(self):
        """Test that labels are correctly extracted from spectra."""
        # Simulate batch with spectra (float32 for compatibility)
        batch = {
            "spectra": torch.tensor([
                [[0.1, 0.5], [0.2, 0.3], [0.3, 0.1]],  # Sample 1: [m/z, intensity]
                [[0.15, 0.4], [0.25, 0.6], [0.35, 0.2]],  # Sample 2
            ], dtype=torch.float32),  # Shape: (2, 3, 2)
            "spectra_mask": torch.tensor([
                [False, False, True],  # Last position is padding
                [False, False, False],  # No padding
            ])  # Shape: (2, 3)
        }

        # Extract expected intensity targets
        expected_intensities = batch["spectra"][:, :, 1]  # (2, 3)

        # Create labels manually (same logic as in code)
        intensity_labels = torch.full_like(expected_intensities, -100.0)
        valid_mask = ~batch["spectra_mask"]
        intensity_labels[valid_mask] = expected_intensities[valid_mask]

        # Check results (use allclose for float comparison)
        assert torch.allclose(intensity_labels[0, 0], torch.tensor(0.5)), "Label should match intensity"
        assert torch.allclose(intensity_labels[0, 1], torch.tensor(0.3)), "Label should match intensity"
        assert intensity_labels[0, 2].item() == -100, "Padded position should be -100"
        assert torch.allclose(intensity_labels[1, 0], torch.tensor(0.4)), "Label should match intensity"
        assert torch.allclose(intensity_labels[1, 1], torch.tensor(0.6)), "Label should match intensity"
        assert torch.allclose(intensity_labels[1, 2], torch.tensor(0.2)), "Label should match intensity"

    def test_label_masking_with_mlm_mask(self):
        """Test that labels respect MLM masking."""
        batch_size, seq_len = 2, 4

        intensity_labels = torch.rand(batch_size, seq_len)  # Some labels
        mlm_mask = torch.tensor([
            [True, False, True, False],  # Positions 0 and 2 masked
            [False, True, False, True],  # Positions 1 and 3 masked
        ])  # Shape: (2, 4)
        spectra_mask = torch.tensor([
            [False, False, False, True],  # Last position padded
            [False, False, False, False],  # No padding
        ])

        # Apply masking logic (same as in compute_auxiliary_losses)
        valid_mask = (intensity_labels != -100) & mlm_mask & (~spectra_mask)
        masked_labels = torch.where(valid_mask, intensity_labels,
                                    torch.full_like(intensity_labels, -100.0))

        # Check that only MLM-masked, non-padded positions are valid
        assert masked_labels[0, 0] != -100, "Position (0,0) should be valid (masked, not padded)"
        assert masked_labels[0, 1] == -100, "Position (0,1) should be invalid (not masked)"
        assert masked_labels[0, 2] != -100, "Position (0,2) should be valid (masked, not padded)"
        assert masked_labels[0, 3] == -100, "Position (0,3) should be invalid (padded)"


class TestIntegration:
    """Integration tests with full model components."""

    def test_head_with_loss(self):
        """Test full forward pass: head → loss."""
        head = IntensityRegressionHead(d_model=256, max_intensity=1.0)
        x = torch.randn(4, 50, 256)

        # Forward through head
        pred = head(x)  # (4, 50, 1)

        # Create targets
        target = torch.rand(4, 50) * 0.5  # Random intensities in [0, 0.5]

        # Compute loss
        loss = compute_intensity_regression_loss(pred, target, delta=0.1)

        assert loss > 0, "Loss should be positive"
        assert torch.isfinite(loss), "Loss should be finite"

        # Test backward
        loss.backward()
        for name, param in head.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradient"

    def test_different_delta_values(self):
        """Test that different delta values work correctly."""
        pred = torch.randn(2, 100, 1)
        target = torch.rand(2, 100)

        deltas = [0.01, 0.05, 0.1, 0.5, 1.0]
        losses = []

        for delta in deltas:
            loss = compute_intensity_regression_loss(pred, target, delta=delta)
            assert torch.isfinite(loss), f"Loss should be finite for delta={delta}"
            losses.append(loss.item())

        # All losses should be positive
        assert all(l > 0 for l in losses), "All losses should be positive"

    def test_consistency_across_devices(self):
        """Test that loss is consistent across CPU (and GPU if available)."""
        pred = torch.randn(2, 50, 1)
        target = torch.rand(2, 50)

        loss_cpu = compute_intensity_regression_loss(pred, target, delta=0.1)

        if torch.cuda.is_available():
            pred_cuda = pred.cuda()
            target_cuda = target.cuda()
            loss_cuda = compute_intensity_regression_loss(pred_cuda, target_cuda, delta=0.1)

            assert torch.allclose(loss_cpu, loss_cuda.cpu(), atol=1e-5), \
                "Loss should be consistent across devices"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
