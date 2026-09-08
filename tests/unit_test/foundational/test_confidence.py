"""
Unit tests for classifier confidence scoring.

Covers:
1. mask_invalid_offsets helper
2. compute_classifier_confidence output shape/keys
3. Expected Calibration Error (ECE)
"""

import pytest
import torch

from instanovo_fm.trainer.losses import (
    mask_invalid_offsets,
    compute_classifier_confidence,
)
from instanovo_fm.trainer.calibration import compute_ece


class TestMaskInvalidOffsets:
    """Test the mask_invalid_offsets helper function."""

    def test_mask_invalid_offsets_basic(self):
        """Test basic masking of invalid offsets in last group."""
        offset_logits = torch.randn(10, 100)
        group_pred = torch.randint(0, 4, (10,))

        group_pred[0] = 4
        group_pred[5] = 4

        n_groups = 5
        last_group_size = 50

        masked = mask_invalid_offsets(offset_logits, group_pred, n_groups, last_group_size)

        assert torch.isinf(masked[0, last_group_size:]).all()
        assert torch.isinf(masked[5, last_group_size:]).all()

        assert not torch.isinf(masked[1, :]).any()
        assert not torch.isinf(masked[2, :]).any()

    def test_mask_invalid_offsets_no_last_group(self):
        """Test that masking does nothing when no samples in last group."""
        offset_logits = torch.randn(10, 100)
        group_pred = torch.randint(0, 4, (10,))

        n_groups = 5
        last_group_size = 50

        masked = mask_invalid_offsets(offset_logits, group_pred, n_groups, last_group_size)

        assert torch.equal(masked, offset_logits)

    def test_mask_invalid_offsets_all_last_group(self):
        """Test masking when all samples are in last group."""
        offset_logits = torch.randn(10, 100)
        group_pred = torch.full((10,), 4)

        n_groups = 5
        last_group_size = 50

        masked = mask_invalid_offsets(offset_logits, group_pred, n_groups, last_group_size)

        assert torch.isinf(masked[:, last_group_size:]).all()
        assert not torch.isinf(masked[:, :last_group_size]).any()


class TestECE:
    """Test Expected Calibration Error."""

    def test_ece_well_calibrated(self):
        """ECE should be low when confidence tracks accuracy."""
        n_samples = 1000
        confidences = torch.rand(n_samples)

        predictions = torch.zeros(n_samples, dtype=torch.long)
        targets = torch.zeros(n_samples, dtype=torch.long)

        for i in range(n_samples):
            if torch.rand(1).item() < confidences[i].item():
                targets[i] = 0
                predictions[i] = 0
            else:
                targets[i] = 1
                predictions[i] = 0

        ece, _, _, _ = compute_ece(confidences, predictions, targets, n_bins=10)

        assert ece < 0.2

    def test_ece_overconfident(self):
        """ECE should be high when confidence >> accuracy."""
        confidences = torch.full((100,), 0.9)
        predictions = torch.zeros(100, dtype=torch.long)
        targets = torch.randint(0, 2, (100,))

        ece, _, _, _ = compute_ece(confidences, predictions, targets, n_bins=10)

        assert ece > 0.2


class TestClassifierConfidenceIntegration:
    """Test compute_classifier_confidence."""

    def test_output_keys_and_shapes(self):
        B, L = 4, 50
        n_groups = 125
        group_size = 100
        last_group_size = 50

        group_logits = torch.randn(B, L, n_groups)
        offset_logits = torch.randn(B, L, group_size)

        conf_dict = compute_classifier_confidence(
            group_logits, offset_logits, n_groups, last_group_size
        )

        assert set(conf_dict.keys()) == {"conf_group", "conf_offset", "conf_joint"}
        for key, tensor in conf_dict.items():
            assert tensor.shape == (B, L), f"{key} wrong shape"

    def test_joint_is_product(self):
        B, L = 2, 10
        n_groups = 5
        group_size = 50
        last_group_size = 50

        group_logits = torch.randn(B, L, n_groups)
        offset_logits = torch.randn(B, L, group_size)

        conf_dict = compute_classifier_confidence(
            group_logits, offset_logits, n_groups, last_group_size
        )

        assert torch.allclose(
            conf_dict["conf_joint"],
            conf_dict["conf_group"] * conf_dict["conf_offset"],
        )

    def test_values_in_unit_interval(self):
        B, L = 2, 10
        n_groups = 5
        group_size = 50
        last_group_size = 50

        group_logits = torch.randn(B, L, n_groups)
        offset_logits = torch.randn(B, L, group_size)

        conf_dict = compute_classifier_confidence(
            group_logits, offset_logits, n_groups, last_group_size
        )

        for key, tensor in conf_dict.items():
            assert (tensor >= 0).all() and (tensor <= 1).all(), f"{key} out of [0, 1]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
