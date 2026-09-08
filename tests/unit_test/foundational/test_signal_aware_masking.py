"""Tests for signal-aware fragment masking strategy."""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from instanovo_fm.data.masking import (
    signal_aware_fragment_mask,
    _build_parent_child_mapping,
    _select_fragment_ions_uniform,
    _enforce_mask_cap,
)


class TestParentChildMapping:
    """Tests for parent-child mapping construction."""

    def test_basic_mapping(self):
        """Test basic parent-child mapping with losses and isotopes."""
        # Setup: 3 base fragments, each with 1 loss and 1 isotope
        feature_types = [
            "base", "loss", "isotope",  # b3+
            "base", "loss", "isotope",  # y5++
            "base", "loss", "isotope",  # b7+
            None, None, None,  # Padding/unmatched
        ]
        parent_annotations = [
            None, "b3+", "b3+",
            None, "y5++", "y5++",
            None, "b7+", "b7+",
            None, None, None,
        ]
        matched_annotations = [
            "b3+", "b3+-H2O", "b3+[+1]",
            "y5++", "y5++-NH3", "y5++[+1]",
            "b7+", "b7+-H2O", "b7+[+1]",
            None, None, None,
        ]
        intensity = torch.tensor([1.0, 0.5, 0.3, 0.9, 0.4, 0.2, 0.8, 0.3, 0.1, 0.0, 0.0, 0.0])
        spectra_mask = torch.tensor([False] * 9 + [True] * 3)

        mapping = _build_parent_child_mapping(
            feature_types, parent_annotations, matched_annotations, intensity, spectra_mask
        )

        assert len(mapping) == 3
        assert "b3+" in mapping
        assert "y5++" in mapping
        assert "b7+" in mapping

        # Check b3+ group
        assert mapping["b3+"]["parent_idx"] == 0
        assert mapping["b3+"]["loss_indices"] == [1]
        assert mapping["b3+"]["isotope_indices"] == [2]

        # Check y5++ group
        assert mapping["y5++"]["parent_idx"] == 3
        assert mapping["y5++"]["loss_indices"] == [4]
        assert mapping["y5++"]["isotope_indices"] == [5]

    def test_no_children(self):
        """Test fragment with no loss/isotope children."""
        feature_types = ["base", "base", None]
        parent_annotations = [None, None, None]
        matched_annotations = ["b3+", "y5+", None]
        intensity = torch.tensor([1.0, 0.9, 0.0])
        spectra_mask = torch.tensor([False, False, True])

        mapping = _build_parent_child_mapping(
            feature_types, parent_annotations, matched_annotations, intensity, spectra_mask
        )

        assert len(mapping) == 2
        assert mapping["b3+"]["loss_indices"] == []
        assert mapping["b3+"]["isotope_indices"] == []

    def test_orphan_children_ignored(self):
        """Test that orphan children (no parent) are ignored."""
        feature_types = ["base", "loss", "isotope"]
        parent_annotations = [None, "b999+", "b999+"]  # Parent doesn't exist
        matched_annotations = ["b3+", "b999+-H2O", "b999+[+1]"]
        intensity = torch.tensor([1.0, 0.5, 0.3])
        spectra_mask = torch.tensor([False, False, False])

        mapping = _build_parent_child_mapping(
            feature_types, parent_annotations, matched_annotations, intensity, spectra_mask
        )

        # Only b3+ should be mapped, orphans ignored
        assert len(mapping) == 1
        assert "b3+" in mapping
        assert mapping["b3+"]["loss_indices"] == []
        assert mapping["b3+"]["isotope_indices"] == []


class TestFragmentSelection:
    """Tests for fragment ion selection."""

    def test_uniform_selection_basic(self):
        """Test uniform random selection of fragments."""
        parent_groups = {
            "b3+": {"parent_idx": 0, "parent_intensity": 1.0, "loss_indices": [1], "isotope_indices": [2]},
            "y5+": {"parent_idx": 3, "parent_intensity": 0.9, "loss_indices": [4], "isotope_indices": [5]},
            "b7+": {"parent_idx": 6, "parent_intensity": 0.8, "loss_indices": [7], "isotope_indices": [8]},
        }
        mask_portion = 0.33  # ~1/3 of fragments
        n_valid = 9
        max_ratio = 0.50

        selected = _select_fragment_ions_uniform(parent_groups, mask_portion, n_valid, max_ratio)

        # Should select 1 fragment (33% of 3)
        assert len(selected) == 1
        assert selected[0] in parent_groups

    def test_selection_respects_cap(self):
        """Test selection respects max_total_mask_ratio cap."""
        # 3 fragments, each with 2 children (3 peaks each)
        parent_groups = {
            "b3+": {"parent_idx": 0, "parent_intensity": 1.0, "loss_indices": [1], "isotope_indices": [2]},
            "y5+": {"parent_idx": 3, "parent_intensity": 0.9, "loss_indices": [4], "isotope_indices": [5]},
            "b7+": {"parent_idx": 6, "parent_intensity": 0.8, "loss_indices": [7], "isotope_indices": [8]},
        }
        mask_portion = 1.0  # Try to select all
        n_valid = 9
        max_ratio = 0.35  # Cap at 35% → max 3 peaks

        selected = _select_fragment_ions_uniform(parent_groups, mask_portion, n_valid, max_ratio)

        # Should select only 1 fragment (3 peaks = 33%, within cap)
        assert len(selected) <= 1

    def test_empty_groups(self):
        """Test selection with no fragment groups."""
        selected = _select_fragment_ions_uniform({}, 0.30, 10, 0.50)
        assert selected == []


class TestMaskCapEnforcement:
    """Tests for hard cap enforcement."""

    def test_enforce_cap_removes_lowest_intensity(self):
        """Test that cap enforcement removes lowest-intensity peaks."""
        batch_mask = torch.tensor([True, True, True, True, False, False])
        spectra_mask = torch.tensor([False, False, False, False, False, True])
        max_ratio = 0.40  # 40% of 5 valid = 2 peaks max
        intensity = torch.tensor([1.0, 0.3, 0.9, 0.2, 0.0, 0.0])

        result = _enforce_mask_cap(batch_mask, spectra_mask, max_ratio, intensity)

        # Should keep 2 highest-intensity: positions 0 (1.0) and 2 (0.9)
        # Should remove: positions 1 (0.3) and 3 (0.2)
        assert result.sum().item() == 2
        assert result[0] == True
        assert result[2] == True
        assert result[1] == False
        assert result[3] == False

    def test_under_cap_no_change(self):
        """Test that under-cap masks are unchanged."""
        batch_mask = torch.tensor([True, False, True, False])
        spectra_mask = torch.tensor([False, False, False, False])
        max_ratio = 0.60  # 60% of 4 = 2.4 → allows 2 peaks
        intensity = torch.tensor([1.0, 0.5, 0.8, 0.3])

        result = _enforce_mask_cap(batch_mask, spectra_mask, max_ratio, intensity)

        # Should be unchanged
        assert torch.equal(result, batch_mask)


class TestSignalAwareFragmentMask:
    """Integration tests for signal_aware_fragment_mask."""

    @pytest.fixture
    def mock_annotation(self):
        """Mock annotation result."""
        return {
            "feature_type": ["base", "loss", "isotope", "base", "loss", None, None],
            "parent_annotation": [None, "b3+", "b3+", None, "y5+", None, None],
            "matched_annotation": ["b3+", "b3+-H2O", "b3+[+1]", "y5+", "y5+-NH3", None, None],
            "mask": np.array([True, True, True, True, True, False, False]),
            "metrics": {"n_base": 2, "n_losses": 2, "n_isotopes": 1},
        }

    def test_fallback_no_peptides(self):
        """Test fallback to thompson_span when no peptides provided."""
        intensity = torch.rand(2, 10)
        spectra_mask = torch.zeros(2, 10, dtype=torch.bool)
        mz = torch.rand(2, 10) * 2500
        charges = torch.tensor([2, 3])

        with patch("instanovo_fm.data.masking.thompson_sampling_span_mask") as mock_thompson:
            mock_thompson.return_value = torch.zeros(2, 10, dtype=torch.bool)

            mask = signal_aware_fragment_mask(
                intensity=intensity,
                spectra_mask=spectra_mask,
                mz=mz,
                charges=charges,
                peptides=None,  # No peptides
            )

            # Should call thompson_span fallback
            assert mock_thompson.called

    def test_fallback_low_quality(self, mock_annotation):
        """Test fallback when backbone coverage / fragment groups below threshold."""
        # Mock annotation with only 2 base fragments (b3+, y5+) → 2 groups, below min_fragment_groups=3
        mock_annotation["metrics"]["n_base"] = 2

        intensity = torch.rand(1, 10)
        spectra_mask = torch.zeros(1, 10, dtype=torch.bool)
        spectra_mask[0, 7:] = True  # Pad last 3
        mz = torch.rand(1, 10) * 2500
        charges = torch.tensor([2])
        peptides = ["PEPTIDE"]

        with patch("instanovo_fm.data.masking._annotate_single_spectrum_worker") as mock_worker:
            mock_worker.return_value = mock_annotation
            with patch("instanovo_fm.data.masking.thompson_sampling_span_mask") as mock_thompson:
                mock_thompson.return_value = torch.zeros(1, 10, dtype=torch.bool)

                mask = signal_aware_fragment_mask(
                    intensity=intensity,
                    spectra_mask=spectra_mask,
                    mz=mz,
                    charges=charges,
                    peptides=peptides,
                    min_backbone_coverage=0.15,
                    min_fragment_groups=3,  # Mock has only 2 groups → fallback
                    num_workers=1,  # Sequential for testing
                )

                # Should fallback due to insufficient fragment groups
                assert mock_thompson.called

    def test_signal_aware_masking_integration(self, mock_annotation):
        """Test full signal-aware masking with sufficient quality."""
        # Enrich mock with enough base fragments to pass quality gate
        # PEPTIDEK has seq_len=8, max_cleavage_sites=7
        # Need ≥3 fragment groups and ≥15% backbone coverage (≥2 cleavage sites)
        mock_annotation["feature_type"] = [
            "base", "loss", "isotope", "base", "loss", "base", None,
        ]
        mock_annotation["matched_annotation"] = [
            "b3+", "b3+-H2O", "b3+[+1]", "y5+", "y5+-NH3", "b5+", None,
        ]
        mock_annotation["parent_annotation"] = [
            None, "b3+", "b3+", None, "y5+", None, None,
        ]
        mock_annotation["metrics"]["n_base"] = 3

        intensity = torch.rand(1, 10)
        spectra_mask = torch.zeros(1, 10, dtype=torch.bool)
        spectra_mask[0, 7:] = True
        mz = torch.rand(1, 10) * 2500
        charges = torch.tensor([2])
        peptides = ["PEPTIDEK"]

        with patch("instanovo_fm.data.masking._annotate_single_spectrum_worker") as mock_worker:
            mock_worker.return_value = mock_annotation

            mask = signal_aware_fragment_mask(
                intensity=intensity,
                spectra_mask=spectra_mask,
                mz=mz,
                charges=charges,
                peptides=peptides,
                mask_portion=0.30,
                min_backbone_coverage=0.15,
                min_fragment_groups=3,
                num_workers=1,
            )

            # Should return a valid mask
            assert mask.shape == (1, 10)
            assert mask.dtype == torch.bool
            # Should have masked some peaks
            assert mask.any()

    def test_mz_denormalization(self):
        """Test that m/z is denormalized before annotation."""
        intensity = torch.rand(1, 5)
        spectra_mask = torch.zeros(1, 5, dtype=torch.bool)
        mz_normalized = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])  # Normalized [0,1]
        charges = torch.tensor([2])
        peptides = ["PEPTIDE"]

        with patch("instanovo_fm.data.masking._annotate_single_spectrum_worker") as mock_worker:
            mock_worker.return_value = None  # Will trigger fallback

            signal_aware_fragment_mask(
                intensity=intensity,
                spectra_mask=spectra_mask,
                mz=mz_normalized,
                charges=charges,
                peptides=peptides,
                max_mz=2500.0,
                normalize_mz=True,  # Should denormalize
                num_workers=1,
            )

            # Check that worker was called with denormalized m/z
            if mock_worker.called:
                args = mock_worker.call_args[0][0]
                mz_arg = args[0]
                # Should be in Daltons (mz_normalized * 2500)
                assert mz_arg[0] == pytest.approx(0.1 * 2500.0, rel=1e-3)


def test_registry_integration():
    """Test that signal_aware_fragment is registered correctly."""
    from instanovo_fm.data.masking import get_mask_function

    mask_fn = get_mask_function("signal_aware_fragment")
    assert mask_fn is signal_aware_fragment_mask
