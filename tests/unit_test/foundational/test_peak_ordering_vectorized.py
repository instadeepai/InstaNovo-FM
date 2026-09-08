"""Tests for vectorized peak ordering in FoundationalDataProcessor.

Validates that the vectorized apply_peak_ordering produces correct results
for all three strategies: sorted, cyclic_shift, and complete_shuffle.
"""

import pytest
import torch

from instanovo_fm.data.data import FoundationalDataProcessor


def _make_processor(ordering: str) -> FoundationalDataProcessor:
    """Create a minimal processor with a specific peak ordering strategy."""
    return FoundationalDataProcessor(
        n_peaks=10,
        peak_ordering=ordering,
        masking_strategy="none",
        annotated=False,
        use_spectrum_utils=False,
    )


class TestSortedOrdering:
    """Tests for the 'sorted' peak ordering strategy."""

    def test_basic_sort(self):
        """Peaks should be sorted by ascending m/z."""
        proc = _make_processor("sorted")
        # B=2, L=5, C=2: unsorted m/z values
        spectra = torch.tensor([
            [[0.5, 1.0], [0.1, 0.5], [0.3, 0.8], [0.0, 0.0], [0.0, 0.0]],
            [[0.9, 0.2], [0.2, 0.7], [0.6, 0.3], [0.4, 0.9], [0.0, 0.0]],
        ])
        mask = torch.tensor([
            [False, False, False, True, True],
            [False, False, False, False, True],
        ])

        result = proc.apply_peak_ordering(spectra, mask)

        # Spectrum 0: valid=[0.5, 0.1, 0.3] → sorted=[0.1, 0.3, 0.5]
        assert torch.allclose(result[0, 0, 0], torch.tensor(0.1))
        assert torch.allclose(result[0, 1, 0], torch.tensor(0.3))
        assert torch.allclose(result[0, 2, 0], torch.tensor(0.5))
        # Padding should stay at the end
        assert torch.allclose(result[0, 3], torch.tensor([0.0, 0.0]))
        assert torch.allclose(result[0, 4], torch.tensor([0.0, 0.0]))

        # Spectrum 1: valid=[0.9, 0.2, 0.6, 0.4] → sorted=[0.2, 0.4, 0.6, 0.9]
        assert torch.allclose(result[1, 0, 0], torch.tensor(0.2))
        assert torch.allclose(result[1, 1, 0], torch.tensor(0.4))
        assert torch.allclose(result[1, 2, 0], torch.tensor(0.6))
        assert torch.allclose(result[1, 3, 0], torch.tensor(0.9))

    def test_already_sorted(self):
        """Already-sorted spectra should remain unchanged."""
        proc = _make_processor("sorted")
        spectra = torch.tensor([
            [[0.1, 0.5], [0.3, 0.8], [0.5, 1.0], [0.0, 0.0], [0.0, 0.0]],
        ])
        mask = torch.tensor([[False, False, False, True, True]])

        result = proc.apply_peak_ordering(spectra, mask)
        assert torch.allclose(result, spectra)

    def test_single_peak(self):
        """A spectrum with a single valid peak should remain unchanged."""
        proc = _make_processor("sorted")
        spectra = torch.tensor([
            [[0.5, 1.0], [0.0, 0.0], [0.0, 0.0]],
        ])
        mask = torch.tensor([[False, True, True]])

        result = proc.apply_peak_ordering(spectra, mask)
        assert torch.allclose(result[0, 0], torch.tensor([0.5, 1.0]))

    def test_all_padding(self):
        """A fully padded spectrum should stay all zeros."""
        proc = _make_processor("sorted")
        spectra = torch.zeros(1, 5, 2)
        mask = torch.ones(1, 5, dtype=torch.bool)

        result = proc.apply_peak_ordering(spectra, mask)
        assert torch.allclose(result, spectra)

    def test_intensities_follow_mz_sort(self):
        """Intensities should follow their paired m/z values after sorting."""
        proc = _make_processor("sorted")
        spectra = torch.tensor([
            [[0.9, 0.1], [0.1, 0.9], [0.5, 0.5], [0.0, 0.0]],
        ])
        mask = torch.tensor([[False, False, False, True]])

        result = proc.apply_peak_ordering(spectra, mask)
        # After sort by m/z: [0.1, 0.9], [0.5, 0.5], [0.9, 0.1]
        assert torch.allclose(result[0, 0], torch.tensor([0.1, 0.9]))
        assert torch.allclose(result[0, 1], torch.tensor([0.5, 0.5]))
        assert torch.allclose(result[0, 2], torch.tensor([0.9, 0.1]))

    def test_full_spectrum_no_padding(self):
        """Spectrum with no padding should sort correctly."""
        proc = _make_processor("sorted")
        spectra = torch.tensor([
            [[0.3, 0.5], [0.1, 0.8], [0.2, 0.3]],
        ])
        mask = torch.zeros(1, 3, dtype=torch.bool)

        result = proc.apply_peak_ordering(spectra, mask)
        assert torch.allclose(result[0, 0, 0], torch.tensor(0.1))
        assert torch.allclose(result[0, 1, 0], torch.tensor(0.2))
        assert torch.allclose(result[0, 2, 0], torch.tensor(0.3))


class TestCyclicShiftOrdering:
    """Tests for the 'cyclic_shift' peak ordering strategy."""

    def test_output_is_permutation_of_sorted(self):
        """Cyclic shift should produce a valid rotation of the sorted spectrum."""
        proc = _make_processor("cyclic_shift")
        spectra = torch.tensor([
            [[0.5, 1.0], [0.1, 0.5], [0.3, 0.8], [0.0, 0.0], [0.0, 0.0]],
        ])
        mask = torch.tensor([[False, False, False, True, True]])

        # Run multiple times to check invariants
        for _ in range(10):
            result = proc.apply_peak_ordering(spectra, mask)

            # Extract valid peaks' m/z values
            valid_mz = result[0, :3, 0]

            # Sorted version
            sorted_mz = torch.tensor([0.1, 0.3, 0.5])

            # Check it's a cyclic rotation of sorted
            # All values should be present
            assert set(valid_mz.tolist()) == set(sorted_mz.tolist())

            # Padding should remain at end
            assert torch.allclose(result[0, 3], torch.tensor([0.0, 0.0]))
            assert torch.allclose(result[0, 4], torch.tensor([0.0, 0.0]))

    def test_single_peak_unchanged(self):
        """Single peak should be unchanged by cyclic shift."""
        proc = _make_processor("cyclic_shift")
        spectra = torch.tensor([
            [[0.5, 1.0], [0.0, 0.0]],
        ])
        mask = torch.tensor([[False, True]])

        result = proc.apply_peak_ordering(spectra, mask)
        assert torch.allclose(result[0, 0], torch.tensor([0.5, 1.0]))


class TestCompleteShuffleOrdering:
    """Tests for the 'complete_shuffle' peak ordering strategy."""

    def test_output_is_permutation_of_sorted(self):
        """Complete shuffle should produce a valid permutation of sorted values."""
        proc = _make_processor("complete_shuffle")
        spectra = torch.tensor([
            [[0.5, 1.0], [0.1, 0.5], [0.3, 0.8], [0.7, 0.2], [0.0, 0.0]],
        ])
        mask = torch.tensor([[False, False, False, False, True]])

        sorted_mz = torch.tensor([0.1, 0.3, 0.5, 0.7])

        for _ in range(10):
            result = proc.apply_peak_ordering(spectra, mask)

            valid_mz = result[0, :4, 0]
            # Same values, possibly different order
            assert sorted(valid_mz.tolist()) == sorted(sorted_mz.tolist())

            # Padding at end
            assert torch.allclose(result[0, 4], torch.tensor([0.0, 0.0]))

    def test_intensities_stay_paired(self):
        """Each peak's intensity should stay with its m/z after shuffling."""
        proc = _make_processor("complete_shuffle")
        # Unique intensity for each peak to verify pairing
        spectra = torch.tensor([
            [[0.3, 0.33], [0.1, 0.11], [0.2, 0.22], [0.0, 0.0]],
        ])
        mask = torch.tensor([[False, False, False, True]])

        known_pairs = {0.1: 0.11, 0.2: 0.22, 0.3: 0.33}

        for _ in range(10):
            result = proc.apply_peak_ordering(spectra, mask)
            for i in range(3):
                mz_val = round(result[0, i, 0].item(), 2)
                int_val = round(result[0, i, 1].item(), 2)
                assert known_pairs[mz_val] == int_val, (
                    f"Peak at position {i}: m/z={mz_val} paired with "
                    f"intensity={int_val}, expected {known_pairs[mz_val]}"
                )

    def test_all_padding_unchanged(self):
        """Fully padded spectrum should remain all zeros."""
        proc = _make_processor("complete_shuffle")
        spectra = torch.zeros(1, 5, 2)
        mask = torch.ones(1, 5, dtype=torch.bool)

        result = proc.apply_peak_ordering(spectra, mask)
        assert torch.allclose(result, spectra)


class TestBatchConsistency:
    """Tests that vectorized ordering handles batches correctly."""

    def test_batch_independence(self):
        """Each spectrum in the batch should be ordered independently."""
        proc = _make_processor("sorted")
        spectra = torch.tensor([
            [[0.5, 1.0], [0.1, 0.5], [0.0, 0.0]],
            [[0.2, 0.3], [0.8, 0.7], [0.4, 0.1]],
        ])
        mask = torch.tensor([
            [False, False, True],
            [False, False, False],
        ])

        result = proc.apply_peak_ordering(spectra, mask)

        # Spectrum 0: [0.1, 0.5] before [0.5, 1.0]
        assert torch.allclose(result[0, 0, 0], torch.tensor(0.1))
        assert torch.allclose(result[0, 1, 0], torch.tensor(0.5))

        # Spectrum 1: [0.2, 0.4, 0.8]
        assert torch.allclose(result[1, 0, 0], torch.tensor(0.2))
        assert torch.allclose(result[1, 1, 0], torch.tensor(0.4))
        assert torch.allclose(result[1, 2, 0], torch.tensor(0.8))

    def test_large_batch(self):
        """Vectorized ordering should handle larger batches without error."""
        proc = _make_processor("sorted")
        B, L = 128, 200
        spectra = torch.rand(B, L, 2)
        # Random padding: last 10-50 positions per spectrum
        mask = torch.zeros(B, L, dtype=torch.bool)
        for i in range(B):
            pad_start = torch.randint(150, 200, (1,)).item()
            mask[i, pad_start:] = True
            spectra[i, pad_start:] = 0.0

        result = proc.apply_peak_ordering(spectra, mask)

        # Verify sorted order for each spectrum
        for i in range(B):
            n_valid = (~mask[i]).sum().item()
            if n_valid > 1:
                valid_mz = result[i, :n_valid, 0]
                # Should be non-decreasing
                assert (valid_mz[1:] >= valid_mz[:-1]).all(), (
                    f"Spectrum {i}: m/z not sorted: {valid_mz}"
                )
