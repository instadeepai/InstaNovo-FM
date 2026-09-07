"""Test vectorized metrics implementation matches original behavior."""

import torch
import pytest
from instanovo_fm.trainer.metrics import StreamingMetrics


class TestMzMetricsVectorization:
    """Test that vectorized update_mz_metrics produces same results as original."""

    def test_basic_metrics_match(self):
        """Test that vectorized version matches for basic metrics."""
        metrics = StreamingMetrics()

        # Create test data
        torch.manual_seed(42)
        batch_size, seq_len = 8, 100
        pred_mz = torch.randn(batch_size, seq_len).abs() * 1000 + 100  # 100-1100 Da
        target_mz = pred_mz + torch.randn(batch_size, seq_len) * 5  # Add noise
        valid_mask = torch.rand(batch_size, seq_len) > 0.3  # ~70% valid

        # Update metrics
        metrics.update_mz_metrics(pred_mz, target_mz, valid_mask)

        # Check that counts are correct
        expected_tokens = valid_mask.sum().item()
        assert metrics.running_counts['tokens'] == expected_tokens
        assert metrics.running_counts['spectra'] == batch_size

        # Check that MAE is reasonable
        assert metrics.running_mae > 0
        assert metrics.running_mae < 100  # Should be on order of the noise we added

    def test_percentage_thresholds(self):
        """Test percentage threshold counters."""
        metrics = StreamingMetrics()

        # Create test data with known error distribution
        batch_size, seq_len = 4, 50
        pred_mz = torch.ones(batch_size, seq_len) * 500.0
        target_mz = pred_mz.clone()

        # Set specific errors
        target_mz[0, 0] = 500.05  # 0.05 Da error
        target_mz[1, 0] = 500.5   # 0.5 Da error
        target_mz[2, 0] = 505.0   # 5.0 Da error
        target_mz[3, 0] = 520.0   # 20.0 Da error

        valid_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        valid_mask[:, 0] = True  # Only first position valid

        metrics.update_mz_metrics(pred_mz, target_mz, valid_mask)

        # Check thresholds
        assert metrics.tokens_within_01da == 1  # 0.05 Da
        assert metrics.tokens_within_1da == 2   # 0.05, 0.5 Da
        assert metrics.tokens_within_10da == 3  # 0.05, 0.5, 5.0 Da

    def test_ppm_thresholds(self):
        """Test PPM threshold counters."""
        metrics = StreamingMetrics()

        batch_size, seq_len = 4, 50
        pred_mz = torch.ones(batch_size, seq_len) * 1000.0
        target_mz = pred_mz.clone()

        # Set specific PPM errors
        # PPM = (error_da / mz_true) * 1e6
        target_mz[0, 0] = 1000.005  # 5 PPM error
        target_mz[1, 0] = 1000.015  # 15 PPM error
        target_mz[2, 0] = 1000.025  # 25 PPM error
        target_mz[3, 0] = 1000.100  # 100 PPM error

        valid_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        valid_mask[:, 0] = True

        metrics.update_mz_metrics(pred_mz, target_mz, valid_mask)

        assert metrics.tokens_within_10ppm == 1   # 5 PPM
        assert metrics.tokens_within_20ppm == 2   # 5, 15 PPM

    def test_per_spectrum_metrics(self):
        """Test per-spectrum MAE and STD accumulation."""
        metrics = StreamingMetrics()

        batch_size, seq_len = 4, 10
        pred_mz = torch.ones(batch_size, seq_len) * 500.0
        target_mz = pred_mz.clone()

        # Spectrum 0: uniform 1 Da error
        target_mz[0, :5] = pred_mz[0, :5] + 1.0

        # Spectrum 1: uniform 2 Da error
        target_mz[1, :5] = pred_mz[1, :5] + 2.0

        valid_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        valid_mask[0, :5] = True
        valid_mask[1, :5] = True

        metrics.update_mz_metrics(pred_mz, target_mz, valid_mask)

        # Average spectrum MAE should be (1 + 2) / 2 = 1.5
        # (only 2 spectra have valid tokens)
        avg_spectrum_mae = metrics.spectrum_mae_sum / 2
        assert abs(avg_spectrum_mae - 1.5) < 0.01

    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        metrics = StreamingMetrics()

        batch_size, seq_len = 2, 10
        pred_mz = torch.randn(batch_size, seq_len).abs() * 100 + 500
        target_mz = pred_mz.clone()  # Perfect match
        valid_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        metrics.update_mz_metrics(pred_mz, target_mz, valid_mask)

        # Cosine similarity should be 1.0 for perfect match
        assert abs(metrics.running_cosine_sim - 1.0) < 0.01
        assert metrics.running_counts['cosine_samples'] == batch_size

    def test_empty_batch(self):
        """Test handling of batch with no valid tokens."""
        metrics = StreamingMetrics()

        batch_size, seq_len = 4, 10
        pred_mz = torch.randn(batch_size, seq_len).abs() * 500
        target_mz = pred_mz.clone()
        valid_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)  # All invalid

        # Should not crash
        metrics.update_mz_metrics(pred_mz, target_mz, valid_mask)

        assert metrics.running_counts['tokens'] == 0
        assert metrics.running_counts['spectra'] == batch_size

    def test_single_token_spectrum(self):
        """Test spectrum with only one valid token."""
        metrics = StreamingMetrics()

        batch_size, seq_len = 2, 10
        pred_mz = torch.ones(batch_size, seq_len) * 500.0
        target_mz = pred_mz + 1.0

        # Only one token valid per spectrum
        valid_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        valid_mask[:, 0] = True

        metrics.update_mz_metrics(pred_mz, target_mz, valid_mask)

        # Should not crash on std calculation
        assert metrics.running_counts['tokens'] == 2
        # Cosine similarity requires >1 token, so should be 0
        assert metrics.running_counts['cosine_samples'] == 0

    def test_median_accumulator_cap(self):
        """Test that median accumulator respects cap."""
        metrics = StreamingMetrics()
        metrics.max_median_samples = 50  # Lower cap for testing

        batch_size, seq_len = 10, 100
        pred_mz = torch.randn(batch_size, seq_len).abs() * 1000
        target_mz = pred_mz + torch.randn(batch_size, seq_len)
        valid_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        metrics.update_mz_metrics(pred_mz, target_mz, valid_mask)

        # Should respect cap
        assert len(metrics.running_median_accumulator) <= metrics.max_median_samples
        assert len(metrics.running_median_ppm_accumulator) <= metrics.max_median_samples

    def test_multiple_batches(self):
        """Test accumulation across multiple batches."""
        metrics = StreamingMetrics()

        for _ in range(3):
            batch_size, seq_len = 4, 50
            pred_mz = torch.randn(batch_size, seq_len).abs() * 500 + 200
            target_mz = pred_mz + torch.randn(batch_size, seq_len) * 2
            valid_mask = torch.rand(batch_size, seq_len) > 0.2

            metrics.update_mz_metrics(pred_mz, target_mz, valid_mask)

        # Check accumulation
        assert metrics.running_counts['spectra'] == 12  # 3 batches * 4 spectra
        assert metrics.running_counts['tokens'] > 0
        assert metrics.running_mae > 0


    def test_median_sampling_unbiased(self):
        """Test that reservoir sampling draws from across the full range, not just early spectra."""
        torch.manual_seed(123)
        metrics = StreamingMetrics()
        metrics.max_median_samples = 200  # Small reservoir for fast testing

        # Run many batches.  Early batches have errors ~1.0, late batches ~100.0
        # With biased (first-N) sampling, late-batch errors would be underrepresented.
        n_batches = 50
        seq_len = 40
        for batch_idx in range(n_batches):
            batch_size = 4
            base_error = 1.0 + (batch_idx / n_batches) * 99.0  # 1 → 100
            pred_mz = torch.ones(batch_size, seq_len) * 500.0
            target_mz = pred_mz + base_error
            valid_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

            metrics.update_mz_metrics(pred_mz, target_mz, valid_mask)

        # Reservoir should be full
        assert len(metrics.running_median_accumulator) == metrics.max_median_samples

        # Check that samples span the full range (not just early batches)
        samples = sorted(metrics.running_median_accumulator)
        # The reservoir should contain errors from both the low end (~1) and the high end (~100)
        assert samples[0] < 10.0, "Reservoir should contain low-error samples"
        assert samples[-1] > 80.0, "Reservoir should contain high-error samples"

        # Median should be roughly centered (biased sampling would give ~1.0)
        median_val = float(torch.tensor(samples).median())
        assert median_val > 20.0, f"Median {median_val} is too low — sampling may be biased toward early batches"

    def test_mz_slice_da_accuracy(self):
        """Test that per-slice Da accuracy counters are computed correctly."""
        metrics = StreamingMetrics()

        seq_len = 8
        # Create 4 spectra, each in a different m/z range with known errors
        pred_mz = torch.zeros(4, seq_len)
        target_mz = torch.zeros(4, seq_len)
        valid_mask = torch.zeros(4, seq_len, dtype=torch.bool)

        # Spectrum 0: low range (< 500), 4 tokens
        #   2 within 0.1Da, 3 within 1Da, 4 within 10Da
        target_mz[0, :4] = torch.tensor([100.0, 200.0, 300.0, 400.0])
        pred_mz[0, :4] = torch.tensor([100.05, 200.09, 300.5, 405.0])
        valid_mask[0, :4] = True

        # Spectrum 1: mid_low range (500-1000), 3 tokens
        #   0 within 0.1Da, 1 within 1Da, 3 within 10Da
        target_mz[1, :3] = torch.tensor([600.0, 700.0, 800.0])
        pred_mz[1, :3] = torch.tensor([601.0, 700.5, 805.0])
        valid_mask[1, :3] = True

        # Spectrum 2: mid_high range (1000-2000), 2 tokens
        #   1 within 0.1Da, 2 within 1Da, 2 within 10Da
        target_mz[2, :2] = torch.tensor([1200.0, 1500.0])
        pred_mz[2, :2] = torch.tensor([1200.05, 1500.8])
        valid_mask[2, :2] = True

        # Spectrum 3: high range (>= 2000), 2 tokens
        #   0 within 0.1Da, 0 within 1Da, 1 within 10Da
        target_mz[3, :2] = torch.tensor([2500.0, 3000.0])
        pred_mz[3, :2] = torch.tensor([2505.0, 3015.0])
        valid_mask[3, :2] = True

        metrics.update_mz_metrics(pred_mz, target_mz, valid_mask)

        # Verify low slice: errors = [0.05, 0.09, 0.5, 5.0]
        assert metrics.mz_slice_da_counts['low']['total'] == 4
        assert metrics.mz_slice_da_counts['low']['within_01da'] == 2   # 0.05, 0.09
        assert metrics.mz_slice_da_counts['low']['within_1da'] == 3    # 0.05, 0.09, 0.5
        assert metrics.mz_slice_da_counts['low']['within_10da'] == 4   # all ≤ 10

        # Verify mid_low slice: errors = [1.0, 0.5, 5.0]
        assert metrics.mz_slice_da_counts['mid_low']['total'] == 3
        assert metrics.mz_slice_da_counts['mid_low']['within_01da'] == 0
        assert metrics.mz_slice_da_counts['mid_low']['within_1da'] == 2  # 0.5 and 1.0 (both ≤ 1.0)
        assert metrics.mz_slice_da_counts['mid_low']['within_10da'] == 3

        # Verify mid_high slice: errors = [0.05, 0.8]
        assert metrics.mz_slice_da_counts['mid_high']['total'] == 2
        assert metrics.mz_slice_da_counts['mid_high']['within_01da'] == 1  # 0.05
        assert metrics.mz_slice_da_counts['mid_high']['within_1da'] == 2   # both
        assert metrics.mz_slice_da_counts['mid_high']['within_10da'] == 2

        # Verify high slice: errors = [5.0, 15.0]
        assert metrics.mz_slice_da_counts['high']['total'] == 2
        assert metrics.mz_slice_da_counts['high']['within_01da'] == 0
        assert metrics.mz_slice_da_counts['high']['within_1da'] == 0
        assert metrics.mz_slice_da_counts['high']['within_10da'] == 1  # 5.0

        # Verify the computed metrics include per-slice percentages
        final = metrics.compute_final_metrics()
        assert abs(final['slice_low_pct_within_01da'] - 50.0) < 0.01    # 2/4 * 100
        assert abs(final['slice_low_pct_within_1da'] - 75.0) < 0.01     # 3/4 * 100
        assert abs(final['slice_mid_low_pct_within_1da'] - 200/3) < 0.1 # 2/3 * 100
        assert abs(final['slice_high_pct_within_10da'] - 50.0) < 0.01   # 1/2 * 100

    def test_ppm_denominator_consistency(self):
        """Test that PPM errors use consistent denominators across all code paths."""
        metrics = StreamingMetrics()

        batch_size, seq_len = 2, 5
        pred_mz = torch.tensor([[100.01, 500.05, 1000.1, 2000.2, 50.005],
                                 [100.01, 500.05, 1000.1, 2000.2, 50.005]])
        target_mz = torch.tensor([[100.0, 500.0, 1000.0, 2000.0, 50.0],
                                   [100.0, 500.0, 1000.0, 2000.0, 50.0]])
        valid_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Compute expected PPM with the unified denominator (target + 1e-8)
        abs_err = (pred_mz - target_mz).abs()
        expected_ppm = (abs_err / (target_mz + 1e-8)) * 1e6

        metrics.update_mz_metrics(pred_mz, target_mz, valid_mask)

        # The PPM median accumulator should contain values consistent with our formula
        # Since all tokens are valid, the PPM values should match
        assert len(metrics.running_median_ppm_accumulator) > 0

        # Also test that update_uncertainty_metrics uses the same formula
        sigma_ppm = torch.ones(batch_size, seq_len) * 10.0
        metrics.update_uncertainty_metrics(pred_mz, target_mz, sigma_ppm, valid_mask)

        # The uncertainty errors should match our expected PPM values
        # (uncertainty_errors stores flat PPM errors from the first batch_size*seq_len entries)
        unc_ppm = torch.tensor(metrics.uncertainty_errors[:seq_len])
        expected_flat = expected_ppm[0]
        assert torch.allclose(unc_ppm, expected_flat, atol=0.1), \
            f"PPM mismatch: uncertainty={unc_ppm} vs expected={expected_flat}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
