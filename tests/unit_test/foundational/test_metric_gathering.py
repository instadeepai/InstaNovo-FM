"""Test multi-GPU metric gathering for StreamingMetrics."""

import torch
import pytest
from unittest.mock import Mock

from instanovo_fm.trainer.metrics import StreamingMetrics


def _make_mock_accelerator(num_processes: int, device: str = "cpu"):
    """Create a mock accelerator for testing gather_across_ranks."""
    mock = Mock()
    mock.num_processes = num_processes
    mock.device = torch.device(device)

    def mock_gather(tensor):
        """Simulate gather by returning the tensor unchanged (single-rank)."""
        return tensor

    mock.gather_for_metrics = Mock(side_effect=mock_gather)
    return mock


def _populate_metrics(metrics: StreamingMetrics, n_tokens: int, mae: float, loss: float,
                      within_20ppm: int = 0, bin_correct: int = 0) -> None:
    """Populate a StreamingMetrics instance with known values."""
    # Simulate accumulation as if update_loss and update_mz_metrics were called
    metrics.running_counts['losses'] = 1
    metrics.running_counts['tokens'] = n_tokens
    metrics.running_counts['spectra'] = n_tokens // 10 or 1
    metrics.running_loss = loss
    metrics.running_mae = mae
    metrics.running_mae_squared = mae ** 2
    metrics.tokens_within_20ppm = within_20ppm
    metrics.tokens_within_10ppm = within_20ppm // 2
    metrics.tokens_within_1da = within_20ppm
    metrics.tokens_within_01da = within_20ppm // 3
    metrics.spectrum_mae_sum = mae * (n_tokens // 10 or 1)
    metrics.spectrum_std_sum = 0.5 * (n_tokens // 10 or 1)
    metrics.bin_acc_group_correct = bin_correct
    metrics.bin_acc_offset_correct = bin_correct
    metrics.bin_acc_bin_correct = bin_correct
    metrics.bin_acc_group_top5_correct = bin_correct
    metrics.bin_acc_offset_top5_correct = bin_correct
    metrics.bin_acc_offset_pm1_correct = bin_correct
    metrics.bin_acc_bin_pm1_correct = bin_correct
    metrics.bin_acc_total = n_tokens
    metrics.bin_only_running_mae = mae * 1.1
    metrics.bin_only_tokens_within_01da = within_20ppm // 4
    metrics.bin_only_tokens_within_1da = within_20ppm
    metrics.bin_only_tokens_within_10ppm = within_20ppm // 2
    metrics.bin_only_tokens_within_20ppm = within_20ppm
    metrics.bin_only_token_count = n_tokens


class TestGatherAcrossRanks:
    """Test gather_across_ranks for multi-GPU metric aggregation."""

    def test_noop_single_process(self):
        """gather_across_ranks is a no-op when num_processes == 1."""
        metrics = StreamingMetrics()
        _populate_metrics(metrics, n_tokens=100, mae=5.0, loss=2.0, within_20ppm=80)

        accel = _make_mock_accelerator(num_processes=1)
        metrics.gather_across_ranks(accel)

        # gather_for_metrics should NOT have been called
        accel.gather_for_metrics.assert_not_called()

        # Values should be unchanged
        assert metrics.running_counts['tokens'] == 100
        assert metrics.running_mae == 5.0
        assert metrics.running_loss == 2.0
        assert metrics.tokens_within_20ppm == 80

    def test_two_ranks_sum_counters(self):
        """SUM counters are correctly aggregated across 2 ranks."""
        # Simulate rank 0: 100 tokens, 80 within 20ppm
        rank0 = StreamingMetrics()
        _populate_metrics(rank0, n_tokens=100, mae=5.0, loss=2.0, within_20ppm=80, bin_correct=60)

        # Simulate rank 1: 200 tokens, 150 within 20ppm
        rank1 = StreamingMetrics()
        _populate_metrics(rank1, n_tokens=200, mae=3.0, loss=1.5, within_20ppm=150, bin_correct=120)

        # Build the mock gather: concatenates both packed tensors
        def mock_gather(tensor):
            """Simulate 2-rank gather by packing rank0 and rank1 tensors."""
            # Build rank1's packed tensor with the same layout
            r1_sums = torch.tensor([
                float(rank1.running_counts['losses']),
                float(rank1.running_counts['tokens']),
                float(rank1.running_counts['spectra']),
                float(rank1.tokens_within_1da),
                float(rank1.tokens_within_01da),
                float(rank1.tokens_within_10ppm),
                float(rank1.tokens_within_20ppm),
                rank1.spectrum_mae_sum,
                rank1.spectrum_std_sum,
                float(rank1.bin_acc_group_correct),
                float(rank1.bin_acc_offset_correct),
                float(rank1.bin_acc_bin_correct),
                float(rank1.bin_acc_group_top5_correct),
                float(rank1.bin_acc_offset_top5_correct),
                float(rank1.bin_acc_offset_pm1_correct),
                float(rank1.bin_acc_bin_pm1_correct),
                float(rank1.bin_acc_total),
                float(rank1.bin_only_tokens_within_01da),
                float(rank1.bin_only_tokens_within_1da),
                float(rank1.bin_only_tokens_within_10ppm),
                float(rank1.bin_only_tokens_within_20ppm),
                float(rank1.bin_only_token_count),
            ], dtype=torch.float64)
            r1_weighted = torch.tensor([
                rank1.running_loss * rank1.running_counts['losses'],
                rank1.running_mae * rank1.running_counts['tokens'],
                rank1.running_mae_squared * rank1.running_counts['tokens'],
                rank1.bin_only_running_mae * rank1.bin_only_token_count,
            ], dtype=torch.float64)
            r1_all = torch.cat([r1_sums, r1_weighted])
            # Concatenate rank0 (tensor arg) and rank1 into (2, N)
            return torch.stack([tensor, r1_all])

        accel = _make_mock_accelerator(num_processes=2)
        accel.gather_for_metrics = Mock(side_effect=mock_gather)

        # Call gather on rank0's metrics
        rank0.gather_across_ranks(accel)

        # Verify SUM counters
        assert rank0.running_counts['tokens'] == 300  # 100 + 200
        assert rank0.running_counts['losses'] == 2     # 1 + 1
        assert rank0.tokens_within_20ppm == 230         # 80 + 150
        assert rank0.bin_acc_total == 300               # 100 + 200
        assert rank0.bin_acc_bin_correct == 180          # 60 + 120
        assert rank0.bin_only_token_count == 300         # 100 + 200

    def test_two_ranks_weighted_averages(self):
        """Weighted averages are correctly recomputed after gathering."""
        rank0 = StreamingMetrics()
        _populate_metrics(rank0, n_tokens=100, mae=5.0, loss=2.0)

        rank1 = StreamingMetrics()
        _populate_metrics(rank1, n_tokens=200, mae=3.0, loss=1.5)

        # Expected weighted average: (5.0*100 + 3.0*200) / (100+200) = 1100/300
        expected_mae = (5.0 * 100 + 3.0 * 200) / 300
        expected_loss = (2.0 * 1 + 1.5 * 1) / 2  # Each rank has 1 loss count
        expected_mae_sq = (25.0 * 100 + 9.0 * 200) / 300

        def mock_gather(tensor):
            r1_sums = torch.tensor([
                float(rank1.running_counts['losses']),
                float(rank1.running_counts['tokens']),
                float(rank1.running_counts['spectra']),
                float(rank1.tokens_within_1da),
                float(rank1.tokens_within_01da),
                float(rank1.tokens_within_10ppm),
                float(rank1.tokens_within_20ppm),
                rank1.spectrum_mae_sum,
                rank1.spectrum_std_sum,
                float(rank1.bin_acc_group_correct),
                float(rank1.bin_acc_offset_correct),
                float(rank1.bin_acc_bin_correct),
                float(rank1.bin_acc_group_top5_correct),
                float(rank1.bin_acc_offset_top5_correct),
                float(rank1.bin_acc_offset_pm1_correct),
                float(rank1.bin_acc_bin_pm1_correct),
                float(rank1.bin_acc_total),
                float(rank1.bin_only_tokens_within_01da),
                float(rank1.bin_only_tokens_within_1da),
                float(rank1.bin_only_tokens_within_10ppm),
                float(rank1.bin_only_tokens_within_20ppm),
                float(rank1.bin_only_token_count),
            ], dtype=torch.float64)
            r1_weighted = torch.tensor([
                rank1.running_loss * rank1.running_counts['losses'],
                rank1.running_mae * rank1.running_counts['tokens'],
                rank1.running_mae_squared * rank1.running_counts['tokens'],
                rank1.bin_only_running_mae * rank1.bin_only_token_count,
            ], dtype=torch.float64)
            r1_all = torch.cat([r1_sums, r1_weighted])
            return torch.stack([tensor, r1_all])

        accel = _make_mock_accelerator(num_processes=2)
        accel.gather_for_metrics = Mock(side_effect=mock_gather)

        rank0.gather_across_ranks(accel)

        assert rank0.running_mae == pytest.approx(expected_mae, rel=1e-10)
        assert rank0.running_loss == pytest.approx(expected_loss, rel=1e-10)
        assert rank0.running_mae_squared == pytest.approx(expected_mae_sq, rel=1e-10)

    def test_list_accumulators_unchanged(self):
        """List-based accumulators (reservoir samples) are NOT modified by gather."""
        metrics = StreamingMetrics()
        _populate_metrics(metrics, n_tokens=100, mae=5.0, loss=2.0)

        # Populate list accumulators with sentinel values
        metrics.running_median_accumulator = [1.0, 2.0, 3.0]
        metrics.running_median_ppm_accumulator = [10.0, 20.0]
        metrics.group_entropies = [0.5, 0.6]
        metrics.conf_joint_list = [0.9, 0.8]
        metrics.aux_intensity_preds = [torch.tensor([1.0])]

        def mock_gather(tensor):
            # Simulate 2-rank gather (second rank has same values)
            return torch.stack([tensor, tensor])

        accel = _make_mock_accelerator(num_processes=2)
        accel.gather_for_metrics = Mock(side_effect=mock_gather)

        rank0_median = list(metrics.running_median_accumulator)
        rank0_ppm = list(metrics.running_median_ppm_accumulator)
        rank0_entropy = list(metrics.group_entropies)
        rank0_conf = list(metrics.conf_joint_list)

        metrics.gather_across_ranks(accel)

        # Lists should be unchanged (same object, same values)
        assert metrics.running_median_accumulator == rank0_median
        assert metrics.running_median_ppm_accumulator == rank0_ppm
        assert metrics.group_entropies == rank0_entropy
        assert metrics.conf_joint_list == rank0_conf
        assert len(metrics.aux_intensity_preds) == 1

    def test_gather_1d_concatenated_format(self):
        """Handle gather_for_metrics returning 1-D concatenated tensor."""
        rank0 = StreamingMetrics()
        _populate_metrics(rank0, n_tokens=100, mae=5.0, loss=2.0, within_20ppm=80)

        rank1 = StreamingMetrics()
        _populate_metrics(rank1, n_tokens=200, mae=3.0, loss=1.5, within_20ppm=150)

        def mock_gather_1d(tensor):
            """Simulate gather that returns flat 1-D concatenation."""
            r1_sums = torch.tensor([
                float(rank1.running_counts['losses']),
                float(rank1.running_counts['tokens']),
                float(rank1.running_counts['spectra']),
                float(rank1.tokens_within_1da),
                float(rank1.tokens_within_01da),
                float(rank1.tokens_within_10ppm),
                float(rank1.tokens_within_20ppm),
                rank1.spectrum_mae_sum,
                rank1.spectrum_std_sum,
                float(rank1.bin_acc_group_correct),
                float(rank1.bin_acc_offset_correct),
                float(rank1.bin_acc_bin_correct),
                float(rank1.bin_acc_group_top5_correct),
                float(rank1.bin_acc_offset_top5_correct),
                float(rank1.bin_acc_offset_pm1_correct),
                float(rank1.bin_acc_bin_pm1_correct),
                float(rank1.bin_acc_total),
                float(rank1.bin_only_tokens_within_01da),
                float(rank1.bin_only_tokens_within_1da),
                float(rank1.bin_only_tokens_within_10ppm),
                float(rank1.bin_only_tokens_within_20ppm),
                float(rank1.bin_only_token_count),
            ], dtype=torch.float64)
            r1_weighted = torch.tensor([
                rank1.running_loss * rank1.running_counts['losses'],
                rank1.running_mae * rank1.running_counts['tokens'],
                rank1.running_mae_squared * rank1.running_counts['tokens'],
                rank1.bin_only_running_mae * rank1.bin_only_token_count,
            ], dtype=torch.float64)
            r1_all = torch.cat([r1_sums, r1_weighted])
            # Return 1-D concatenation (some accelerate versions do this)
            return torch.cat([tensor, r1_all])

        accel = _make_mock_accelerator(num_processes=2)
        accel.gather_for_metrics = Mock(side_effect=mock_gather_1d)

        rank0.gather_across_ranks(accel)

        # Should still produce correct results
        assert rank0.running_counts['tokens'] == 300
        assert rank0.tokens_within_20ppm == 230
        expected_mae = (5.0 * 100 + 3.0 * 200) / 300
        assert rank0.running_mae == pytest.approx(expected_mae, rel=1e-10)

    def test_zero_tokens_division_safe(self):
        """No division by zero when both ranks have zero tokens."""
        metrics = StreamingMetrics()
        # Leave all counters at default zero

        def mock_gather(tensor):
            return torch.stack([tensor, tensor])

        accel = _make_mock_accelerator(num_processes=2)
        accel.gather_for_metrics = Mock(side_effect=mock_gather)

        # Should not raise
        metrics.gather_across_ranks(accel)

        assert metrics.running_counts['tokens'] == 0
        assert metrics.running_mae == 0.0
        assert metrics.running_loss == 0.0
