"""Expected Calibration Error (ECE) helper for classifier confidence metrics.

Used by StreamingMetrics to report ece_group and ece_offset during validation —
the "when confident, is it right?" signal for the classification mz_head.

Reference:
    "On Calibration of Modern Neural Networks" (Guo et al., ICML 2017)
"""

from typing import Tuple

import torch


def compute_ece(
    confidences: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15,
) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Expected Calibration Error (ECE) and reliability diagram data.

    ECE measures the expected difference between confidence and accuracy.
    Well-calibrated models have ECE close to 0.

    Args:
        confidences: Confidence scores of shape (N,)
        predictions: Predicted class indices of shape (N,)
        targets: Ground truth class indices of shape (N,)
        n_bins: Number of bins for calibration (default: 15)

    Returns:
        Tuple of (ece, bin_accuracies, bin_confidences, bin_counts) where:
            - ece: Expected Calibration Error (scalar)
            - bin_accuracies: Accuracy per bin, shape (n_bins,)
            - bin_confidences: Average confidence per bin, shape (n_bins,)
            - bin_counts: Number of samples per bin, shape (n_bins,)
    """
    confidences = confidences.cpu().float()
    predictions = predictions.cpu().long()
    targets = targets.cpu().long()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_accuracies = torch.zeros(n_bins)
    bin_confidences = torch.zeros(n_bins)
    bin_counts = torch.zeros(n_bins)

    accuracies = (predictions == targets).float()

    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        if in_bin.any():
            bin_counts[i] = in_bin.sum().item()
            bin_accuracies[i] = accuracies[in_bin].mean().item()
            bin_confidences[i] = confidences[in_bin].mean().item()

    bin_weights = bin_counts / bin_counts.sum()
    ece = (bin_weights * torch.abs(bin_accuracies - bin_confidences)).sum().item()

    return ece, bin_accuracies, bin_confidences, bin_counts
