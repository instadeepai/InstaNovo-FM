"""Intensity metrics ported from spectrum_lens (pure numpy)."""

from __future__ import annotations

import numpy as np


def compute_intensity_metrics(
    intensity_array: np.ndarray,
    matched_observed_indices: set[int] | list[int],
) -> dict[str, float]:
    """Compute TIC explained, SNR estimate, and dynamic range."""
    intensity_array = np.asarray(intensity_array, dtype=float).flatten()
    if intensity_array.size == 0:
        return {"tic_explained": 0.0, "snr_estimate": 0.0, "dynamic_range": 0.0}

    matched_indices = {int(i) for i in matched_observed_indices}
    matched_intensities = intensity_array[list(matched_indices)] if matched_indices else np.array([])

    total_intensity = float(intensity_array.sum())
    matched_sum = float(matched_intensities.sum()) if matched_intensities.size else 0.0
    tic_explained = matched_sum / total_intensity if total_intensity > 0 else 0.0

    unmatched_mask = np.ones(len(intensity_array), dtype=bool)
    if matched_indices:
        unmatched_mask[list(matched_indices)] = False
    unmatched_intensities = intensity_array[unmatched_mask]

    median_matched = float(np.median(matched_intensities)) if matched_intensities.size else 0.0
    median_unmatched = float(np.median(unmatched_intensities)) if unmatched_intensities.size else 1e-6
    snr_estimate = median_matched / median_unmatched if median_unmatched > 0 else 0.0

    min_int = float(intensity_array.min())
    dynamic_range = float(intensity_array.max()) / min_int if min_int > 0 else 0.0

    return {
        "tic_explained": float(tic_explained),
        "snr_estimate": float(snr_estimate),
        "dynamic_range": float(dynamic_range),
    }
