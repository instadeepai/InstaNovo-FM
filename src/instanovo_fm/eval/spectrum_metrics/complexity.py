"""Spectrum complexity metrics ported from spectrum_lens."""

from __future__ import annotations

import numpy as np

MZ_BIN_EDGES = [0, 50, 2500, 3000, 3500, 4000, 4500, 5000]


def calculate_peak_distribution_uniformity(mz_array: np.ndarray) -> float:
    """Return normalised Shannon entropy of peak counts across m/z bins."""
    mz_array = np.asarray(mz_array, dtype=float).flatten()
    if mz_array.size < 2:
        return 0.0
    counts, _ = np.histogram(mz_array, bins=MZ_BIN_EDGES)
    total = counts.sum()
    if total == 0:
        return 0.0
    proportions = counts / total
    nonzero = proportions[proportions > 0]
    entropy = -float(np.sum(nonzero * np.log(nonzero)))
    max_entropy = np.log(len(nonzero)) if len(nonzero) > 1 else 1.0
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def compute_complexity_metrics(mz_array: np.ndarray) -> dict[str, float]:
    """Compute peak distribution uniformity and peak density for one spectrum."""
    mz_array = np.asarray(mz_array, dtype=float).flatten()
    if mz_array.size == 0:
        return {"peak_distribution_uniformity": 0.0, "peak_density": 0.0}
    mz_span = float(mz_array.max() - mz_array.min())
    peak_density = len(mz_array) / mz_span if mz_span > 0 else 0.0
    return {
        "peak_distribution_uniformity": calculate_peak_distribution_uniformity(mz_array),
        "peak_density": float(peak_density),
    }
