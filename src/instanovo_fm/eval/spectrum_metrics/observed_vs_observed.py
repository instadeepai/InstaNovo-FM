"""Observed-vs-observed spectral similarity (block A)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return None
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        return None
    return float(np.dot(a, b) / (norm_a * norm_b))


def _spectral_angle_from_cosine(cos_val: float) -> float:
    clipped = max(-1.0, min(1.0, cos_val))
    return 1.0 - (2.0 * math.acos(clipped) / math.pi)


def _bin_spectra(
    mz_a: np.ndarray,
    int_a: np.ndarray,
    mz_b: np.ndarray,
    int_b: np.ndarray,
    *,
    bin_width: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    if mz_a.size == 0 or mz_b.size == 0:
        return np.array([]), np.array([])
    max_mz = max(float(mz_a.max()), float(mz_b.max())) + 1.0
    bins = np.arange(0.0, max_mz + bin_width, bin_width)
    hist_a = np.zeros(len(bins) - 1, dtype=float)
    hist_b = np.zeros(len(bins) - 1, dtype=float)
    for mz, intensity in zip(mz_a, int_a, strict=False):
        idx = int(np.searchsorted(bins, mz, side="right") - 1)
        if 0 <= idx < len(hist_a):
            hist_a[idx] += intensity
    for mz, intensity in zip(mz_b, int_b, strict=False):
        idx = int(np.searchsorted(bins, mz, side="right") - 1)
        if 0 <= idx < len(hist_b):
            hist_b[idx] += intensity
    return hist_a, hist_b


def _match_peaks(
    mz_a: np.ndarray,
    int_a: np.ndarray,
    mz_b: np.ndarray,
    int_b: np.ndarray,
    *,
    tolerance_da: float,
) -> tuple[int, float]:
    if mz_a.size == 0 or mz_b.size == 0:
        return 0, 0.0
    matched = 0
    matched_intensity = 0.0
    used_b: set[int] = set()
    order = np.argsort(int_a)[::-1]
    for idx_a in order:
        deltas = np.abs(mz_b - mz_a[idx_a])
        candidate_indices = np.where(deltas <= tolerance_da)[0]
        best_idx = None
        best_delta = float("inf")
        for idx_b in candidate_indices:
            if idx_b in used_b:
                continue
            delta = float(deltas[idx_b])
            if delta < best_delta:
                best_delta = delta
                best_idx = int(idx_b)
        if best_idx is not None:
            used_b.add(best_idx)
            matched += 1
            matched_intensity += float(int_a[idx_a])
    return matched, matched_intensity


def score_observed_vs_observed(
    mz_a: np.ndarray,
    int_a: np.ndarray,
    mz_b: np.ndarray,
    int_b: np.ndarray,
    *,
    tolerance_da: float = 0.05,
    bin_width: float = 0.1,
) -> dict[str, Any]:
    """Score two observed spectra against each other."""
    mz_a = np.asarray(mz_a, dtype=float).flatten()
    int_a = np.asarray(int_a, dtype=float).flatten()
    mz_b = np.asarray(mz_b, dtype=float).flatten()
    int_b = np.asarray(int_b, dtype=float).flatten()

    hist_a, hist_b = _bin_spectra(mz_a, int_a, mz_b, int_b, bin_width=bin_width)
    cosine = _cosine_similarity(hist_a, hist_b)
    pearson = None
    if hist_a.size >= 2 and hist_a.std() > 0 and hist_b.std() > 0:
        pearson = float(np.corrcoef(hist_a, hist_b)[0, 1])

    matched_count, matched_intensity = _match_peaks(mz_a, int_a, mz_b, int_b, tolerance_da=tolerance_da)
    total_intensity = float(int_a.sum())
    explained = matched_intensity / total_intensity if total_intensity > 0 else None

    return {
        "spectral_angle": _spectral_angle_from_cosine(cosine) if cosine is not None else None,
        "cosine_similarity": cosine,
        "pearson_correlation": pearson,
        "matched_peak_count": int(matched_count),
        "explained_intensity_fraction": explained,
    }
