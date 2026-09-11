"""Adapter around proteomics-mcp scoring (optional dependency)."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from typing import Any, cast

import numpy as np

try:
    from proteomics_mcp.core.annotation import generate_terminal_fragments
    from proteomics_mcp.core.scoring import score_candidate_spectrum
    from proteomics_mcp.models.schemas import CandidatePSM, ObservedSpectrum, PredictedFragment, PredictedSpectrum

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


def _to_list(values: Any) -> list[float]:
    if values is None:
        return []
    arr = np.asarray(values, dtype=float).flatten()
    return [float(v) for v in arr.tolist()]


def observed_spectrum_from_arrays(
    mz: Any,
    intensity: Any,
    *,
    precursor_mz: float | None = None,
    precursor_charge: int | None = None,
    spectrum_id: str | None = None,
) -> Any:
    """Build an MCP ObservedSpectrum from numpy peak arrays."""
    if not MCP_AVAILABLE:
        raise RuntimeError("proteomics-mcp is not installed. Install with: uv sync --extra proteomics-metrics")
    return ObservedSpectrum(
        mz=_to_list(mz),
        intensity=_to_list(intensity),
        spectrum_id=spectrum_id,
        precursor_mz=precursor_mz,
        precursor_charge=precursor_charge,
        source="instanovo_cross_set",
    )


@lru_cache(maxsize=4096)
def _build_predicted_spectrum_cached(
    peptidoform: str,
    precursor_charge: int,
    ion_types: str,
    max_ion_charge: int,
) -> Any:
    """Memoized fragment-ion generation, keyed on (peptide, charge, ion_types, max_ion_charge).

    Theoretical fragments only depend on these four inputs, not on which query is
    being scored against them. Without this cache, a "hub" library peptide shared
    by thousands of queries (see cross-set annotation transfer margin analysis)
    would recompute the identical fragment list once per query.
    """
    raw_fragments = generate_terminal_fragments(
        peptidoform,
        ion_types=ion_types,
        max_ion_charge=max_ion_charge,
        max_isotope=0,
        neutral_losses=False,
    )
    fragments = [
        PredictedFragment(
            mz=float(item["mz"]),
            intensity=1.0,
            annotation=str(item["annotation"]),
            ion_type=item.get("ion_type"),
            ordinal=item.get("ordinal"),
            charge=int(item.get("charge") or 1),
            neutral_loss=item.get("neutral_loss"),
        )
        for item in raw_fragments
    ]
    return PredictedSpectrum(
        peptidoform=peptidoform,
        precursor_charge=precursor_charge,
        fragments=fragments,
        predictor_name="terminal_fragments",
        predictor_version="proteomics_mcp",
        model="terminal",
    )


def build_predicted_spectrum(
    peptidoform: str,
    precursor_charge: int,
    *,
    ion_types: str = "by",
    max_ion_charge: int = 2,
) -> Any:
    """Build an MCP PredictedSpectrum from a peptidoform sequence (memoized)."""
    if not MCP_AVAILABLE:
        raise RuntimeError("proteomics-mcp is not installed. Install with: uv sync --extra proteomics-metrics")
    if not peptidoform:
        return PredictedSpectrum(peptidoform="", precursor_charge=precursor_charge or 2, fragments=[])
    return _build_predicted_spectrum_cached(peptidoform, precursor_charge, ion_types, max_ion_charge)


def _flatten_mcp_result(result: Any) -> dict[str, Any]:
    if is_dataclass(result):
        payload = asdict(cast(Any, result))
    elif isinstance(result, dict):
        payload = result
    else:
        payload = dict(result)

    annotation_summary = payload.get("annotation_summary") or {}
    flat: dict[str, Any] = {
        "spectral_angle": payload.get("spectral_angle"),
        "cosine_similarity": payload.get("cosine_similarity"),
        "pearson_correlation": payload.get("pearson_correlation"),
        "xcorr": payload.get("xcorr"),
        "log2_xcorr": payload.get("log2_xcorr"),
        "hyperscore": payload.get("hyperscore"),
        "log2_hyperscore": payload.get("log2_hyperscore"),
        "matched_ion_count": payload.get("matched_ion_count"),
        "total_predicted_ions": payload.get("total_predicted_ions"),
        "matched_ion_fraction": payload.get("matched_ion_fraction"),
        "explained_intensity_fraction": payload.get("explained_intensity_fraction"),
        "precursor_signal_fraction": payload.get("precursor_signal_fraction"),
        "fragment_mass_error_mean_ppm": payload.get("fragment_mass_error_mean_ppm"),
        "fragment_mass_error_abs_mean_ppm": payload.get("fragment_mass_error_abs_mean_ppm"),
        "precursor_mass_error_ppm": payload.get("precursor_mass_error_ppm"),
        "precursor_mass_error_da": payload.get("precursor_mass_error_da"),
        "annotated_peak_count": annotation_summary.get("annotated_peak_count"),
        "annotated_intensity_fraction": annotation_summary.get("annotated_intensity_fraction"),
        "ion_type_counts_json": json.dumps(annotation_summary.get("ion_type_counts") or {}, sort_keys=True),
    }

    matched_peaks = payload.get("matched_peaks") or []
    matched_details = []
    matched_observed_indices: list[int] = []
    for peak in matched_peaks:
        if not isinstance(peak, dict):
            continue
        matched_details.append(
            {
                "theoretical_annotation": peak.get("annotation"),
                "observed_mz": peak.get("observed_mz"),
                "predicted_mz": peak.get("predicted_mz"),
            }
        )
        if peak.get("observed_index") is not None:
            matched_observed_indices.append(int(peak["observed_index"]))
    flat["_matched_details"] = matched_details
    flat["_matched_observed_indices"] = matched_observed_indices
    return flat


def score_observed_vs_theoretical(
    *,
    observed_mz: Any,
    observed_intensity: Any,
    peptidoform: str,
    precursor_mz: float | None,
    precursor_charge: int | None,
    tolerance_da: float = 0.05,
    ion_types: str = "by",
    max_ion_charge: int = 2,
) -> dict[str, Any]:
    """Score observed peaks against a predicted fragment spectrum via MCP."""
    if not MCP_AVAILABLE:
        raise RuntimeError("proteomics-mcp is not installed. Install with: uv sync --extra proteomics-metrics")

    observed = observed_spectrum_from_arrays(
        observed_mz,
        observed_intensity,
        precursor_mz=precursor_mz,
        precursor_charge=precursor_charge,
    )
    charge = int(precursor_charge or 2)
    predicted = build_predicted_spectrum(peptidoform, charge, ion_types=ion_types, max_ion_charge=max_ion_charge)
    candidate = CandidatePSM(
        candidate_id="cross_set",
        source_engine="instanovo_cross_set",
        peptidoform=peptidoform or "",
        precursor_mz=precursor_mz,
        precursor_charge=charge,
    )
    result = score_candidate_spectrum(
        candidate=candidate,
        observed=observed,
        predicted=predicted,
        tolerance=tolerance_da,
        tolerance_unit="Da",
        annotation_ion_types=ion_types,
        annotation_max_ion_charge=max_ion_charge,
    )
    return _flatten_mcp_result(result)
