"""Parallel evidence scoring workers for cross-set annotation transfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from instanovo_fm.eval.spectrum_metrics.complexity import compute_complexity_metrics
from instanovo_fm.eval.spectrum_metrics.intensity import compute_intensity_metrics
from instanovo_fm.eval.spectrum_metrics.mcp_scoring import MCP_AVAILABLE, score_observed_vs_theoretical
from instanovo_fm.eval.spectrum_metrics.observed_vs_observed import score_observed_vs_observed
from instanovo_fm.eval.spectrum_metrics.sequencing import compute_sequencing_metrics


@dataclass(frozen=True)
class SpectrumRecord:
    """Pickle-friendly spectrum payload for parallel evidence workers."""

    mz: tuple[float, ...]
    intensity: tuple[float, ...]
    precursor_mz: float | None
    precursor_charge: int | None
    peptide: str
    unmodified_peptide: str


@dataclass(frozen=True)
class QueryRankWorkItem:
    """Work item for scoring one query-library candidate pair."""

    query_index: int
    rank: int
    library_index: int
    query: SpectrumRecord
    library: SpectrumRecord
    score_blocks: tuple[str, ...]
    tolerance_da: float
    ion_types: str
    max_ion_charge: int


@dataclass(frozen=True)
class LibrarySelfWorkItem:
    """Work item for library observed-vs-theoretical self-consistency (block C)."""

    library_index: int
    library: SpectrumRecord
    tolerance_da: float
    ion_types: str
    max_ion_charge: int


def _prefix_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if key.startswith("_"):
            continue
        out[f"{prefix}__{key}"] = value
    return out


def _sequence_length(peptide: str) -> int:
    clean = "".join(ch for ch in peptide if ch.isalpha())
    return len(clean)


def _extend_with_lens_metrics(
    metrics: dict[str, Any],
    *,
    prefix: str,
    matched_details: list[dict[str, Any]],
    matched_observed_indices: list[int],
    mz: np.ndarray,
    intensity: np.ndarray,
    peptide: str,
) -> dict[str, Any]:
    out = dict(metrics)
    out.update(_prefix_metrics(compute_complexity_metrics(mz), prefix))
    out.update(
        _prefix_metrics(
            compute_intensity_metrics(intensity, matched_observed_indices),
            prefix,
        )
    )
    out.update(
        _prefix_metrics(
            compute_sequencing_metrics(
                matched_details,
                sequence_length=_sequence_length(peptide),
            ),
            prefix,
        )
    )
    return out


def score_library_self(item: LibrarySelfWorkItem) -> tuple[int, dict[str, Any]]:
    """Score library observed vs library theoretical (block C)."""
    if not MCP_AVAILABLE:
        return item.library_index, {}
    lib = item.library
    if not (lib.peptide or lib.unmodified_peptide):
        return item.library_index, {}
    try:
        mz = np.asarray(lib.mz, dtype=float)
        intensity = np.asarray(lib.intensity, dtype=float)
        mcp = score_observed_vs_theoretical(
            observed_mz=mz,
            observed_intensity=intensity,
            peptidoform=lib.peptide or lib.unmodified_peptide,
            precursor_mz=lib.precursor_mz,
            precursor_charge=lib.precursor_charge,
            tolerance_da=item.tolerance_da,
            ion_types=item.ion_types,
            max_ion_charge=item.max_ion_charge,
        )
    except Exception:
        return item.library_index, {}
    matched_details = mcp.pop("_matched_details", [])
    matched_observed_indices = mcp.pop("_matched_observed_indices", [])
    metrics = _prefix_metrics(mcp, "lib_obs__lib_theo")
    metrics = _extend_with_lens_metrics(
        metrics,
        prefix="lib_obs__lib_theo",
        matched_details=matched_details,
        matched_observed_indices=matched_observed_indices,
        mz=mz,
        intensity=intensity,
        peptide=lib.unmodified_peptide or lib.peptide,
    )
    return item.library_index, metrics


def score_query_rank(item: QueryRankWorkItem, library_self_cache: dict[int, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Score blocks A/B (+ attach cached C) for one query-library pair."""
    out: dict[str, Any] = {
        "query_index": item.query_index,
        "rank": item.rank,
        "library_index": item.library_index,
    }
    query = item.query
    library = item.library
    q_mz = np.asarray(query.mz, dtype=float)
    q_int = np.asarray(query.intensity, dtype=float)
    l_mz = np.asarray(library.mz, dtype=float)
    l_int = np.asarray(library.intensity, dtype=float)

    if "A" in item.score_blocks:
        block_a = score_observed_vs_observed(q_mz, q_int, l_mz, l_int, tolerance_da=item.tolerance_da)
        out.update(_prefix_metrics(block_a, "q_obs__lib_obs"))

    if MCP_AVAILABLE and "B" in item.score_blocks:
        try:
            block_b = score_observed_vs_theoretical(
                observed_mz=q_mz,
                observed_intensity=q_int,
                peptidoform=library.peptide or library.unmodified_peptide,
                precursor_mz=query.precursor_mz,
                precursor_charge=query.precursor_charge,
                tolerance_da=item.tolerance_da,
                ion_types=item.ion_types,
                max_ion_charge=item.max_ion_charge,
            )
        except Exception:
            block_b = {}
        if block_b:
            matched_details = block_b.pop("_matched_details", [])
            matched_observed_indices = block_b.pop("_matched_observed_indices", [])
            out.update(_prefix_metrics(block_b, "q_obs__lib_theo"))
            out = _extend_with_lens_metrics(
                out,
                prefix="q_obs__lib_theo",
                matched_details=matched_details,
                matched_observed_indices=matched_observed_indices,
                mz=q_mz,
                intensity=q_int,
                peptide=library.unmodified_peptide or library.peptide,
            )

    if library_self_cache is not None and item.library_index in library_self_cache:
        out.update(library_self_cache[item.library_index])

    return out


def score_query_rank_worker(args: tuple[QueryRankWorkItem, dict[int, dict[str, Any]] | None]) -> dict[str, Any]:
    """Process-pool entry point for ``score_query_rank``."""
    item, cache = args
    return score_query_rank(item, library_self_cache=cache)
