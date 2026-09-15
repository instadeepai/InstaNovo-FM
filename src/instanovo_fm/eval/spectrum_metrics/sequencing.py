"""Sequencing metrics ported from spectrum_lens."""

from __future__ import annotations

import re
from typing import Any

_ION_ANNOTATION_RE = re.compile(r"^([abcxyz])(\d+)", re.IGNORECASE)


def _parse_ion_annotation(annotation: str) -> tuple[str, int] | None:
    match = _ION_ANNOTATION_RE.match(str(annotation))
    if not match:
        return None
    return match.group(1).lower(), int(match.group(2))


def _longest_consecutive(nums: list[int]) -> int:
    if not nums:
        return 0
    nums = sorted(set(nums))
    longest = 1
    current = 1
    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1] + 1:
            current += 1
        else:
            current = 1
        longest = max(longest, current)
    return longest


def compute_sequencing_metrics(
    matched_details: list[dict[str, Any]],
    *,
    sequence_length: int = 0,
    n_terminal_ions: tuple[str, ...] = ("a", "b", "c"),
    c_terminal_ions: tuple[str, ...] = ("x", "y", "z"),
) -> dict[str, float | int]:
    """Compute consecutive-ion and residue-coverage metrics from match details."""
    ion_indices: dict[str, list[int]] = {ion_type: [] for ion_type in ["a", "b", "c", "x", "y", "z"]}
    for match in matched_details:
        anno = match.get("theoretical_annotation") or match.get("annotation") or ""
        parsed = _parse_ion_annotation(str(anno))
        if parsed:
            ion_type, pos = parsed
            if ion_type in ion_indices:
                ion_indices[ion_type].append(pos)

    consecutive_counts = {f"consecutive_{ion}_ions": _longest_consecutive(indices) for ion, indices in ion_indices.items()}
    max_consecutive = max(consecutive_counts.values()) if consecutive_counts else 0

    n_term_count = sum(len(ion_indices[t]) for t in n_terminal_ions)
    c_term_count = sum(len(ion_indices[t]) for t in c_terminal_ions)

    n_term_sets = {t: set(ion_indices[t]) for t in n_terminal_ions}
    c_term_sets = {t: set(ion_indices[t]) for t in c_terminal_ions}

    covered_positions = 0
    if sequence_length > 0:
        for i in range(1, sequence_length + 1):
            n_term_covered = any(i in n_term_sets[t] for t in n_terminal_ions)
            c_term_covered = any((sequence_length - i + 1) in c_term_sets[t] for t in c_terminal_ions)
            if n_term_covered or c_term_covered:
                covered_positions += 1
        residue_evidence_coverage = covered_positions / sequence_length
    else:
        residue_evidence_coverage = 0.0

    return {
        "consecutive_ion_series": int(max_consecutive),
        **{k: int(v) for k, v in consecutive_counts.items()},
        "n_term_vs_c_term_bias": float(n_term_count / (c_term_count + 1e-6)),
        "residue_evidence_coverage": float(residue_evidence_coverage),
    }
