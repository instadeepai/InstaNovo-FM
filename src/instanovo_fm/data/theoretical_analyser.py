#!/usr/bin/env python
"""Theoretical Spectrum Analyzer for Foundation Model Training Data.

Performs post-hoc characterization of MS/MS data quality through theoretical
spectrum matching.  Provides insights for:

1. **Training signal quality** — what fraction of peaks are informative (annotated
   fragment ions) vs noise (unannotated)?
2. **Mass error characterization** — are mass errors compatible with the chosen
   binning resolution?
3. **Ion coverage** — which theoretical ions are consistently observed or missed?

This is a diagnostic/analysis tool — it is **never** called during training.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from omegaconf import DictConfig

from instanovo.__init__ import console
from instanovo.common import DataProcessor
from instanovo_fm.data import FoundationalDataProcessor
from instanovo_fm.utils.theoretical_spectra import (
    DEFAULT_CID_DA_TOL,
    DEFAULT_CUSTOM_IONS,
    _da_tol_for_fragmentation,
    compute_theoretical_precursor_mz,
    detect_custom_ions,
    generate_theoretical_spectrum,
    generate_theoretical_spectrum_rustyms,
    match_theoretical_to_experimental,
    match_with_conditional_features,
)
from instanovo_fm.utils.modifications import extract_modification_types
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger

# ─── Shared colour palettes ────────────────────────────────────────────────
ION_TYPE_COLORS: Dict[str, str] = {
    "b": "#1f77b4",
    "y": "#d62728",
    "a": "#2ca02c",
    "c": "#9467bd",
    "x": "#8c564b",
    "z": "#e377c2",
}

SIGNAL_CATEGORY_COLORS: Dict[str, str] = {
    "fragment_base": "#2ca02c",
    "fragment_loss": "#98df8a",
    "fragment_isotope": "#aec7e8",
    "precursor": "#ffbb78",
    "other_annotated": "#c7c7c7",
    "unannotated": "#d62728",
}

FRAG_TYPE_COLORS: Dict[str, str] = {
    "HCD": "#1f77b4",
    "CID": "#ff7f0e",
    "HCID": "#2ca02c",
    "ETD": "#d62728",
    "ECD": "#9467bd",
    "UVPD": "#8c564b",
    "unknown": "#7f7f7f",
}


# ─── Multiprocessing workers for per-peptide analyses ──────────────────────
# These module-level functions parallelise the per-peptide loops in
# `calculate_complementary_pair_analysis` and `calculate_mass_gap_analysis`.
# Each worker processes a chunk of peptide groups independently and returns
# the per-gap/per-pair records plus per-spectrum summaries. The main process
# concatenates results and runs the existing aggregation code.

# Worker-local state (populated by the initialiser; read by per-peptide fns).
_WORKER_STATE: Dict[str, Any] = {}


def _tokenize_sequence_standalone(seq: str) -> List[str]:
    """Module-level copy of TheoreticalAnalyser._tokenize_sequence for workers.

    Kept in sync with the class's staticmethod. Workers call this instead of
    importing the class, to avoid circular-import risks and keep the worker
    surface small.
    """
    residues: List[str] = []
    i = 0
    while i < len(seq):
        if seq[i].isupper():
            residue = seq[i]
            if i + 1 < len(seq) and seq[i + 1] in ("(", "["):
                bracket = ")" if seq[i + 1] == "(" else "]"
                end = seq.find(bracket, i + 2)
                if end != -1:
                    residue = seq[i:end + 1]
                    i = end + 1
                    residues.append(residue)
                    continue
            residues.append(residue)
        i += 1
    return residues


def _classify_mz_range_standalone(
    mz: float,
    mz_range_boundaries: Dict[str, Tuple[float, float]],
    mz_range_order: List[str],
) -> str:
    """Module-level copy of _classify_mz_range for workers."""
    for range_name in mz_range_order:
        low, high = mz_range_boundaries[range_name]
        if low <= mz < high:
            return range_name
    return mz_range_order[-1]


def _init_theo_worker(
    aa_codes: List[str],
    aa_masses: np.ndarray,
    aa_masses_sorted: np.ndarray,
    proton_mass: float,
    mz_range_boundaries: Dict[str, Tuple[float, float]],
    mz_range_order: List[str],
    mass_gap_ppm_tol: float,
) -> None:
    """Pool initialiser — stores shared constants in worker-local state."""
    global _WORKER_STATE
    _WORKER_STATE = {
        "aa_codes": aa_codes,
        "aa_masses": aa_masses,
        "aa_masses_sorted": aa_masses_sorted,
        "proton_mass": proton_mass,
        "mz_range_boundaries": mz_range_boundaries,
        "mz_range_order": mz_range_order,
        "mass_gap_ppm_tol": mass_gap_ppm_tol,
    }


def _process_batch_complementary_pair(
    batch: List[Tuple[str, Dict[str, np.ndarray], Dict[str, Any]]],
) -> List[Optional[Dict[str, Any]]]:
    """Batch wrapper: process many peptides per pickle to amortise Pool overhead."""
    return [_process_peptide_complementary_pair(a) for a in batch]


def _process_batch_mass_gap(
    batch: List[Tuple[str, Dict[str, np.ndarray]]],
) -> List[Optional[Dict[str, Any]]]:
    """Batch wrapper: process many peptides per pickle to amortise Pool overhead."""
    return [_process_peptide_mass_gap(a) for a in batch]


def _process_peptide_complementary_pair(
    args: Tuple[str, Dict[str, np.ndarray], Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Worker: process one peptide group for complementary b/y pair analysis.

    Input args are pre-extracted numpy arrays (not a DataFrame) to minimise
    pickling overhead when this function runs in a worker process.

    Returns a dict with:
      - `pair_records`: per-pair records (deviations, positions, frag_types)
      - `per_spectrum`: per-spectrum summary (pair_fraction, n_pairs, ...)
    or None if the peptide lacks required metadata.
    """
    peptide, arrays, meta = args
    proton_mass = _WORKER_STATE["proton_mass"]

    if meta is None:
        return None

    prec_charge = meta["precursor_charge"]
    prec_mz = meta["precursor_mz"]
    if np.isnan(prec_mz):
        return None

    seq_residues = _tokenize_sequence_standalone(peptide)
    seq_len = len(seq_residues)
    if seq_len < 2:
        return None

    positions = arrays["position"]
    charges = arrays["charge"]
    exp_mzs = arrays["exp_mz"]
    ion_types = arrays["ion_type"]
    frag_type = arrays["frag_type"]

    # Neutral precursor mass from observed (or theoretical) m/z
    prec_neutral = (prec_mz - proton_mass) * prec_charge

    # Build position -> {charge: exp_mz} maps for b and y
    b_map: Dict[int, Dict[int, float]] = {}
    y_map: Dict[int, Dict[int, float]] = {}
    for i in range(len(positions)):
        pos = int(positions[i])
        charge = int(charges[i])
        exp_mz = float(exp_mzs[i])
        if pos < 1:
            continue
        target = b_map if ion_types[i] == "b" else y_map
        target.setdefault(pos, {})[charge] = exp_mz

    # Check each cleavage site
    n_possible = seq_len - 1
    n_pairs = 0
    spectrum_devs: List[float] = []
    pair_records: List[Dict[str, Any]] = []

    for site in range(1, seq_len):
        b_pos = site
        y_pos = seq_len - site
        if b_pos not in b_map or y_pos not in y_map:
            continue

        b_charges = b_map[b_pos]
        y_charges = y_map[y_pos]
        best_dev = None
        for zb, mz_b in b_charges.items():
            for zy, mz_y in y_charges.items():
                m_b = (mz_b - proton_mass) * zb
                m_y = (mz_y - proton_mass) * zy
                observed_sum = m_b + m_y
                dev_da = observed_sum - prec_neutral
                if best_dev is None or abs(dev_da) < abs(best_dev):
                    best_dev = dev_da

        if best_dev is not None:
            n_pairs += 1
            spectrum_devs.append(best_dev)
            dev_ppm = (
                best_dev / prec_neutral * 1e6 if prec_neutral > 0 else 0.0
            )
            rel_pos = site / seq_len
            pair_records.append({
                "dev_da": best_dev,
                "dev_ppm": dev_ppm,
                "rel_pos": rel_pos,
                "frag_type": frag_type,
            })

    return {
        "pair_records": pair_records,
        "per_spectrum": {
            "pair_fraction": n_pairs / max(n_possible, 1),
            "n_pairs": n_pairs,
            "mean_dev_da": float(np.mean(spectrum_devs)) if spectrum_devs else 0.0,
            "frag_type": frag_type,
            "precursor_charge": prec_charge,
            "seq_len": seq_len,
        },
    }


def _process_peptide_mass_gap(
    args: Tuple[str, Dict[str, np.ndarray]],
) -> Optional[Dict[str, Any]]:
    """Worker: process one peptide group for mass gap validation.

    Input args are pre-extracted numpy arrays (not a DataFrame) to minimise
    pickling overhead when this function runs in a worker process.

    Returns a dict with:
      - `gap_records`: per-gap records (gap_da, errors, is_valid, ...)
      - `per_spectrum`: per-spectrum summary (match_rate, correct_rate, ...)
    or None if the peptide produces no gaps.
    """
    peptide, arrays = args
    aa_codes = _WORKER_STATE["aa_codes"]
    aa_masses = _WORKER_STATE["aa_masses"]
    aa_masses_sorted = _WORKER_STATE["aa_masses_sorted"]
    mz_range_boundaries = _WORKER_STATE["mz_range_boundaries"]
    mz_range_order = _WORKER_STATE["mz_range_order"]
    ppm_tol = _WORKER_STATE["mass_gap_ppm_tol"]

    frag_type = arrays["frag_type"]
    all_positions = arrays["position"]
    all_charges = arrays["charge"]
    all_mzs = arrays["exp_mz"]
    all_ion_types = arrays["ion_type"]

    seq_residues = _tokenize_sequence_standalone(peptide) if peptide else []
    seq_len = len(seq_residues) if seq_residues else 0

    n_gaps = 0
    n_valid = 0
    n_correct = 0
    valid_left: Set[int] = set()
    valid_right: Set[int] = set()
    gap_records: List[Dict[str, Any]] = []

    # Group by (ion_type, charge) using numpy operations
    # Build unique (ion_type, charge) pairs
    ic_pairs = set(zip(all_ion_types.tolist(), all_charges.tolist()))
    for ion_type, charge in ic_pairs:
        charge = int(charge)
        if charge < 1:
            continue

        # Select rows matching this (ion_type, charge)
        mask = (all_ion_types == ion_type) & (all_charges == charge)
        sub_positions = all_positions[mask]
        sub_mzs = all_mzs[mask]

        # Sort by position
        sort_idx = np.argsort(sub_positions)
        positions = sub_positions[sort_idx]
        mz_values = sub_mzs[sort_idx]

        for k in range(len(positions) - 1):
            pos_i = int(positions[k])
            pos_j = int(positions[k + 1])
            if pos_j != pos_i + 1:
                continue  # Not consecutive

            gap_mz = mz_values[k + 1] - mz_values[k]
            gap_da = abs(gap_mz * charge)

            diffs = np.abs(aa_masses_sorted - gap_da)
            best_idx = int(np.argmin(diffs))
            best_mass = float(aa_masses_sorted[best_idx])
            match_error_da = gap_da - best_mass
            match_error_ppm = (
                match_error_da / best_mass * 1e6 if best_mass > 0 else 0.0
            )

            is_valid = abs(match_error_ppm) <= ppm_tol

            n_ambiguous = int(
                np.sum(
                    np.abs(aa_masses - gap_da)
                    / np.maximum(aa_masses, 1e-9)
                    * 1e6
                    <= ppm_tol
                )
            )

            is_correct = False
            true_aa = ""
            if seq_residues and seq_len > 0:
                if ion_type == "b" and pos_i < seq_len:
                    true_aa = seq_residues[pos_i]
                elif ion_type == "y" and (seq_len - pos_j) >= 0:
                    true_aa = seq_residues[seq_len - pos_j]

                if true_aa and len(true_aa) == 1:
                    true_mass_idx = (
                        aa_codes.index(true_aa) if true_aa in aa_codes else -1
                    )
                    if true_mass_idx >= 0:
                        true_mass = aa_masses[true_mass_idx]
                        true_err_ppm = abs(
                            (gap_da - true_mass) / true_mass * 1e6
                        )
                        is_correct = true_err_ppm <= ppm_tol

            mid_mz = (mz_values[k] + mz_values[k + 1]) / 2
            mz_range = _classify_mz_range_standalone(
                mid_mz, mz_range_boundaries, mz_range_order
            )

            n_gaps += 1
            if is_valid:
                n_valid += 1
                valid_right.add(pos_i)
                valid_left.add(pos_j)
            if is_correct:
                n_correct += 1

            gap_records.append({
                "gap_da": gap_da,
                "match_error_da": match_error_da,
                "match_error_ppm": match_error_ppm,
                "is_valid": is_valid,
                "is_correct": is_correct,
                "n_ambiguous": n_ambiguous,
                "ion_type": ion_type,
                "charge": charge,
                "frag_type": frag_type,
                "mz_range": mz_range,
                "position_left": pos_i,
                "true_aa": true_aa,
            })

    if n_gaps == 0:
        return None

    two_sided_positions = valid_left & valid_right
    candidate_positions = valid_left | valid_right
    two_sided_rate = (
        len(two_sided_positions) / len(candidate_positions)
        if candidate_positions
        else 0.0
    )

    return {
        "gap_records": gap_records,
        "per_spectrum": {
            "match_rate": n_valid / n_gaps,
            "correct_rate": n_correct / n_gaps,
            "frag_type": frag_type,
            "two_sided_rate": two_sided_rate,
        },
    }


class TheoreticalAnalyser:
    """Analyzer for theoretical spectrum generation, matching, and signal composition.

    Produces three categories of insight:

    1. **Annotation coverage** — per-spectrum and aggregate statistics on how many
       experimental peaks can be explained by theoretical fragment ions.
    2. **Mass error analysis** — distribution of PPM errors between matched
       experimental and theoretical peaks, including systematic bias detection.
    3. **Signal composition** — breakdown of annotated peak types (b/y ions,
       neutral losses, isotopes, precursor, unannotated) by count and
       intensity weight.
    """

    def __init__(self, config: DictConfig, output_dir: Optional[Path] = None):
        self.config = config

        if output_dir is None:
            self.output_dir = Path("analysis_output") / "theoretical_analysis"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model configuration
        self.max_mz = config.model.get("max_mz", 2500.0)
        self.min_mz = config.model.get("min_mz", 50.0)

        # Analysis configuration — supports both new task_configs and legacy flat
        analysis_config = config.get("analysis", {})
        _tc_raw = analysis_config.get("task_configs", {})
        _theo_cfg = (
            dict(_tc_raw.get("theoretical", {}))
            if hasattr(_tc_raw, "get")
            else {}
        )

        def _tc(key: str, default: Any = None) -> Any:
            """Read from task_configs.theoretical first, then flat analysis config."""
            if key in _theo_cfg:
                return _theo_cfg[key]
            return analysis_config.get(key, default)

        self.ppm_tol = _tc("ppm_tol", 10.0)
        self.cid_da_tol = _tc("cid_da_tol", DEFAULT_CID_DA_TOL)
        self.enable_theoretical = _tc("enable_theoretical", True)
        self.keep_modifications = _tc("keep_modifications", True)
        self.theoretical_engine = _tc("theoretical_engine", "pyopenms")

        # Conditional annotation configuration
        self.use_conditional_annotation = _tc("use_conditional_annotation", True)
        self.add_precursor = _tc("add_precursor", True)
        self.add_losses = _tc("add_losses", True)
        self.loss_types = tuple(_tc("loss_types", ["H2O", "NH3"]))
        self.add_isotopes = _tc("add_isotopes", True)
        self.isotope_intensity_threshold = _tc("isotope_intensity_threshold", 0.02)
        self.max_isotope = _tc("max_isotope", 4)

        # Mass error analysis
        self.enable_mass_error_analysis = _tc("enable_mass_error_analysis", True)

        # m/z range classification
        self.mz_range_boundaries = _tc(
            "mz_range_boundaries",
            {
                "immonium_internal": (0, 200),
                "core_fragment": (200, 800),
                "extended_fragment": (800, 1500),
                "high_mass_fragment": (1500, float("inf")),
            },
        )
        self.mz_range_order = [
            "immonium_internal",
            "core_fragment",
            "extended_fragment",
            "high_mass_fragment",
        ]

        # Custom ion detection configuration
        self.enable_custom_ions = _tc("enable_custom_ions", False)
        custom_ions_cfg = _tc("custom_ions", None)
        self.custom_ions = dict(custom_ions_cfg) if custom_ions_cfg is not None else None

        # Stratified analysis
        self.enable_stratified_analysis = _tc("enable_stratified_analysis", True)

        # PTM-stratified analysis
        self.enable_modification_analysis = _tc("enable_modification_analysis", False)

        # Complementary b/y pair analysis
        self.enable_complementary_pair_analysis = _tc(
            "enable_complementary_pair_analysis", True
        )

        # Mass gap validation
        self.enable_mass_gap_analysis = _tc("enable_mass_gap_analysis", True)
        self.mass_gap_ppm_tol = _tc("mass_gap_ppm_tol", 20.0)

        # Quality gate analysis configuration
        theo_analysis_config = _tc("theoretical_analysis", {})
        self.enable_quality_gate = theo_analysis_config.get(
            "enable_quality_gate_analysis", True
        )
        self.quality_gate_seq_len_bins = list(
            theo_analysis_config.get(
                "quality_gate_sequence_length_bins", [7, 10, 15, 20, 25, 30]
            )
        )
        # When true, apply the gate as a hard filter before running the
        # downstream analyses (coverage, signal composition, mass error,
        # neutral loss, complementary pair, mass gap, modification matching
        # quality, stratified). When false, every analysis runs on the full
        # sequence-available population and the gate is purely diagnostic.
        # Has no effect when enable_quality_gate is False.
        self.apply_quality_gate_filter = theo_analysis_config.get(
            "apply_quality_gate_filter", True
        )

        # Backbone coverage quality gate thresholds
        self.min_backbone_coverage = _tc("min_backbone_coverage", 0.15)
        self.min_fragment_groups = _tc("min_fragment_groups", 3)

        # Parallelism for aggregation-phase per-peptide analyses (complementary
        # pair + mass gap). Reads `analysis.n_workers` like SpectrumAnalyser.
        n_workers_cfg = analysis_config.get("n_workers", None)
        if n_workers_cfg is not None:
            self.n_workers = int(n_workers_cfg)
        else:
            self.n_workers = max(1, cpu_count() - 1)

        # Check availability
        self.theoretical_available = self.enable_theoretical
        if self.enable_theoretical:
            try:
                from instanovo_fm.utils.theoretical_spectra import (
                    generate_theoretical_spectrum,
                )

                try:
                    import rustyms  # noqa: F401

                    self.rustyms_available = True
                except ImportError:
                    self.rustyms_available = False
            except ImportError:
                self.theoretical_available = False
                self.rustyms_available = False
                logger.warning(
                    "Theoretical spectra module not available. "
                    "Skipping theoretical analysis."
                )
        else:
            logger.info("Theoretical analysis disabled by configuration.")
            self.rustyms_available = False

        # Cache: (sequence, charge, ion_types) -> (theo_mz, theo_annotations)
        # ion_types is part of the key because _ion_types_for_mode returns
        # different sets for HCD/CID vs ETD/UVPD; reusing an (a,b,y) cache
        # entry for an ETD spectrum would silently mis-annotate c/z ions.
        self.theoretical_cache: Dict[
            Tuple[str, int, Tuple[str, ...]], Tuple[np.ndarray, List[str]]
        ] = {}

        # Aggregated results (populated by aggregate_results)
        self.results: Dict[str, Any] = {}

        logger.debug(
            f"Theoretical analyzer initialized (output: {self.output_dir})"
        )

    # =========================================================================
    # Static Utilities
    # =========================================================================

    @staticmethod
    def _extract_field(
        data: Dict[str, Any], field_names: list[str], convert_fn=None
    ):
        """Extract first available field from *data*."""
        for field in field_names:
            if field in data and data[field] is not None:
                value = data[field]
                return convert_fn(value) if convert_fn else value
        return None

    @staticmethod
    def _ion_types_for_mode(frag_type: Optional[str]) -> tuple[str, ...]:
        """Return ion types appropriate for the given fragmentation method."""
        if not frag_type:
            return ("b", "y")
        ft = str(frag_type).strip().upper()
        if ft in ("CID", "HCD", "HCID"):
            return ("a", "b", "y")
        if ft in ("ETD", "ECD"):
            return ("c", "z")
        if ft == "UVPD":
            return ("a", "b", "c", "x", "y", "z")
        return ("b", "y")

    @staticmethod
    def _primary_pair_for_mode(frag_type: Optional[str]) -> tuple[str, str]:
        """Return the canonical (N-terminal, C-terminal) base ion pair
        for the given fragmentation method.

        Used by per-series visualisations / CSV rows that want to show
        one N/C complementary pair per frag_type: b/y for collisional
        activation (HCD/HCID/CID, UVPD), c/z for electron-driven methods
        (ETD/ECD).
        """
        if not frag_type:
            return ("b", "y")
        ft = str(frag_type).strip().upper()
        if ft in ("ETD", "ECD"):
            return ("c", "z")
        return ("b", "y")

    @staticmethod
    def _parse_peak_label(
        feature_type: Optional[str],
        annotation: Optional[str],
        parent_annotation: Optional[str],
    ) -> str:
        """Parse peak annotation into a detailed label.

        Used by both this analyser and :class:`MaskingAnalyser` to classify
        peaks into categories such as ``b-ion``, ``y-loss``, ``precursor``,
        ``unannotated``, etc.
        """
        if feature_type is None:
            return "unannotated"

        inferred_type = feature_type
        ion_annotation = annotation or ""
        if feature_type in ("loss", "isotope") and parent_annotation:
            ion_annotation = parent_annotation

        if isinstance(ion_annotation, str):
            if inferred_type is None:
                if ion_annotation.startswith("p^") or ion_annotation.startswith("p-"):
                    inferred_type = "precursor"
                elif "[+" in ion_annotation:
                    inferred_type = "isotope"
                elif "-" in ion_annotation and any(
                    loss in ion_annotation
                    for loss in ("H2O", "NH3", "CO", "H3PO4")
                ):
                    inferred_type = "loss"
                else:
                    inferred_type = "base"
        else:
            ion_annotation = ""

        if isinstance(ion_annotation, str) and (
            ion_annotation.startswith("p^") or ion_annotation.startswith("p-")
        ):
            if inferred_type == "isotope":
                return "precursor-isotope"
            return "precursor"

        if not isinstance(ion_annotation, str) or len(ion_annotation) == 0:
            return "other"

        ion_type_char = ion_annotation[0].lower()
        suffix_map = {"loss": "-loss", "isotope": "-isotope"}
        if ion_type_char in ("b", "y"):
            suffix = suffix_map.get(inferred_type, "-ion")
            return f"{ion_type_char}{suffix}"

        if inferred_type == "loss":
            return f"{ion_type_char}-loss"
        if inferred_type == "isotope":
            return f"{ion_type_char}-isotope"
        if inferred_type == "base":
            return f"{ion_type_char}-ion"
        if inferred_type == "precursor":
            return "precursor"
        return "other"

    @staticmethod
    def _group_fragment_key(
        feature_type: Optional[str],
        annotation: Optional[str],
        parent_annotation: Optional[str],
    ) -> Optional[str]:
        """Group base peaks with their isotopes for completeness analysis.

        Used by :class:`MaskingAnalyser` to evaluate fragment-group masking.
        """
        ann = annotation or ""
        ftype = feature_type

        if ftype is None and isinstance(ann, str):
            if ann.startswith("p^") or ann.startswith("p-"):
                ftype = "precursor"
            elif "[+" in ann:
                ftype = "isotope"
            elif "-" in ann and any(
                loss in ann for loss in ("H2O", "NH3", "CO", "H3PO4")
            ):
                ftype = "loss"
            else:
                ftype = "base"

        if ftype == "loss":
            return None
        if ftype == "isotope":
            if parent_annotation:
                return parent_annotation
            if isinstance(ann, str) and "[+" in ann:
                return ann.split("[")[0]
            return None
        if ftype in ("base", "precursor"):
            return ann if isinstance(ann, str) else None
        return None

    @staticmethod
    def _extract_ion_type(annotation: str) -> str:
        """Extract base ion type character (e.g. ``'b'``, ``'y'``)."""
        if not annotation:
            return "unknown"
        clean = annotation.split("-")[0].split("[")[0]
        for char in clean:
            if char.isalpha():
                return char.lower()
        return "unknown"

    _KNOWN_LOSS_NAMES = {"H2O", "NH3", "H3PO4", "SO3", "CO"}

    @staticmethod
    def _extract_loss_type(annotation: str) -> Optional[str]:
        """Extract neutral loss type from an annotation string.

        Examples::

            >>> TheoreticalAnalyser._extract_loss_type("b3+-H2O")
            'H2O'
            >>> TheoreticalAnalyser._extract_loss_type("y5+-NH3")
            'NH3'
            >>> TheoreticalAnalyser._extract_loss_type("b3+")
            None
        """
        if not annotation or "-" not in annotation:
            return None
        # Split on '-' and check trailing parts against known losses
        parts = annotation.split("-")
        for part in parts[1:]:
            # Strip trailing charge indicators ('+', '++'), PSI mzPAF '^z', and isotope brackets
            clean = part.split("[")[0].split("^")[0].strip("+")
            if clean in TheoreticalAnalyser._KNOWN_LOSS_NAMES:
                return clean
        return None

    def _classify_mz_range(self, mz: float) -> str:
        """Classify an m/z value into a named range."""
        for range_name in self.mz_range_order:
            low, high = self.mz_range_boundaries[range_name]
            if low <= mz < high:
                return range_name
        return self.mz_range_order[-1]

    def _extract_charge_from_annotation(self, annotation: str) -> int:
        """Extract charge state from an annotation string.

        Handles both formats:
        - Fragment ions: ``'b3+'`` → 1, ``'y5++'`` → 2 (count ``+`` symbols)
        - Precursor ions (PSI mzPAF): ``'p^2'`` → 2, ``'p-H2O^3'`` → 3 (parse ``^z``)
        """
        if not annotation:
            return 1
        # PSI mzPAF '^z' notation — precursor ions
        caret_idx = annotation.rfind('^')
        if caret_idx != -1:
            charge_str = ''
            for ch in annotation[caret_idx + 1:]:
                if ch.isdigit():
                    charge_str += ch
                else:
                    break
            if charge_str:
                return int(charge_str)
        # Fragment ion format — strip isotope brackets then count '+' symbols
        import re
        clean_ann = re.sub(r"\[\+\d+\]", "", annotation)
        count = clean_ann.count("+")
        return max(1, count)

    @staticmethod
    def _tokenize_sequence(seq: str) -> List[str]:
        """Tokenize a peptide sequence into residues, handling modifications.

        Parses sequences like ``'C(Carbamidomethyl)PEPTIDE'`` or
        ``'M[UNIMOD:35]K'`` into individual residue tokens.
        """
        residues: List[str] = []
        i = 0
        while i < len(seq):
            if seq[i].isupper():
                residue = seq[i]
                if i + 1 < len(seq) and seq[i + 1] in ("(", "["):
                    bracket = ")" if seq[i + 1] == "(" else "]"
                    end = seq.find(bracket, i + 2)
                    if end != -1:
                        residue = seq[i:end + 1]
                        i = end + 1
                    else:
                        i += 1
                else:
                    i += 1
                residues.append(residue)
            else:
                i += 1
        return residues

    @staticmethod
    def _extract_fragment_position(annotation: str) -> int:
        """Extract fragment position number (e.g. ``'b3+'`` -> ``3``)."""
        import re

        if not annotation or not isinstance(annotation, str):
            return -1
        if annotation.startswith("p^") or annotation.startswith("p-"):
            return -1
        match = re.match(r"^[a-z](\d+)", annotation.lower())
        if match:
            return int(match.group(1))
        return -1

    @staticmethod
    def _compute_ion_ladders(positions: Set[int]) -> List[List[int]]:
        """Find consecutive position runs from a set of fragment positions.

        E.g., ``{3, 4, 5, 7, 8}`` → ``[[3, 4, 5], [7, 8]]``.
        """
        if not positions:
            return []
        sorted_pos = sorted(positions)
        ladders: List[List[int]] = [[sorted_pos[0]]]
        for p in sorted_pos[1:]:
            if p == ladders[-1][-1] + 1:
                ladders[-1].append(p)
            else:
                ladders.append([p])
        return ladders

    # =========================================================================
    # Custom Ion Utilities
    # =========================================================================

    @staticmethod
    def _classify_ion_category(group_name: str) -> str:
        """Map an ion group key to a high-level category."""
        if group_name.startswith("glycan_"):
            return "glycan"
        if group_name.startswith("immonium_"):
            return "immonium"
        if group_name.startswith("TMT_") or group_name.startswith("TMTpro_"):
            return "TMT"
        if group_name.startswith("iTRAQ_"):
            return "iTRAQ"
        return "other"

    def _detect_custom_ions_per_spectrum(
        self,
        valid_mz: np.ndarray,
        valid_intensity: np.ndarray,
        theoretical_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run custom ion detection and compute coverage lift for one spectrum.

        Uses the conditional two-pass strategy: detect monoisotopic custom
        ions first, then check for isotope peaks of detected ions.

        Returns a dict with ``detection`` (raw result from
        :func:`detect_custom_ions`), ``coverage_by_mz_range`` (per-range
        counts of peaks explained), and ``n_custom_explaining_unannotated``.
        """
        detection = detect_custom_ions(
            exp_mz=valid_mz,
            exp_intensity=valid_intensity,
            custom_ions=self.custom_ions,
            ppm_tol=self.ppm_tol,
            return_details=True,
            add_isotopes=True,
            max_isotope=self.max_isotope,
            isotope_intensity_threshold=self.isotope_intensity_threshold,
        )

        # Build annotated_mask (from fragment matching), default all-False
        raw_mask = theoretical_analysis.get("annotated_mask", [])
        if len(raw_mask) == len(valid_mz):
            annotated_mask = np.asarray(raw_mask, dtype=bool)
        else:
            annotated_mask = np.zeros(len(valid_mz), dtype=bool)

        # Initialise per-m/z-range counters
        coverage_by_mz_range: Dict[str, Dict[str, int]] = {}
        for range_name in self.mz_range_order:
            low, high = self.mz_range_boundaries[range_name]
            in_range = (valid_mz >= low) & (valid_mz < high)
            coverage_by_mz_range[range_name] = {
                "n_peaks": int(in_range.sum()),
                "n_fragment_annotated": int((in_range & annotated_mask).sum()),
                "n_custom_only": 0,
            }

        # Track which peak indices have already been counted to avoid
        # double-counting when multiple ion groups resolve to the same peak.
        counted_indices: set = set()

        def _count_coverage_lift(mz_val: float) -> None:
            """Check if *mz_val* explains a previously-unannotated peak."""
            idx = np.searchsorted(valid_mz, mz_val)
            best = None
            for ci in (idx - 1, idx):
                if 0 <= ci < len(valid_mz):
                    if (
                        abs(valid_mz[ci] - mz_val)
                        / max(mz_val, 1e-12)
                        * 1e6
                        < self.ppm_tol * 2
                    ):
                        best = ci
                        break
            if best is not None and not annotated_mask[best] and best not in counted_indices:
                counted_indices.add(best)
                peak_mz = valid_mz[best]
                for range_name in self.mz_range_order:
                    low, high = self.mz_range_boundaries[range_name]
                    if low <= peak_mz < high:
                        coverage_by_mz_range[range_name]["n_custom_only"] += 1
                        break

        # Count coverage lift from monoisotopic matches AND isotope matches
        for group_data in detection.values():
            if not group_data.get("found"):
                continue
            # Monoisotopic matches
            for mz_val in group_data.get("matched_mz", []):
                _count_coverage_lift(mz_val)
            # Isotope matches (conditional Pass 2 results)
            for iso_match in group_data.get("isotope_matches", []):
                _count_coverage_lift(iso_match["matched_mz"])

        return {
            "detection": detection,
            "coverage_by_mz_range": coverage_by_mz_range,
            "n_custom_explaining_unannotated": len(counted_indices),
        }

    # =========================================================================
    # Core Per-Spectrum Analysis
    # =========================================================================

    def analyze_spectrum(
        self,
        spectrum_data: Dict[str, Any],
        valid_mz: np.ndarray,
        valid_intensity: np.ndarray,
        mlm_mask: Optional[np.ndarray],
        processor: FoundationalDataProcessor,
        batch_result: Dict[str, Any],
        theoretical_cache: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Perform theoretical analysis on a single spectrum.

        Generates a theoretical spectrum from the peptide sequence, matches it
        to the experimental peaks, and collects per-peak mass error data.

        Args:
            spectrum_data: Raw spectrum metadata dictionary.
            valid_mz: Filtered m/z array (peaks within valid range).
            valid_intensity: Filtered intensity array (normalized).
            mlm_mask: Unused (kept for interface compatibility with callers).
            processor: FoundationalDataProcessor instance.
            batch_result: Batch processing result with metadata.
            theoretical_cache: Optional external cache to merge.

        Returns:
            Dictionary with keys ``theoretical_analysis``, ``mass_error_data``,
            ``unmatched_theo_data``, and ``masking_analysis`` (always ``{}``).
        """
        if theoretical_cache is not None:
            self.theoretical_cache.update(theoretical_cache)

        sequence = self._extract_field(
            spectrum_data, ["sequence", "modified_peptide", "peptide"]
        )
        precursor_charge = self._extract_field(
            spectrum_data, ["precursor_charge", "charge"], int
        )

        mass_error_data: List[Dict] = []
        unmatched_theo_data: List[Dict] = []
        theoretical_analysis: Dict[str, Any] = {}

        if (
            sequence is not None
            and precursor_charge is not None
            and self.theoretical_available
        ):
            frag_type_raw = (
                batch_result.get("frag_type", [None])[0]
                if "frag_type" in batch_result
                else None
            )
            frag_type = str(frag_type_raw) if frag_type_raw is not None else None
            ion_types_used = self._ion_types_for_mode(frag_type)

            if self.theoretical_engine == "pyopenms":
                clean_sequence = DataProcessor.clean_peptide_for_pyopenms(
                    sequence, keep_modifications=self.keep_modifications
                )
            else:
                clean_sequence = sequence

            if clean_sequence is not None:
                try:
                    theoretical_analysis, mass_error_data, unmatched_theo_data = (
                        self._match_and_collect(
                            clean_sequence=clean_sequence,
                            sequence=sequence,
                            precursor_charge=precursor_charge,
                            ion_types_used=ion_types_used,
                            frag_type=frag_type,
                            valid_mz=valid_mz,
                            valid_intensity=valid_intensity,
                        )
                    )
                    theoretical_analysis["modifications_stripped"] = False
                except Exception as e:
                    # Fallback: strip modifications (PyOpenMS only)
                    if (
                        self.keep_modifications
                        and self.theoretical_engine == "pyopenms"
                    ):
                        logger.warning(
                            f"PyOpenMS failed with modified sequence "
                            f"'{clean_sequence[:50]}...': {e}. "
                            "Trying without modifications."
                        )
                        clean_unmod = DataProcessor.clean_peptide_for_pyopenms(
                            sequence, keep_modifications=False
                        )
                        if clean_unmod is not None:
                            try:
                                theoretical_analysis, mass_error_data, unmatched_theo_data = (
                                    self._match_and_collect(
                                        clean_sequence=clean_unmod,
                                        sequence=sequence,
                                        precursor_charge=precursor_charge,
                                        ion_types_used=ion_types_used,
                                        frag_type=frag_type,
                                        valid_mz=valid_mz,
                                        valid_intensity=valid_intensity,
                                    )
                                )
                                theoretical_analysis["modifications_stripped"] = True
                            except Exception as e2:
                                logger.error(
                                    f"Failed even without modifications: {e2}"
                                )
                                theoretical_analysis = self._empty_analysis(
                                    n_peaks=len(valid_mz),
                                    sequence_available=True,
                                    error="Failed to generate theoretical spectrum",
                                    precursor_charge=precursor_charge,
                                )
                        else:
                            theoretical_analysis = self._empty_analysis(
                                n_peaks=len(valid_mz),
                                sequence_available=True,
                                error="Failed to clean peptide sequence",
                                precursor_charge=precursor_charge,
                            )
                    else:
                        logger.error(f"Theoretical analysis failed: {e}")
                        theoretical_analysis = self._empty_analysis(
                            n_peaks=len(valid_mz),
                            sequence_available=True,
                            error=str(e),
                            precursor_charge=precursor_charge,
                        )
            else:
                theoretical_analysis = self._empty_analysis(
                    n_peaks=len(valid_mz),
                    sequence_available=True,
                    error="Failed to clean peptide sequence",
                    precursor_charge=precursor_charge,
                )
        else:
            theoretical_analysis = self._empty_analysis(
                n_peaks=len(valid_mz), sequence_available=False
            )

        # Custom ion detection (independent of sequence availability)
        if self.enable_custom_ions:
            theoretical_analysis["custom_ion_data"] = self._detect_custom_ions_per_spectrum(
                valid_mz, valid_intensity, theoretical_analysis,
            )

            # Append custom ion matches to mass_error_data for binning analysis
            if self.enable_mass_error_analysis:
                custom_ion_data = theoretical_analysis.get("custom_ion_data", {})
                detection = custom_ion_data.get("detection", {})
                ion_library = self.custom_ions or DEFAULT_CUSTOM_IONS

                # Extract frag_type for custom ion entries
                frag_type_raw = (
                    batch_result.get("frag_type", [None])[0]
                    if "frag_type" in batch_result
                    else None
                )
                frag_type_ci = str(frag_type_raw) if frag_type_raw is not None else None

                for group_name, group_data in detection.items():
                    if not group_data.get("found"):
                        continue
                    target_mz_list = ion_library.get(group_name, [])
                    if not target_mz_list:
                        continue
                    for exp_mz_val in group_data.get("matched_mz", []):
                        # Find closest target m/z for this match
                        closest_target = min(
                            target_mz_list, key=lambda t: abs(t - exp_mz_val)
                        )
                        delta_da = abs(exp_mz_val - closest_target)
                        delta_ppm = (
                            delta_da / closest_target * 1e6
                            if closest_target > 0
                            else 0.0
                        )
                        signed_ppm = (
                            (exp_mz_val - closest_target) / closest_target * 1e6
                            if closest_target > 0
                            else 0.0
                        )
                        mass_error_data.append(
                            {
                                "theo_mz": float(closest_target),
                                "exp_mz": float(exp_mz_val),
                                "delta_mz_da": float(delta_da),
                                "delta_mz_ppm": float(delta_ppm),
                                "signed_ppm": float(signed_ppm),
                                "ion_type": "custom",
                                "charge": 1,
                                "mz_range": self._classify_mz_range(closest_target),
                                "frag_type": frag_type_ci,
                                "feature_type": "custom",
                                "peptide": sequence,
                                "position": -1,
                                "annotation": group_name,
                            }
                        )

        # Precursor mass validation (pyopenms only, requires successful match)
        if (
            self.theoretical_engine == "pyopenms"
            and theoretical_analysis.get("n_matched", 0) > 0
            and precursor_charge is not None
        ):
            observed_mz = (
                float(batch_result["precursor_mz"][0])
                if "precursor_mz" in batch_result
                else None
            )
            if observed_mz is not None:
                prec_validation = self._validate_precursor_mass(
                    clean_sequence=theoretical_analysis.get(
                        "clean_sequence", clean_sequence or ""
                    ),
                    precursor_charge=precursor_charge,
                    observed_precursor_mz=observed_mz,
                )
                theoretical_analysis.update(prec_validation)

        return {
            "theoretical_analysis": theoretical_analysis,
            "mass_error_data": mass_error_data,
            "unmatched_theo_data": unmatched_theo_data,
            "masking_analysis": {},
        }

    def _match_and_collect(
        self,
        clean_sequence: str,
        sequence: str,
        precursor_charge: int,
        ion_types_used: Tuple[str, ...],
        frag_type: Optional[str],
        valid_mz: np.ndarray,
        valid_intensity: np.ndarray,
    ) -> Tuple[Dict[str, Any], List[Dict], List[Dict]]:
        """Generate theoretical spectrum, match, and collect per-peak data.

        Returns:
            (theoretical_analysis dict, mass_error_data list, unmatched_theo_data list)
        """
        # --- Generate or fetch from cache ---
        # ion_types must be in the cache key: _ion_types_for_mode returns
        # different sets per fragmentation method (a/b/y for HCD/CID/HCID,
        # c/z for ETD/ECD, all six for UVPD), and reusing a cached entry
        # with a mismatched ion set would silently produce wrong annotations.
        cache_key = (clean_sequence, precursor_charge, tuple(ion_types_used))
        if cache_key in self.theoretical_cache:
            theo_mz, theo_annotations = self.theoretical_cache[cache_key]
            theo_annotations_list = (
                theo_annotations if isinstance(theo_annotations, list) else []
            )
        else:
            theo_mz, theo_annotations = generate_theoretical_spectrum(
                peptide=clean_sequence,
                precursor_charge=precursor_charge,
                ion_types=ion_types_used,
                max_charge=precursor_charge - 1,
                add_losses=False,
                add_isotopes=False,
                add_precursor=self.add_precursor,
                add_custom_ions=False,
                custom_ions=None,
                engine=self.theoretical_engine,
                fragmentation_type=frag_type,
            )
            self.theoretical_cache[cache_key] = (theo_mz, theo_annotations)
            theo_annotations_list = (
                theo_annotations if isinstance(theo_annotations, list) else []
            )

        # --- Match experimental to theoretical ---
        # Auto-select Da tolerance for low-res CID; None = use ppm_tol
        da_tol = (
            _da_tol_for_fragmentation(frag_type, self.cid_da_tol)
            if self.cid_da_tol is not None
            else None
        )
        if self.use_conditional_annotation:
            match_results = match_with_conditional_features(
                exp_mz=valid_mz,
                exp_intensity=valid_intensity,
                peptide=clean_sequence,
                precursor_charge=precursor_charge,
                ppm_tol=self.ppm_tol,
                da_tol=da_tol,
                ion_types=ion_types_used,
                max_charge=precursor_charge - 1 if precursor_charge > 1 else 1,
                add_losses=self.add_losses,
                loss_types=self.loss_types,
                add_isotopes=self.add_isotopes,
                max_isotope=self.max_isotope,
                isotope_intensity_threshold=self.isotope_intensity_threshold,
                add_precursor=self.add_precursor,
                engine=self.theoretical_engine,
                fragmentation_type=frag_type,
            )
        else:
            # Non-conditional path: the cache above only holds bases + precursor
            # (add_losses=False, add_isotopes=False) because the conditional
            # path generates losses/isotopes itself. Here we must regenerate
            # so self.add_losses / self.add_isotopes are actually honoured.
            if self.add_losses or self.add_isotopes:
                full_theo_mz, full_theo_ann = generate_theoretical_spectrum(
                    peptide=clean_sequence,
                    precursor_charge=precursor_charge,
                    ion_types=ion_types_used,
                    max_charge=precursor_charge - 1 if precursor_charge > 1 else 1,
                    add_losses=self.add_losses,
                    loss_types=self.loss_types,
                    add_isotopes=self.add_isotopes,
                    max_isotope=self.max_isotope,
                    add_precursor=self.add_precursor,
                    precursor_isotopes=self.add_precursor,
                    max_precursor_isotope=self.max_isotope,
                    engine=self.theoretical_engine,
                    fragmentation_type=frag_type,
                )
                theo_mz = full_theo_mz
                theo_annotations = full_theo_ann
                theo_annotations_list = (
                    full_theo_ann if isinstance(full_theo_ann, list) else []
                )
            match_results = match_theoretical_to_experimental(
                exp_mz=valid_mz,
                exp_intensity=valid_intensity,
                theo_mz=theo_mz,
                ppm_tol=self.ppm_tol,
                da_tol=da_tol,
            )

        # --- Build annotation arrays ---
        annotated_mask = match_results["mask"]
        match_idx = match_results["match_idx"]

        if self.use_conditional_annotation and "matched_annotation" in match_results:
            matched_annotations = match_results["matched_annotation"]
        else:
            matched_annotations = []
            if theo_annotations_list:
                for exp_idx in range(len(valid_mz)):
                    theo_idx = match_idx[exp_idx]
                    if 0 <= theo_idx < len(theo_annotations_list):
                        matched_annotations.append(theo_annotations_list[theo_idx])
                    else:
                        matched_annotations.append("")
            else:
                matched_annotations = [""] * len(valid_mz)

        feature_types = match_results.get("feature_type", None)
        parent_annotations = match_results.get("parent_annotation", None)
        if not feature_types:
            feature_types = [
                "base" if (ann is not None and ann != "") else None
                for ann in matched_annotations
            ]
        if not parent_annotations:
            parent_annotations = [None] * len(matched_annotations)

        annotation_labels = [
            self._parse_peak_label(
                feature_types[i] if i < len(feature_types) else None,
                matched_annotations[i] if i < len(matched_annotations) else None,
                parent_annotations[i] if i < len(parent_annotations) else None,
            )
            for i in range(len(matched_annotations))
        ]

        label_counts: Dict[str, int] = {}
        for label in annotation_labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        # Per-label intensity accumulation (for intensity-weighted composition)
        label_intensity: Dict[str, float] = {}
        for i, label in enumerate(annotation_labels):
            if i < len(valid_intensity):
                label_intensity[label] = label_intensity.get(label, 0.0) + float(
                    valid_intensity[i]
                )

        # --- Per-peak neutral loss details ---
        loss_details: List[Dict[str, Any]] = []
        for i in range(len(matched_annotations)):
            ft = feature_types[i] if i < len(feature_types) else None
            if ft != "loss":
                continue
            ann = matched_annotations[i] if i < len(matched_annotations) else ""
            loss_type = self._extract_loss_type(ann)
            if loss_type is None:
                continue

            # Parent base ion info
            parent_ann = parent_annotations[i] if i < len(parent_annotations) else None
            ion_series = self._extract_ion_type(parent_ann) if parent_ann else "unknown"

            # Intensity ratio: loss peak / parent peak
            loss_intensity = float(valid_intensity[i])
            parent_intensity = None
            if parent_ann:
                for j in range(len(matched_annotations)):
                    if (
                        j < len(feature_types)
                        and feature_types[j] == "base"
                        and matched_annotations[j] == parent_ann
                    ):
                        parent_intensity = float(valid_intensity[j])
                        break
            intensity_ratio = (
                loss_intensity / parent_intensity
                if parent_intensity is not None and parent_intensity > 1e-12
                else None
            )

            loss_details.append(
                {
                    "loss_type": loss_type,
                    "annotation": ann,
                    "parent_annotation": parent_ann or "",
                    "ion_series": ion_series,
                    "loss_intensity": loss_intensity,
                    "parent_intensity": parent_intensity,
                    "intensity_ratio": intensity_ratio,
                }
            )

        # --- Annotation statistics ---
        n_annotated = int(annotated_mask.sum())
        n_unannotated = len(annotated_mask) - n_annotated
        total_intensity = float(valid_intensity.sum())
        annotated_intensity = float(valid_intensity[annotated_mask].sum())
        unannotated_intensity = float(valid_intensity[~annotated_mask].sum())

        theoretical_analysis: Dict[str, Any] = {
            "sequence_available": True,
            "clean_sequence": clean_sequence,
            "precursor_charge": precursor_charge,
            "n_theoretical": len(theo_mz),
            "n_matched": match_results["metrics"]["n_matched"],
            "match_rate": match_results["metrics"]["n_matched"]
            / max(len(theo_mz), 1),
            "frac_intensity": match_results["metrics"]["frac_intensity"],
            "median_ppm": match_results["metrics"]["median_abs_ppm"],
            "mean_ppm": match_results["metrics"]["mean_ppm_bias"],
            "n_annotated_peaks": n_annotated,
            "n_unannotated_peaks": n_unannotated,
            "annotated_fraction": n_annotated / max(len(annotated_mask), 1),
            "unannotated_fraction": n_unannotated / max(len(annotated_mask), 1),
            "annotated_intensity_fraction": annotated_intensity
            / max(total_intensity, 1e-12),
            "unannotated_intensity_fraction": unannotated_intensity
            / max(total_intensity, 1e-12),
            "annotated_mask": annotated_mask.tolist(),
            "theo_annotations": matched_annotations,
            "annotation_labels": annotation_labels,
            "annotation_label_counts": label_counts,
            "annotation_label_intensities": label_intensity,
            "_annotated_intensity_values": valid_intensity[annotated_mask].tolist(),
            "_unannotated_intensity_values": valid_intensity[~annotated_mask].tolist(),
        }

        if self.use_conditional_annotation and "n_base" in match_results["metrics"]:
            theoretical_analysis.update(
                {
                    "n_base": match_results["metrics"]["n_base"],
                    "n_losses": match_results["metrics"]["n_losses"],
                    "n_isotopes": match_results["metrics"]["n_isotopes"],
                    "feature_types": feature_types,
                    "parent_annotations": parent_annotations,
                    "loss_details": loss_details,
                }
            )

        # --- Per-peak mass error data ---
        mass_error_data: List[Dict] = []
        unmatched_theo_data: List[Dict] = []

        if (
            self.enable_mass_error_analysis
            and match_results
            and len(match_results.get("ppm_error", [])) > 0
        ):
            ppm_errors = match_results["ppm_error"]
            matched_theo_mz = match_results.get(
                "matched_theo_mz", np.full(len(valid_mz), np.nan)
            )

            # Unmatched theoretical ions
            matched_theo_indices = set(match_idx[match_idx >= 0])
            for theo_idx in set(range(len(theo_mz))) - matched_theo_indices:
                theo_mz_val = float(theo_mz[theo_idx])
                theo_ann = (
                    theo_annotations_list[theo_idx]
                    if theo_annotations_list and theo_idx < len(theo_annotations_list)
                    else ""
                )
                unmatched_theo_data.append(
                    {
                        "theo_mz": theo_mz_val,
                        "mz_range": self._classify_mz_range(theo_mz_val),
                        "ion_type": self._extract_ion_type(theo_ann)
                        if theo_ann
                        else "unknown",
                        "frag_type": frag_type,
                    }
                )

            # Matched peak mass errors
            for i, (exp_mz_i, ppm_err, theo_mz_i, ann) in enumerate(
                zip(valid_mz, ppm_errors, matched_theo_mz, matched_annotations)
            ):
                if not np.isnan(ppm_err) and not np.isnan(theo_mz_i):
                    mass_error_data.append(
                        {
                            "theo_mz": float(theo_mz_i),
                            "exp_mz": float(exp_mz_i),
                            "delta_mz_da": float(abs(exp_mz_i - theo_mz_i)),
                            "delta_mz_ppm": float(abs(ppm_err)),
                            "signed_ppm": float(ppm_err),
                            "ion_type": self._extract_ion_type(ann)
                            if ann
                            else "unknown",
                            "charge": self._extract_charge_from_annotation(ann)
                            if ann
                            else 1,
                            "mz_range": self._classify_mz_range(theo_mz_i),
                            "frag_type": frag_type,
                            "feature_type": feature_types[i]
                            if feature_types and i < len(feature_types)
                            else "base",
                            "peptide": sequence,
                            "position": self._extract_fragment_position(ann)
                            if ann
                            else -1,
                            "annotation": ann if ann else "",
                        }
                    )

        return theoretical_analysis, mass_error_data, unmatched_theo_data

    @staticmethod
    def _empty_analysis(
        n_peaks: int,
        sequence_available: bool,
        error: Optional[str] = None,
        precursor_charge: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return a default (empty) theoretical analysis dict."""
        result: Dict[str, Any] = {
            "sequence_available": sequence_available,
            "precursor_charge": precursor_charge,
            "n_theoretical": 0,
            "n_matched": 0,
            "match_rate": 0.0,
            "frac_intensity": 0.0,
            "median_ppm": np.nan,
            "mean_ppm": np.nan,
            "n_annotated_peaks": 0,
            "n_unannotated_peaks": n_peaks,
            "annotated_fraction": 0.0,
            "unannotated_fraction": 1.0,
            "annotated_intensity_fraction": 0.0,
            "unannotated_intensity_fraction": 1.0,
            "modifications_stripped": False,
        }
        if error:
            result["error"] = error
        return result

    # Physical constants
    _NEUTRON_MASS = 1.003355
    _PROTON_MASS = 1.007276466812
    _H2O_MASS = 18.0105647  # monoisotopic water mass

    # Standard amino acid residue masses (monoisotopic, Da)
    # Order: G A S P V T C L I N D Q K E M H F R Y W
    _AA_CODES = [
        "G", "A", "S", "P", "V", "T", "C", "L", "I", "N",
        "D", "Q", "K", "E", "M", "H", "F", "R", "Y", "W",
    ]
    _AA_MASSES = np.array([
        57.021464, 71.037114, 87.032028, 97.052764, 99.068414,   # G A S P V
        101.047670, 103.009185, 113.084064, 113.084064, 114.042927,  # T C L I N
        115.026943, 128.058578, 128.094963, 129.042593, 131.040485,  # D Q K E M
        137.058912, 147.068414, 156.101111, 163.063329, 186.079313,  # H F R Y W
    ], dtype=np.float64)
    _AA_MASS_TO_CODE = dict(zip(_AA_MASSES, _AA_CODES))
    _AA_MASSES_SORTED = np.sort(_AA_MASSES)

    def _validate_precursor_mass(
        self,
        clean_sequence: str,
        precursor_charge: int,
        observed_precursor_mz: float,
        ppm_threshold: float = 20.0,
    ) -> Dict[str, Any]:
        """Compare theoretical and observed precursor m/z.

        Accounts for **isotope peak selection**: the mass spectrometer may
        isolate a non-monoisotopic precursor peak (M+1, M+2, M+3).  The
        function tries offsets of 0..3 neutron masses / charge and picks
        the best PPM match.

        Note: implicit Carbamidomethylation on Cys is handled upstream by
        ``DataProcessor.clean_peptide_for_pyopenms()``, so the
        *clean_sequence* passed here already contains explicit CAM
        annotations.

        Parameters
        ----------
        clean_sequence : str
            PyOpenMS-compatible peptide sequence (with CAM already applied).
        precursor_charge : int
            Precursor charge state.
        observed_precursor_mz : float
            Observed precursor m/z from the spectrum.
        ppm_threshold : float
            PPM threshold above which a warning is raised.

        Returns
        -------
        Dict[str, Any]
            Keys: theoretical_precursor_mz, observed_precursor_mz,
            precursor_ppm_error, precursor_mass_warning,
            precursor_isotope_offset.
        """
        try:
            theo_mz = compute_theoretical_precursor_mz(
                clean_sequence, precursor_charge
            )
        except Exception:
            return {
                "theoretical_precursor_mz": np.nan,
                "observed_precursor_mz": observed_precursor_mz,
                "precursor_ppm_error": np.nan,
                "precursor_mass_warning": False,
                "precursor_isotope_offset": 0,
            }

        # Try M+0 through M+3 isotope offsets, pick the best match
        best_ppm = float("inf")
        best_mz = theo_mz
        best_iso = 0
        for iso in range(4):
            iso_shift = iso * self._NEUTRON_MASS / precursor_charge
            candidate_mz = theo_mz + iso_shift
            ppm = abs(observed_precursor_mz - candidate_mz) / candidate_mz * 1e6
            if ppm < best_ppm:
                best_ppm = ppm
                best_mz = candidate_mz
                best_iso = iso

        return {
            "theoretical_precursor_mz": float(best_mz),
            "observed_precursor_mz": float(observed_precursor_mz),
            "precursor_ppm_error": float(best_ppm),
            "precursor_mass_warning": bool(best_ppm > ppm_threshold),
            "precursor_isotope_offset": int(best_iso),
        }

    # =========================================================================
    # Masking Utility (used by MaskingAnalyser — do not remove)
    # =========================================================================

    @staticmethod
    def _analyze_masking_effect(
        theoretical_analysis: Dict[str, Any],
        valid_intensity: np.ndarray,
        mlm_mask: np.ndarray,
        matched_annotations: List[str],
        feature_types: List[Optional[str]],
        parent_annotations: List[Optional[str]],
        custom_ion_peak_mask: Optional[np.ndarray] = None,
        frag_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze masking effect on annotated vs unannotated peaks.

        This is a static utility consumed by :class:`MaskingAnalyser`.  It
        computes masking statistics including annotated/unannotated masking
        ratios, intensity-based metrics, per-annotation-type counts, and
        fragment-group completeness.
        """
        if not theoretical_analysis.get("sequence_available", False):
            return {}
        if "annotated_mask" not in theoretical_analysis:
            return {}

        annotated_mask = np.array(theoretical_analysis["annotated_mask"])
        masked_peaks = mlm_mask

        n_annotated_masked = int((annotated_mask & masked_peaks).sum())
        n_unannotated_masked = int((masked_peaks & ~annotated_mask).sum())
        n_annotated_unmasked = int((annotated_mask & ~masked_peaks).sum())
        n_unannotated_unmasked = int((~annotated_mask & ~masked_peaks).sum())

        total_masked = int(masked_peaks.sum())
        annotated_fraction_of_masked = float(n_annotated_masked) / max(
            total_masked, 1
        )

        # Intensity-based metrics
        annotated_int_masked = float(
            valid_intensity[annotated_mask & masked_peaks].sum()
        )
        unannotated_int_masked = float(
            valid_intensity[~annotated_mask & masked_peaks].sum()
        )
        annotated_int_unmasked = float(
            valid_intensity[annotated_mask & ~masked_peaks].sum()
        )
        unannotated_int_unmasked = float(
            valid_intensity[~annotated_mask & ~masked_peaks].sum()
        )
        total_int_masked = annotated_int_masked + unannotated_int_masked
        annotated_int_frac_masked = annotated_int_masked / max(total_int_masked, 1e-12)
        total_annotated_int = annotated_int_masked + annotated_int_unmasked
        total_unannotated_int = unannotated_int_masked + unannotated_int_unmasked
        ann_int_masked_frac = annotated_int_masked / max(total_annotated_int, 1e-12)
        unann_int_masked_frac = unannotated_int_masked / max(
            total_unannotated_int, 1e-12
        )

        # Per-annotation type
        annotation_labels = theoretical_analysis.get("annotation_labels", [])
        annotated_type_total: Dict[str, int] = {}
        annotated_type_masked: Dict[str, int] = {}
        annotated_type_intensity_total: Dict[str, float] = {}
        annotated_type_intensity_masked: Dict[str, float] = {}
        for idx, label in enumerate(annotation_labels):
            if label == "unannotated":
                continue
            annotated_type_total[label] = annotated_type_total.get(label, 0) + 1
            peak_int = float(valid_intensity[idx]) if idx < len(valid_intensity) else 0.0
            annotated_type_intensity_total[label] = (
                annotated_type_intensity_total.get(label, 0.0) + peak_int
            )
            if idx < len(masked_peaks) and masked_peaks[idx]:
                annotated_type_masked[label] = (
                    annotated_type_masked.get(label, 0) + 1
                )
                annotated_type_intensity_masked[label] = (
                    annotated_type_intensity_masked.get(label, 0.0) + peak_int
                )
        annotated_type_mask_ratio = {
            label: masked / max(annotated_type_total.get(label, 0), 1)
            for label, masked in annotated_type_masked.items()
        }
        annotated_type_intensity_mask_ratio = {
            label: annotated_type_intensity_masked.get(label, 0.0)
            / max(annotated_type_intensity_total.get(label, 0.0), 1e-12)
            for label in annotated_type_intensity_total
        }

        # Fragment group completeness — split precursor from fragment groups
        fragment_group_totals: Dict[str, int] = {}
        fragment_group_masked: Dict[str, int] = {}
        precursor_group_totals: Dict[str, int] = {}
        precursor_group_masked: Dict[str, int] = {}
        precursor_peak_indices: List[int] = []
        precursor_base_key: Optional[str] = None

        for idx, label in enumerate(annotation_labels):
            if label == "unannotated":
                continue
            ft = feature_types[idx] if idx < len(feature_types) else None
            ann = matched_annotations[idx] if idx < len(matched_annotations) else None
            parent = parent_annotations[idx] if idx < len(parent_annotations) else None
            group_key = TheoreticalAnalyser._group_fragment_key(ft, ann, parent)
            if not group_key:
                continue

            # Determine if this is a precursor peak
            is_precursor = label in ("precursor", "precursor-isotope")
            if is_precursor:
                precursor_group_totals[group_key] = (
                    precursor_group_totals.get(group_key, 0) + 1
                )
                precursor_peak_indices.append(idx)
                if masked_peaks[idx]:
                    precursor_group_masked[group_key] = (
                        precursor_group_masked.get(group_key, 0) + 1
                    )
                # Track the base precursor key (monoisotopic: p^...)
                ann_str = ann or ""
                if ft == "precursor" and isinstance(ann_str, str) and ann_str.startswith("p^"):
                    precursor_base_key = group_key
            else:
                fragment_group_totals[group_key] = (
                    fragment_group_totals.get(group_key, 0) + 1
                )
                if masked_peaks[idx]:
                    fragment_group_masked[group_key] = (
                        fragment_group_masked.get(group_key, 0) + 1
                    )

        # --- Split into "primary" (training-relevant) vs "all" views ---
        # The headline fragment-group metrics are reported against the
        # frag_type's canonical backbone ion pair (b/y for collisional
        # activation, c/z for ETD/ECD). The reason: signal_aware_fragment
        # only considers that pair as maskable, so counting a-ions /
        # x-ions in the denominator was inflating "unmasked groups"
        # across every strategy. The "all-types" counts are retained
        # under ``_all_*`` keys for diagnostics.
        primary_pair = TheoreticalAnalyser._primary_pair_for_mode(frag_type)
        primary_letters = set(primary_pair)

        def _first_alpha(key: str) -> str:
            for ch in key:
                if ch.isalpha():
                    return ch.lower()
            return ""

        def _is_primary_group(key: str) -> bool:
            return _first_alpha(key) in primary_letters

        fragment_group_totals_all = fragment_group_totals
        fragment_group_masked_all = fragment_group_masked
        fragment_group_totals_primary = {
            k: v for k, v in fragment_group_totals_all.items() if _is_primary_group(k)
        }
        fragment_group_masked_primary = {
            k: v for k, v in fragment_group_masked_all.items() if _is_primary_group(k)
        }

        def _group_stats(
            totals: Dict[str, int], masked: Dict[str, int]
        ) -> Dict[str, Any]:
            """Compute group-level mask statistics for one totals/masked
            pair (primary or all). Returns counts + full-mask ratio +
            per-group mask fractions so callers can reconstruct both the
            ratio ("how complete is each group") and the count views
            ("how many groups are fully/partial/un-masked")."""
            count = len(totals)
            if not count:
                return {
                    "count": 0,
                    "fractions": [],
                    "full_mask_ratio": 0.0,
                    "avg_mask_fraction": 0.0,
                    "weighted_mask_fraction": 0.0,
                    "n_fully_masked": 0,
                    "n_partially_masked": 0,
                    "n_unmasked": 0,
                }
            fractions = [
                masked.get(key, 0) / max(total, 1)
                for key, total in totals.items()
            ]
            n_fully = sum(
                1 for key, total in totals.items()
                if masked.get(key, 0) >= total
            )
            n_partial = sum(
                1 for key, total in totals.items()
                if 0 < masked.get(key, 0) < total
            )
            n_unmasked = sum(1 for key in totals if masked.get(key, 0) == 0)
            return {
                "count": count,
                "fractions": fractions,
                "full_mask_ratio": n_fully / count,
                "avg_mask_fraction": float(np.mean(fractions)),
                "weighted_mask_fraction": (
                    sum(masked.values()) / max(sum(totals.values()), 1)
                ),
                "n_fully_masked": n_fully,
                "n_partially_masked": n_partial,
                "n_unmasked": n_unmasked,
            }

        primary_stats = _group_stats(
            fragment_group_totals_primary, fragment_group_masked_primary
        )
        all_stats = _group_stats(fragment_group_totals_all, fragment_group_masked_all)

        # Headline metrics use the primary view.
        fragment_group_count = primary_stats["count"]
        group_mask_fractions = primary_stats["fractions"]
        fragment_group_full_mask_ratio = primary_stats["full_mask_ratio"]
        fragment_group_avg_mask_fraction = primary_stats["avg_mask_fraction"]
        fragment_group_weighted_mask_fraction = primary_stats["weighted_mask_fraction"]
        n_fragment_groups_total = primary_stats["count"]
        n_fragment_groups_fully_masked = primary_stats["n_fully_masked"]
        n_fragment_groups_partially_masked = primary_stats["n_partially_masked"]
        n_fragment_groups_unmasked = primary_stats["n_unmasked"]
        n_fragment_groups_any_masked = (
            n_fragment_groups_fully_masked + n_fragment_groups_partially_masked
        )

        # --- Per-ion-series group masking ---
        series_group_total: Dict[str, int] = {}
        series_group_any_masked: Dict[str, int] = {}
        series_group_fully_masked: Dict[str, int] = {}
        series_group_partially_masked: Dict[str, int] = {}
        series_group_unmasked: Dict[str, int] = {}
        for key, total in fragment_group_totals.items():
            # Extract ion series from the first alphabetic char of the group key
            series = "unknown"
            for ch in key:
                if ch.isalpha():
                    series = ch.lower()
                    break
            series_group_total[series] = series_group_total.get(series, 0) + 1
            masked_count = fragment_group_masked.get(key, 0)
            if masked_count > 0:
                series_group_any_masked[series] = (
                    series_group_any_masked.get(series, 0) + 1
                )
            if masked_count >= total:
                series_group_fully_masked[series] = (
                    series_group_fully_masked.get(series, 0) + 1
                )
            elif masked_count > 0:
                series_group_partially_masked[series] = (
                    series_group_partially_masked.get(series, 0) + 1
                )
            else:
                series_group_unmasked[series] = (
                    series_group_unmasked.get(series, 0) + 1
                )

        # --- Precursor ion group metrics ---
        precursor_n_peaks = sum(precursor_group_totals.values())
        precursor_n_masked = sum(precursor_group_masked.values())
        precursor_base_masked = bool(
            precursor_base_key
            and precursor_group_masked.get(precursor_base_key, 0) > 0
        )
        precursor_group_mask_fraction = (
            precursor_n_masked / max(precursor_n_peaks, 1)
        )
        precursor_fully_masked = bool(
            precursor_n_peaks > 0 and precursor_n_masked >= precursor_n_peaks
        )

        # Mask budget fractions (count-based and intensity-based)
        precursor_mask_budget_fraction = (
            precursor_n_masked / max(total_masked, 1)
        )
        precursor_masked_intensity = float(
            sum(
                valid_intensity[i]
                for i in precursor_peak_indices
                if i < len(masked_peaks) and masked_peaks[i]
            )
        )
        precursor_intensity_budget_fraction = (
            precursor_masked_intensity / max(total_int_masked, 1e-12)
        )

        # Precursor share of the *annotated* masking signal
        precursor_fraction_of_annotated_masked = (
            precursor_n_masked / max(n_annotated_masked, 1)
        )
        precursor_intensity_fraction_of_annotated_masked = (
            precursor_masked_intensity / max(annotated_int_masked, 1e-12)
        )

        # Precursor leakage: base precursor masked but some children unmasked
        precursor_has_leakage = False
        if precursor_base_masked and precursor_n_peaks > 1:
            precursor_has_leakage = precursor_n_masked < precursor_n_peaks

        # --- Custom ion masking metrics ---
        custom_n_peaks = 0
        custom_n_masked = 0
        custom_mask_ratio = 0.0
        custom_mask_budget_fraction = 0.0
        custom_intensity_budget_fraction = 0.0
        if custom_ion_peak_mask is not None and len(custom_ion_peak_mask) == len(masked_peaks):
            custom_n_peaks = int(custom_ion_peak_mask.sum())
            custom_masked_arr = custom_ion_peak_mask & masked_peaks
            custom_n_masked = int(custom_masked_arr.sum())
            custom_mask_ratio = custom_n_masked / max(custom_n_peaks, 1)
            custom_mask_budget_fraction = custom_n_masked / max(total_masked, 1)
            custom_masked_int = float(valid_intensity[custom_masked_arr].sum())
            custom_intensity_budget_fraction = custom_masked_int / max(total_int_masked, 1e-12)

        n_ann = theoretical_analysis["n_annotated_peaks"]
        n_unann = theoretical_analysis["n_unannotated_peaks"]
        return {
            "n_annotated_masked": n_annotated_masked,
            "n_unannotated_masked": n_unannotated_masked,
            "n_annotated_unmasked": n_annotated_unmasked,
            "n_unannotated_unmasked": n_unannotated_unmasked,
            "annotated_mask_ratio": float(n_annotated_masked) / max(n_ann, 1),
            "unannotated_mask_ratio": float(n_unannotated_masked) / max(n_unann, 1),
            "overall_mask_ratio": float(total_masked) / max(len(masked_peaks), 1),
            "annotated_preservation_ratio": float(n_annotated_unmasked)
            / max(n_ann, 1),
            "annotated_fraction_of_masked_peaks": annotated_fraction_of_masked,
            "unannotated_fraction_of_masked_peaks": 1.0
            - annotated_fraction_of_masked,
            "annotated_intensity_fraction_of_masked": float(
                annotated_int_frac_masked
            ),
            "unannotated_intensity_fraction_of_masked": float(
                1.0 - annotated_int_frac_masked
            ),
            "annotated_intensity_masked_fraction_of_annotated": float(
                ann_int_masked_frac
            ),
            "annotated_intensity_unmasked_fraction_of_annotated": float(
                1.0 - ann_int_masked_frac
            ),
            "unannotated_intensity_masked_fraction_of_unannotated": float(
                unann_int_masked_frac
            ),
            "unannotated_intensity_unmasked_fraction_of_unannotated": float(
                1.0 - unann_int_masked_frac
            ),
            "annotated_type_total": annotated_type_total,
            "annotated_type_masked": annotated_type_masked,
            "annotated_type_mask_ratio": annotated_type_mask_ratio,
            "annotated_type_intensity_total": annotated_type_intensity_total,
            "annotated_type_intensity_masked": annotated_type_intensity_masked,
            "annotated_type_intensity_mask_ratio": annotated_type_intensity_mask_ratio,
            # Backward-compatible fragment group ratio metrics
            "fragment_group_count": int(fragment_group_count),
            "fragment_group_full_mask_ratio": float(fragment_group_full_mask_ratio),
            "fragment_group_avg_mask_fraction": float(
                fragment_group_avg_mask_fraction
            ),
            "fragment_group_weighted_mask_fraction": float(
                fragment_group_weighted_mask_fraction
            ),
            # Fragment group count breakdown (primary view: restricted to
            # the frag_type's canonical backbone ion pair)
            "n_fragment_groups_total": int(n_fragment_groups_total),
            "n_fragment_groups_fully_masked": int(n_fragment_groups_fully_masked),
            "n_fragment_groups_partially_masked": int(
                n_fragment_groups_partially_masked
            ),
            "n_fragment_groups_unmasked": int(n_fragment_groups_unmasked),
            "n_fragment_groups_any_masked": int(n_fragment_groups_any_masked),
            "group_mask_fractions": group_mask_fractions,
            # Metadata: which ion-series define the "primary" view. Lets
            # downstream figures caption their subtitle with the filter
            # actually applied.
            "primary_ion_series": list(primary_pair),
            # Diagnostic "all ion types" counterparts. These reflect
            # every non-precursor group the theoretical generator
            # produced for this frag_type (a/b/y for HCD, c/z for ETD,
            # etc.), so they match the denominator the previous
            # unrestricted view used. Useful for sanity-checking the
            # primary-view headline numbers.
            "n_fragment_groups_total_all": int(all_stats["count"]),
            "n_fragment_groups_fully_masked_all": int(all_stats["n_fully_masked"]),
            "n_fragment_groups_partially_masked_all": int(
                all_stats["n_partially_masked"]
            ),
            "n_fragment_groups_unmasked_all": int(all_stats["n_unmasked"]),
            "fragment_group_full_mask_ratio_all": float(all_stats["full_mask_ratio"]),
            "fragment_group_avg_mask_fraction_all": float(
                all_stats["avg_mask_fraction"]
            ),
            "fragment_group_weighted_mask_fraction_all": float(
                all_stats["weighted_mask_fraction"]
            ),
            # Per-ion-series group masking (from the *all* view, which is
            # the right scope for this breakdown — users reading this panel
            # want to see all series present, not just the primary pair).
            "series_group_total": series_group_total,
            "series_group_any_masked": series_group_any_masked,
            "series_group_fully_masked": series_group_fully_masked,
            "series_group_partially_masked": series_group_partially_masked,
            "series_group_unmasked": series_group_unmasked,
            # Precursor ion group metrics
            "precursor_n_peaks": int(precursor_n_peaks),
            "precursor_n_masked": int(precursor_n_masked),
            "precursor_base_masked": bool(precursor_base_masked),
            "precursor_group_mask_fraction": float(precursor_group_mask_fraction),
            "precursor_fully_masked": bool(precursor_fully_masked),
            "precursor_mask_budget_fraction": float(precursor_mask_budget_fraction),
            "precursor_intensity_budget_fraction": float(
                precursor_intensity_budget_fraction
            ),
            "precursor_fraction_of_annotated_masked": float(
                precursor_fraction_of_annotated_masked
            ),
            "precursor_intensity_fraction_of_annotated_masked": float(
                precursor_intensity_fraction_of_annotated_masked
            ),
            "precursor_has_leakage": bool(precursor_has_leakage),
            # Custom ion masking metrics
            "custom_n_peaks": int(custom_n_peaks),
            "custom_n_masked": int(custom_n_masked),
            "custom_mask_ratio": float(custom_mask_ratio),
            "custom_mask_budget_fraction": float(custom_mask_budget_fraction),
            "custom_intensity_budget_fraction": float(custom_intensity_budget_fraction),
        }

    # =========================================================================
    # Aggregation
    # =========================================================================

    def aggregate_results(
        self,
        per_spectrum_results: List[Dict[str, Any]],
        mass_error_data: Optional[List[Dict]] = None,
        unmatched_theo_data: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Aggregate theoretical analysis results across all spectra.

        Computes overall matching statistics, coverage, signal composition,
        and ion-type performance.
        """
        valid_results = [
            r for r in per_spectrum_results if r.get("sequence_available", False)
        ]
        if not valid_results:
            logger.warning("No valid theoretical results to aggregate")
            return {"error": "No data"}

        logger.info(
            f"Aggregating theoretical analysis for {len(valid_results):,d} spectra..."
        )

        # --- Per-spectrum summary arrays ---
        match_rates = [r.get("match_rate", 0) for r in valid_results]
        frac_intensities = [r.get("frac_intensity", 0) for r in valid_results]
        median_ppms = [r.get("median_ppm", np.nan) for r in valid_results]
        mean_ppms = [r.get("mean_ppm", np.nan) for r in valid_results]
        annotated_fracs = [r.get("annotated_fraction", 0) for r in valid_results]
        n_theoretical = [r.get("n_theoretical", 0) for r in valid_results]
        n_matched = [r.get("n_matched", 0) for r in valid_results]
        n_ann = [r.get("n_annotated_peaks", 0) for r in valid_results]
        n_unann = [r.get("n_unannotated_peaks", 0) for r in valid_results]

        overall_stats = {
            "n_spectra_analyzed": len(valid_results),
            "avg_match_rate": float(np.mean(match_rates)),
            "std_match_rate": float(np.std(match_rates)),
            "median_match_rate": float(np.median(match_rates)),
            "avg_frac_intensity": float(np.mean(frac_intensities)),
            "std_frac_intensity": float(np.std(frac_intensities)),
            "avg_median_ppm": float(np.nanmean(median_ppms)),
            "std_median_ppm": float(np.nanstd(median_ppms)),
            "avg_mean_ppm": float(np.nanmean(mean_ppms)),
            "std_mean_ppm": float(np.nanstd(mean_ppms)),
            "avg_annotated_fraction": float(np.mean(annotated_fracs)),
            "std_annotated_fraction": float(np.std(annotated_fracs)),
            "total_theoretical_ions": int(np.sum(n_theoretical)),
            "total_matched_ions": int(np.sum(n_matched)),
            "total_annotated_peaks": int(np.sum(n_ann)),
            "total_unannotated_peaks": int(np.sum(n_unann)),
        }

        # --- DataFrames ---
        mass_error_df = None
        if mass_error_data and len(mass_error_data) > 0:
            mass_error_df = pd.DataFrame(mass_error_data)
            logger.debug(
                f"Collected {len(mass_error_df):,d} matched peaks for mass error analysis"
            )

        unmatched_theo_df = None
        if unmatched_theo_data and len(unmatched_theo_data) > 0:
            unmatched_theo_df = pd.DataFrame(unmatched_theo_data)
            logger.debug(
                f"Collected {len(unmatched_theo_df):,d} unmatched theoretical ions"
            )

        self.results = {
            "overall_stats": overall_stats,
            "per_spectrum": per_spectrum_results,
            "mass_error_df": mass_error_df,
            "unmatched_theo_df": unmatched_theo_df,
        }

        # =================================================================
        # Phase A — full-population analyses (must precede filtering)
        # =================================================================
        # Fragment group analysis is run on the full sequence-available
        # population first because (a) the quality gate threshold is
        # defined in terms of its per-spectrum coverage / n_groups arrays
        # and (b) the gate's diagnostic outputs (rejection by frag_type /
        # charge / instrument / seq-len, modification-rejection-by-type)
        # need the full population to be meaningful.
        # The full-pop FGA result is cached as `_full_fragment_group_analysis`
        # and the visualization-facing `fragment_group_analysis` slot is
        # re-populated in Phase C with the gated-pop run, so plots reflect
        # what survived the gate.
        self.calculate_fragment_group_analysis()
        self.results["_full_fragment_group_analysis"] = self.results.get(
            "fragment_group_analysis", {}
        )

        if self.enable_quality_gate:
            self.analyze_quality_gate()

        # Modification quality-gate diagnostic (rejection rate per PTM
        # type) — must run on full population, before filtering removes
        # the rejected spectra. The remaining modification analyses
        # (prevalence, matching_quality, mass_error_by_mod, signal
        # composition) run later on the gated population.
        if self.enable_modification_analysis:
            self._compute_modification_quality_gate_diagnostic(per_spectrum_results)

        # =================================================================
        # Phase B — apply quality-gate filter (default ON)
        # =================================================================
        # Cache the unfiltered views so analyses that need both populations
        # (quality_gate_analysis itself was already run above) can still
        # reach them, and so we have an audit trail of what was removed.
        self.results["_full_per_spectrum"] = per_spectrum_results
        self.results["_full_mass_error_df"] = mass_error_df
        self.results["_full_unmatched_theo_df"] = unmatched_theo_df

        filter_metadata = self._apply_quality_gate_filter(
            per_spectrum_results, mass_error_df, unmatched_theo_df
        )
        self.results["filter_metadata"] = filter_metadata

        # Re-bind locals to whatever Phase B left in self.results so the
        # downstream conditionals see the gated views consistently.
        per_spectrum_results = self.results["per_spectrum"]
        mass_error_df = self.results["mass_error_df"]
        unmatched_theo_df = self.results["unmatched_theo_df"]

        # =================================================================
        # Phase C — gated-population analyses
        # =================================================================
        # All of the analyses below read self.results, which now points at
        # the gated views (when filtering is enabled). When filtering is
        # disabled they read the full population — same as before.

        # Re-run fragment group analysis on the gated subset so that the
        # visualization (fragment_group_analysis.png) and the per-frag_type
        # summary CSV reflect the population that survived the gate. The
        # full-pop result is preserved under _full_fragment_group_analysis
        # for any consumer that still wants the unfiltered view.
        self.calculate_fragment_group_analysis()

        if mass_error_df is not None and len(mass_error_df) > 0:
            self.calculate_coverage_statistics()
            self.calculate_signal_composition()
            self.analyze_ion_type_performance()
            self.calculate_mass_error_summary()

        # Custom ion analysis (independent of mass error data)
        if self.enable_custom_ions:
            self.calculate_custom_ion_analysis()

        # Neutral loss analysis (requires loss_details from conditional annotation)
        if self.add_losses:
            self.calculate_neutral_loss_analysis()

        # Blur σ calibration analysis (requires mass_error_df)
        if mass_error_df is not None and len(mass_error_df) > 0:
            self.calculate_blur_sigma_analysis()

        # Complementary b/y pair analysis
        if self.enable_complementary_pair_analysis and mass_error_df is not None:
            self.calculate_complementary_pair_analysis()

        # Mass gap validation against amino acid masses
        if self.enable_mass_gap_analysis and mass_error_df is not None:
            self.calculate_mass_gap_analysis()

        # PTM-stratified analysis (matching quality + per-PTM mass error,
        # on gated pop). The rejection-by-PTM diagnostic was already
        # computed in Phase A.
        if self.enable_modification_analysis:
            self.calculate_modification_analysis()

        # Stratified analysis (per frag_type) — also on gated pop.
        if self.enable_stratified_analysis:
            self._stratify_by_frag_type()

        logger.info("Theoretical analysis aggregation complete")
        return self.results

    # =========================================================================
    # Quality-gate filter (Phase B helpers)
    # =========================================================================

    def _apply_quality_gate_filter(
        self,
        per_spectrum_results: List[Dict[str, Any]],
        mass_error_df: Optional[pd.DataFrame],
        unmatched_theo_df: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        """Filter per_spectrum / mass_error_df / unmatched_theo_df by the gate.

        Reads the pass/fail mask produced by ``analyze_quality_gate`` (which
        is aligned to ``fga_valid``, the subset of per_spectrum with
        sequence_available + feature_types + theo_annotations). Maps that
        mask back to per_spectrum positions so we can also filter
        DataFrames whose ``spectrum_idx`` indexes per_spectrum directly.

        When the gate is disabled, the filter is disabled, or there is no
        gate output to read from, this is a no-op (the views in
        ``self.results`` already point at the full population).

        Returns a metadata dict describing the filter outcome.
        """
        n_total = len(per_spectrum_results)
        meta: Dict[str, Any] = {
            "gate_enabled": bool(self.enable_quality_gate),
            "filter_applied": False,
            "filter_requested": bool(self.apply_quality_gate_filter),
            "min_backbone_coverage": float(self.min_backbone_coverage),
            "min_fragment_groups": int(self.min_fragment_groups),
            "n_total_spectra": n_total,
            "n_passed_spectra": n_total,
            "n_rejected_spectra": 0,
            "rejection_rate": 0.0,
        }

        if not self.enable_quality_gate or not self.apply_quality_gate_filter:
            return meta

        qga = self.results.get("quality_gate_analysis", {})
        below_mask = qga.get("_below_mask")
        if below_mask is None:
            logger.warning(
                "apply_quality_gate_filter is True but quality_gate_analysis "
                "did not produce a below_mask; skipping filter"
            )
            return meta

        # below_mask is aligned to fga_valid (sequence_available + feature_types
        # + theo_annotations). Translate back to per_spectrum positions.
        fga_valid_positions = [
            i for i, r in enumerate(per_spectrum_results)
            if r.get("sequence_available", False)
            and r.get("feature_types")
            and r.get("theo_annotations")
        ]
        if len(fga_valid_positions) != len(below_mask):
            logger.warning(
                f"Quality-gate filter: fga_valid positions ({len(fga_valid_positions)}) "
                f"!= below_mask length ({len(below_mask)}); skipping filter"
            )
            return meta

        below_mask_arr = np.asarray(below_mask, dtype=bool)
        # Spectra absent from fga_valid never had a chance to pass — treat
        # them as rejected too. This matches the spirit of the gate: only
        # spectra with enough annotated structure to evaluate coverage can
        # be considered "passing".
        pass_indices: Set[int] = {
            pos for pos, fail in zip(fga_valid_positions, below_mask_arr) if not fail
        }

        n_passed = len(pass_indices)
        n_rejected = n_total - n_passed
        meta.update({
            "filter_applied": True,
            "n_passed_spectra": n_passed,
            "n_rejected_spectra": n_rejected,
            "rejection_rate": n_rejected / max(n_total, 1),
        })

        # Filter per_spectrum, preserving original ordering. spectrum_idx
        # values stamped on mass_error rows index INTO per_spectrum_results
        # (i.e. before filtering), so we filter the DataFrames using the
        # same per-spectrum index set rather than re-numbering rows.
        filtered_per_spectrum = [
            per_spectrum_results[i] for i in range(n_total) if i in pass_indices
        ]

        filtered_mass_error_df = mass_error_df
        if mass_error_df is not None and "spectrum_idx" in mass_error_df.columns:
            row_mask = mass_error_df["spectrum_idx"].isin(pass_indices)
            filtered_mass_error_df = mass_error_df.loc[row_mask].copy()

        filtered_unmatched_theo_df = unmatched_theo_df
        if (
            unmatched_theo_df is not None
            and "spectrum_idx" in unmatched_theo_df.columns
        ):
            row_mask = unmatched_theo_df["spectrum_idx"].isin(pass_indices)
            filtered_unmatched_theo_df = unmatched_theo_df.loc[row_mask].copy()

        # Replace the views in self.results so downstream analyses see the
        # gated population.
        self.results["per_spectrum"] = filtered_per_spectrum
        self.results["mass_error_df"] = filtered_mass_error_df
        self.results["unmatched_theo_df"] = filtered_unmatched_theo_df

        logger.info(
            f"Quality-gate filter applied: {n_passed:,d}/{n_total:,d} spectra "
            f"passed ({(1 - meta['rejection_rate']) * 100:.1f}%); "
            f"{n_rejected:,d} rejected"
        )
        return meta

    def _compute_modification_quality_gate_diagnostic(
        self,
        full_per_spectrum: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Compute rejection rate per PTM type on the full population.

        Stored under ``self.results["modification_quality_gate"]`` so
        ``calculate_modification_analysis`` (which now runs on the gated
        population) does not have to maintain its own copy of this
        full-population diagnostic.
        """
        # Read from the cached full-pop FGA so this method keeps working
        # if the visible `fragment_group_analysis` slot is later replaced
        # by the gated-pop run in Phase C.
        fga = self.results.get(
            "_full_fragment_group_analysis",
            self.results.get("fragment_group_analysis", {}),
        )
        coverage_arr = np.asarray(fga.get("_per_spectrum_coverage", []))
        n_groups_arr = np.asarray(fga.get("_per_spectrum_n_groups", []))
        if coverage_arr.size == 0:
            return {}

        fga_valid = [
            r for r in full_per_spectrum
            if r.get("sequence_available", False)
            and r.get("feature_types")
            and r.get("theo_annotations")
        ]
        if len(fga_valid) != len(coverage_arr):
            return {}

        below_mask = (
            (coverage_arr < self.min_backbone_coverage)
            | (n_groups_arr < self.min_fragment_groups)
        )
        overall_rej_rate = float(below_mask.sum()) / max(len(below_mask), 1)

        fga_mod_types: List[List[str]] = []
        for r in fga_valid:
            seq = r.get("_metadata", {}).get("sequence") or r.get("clean_sequence", "")
            fga_mod_types.append(extract_modification_types(seq) if seq else [])

        type_indices: Dict[str, List[int]] = {"Unmodified": []}
        for i, mtypes in enumerate(fga_mod_types):
            if not mtypes:
                type_indices["Unmodified"].append(i)
            else:
                for mt in mtypes:
                    type_indices.setdefault(mt, []).append(i)

        by_mod: Dict[str, Dict[str, Any]] = {}
        for mod_type, indices in type_indices.items():
            if len(indices) < 2:
                continue
            idx_arr = np.array(indices)
            n_rej = int(below_mask[idx_arr].sum())
            rej_rate = n_rej / max(len(indices), 1)
            by_mod[mod_type] = {
                "n_total": len(indices),
                "n_rejected": n_rej,
                "rejection_rate": rej_rate,
                "relative_risk": rej_rate / max(overall_rej_rate, 1e-9),
            }

        result = {
            "overall_rejection_rate": overall_rej_rate,
            "by_modification": by_mod,
        }
        self.results["modification_quality_gate"] = result
        return result

    # =========================================================================
    # Modification (PTM) Stratified Analysis
    # =========================================================================

    def calculate_modification_analysis(self) -> Dict[str, Any]:
        """Analyse theoretical matching quality stratified by modification type.

        Produces five analysis dimensions:

        A. **Prevalence** — how common each modification type is.
        B. **Per-modification matching quality** — match rate, intensity
           coverage, median PPM, annotated fraction per type.
        C. **Modification-aware mass error** — systematic bias per type.
        D. **Signal composition by modification** — annotated vs unannotated.
        E. **Quality gate rejection by modification** — whether modified
           spectra are disproportionately rejected.

        Results stored in ``self.results["modification_analysis"]``.
        """
        per_spectrum = self.results.get("per_spectrum", [])
        valid = [r for r in per_spectrum if r.get("sequence_available", False)]
        if not valid:
            logger.debug("No valid spectra for modification analysis")
            return {}

        # ── A. Prevalence ──────────────────────────────────────────────
        type_counts: Counter = Counter()
        mod_label_per_spectrum: List[str] = []
        mod_types_per_spectrum: List[List[str]] = []
        n_modified = 0
        n_unmodified = 0

        for r in valid:
            seq = r.get("_metadata", {}).get("sequence") or r.get("clean_sequence", "")
            mod_types = extract_modification_types(seq) if seq else []
            mod_types_per_spectrum.append(mod_types)
            if mod_types:
                n_modified += 1
                for mt in mod_types:
                    type_counts[mt] += 1
                mod_label_per_spectrum.append(" + ".join(sorted(mod_types)))
            else:
                n_unmodified += 1
                mod_label_per_spectrum.append("Unmodified")

        n_total = len(valid)
        prevalence = {
            "n_total": n_total,
            "n_modified": n_modified,
            "n_unmodified": n_unmodified,
            "modified_fraction": n_modified / max(n_total, 1),
            "per_type": {
                mt: {
                    "n_spectra": cnt,
                    "spectra_fraction": cnt / max(n_total, 1),
                }
                for mt, cnt in type_counts.most_common()
            },
        }

        # ── B. Per-modification matching quality ───────────────────────
        # Group spectra indices by type (+ Unmodified baseline)
        type_to_indices: Dict[str, List[int]] = {"Unmodified": []}
        for i, (mtypes, _) in enumerate(
            zip(mod_types_per_spectrum, valid)
        ):
            if not mtypes:
                type_to_indices["Unmodified"].append(i)
            else:
                for mt in mtypes:
                    type_to_indices.setdefault(mt, []).append(i)

        metrics_keys = ["match_rate", "frac_intensity", "median_ppm", "annotated_fraction"]
        matching_quality: Dict[str, Dict[str, Dict[str, float]]] = {}
        # Internal arrays for visualization
        _match_rate_by_type: Dict[str, List[float]] = {}
        _frac_intensity_by_type: Dict[str, List[float]] = {}
        _ppm_by_type: Dict[str, List[float]] = {}

        for mod_type, indices in type_to_indices.items():
            if len(indices) < 2:
                continue
            vals: Dict[str, List[float]] = {k: [] for k in metrics_keys}
            for idx in indices:
                r = valid[idx]
                for k in metrics_keys:
                    v = r.get(k, np.nan)
                    if not (isinstance(v, float) and np.isnan(v)):
                        vals[k].append(float(v))

            stats: Dict[str, Dict[str, float]] = {}
            for k in metrics_keys:
                arr = np.array(vals[k]) if vals[k] else np.array([])
                if len(arr) > 0:
                    stats[k] = {
                        "mean": float(np.mean(arr)),
                        "median": float(np.median(arr)),
                        "std": float(np.std(arr)),
                        "q25": float(np.percentile(arr, 25)),
                        "q75": float(np.percentile(arr, 75)),
                    }
                else:
                    stats[k] = {
                        "mean": 0.0, "median": 0.0, "std": 0.0,
                        "q25": 0.0, "q75": 0.0,
                    }
            matching_quality[mod_type] = stats

            _match_rate_by_type[mod_type] = vals["match_rate"]
            _frac_intensity_by_type[mod_type] = vals["frac_intensity"]
            _ppm_by_type[mod_type] = [
                v for v in vals["median_ppm"] if not np.isnan(v)
            ]

        # ── C. Modification-aware mass error ───────────────────────────
        mass_error_by_mod: Dict[str, Dict[str, float]] = {}
        mass_error_df = self.results.get("mass_error_df")
        if mass_error_df is not None and len(mass_error_df) > 0 and "peptide" in mass_error_df.columns:
            for mod_type, indices in type_to_indices.items():
                if len(indices) < 2:
                    continue
                # Get sequences for these indices
                seqs = set()
                for idx in indices:
                    r = valid[idx]
                    seq = r.get("_metadata", {}).get("sequence") or r.get("clean_sequence", "")
                    if seq:
                        seqs.add(seq)
                if not seqs:
                    continue
                # Filter mass_error_df rows matching these sequences
                mask = mass_error_df["peptide"].isin(seqs)
                subset = mass_error_df.loc[mask, "signed_ppm"]
                if len(subset) < 2:
                    continue
                arr = subset.values
                mass_error_by_mod[mod_type] = {
                    "mean_signed_ppm": float(np.mean(arr)),
                    "median_signed_ppm": float(np.median(arr)),
                    "std_signed_ppm": float(np.std(arr)),
                    "mean_abs_ppm": float(np.mean(np.abs(arr))),
                    "median_abs_ppm": float(np.median(np.abs(arr))),
                }

        # ── D. Signal composition by modification ─────────────────────
        signal_composition: Dict[str, Dict[str, float]] = {}
        for mod_type, indices in type_to_indices.items():
            if len(indices) < 2:
                continue
            ann_fracs = []
            unann_fracs = []
            for idx in indices:
                r = valid[idx]
                ann_fracs.append(r.get("annotated_fraction", 0.0))
                unann_fracs.append(r.get("unannotated_fraction", 1.0))
            signal_composition[mod_type] = {
                "mean_annotated_fraction": float(np.mean(ann_fracs)),
                "mean_unannotated_fraction": float(np.mean(unann_fracs)),
            }

        # ── E. Quality gate rejection by modification ──────────────────
        # Pre-computed on the FULL (pre-filter) population by
        # _compute_modification_quality_gate_diagnostic. We only surface
        # it here for backward-compatible JSON output.
        mod_qg = self.results.get("modification_quality_gate", {})
        quality_gate_by_mod: Dict[str, Dict[str, Any]] = dict(
            mod_qg.get("by_modification", {})
        )

        # ── Diagnostic counters ────────────────────────────────────────
        n_modifications_stripped = sum(
            1 for r in valid if r.get("modifications_stripped", False)
        )
        n_precursor_mass_warnings = sum(
            1 for r in valid if r.get("precursor_mass_warning", False)
        )
        n_isotope_corrected = sum(
            1 for r in valid if r.get("precursor_isotope_offset", 0) > 0
        )

        result = {
            "prevalence": prevalence,
            "matching_quality": matching_quality,
            "mass_error_by_modification": mass_error_by_mod,
            "signal_composition": signal_composition,
            "quality_gate_by_modification": quality_gate_by_mod,
            "diagnostics": {
                "n_modifications_stripped": n_modifications_stripped,
                "n_precursor_mass_warnings": n_precursor_mass_warnings,
                "n_isotope_corrected": n_isotope_corrected,
            },
            # Internal arrays for visualization (prefixed with _)
            "_match_rate_by_type": _match_rate_by_type,
            "_frac_intensity_by_type": _frac_intensity_by_type,
            "_ppm_by_type": _ppm_by_type,
            "_mod_label_per_spectrum": mod_label_per_spectrum,
        }
        self.results["modification_analysis"] = result
        logger.info(
            f"Modification analysis: {n_modified:,d}/{n_total:,d} modified "
            f"({len(type_counts)} unique types)"
        )
        return result

    def visualize_modification_analysis(self) -> Figure:
        """Generate 2x3 panel figure for modification analysis.

        Panels:
          A — Prevalence bar chart (top 15 types)
          B — Match rate box plots per type
          C — Intensity coverage box plots per type
          D — Mass error bias per type (bar + error bars)
          E — Quality gate rejection rate per type
          F — Signal composition (annotated vs unannotated) per type
        """
        mod = self.results.get("modification_analysis", {})
        if not mod:
            logger.debug("No modification analysis to visualize")
            return plt.figure()

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            "Modification (PTM) Stratified Analysis",
            fontsize=16,
            fontweight="bold",
        )

        prevalence = mod.get("prevalence", {})
        per_type = prevalence.get("per_type", {})
        matching_quality = mod.get("matching_quality", {})
        mass_error = mod.get("mass_error_by_modification", {})
        qg_by_mod = mod.get("quality_gate_by_modification", {})
        signal_comp = mod.get("signal_composition", {})
        _match_rate_by_type = mod.get("_match_rate_by_type", {})
        _frac_intensity_by_type = mod.get("_frac_intensity_by_type", {})

        # Top types for box/bar plots (top 8 + Unmodified)
        sorted_types = sorted(
            per_type.keys(), key=lambda t: per_type[t]["n_spectra"], reverse=True
        )
        box_types = ["Unmodified"] + [t for t in sorted_types if t != "Unmodified"][:8]
        box_types = [t for t in box_types if t in matching_quality]

        # ── Panel A: Prevalence (top 15) ───────────────────────────────
        ax = axes[0, 0]
        top_15 = sorted_types[:15]
        if top_15:
            counts = [per_type[t]["n_spectra"] for t in top_15]
            y_pos = np.arange(len(top_15))
            ax.barh(y_pos, counts, color="#1f77b4", edgecolor="white")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_15, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Spectra count")
            ax.set_title("A. Modification Prevalence (top 15)")
        else:
            ax.text(0.5, 0.5, "No modifications found", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("A. Modification Prevalence")

        # ── Panel B: Match rate box plots ──────────────────────────────
        ax = axes[0, 1]
        if box_types and _match_rate_by_type:
            data_b = [
                _match_rate_by_type.get(t, [])
                for t in box_types
            ]
            bp = ax.boxplot(
                data_b,
                labels=[t[:15] for t in box_types],
                patch_artist=True,
                showfliers=False,
            )
            colors = ["#2ca02c" if t == "Unmodified" else "#1f77b4" for t in box_types]
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.6)
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            ax.set_ylabel("Match rate")
            # Autoscale from the actual IQR ranges. The hard [0, 1.05]
            # limit previously crushed the box bodies because match_rate
            # values exceed 1.0 (the conditional-match numerator counts
            # loss / isotope features in addition to base ions, while the
            # denominator here is the cached base-ion theoretical count —
            # a pre-existing denominator mismatch we noted in the review).
            # We set ymin = 0 and ymax = max IQR upper hinge + 10 % so
            # both the body of the distribution and any >1 values are
            # visible at a useful scale.
            all_vals = [v for series in data_b for v in series if np.isfinite(v)]
            if all_vals:
                ymax_data = float(np.percentile(all_vals, 99))
                ymax = max(1.05, ymax_data * 1.05)
                ax.set_ylim(0, ymax)
            # Reference line at 1.0 so the reader can see when a type's
            # values exceed the nominal 100 % ceiling.
            ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)
            ax.set_title("B. Match Rate by Type")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("B. Match Rate by Type")

        # ── Panel C: Intensity coverage box plots ──────────────────────
        ax = axes[0, 2]
        if box_types and _frac_intensity_by_type:
            data_c = [
                _frac_intensity_by_type.get(t, [])
                for t in box_types
            ]
            bp = ax.boxplot(
                data_c,
                labels=[t[:15] for t in box_types],
                patch_artist=True,
                showfliers=False,
            )
            colors = ["#2ca02c" if t == "Unmodified" else "#d62728" for t in box_types]
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.6)
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            ax.set_ylabel("Intensity coverage")
            ax.set_ylim(0, 1.05)
            ax.set_title("C. Intensity Coverage by Type")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("C. Intensity Coverage by Type")

        # ── Panel D: Mass error bias ───────────────────────────────────
        ax = axes[1, 0]
        if mass_error:
            me_types = [t for t in box_types if t in mass_error]
            if me_types:
                means = [mass_error[t]["mean_signed_ppm"] for t in me_types]
                stds = [mass_error[t]["std_signed_ppm"] for t in me_types]
                x_pos = np.arange(len(me_types))
                ax.bar(x_pos, means, yerr=stds, capsize=3, color="#ff7f0e",
                       edgecolor="white", alpha=0.8)
                ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
                ax.set_xticks(x_pos)
                ax.set_xticklabels([t[:15] for t in me_types], rotation=45,
                                   fontsize=7, ha="right")
                ax.set_ylabel("Mean signed PPM")
                ax.set_title("D. Mass Error Bias by Type")
            else:
                ax.text(0.5, 0.5, "No mass error data", ha="center",
                        va="center", transform=ax.transAxes)
                ax.set_title("D. Mass Error Bias by Type")
        else:
            ax.text(0.5, 0.5, "No mass error data", ha="center",
                    va="center", transform=ax.transAxes)
            ax.set_title("D. Mass Error Bias by Type")

        # ── Panel E: Quality gate rejection ────────────────────────────
        ax = axes[1, 1]
        if qg_by_mod and box_types:
            # Show all box_types for consistency; types not in qg_by_mod
            # get rejection_rate = 0 (too few spectra to compute).
            rej_rates = [
                qg_by_mod[t]["rejection_rate"] if t in qg_by_mod else 0.0
                for t in box_types
            ]
            x_pos = np.arange(len(box_types))
            ax.bar(x_pos, rej_rates, color="#d62728", edgecolor="white",
                   alpha=0.8)
            # Overall rejection rate as reference
            qga = self.results.get("quality_gate_analysis", {})
            overall_rej = qga.get("overall_rejection_rate", 0)
            if overall_rej > 0:
                ax.axhline(overall_rej, color="black", linewidth=1.2,
                           linestyle="--", label=f"Overall ({overall_rej*100:.1f}%)")
                ax.legend(fontsize=8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([t[:15] for t in box_types], rotation=45,
                               fontsize=7, ha="right")
            ax.set_ylabel("Rejection rate")
            ax.set_title("E. Quality Gate Rejection by Type")
        else:
            ax.text(0.5, 0.5, "No quality gate data", ha="center",
                    va="center", transform=ax.transAxes)
            ax.set_title("E. Quality Gate Rejection by Type")

        # ── Panel F: Signal composition ────────────────────────────────
        ax = axes[1, 2]
        if signal_comp:
            sc_types = [t for t in box_types if t in signal_comp]
            if sc_types:
                ann_vals = [signal_comp[t]["mean_annotated_fraction"] for t in sc_types]
                unann_vals = [signal_comp[t]["mean_unannotated_fraction"] for t in sc_types]
                x_pos = np.arange(len(sc_types))
                ax.bar(x_pos, ann_vals, label="Annotated", color="#2ca02c",
                       edgecolor="white")
                ax.bar(x_pos, unann_vals, bottom=ann_vals, label="Unannotated",
                       color="#d62728", edgecolor="white")
                ax.set_xticks(x_pos)
                ax.set_xticklabels([t[:15] for t in sc_types], rotation=45,
                                   fontsize=7, ha="right")
                ax.set_ylabel("Fraction")
                ax.set_title("F. Signal Composition by Type")
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, "No signal data", ha="center",
                        va="center", transform=ax.transAxes)
                ax.set_title("F. Signal Composition by Type")
        else:
            ax.text(0.5, 0.5, "No signal data", ha="center",
                    va="center", transform=ax.transAxes)
            ax.set_title("F. Signal Composition by Type")

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        output_path = self.output_dir / "modification_analysis.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"Saved modification analysis: {output_path}")
        return fig

    def _stratify_by_frag_type(self) -> None:
        """Create per-frag_type sub-analysers for stratified visualization.

        Each sub-analyser gets a filtered view of per_spectrum, mass_error_df,
        and unmatched_theo_df, then runs the core aggregation methods.  The
        sub-analysers are stored under ``_stratified_analysers`` (underscore
        prefix = skipped by ``save_results``).
        """
        per_spectrum = self.results.get("per_spectrum", [])
        valid = [r for r in per_spectrum if r.get("sequence_available", False)]
        if not valid:
            return

        # Determine unique frag_types
        frag_types: Dict[str, List[int]] = {}
        for i, r in enumerate(valid):
            ft = str(r.get("_metadata", {}).get("frag_type", "unknown"))
            frag_types.setdefault(ft, []).append(i)

        if len(frag_types) <= 1:
            logger.debug("Single frag_type -- skipping stratified analysis")
            return

        mass_error_df = self.results.get("mass_error_df")
        unmatched_theo_df = self.results.get("unmatched_theo_df")

        stratified: Dict[str, "TheoreticalAnalyser"] = {}
        for ft, indices in frag_types.items():
            if len(indices) < 10:
                continue

            sub_dir = self.output_dir / f"by_frag_type/{ft}"
            sub = TheoreticalAnalyser(self.config, output_dir=sub_dir)

            # Filter per-spectrum results
            sub_ps = [valid[i] for i in indices]

            # Filter DataFrames by frag_type column (preferred) or spectrum_idx
            sub_me_df = None
            sub_unmatched_df = None
            if mass_error_df is not None:
                if "frag_type" in mass_error_df.columns:
                    sub_me_df = mass_error_df[
                        mass_error_df["frag_type"] == ft
                    ].copy()
                elif "spectrum_idx" in mass_error_df.columns:
                    sub_me_df = mass_error_df[
                        mass_error_df["spectrum_idx"].isin(set(indices))
                    ].copy()
                if sub_me_df is not None and len(sub_me_df) == 0:
                    sub_me_df = None
            if unmatched_theo_df is not None:
                if "frag_type" in unmatched_theo_df.columns:
                    sub_unmatched_df = unmatched_theo_df[
                        unmatched_theo_df["frag_type"] == ft
                    ].copy()
                elif "spectrum_idx" in unmatched_theo_df.columns:
                    sub_unmatched_df = unmatched_theo_df[
                        unmatched_theo_df["spectrum_idx"].isin(set(indices))
                    ].copy()
                if sub_unmatched_df is not None and len(sub_unmatched_df) == 0:
                    sub_unmatched_df = None

            # Populate results directly (no recursive aggregation)
            sub.results = {
                "per_spectrum": sub_ps,
                "mass_error_df": sub_me_df,
                "unmatched_theo_df": sub_unmatched_df,
            }

            # Run core aggregation methods
            if sub_me_df is not None and len(sub_me_df) > 0:
                sub.calculate_coverage_statistics()
                sub.calculate_signal_composition()
                sub.analyze_ion_type_performance()
                sub.calculate_mass_error_summary()

            stratified[ft] = sub
            logger.debug(
                f"Stratified: {ft} -- {len(sub_ps)} spectra"
            )

        self.results["_stratified_analysers"] = stratified

    def calculate_coverage_statistics(self) -> Dict[str, Any]:
        """Calculate theoretical ion coverage by m/z range and ion type."""
        mass_error_df = self.results.get("mass_error_df")
        unmatched_theo_df = self.results.get("unmatched_theo_df")
        if mass_error_df is None:
            return {}

        # By m/z range
        coverage_by_mz_range = {}
        for mz_range in self.mz_range_order:
            n_matched = len(mass_error_df[mass_error_df["mz_range"] == mz_range])
            n_unmatched = 0
            if unmatched_theo_df is not None:
                n_unmatched = len(
                    unmatched_theo_df[unmatched_theo_df["mz_range"] == mz_range]
                )
            n_total = n_matched + n_unmatched
            coverage_by_mz_range[mz_range] = {
                "n_theoretical": n_total,
                "n_matched": n_matched,
                "n_unmatched": n_unmatched,
                "coverage_rate": float(n_matched / max(n_total, 1)),
            }

        # By ion type
        coverage_by_ion_type = {}
        if "ion_type" in mass_error_df.columns:
            for ion_type in mass_error_df["ion_type"].unique():
                if not ion_type or ion_type == "unknown":
                    continue
                n_matched = len(mass_error_df[mass_error_df["ion_type"] == ion_type])
                n_unmatched = 0
                if (
                    unmatched_theo_df is not None
                    and "ion_type" in unmatched_theo_df.columns
                ):
                    n_unmatched = len(
                        unmatched_theo_df[unmatched_theo_df["ion_type"] == ion_type]
                    )
                n_total = n_matched + n_unmatched
                coverage_by_ion_type[ion_type] = {
                    "n_theoretical": n_total,
                    "n_matched": n_matched,
                    "n_unmatched": n_unmatched,
                    "coverage_rate": float(n_matched / max(n_total, 1)),
                }

        total_matched = len(mass_error_df)
        total_unmatched = len(unmatched_theo_df) if unmatched_theo_df is not None else 0
        total_theoretical = total_matched + total_unmatched

        coverage_stats = {
            "overall_coverage_rate": float(
                total_matched / max(total_theoretical, 1)
            ),
            "total_theoretical_ions": total_theoretical,
            "total_matched": total_matched,
            "total_unmatched": total_unmatched,
            "coverage_by_mz_range": coverage_by_mz_range,
            "coverage_by_ion_type": coverage_by_ion_type,
        }

        self.results["coverage_stats"] = coverage_stats
        logger.debug(
            f"Coverage: {coverage_stats['overall_coverage_rate']*100:.1f}% "
            f"({total_matched:,d}/{total_theoretical:,d})"
        )
        return coverage_stats

    def calculate_signal_composition(self) -> Dict[str, Any]:
        """Calculate training signal composition — peak types by count and intensity.

        This analysis answers a fundamental question for the foundation model:
        *What is the model learning to reconstruct?*  A spectrum's peaks consist
        of informative fragment ions (b/y, losses, isotopes) and noise
        (unannotated peaks).  Understanding this composition informs masking
        strategy design and loss weighting.
        """
        per_spectrum = self.results.get("per_spectrum", [])
        valid_results = [
            r for r in per_spectrum if r.get("sequence_available", False)
        ]
        if not valid_results:
            return {}

        # Aggregate label counts across all spectra
        total_by_label: Counter = Counter()
        for r in valid_results:
            counts = r.get("annotation_label_counts", {})
            for label, count in counts.items():
                total_by_label[label] += count

        grand_total = sum(total_by_label.values())
        if grand_total == 0:
            return {}

        # Count-based composition
        count_composition = {
            label: {
                "count": count,
                "fraction": count / grand_total,
            }
            for label, count in total_by_label.most_common()
        }

        # Group into high-level categories. Every ion-series listed under
        # fragment_base must have matching -loss/-isotope entries here; x/z
        # arise from ETD/UVPD data and their loss/isotope labels were
        # previously silently dropped into "other".
        category_map = {
            "fragment_base": ["b-ion", "y-ion", "a-ion", "c-ion", "x-ion", "z-ion"],
            "fragment_loss": [
                "b-loss", "y-loss", "a-loss", "c-loss", "x-loss", "z-loss",
            ],
            "fragment_isotope": [
                "b-isotope", "y-isotope", "a-isotope", "c-isotope",
                "x-isotope", "z-isotope",
            ],
            "precursor": ["precursor", "precursor-isotope"],
            "other_annotated": ["other"],
            "unannotated": ["unannotated"],
        }
        category_counts: Dict[str, int] = {}
        for category, labels in category_map.items():
            category_counts[category] = sum(
                total_by_label.get(label, 0) for label in labels
            )
        category_fractions = {
            cat: count / grand_total for cat, count in category_counts.items()
        }

        # Informative vs noise summary
        informative_count = sum(
            v
            for k, v in category_counts.items()
            if k not in ("unannotated", "other_annotated")
        )

        # --- Intensity-weighted composition (Michalski 2012 insight) ---
        total_intensity_by_label: Dict[str, float] = {}
        annotated_intensities: List[float] = []
        unannotated_intensities: List[float] = []
        for r in valid_results:
            intensities = r.get("annotation_label_intensities", {})
            for label, intensity in intensities.items():
                total_intensity_by_label[label] = (
                    total_intensity_by_label.get(label, 0.0) + intensity
                )
            annotated_intensities.extend(
                r.get("_annotated_intensity_values", [])
            )
            unannotated_intensities.extend(
                r.get("_unannotated_intensity_values", [])
            )

        grand_total_intensity = sum(total_intensity_by_label.values())
        if grand_total_intensity > 0:
            intensity_composition = {
                label: {
                    "intensity": total_intensity_by_label.get(label, 0.0),
                    "fraction": total_intensity_by_label.get(label, 0.0)
                    / grand_total_intensity,
                }
                for label in total_by_label
            }
            category_intensity: Dict[str, float] = {}
            for category, labels in category_map.items():
                category_intensity[category] = sum(
                    total_intensity_by_label.get(label, 0.0) for label in labels
                )
            category_intensity_fractions = {
                cat: val / grand_total_intensity
                for cat, val in category_intensity.items()
            }
            informative_intensity = sum(
                v
                for k, v in category_intensity.items()
                if k not in ("unannotated", "other_annotated")
            )
        else:
            intensity_composition = {}
            category_intensity_fractions = {}
            informative_intensity = 0.0

        signal_composition = {
            "total_peaks": grand_total,
            "count_composition": count_composition,
            "category_counts": category_counts,
            "category_fractions": category_fractions,
            "informative_fraction": informative_count / grand_total,
            "noise_fraction": category_counts.get("unannotated", 0) / grand_total,
            "intensity_composition": intensity_composition,
            "category_intensity_fractions": category_intensity_fractions,
            "informative_intensity_fraction": informative_intensity
            / max(grand_total_intensity, 1e-12),
            "noise_intensity_fraction": category_intensity.get("unannotated", 0.0)
            / max(grand_total_intensity, 1e-12)
            if grand_total_intensity > 0
            else 0.0,
            "_annotated_intensities": annotated_intensities,
            "_unannotated_intensities": unannotated_intensities,
        }

        self.results["signal_composition"] = signal_composition
        logger.debug(
            f"Signal composition: {signal_composition['informative_fraction']*100:.1f}% informative, "
            f"{signal_composition['noise_fraction']*100:.1f}% unannotated"
        )
        return signal_composition

    def analyze_ion_type_performance(self) -> Dict[str, Any]:
        """Analyze per-ion-type matching performance (mass accuracy, counts)."""
        mass_error_df = self.results.get("mass_error_df")
        if mass_error_df is None or "ion_type" not in mass_error_df.columns:
            return {}

        ion_type_stats = {}
        for ion_type in mass_error_df["ion_type"].unique():
            if not ion_type or ion_type == "unknown":
                continue

            ion_df = mass_error_df[mass_error_df["ion_type"] == ion_type]
            ppm_errors = ion_df["delta_mz_ppm"].values

            charge_distribution = {}
            if "charge" in ion_df.columns:
                charge_counts = ion_df["charge"].value_counts().to_dict()
                charge_distribution = {
                    int(k): int(v)
                    for k, v in charge_counts.items()
                    if not pd.isna(k)
                }

            ion_type_stats[ion_type] = {
                "n_matched": len(ion_df),
                "mean_ppm_error": float(np.mean(ppm_errors)),
                "std_ppm_error": float(np.std(ppm_errors)),
                "median_abs_ppm_error": float(np.median(np.abs(ppm_errors))),
                "charge_distribution": charge_distribution,
            }

        self.results["ion_type_performance"] = ion_type_stats
        logger.debug(f"Ion type performance: {len(ion_type_stats)} types analyzed")
        return ion_type_stats

    def calculate_mass_error_summary(self) -> Dict[str, Any]:
        """Calculate mass error summary with systematic bias detection.

        Folds in the essential metrics from the former
        ``calculate_annotation_quality_metrics`` method: systematic bias,
        error percentiles, and multiple-assignment rate.
        """
        mass_error_df = self.results.get("mass_error_df")
        if mass_error_df is None or len(mass_error_df) == 0:
            return {}

        signed_ppm = mass_error_df["signed_ppm"].values
        abs_ppm = mass_error_df["delta_mz_ppm"].values

        # Systematic bias
        systematic_bias_ppm = float(np.mean(signed_ppm))
        systematic_bias_std = float(np.std(signed_ppm))

        try:
            from scipy import stats as scipy_stats

            _, p_value = scipy_stats.ttest_1samp(signed_ppm, 0)
            p_value = float(p_value)
            bias_significant = p_value < 0.05
        except ImportError:
            p_value = float("nan")
            bias_significant = False

        # Percentiles
        percentiles = {
            "p50": float(np.percentile(abs_ppm, 50)),
            "p90": float(np.percentile(abs_ppm, 90)),
            "p95": float(np.percentile(abs_ppm, 95)),
            "p99": float(np.percentile(abs_ppm, 99)),
        }

        # Multiple assignment rate (same theo ion matched to >1 exp peak)
        multiple_assignment_rate = float("nan")
        n_multiple = 0
        if "theo_mz" in mass_error_df.columns:
            theo_mz_counts = mass_error_df["theo_mz"].value_counts()
            n_multiple = int((theo_mz_counts > 1).sum())
            multiple_assignment_rate = n_multiple / max(len(theo_mz_counts), 1)

        mass_error_summary = {
            "n_matched_peaks": len(mass_error_df),
            "systematic_bias_ppm": systematic_bias_ppm,
            "systematic_bias_std_ppm": systematic_bias_std,
            "systematic_bias_significant": bias_significant,
            "systematic_bias_p_value": p_value,
            "percentiles": percentiles,
            "multiple_assignment_rate": float(multiple_assignment_rate),
            "n_multiple_assignments": n_multiple,
        }

        self.results["mass_error_summary"] = mass_error_summary
        logger.debug(
            f"Mass error: bias={systematic_bias_ppm:.2f} ppm "
            f"(p={p_value:.3f}), P50={percentiles['p50']:.2f} ppm"
        )
        return mass_error_summary

    # =========================================================================
    # Custom Ion Analysis (Aggregation)
    # =========================================================================

    def calculate_custom_ion_analysis(self) -> Dict[str, Any]:
        """Aggregate custom ion detection data across all analysed spectra.

        Produces five categories of insight:

        1. **Prevalence** — hit rate per ion group and per high-level category
           (glycan, immonium, TMT, iTRAQ).
        2. **Intensity profile** — how prominent are custom ions when present?
           (max matched intensity per detection, grouped by category).
        3. **Mass accuracy** — PPM error of custom-ion matches, validating the
           m/z values in the ion library.
        4. **Co-occurrence** — which ion categories appear together across
           spectra?  Reveals dataset composition (e.g. TMT-labelled fraction,
           glycoproteomics fraction).
        5. **Coverage lift** — how many previously-unannotated peaks become
           explained when custom-ion detection is added to fragment matching,
           broken down by m/z range.
        """
        per_spectrum = self.results.get("per_spectrum", [])
        valid = [r for r in per_spectrum if "custom_ion_data" in r]
        if not valid:
            logger.debug("No custom ion data to aggregate")
            return {}

        ion_library = self.custom_ions or DEFAULT_CUSTOM_IONS
        categories = ["glycan", "immonium", "TMT", "iTRAQ"]
        n_spectra = len(valid)
        logger.debug(
            f"Aggregating custom ion data from {n_spectra:,d} spectra"
        )

        # Accumulators
        group_hit_counts: Dict[str, int] = {g: 0 for g in ion_library}
        category_hit_counts: Dict[str, int] = {c: 0 for c in categories}
        category_intensities: Dict[str, List[float]] = {c: [] for c in categories}
        ppm_errors_all: List[float] = []
        ppm_errors_by_cat: Dict[str, List[float]] = {c: [] for c in categories}
        per_spectrum_categories: List[set] = []
        # Per-m/z-range coverage accumulators
        range_totals: Dict[str, Dict[str, int]] = {
            rn: {"n_peaks": 0, "n_fragment_annotated": 0, "n_custom_only": 0}
            for rn in self.mz_range_order
        }

        for r in valid:
            cid = r["custom_ion_data"]
            detection = cid.get("detection", {})
            cats_present: set = set()

            for group_name, group_data in detection.items():
                if not group_data.get("found", False):
                    continue

                group_hit_counts[group_name] = (
                    group_hit_counts.get(group_name, 0) + 1
                )
                cat = self._classify_ion_category(group_name)
                cats_present.add(cat)

                # Intensity
                max_int = group_data.get("max_intensity", 0.0)
                if max_int > 0:
                    category_intensities[cat].append(max_int)

                # PPM errors (compute from matched_mz vs closest target)
                target_list = ion_library.get(group_name, [])
                for matched_mz_val in group_data.get("matched_mz", []):
                    if target_list:
                        closest = min(
                            target_list,
                            key=lambda t: abs(t - matched_mz_val),
                        )
                        ppm = (
                            (matched_mz_val - closest) / closest * 1e6
                        )
                        ppm_errors_all.append(ppm)
                        ppm_errors_by_cat[cat].append(ppm)

            for cat in cats_present:
                category_hit_counts[cat] = (
                    category_hit_counts.get(cat, 0) + 1
                )
            per_spectrum_categories.append(cats_present)

            # Coverage lift per m/z range
            for rn, rd in cid.get("coverage_by_mz_range", {}).items():
                if rn in range_totals:
                    for key in ("n_peaks", "n_fragment_annotated", "n_custom_only"):
                        range_totals[rn][key] += rd.get(key, 0)

        # ------------------------------------------------------------------
        # Derived statistics
        # ------------------------------------------------------------------
        # Hit rates
        group_hit_rates = {
            g: {"n_found": c, "hit_rate": c / max(n_spectra, 1)}
            for g, c in group_hit_counts.items()
        }
        category_hit_rates = {
            c: {"n_found": n, "hit_rate": n / max(n_spectra, 1)}
            for c, n in category_hit_counts.items()
        }

        # Intensity stats
        category_intensity_stats = {}
        for cat, vals in category_intensities.items():
            if vals:
                arr = np.array(vals)
                category_intensity_stats[cat] = {
                    "n_detections": len(vals),
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "std": float(arr.std()),
                    "q25": float(np.percentile(arr, 25)),
                    "q75": float(np.percentile(arr, 75)),
                }
            else:
                category_intensity_stats[cat] = {
                    "n_detections": 0,
                    "mean": 0.0,
                    "median": 0.0,
                    "std": 0.0,
                    "q25": 0.0,
                    "q75": 0.0,
                }

        # PPM error stats
        ppm_summary = {}
        if ppm_errors_all:
            arr = np.array(ppm_errors_all)
            ppm_summary = {
                "n": len(arr),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "median_abs": float(np.median(np.abs(arr))),
                "p95_abs": float(np.percentile(np.abs(arr), 95)),
            }

        # Co-occurrence (Jaccard + conditional probabilities)
        co_occurrence: Dict[str, Dict[str, float]] = {}
        for i, cat_a in enumerate(categories):
            for j, cat_b in enumerate(categories):
                if i > j:
                    continue
                both = sum(
                    1
                    for s in per_spectrum_categories
                    if cat_a in s and cat_b in s
                )
                either = sum(
                    1
                    for s in per_spectrum_categories
                    if cat_a in s or cat_b in s
                )
                co_occurrence[f"{cat_a}__x__{cat_b}"] = {
                    "jaccard": both / max(either, 1),
                    "n_both": both,
                    "n_either": either,
                    "p_b_given_a": both
                    / max(category_hit_counts.get(cat_a, 0), 1),
                    "p_a_given_b": both
                    / max(category_hit_counts.get(cat_b, 0), 1),
                }

        # Coverage lift
        total_unannotated = sum(
            rd["n_peaks"] - rd["n_fragment_annotated"]
            for rd in range_totals.values()
        )
        total_custom_lift = sum(
            rd["n_custom_only"] for rd in range_totals.values()
        )
        coverage_lift = {
            "total_unannotated_peaks": total_unannotated,
            "total_explained_by_custom_ions": total_custom_lift,
            "overall_lift_fraction": total_custom_lift
            / max(total_unannotated, 1),
            "by_mz_range": range_totals,
        }

        analysis: Dict[str, Any] = {
            "n_spectra": n_spectra,
            "group_hit_rates": group_hit_rates,
            "category_hit_rates": category_hit_rates,
            "category_intensity_stats": category_intensity_stats,
            "ppm_summary": ppm_summary,
            "co_occurrence": co_occurrence,
            "coverage_lift": coverage_lift,
            # Raw arrays kept for visualization (not serialised to JSON)
            "_category_intensities": category_intensities,
            "_ppm_errors_all": ppm_errors_all,
            "_ppm_errors_by_cat": ppm_errors_by_cat,
            "_per_spectrum_categories": per_spectrum_categories,
        }

        self.results["custom_ion_analysis"] = analysis
        logger.debug(
            f"Custom ions: {n_spectra:,d} spectra, "
            f"coverage lift {coverage_lift['overall_lift_fraction'] * 100:.2f}%"
        )
        return analysis

    # =========================================================================
    # Neutral Loss Analysis (Aggregation)
    # =========================================================================

    def calculate_neutral_loss_analysis(self) -> Dict[str, Any]:
        """Aggregate neutral loss statistics across all analysed spectra.

        Produces four categories of insight:

        1. **Prevalence** — per-loss-type: spectra hit rate, total count,
           mean count per spectrum.
        2. **Intensity ratio** — loss-to-parent intensity ratio by loss type
           (mean, median, quartiles).
        3. **Mass error** — |PPM| error by loss type vs base ions.
        4. **Ion series cross-tab** — loss_type x ion_series counts.
        """
        per_spectrum = self.results.get("per_spectrum", [])
        valid = [r for r in per_spectrum if r.get("loss_details")]
        if not valid:
            logger.debug("No neutral loss details to aggregate")
            return {}

        n_total_spectra = len(
            [r for r in per_spectrum if r.get("sequence_available", False)]
        )
        n_spectra_with_losses = len(valid)

        # --- Prevalence ---
        loss_type_counts: Dict[str, int] = {}
        loss_type_spectra: Dict[str, int] = {}
        # --- Intensity ratios ---
        intensity_ratios: Dict[str, List[float]] = {}
        # --- Ion series cross-tab ---
        crosstab_counts: Dict[str, Dict[str, int]] = {}

        for r in valid:
            loss_types_in_spectrum: set = set()
            for ld in r["loss_details"]:
                lt = ld["loss_type"]
                loss_type_counts[lt] = loss_type_counts.get(lt, 0) + 1
                loss_types_in_spectrum.add(lt)

                # Intensity ratio
                ir = ld.get("intensity_ratio")
                if ir is not None:
                    intensity_ratios.setdefault(lt, []).append(ir)

                # Ion series cross-tab
                ion_s = ld.get("ion_series", "unknown")
                if lt not in crosstab_counts:
                    crosstab_counts[lt] = {}
                crosstab_counts[lt][ion_s] = crosstab_counts[lt].get(ion_s, 0) + 1

            for lt in loss_types_in_spectrum:
                loss_type_spectra[lt] = loss_type_spectra.get(lt, 0) + 1

        # Build prevalence dict
        prevalence: Dict[str, Dict[str, Any]] = {}
        for lt in sorted(loss_type_counts.keys()):
            total_count = loss_type_counts[lt]
            n_spectra_hit = loss_type_spectra.get(lt, 0)
            prevalence[lt] = {
                "total_count": total_count,
                "n_spectra_hit": n_spectra_hit,
                "spectra_hit_rate": n_spectra_hit / max(n_total_spectra, 1),
                "mean_per_spectrum": total_count / max(n_spectra_with_losses, 1),
            }

        # Build intensity ratio stats
        intensity_ratio_stats: Dict[str, Dict[str, float]] = {}
        for lt in sorted(intensity_ratios.keys()):
            vals = np.array(intensity_ratios[lt])
            intensity_ratio_stats[lt] = {
                "n": len(vals),
                "mean": float(vals.mean()),
                "median": float(np.median(vals)),
                "std": float(vals.std()),
                "q25": float(np.percentile(vals, 25)),
                "q75": float(np.percentile(vals, 75)),
            }

        # Mass error by loss type (from mass_error_df)
        mass_error_by_loss_type: Dict[str, Dict[str, float]] = {}
        mass_error_df = self.results.get("mass_error_df")
        if mass_error_df is not None and "annotation" in mass_error_df.columns:
            loss_rows = mass_error_df[mass_error_df["feature_type"] == "loss"]
            if len(loss_rows) > 0:
                loss_rows = loss_rows.copy()
                loss_rows["loss_type"] = loss_rows["annotation"].apply(
                    self._extract_loss_type
                )
                for lt in loss_rows["loss_type"].dropna().unique():
                    subset = loss_rows[loss_rows["loss_type"] == lt]
                    abs_ppm = subset["delta_mz_ppm"].values
                    mass_error_by_loss_type[lt] = {
                        "n": len(abs_ppm),
                        "mean_abs_ppm": float(np.mean(abs_ppm)),
                        "median_abs_ppm": float(np.median(abs_ppm)),
                        "std_abs_ppm": float(np.std(abs_ppm)),
                        "p95_abs_ppm": float(np.percentile(abs_ppm, 95)),
                    }

            # Also compute base ion mass error for comparison
            base_rows = mass_error_df[mass_error_df["feature_type"] == "base"]
            if len(base_rows) > 0:
                abs_ppm = base_rows["delta_mz_ppm"].values
                mass_error_by_loss_type["_base_ions"] = {
                    "n": len(abs_ppm),
                    "mean_abs_ppm": float(np.mean(abs_ppm)),
                    "median_abs_ppm": float(np.median(abs_ppm)),
                    "std_abs_ppm": float(np.std(abs_ppm)),
                    "p95_abs_ppm": float(np.percentile(abs_ppm, 95)),
                }

        analysis: Dict[str, Any] = {
            "n_total_spectra": n_total_spectra,
            "n_spectra_with_losses": n_spectra_with_losses,
            "prevalence": prevalence,
            "intensity_ratio_stats": intensity_ratio_stats,
            "mass_error_by_loss_type": mass_error_by_loss_type,
            "crosstab": crosstab_counts,
            # Raw arrays for visualization (not serialised to JSON)
            "_intensity_ratios": intensity_ratios,
        }

        self.results["neutral_loss_analysis"] = analysis
        logger.debug(
            f"Neutral losses: {n_spectra_with_losses:,d} spectra, "
            f"{len(prevalence)} loss types"
        )
        return analysis

    # =========================================================================
    # Fragment Group Analysis
    # =========================================================================

    def calculate_fragment_group_analysis(self) -> Dict[str, Any]:
        """Analyse fragment ion group structure across spectra.

        For each spectrum with annotations, extracts base fragment ions and
        computes:

        1. **Fragment count** — number of unique (ion_type, position) groups.
        2. **Backbone cleavage coverage** — fraction of possible cleavage
           sites observed (mapped from N-terminal and C-terminal ions).
        3. **Charge multiplicity** — how many charge states each group
           appears at (1×, 2×, 3+×).
        4. **Charge distribution** — fraction of fragment observations at
           each charge state.
        5. **Charge by fragment size** — mean fragment charge as a function
           of relative position, stratified by precursor charge.

        Results are stored in ``self.results["fragment_group_analysis"]``.
        """
        per_spectrum = self.results.get("per_spectrum", [])
        valid = [
            r
            for r in per_spectrum
            if r.get("sequence_available", False)
            and r.get("feature_types")
            and r.get("theo_annotations")
        ]
        if not valid:
            logger.debug("No annotated spectra for fragment group analysis")
            return {}

        # N-terminal ion types (cleavage_site = position)
        n_terminal_ions = {"b", "a", "c"}
        # C-terminal ion types (cleavage_site = seq_len - position)
        c_terminal_ions = {"y", "x", "z"}

        # Per-series breakdown: track every backbone ion series so the
        # visualisation can pick the pair relevant to each fragmentation
        # method (b/y for HCD/CID/HCID, c/z for ETD/ECD). All six are
        # always accumulated; series the theoretical generator did not
        # emit for a given frag_type simply stay at 0.
        primary_series = ("a", "b", "c", "x", "y", "z")

        # Per-spectrum accumulators
        per_spectrum_n_groups: List[float] = []
        per_spectrum_coverage: List[float] = []
        per_spectrum_frag_type: List[str] = []
        per_spectrum_complementary: List[float] = []
        per_spectrum_n_complementary: List[int] = []
        per_spectrum_n_either: List[int] = []
        # Per-spectrum per-ion-series counts: [{series: count}]
        per_spectrum_n_by_series: List[Dict[str, int]] = []
        # Per-spectrum per-ion-series coverage: [{series: fraction}]
        per_spectrum_coverage_by_series: List[Dict[str, float]] = []

        # Ladder analysis accumulators (consecutive position runs per ion series)
        per_spectrum_max_ladder: Dict[str, List[int]] = {s: [] for s in primary_series}
        per_spectrum_n_ladders: Dict[str, List[int]] = {s: [] for s in primary_series}
        all_ladder_lengths: Dict[str, List[int]] = {s: [] for s in primary_series}
        per_spectrum_mean_peaks_per_pos: Dict[str, List[float]] = {
            s: [] for s in primary_series
        }
        # Raw per-group peak counts (one entry per fragment ion group across all spectra)
        all_group_peak_counts: Dict[str, List[int]] = {s: [] for s in primary_series}

        # Charge multiplicity counts across all spectra
        charge_mult_counts = {1: 0, 2: 0, "3+": 0}  # groups with 1, 2, 3+ charges
        total_groups = 0

        # Per-ion-series multiplicity counts (overall)
        series_mult_counts: Dict[str, Dict] = {
            s: {1: 0, 2: 0, "3+": 0, "total": 0} for s in primary_series
        }

        # Charge distribution across all fragment observations
        charge_obs_counts = {1: 0, 2: 0, "3+": 0}
        total_observations = 0

        # Per frag_type accumulators (now includes per-ion-series sub-dicts)
        frag_type_data: Dict[str, Dict[str, Any]] = {}

        def _init_frag_type_entry() -> Dict[str, Any]:
            """Create a fresh frag_type accumulator."""
            entry: Dict[str, Any] = {
                "n_groups": [],
                "coverage": [],
                "mult_1": 0,
                "mult_2": 0,
                "mult_3+": 0,
                "total_groups": 0,
                "charge_1": 0,
                "charge_2": 0,
                "charge_3+": 0,
                "total_obs": 0,
            }
            # Per-ion-series sub-accumulators
            for s in primary_series:
                entry[f"n_groups_{s}"] = []  # per-spectrum count of this series
                entry[f"coverage_{s}"] = []  # per-spectrum coverage of this series
                entry[f"mult_1_{s}"] = 0
                entry[f"mult_2_{s}"] = 0
                entry[f"mult_3+_{s}"] = 0
                entry[f"total_groups_{s}"] = 0
                # Ladder accumulators per ion series
                entry[f"max_ladder_{s}"] = []
                entry[f"all_ladder_lengths_{s}"] = []
                entry[f"mean_peaks_per_pos_{s}"] = []
                entry[f"all_group_peak_counts_{s}"] = []
            return entry

        # Fragment observations for Panel D: (rel_pos, charge, prec_charge, frag_type)
        fragment_observations: List[Tuple[float, int, int, str]] = []

        for r in valid:
            feature_types = r["feature_types"]
            annotations = r["theo_annotations"]
            seq = r.get("clean_sequence", "")
            seq_len = len(seq) if seq else 0
            precursor_charge = r.get("precursor_charge", 2)
            meta = r.get("_metadata", {})
            frag_type = str(meta.get("frag_type", "unknown"))

            # Collect base fragment groups: {(ion_type, position): {charges}}
            groups: Dict[Tuple[str, int], set] = {}
            for ft, ann in zip(feature_types, annotations):
                if ft != "base" or not ann:
                    continue
                ion_type = self._extract_ion_type(ann)
                position = self._extract_fragment_position(ann)
                if position < 1 or ion_type == "unknown":
                    continue
                charge = self._extract_charge_from_annotation(ann)
                key = (ion_type, position)
                groups.setdefault(key, set()).add(charge)

            n_groups = len(groups)
            per_spectrum_n_groups.append(n_groups)
            per_spectrum_frag_type.append(frag_type)

            # Complementary ion pairs: b_i and y_{seq_len - i} share cleavage site i
            b_positions = {pos for (ion, pos) in groups if ion == "b"}
            y_positions = {pos for (ion, pos) in groups if ion == "y"}

            # --- Ladder analysis: consecutive position runs per ion series ---
            # For each fragment ion group at a specific charge state
            # (base ion + isotopes), measure the m/z-index span = total peaks
            # (including interleaved noise) between the first and last peak.
            # This tells us the actual span needed in thompson span masking.
            #
            # Key by (ion_type, position, charge) because the same ion at
            # different charge states occupies completely different m/z regions
            # (e.g. y7+ at ~800 m/z vs y7++ at ~400 m/z).  Losses excluded:
            # they are ~17-18 Da away, not m/z-consecutive.

            parent_annotations = r.get("parent_annotations")
            # (ion_type, position, charge) -> list of peak indices
            charge_group_indices: Dict[Tuple[str, int, int], List[int]] = {}
            for peak_idx, (ft, ann, p_ann) in enumerate(zip(
                feature_types,
                annotations,
                parent_annotations if parent_annotations else [None] * len(annotations),
            )):
                if not ann or ft not in ("base", "isotope"):
                    continue
                ref_ann = p_ann if (ft == "isotope" and p_ann) else ann
                it = self._extract_ion_type(ref_ann)
                pos = self._extract_fragment_position(ref_ann)
                charge = self._extract_charge_from_annotation(ann)
                if pos >= 1 and it != "unknown":
                    charge_group_indices.setdefault(
                        (it, pos, charge), []
                    ).append(peak_idx)

            # Compute m/z-index span per charge-specific group
            charge_group_spans: Dict[Tuple[str, int, int], int] = {}
            for key, indices in charge_group_indices.items():
                charge_group_spans[key] = max(indices) - min(indices) + 1

            series_positions: Dict[str, set] = {
                s: {pos for (ion, pos) in groups if ion == s}
                for s in primary_series
            }
            for s in primary_series:
                ladders = self._compute_ion_ladders(series_positions[s])
                lengths = [len(lad) for lad in ladders]
                max_lad = max(lengths) if lengths else 0
                per_spectrum_max_ladder[s].append(max_lad)
                per_spectrum_n_ladders[s].append(len(ladders))
                all_ladder_lengths[s].extend(lengths)
                # Annotated peaks per position (sum across charge states)
                ppp_vals = [
                    sum(
                        len(idxs)
                        for (it, p, c), idxs in charge_group_indices.items()
                        if it == s and p == pos
                    )
                    for lad in ladders
                    for pos in lad
                ]
                per_spectrum_mean_peaks_per_pos[s].append(
                    float(np.mean(ppp_vals)) if ppp_vals else 0.0
                )
                # m/z-index span per charge-specific group
                for (it, pos, charge), span in charge_group_spans.items():
                    if it == s:
                        all_group_peak_counts[s].append(span)

            if seq_len > 1:
                n_complementary = sum(
                    1
                    for site in range(1, seq_len)
                    if site in b_positions and (seq_len - site) in y_positions
                )
                n_either = sum(
                    1
                    for site in range(1, seq_len)
                    if site in b_positions or (seq_len - site) in y_positions
                )
                comp_frac = n_complementary / max(n_either, 1)
            else:
                n_complementary = 0
                n_either = 0
                comp_frac = 0.0
            per_spectrum_complementary.append(comp_frac)
            per_spectrum_n_complementary.append(n_complementary)
            per_spectrum_n_either.append(n_either)

            # Per-ion-series group counts for this spectrum
            series_counts: Dict[str, int] = {}
            for s in primary_series:
                series_counts[s] = sum(
                    1 for (it, _) in groups if it == s
                )
            per_spectrum_n_by_series.append(series_counts)

            # Backbone cleavage coverage (overall and per-ion-series)
            if seq_len > 1:
                max_sites = seq_len - 1
                cleavage_sites: set = set()
                series_cleavage_sites: Dict[str, set] = {
                    s: set() for s in primary_series
                }
                for (ion_type, position), charges in groups.items():
                    if ion_type in n_terminal_ions:
                        site = position
                    elif ion_type in c_terminal_ions:
                        site = seq_len - position
                    else:
                        continue
                    if 0 < site < seq_len:
                        cleavage_sites.add(site)
                        if ion_type in primary_series:
                            series_cleavage_sites[ion_type].add(site)
                coverage = len(cleavage_sites) / max_sites
                series_coverage = {
                    s: len(sites) / max_sites
                    for s, sites in series_cleavage_sites.items()
                }
            else:
                coverage = 0.0
                series_coverage = {s: 0.0 for s in primary_series}
            per_spectrum_coverage.append(coverage)
            per_spectrum_coverage_by_series.append(series_coverage)

            # Charge multiplicity + distribution (overall and per-series)
            for (ion_type, position), charges in groups.items():
                n_charges = len(charges)
                if n_charges == 1:
                    charge_mult_counts[1] += 1
                elif n_charges == 2:
                    charge_mult_counts[2] += 1
                else:
                    charge_mult_counts["3+"] += 1
                total_groups += 1

                # Per-ion-series multiplicity (overall)
                if ion_type in primary_series:
                    smc = series_mult_counts[ion_type]
                    if n_charges == 1:
                        smc[1] += 1
                    elif n_charges == 2:
                        smc[2] += 1
                    else:
                        smc["3+"] += 1
                    smc["total"] += 1

                # Per-observation charge distribution + fragment obs for Panel D
                for c in charges:
                    if c == 1:
                        charge_obs_counts[1] += 1
                    elif c == 2:
                        charge_obs_counts[2] += 1
                    else:
                        charge_obs_counts["3+"] += 1
                    total_observations += 1

                    # Record for Panel D (relative position)
                    if seq_len > 1:
                        rel_pos = position / (seq_len - 1)
                        fragment_observations.append(
                            (rel_pos, c, precursor_charge, frag_type)
                        )

            # Accumulate per frag_type
            if frag_type not in frag_type_data:
                frag_type_data[frag_type] = _init_frag_type_entry()
            ftd = frag_type_data[frag_type]
            ftd["n_groups"].append(n_groups)
            ftd["coverage"].append(coverage)
            for s in primary_series:
                ftd[f"n_groups_{s}"].append(series_counts[s])
                ftd[f"coverage_{s}"].append(series_coverage[s])
                ftd[f"max_ladder_{s}"].append(per_spectrum_max_ladder[s][-1])
                # Ladder lengths for this spectrum: slice from the global accumulator
                # (n_ladders tells us how many were appended for this spectrum)
                n_lad = per_spectrum_n_ladders[s][-1]
                if n_lad > 0:
                    ftd[f"all_ladder_lengths_{s}"].extend(
                        all_ladder_lengths[s][-n_lad:]
                    )
                ftd[f"mean_peaks_per_pos_{s}"].append(
                    per_spectrum_mean_peaks_per_pos[s][-1]
                )
                # m/z-index span per charge-specific group for this frag_type
                for (it, pos, charge), span in charge_group_spans.items():
                    if it == s:
                        ftd[f"all_group_peak_counts_{s}"].append(span
                    )
            for (ion_type, position), charges in groups.items():
                n_charges = len(charges)
                if n_charges == 1:
                    ftd["mult_1"] += 1
                elif n_charges == 2:
                    ftd["mult_2"] += 1
                else:
                    ftd["mult_3+"] += 1
                ftd["total_groups"] += 1
                # Per-ion-series multiplicity within this frag_type
                if ion_type in primary_series:
                    if n_charges == 1:
                        ftd[f"mult_1_{ion_type}"] += 1
                    elif n_charges == 2:
                        ftd[f"mult_2_{ion_type}"] += 1
                    else:
                        ftd[f"mult_3+_{ion_type}"] += 1
                    ftd[f"total_groups_{ion_type}"] += 1
                for c in charges:
                    if c == 1:
                        ftd["charge_1"] += 1
                    elif c == 2:
                        ftd["charge_2"] += 1
                    else:
                        ftd["charge_3+"] += 1
                    ftd["total_obs"] += 1

        def _dist_stats(values: List[float]) -> Dict[str, float]:
            """Compute distribution summary statistics."""
            arr = np.array(values)
            if len(arr) == 0:
                return {
                    "mean": 0.0, "median": 0.0, "std": 0.0,
                    "q25": 0.0, "q75": 0.0, "min": 0.0, "max": 0.0,
                }
            return {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std()),
                "q25": float(np.percentile(arr, 25)),
                "q75": float(np.percentile(arr, 75)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }

        def _mult_fracs(m1: int, m2: int, m3: int, total: int) -> Dict[str, float]:
            """Compute multiplicity fractions."""
            if total == 0:
                return {"1": 0.0, "2": 0.0, "3+": 0.0}
            return {
                "1": m1 / total,
                "2": m2 / total,
                "3+": m3 / total,
            }

        def _charge_fracs(c1: int, c2: int, c3: int, total: int) -> Dict[str, float]:
            """Compute charge distribution fractions."""
            if total == 0:
                return {"1": 0.0, "2": 0.0, "3+": 0.0}
            return {
                "1": c1 / total,
                "2": c2 / total,
                "3+": c3 / total,
            }

        # Build overall statistics
        overall: Dict[str, Any] = {
            "fragment_count": _dist_stats(per_spectrum_n_groups),
            "backbone_coverage": _dist_stats(per_spectrum_coverage),
            "charge_multiplicity": _mult_fracs(
                charge_mult_counts[1],
                charge_mult_counts[2],
                charge_mult_counts["3+"],
                total_groups,
            ),
            "charge_distribution": _charge_fracs(
                charge_obs_counts[1],
                charge_obs_counts[2],
                charge_obs_counts["3+"],
                total_observations,
            ),
        }

        # Per-ion-series overall stats
        by_ion_series: Dict[str, Dict[str, Any]] = {}
        for s in primary_series:
            s_counts = [d[s] for d in per_spectrum_n_by_series]
            s_coverages = [d[s] for d in per_spectrum_coverage_by_series]
            smc = series_mult_counts[s]
            by_ion_series[s] = {
                "fragment_count": _dist_stats(s_counts),
                "backbone_coverage": _dist_stats(s_coverages),
                "charge_multiplicity": _mult_fracs(
                    smc[1], smc[2], smc["3+"], smc["total"]
                ),
            }
        overall["by_ion_series"] = by_ion_series

        # ---- Ladder analysis (overall) ----
        def _ladder_stats_for_series(
            max_ladder_list: List[int],
            all_lengths: List[int],
            mean_ppp: List[float],
            group_mz_spans: Optional[List[int]] = None,
        ) -> Dict[str, Any]:
            """Build ladder summary for one ion series."""
            stats: Dict[str, Any] = {
                "max_ladder_length": _dist_stats([float(x) for x in max_ladder_list]),
                "n_spectra": len(max_ladder_list),
                "all_ladder_lengths": {
                    "mean": float(np.mean(all_lengths)) if all_lengths else 0.0,
                    "median": float(np.median(all_lengths)) if all_lengths else 0.0,
                    "p75": float(np.percentile(all_lengths, 75)) if all_lengths else 0.0,
                    "p90": float(np.percentile(all_lengths, 90)) if all_lengths else 0.0,
                    "total_count": len(all_lengths),
                },
                "mean_annotated_peaks_per_position": _dist_stats(mean_ppp),
            }
            if group_mz_spans:
                stats["group_mz_span"] = {
                    "mean": float(np.mean(group_mz_spans)),
                    "median": float(np.median(group_mz_spans)),
                    "p75": float(np.percentile(group_mz_spans, 75)),
                    "p90": float(np.percentile(group_mz_spans, 90)),
                    "total_groups": len(group_mz_spans),
                }
            return stats

        ladder_analysis: Dict[str, Any] = {}
        for s in primary_series:
            ladder_analysis[s] = _ladder_stats_for_series(
                per_spectrum_max_ladder[s],
                all_ladder_lengths[s],
                per_spectrum_mean_peaks_per_pos[s],
                all_group_peak_counts[s],
            )

        # Combined span recommendation
        # group_mz_span directly measures how many consecutive m/z-sorted peaks
        # (including interleaved noise) span a single fragment ion group.
        # This is exactly the span needed to mask one complete group.
        combined_group_spans: List[int] = []
        combined_lengths: List[int] = []
        for s in primary_series:
            combined_group_spans.extend(all_group_peak_counts[s])
            combined_lengths.extend(all_ladder_lengths[s])
        if combined_group_spans:
            med_span = float(np.median(combined_group_spans))
            p75_span = float(np.percentile(combined_group_spans, 75))
            p90_span = float(np.percentile(combined_group_spans, 90))
            med_ladder = float(np.median(combined_lengths)) if combined_lengths else 0.0
            ladder_analysis["combined"] = {
                "median_ladder_length_positions": med_ladder,
                "median_group_mz_span": med_span,
                "p75_group_mz_span": p75_span,
                "p90_group_mz_span": p90_span,
                # span_min should cover most groups (use p75);
                # span_max to cover larger groups (use p90)
                "suggested_span_min": max(2, int(round(p75_span))),
                "suggested_span_max": max(4, int(round(p90_span))),
                "reasoning": (
                    f"Group size: median={med_span:.0f}, "
                    f"p75={p75_span:.0f}, p90={p90_span:.0f} peaks "
                    f"(base + isotope per charge state). "
                    f"Median ladder = {med_ladder:.1f} consecutive positions"
                ),
            }

        # Per frag_type ladder breakdown
        ladder_by_frag_type: Dict[str, Dict[str, Any]] = {}
        for ft, ftd in sorted(frag_type_data.items()):
            ft_ladder: Dict[str, Any] = {}
            for s in primary_series:
                ft_ladder[s] = _ladder_stats_for_series(
                    ftd[f"max_ladder_{s}"],
                    ftd[f"all_ladder_lengths_{s}"],
                    ftd[f"mean_peaks_per_pos_{s}"],
                    ftd[f"all_group_peak_counts_{s}"],
                )
            ladder_by_frag_type[ft] = ft_ladder
        ladder_analysis["by_frag_type"] = ladder_by_frag_type

        # Build per frag_type statistics
        by_frag_type: Dict[str, Dict[str, Any]] = {}
        for ft, ftd in sorted(frag_type_data.items()):
            ft_entry: Dict[str, Any] = {
                "fragment_count": _dist_stats(ftd["n_groups"]),
                "backbone_coverage": _dist_stats(ftd["coverage"]),
                "charge_multiplicity": _mult_fracs(
                    ftd["mult_1"], ftd["mult_2"], ftd["mult_3+"], ftd["total_groups"]
                ),
                "charge_distribution": _charge_fracs(
                    ftd["charge_1"],
                    ftd["charge_2"],
                    ftd["charge_3+"],
                    ftd["total_obs"],
                ),
            }
            # Per-ion-series within this frag_type
            ft_by_series: Dict[str, Dict[str, Any]] = {}
            for s in primary_series:
                ft_by_series[s] = {
                    "fragment_count": _dist_stats(ftd[f"n_groups_{s}"]),
                    "backbone_coverage": _dist_stats(ftd[f"coverage_{s}"]),
                    "charge_multiplicity": _mult_fracs(
                        ftd[f"mult_1_{s}"],
                        ftd[f"mult_2_{s}"],
                        ftd[f"mult_3+_{s}"],
                        ftd[f"total_groups_{s}"],
                    ),
                }
            ft_entry["by_ion_series"] = ft_by_series
            by_frag_type[ft] = ft_entry

        # Complementary pair summary statistics
        comp_arr = np.array(per_spectrum_complementary)
        complementary_pair_stats = _dist_stats(per_spectrum_complementary)

        analysis: Dict[str, Any] = {
            "n_spectra": len(valid),
            "overall": overall,
            "by_frag_type": by_frag_type,
            "complementary_pair_fraction": complementary_pair_stats,
            "ladder_analysis": ladder_analysis,
            # Internal arrays for visualization (not serialized to JSON)
            "_per_spectrum_n_groups": per_spectrum_n_groups,
            "_per_spectrum_n_by_series": per_spectrum_n_by_series,
            "_per_spectrum_coverage": per_spectrum_coverage,
            "_per_spectrum_coverage_by_series": per_spectrum_coverage_by_series,
            "_per_spectrum_frag_type": per_spectrum_frag_type,
            "_per_spectrum_complementary": per_spectrum_complementary,
            "_fragment_observations": fragment_observations,
            "_per_spectrum_max_ladder": per_spectrum_max_ladder,
            "_all_ladder_lengths": all_ladder_lengths,
            "_per_spectrum_mean_peaks_per_pos": per_spectrum_mean_peaks_per_pos,
            "_all_group_peak_counts": all_group_peak_counts,
        }

        self.results["fragment_group_analysis"] = analysis

        fc = overall["fragment_count"]
        bc = overall["backbone_coverage"]
        logger.debug(
            f"Fragment groups: {fc['mean']:.1f} +/- {fc['std']:.1f}, "
            f"backbone coverage: {bc['mean']*100:.1f}%"
        )
        return analysis

    # =========================================================================
    # Blur σ Calibration Analysis
    # =========================================================================

    def calculate_blur_sigma_analysis(self) -> Dict[str, Any]:
        """Compute within-group spread and between-group gap distributions.

        These distributions guide the selection of the Gaussian blur σ for
        masked peak encoding.  The ideal σ satisfies:

            within-group spread  <  σ  <  between-group gap

        so that the model cannot trivially resolve group members from the
        blur alone, yet different fragment groups receive distinguishable
        blurred encodings.

        **Within-group spread**: For each fragment group (base + isotopes +
        losses sharing the same parent), the m/z distance from lowest to
        highest member.

        **Between-group gap**: For adjacent fragment groups in each spectrum
        (sorted by centroid m/z), the edge-to-edge m/z distance.

        Results stored in ``self.results["blur_sigma_analysis"]``.
        """
        mass_error_df = self.results.get("mass_error_df")
        if mass_error_df is None or mass_error_df.empty:
            logger.debug("No mass error data for blur σ analysis")
            return {}

        # Need: peptide, exp_mz, annotation, feature_type, ion_type, charge, position
        required = {"peptide", "exp_mz", "annotation", "feature_type", "ion_type", "charge", "position"}
        if not required.issubset(mass_error_df.columns):
            logger.debug(f"Missing columns for blur σ analysis: {required - set(mass_error_df.columns)}")
            return {}

        df = mass_error_df.copy()

        # --- Assign each peak to its parent fragment group ---
        # Base ions: group = (peptide, ion_type, position, charge)
        # Isotopes: parent is the annotation without the [+N] suffix
        # Losses: parent is the annotation without the -H2O/-NH3 suffix
        # We group by the BASE annotation to collect all members of a group.

        def _extract_parent_key(row: pd.Series) -> str:
            """Map each peak to its parent fragment group key."""
            ann = row["annotation"]
            ft = row["feature_type"]
            pep = row["peptide"]
            if not ann:
                return ""
            # For isotopes: strip [+N] suffix → parent base annotation
            if ft == "isotope":
                parent = re.sub(r"\[\+\d+\]$", "", ann)
                return f"{pep}|{parent}"
            # For losses: strip -H2O, -NH3, etc → parent base annotation
            if ft == "loss":
                parent = re.sub(r"-[A-Za-z0-9]+$", "", ann)
                return f"{pep}|{parent}"
            # Base ions: use annotation directly
            return f"{pep}|{ann}"

        df["group_key"] = df.apply(_extract_parent_key, axis=1)
        df = df[df["group_key"] != ""]

        # --- Within-group spread ---
        group_stats = df.groupby("group_key")["exp_mz"].agg(["min", "max", "mean", "count"])
        group_stats["spread_da"] = group_stats["max"] - group_stats["min"]
        # Only include groups with 2+ members (single-peak groups have spread=0)
        multi_member = group_stats[group_stats["count"] >= 2]
        within_group_spreads = multi_member["spread_da"].values

        # Split by type: isotope-only vs full (with losses)
        # Groups with spread < 5 Da are likely isotope-only; > 5 Da include losses
        isotope_only_spreads = within_group_spreads[within_group_spreads < 5.0]
        with_loss_spreads = within_group_spreads[within_group_spreads >= 5.0]

        # --- Between-group gap (per spectrum) ---
        # Compute two types of gaps:
        # 1. Same-series: gaps within the b-ladder or y-ladder (amino acid mass gaps)
        # 2. All-series: gaps between any adjacent groups in m/z order (spectrum density)

        # Extract ion_type and charge per group from the base ion annotation
        def _parse_key(key: str) -> tuple:
            """Extract (ion_type, charge) from group key like 'peptide|b3+'."""
            ann = key.split("|", 1)[1] if "|" in key else key
            ion_type = ""
            charge = 1
            if ann:
                ion_type = next((c for c in ann if c.isalpha()), "")
                charge = ann.count("+") or 1
            return ion_type, charge

        parsed = [_parse_key(k) for k in group_stats.index]
        group_stats["ion_type"] = [p[0] for p in parsed]
        group_stats["charge"] = [p[1] for p in parsed]

        same_series_gaps = []
        all_series_gaps = []

        for peptide, pep_groups in group_stats.groupby(
            group_stats.index.map(lambda k: k.split("|")[0])
        ):
            if len(pep_groups) < 2:
                continue

            # All-series gaps (any adjacent groups in m/z order)
            sorted_all = pep_groups.sort_values("mean")
            maxes_all = sorted_all["max"].values
            mins_all = sorted_all["min"].values
            for i in range(len(sorted_all) - 1):
                gap = mins_all[i + 1] - maxes_all[i]
                if gap > 0:
                    all_series_gaps.append(gap)

            # Same-series, same-charge gaps (within b+, b++, y+, y++ separately)
            # This correctly separates the z=1 ladder (gaps ~57-186 Da) from
            # the z=2 ladder (gaps ~28-93 Da) — mixing charges produces
            # spurious small gaps from interleaved charge-state groups.
            for ion_type in ("b", "y"):
                for charge in pep_groups["charge"].unique():
                    series_groups = pep_groups[
                        (pep_groups["ion_type"] == ion_type)
                        & (pep_groups["charge"] == charge)
                    ]
                    if len(series_groups) < 2:
                        continue
                    sorted_series = series_groups.sort_values("mean")
                    maxes_s = sorted_series["max"].values
                    mins_s = sorted_series["min"].values
                    for i in range(len(sorted_series) - 1):
                        gap = mins_s[i + 1] - maxes_s[i]
                        if gap > 0:
                            same_series_gaps.append(gap)

        same_series_gaps = np.array(same_series_gaps) if same_series_gaps else np.array([])
        all_series_gaps = np.array(all_series_gaps) if all_series_gaps else np.array([])

        # --- Summary statistics ---
        sigma_values = [5.0, 10.0, 15.0, 25.0, 50.0]

        def _percentile_above(arr: np.ndarray, threshold: float) -> float:
            """Fraction of values > threshold."""
            if len(arr) == 0:
                return 0.0
            return float(np.mean(arr > threshold))

        sigma_analysis = {}
        # Use same-series gaps as the primary metric for sigma selection
        # (these are the amino acid mass gaps the blur must distinguish)
        primary_gaps = same_series_gaps if len(same_series_gaps) > 0 else all_series_gaps
        for sigma in sigma_values:
            sigma_analysis[f"sigma_{int(sigma)}"] = {
                "within_group_pct_below": float(np.mean(within_group_spreads <= sigma)) if len(within_group_spreads) > 0 else 0.0,
                "isotope_only_pct_below": float(np.mean(isotope_only_spreads <= sigma)) if len(isotope_only_spreads) > 0 else 0.0,
                "same_series_pct_above": _percentile_above(same_series_gaps, sigma),
                "all_series_pct_above": _percentile_above(all_series_gaps, sigma),
            }

        def _safe_stats(arr: np.ndarray, name: str) -> dict:
            if len(arr) == 0:
                return {k: 0.0 for k in ["mean", "median", "p25", "p75", "p5", "p95", "min", "max"]}
            return {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "p5": float(np.percentile(arr, 5)),
                "p95": float(np.percentile(arr, 95)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

        analysis = {
            "within_group_spreads": within_group_spreads.tolist(),
            "isotope_only_spreads": isotope_only_spreads.tolist(),
            "with_loss_spreads": with_loss_spreads.tolist(),
            "same_series_gaps": same_series_gaps.tolist(),
            "all_series_gaps": all_series_gaps.tolist(),
            "n_groups_total": int(len(group_stats)),
            "n_groups_multi_member": int(len(multi_member)),
            "n_same_series_gaps": int(len(same_series_gaps)),
            "n_all_series_gaps": int(len(all_series_gaps)),
            "within_group_stats": _safe_stats(within_group_spreads, "within"),
            "isotope_only_stats": _safe_stats(isotope_only_spreads, "isotope_only"),
            "same_series_gap_stats": _safe_stats(same_series_gaps, "same_series"),
            "all_series_gap_stats": _safe_stats(all_series_gaps, "all_series"),
            "sigma_analysis": sigma_analysis,
        }

        self.results["blur_sigma_analysis"] = analysis

        wg = analysis["within_group_stats"]
        ss = analysis["same_series_gap_stats"]
        logger.debug(
            f"Blur σ analysis: within-group spread median={wg['median']:.1f} Da "
            f"(p95={wg['p95']:.1f}), same-series gap median={ss['median']:.1f} Da "
            f"(p5={ss['p5']:.1f}), n_groups={analysis['n_groups_multi_member']}"
        )
        return analysis

    def visualize_blur_sigma_analysis(self) -> Figure:
        """Blur σ calibration figure: within-group spread vs between-group gap.

        Three panels:
        A) Within-group m/z spread distribution (isotope-only vs with-losses)
        B) Between-group m/z gap distribution
        C) CDF comparison with σ sweet-spot overlay
        """
        bsa = self.results.get("blur_sigma_analysis", {})
        if not bsa:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.text(0.5, 0.5, "No blur σ analysis data", ha="center", va="center")
            return fig

        within = np.array(bsa["within_group_spreads"])
        iso_only = np.array(bsa["isotope_only_spreads"])
        with_loss = np.array(bsa["with_loss_spreads"])
        same_series = np.array(bsa["same_series_gaps"])
        all_series = np.array(bsa["all_series_gaps"])
        sigma_data = bsa["sigma_analysis"]
        wg_stats = bsa["within_group_stats"]
        ss_stats = bsa["same_series_gap_stats"]
        as_stats = bsa["all_series_gap_stats"]

        sigma_values = [5.0, 10.0, 15.0, 25.0, 50.0]
        sigma_colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]

        # Compute isotope-only stats for the CDF panel
        iso_stats = bsa.get("isotope_only_stats", {})
        if not iso_stats and len(iso_only) > 0:
            iso_stats = {
                "median": float(np.median(iso_only)),
                "p95": float(np.percentile(iso_only, 95)),
            }

        fig, axes = plt.subplots(4, 1, figsize=(12, 18))

        # --- Panel A: Within-group spread ---
        ax = axes[0]
        bins_a = np.linspace(0, min(50, np.percentile(within, 99) * 1.2) if len(within) > 0 else 50, 60)
        if len(iso_only) > 0:
            ax.hist(iso_only, bins=bins_a, alpha=0.6, color="#aec7e8", label=f"Isotope-only (n={len(iso_only):,})", edgecolor="white", linewidth=0.5)
        if len(with_loss) > 0:
            ax.hist(with_loss, bins=bins_a, alpha=0.6, color="#98df8a", label=f"With losses (n={len(with_loss):,})", edgecolor="white", linewidth=0.5)
        for _sig_i, (sigma, color) in enumerate(zip(sigma_values, sigma_colors)):
            ax.axvline(sigma, color=color, linestyle="--", linewidth=1.5, alpha=0.8, label=f"σ={int(sigma)} Da")
        ax.set_xlabel("Within-group m/z spread (Da)")
        ax.set_ylabel("Count")
        ax.set_title(
            f"A) Within-Group Spread — σ should exceed this\n"
            f"All groups: Median={wg_stats['median']:.1f} Da, P95={wg_stats['p95']:.1f} Da | "
            f"Isotope-only: Median={iso_stats.get('median', 0):.1f} Da, P95={iso_stats.get('p95', 0):.1f} Da"
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

        # --- Panel B: Between-group gap (same-series = ladder gaps) ---
        ax = axes[1]
        max_plot = 200
        if len(same_series) > 0:
            bins_b = np.linspace(0, min(max_plot, np.percentile(same_series, 99) * 1.1), 80)
            ax.hist(same_series, bins=bins_b, alpha=0.7, color="#c5b0d5", edgecolor="white", linewidth=0.5,
                    label=f"Same-series ladder gaps (n={len(same_series):,})")
        if len(all_series) > 0:
            bins_b2 = np.linspace(0, min(max_plot, np.percentile(all_series, 99) * 1.1), 80)
            ax.hist(all_series, bins=bins_b2, alpha=0.3, color="#aec7e8", edgecolor="white", linewidth=0.5,
                    label=f"All-series gaps (n={len(all_series):,})")
        for _sig_i, (sigma, color) in enumerate(zip(sigma_values, sigma_colors)):
            ax.axvline(sigma, color=color, linestyle="--", linewidth=1.5, alpha=0.8, label=f"σ={int(sigma)} Da")
        ax.set_xlabel("Between-group edge-to-edge gap (Da)")
        ax.set_ylabel("Count")
        ax.set_title(
            f"B) Between-Group Gap — σ should be below this\n"
            f"Same-series: Median={ss_stats['median']:.1f} Da, P5={ss_stats['p5']:.1f} Da | "
            f"All-series: Median={as_stats['median']:.1f} Da, P5={as_stats['p5']:.1f} Da"
        )
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

        # --- Panel C: CDF comparison with sweet-spot ---
        ax = axes[2]
        x_range = np.linspace(0, 80, 500)

        if len(within) > 0:
            within_sorted = np.sort(within)
            within_cdf = np.searchsorted(within_sorted, x_range) / len(within_sorted)
            ax.plot(x_range, within_cdf, color="#d62728", linewidth=2, label="Within-group spread CDF")

        # Use same-series gaps as primary (these are the amino acid mass ladder gaps)
        gap_arr = same_series if len(same_series) > 0 else all_series
        gap_label = "same-series" if len(same_series) > 0 else "all-series"
        if len(gap_arr) > 0:
            gap_sorted = np.sort(gap_arr)
            gap_cdf = np.searchsorted(gap_sorted, x_range) / len(gap_sorted)
            ax.plot(x_range, 1 - gap_cdf, color="#1f77b4", linewidth=2, label=f"P({gap_label} gap > σ)")

            # Sweet spot: region where within CDF is high AND 1-gap CDF is high
            if len(within) > 0:
                sweet = np.minimum(within_cdf, 1 - gap_cdf)
                ax.fill_between(x_range, 0, sweet, alpha=0.15, color="#2ca02c", label="Sweet spot")

        # Also show all-series gaps if different from primary
        if len(same_series) > 0 and len(all_series) > 0:
            all_sorted = np.sort(all_series)
            all_cdf = np.searchsorted(all_sorted, x_range) / len(all_sorted)
            ax.plot(x_range, 1 - all_cdf, color="#1f77b4", linewidth=1, linestyle=":", alpha=0.5,
                    label="P(all-series gap > σ)")

        # Find optimal σ that maximizes the sweet-spot (min of both curves)
        optimal_sigma = None
        if len(within) > 0 and len(gap_arr) > 0:
            sweet_values = np.minimum(within_cdf, 1 - gap_cdf)
            optimal_idx = np.argmax(sweet_values)
            optimal_sigma = x_range[optimal_idx]
            ax.axvline(optimal_sigma, color="black", linestyle="-", linewidth=2, alpha=0.6)
            ax.annotate(
                f"Optimal σ ≈ {optimal_sigma:.0f} Da",
                xy=(optimal_sigma, sweet_values[optimal_idx] + 0.05),
                fontsize=9, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="black"),
            )

        for _sig_i, (sigma, color) in enumerate(zip(sigma_values, sigma_colors)):
            sd = sigma_data.get(f"sigma_{int(sigma)}", {})
            pct_within = sd.get("within_group_pct_below", 0)
            pct_same = sd.get("same_series_pct_above", 0)
            ax.axvline(sigma, color=color, linestyle="--", linewidth=1.5, alpha=0.8)
            ax.annotate(
                f"σ={int(sigma)}\n{pct_within:.0%} hidden\n{pct_same:.0%} distinct",
                xy=(sigma, 0.03 + 0.24 * (_sig_i % 2)), fontsize=10, ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.15),
            )

        ax.set_xlabel("σ (Da)")
        ax.set_ylabel("Probability")
        ax.set_title(
            "C) Blur σ Sweet Spot\n"
            "Red = P(group spread ≤ σ) — higher means more groups hidden by blur\n"
            "Blue = P(between-group gap > σ) — higher means more groups distinguishable"
        )
        ax.legend(fontsize=11, loc="center right")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

        # --- Panel D: Isotope-only sweet spot (for visible-intensity mode) ---
        ax = axes[3]
        x_range_d = np.linspace(0, 40, 500)  # narrower range for isotope-only

        if len(iso_only) > 0:
            iso_sorted = np.sort(iso_only)
            iso_cdf = np.searchsorted(iso_sorted, x_range_d) / len(iso_sorted)
            ax.plot(x_range_d, iso_cdf, color="#d62728", linewidth=2, label="Isotope-only spread CDF")

        # Show BOTH gap curves: same-series (ladder) and all-series (cross-series)
        # The model doesn't know the ion series — a blurred b₅ near y₃ is confusing
        # regardless of series. All-series gaps are the binding constraint.
        if len(gap_arr) > 0:
            gap_sorted_d = np.sort(gap_arr)
            gap_cdf_d = np.searchsorted(gap_sorted_d, x_range_d) / len(gap_sorted_d)
            ax.plot(x_range_d, 1 - gap_cdf_d, color="#1f77b4", linewidth=2, label=f"P({gap_label} gap > σ)")

        if len(all_series) > 0:
            all_sorted_d = np.sort(all_series)
            all_cdf_d = np.searchsorted(all_sorted_d, x_range_d) / len(all_sorted_d)
            ax.plot(x_range_d, 1 - all_cdf_d, color="#1f77b4", linewidth=1.5, linestyle="--",
                    alpha=0.7, label="P(all-series gap > σ)")

            # Use all-series for the sweet spot — it's the binding constraint
            # since the model can confuse b-ion with y-ion groups at similar m/z
            if len(iso_only) > 0:
                sweet_d = np.minimum(iso_cdf, 1 - all_cdf_d)
                ax.fill_between(x_range_d, 0, sweet_d, alpha=0.15, color="#2ca02c", label="Sweet spot (all-series)")

                optimal_idx_d = np.argmax(sweet_d)
                optimal_sigma_d = x_range_d[optimal_idx_d]
                ax.axvline(optimal_sigma_d, color="black", linestyle="-", linewidth=2, alpha=0.6)
                ax.annotate(
                    f"Optimal σ ≈ {optimal_sigma_d:.0f} Da",
                    xy=(optimal_sigma_d, sweet_d[optimal_idx_d] + 0.05),
                    fontsize=9, fontweight="bold", ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="black"),
                )

        for _sig_i, (sigma, color) in enumerate(zip(sigma_values, sigma_colors)):
            if sigma <= 40:
                sd = sigma_data.get(f"sigma_{int(sigma)}", {})
                pct_iso = float(np.mean(iso_only <= sigma)) if len(iso_only) > 0 else 0.0
                pct_all = sd.get("all_series_pct_above", 0)
                ax.axvline(sigma, color=color, linestyle="--", linewidth=1.5, alpha=0.8)
                ax.annotate(
                    f"σ={int(sigma)}\n{pct_iso:.0%} hidden\n{pct_all:.0%} distinct",
                    xy=(sigma, 0.03 + 0.24 * (_sig_i % 2)), fontsize=10, ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.15),
                )

        ax.set_xlabel("σ (Da)")
        ax.set_ylabel("Probability")
        ax.set_title(
            "D) Isotope-Only Sweet Spot (visible-intensity mode: mask_intensity=false)\n"
            "All-series gaps (dashed) are the binding constraint — model can confuse b-ions with nearby y-ions"
        )
        ax.legend(fontsize=11, loc="center right")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, 40)
        ax.grid(alpha=0.3)

        fig.suptitle("Blur σ Calibration: Data-Driven Guidance for Mask Encoding", fontsize=13, fontweight="bold", y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        fig.savefig(self.output_dir / "blur_sigma_analysis.png", dpi=300, bbox_inches="tight")
        # Clean vector versions (figure convention): drop the super-title AND the per-panel
        # subtitles (their content moves to the figure caption), and enlarge the legends,
        # sigma annotation boxes, axis labels and ticks for readability.
        fig.suptitle("")
        for _a in fig.axes:
            _a.set_title("")
            _leg = _a.get_legend()
            if _leg is not None:
                for _t in _leg.get_texts():
                    _t.set_fontsize(13)
            for _txt in _a.texts:  # sigma annotation boxes + "optimal sigma" labels
                _txt.set_fontsize(max(_txt.get_fontsize(), 11))
            _a.xaxis.label.set_size(13)
            _a.yaxis.label.set_size(13)
            _a.tick_params(labelsize=11)
        fig.tight_layout(rect=[0, 0, 1, 1])
        fig.savefig(self.output_dir / "blur_sigma_analysis.svg", bbox_inches="tight",
                    metadata={"Title": "Blur-sigma calibration (LCFM)"})
        fig.savefig(self.output_dir / "blur_sigma_analysis.pdf", bbox_inches="tight")
        return fig

    # =========================================================================
    # Complementary b/y Pair Analysis
    # =========================================================================

    def calculate_complementary_pair_analysis(self) -> Dict[str, Any]:
        """Analyse complementary b/y ion pairs (b_i + y_{n-i} = M + H2O).

        For each spectrum with annotated base fragment ions, identifies
        cleavage sites where both a b-ion and its complementary y-ion are
        observed.  Validates the conservation law by computing the neutral
        mass sum of each pair and comparing to the theoretical precursor
        neutral mass plus water.

        Results are stored in ``self.results["complementary_pair_analysis"]``.
        """
        mass_error_df = self.results.get("mass_error_df")
        per_spectrum = self.results.get("per_spectrum", [])
        if mass_error_df is None or mass_error_df.empty:
            return {}

        # Filter to base b/y ions only
        base_df = mass_error_df[
            (mass_error_df["feature_type"] == "base")
            & (mass_error_df["ion_type"].isin(["b", "y"]))
        ].copy()
        if base_df.empty:
            return {}

        # Build per-peptide metadata lookup, keyed by the *raw* sequence so
        # it matches mass_error_df["peptide"] directly.  Using clean_sequence
        # plus an uppercase-only fallback previously collapsed PTM variants
        # that differed only in mod mass/UniMod number (e.g. M[UNIMOD:35] vs
        # M[UNIMOD:28] at the same position): stripping lowercase + digits
        # reduced both to the same "backbone", letting the first-seen
        # variant's precursor_mz leak into all others — biasing dev_da by
        # the PTM Δmass.
        peptide_meta: Dict[str, Dict[str, Any]] = {}
        for r in per_spectrum:
            if not r.get("sequence_available"):
                continue
            raw_seq = r.get("_metadata", {}).get("sequence")
            if not raw_seq:
                # Fall back to clean_sequence only when raw is genuinely
                # unavailable; groupby in mass_error_df uses raw, so this
                # fallback may not match but at least doesn't cross PTMs.
                raw_seq = r.get("clean_sequence", "")
            if not raw_seq:
                continue
            prec_charge = r.get("precursor_charge", 2)
            # Prefer observed precursor m/z; fall back to theoretical
            obs_mz = r.get("observed_precursor_mz", np.nan)
            theo_mz = r.get("theoretical_precursor_mz", np.nan)
            prec_mz = obs_mz if not np.isnan(obs_mz) else theo_mz
            peptide_meta[raw_seq] = {
                "precursor_charge": prec_charge,
                "precursor_mz": prec_mz,
            }

        # Per-pair accumulators
        pair_deviations_da: List[float] = []
        pair_deviations_ppm: List[float] = []
        pair_relative_positions: List[float] = []
        pair_frag_types: List[str] = []

        # Per-spectrum accumulators
        per_spectrum_pair_fraction: List[float] = []
        per_spectrum_n_pairs: List[int] = []
        per_spectrum_mean_dev_da: List[float] = []
        per_spectrum_frag_type: List[str] = []
        per_spectrum_precursor_charge: List[int] = []
        per_spectrum_seq_len: List[int] = []

        # Build worker args as pre-extracted numpy arrays (avoids DataFrame
        # pickling overhead when dispatching to worker processes).
        worker_args: List[Tuple[str, Dict[str, np.ndarray], Optional[Dict[str, Any]]]] = []
        for peptide, group in base_df.groupby("peptide"):
            meta = peptide_meta.get(peptide)
            if meta is None:
                continue
            arrays = {
                "position": group["position"].values.astype(np.int32),
                "charge": group["charge"].values.astype(np.int32),
                "exp_mz": group["exp_mz"].values.astype(np.float64),
                "ion_type": group["ion_type"].values.astype("U4"),
                "frag_type": str(group["frag_type"].iloc[0] or "unknown"),
            }
            worker_args.append((peptide, arrays, meta))

        # Dispatch: parallel for large workloads, sequential otherwise
        # Batch peptides per pickle to amortise per-task Pool overhead —
        # each Pool task processes a batch of peptides, not a single one.
        results: List[Optional[Dict[str, Any]]] = []
        _PARALLEL_THRESHOLD = 500
        _BATCH_SIZE = 500  # peptides per worker task
        if len(worker_args) >= _PARALLEL_THRESHOLD and self.n_workers > 1:
            logger.info(
                f"Complementary pair analysis: dispatching {len(worker_args):,d} peptides "
                f"to {self.n_workers} workers ({_BATCH_SIZE}/batch)"
            )
            init_args = (
                self._AA_CODES,
                self._AA_MASSES,
                self._AA_MASSES_SORTED,
                self._PROTON_MASS,
                self.mz_range_boundaries,
                self.mz_range_order,
                self.mass_gap_ppm_tol,
            )
            batches = [
                worker_args[i:i + _BATCH_SIZE]
                for i in range(0, len(worker_args), _BATCH_SIZE)
            ]
            with Pool(
                processes=self.n_workers,
                initializer=_init_theo_worker,
                initargs=init_args,
            ) as pool:
                batch_results = pool.map(_process_batch_complementary_pair, batches)
            # Flatten list-of-lists
            for br in batch_results:
                results.extend(br)
        else:
            # Sequential fallback (needs worker-state initialised once in-process)
            _init_theo_worker(
                self._AA_CODES,
                self._AA_MASSES,
                self._AA_MASSES_SORTED,
                self._PROTON_MASS,
                self.mz_range_boundaries,
                self.mz_range_order,
                self.mass_gap_ppm_tol,
            )
            results = [_process_peptide_complementary_pair(a) for a in worker_args]

        # Flatten results into the same accumulators used by the old code
        for res in results:
            if res is None:
                continue
            for pr in res["pair_records"]:
                pair_deviations_da.append(pr["dev_da"])
                pair_deviations_ppm.append(pr["dev_ppm"])
                pair_relative_positions.append(pr["rel_pos"])
                pair_frag_types.append(pr["frag_type"])
            ps = res["per_spectrum"]
            per_spectrum_pair_fraction.append(ps["pair_fraction"])
            per_spectrum_n_pairs.append(ps["n_pairs"])
            per_spectrum_mean_dev_da.append(ps["mean_dev_da"])
            per_spectrum_frag_type.append(ps["frag_type"])
            per_spectrum_precursor_charge.append(ps["precursor_charge"])
            per_spectrum_seq_len.append(ps["seq_len"])

        if not per_spectrum_pair_fraction:
            return {}

        # --- Aggregate statistics ---
        def _dist(vals: List[float]) -> Dict[str, float]:
            a = np.array(vals)
            if len(a) == 0:
                return {"mean": 0.0, "median": 0.0, "std": 0.0, "q25": 0.0, "q75": 0.0}
            return {
                "mean": float(a.mean()),
                "median": float(np.median(a)),
                "std": float(a.std()),
                "q25": float(np.percentile(a, 25)),
                "q75": float(np.percentile(a, 75)),
            }

        # Positional coverage: bin relative positions into deciles
        pos_bins = np.linspace(0, 1, 11)
        pos_bin_labels = [f"{pos_bins[i]:.1f}-{pos_bins[i+1]:.1f}" for i in range(10)]
        pos_arr = np.array(pair_relative_positions) if pair_relative_positions else np.array([])
        pos_bin_counts = [0] * 10
        if len(pos_arr) > 0:
            bin_idx = np.clip(np.digitize(pos_arr, pos_bins) - 1, 0, 9)
            for i in range(10):
                pos_bin_counts[i] = int(np.sum(bin_idx == i))

        # Stratify by fragmentation type
        by_frag_type: Dict[str, Dict[str, Any]] = {}
        for ft in set(per_spectrum_frag_type):
            mask = [f == ft for f in per_spectrum_frag_type]
            ft_fracs = [v for v, m in zip(per_spectrum_pair_fraction, mask) if m]
            by_frag_type[ft] = {
                "pair_fraction": _dist(ft_fracs),
                "n_spectra": len(ft_fracs),
            }

        # Stratify by precursor charge
        by_charge: Dict[str, Dict[str, Any]] = {}
        for z in sorted(set(per_spectrum_precursor_charge)):
            mask = [c == z for c in per_spectrum_precursor_charge]
            z_fracs = [v for v, m in zip(per_spectrum_pair_fraction, mask) if m]
            by_charge[str(z)] = {
                "pair_fraction": _dist(z_fracs),
                "n_spectra": len(z_fracs),
            }

        # Stratify by sequence length bins
        len_bins = [0, 7, 10, 15, 20, 25, 30, 999]
        by_seq_len: Dict[str, Dict[str, Any]] = {}
        for i in range(len(len_bins) - 1):
            lo, hi = len_bins[i], len_bins[i + 1]
            label = f"{lo}-{hi}" if hi < 999 else f"{lo}+"
            mask = [lo <= sl < hi for sl in per_spectrum_seq_len]
            sl_fracs = [v for v, m in zip(per_spectrum_pair_fraction, mask) if m]
            if sl_fracs:
                by_seq_len[label] = {
                    "pair_fraction": _dist(sl_fracs),
                    "n_spectra": len(sl_fracs),
                }

        analysis: Dict[str, Any] = {
            "n_spectra": len(per_spectrum_pair_fraction),
            "total_pairs": sum(per_spectrum_n_pairs),
            "pair_fraction": _dist(per_spectrum_pair_fraction),
            "deviation_da": _dist(pair_deviations_da) if pair_deviations_da else _dist([]),
            "deviation_ppm": _dist(pair_deviations_ppm) if pair_deviations_ppm else _dist([]),
            "abs_deviation_da": _dist(
                [abs(d) for d in pair_deviations_da]
            ) if pair_deviations_da else _dist([]),
            "positional_coverage": {
                "bin_labels": pos_bin_labels,
                "bin_counts": pos_bin_counts,
                "total_pairs": len(pair_relative_positions),
            },
            "by_frag_type": by_frag_type,
            "by_precursor_charge": by_charge,
            "by_seq_len": by_seq_len,
            # Internal arrays for visualization (prefixed with _)
            "_per_spectrum_pair_fraction": per_spectrum_pair_fraction,
            "_per_spectrum_frag_type": per_spectrum_frag_type,
            "_per_spectrum_precursor_charge": per_spectrum_precursor_charge,
            "_pair_deviations_da": pair_deviations_da,
            "_pair_deviations_ppm": pair_deviations_ppm,
            "_pair_relative_positions": pair_relative_positions,
        }

        self.results["complementary_pair_analysis"] = analysis
        logger.info(
            f"Complementary pair analysis: {analysis['total_pairs']:,d} pairs "
            f"across {analysis['n_spectra']:,d} spectra, "
            f"mean pair fraction {analysis['pair_fraction']['mean']*100:.1f}%"
        )
        return analysis

    # =========================================================================
    # Mass Gap Validation Against Amino Acid Masses
    # =========================================================================

    def calculate_mass_gap_analysis(self) -> Dict[str, Any]:
        """Validate mass gaps between consecutive same-series ions.

        For each pair of consecutive same-series, same-charge base fragment
        ions in a ladder, computes the neutral mass gap and checks whether
        it matches one of the 20 standard amino acid residue masses.

        Produces:
        1. **Match rate** — fraction of gaps matching a valid AA mass.
        2. **Sequence recovery** — fraction where the matched AA is correct.
        3. **Ambiguity** — how many AAs match per gap within tolerance.
        4. **Two-sided constraints** — positions with both neighbours valid.

        Results are stored in ``self.results["mass_gap_analysis"]``.
        """
        mass_error_df = self.results.get("mass_error_df")
        per_spectrum = self.results.get("per_spectrum", [])
        if mass_error_df is None or mass_error_df.empty:
            return {}

        # Filter to base fragment ions only
        base_df = mass_error_df[mass_error_df["feature_type"] == "base"].copy()
        if base_df.empty:
            return {}

        # No need for per_spectrum lookup for mass gap analysis — the peptide
        # string in mass_error_df can be tokenized directly for ground truth.

        ppm_tol = self.mass_gap_ppm_tol

        # Per-gap accumulators
        gap_records: List[Dict[str, Any]] = []

        # Per-spectrum accumulators
        per_spectrum_match_rate: List[float] = []
        per_spectrum_correct_rate: List[float] = []
        per_spectrum_frag_type_list: List[str] = []
        per_spectrum_two_sided_rate: List[float] = []

        # Build worker args as pre-extracted numpy arrays (avoids DataFrame
        # pickling overhead when dispatching to worker processes).
        worker_args: List[Tuple[str, Dict[str, np.ndarray]]] = []
        for peptide, pep_group in base_df.groupby("peptide"):
            arrays = {
                "position": pep_group["position"].values.astype(np.int32),
                "charge": pep_group["charge"].values.astype(np.int32),
                "exp_mz": pep_group["exp_mz"].values.astype(np.float64),
                "ion_type": pep_group["ion_type"].values.astype("U4"),
                "frag_type": str(pep_group["frag_type"].iloc[0] or "unknown"),
            }
            worker_args.append((peptide, arrays))

        # Dispatch: parallel for large workloads, sequential otherwise
        # Batch peptides per pickle to amortise per-task Pool overhead.
        results: List[Optional[Dict[str, Any]]] = []
        _PARALLEL_THRESHOLD = 500
        _BATCH_SIZE = 500
        if len(worker_args) >= _PARALLEL_THRESHOLD and self.n_workers > 1:
            logger.info(
                f"Mass gap analysis: dispatching {len(worker_args):,d} peptides "
                f"to {self.n_workers} workers ({_BATCH_SIZE}/batch)"
            )
            init_args = (
                self._AA_CODES,
                self._AA_MASSES,
                self._AA_MASSES_SORTED,
                self._PROTON_MASS,
                self.mz_range_boundaries,
                self.mz_range_order,
                ppm_tol,
            )
            batches = [
                worker_args[i:i + _BATCH_SIZE]
                for i in range(0, len(worker_args), _BATCH_SIZE)
            ]
            with Pool(
                processes=self.n_workers,
                initializer=_init_theo_worker,
                initargs=init_args,
            ) as pool:
                batch_results = pool.map(_process_batch_mass_gap, batches)
            for br in batch_results:
                results.extend(br)
        else:
            # Sequential fallback
            _init_theo_worker(
                self._AA_CODES,
                self._AA_MASSES,
                self._AA_MASSES_SORTED,
                self._PROTON_MASS,
                self.mz_range_boundaries,
                self.mz_range_order,
                ppm_tol,
            )
            results = [_process_peptide_mass_gap(a) for a in worker_args]

        # Flatten results into accumulators
        for res in results:
            if res is None:
                continue
            gap_records.extend(res["gap_records"])
            ps = res["per_spectrum"]
            per_spectrum_match_rate.append(ps["match_rate"])
            per_spectrum_correct_rate.append(ps["correct_rate"])
            per_spectrum_frag_type_list.append(ps["frag_type"])
            per_spectrum_two_sided_rate.append(ps["two_sided_rate"])

        if not gap_records:
            return {}

        gap_df = pd.DataFrame(gap_records)

        # --- Aggregate statistics ---
        def _dist(vals: List[float]) -> Dict[str, float]:
            a = np.array(vals)
            if len(a) == 0:
                return {"mean": 0.0, "median": 0.0, "std": 0.0, "q25": 0.0, "q75": 0.0}
            return {
                "mean": float(a.mean()),
                "median": float(np.median(a)),
                "std": float(a.std()),
                "q25": float(np.percentile(a, 25)),
                "q75": float(np.percentile(a, 75)),
            }

        n_total = len(gap_df)
        n_valid_total = int(gap_df["is_valid"].sum())
        n_correct_total = int(gap_df["is_correct"].sum())

        # Ambiguity distribution
        ambiguity_counts = gap_df["n_ambiguous"].value_counts().sort_index()
        ambiguity_dist = {
            int(k): int(v) for k, v in ambiguity_counts.items()
        }

        # Match error distribution (valid gaps only)
        valid_gaps = gap_df[gap_df["is_valid"]]
        match_errors_ppm = valid_gaps["match_error_ppm"].tolist() if len(valid_gaps) > 0 else []

        # Stratify by fragmentation type
        by_frag_type: Dict[str, Dict[str, Any]] = {}
        for ft in set(per_spectrum_frag_type_list):
            mask = [f == ft for f in per_spectrum_frag_type_list]
            ft_match = [v for v, m in zip(per_spectrum_match_rate, mask) if m]
            ft_correct = [v for v, m in zip(per_spectrum_correct_rate, mask) if m]
            by_frag_type[ft] = {
                "match_rate": _dist(ft_match),
                "correct_rate": _dist(ft_correct),
                "n_spectra": len(ft_match),
            }

        # Stratify by charge
        by_charge: Dict[str, Dict[str, Any]] = {}
        for z in sorted(gap_df["charge"].unique()):
            z_df = gap_df[gap_df["charge"] == z]
            n_z = len(z_df)
            by_charge[str(int(z))] = {
                "n_gaps": n_z,
                "match_rate": float(z_df["is_valid"].mean()) if n_z > 0 else 0.0,
                "correct_rate": float(z_df["is_correct"].mean()) if n_z > 0 else 0.0,
            }

        # Stratify by m/z range
        by_mz_range: Dict[str, Dict[str, Any]] = {}
        for rng in gap_df["mz_range"].unique():
            r_df = gap_df[gap_df["mz_range"] == rng]
            n_r = len(r_df)
            by_mz_range[rng] = {
                "n_gaps": n_r,
                "match_rate": float(r_df["is_valid"].mean()) if n_r > 0 else 0.0,
                "correct_rate": float(r_df["is_correct"].mean()) if n_r > 0 else 0.0,
            }

        # Stratify by ion type
        by_ion_type: Dict[str, Dict[str, Any]] = {}
        for it in gap_df["ion_type"].unique():
            it_df = gap_df[gap_df["ion_type"] == it]
            n_it = len(it_df)
            by_ion_type[it] = {
                "n_gaps": n_it,
                "match_rate": float(it_df["is_valid"].mean()) if n_it > 0 else 0.0,
                "correct_rate": float(it_df["is_correct"].mean()) if n_it > 0 else 0.0,
            }

        analysis: Dict[str, Any] = {
            "n_spectra": len(per_spectrum_match_rate),
            "n_gaps": n_total,
            "n_valid": n_valid_total,
            "n_correct": n_correct_total,
            "overall_match_rate": n_valid_total / max(n_total, 1),
            "overall_correct_rate": n_correct_total / max(n_total, 1),
            "ppm_tolerance": ppm_tol,
            "per_spectrum_match_rate": _dist(per_spectrum_match_rate),
            "per_spectrum_correct_rate": _dist(per_spectrum_correct_rate),
            "per_spectrum_two_sided_rate": _dist(per_spectrum_two_sided_rate),
            "match_error_ppm": _dist(match_errors_ppm),
            "ambiguity_distribution": ambiguity_dist,
            "by_frag_type": by_frag_type,
            "by_charge": by_charge,
            "by_mz_range": by_mz_range,
            "by_ion_type": by_ion_type,
            # Internal arrays for visualization (prefixed with _)
            "_per_spectrum_match_rate": per_spectrum_match_rate,
            "_per_spectrum_correct_rate": per_spectrum_correct_rate,
            "_per_spectrum_frag_type": per_spectrum_frag_type_list,
            "_per_spectrum_two_sided_rate": per_spectrum_two_sided_rate,
            "_match_errors_ppm": match_errors_ppm,
            "_gap_df": gap_df,
        }

        self.results["mass_gap_analysis"] = analysis
        logger.info(
            f"Mass gap analysis: {n_total:,d} gaps, "
            f"{n_valid_total:,d} valid ({analysis['overall_match_rate']*100:.1f}%), "
            f"{n_correct_total:,d} correct ({analysis['overall_correct_rate']*100:.1f}%)"
        )
        return analysis

    # =========================================================================
    # Quality Gate Analysis
    # =========================================================================

    def analyze_quality_gate(self) -> Dict[str, Any]:
        """Analyse which spectra fail the backbone-coverage quality gate.

        Uses per-spectrum backbone coverage and fragment group counts from
        the fragment group analysis.  A spectrum is rejected when
        ``backbone_coverage < min_backbone_coverage`` **or**
        ``n_fragment_groups < min_fragment_groups``.

        Produces rejection-rate breakdowns by metadata dimension, diagnostic
        comparisons between populations above/below the threshold,
        sequence-length vs rejection curves, and a cross-tabulation heatmap.
        """
        # The gate is intrinsically a full-population diagnostic. Read from
        # the cached full-pop FGA so this method keeps working even after
        # the visible `fragment_group_analysis` slot is replaced by the
        # gated-pop run in Phase C.
        fga = self.results.get(
            "_full_fragment_group_analysis",
            self.results.get("fragment_group_analysis", {}),
        )
        coverage_arr = np.array(fga.get("_per_spectrum_coverage", []))
        n_groups_arr = np.array(fga.get("_per_spectrum_n_groups", []))

        if len(coverage_arr) == 0:
            logger.debug("No fragment group data for quality gate analysis")
            return {}

        # The fragment group analysis filters with the same predicate:
        #   sequence_available AND feature_types AND theo_annotations
        # We must use the *same* filter to align array indices.
        per_spectrum = self.results.get("per_spectrum", [])
        valid = [
            r
            for r in per_spectrum
            if r.get("sequence_available", False)
            and r.get("feature_types")
            and r.get("theo_annotations")
        ]

        if len(valid) != len(coverage_arr):
            logger.warning(
                f"Quality gate: valid spectra ({len(valid)}) != coverage array "
                f"({len(coverage_arr)}); skipping"
            )
            return {}

        # Failure masks
        fail_coverage = coverage_arr < self.min_backbone_coverage
        fail_groups = n_groups_arr < self.min_fragment_groups
        below_mask = fail_coverage | fail_groups

        n_below = int(below_mask.sum())
        n_above = len(valid) - n_below
        overall_rejection = n_below / max(len(valid), 1)

        # Failure mode breakdown
        n_fail_both = int((fail_coverage & fail_groups).sum())
        n_fail_coverage_only = int((fail_coverage & ~fail_groups).sum())
        n_fail_groups_only = int((~fail_coverage & fail_groups).sum())

        # A. Rejection rate by dimension
        rejection_by_dimension = self._quality_gate_rejection_by_dimension(
            valid, below_mask, overall_rejection
        )

        # B. Diagnostic comparison (below vs above)
        diagnostic_comparison = self._quality_gate_diagnostic_comparison(
            valid, below_mask, coverage_arr, n_groups_arr
        )

        # C. Sequence length vs rejection rate
        seq_len_analysis = self._quality_gate_seq_len_analysis(
            valid, coverage_arr, below_mask
        )

        # D. Cross-tabulation heatmap data
        cross_tab = self._quality_gate_cross_tabulation(valid, below_mask)

        analysis: Dict[str, Any] = {
            "min_backbone_coverage": self.min_backbone_coverage,
            "min_fragment_groups": self.min_fragment_groups,
            "n_total": len(valid),
            "n_below": n_below,
            "n_above": n_above,
            "n_fail_coverage_only": n_fail_coverage_only,
            "n_fail_groups_only": n_fail_groups_only,
            "n_fail_both": n_fail_both,
            "overall_rejection_rate": overall_rejection,
            "rejection_by_dimension": rejection_by_dimension,
            "diagnostic_comparison": diagnostic_comparison,
            "seq_len_analysis": seq_len_analysis,
            "cross_tabulation": cross_tab,
            # Internal arrays for visualization (not serialised)
            "_coverage_arr": coverage_arr,
            "_n_groups_arr": n_groups_arr,
            "_below_mask": below_mask,
            "_fail_coverage": fail_coverage,
            "_fail_groups": fail_groups,
        }

        self.results["quality_gate_analysis"] = analysis
        logger.debug(
            f"Quality gate: rejected {n_below:,d}/{len(valid):,d} "
            f"({overall_rejection*100:.1f}%)"
        )
        return analysis

    # -- Quality gate helpers --------------------------------------------------

    def _quality_gate_rejection_by_dimension(
        self,
        valid: List[Dict],
        below_mask: np.ndarray,
        overall_rejection: float,
    ) -> Dict[str, Dict[str, Any]]:
        """Compute rejection rate per metadata group for several dimensions."""
        dimensions = [
            "frag_type",
            "search_project",
            "search_instrument",
            "precursor_charge",
        ]
        result: Dict[str, Dict[str, Any]] = {}

        for dim in dimensions:
            groups: Dict[str, List[bool]] = {}
            for i, r in enumerate(valid):
                if dim == "precursor_charge":
                    val = str(r.get("precursor_charge", "unknown"))
                else:
                    meta = r.get("_metadata", {})
                    val = str(meta.get(dim, "unknown"))
                groups.setdefault(val, []).append(bool(below_mask[i]))

            dim_result: Dict[str, Any] = {}
            for g, vals in sorted(
                groups.items(), key=lambda x: -sum(x[1]) / max(len(x[1]), 1)
            ):
                n_total = len(vals)
                n_below = sum(vals)
                rate = n_below / max(n_total, 1)
                rr = rate / max(overall_rejection, 1e-9)
                dim_result[g] = {
                    "n_total": n_total,
                    "n_below": n_below,
                    "rejection_rate": rate,
                    "relative_risk": rr,
                }

            result[dim] = dim_result

        return result

    def _quality_gate_diagnostic_comparison(
        self,
        valid: List[Dict],
        below_mask: np.ndarray,
        coverage_arr: np.ndarray,
        n_groups_arr: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare feature distributions between below vs above populations."""
        # Features: (name, extractor(record, index) -> value)
        features: List[Tuple[str, Any]] = [
            ("backbone_coverage", lambda r, i: float(coverage_arr[i])),
            ("n_fragment_groups", lambda r, i: float(n_groups_arr[i])),
            (
                "annotated_fraction",
                lambda r, i: r.get("annotated_fraction"),
            ),
            ("frac_intensity", lambda r, i: r.get("frac_intensity")),
            (
                "n_valid_peaks",
                lambda r, i: r.get("_spectrum_stats", {}).get("n_valid_peaks"),
            ),
            (
                "sequence_length",
                lambda r, i: (
                    len(r["clean_sequence"]) if r.get("clean_sequence") else None
                ),
            ),
            ("n_matched", lambda r, i: r.get("n_matched")),
        ]

        def _stats(arr: np.ndarray) -> Dict[str, float]:
            return {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "q25": float(np.percentile(arr, 25)),
                "q75": float(np.percentile(arr, 75)),
            }

        result: Dict[str, Dict[str, Any]] = {}
        for fname, extractor in features:
            vals_below: List[float] = []
            vals_above: List[float] = []
            for i, r in enumerate(valid):
                v = extractor(r, i)
                if v is not None and np.isfinite(v):
                    if below_mask[i]:
                        vals_below.append(float(v))
                    else:
                        vals_above.append(float(v))

            if not vals_below or not vals_above:
                continue

            arr_b = np.array(vals_below)
            arr_a = np.array(vals_above)

            entry: Dict[str, Any] = {
                "below": _stats(arr_b),
                "above": _stats(arr_a),
                "n_below": len(arr_b),
                "n_above": len(arr_a),
            }

            # KS test + Cohen's d (graceful fallback)
            try:
                from scipy.stats import ks_2samp

                stat, pval = ks_2samp(arr_b, arr_a)
                entry["ks_statistic"] = float(stat)
                entry["ks_pvalue"] = float(pval)
            except (ImportError, ValueError):
                pass

            pooled_std = np.sqrt(
                (arr_b.var() * len(arr_b) + arr_a.var() * len(arr_a))
                / max(len(arr_b) + len(arr_a), 1)
            )
            if pooled_std > 0:
                entry["cohens_d"] = float(
                    (arr_a.mean() - arr_b.mean()) / pooled_std
                )

            result[fname] = entry

        return result

    def _quality_gate_seq_len_analysis(
        self,
        valid: List[Dict],
        coverage_arr: np.ndarray,
        below_mask: np.ndarray,
    ) -> Dict[str, Any]:
        """Rejection rate by sequence length bucket + Spearman correlation."""
        seq_lens = np.array(
            [
                len(r["clean_sequence"]) if r.get("clean_sequence") else 0
                for r in valid
            ]
        )

        edges = [0] + self.quality_gate_seq_len_bins + [999]
        bins: List[Dict[str, Any]] = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (seq_lens >= lo) & (seq_lens < hi)
            n_total = int(mask.sum())
            if n_total == 0:
                continue
            n_below = int((mask & below_mask).sum())
            mean_coverage = float(coverage_arr[mask].mean())
            bins.append(
                {
                    "bin": f"{lo}-{hi}",
                    "n_total": n_total,
                    "n_below": n_below,
                    "rejection_rate": n_below / max(n_total, 1),
                    "mean_backbone_coverage": mean_coverage,
                }
            )

        # Spearman correlation: seq_len vs backbone_coverage
        spearman_rho = None
        spearman_pval = None
        nonzero = seq_lens > 0
        if nonzero.sum() > 2:
            try:
                from scipy.stats import spearmanr

                rho, pval = spearmanr(seq_lens[nonzero], coverage_arr[nonzero])
                spearman_rho = float(rho)
                spearman_pval = float(pval)
            except (ImportError, ValueError):
                pass

        return {
            "bins": bins,
            "spearman_rho": spearman_rho,
            "spearman_pval": spearman_pval,
        }

    def _quality_gate_cross_tabulation(
        self,
        valid: List[Dict],
        below_mask: np.ndarray,
    ) -> Dict[str, Any]:
        """Rejection rate cross-tab: frag_type x precursor_charge."""
        counts: Dict[str, Dict[str, List[bool]]] = {}
        for i, r in enumerate(valid):
            frag = str(r.get("_metadata", {}).get("frag_type", "unknown"))
            charge = str(r.get("precursor_charge", "?"))
            counts.setdefault(frag, {}).setdefault(charge, []).append(
                bool(below_mask[i])
            )

        heatmap: Dict[str, Dict[str, Any]] = {}
        for frag, charges in sorted(counts.items()):
            heatmap[frag] = {}
            for charge, vals in sorted(charges.items()):
                n = len(vals)
                nb = sum(vals)
                heatmap[frag][charge] = {
                    "n_total": n,
                    "n_below": nb,
                    "rejection_rate": nb / max(n, 1),
                }

        return {"frag_type_x_charge": heatmap}

    # =========================================================================
    # Output
    # =========================================================================

    def save_results(self) -> None:
        """Save analysis results to JSON and CSV files."""
        logger.debug("Saving theoretical analysis results...")
        if not self.results:
            logger.warning("No results to save")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        def convert_for_json(obj):
            """Recursively convert numpy types for JSON serialization.

            Drops any dict entries whose key starts with ``_`` — those are
            reserved for per-analyser internal state (raw per-peak arrays,
            per-spectrum masks, etc.) that can be hundreds of MB at scale
            and have previously blown up the JSON writer mid-dump, leaving
            a truncated file on disk (see signal_composition's
            ``_annotated_intensities`` / ``_unannotated_intensities``).
            """
            if isinstance(obj, dict):
                return {
                    k: convert_for_json(v)
                    for k, v in obj.items()
                    if not (isinstance(k, str) and k.startswith("_"))
                }
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Summary JSON
        summary_data = {
            "overall_stats": convert_for_json(
                self.results.get("overall_stats", {})
            ),
            "coverage_stats": convert_for_json(
                self.results.get("coverage_stats", {})
            ),
            "signal_composition": convert_for_json(
                self.results.get("signal_composition", {})
            ),
            "mass_error_summary": convert_for_json(
                self.results.get("mass_error_summary", {})
            ),
            "ion_type_performance": convert_for_json(
                self.results.get("ion_type_performance", {})
            ),
            "analysis_config": {
                "ppm_tolerance": float(self.ppm_tol),
                "use_conditional_annotation": bool(self.use_conditional_annotation),
                "add_losses": bool(self.add_losses),
                "add_isotopes": bool(self.add_isotopes),
                "theoretical_engine": str(self.theoretical_engine),
                "min_backbone_coverage": float(self.min_backbone_coverage),
                "min_fragment_groups": int(self.min_fragment_groups),
                "apply_quality_gate_filter": bool(self.apply_quality_gate_filter),
            },
            # Always-on, top-level so JSON consumers can immediately tell
            # whether the rest of this file describes the gated population.
            "filter_metadata": convert_for_json(
                self.results.get("filter_metadata", {})
            ),
        }
        # Add fragment group analysis (exclude internal arrays)
        fga = self.results.get("fragment_group_analysis", {})
        if fga:
            summary_data["fragment_group_analysis"] = convert_for_json(
                {k: v for k, v in fga.items() if not k.startswith("_")}
            )
        # Add quality gate summary (exclude internal arrays)
        qga = self.results.get("quality_gate_analysis", {})
        if qga:
            summary_data["quality_gate_analysis"] = convert_for_json(
                {k: v for k, v in qga.items() if not k.startswith("_")}
            )
        # Add modification analysis (exclude internal arrays)
        mod_analysis = self.results.get("modification_analysis", {})
        if mod_analysis:
            summary_data["modification_analysis"] = convert_for_json(
                {k: v for k, v in mod_analysis.items() if not k.startswith("_")}
            )
        # Add complementary pair analysis (exclude internal arrays)
        cpa = self.results.get("complementary_pair_analysis", {})
        if cpa:
            summary_data["complementary_pair_analysis"] = convert_for_json(
                {k: v for k, v in cpa.items() if not k.startswith("_")}
            )
        # Add mass gap analysis (exclude internal arrays and DataFrames)
        mga = self.results.get("mass_gap_analysis", {})
        if mga:
            summary_data["mass_gap_analysis"] = convert_for_json(
                {
                    k: v for k, v in mga.items()
                    if not k.startswith("_") and not isinstance(v, pd.DataFrame)
                }
            )
        # Add blur sigma analysis (exclude raw distributions to keep JSON small)
        bsa = self.results.get("blur_sigma_analysis", {})
        if bsa:
            summary_data["blur_sigma_analysis"] = convert_for_json(
                {
                    k: v for k, v in bsa.items()
                    if k not in ("within_group_spreads", "isotope_only_spreads",
                                 "with_loss_spreads", "same_series_gaps", "all_series_gaps")
                }
            )
        # Atomic write: serialise to a temp file in the same directory
        # (same filesystem for rename to be atomic), fsync, then rename.
        # Previously a plain ``open(...).write`` left truncated files on
        # disk whenever the process was killed mid-dump (OOM, SIGINT,
        # disk-full) — consumers then hit JSONDecodeError without any
        # way to distinguish "analyser didn't run" from "write died
        # halfway". rename()-based replacement guarantees the file on
        # disk is either the previous run or the new complete one.
        summary_path = self.output_dir / "theoretical_analysis_summary.json"
        tmp_path = summary_path.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(summary_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, summary_path)
        except BaseException:
            # BaseException catches KeyboardInterrupt too — we still want
            # to clean up the partial tmp file before propagating.
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise
        logger.debug(f"Saved summary: {summary_path}")

        # Mass error CSV (save only essential columns; derivable columns omitted to reduce file size)
        if self.results.get("mass_error_df") is not None:
            df = self.results["mass_error_df"]
            save_cols = [c for c in [
                "theo_mz", "exp_mz", "signed_ppm",
                "ion_type", "charge", "frag_type", "feature_type",
            ] if c in df.columns]
            csv_path = self.output_dir / "mass_error_data.csv"
            df[save_cols].to_csv(csv_path, index=False, float_format="%.6f")
            logger.debug(f"Saved mass error data: {csv_path} ({len(df):,d} peaks, {len(save_cols)} columns)")

        # Unmatched theoretical ions CSV (omit derivable mz_range)
        if self.results.get("unmatched_theo_df") is not None:
            df = self.results["unmatched_theo_df"]
            save_cols = [c for c in ["theo_mz", "ion_type", "frag_type"] if c in df.columns]
            csv_path = self.output_dir / "unmatched_theoretical_ions.csv"
            df[save_cols].to_csv(csv_path, index=False, float_format="%.6f")
            logger.debug(f"Saved unmatched ions: {csv_path} ({len(df):,d} ions)")

        # Ion type statistics CSV
        ion_perf = self.results.get("ion_type_performance", {})
        if ion_perf:
            records = [
                {
                    "ion_type": it,
                    "n_matched": s["n_matched"],
                    "mean_ppm_error": s["mean_ppm_error"],
                    "std_ppm_error": s["std_ppm_error"],
                    "median_abs_ppm_error": s["median_abs_ppm_error"],
                }
                for it, s in ion_perf.items()
            ]
            if records:
                csv_path = self.output_dir / "ion_type_statistics.csv"
                pd.DataFrame(records).to_csv(
                    csv_path, index=False, float_format="%.6f"
                )
                logger.debug(f"Saved ion type statistics: {csv_path}")

        # Custom ion analysis JSON + CSV
        cia = self.results.get("custom_ion_analysis", {})
        if cia:
            # JSON (exclude raw arrays)
            cia_json = {
                k: convert_for_json(v)
                for k, v in cia.items()
                if not k.startswith("_")
            }
            json_path = self.output_dir / "custom_ion_analysis.json"
            with open(json_path, "w") as f:
                json.dump(cia_json, f, indent=2)
            logger.debug(f"Saved custom ion analysis: {json_path}")

            # CSV: per-group hit rates
            ghr = cia.get("group_hit_rates", {})
            if ghr:
                records = [
                    {
                        "group": g,
                        "category": self._classify_ion_category(g),
                        "n_found": d["n_found"],
                        "hit_rate": d["hit_rate"],
                    }
                    for g, d in ghr.items()
                ]
                csv_path = self.output_dir / "custom_ion_hit_rates.csv"
                pd.DataFrame(records).to_csv(
                    csv_path, index=False, float_format="%.6f"
                )
                logger.debug(f"Saved custom ion hit rates: {csv_path}")

        # Neutral loss analysis JSON + CSV
        nla = self.results.get("neutral_loss_analysis", {})
        if nla:
            nla_json = {
                k: convert_for_json(v)
                for k, v in nla.items()
                if not k.startswith("_")
            }
            json_path = self.output_dir / "neutral_loss_analysis.json"
            with open(json_path, "w") as f:
                json.dump(nla_json, f, indent=2)
            logger.debug(f"Saved neutral loss analysis: {json_path}")

            # CSV: per-loss-type prevalence
            prev = nla.get("prevalence", {})
            if prev:
                records = [
                    {
                        "loss_type": lt,
                        "total_count": d["total_count"],
                        "n_spectra_hit": d["n_spectra_hit"],
                        "spectra_hit_rate": d["spectra_hit_rate"],
                        "mean_per_spectrum": d["mean_per_spectrum"],
                    }
                    for lt, d in prev.items()
                ]
                csv_path = self.output_dir / "neutral_loss_prevalence.csv"
                pd.DataFrame(records).to_csv(
                    csv_path, index=False, float_format="%.6f"
                )
                logger.debug(f"Saved neutral loss prevalence: {csv_path}")

        # Fragment group analysis JSON + CSV
        fga = self.results.get("fragment_group_analysis", {})
        if fga:
            # JSON (exclude internal arrays prefixed with _)
            fga_json = {
                k: convert_for_json(v)
                for k, v in fga.items()
                if not k.startswith("_")
            }
            json_path = self.output_dir / "fragment_group_analysis.json"
            with open(json_path, "w") as f:
                json.dump(fga_json, f, indent=2)
            logger.debug(f"Saved fragment group analysis: {json_path}")

            # CSV: per-frag_type summary
            by_ft = fga.get("by_frag_type", {})
            if by_ft:
                records = []
                for ft, stats in by_ft.items():
                    fc = stats["fragment_count"]
                    bc = stats["backbone_coverage"]
                    cm = stats["charge_multiplicity"]
                    row: Dict[str, Any] = {
                        "frag_type": ft,
                        "mean_fragment_count": fc["mean"],
                        "median_fragment_count": fc["median"],
                        "std_fragment_count": fc["std"],
                        "mean_backbone_coverage": bc["mean"],
                        "median_backbone_coverage": bc["median"],
                        "std_backbone_coverage": bc["std"],
                        "pct_1_charge": cm["1"] * 100,
                        "pct_2_charge": cm["2"] * 100,
                        "pct_3plus_charge": cm["3+"] * 100,
                    }
                    # Per-ion-series columns. All six base ion series are
                    # emitted so downstream consumers can read the pair
                    # relevant to each frag_type (b/y for collisional,
                    # c/z for ETD/ECD) without needing to parse the label.
                    for s in ("a", "b", "c", "x", "y", "z"):
                        s_stats = stats.get("by_ion_series", {}).get(s, {})
                        s_fc = s_stats.get("fragment_count", {})
                        s_cm = s_stats.get("charge_multiplicity", {})
                        row[f"mean_{s}_count"] = s_fc.get("mean", 0)
                        row[f"median_{s}_count"] = s_fc.get("median", 0)
                        row[f"pct_{s}_2plus_charge"] = (
                            s_cm.get("2", 0) + s_cm.get("3+", 0)
                        ) * 100
                    records.append(row)
                csv_path = self.output_dir / "fragment_group_summary.csv"
                pd.DataFrame(records).to_csv(
                    csv_path, index=False, float_format="%.6f"
                )
                logger.debug(f"Saved fragment group summary: {csv_path}")

        # Quality gate analysis JSON + CSVs
        qga = self.results.get("quality_gate_analysis", {})
        if qga:
            # JSON (exclude internal arrays)
            qga_json = {
                k: convert_for_json(v)
                for k, v in qga.items()
                if not k.startswith("_")
            }
            json_path = self.output_dir / "quality_gate_analysis.json"
            with open(json_path, "w") as f:
                json.dump(qga_json, f, indent=2)
            logger.debug(f"Saved quality gate analysis: {json_path}")

            # Per-dimension rejection CSVs
            for dim, groups in qga.get("rejection_by_dimension", {}).items():
                records = [
                    {
                        "group": g,
                        "n_total": s["n_total"],
                        "n_below": s["n_below"],
                        "rejection_rate": s["rejection_rate"],
                        "relative_risk": s["relative_risk"],
                    }
                    for g, s in groups.items()
                ]
                if records:
                    csv_path = self.output_dir / f"quality_gate_rejection_by_{dim}.csv"
                    pd.DataFrame(records).to_csv(
                        csv_path, index=False, float_format="%.6f"
                    )
                    logger.debug(f"Saved: {csv_path}")

        logger.info(f"Theoretical analysis results saved to {self.output_dir}")

    def print_summary(self) -> None:
        """Print analysis summary to console."""
        if not self.results:
            logger.warning("No results to summarize")
            return

        logger.info("=" * 70)
        logger.info("THEORETICAL ANALYSIS SUMMARY")
        logger.info("=" * 70)

        # Surface filter state up-front so the reader knows whether the
        # numbers below describe the full or the gated population.
        fm = self.results.get("filter_metadata", {})
        if fm:
            logger.info("")
            logger.info("Quality-gate filter:")
            if fm.get("filter_applied"):
                logger.info(
                    f"  APPLIED  (min_backbone_coverage>={fm['min_backbone_coverage']}, "
                    f"min_fragment_groups>={fm['min_fragment_groups']})"
                )
                logger.info(
                    f"  Passed:  {fm['n_passed_spectra']:,d} / "
                    f"{fm['n_total_spectra']:,d} "
                    f"({(1 - fm['rejection_rate']) * 100:.1f}%)"
                )
                logger.info(
                    f"  Rejected:{fm['n_rejected_spectra']:,d} "
                    f"({fm['rejection_rate'] * 100:.1f}%)"
                )
                logger.info(
                    "  -> Phase-C analyses below are computed on the gated population."
                )
            else:
                if fm.get("filter_requested") and not fm.get("gate_enabled"):
                    reason = "quality gate disabled"
                elif fm.get("filter_requested"):
                    reason = "no gate output available"
                else:
                    reason = "filter disabled in config"
                logger.info(f"  NOT APPLIED ({reason})")
                logger.info(
                    "  -> Analyses below run on the full sequence-available population."
                )

        overall = self.results.get("overall_stats", {})
        if overall:
            logger.info(f"  Spectra analyzed:       {overall['n_spectra_analyzed']:,d}")
            logger.info(
                f"  Avg match rate:         "
                f"{overall['avg_match_rate']*100:.1f}% "
                f"+/- {overall['std_match_rate']*100:.1f}%"
            )
            logger.info(
                f"  Avg intensity coverage: "
                f"{overall['avg_frac_intensity']*100:.1f}% "
                f"+/- {overall['std_frac_intensity']*100:.1f}%"
            )
            logger.info(
                f"  Avg annotated fraction: "
                f"{overall['avg_annotated_fraction']*100:.1f}% "
                f"+/- {overall['std_annotated_fraction']*100:.1f}%"
            )
            logger.info(
                f"  Total annotated peaks:  {overall['total_annotated_peaks']:,d}"
            )
            logger.info(
                f"  Total unannotated:      {overall['total_unannotated_peaks']:,d}"
            )

        # Signal composition
        sig = self.results.get("signal_composition", {})
        if sig:
            logger.info("")
            logger.info("Signal Composition:")
            logger.info(
                f"  Informative (fragment ions): "
                f"{sig['informative_fraction']*100:.1f}%"
            )
            logger.info(
                f"  Unannotated (noise):         "
                f"{sig['noise_fraction']*100:.1f}%"
            )
            cats = sig.get("category_fractions", {})
            for cat in [
                "fragment_base",
                "fragment_loss",
                "fragment_isotope",
                "precursor",
            ]:
                if cat in cats:
                    logger.info(f"    {cat:22s} {cats[cat]*100:5.1f}%")

        # Mass error
        me = self.results.get("mass_error_summary", {})
        if me:
            logger.info("")
            logger.info("Mass Error:")
            logger.info(f"  Systematic bias: {me['systematic_bias_ppm']:.2f} ppm")
            percs = me.get("percentiles", {})
            logger.info(
                f"  |PPM| P50={percs.get('p50', 0):.2f}  "
                f"P90={percs.get('p90', 0):.2f}  "
                f"P95={percs.get('p95', 0):.2f}  "
                f"P99={percs.get('p99', 0):.2f}"
            )

        # Coverage
        cov = self.results.get("coverage_stats", {})
        if cov:
            logger.info("")
            logger.info(
                f"Ion Coverage: {cov['overall_coverage_rate']*100:.1f}% "
                f"({cov['total_matched']:,d}/{cov['total_theoretical_ions']:,d})"
            )

        # Custom ion analysis
        cia = self.results.get("custom_ion_analysis", {})
        if cia:
            logger.info("")
            logger.info("Custom Ion Detection:")
            chr_ = cia.get("category_hit_rates", {})
            for cat in ["glycan", "immonium", "TMT", "iTRAQ"]:
                d = chr_.get(cat, {})
                rate = d.get("hit_rate", 0)
                if rate > 0:
                    logger.info(
                        f"  {cat:<12s} {rate*100:5.1f}%  "
                        f"({d['n_found']:,d}/{cia['n_spectra']:,d})"
                    )
            # Top individual groups
            ghr = cia.get("group_hit_rates", {})
            top_groups = sorted(
                ghr.items(), key=lambda x: x[1]["hit_rate"], reverse=True
            )[:5]
            if top_groups and top_groups[0][1]["hit_rate"] > 0:
                logger.info("  Top groups:")
                for g, d in top_groups:
                    if d["hit_rate"] > 0:
                        logger.info(
                            f"    {g:<35s} {d['hit_rate']*100:5.1f}%"
                        )
            cl = cia.get("coverage_lift", {})
            if cl:
                logger.info(
                    f"  Coverage lift: "
                    f"{cl['total_explained_by_custom_ions']:,d} / "
                    f"{cl['total_unannotated_peaks']:,d} "
                    f"unannotated peaks now explained "
                    f"({cl['overall_lift_fraction']*100:.2f}%)"
                )

        # Neutral loss analysis
        nla = self.results.get("neutral_loss_analysis", {})
        if nla:
            logger.info("")
            logger.info("Neutral Loss Analysis:")
            prev = nla.get("prevalence", {})
            for lt, p in prev.items():
                logger.info(
                    f"  {lt:<8s} {p['spectra_hit_rate']*100:5.1f}% spectra  "
                    f"({p['total_count']:,d} total, "
                    f"{p['mean_per_spectrum']:.1f}/spectrum)"
                )
            irs = nla.get("intensity_ratio_stats", {})
            if irs:
                logger.info("  Intensity ratios (loss/parent):")
                for lt, stats in irs.items():
                    logger.info(
                        f"    {lt:<8s} median={stats['median']:.3f}  "
                        f"mean={stats['mean']:.3f}  "
                        f"IQR=[{stats['q25']:.3f}, {stats['q75']:.3f}]"
                    )

        # Fragment group analysis
        fga = self.results.get("fragment_group_analysis", {})
        if fga:
            logger.info("")
            logger.info("Fragment Ion Groups:")
            fc = fga["overall"]["fragment_count"]
            bc = fga["overall"]["backbone_coverage"]
            cm = fga["overall"]["charge_multiplicity"]
            multi_pct = (cm["2"] + cm["3+"]) * 100
            logger.info(
                f"  Unique fragments/spectrum: "
                f"{fc['mean']:.1f} +/- {fc['std']:.1f} "
                f"(median: {fc['median']:.0f})"
            )
            by_series = fga["overall"].get("by_ion_series", {})
            # Report any series with non-zero activity so ETD/ECD runs
            # (which emit c/z rather than b/y) still get a summary line.
            for s in ("a", "b", "c", "x", "y", "z"):
                sfc = by_series.get(s, {}).get("fragment_count", {})
                if sfc and sfc.get("mean", 0) > 0:
                    logger.info(
                        f"    {s}-ions: {sfc['mean']:.1f} +/- {sfc['std']:.1f} "
                        f"(median: {sfc['median']:.0f})"
                    )
            logger.info(
                f"  Backbone coverage:         "
                f"{bc['mean']*100:.1f}% +/- {bc['std']*100:.1f}%"
            )
            logger.info(
                f"  Multi-charge groups:       "
                f"{multi_pct:.1f}% of groups at 2+ charge states"
            )
            by_ft = fga.get("by_frag_type", {})
            if by_ft:
                parts = []
                for ft, stats in sorted(by_ft.items()):
                    ft_series = stats.get("by_ion_series", {})
                    # Show the N/C pair native to this fragmentation method
                    # (c/z for ETD/ECD, b/y otherwise) — reporting a fixed
                    # b/y pair shows 0.0/0.0 for electron-driven runs and
                    # misrepresents fragmentation quality.
                    n_key, c_key = self._primary_pair_for_mode(ft)
                    n_m = ft_series.get(n_key, {}).get(
                        "fragment_count", {}
                    ).get("mean", 0)
                    c_m = ft_series.get(c_key, {}).get(
                        "fragment_count", {}
                    ).get("mean", 0)
                    parts.append(
                        f"{ft}: {stats['fragment_count']['mean']:.1f} "
                        f"({n_key}={n_m:.1f}, {c_key}={c_m:.1f}), "
                        f"{stats['backbone_coverage']['mean']*100:.1f}% cov"
                    )
                logger.info(f"  By frag_type:  {' | '.join(parts)}")

            # Ladder analysis
            ladder = fga.get("ladder_analysis", {})
            if ladder:
                logger.info("")
                logger.info("Ion Ladder Analysis (consecutive fragment runs):")
                # Report any ion series with activity so ETD runs (c/z)
                # get a log line rather than being silently skipped.
                for s in ("a", "b", "c", "x", "y", "z"):
                    sl = ladder.get(s, {})
                    ml_check = sl.get("max_ladder_length", {})
                    if not ml_check or ml_check.get("mean", 0) == 0:
                        continue
                    ml = sl.get("max_ladder_length", {})
                    al = sl.get("all_ladder_lengths", {})
                    gs = sl.get("group_mz_span", {})
                    if ml:
                        logger.info(
                            f"  {s}-ion max ladder: "
                            f"{ml['mean']:.1f} +/- {ml['std']:.1f} "
                            f"(median: {ml['median']:.0f})"
                        )
                        logger.info(
                            f"    All ladders: median={al.get('median', 0):.1f}, "
                            f"p75={al.get('p75', 0):.1f}, "
                            f"p90={al.get('p90', 0):.1f} "
                            f"(n={al.get('total_count', 0):,d})"
                        )
                        if gs:
                            logger.info(
                                f"    Group m/z span: "
                                f"median={gs.get('median', 0):.0f}, "
                                f"p75={gs.get('p75', 0):.0f}, "
                                f"p90={gs.get('p90', 0):.0f} "
                                f"peaks (incl. noise)"
                            )
                rec = ladder.get("combined", {})
                if rec:
                    logger.info(
                        f"  Span recommendation: "
                        f"span_min={rec['suggested_span_min']}, "
                        f"span_max={rec['suggested_span_max']} "
                        f"({rec['reasoning']})"
                    )

        # Quality gate
        qga = self.results.get("quality_gate_analysis", {})
        if qga:
            logger.info("")
            logger.info("Quality Gate Analysis:")
            logger.info(
                f"  Thresholds:       coverage >= {qga['min_backbone_coverage']}, "
                f"groups >= {qga['min_fragment_groups']}"
            )
            logger.info(
                f"  Rejected:         {qga['n_below']:,d} / {qga['n_total']:,d} "
                f"({qga['overall_rejection_rate']*100:.1f}%)"
            )
            logger.info(
                f"  Fail breakdown:   coverage only: {qga['n_fail_coverage_only']:,d}, "
                f"groups only: {qga['n_fail_groups_only']:,d}, "
                f"both: {qga['n_fail_both']:,d}"
            )
            # Highest-rejection dimension
            dim_data = qga.get("rejection_by_dimension", {})
            worst_dim = None
            worst_rate = 0.0
            for dim_name, groups in dim_data.items():
                for g, stats in groups.items():
                    if (
                        stats["n_total"] >= 10
                        and stats["rejection_rate"] > worst_rate
                    ):
                        worst_rate = stats["rejection_rate"]
                        worst_dim = f"{dim_name}/{g}"
            if worst_dim:
                logger.info(
                    f"  Highest rejection: {worst_dim} "
                    f"({worst_rate*100:.1f}%)"
                )

        # Modification analysis
        mod = self.results.get("modification_analysis", {})
        if mod:
            logger.info("")
            logger.info("Modification Analysis:")
            prev = mod.get("prevalence", {})
            logger.info(
                f"  Modified spectra:       "
                f"{prev.get('n_modified', 0):,d} / {prev.get('n_total', 0):,d} "
                f"({prev.get('modified_fraction', 0)*100:.1f}%)"
            )
            per_type = prev.get("per_type", {})
            if per_type:
                top_types = sorted(
                    per_type.items(),
                    key=lambda x: x[1].get("n_spectra", 0),
                    reverse=True,
                )[:8]
                for mod_name, stats in top_types:
                    logger.info(
                        f"    {mod_name:<25s} "
                        f"{stats['n_spectra']:>6,d} spectra "
                        f"({stats['spectra_fraction']*100:5.1f}%)"
                    )
            # Per-type match rate deltas
            matching_quality = mod.get("matching_quality", {})
            unmod_mean = matching_quality.get("Unmodified", {}).get(
                "match_rate", {}
            ).get("mean", 0)
            if matching_quality and unmod_mean > 0:
                logger.info("  Match rate vs Unmodified baseline:")
                for mod_name, metrics in sorted(matching_quality.items()):
                    if mod_name == "Unmodified":
                        continue
                    mr = metrics.get("match_rate", {})
                    delta = mr.get("mean", 0) - unmod_mean
                    logger.info(
                        f"    {mod_name:<25s} "
                        f"{mr.get('mean', 0)*100:5.1f}% "
                        f"(delta: {delta*100:+.1f}%)"
                    )
            # Diagnostic counters
            diag = mod.get("diagnostics", {})
            n_stripped = diag.get("n_modifications_stripped", 0)
            n_prec_warn = diag.get("n_precursor_mass_warnings", 0)
            n_iso = diag.get("n_isotope_corrected", 0)
            if any((n_stripped, n_prec_warn, n_iso)):
                logger.info(
                    f"  Precursor validation:  "
                    f"isotope-corrected={n_iso:,d}, "
                    f"warnings={n_prec_warn:,d}"
                )
                if n_stripped > 0:
                    logger.info(
                        f"  Modifications stripped (fallback): {n_stripped:,d}"
                    )

        # Complementary pair analysis
        cpa = self.results.get("complementary_pair_analysis", {})
        if cpa:
            logger.info("")
            logger.info("Complementary b/y Pair Analysis:")
            pf = cpa.get("pair_fraction", {})
            logger.info(
                f"  Pair fraction:    "
                f"{pf.get('mean', 0)*100:.1f}% +/- {pf.get('std', 0)*100:.1f}% "
                f"(median: {pf.get('median', 0)*100:.1f}%)"
            )
            logger.info(
                f"  Total pairs:      {cpa.get('total_pairs', 0):,d} "
                f"across {cpa.get('n_spectra', 0):,d} spectra"
            )
            dev = cpa.get("deviation_da", {})
            abs_dev = cpa.get("abs_deviation_da", {})
            logger.info(
                f"  Sum deviation:    median={dev.get('median', 0):.5f} Da "
                f"(|dev| median={abs_dev.get('median', 0):.5f} Da)"
            )
            by_ft = cpa.get("by_frag_type", {})
            if by_ft:
                parts = []
                for ft, stats in sorted(by_ft.items()):
                    fp = stats.get("pair_fraction", {})
                    parts.append(
                        f"{ft}: {fp.get('mean', 0)*100:.1f}% (n={stats.get('n_spectra', 0)})"
                    )
                logger.info(f"  By frag_type:     {' | '.join(parts)}")

        # Mass gap analysis
        mga = self.results.get("mass_gap_analysis", {})
        if mga:
            logger.info("")
            logger.info("Mass Gap Analysis:")
            logger.info(
                f"  Valid AA match:   "
                f"{mga.get('overall_match_rate', 0)*100:.1f}% "
                f"({mga.get('n_valid', 0):,d}/{mga.get('n_gaps', 0):,d} gaps)"
            )
            logger.info(
                f"  Correct AA:       "
                f"{mga.get('overall_correct_rate', 0)*100:.1f}% "
                f"({mga.get('n_correct', 0):,d}/{mga.get('n_gaps', 0):,d} gaps)"
            )
            me = mga.get("match_error_ppm", {})
            logger.info(
                f"  Match error:      "
                f"median={me.get('median', 0):.2f} ppm "
                f"(IQR=[{me.get('q25', 0):.2f}, {me.get('q75', 0):.2f}])"
            )
            ts = mga.get("per_spectrum_two_sided_rate", {})
            logger.info(
                f"  Two-sided rate:   "
                f"{ts.get('mean', 0)*100:.1f}% +/- {ts.get('std', 0)*100:.1f}%"
            )
            ambig = mga.get("ambiguity_distribution", {})
            if ambig:
                total_a = sum(ambig.values())
                unambig = ambig.get(1, 0)
                logger.info(
                    f"  Unambiguous (1 AA): "
                    f"{unambig/max(total_a,1)*100:.1f}% of valid gaps"
                )
            by_ft = mga.get("by_frag_type", {})
            if by_ft:
                parts = []
                for ft, stats in sorted(by_ft.items()):
                    mr = stats.get("match_rate", {}).get("mean", 0)
                    cr = stats.get("correct_rate", {}).get("mean", 0)
                    parts.append(
                        f"{ft}: match={mr*100:.1f}%, correct={cr*100:.1f}%"
                    )
                logger.info(f"  By frag_type:     {' | '.join(parts)}")

        logger.info("=" * 70)

    # =========================================================================
    # Visualization
    # =========================================================================

    def generate_visualizations(self) -> None:
        """Generate all theoretical analysis visualizations."""
        logger.debug("Generating theoretical analysis visualizations...")
        if not self.results:
            logger.warning("No results available for visualization")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        mass_error_df = self.results.get("mass_error_df")
        has_mass_error = mass_error_df is not None and (
            not isinstance(mass_error_df, pd.DataFrame) or not mass_error_df.empty
        )

        figures = []
        if has_mass_error:
            figures.extend([
                ("annotation summary", self.visualize_annotation_summary),
                ("mass error analysis", self.visualize_mass_error_analysis),
                ("ion type summary", self.visualize_ion_type_summary),
            ])
        else:
            logger.warning("No mass error data -- skipping fragment-matching figures")

        sig_comp = self.results.get("signal_composition", {})
        if sig_comp.get("_annotated_intensities") or sig_comp.get(
            "_unannotated_intensities"
        ):
            figures.append(
                (
                    "annotated intensity comparison",
                    self.visualize_annotated_intensity_comparison,
                )
            )

        if self.results.get("custom_ion_analysis"):
            figures.append(
                ("custom ion analysis", self.visualize_custom_ion_analysis)
            )

        if self.results.get("neutral_loss_analysis"):
            figures.append(
                ("neutral loss analysis", self.visualize_neutral_loss_analysis)
            )

        if self.results.get("fragment_group_analysis"):
            figures.append(
                ("fragment group analysis", self.visualize_fragment_group_analysis)
            )
            if self.results["fragment_group_analysis"].get("ladder_analysis"):
                figures.append(
                    ("ladder length analysis", self.visualize_ladder_analysis)
                )

        if self.results.get("quality_gate_analysis"):
            figures.extend([
                ("quality gate overview", self.visualize_quality_gate_overview),
                ("quality gate diagnostics", self.visualize_quality_gate_diagnostics),
            ])

        if self.results.get("modification_analysis"):
            figures.append(
                ("modification analysis", self.visualize_modification_analysis)
            )

        if self.results.get("complementary_pair_analysis"):
            figures.append(
                ("complementary pair analysis", self.visualize_complementary_pair_analysis)
            )

        if self.results.get("mass_gap_analysis"):
            figures.append(
                ("mass gap analysis", self.visualize_mass_gap_analysis)
            )

        if self.results.get("blur_sigma_analysis"):
            figures.append(
                ("blur sigma analysis", self.visualize_blur_sigma_analysis)
            )

        for name, method in figures:
            try:
                method()
            except Exception as e:
                logger.error(f"Failed to generate {name}: {e}")

        # Stratified (per-frag_type) visualizations
        stratified = self.results.get("_stratified_analysers", {})
        for ft, sub_analyser in stratified.items():
            try:
                logger.debug(f"Generating stratified figures for frag_type={ft}...")
                sub_analyser.generate_visualizations()
            except Exception as e:
                logger.error(f"Failed stratified viz for {ft}: {e}")

        logger.info("Theoretical analysis visualizations complete")

    def visualize_annotation_summary(self) -> Figure:
        """Annotation coverage and signal composition (2x2 grid).

        Panels:
          A — Coverage by m/z range (bar chart)
          B — Signal composition by category (horizontal bar)
          C — Match rate distribution across spectra (histogram)
          D — Coverage & quality statistics (text box)
        """
        logger.debug("Generating annotation summary...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Annotation Coverage & Signal Composition",
            fontsize=16,
            fontweight="bold",
        )

        coverage_stats = self.results.get("coverage_stats", {})
        overall_stats = self.results.get("overall_stats", {})
        signal_comp = self.results.get("signal_composition", {})

        # --- Panel A: Coverage by m/z range ---
        ax = axes[0, 0]
        if coverage_stats.get("coverage_by_mz_range"):
            mz_ranges = list(coverage_stats["coverage_by_mz_range"].keys())
            rates = [
                coverage_stats["coverage_by_mz_range"][r]["coverage_rate"]
                for r in mz_ranges
            ]
            n_matched = [
                coverage_stats["coverage_by_mz_range"][r]["n_matched"]
                for r in mz_ranges
            ]
            n_total = [
                coverage_stats["coverage_by_mz_range"][r]["n_theoretical"]
                for r in mz_ranges
            ]

            x_pos = np.arange(len(mz_ranges))
            bars = ax.bar(x_pos, rates, color="steelblue", alpha=0.7)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(mz_ranges, rotation=45, ha="right")
            ax.set_ylabel("Coverage Rate")
            ax.set_title("A. Coverage by m/z Range")
            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", alpha=0.3)
            for bar, matched, total in zip(bars, n_matched, n_total):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{matched:,d}/{total:,d}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        else:
            ax.text(
                0.5, 0.5, "No coverage data",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_title("A. Coverage by m/z Range")

        # --- Panel B: Signal composition by category (grouped: count + intensity) ---
        ax = axes[0, 1]
        if signal_comp.get("category_fractions"):
            cats = signal_comp["category_fractions"]
            int_cats = signal_comp.get("category_intensity_fractions", {})
            cat_order = [
                "fragment_base",
                "fragment_loss",
                "fragment_isotope",
                "precursor",
                "other_annotated",
                "unannotated",
            ]
            cat_labels = [c for c in cat_order if c in cats and cats[c] > 0]
            count_values = [cats[c] * 100 for c in cat_labels]
            int_values = [int_cats.get(c, 0) * 100 for c in cat_labels]
            colors = [SIGNAL_CATEGORY_COLORS.get(c, "#999999") for c in cat_labels]

            bar_height = 0.35
            y_pos = np.arange(len(cat_labels))
            # Count fraction bars (solid)
            bars_count = ax.barh(
                y_pos - bar_height / 2, count_values, bar_height,
                color=colors, alpha=0.8, label="By count",
            )
            # Intensity fraction bars (hatched)
            bars_int = ax.barh(
                y_pos + bar_height / 2, int_values, bar_height,
                color=colors, alpha=0.5, hatch="//", label="By intensity",
            )
            ax.set_yticks(y_pos)
            ax.set_yticklabels(cat_labels, fontsize=9)
            ax.set_xlabel("% of All Peaks")
            # The bars show the AGGREGATE fraction (summed over every peak
            # in the dataset) which by construction totals 100 %. Surface
            # the matching aggregate annotated-vs-unannotated split in the
            # title (so the numbers reconcile with the bars) and keep the
            # per-spectrum mean ± std on a second line so the reader sees
            # the spread across spectra. The two values differ because
            # peak counts per spectrum vary: aggregate weights long
            # spectra more, per-spectrum mean weights every spectrum
            # equally.
            agg_annot_count = sum(
                cats.get(c, 0.0) for c in cat_labels if c != "unannotated"
            ) * 100.0
            agg_annot_int = sum(
                int_cats.get(c, 0.0) for c in cat_labels if c != "unannotated"
            ) * 100.0
            ann_per = [
                float(r.get("annotated_fraction", 0))
                for r in self.results.get("per_spectrum", [])
                if r.get("sequence_available", False)
            ]
            int_per = [
                float(r.get("annotated_intensity_fraction", 0))
                for r in self.results.get("per_spectrum", [])
                if r.get("sequence_available", False)
            ]
            if ann_per:
                _a = np.asarray(ann_per)
                _i = np.asarray(int_per) if int_per else _a
                title = (
                    "B. Signal Composition (Count vs Intensity)\n"
                    f"aggregate annotated: count {agg_annot_count:.1f}%, "
                    f"intensity {agg_annot_int:.1f}% "
                    f"(sum to 100% with unannotated)\n"
                    f"per-spectrum annotated: count {_a.mean()*100:.1f}±{_a.std()*100:.1f}%, "
                    f"intensity {_i.mean()*100:.1f}±{_i.std()*100:.1f}%"
                )
            else:
                title = (
                    "B. Signal Composition (Count vs Intensity)\n"
                    f"aggregate annotated: count {agg_annot_count:.1f}%, "
                    f"intensity {agg_annot_int:.1f}%"
                )
            ax.set_title(title, fontsize=10)
            ax.grid(axis="x", alpha=0.3)
            ax.legend(fontsize=8, loc="lower right")
            for bar, val in zip(bars_count, count_values):
                if val > 2:
                    ax.text(
                        bar.get_width() + 0.3,
                        bar.get_y() + bar.get_height() / 2.0,
                        f"{val:.1f}%",
                        ha="left", va="center", fontsize=7,
                    )
            for bar, val in zip(bars_int, int_values):
                if val > 2:
                    ax.text(
                        bar.get_width() + 0.3,
                        bar.get_y() + bar.get_height() / 2.0,
                        f"{val:.1f}%",
                        ha="left", va="center", fontsize=7,
                    )
        else:
            ax.text(
                0.5, 0.5, "No signal composition data",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_title("B. Signal Composition (Count vs Intensity)")

        # --- Panel C: Annotated fraction distribution ---
        ax = axes[1, 0]
        per_spectrum = self.results.get("per_spectrum", [])
        annotated_fracs = [
            r.get("annotated_fraction", 0)
            for r in per_spectrum
            if r.get("sequence_available", False)
        ]
        if annotated_fracs:
            af_arr = np.asarray(annotated_fracs, dtype=float)
            af_mean = float(af_arr.mean())
            af_std = float(af_arr.std())
            af_med = float(np.median(af_arr))
            af_q25, af_q75 = (
                float(np.percentile(af_arr, 25)),
                float(np.percentile(af_arr, 75)),
            )
            ax.hist(
                af_arr, bins=30, color="green", alpha=0.6, edgecolor="black",
            )
            ax.axvline(
                af_mean, color="red", linestyle="--", linewidth=2,
                label=f"Mean: {af_mean*100:.1f}% ± {af_std*100:.1f}%",
            )
            ax.axvline(
                af_med, color="black", linestyle=":", linewidth=1.5,
                label=f"Median: {af_med*100:.1f}% (IQR {af_q25*100:.1f}–{af_q75*100:.1f}%)",
            )
            ax.set_xlabel("Annotated Fraction")
            ax.set_ylabel("Count")
            ax.set_xlim(0, 1.0)
            ax.set_title(f"C. Annotated Fraction Distribution (n={len(af_arr):,d})")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
        else:
            ax.text(
                0.5, 0.5, "No annotation data",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_title("C. Annotated Fraction Distribution")

        # --- Panel D: Annotated fraction vs intensity fraction scatter ---
        ax = axes[1, 1]
        valid_ps = [
            r for r in per_spectrum if r.get("sequence_available", False)
        ]
        ann_fracs_d = [r.get("annotated_fraction", 0) for r in valid_ps]
        int_fracs_d = [
            r.get("annotated_intensity_fraction", 0) for r in valid_ps
        ]
        if ann_fracs_d and int_fracs_d:
            # Subsample if too many points
            n_pts = len(ann_fracs_d)
            if n_pts > 10000:
                sample_idx = np.random.choice(n_pts, 10000, replace=False)
                ann_fracs_d = [ann_fracs_d[i] for i in sample_idx]
                int_fracs_d = [int_fracs_d[i] for i in sample_idx]
                valid_ps_sub = [valid_ps[i] for i in sample_idx]
            else:
                valid_ps_sub = valid_ps

            # Color by frag_type
            frag_types_d = [
                str(r.get("_metadata", {}).get("frag_type", "unknown"))
                for r in valid_ps_sub
            ]
            unique_ft = sorted(set(frag_types_d))
            for ft in unique_ft:
                mask = [f == ft for f in frag_types_d]
                x_pts = [ann_fracs_d[i] for i, m in enumerate(mask) if m]
                y_pts = [int_fracs_d[i] for i, m in enumerate(mask) if m]
                color = FRAG_TYPE_COLORS.get(ft, "#7f7f7f")
                ax.scatter(
                    x_pts, y_pts, alpha=0.3, s=5, color=color, label=ft,
                )
            # y=x reference line
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="y=x")
            ax.set_xlabel("Annotated Fraction (by count)")
            ax.set_ylabel("Annotated Fraction (by intensity)")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=7, markerscale=3)
            ax.grid(alpha=0.3)
        else:
            ax.text(
                0.5, 0.5, "No per-spectrum data",
                ha="center", va="center", transform=ax.transAxes,
            )
        ax.set_title("D. Count vs Intensity Annotation")

        plt.tight_layout()
        output_path = self.output_dir / "annotation_summary.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.debug(f"Saved: {output_path}")
        plt.close(fig)
        return fig

    def visualize_annotated_intensity_comparison(self) -> Figure:
        """Annotated vs unannotated peak intensity distributions (1x2).

        Panels:
          A — Overlapping log10(intensity) histograms (density-normalized)
          B — Box/violin side-by-side comparison with median annotations
        """
        logger.debug("Generating annotated intensity comparison...")
        sig_comp = self.results.get("signal_composition", {})
        ann_int = sig_comp.get("_annotated_intensities", [])
        unann_int = sig_comp.get("_unannotated_intensities", [])
        if not ann_int and not unann_int:
            logger.warning("No intensity data for annotated/unannotated comparison")
            return None

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            "Annotated vs Unannotated Peak Intensities",
            fontsize=14,
            fontweight="bold",
        )

        # Convert to log10, filtering zeros
        ann_log = np.log10(np.array([v for v in ann_int if v > 0]) + 1e-12)
        unann_log = np.log10(np.array([v for v in unann_int if v > 0]) + 1e-12)

        # --- Panel A: Overlapping histograms ---
        ax = axes[0]
        if len(ann_log) > 0:
            ax.hist(
                ann_log, bins=50, density=True, alpha=0.6,
                color="#2ca02c", edgecolor="black", linewidth=0.3,
                label=f"Annotated (n={len(ann_log):,d})",
            )
        if len(unann_log) > 0:
            ax.hist(
                unann_log, bins=50, density=True, alpha=0.6,
                color="#d62728", edgecolor="black", linewidth=0.3,
                label=f"Unannotated (n={len(unann_log):,d})",
            )
        ax.set_xlabel("log₁₀(Intensity)")
        ax.set_ylabel("Density")
        ax.set_title("A. Intensity Distribution")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # --- Panel B: Box plot comparison ---
        ax = axes[1]
        box_data = []
        box_labels = []
        box_colors = []
        if len(ann_log) > 0:
            box_data.append(ann_log)
            box_labels.append(f"Annotated\n(n={len(ann_log):,d})")
            box_colors.append("#2ca02c")
        if len(unann_log) > 0:
            box_data.append(unann_log)
            box_labels.append(f"Unannotated\n(n={len(unann_log):,d})")
            box_colors.append("#d62728")
        if box_data:
            bp = ax.boxplot(
                box_data, labels=box_labels, patch_artist=True,
                showfliers=False, widths=0.5,
            )
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
            # Annotate medians
            for i, vals in enumerate(box_data, start=1):
                median_val = float(np.median(vals))
                ax.text(
                    i, median_val, f"  {median_val:.2f}",
                    ha="left", va="center", fontsize=8, fontweight="bold",
                )
        ax.set_ylabel("log₁₀(Intensity)")
        ax.set_title("B. Intensity Comparison")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "annotated_intensity_comparison.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.debug(f"Saved: {output_path}")
        plt.close(fig)
        return fig

    def visualize_mass_error_analysis(self) -> Figure:
        """Mass error characterization with systematic bias (2x2 grid).

        Panels:
          A — |PPM| error by feature type (box plot)
          B — |PPM| error vs m/z (scatter)
          C — PPM error distribution (histogram) with bias line
          D — Systematic bias across m/z (binned mean + trend)
        """
        logger.debug("Generating mass error analysis...")

        mass_error_df = self.results.get("mass_error_df")
        if mass_error_df is None or len(mass_error_df) == 0:
            logger.warning("No mass error data for visualization")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Mass Error Characterization", fontsize=16, fontweight="bold"
        )

        # Subsample for scatter plots
        n = len(mass_error_df)
        idx = (
            np.random.choice(n, 10000, replace=False) if n > 10000 else np.arange(n)
        )

        # --- Panel A: |PPM| error by feature type (box plot) ---
        ax = axes[0, 0]
        ft_order = ["base", "loss", "isotope", "precursor"]
        ft_colors = {
            "base": "#2ca02c",
            "loss": "#98df8a",
            "isotope": "#aec7e8",
            "precursor": "#ffbb78",
        }
        if "feature_type" in mass_error_df.columns:
            box_data = []
            box_labels = []
            box_colors = []
            for ft in ft_order:
                subset = mass_error_df.loc[
                    mass_error_df["feature_type"] == ft, "delta_mz_ppm"
                ]
                if len(subset) > 0:
                    box_data.append(subset.values)
                    box_labels.append(ft)
                    box_colors.append(ft_colors.get(ft, "#7f7f7f"))
            if box_data:
                # Clip y-axis to 99th percentile for readability
                all_ppm = np.concatenate(box_data)
                y_clip = float(np.percentile(all_ppm, 99))
                bp = ax.boxplot(
                    box_data, labels=box_labels, patch_artist=True,
                    showfliers=False,
                )
                for patch, color in zip(bp["boxes"], box_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
                ax.set_ylim(0, y_clip * 1.1)
                ax.set_ylabel("|PPM| Error")
                ax.grid(axis="y", alpha=0.3)
                for i, vals in enumerate(box_data, start=1):
                    ax.text(
                        i, ax.get_ylim()[1] * 0.95,
                        f"n={len(vals):,d}",
                        ha="center", va="top", fontsize=7,
                    )
            else:
                ax.text(
                    0.5, 0.5, "No feature type data",
                    ha="center", va="center", transform=ax.transAxes,
                )
        else:
            ax.text(
                0.5, 0.5, "No feature type column",
                ha="center", va="center", transform=ax.transAxes,
            )
        ax.set_title("A. Mass Error by Feature Type")

        # --- Panel B: |PPM| vs m/z ---
        ax = axes[0, 1]
        x = mass_error_df["theo_mz"].values[idx]
        y_ppm = mass_error_df["delta_mz_ppm"].values[idx]
        scatter = ax.scatter(x, y_ppm, c=y_ppm, cmap="plasma", alpha=0.3, s=1)
        ax.set_xlabel("m/z (Da)")
        ax.set_ylabel("|Δm| (PPM)")
        ax.set_title("B. Absolute PPM Error")
        plt.colorbar(scatter, ax=ax, label="|Δm| (PPM)")
        ax.grid(alpha=0.3)

        # --- Panel C: PPM error distribution ---
        ax = axes[1, 0]
        signed_ppm = mass_error_df["signed_ppm"].values
        ax.hist(signed_ppm, bins=50, color="green", alpha=0.6, edgecolor="black")
        ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero")
        mean_ppm = float(np.mean(signed_ppm))
        ax.axvline(
            mean_ppm,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_ppm:.2f} ppm",
        )
        ax.set_xlabel("Signed PPM Error")
        ax.set_ylabel("Count")
        ax.set_title("C. PPM Error Distribution")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Percentile annotations
        me = self.results.get("mass_error_summary", {})
        percs = me.get("percentiles", {})
        if percs:
            ax.text(
                0.98,
                0.98,
                f"|PPM| P50={percs.get('p50', 0):.1f}\n"
                f"|PPM| P95={percs.get('p95', 0):.1f}\n"
                f"|PPM| P99={percs.get('p99', 0):.1f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        # --- Panel D: Systematic bias vs m/z ---
        ax = axes[1, 1]
        mz_bins = np.arange(0, 2500, 100)
        bin_centers = (mz_bins[:-1] + mz_bins[1:]) / 2
        binned_means = []
        binned_stds = []
        for i in range(len(mz_bins) - 1):
            subset = mass_error_df[
                (mass_error_df["theo_mz"] >= mz_bins[i])
                & (mass_error_df["theo_mz"] < mz_bins[i + 1])
            ]
            if len(subset) > 10:
                binned_means.append(float(np.mean(subset["signed_ppm"])))
                binned_stds.append(float(np.std(subset["signed_ppm"])))
            else:
                binned_means.append(np.nan)
                binned_stds.append(np.nan)

        binned_means = np.array(binned_means)
        binned_stds = np.array(binned_stds)
        valid = ~np.isnan(binned_means)

        ax.errorbar(
            bin_centers[valid],
            binned_means[valid],
            yerr=binned_stds[valid],
            fmt="o-",
            color="steelblue",
            alpha=0.7,
            markersize=3,
            capsize=2,
            label="Mean +/- Std",
        )
        ax.axhline(0, color="red", linestyle="--", linewidth=2, label="No bias")

        # Linear trend
        if np.sum(valid) > 5:
            from numpy.polynomial import polynomial as P

            coefs = P.polyfit(bin_centers[valid], binned_means[valid], 1)
            fit_line = P.polyval(bin_centers, coefs)
            ax.plot(
                bin_centers,
                fit_line,
                "--",
                color="orange",
                linewidth=2,
                label="Linear trend",
            )

        ax.set_xlabel("m/z (Da)")
        ax.set_ylabel("Mean Signed PPM Error")
        ax.set_title("D. Systematic Bias Across m/z")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Add bias significance text
        if me:
            p_val = me.get("systematic_bias_p_value", float("nan"))
            bias = me.get("systematic_bias_ppm", 0)
            sig_text = f"Overall bias: {bias:.2f} ppm"
            if not np.isnan(p_val):
                sig_text += f"\np = {p_val:.3f}"
                if me.get("systematic_bias_significant", False):
                    sig_text += " (significant)"
            ax.text(
                0.02,
                0.98,
                sig_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.tight_layout()
        output_path = self.output_dir / "mass_error_analysis.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.debug(f"Saved: {output_path}")
        plt.close(fig)
        return fig

    def visualize_ion_type_summary(self) -> Figure:
        """Ion type performance summary (2x2).

        Panels:
          A — Ion type match counts with b:y ratio annotation
          B — Coverage rate by ion type
          C — Mass accuracy by ion type (median |PPM| with IQR bars)
          D — Charge distribution per ion type (stacked bar)
        """
        logger.debug("Generating ion type summary...")

        ion_perf = self.results.get("ion_type_performance", {})
        coverage_stats = self.results.get("coverage_stats", {})
        if not ion_perf:
            logger.warning("No ion type data for visualization")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Ion Type Performance", fontsize=14, fontweight="bold")

        ion_types = sorted(ion_perf.keys())
        x_pos = np.arange(len(ion_types))
        bar_colors = [ION_TYPE_COLORS.get(it, "#7f7f7f") for it in ion_types]

        # --- Panel A: Coverage rate + b:y ratio ---
        ax = axes[0, 0]
        ion_cov_a = coverage_stats.get("coverage_by_ion_type", {})
        cov_rates_a = [
            ion_cov_a.get(it, {}).get("coverage_rate", 0) for it in ion_types
        ]
        n_matched_a = [ion_perf[it]["n_matched"] for it in ion_types]
        n_theo_a = [
            ion_cov_a.get(it, {}).get("n_theoretical", 0) for it in ion_types
        ]
        bars = ax.bar(x_pos, cov_rates_a, color=bar_colors, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ion_types)
        ax.set_ylabel("Coverage Rate (matched / theoretical)")
        ax.set_ylim(0, 1.0)
        ax.set_title("A. Coverage Rate by Ion Type")
        ax.grid(axis="y", alpha=0.3)
        # b:y ratio annotation
        b_count = ion_perf.get("b", {}).get("n_matched", 0)
        y_count = ion_perf.get("y", {}).get("n_matched", 0)
        if y_count > 0:
            ratio_text = f"b:y = {b_count/y_count:.2f}"
        elif b_count > 0:
            ratio_text = f"b:y = {b_count}:0"
        else:
            ratio_text = ""
        if ratio_text:
            ax.text(
                0.95, 0.95, ratio_text, transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
        for i, bar in enumerate(bars):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{n_matched_a[i]:,d}/{n_theo_a[i]:,d}",
                ha="center", va="bottom", fontsize=7,
            )

        # --- Panel B: Matched vs theoretical counts (grouped bar) ---
        ax = axes[0, 1]
        ion_cov = coverage_stats.get("coverage_by_ion_type", {})
        if ion_cov:
            cov_types = [it for it in ion_types if it in ion_cov]
            matched_counts = [ion_cov[it]["n_matched"] for it in cov_types]
            theo_counts = [ion_cov[it]["n_theoretical"] for it in cov_types]
            cx_pos = np.arange(len(cov_types))
            w = 0.35
            cov_colors = [ION_TYPE_COLORS.get(it, "#7f7f7f") for it in cov_types]
            ax.bar(
                cx_pos - w / 2, theo_counts, w,
                color=cov_colors, alpha=0.3,
                edgecolor=cov_colors, linewidth=1.2, label="Theoretical",
            )
            ax.bar(
                cx_pos + w / 2, matched_counts, w,
                color=cov_colors, alpha=0.7, label="Matched",
            )
            ax.set_xticks(cx_pos)
            ax.set_xticklabels(cov_types)
            ax.set_ylabel("Ion Count")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
        else:
            ax.text(
                0.5, 0.5, "No coverage data",
                ha="center", va="center", transform=ax.transAxes,
            )
        ax.set_title("B. Theoretical vs Matched Counts")

        # --- Panel C: Mass accuracy (median + IQR) ---
        ax = axes[1, 0]
        mass_error_df = self.results.get("mass_error_df")
        median_errors = [ion_perf[it]["median_abs_ppm_error"] for it in ion_types]
        iqr_lower = []
        iqr_upper = []
        if mass_error_df is not None and "ion_type" in mass_error_df.columns:
            for it in ion_types:
                subset = mass_error_df.loc[
                    mass_error_df["ion_type"] == it, "delta_mz_ppm"
                ]
                if len(subset) > 0:
                    q25, q75 = float(np.percentile(subset, 25)), float(
                        np.percentile(subset, 75)
                    )
                    med = float(np.median(subset))
                    iqr_lower.append(med - q25)
                    iqr_upper.append(q75 - med)
                else:
                    iqr_lower.append(0)
                    iqr_upper.append(0)
        else:
            iqr_lower = [0] * len(ion_types)
            iqr_upper = [0] * len(ion_types)
        ax.bar(
            x_pos,
            median_errors,
            yerr=[iqr_lower, iqr_upper],
            color="coral",
            alpha=0.7,
            capsize=5,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ion_types)
        ax.set_ylabel("Median |PPM| Error")
        ax.set_title("C. Mass Error by Ion Type (IQR)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)

        # --- Panel D: Charge distribution per ion type (normalized stacked bar) ---
        ax = axes[1, 1]
        charge_keys_set: set = set()
        for it in ion_types:
            charge_keys_set.update(ion_perf[it].get("charge_distribution", {}).keys())
        charge_keys = sorted(charge_keys_set)
        if charge_keys:
            charge_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
            # Compute totals per ion type for normalization
            totals = np.array([
                sum(
                    ion_perf[it].get("charge_distribution", {}).get(ck, 0)
                    for ck in charge_keys
                )
                for it in ion_types
            ], dtype=float)
            totals[totals == 0] = 1.0  # avoid division by zero
            bottom = np.zeros(len(ion_types))
            for ci, ck in enumerate(charge_keys):
                raw = np.array([
                    ion_perf[it].get("charge_distribution", {}).get(ck, 0)
                    for it in ion_types
                ], dtype=float)
                fracs = raw / totals
                color = charge_colors[ci % len(charge_colors)]
                ax.bar(
                    x_pos, fracs, bottom=bottom, color=color, alpha=0.7,
                    label=f"z={ck}",
                )
                bottom += fracs
            ax.set_xticks(x_pos)
            ax.set_xticklabels(ion_types)
            ax.set_ylabel("Fraction")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8, title="Charge", loc="upper right")
            ax.grid(axis="y", alpha=0.3)
            # Annotate total count per ion type
            for i, it in enumerate(ion_types):
                ax.text(
                    i, 1.01, f"n={int(totals[i]):,d}",
                    ha="center", va="bottom", fontsize=7,
                )
        else:
            ax.text(
                0.5, 0.5, "No charge data",
                ha="center", va="center", transform=ax.transAxes,
            )
        ax.set_title("D. Charge Distribution per Ion Type")

        plt.tight_layout()
        output_path = self.output_dir / "ion_type_summary.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.debug(f"Saved: {output_path}")
        plt.close(fig)
        return fig

    def visualize_custom_ion_analysis(self) -> Figure:
        """Custom ion diagnostic analysis (2x2 grid).

        Panels:
          A — Hit rate by ion group (top 25), coloured by category
          B — Intensity distribution by category (box plot)
          C — Category co-occurrence heatmap (Jaccard similarity)
          D — Coverage impact by m/z range (stacked bar:
              fragment-annotated / custom-ion-explained / remaining unannotated)
        """
        logger.debug("Generating custom ion analysis figure...")

        cia = self.results.get("custom_ion_analysis", {})
        if not cia:
            logger.warning("No custom ion data for visualization")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Custom Ion Diagnostic Analysis",
            fontsize=16,
            fontweight="bold",
        )

        cat_colors = {
            "glycan": "#1f77b4",
            "immonium": "#2ca02c",
            "TMT": "#ff7f0e",
            "iTRAQ": "#9467bd",
            "other": "#7f7f7f",
        }
        categories = ["glycan", "immonium", "TMT", "iTRAQ"]

        # --- Panel A: Hit rate by ion group (top N) -------------------------
        ax = axes[0, 0]
        ghr = cia.get("group_hit_rates", {})
        if ghr:
            sorted_groups = sorted(
                ghr.items(), key=lambda x: x[1]["hit_rate"], reverse=True
            )
            # Keep top 25 with non-zero hit rate
            top = [(g, d) for g, d in sorted_groups if d["hit_rate"] > 0][:25]
            if top:
                labels = [g for g, _ in top]
                rates = [d["hit_rate"] * 100 for _, d in top]
                colors = [
                    cat_colors.get(self._classify_ion_category(g), "#7f7f7f")
                    for g, _ in top
                ]
                y_pos = np.arange(len(labels))
                ax.barh(y_pos, rates, color=colors, alpha=0.8)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels, fontsize=7)
                ax.set_xlabel("Hit Rate (%)")
                ax.invert_yaxis()
                ax.grid(axis="x", alpha=0.3)
                # Legend for categories
                from matplotlib.patches import Patch
                legend_handles = [
                    Patch(facecolor=cat_colors[c], label=c)
                    for c in categories
                    if any(
                        self._classify_ion_category(g) == c
                        for g, _ in top
                    )
                ]
                if legend_handles:
                    ax.legend(
                        handles=legend_handles, fontsize=7, loc="lower right"
                    )
            else:
                ax.text(
                    0.5, 0.5, "No ions detected",
                    ha="center", va="center", transform=ax.transAxes,
                )
        ax.set_title("A. Custom Ion Hit Rates (top groups)")

        # --- Panel B: Intensity distribution by category --------------------
        ax = axes[0, 1]
        cat_int = cia.get("_category_intensities", {})
        box_data = []
        box_labels = []
        box_colors = []
        for cat in categories:
            vals = cat_int.get(cat, [])
            if vals:
                box_data.append(vals)
                box_labels.append(cat)
                box_colors.append(cat_colors[cat])

        if box_data:
            bp = ax.boxplot(
                box_data,
                labels=box_labels,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker=".", markersize=2, alpha=0.3),
            )
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.set_ylabel("Max Matched Intensity (normalised)")
            ax.grid(axis="y", alpha=0.3)
            # Add sample counts
            for i, (cat, vals) in enumerate(
                zip(box_labels, box_data), start=1
            ):
                ax.text(
                    i, ax.get_ylim()[1] * 0.95,
                    f"n={len(vals):,d}",
                    ha="center", va="top", fontsize=7,
                )
        else:
            ax.text(
                0.5, 0.5, "No intensity data",
                ha="center", va="center", transform=ax.transAxes,
            )
        ax.set_title("B. Intensity Distribution by Category")

        # --- Panel C: Co-occurrence heatmap ---------------------------------
        ax = axes[1, 0]
        co_occ = cia.get("co_occurrence", {})
        # Only include categories with >1% hit rate (avoids sparse heatmaps)
        active_cats = [
            c for c in categories
            if cia.get("category_hit_rates", {}).get(c, {}).get("hit_rate", 0)
            > 0.01
        ]
        if len(active_cats) >= 2 and co_occ:
            n_cats = len(active_cats)
            matrix = np.zeros((n_cats, n_cats))
            for i, ca in enumerate(active_cats):
                for j, cb in enumerate(active_cats):
                    # Jaccard is symmetric; lookup both orderings
                    key = f"{ca}__x__{cb}"
                    rev_key = f"{cb}__x__{ca}"
                    entry = co_occ.get(key) or co_occ.get(rev_key)
                    if entry:
                        matrix[i, j] = entry["jaccard"]
                    elif i == j:
                        matrix[i, j] = 1.0

            im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(n_cats))
            ax.set_xticklabels(active_cats, rotation=45, ha="right")
            ax.set_yticks(range(n_cats))
            ax.set_yticklabels(active_cats)
            plt.colorbar(im, ax=ax, label="Jaccard Similarity", shrink=0.8)
            # Annotate cells
            for i in range(n_cats):
                for j in range(n_cats):
                    val = matrix[i, j]
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if val > 0.5 else "black",
                    )
        else:
            ax.text(
                0.5, 0.5, "Not enough categories detected\n(need ≥2 with >1% hit rate)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9,
            )
        ax.set_title("C. Category Co-occurrence (Jaccard)")

        # --- Panel D: Coverage impact by m/z range --------------------------
        ax = axes[1, 1]
        cl = cia.get("coverage_lift", {})
        range_data = cl.get("by_mz_range", {})
        if range_data:
            range_names = [
                rn for rn in self.mz_range_order if rn in range_data
            ]
            n_peaks = [range_data[rn]["n_peaks"] for rn in range_names]
            n_frag = [
                range_data[rn]["n_fragment_annotated"] for rn in range_names
            ]
            n_custom = [
                range_data[rn]["n_custom_only"] for rn in range_names
            ]
            n_remaining = [
                p - f - c for p, f, c in zip(n_peaks, n_frag, n_custom)
            ]

            # Convert to fractions
            fracs_frag = [
                f / max(p, 1) for f, p in zip(n_frag, n_peaks)
            ]
            fracs_custom = [
                c / max(p, 1) for c, p in zip(n_custom, n_peaks)
            ]
            fracs_remain = [
                r / max(p, 1) for r, p in zip(n_remaining, n_peaks)
            ]

            x_pos = np.arange(len(range_names))
            bar_width = 0.6
            bars_frag = ax.bar(
                x_pos, fracs_frag, bar_width,
                label="Fragment-annotated", color="#2ca02c", alpha=0.8,
            )
            bars_custom = ax.bar(
                x_pos, fracs_custom, bar_width,
                bottom=fracs_frag,
                label="Custom-ion explained", color="#1f77b4", alpha=0.8,
            )
            bars_remain = ax.bar(
                x_pos, fracs_remain, bar_width,
                bottom=[f + c for f, c in zip(fracs_frag, fracs_custom)],
                label="Remaining unannotated", color="#d62728", alpha=0.5,
            )
            ax.set_xticks(x_pos)
            ax.set_xticklabels(range_names, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("Fraction of Peaks")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(axis="y", alpha=0.3)

            # Annotate custom-ion counts on bars
            for i, (cf, cc, rn) in enumerate(
                zip(fracs_frag, fracs_custom, range_names)
            ):
                if n_custom[i] > 0:
                    ax.text(
                        i, cf + cc / 2,
                        f"+{n_custom[i]:,d}",
                        ha="center", va="center", fontsize=7,
                        fontweight="bold", color="white",
                    )
        else:
            ax.text(
                0.5, 0.5, "No coverage data",
                ha="center", va="center", transform=ax.transAxes,
            )
        ax.set_title("D. Coverage Impact by m/z Range")

        plt.tight_layout()
        output_path = self.output_dir / "custom_ion_analysis.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.debug(f"Saved: {output_path}")
        plt.close(fig)
        return fig

    def visualize_neutral_loss_analysis(self) -> Figure:
        """Neutral loss diagnostic analysis (2x2 grid).

        Panels:
          A — Prevalence bars (total count) + twin-axis line (% spectra hit rate)
          B — Box plot of loss-to-parent intensity ratio per loss type
          C — Box plot of |PPM| error per loss type + base ions (reference)
          D — Heatmap of loss_type x ion_series counts
        """
        logger.debug("Generating neutral loss analysis figure...")

        nla = self.results.get("neutral_loss_analysis", {})
        if not nla:
            logger.warning("No neutral loss data for visualization")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Neutral Loss Analysis",
            fontsize=16,
            fontweight="bold",
        )

        prevalence = nla.get("prevalence", {})
        intensity_ratio_stats = nla.get("intensity_ratio_stats", {})
        mass_error_by_lt = nla.get("mass_error_by_loss_type", {})
        crosstab = nla.get("crosstab", {})
        intensity_ratios_raw = nla.get("_intensity_ratios", {})

        loss_types = sorted(prevalence.keys())
        loss_colors = {
            "H2O": "#1f77b4",
            "NH3": "#ff7f0e",
            "H3PO4": "#2ca02c",
            "SO3": "#d62728",
            "CO": "#9467bd",
        }

        # --- Panel A: Prevalence (spectra hit rate) ---
        ax = axes[0, 0]
        if loss_types:
            x_pos = np.arange(len(loss_types))
            total_counts = [prevalence[lt]["total_count"] for lt in loss_types]
            hit_rates = [
                prevalence[lt]["spectra_hit_rate"] for lt in loss_types
            ]
            n_spectra_hit = [
                prevalence[lt].get("n_spectra_hit", 0) for lt in loss_types
            ]
            colors = [loss_colors.get(lt, "#7f7f7f") for lt in loss_types]

            bars = ax.bar(x_pos, hit_rates, color=colors, alpha=0.7)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(loss_types)
            ax.set_ylabel("Spectra Hit Rate")
            ax.set_ylim(0, min(1.0, max(hit_rates) * 1.3) if hit_rates else 1.0)
            ax.grid(axis="y", alpha=0.3)

            for bar, count, n_hit in zip(bars, total_counts, n_spectra_hit):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{n_hit:,d} spectra\n({count:,d} peaks)",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        else:
            ax.text(
                0.5, 0.5, "No loss data",
                ha="center", va="center", transform=ax.transAxes,
            )
        ax.set_title("A. Neutral Loss Prevalence")

        # --- Panel B: Intensity ratio box plot ---
        ax = axes[0, 1]
        box_data = []
        box_labels = []
        box_colors = []
        for lt in loss_types:
            vals = intensity_ratios_raw.get(lt, [])
            if vals:
                box_data.append(vals)
                box_labels.append(lt)
                box_colors.append(loss_colors.get(lt, "#7f7f7f"))

        if box_data:
            bp = ax.boxplot(
                box_data,
                labels=box_labels,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker=".", markersize=2, alpha=0.3),
            )
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.set_ylabel("Loss / Parent Intensity Ratio")
            ax.grid(axis="y", alpha=0.3)
            for i, (lt, vals) in enumerate(zip(box_labels, box_data), start=1):
                ax.text(
                    i,
                    ax.get_ylim()[1] * 0.95,
                    f"n={len(vals):,d}",
                    ha="center",
                    va="top",
                    fontsize=7,
                )
        else:
            ax.text(
                0.5, 0.5, "No intensity ratio data",
                ha="center", va="center", transform=ax.transAxes,
            )
        ax.set_title("B. Loss-to-Parent Intensity Ratio")

        # --- Panel C: Mass error by loss type ---
        ax = axes[1, 0]
        mass_error_df = self.results.get("mass_error_df")
        if mass_error_df is not None and "annotation" in mass_error_df.columns:
            # Collect box data from raw DataFrame for proper box plots
            me_box_data = []
            me_box_labels = []
            me_box_colors = []

            # Base ions reference
            base_rows = mass_error_df[mass_error_df["feature_type"] == "base"]
            if len(base_rows) > 0:
                me_box_data.append(base_rows["delta_mz_ppm"].values)
                me_box_labels.append("base")
                me_box_colors.append("#999999")

            # Loss types
            loss_rows = mass_error_df[mass_error_df["feature_type"] == "loss"].copy()
            if len(loss_rows) > 0:
                loss_rows["loss_type"] = loss_rows["annotation"].apply(
                    self._extract_loss_type
                )
                for lt in loss_types:
                    subset = loss_rows[loss_rows["loss_type"] == lt]
                    if len(subset) > 0:
                        me_box_data.append(subset["delta_mz_ppm"].values)
                        me_box_labels.append(lt)
                        me_box_colors.append(loss_colors.get(lt, "#7f7f7f"))

            if me_box_data:
                bp = ax.boxplot(
                    me_box_data,
                    labels=me_box_labels,
                    patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker=".", markersize=2, alpha=0.3),
                )
                for patch, color in zip(bp["boxes"], me_box_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
                ax.set_ylabel("|PPM| Error")
                ax.grid(axis="y", alpha=0.3)
                for i, (lbl, vals) in enumerate(
                    zip(me_box_labels, me_box_data), start=1
                ):
                    ax.text(
                        i,
                        ax.get_ylim()[1] * 0.95,
                        f"n={len(vals):,d}",
                        ha="center",
                        va="top",
                        fontsize=7,
                    )
            else:
                ax.text(
                    0.5, 0.5, "No mass error data",
                    ha="center", va="center", transform=ax.transAxes,
                )
        else:
            ax.text(
                0.5, 0.5, "No mass error data",
                ha="center", va="center", transform=ax.transAxes,
            )
        ax.set_title("C. Mass Error: Loss Types vs Base Ions")

        # --- Panel D: Heatmap of loss_type x ion_series (row-normalized) ---
        ax = axes[1, 1]
        if crosstab:
            all_series = sorted(
                {s for lt_dict in crosstab.values() for s in lt_dict}
            )
            ct_loss_types = sorted(crosstab.keys())
            raw_matrix = np.zeros((len(ct_loss_types), len(all_series)))
            for i, lt in enumerate(ct_loss_types):
                for j, s in enumerate(all_series):
                    raw_matrix[i, j] = crosstab[lt].get(s, 0)

            # Row-normalize: fraction of each loss type per ion series
            row_sums = raw_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            frac_matrix = raw_matrix / row_sums

            im = ax.imshow(frac_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(range(len(all_series)))
            ax.set_xticklabels(all_series, rotation=45, ha="right")
            ax.set_yticks(range(len(ct_loss_types)))
            ax.set_yticklabels(ct_loss_types)
            plt.colorbar(im, ax=ax, label="Fraction within loss type", shrink=0.8)
            # Annotate cells with fraction and raw count
            for i in range(len(ct_loss_types)):
                for j in range(len(all_series)):
                    frac = frac_matrix[i, j]
                    raw = int(raw_matrix[i, j])
                    if raw > 0:
                        ax.text(
                            j, i, f"{frac:.0%}\n({raw:,d})",
                            ha="center", va="center", fontsize=7,
                            color="white"
                            if frac > 0.5
                            else "black",
                        )
        else:
            ax.text(
                0.5, 0.5, "No cross-tab data",
                ha="center", va="center", transform=ax.transAxes,
            )
        ax.set_title("D. Loss Type x Ion Series")

        plt.tight_layout()
        output_path = self.output_dir / "neutral_loss_analysis.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.debug(f"Saved: {output_path}")
        plt.close(fig)
        return fig

    # -- Fragment group visualizations -----------------------------------------

    def visualize_fragment_group_analysis(self) -> Figure:
        """Fragment ion group analysis (3x2 grid, 14x15, 300 DPI).

        Panels:
          A — Fragment Ion Count by Fragmentation Type (box plot)
          B — Charge State Multiplicity (grouped bar)
          C — Backbone Cleavage Coverage (box plot)
          D — Fragment Charge vs Size by Precursor Charge (line plot)
          E — Complementary Ion Pair Fraction (histogram)
          F — Complementary Fraction by Precursor Charge (box plot)
        """
        logger.debug("Generating fragment group analysis figure...")

        fga = self.results.get("fragment_group_analysis", {})
        if not fga:
            logger.warning("No fragment group data for visualization")
            return None

        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        fig.suptitle(
            "Fragment Ion Group Analysis",
            fontsize=16,
            fontweight="bold",
        )

        by_frag_type = fga.get("by_frag_type", {})
        frag_types = sorted(by_frag_type.keys())
        n_by_series_raw = fga.get("_per_spectrum_n_by_series", [])
        cov_by_series_raw = fga.get("_per_spectrum_coverage_by_series", [])
        frag_type_raw = fga.get("_per_spectrum_frag_type", [])
        frag_obs = fga.get("_fragment_observations", [])
        comp_raw = fga.get("_per_spectrum_complementary", [])
        # TOTAL per-spectrum fragment count (sum across b/y/a/c/x/z). This
        # is the quantity the quality gate is defined on — it must be used
        # for Panel A so the reader can confirm every box sits above the
        # threshold.
        n_groups_total_raw = fga.get("_per_spectrum_n_groups", [])

        from matplotlib.patches import Patch

        ion_series_colors = {
            s: ION_TYPE_COLORS.get(s, "#888888")
            for s in ("a", "b", "c", "x", "y", "z")
        }

        def _grouped_box_panel(
            ax,
            frag_types: List[str],
            per_spectrum_dicts: list,
            frag_type_list: list,
            series_keys_for_ft,
            ylabel: str,
            title: str,
            fmt_median="{:.0f}",
            ylim=None,
        ):
            """Draw per-ion-series box plots grouped by frag_type.

            Parameters
            ----------
            per_spectrum_dicts : list[dict]
                Each element maps series key -> value for one spectrum.
            series_keys_for_ft : tuple or callable
                Either a fixed tuple of series keys (applied to every
                frag_type) or a callable ``ft -> tuple[str, ...]`` that
                returns the series to show for each frag_type. The
                callable form is used to pick c/z for ETD/ECD and b/y
                for collisional methods so panels aren't cluttered with
                ion series the fragmentation method never produces.
            """
            if callable(series_keys_for_ft):
                series_getter = series_keys_for_ft
            else:
                _fixed = tuple(series_keys_for_ft)
                series_getter = lambda _ft: _fixed  # noqa: E731

            # Uniform spacing that accommodates the widest per-ft series
            # count so differently-sized groups still align cleanly.
            box_width = 0.35
            gap = 1.0
            per_ft_keys = {ft: tuple(series_getter(ft)) for ft in frag_types}
            max_n_series = max(
                (len(keys) for keys in per_ft_keys.values()),
                default=1,
            )
            group_stride = max_n_series * box_width + gap

            box_data: List[List[float]] = []
            positions: List[float] = []
            colors: List[str] = []
            group_centers: List[float] = []
            legend_keys: List[str] = []

            for g_idx, ft in enumerate(frag_types):
                keys = per_ft_keys[ft]
                group_left = g_idx * group_stride
                # Centre the (possibly smaller) series block within the
                # uniform slot so labels still land under their boxes.
                n_series_ft = len(keys)
                offset = (max_n_series - n_series_ft) * box_width / 2
                for s_idx, s in enumerate(keys):
                    vals = [
                        d.get(s, 0)
                        for d, f in zip(per_spectrum_dicts, frag_type_list)
                        if f == ft
                    ]
                    if vals:
                        pos = group_left + offset + s_idx * box_width
                        box_data.append(vals)
                        positions.append(pos)
                        colors.append(ion_series_colors.get(s, "#888888"))
                        if s not in legend_keys:
                            legend_keys.append(s)
                group_centers.append(
                    group_left + (max_n_series - 1) * box_width / 2
                )

            if box_data:
                bp = ax.boxplot(
                    box_data,
                    positions=positions,
                    widths=box_width * 0.8,
                    patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker=".", markersize=2, alpha=0.3),
                )
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
                for pos, vals in zip(positions, box_data):
                    median_val = float(np.median(vals))
                    ax.text(
                        pos,
                        median_val,
                        fmt_median.format(median_val),
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        fontweight="bold",
                    )
                ax.set_xticks(group_centers)
                ax.set_xticklabels(frag_types)
                ax.set_ylabel(ylabel)
                ax.grid(axis="y", alpha=0.3)
                ax.legend(
                    handles=[
                        Patch(
                            facecolor=ion_series_colors.get(s, "#888888"),
                            alpha=0.6,
                            label=f"{s}-ions",
                        )
                        for s in legend_keys
                    ],
                    fontsize=8,
                    loc="upper right",
                )
                if ylim is not None:
                    ax.set_ylim(ylim)
                ymin, ymax = ax.get_ylim()
                summary_y = ymin - 0.06 * (ymax - ymin)
                for g_idx, ft in enumerate(frag_types):
                    parts = []
                    for s in per_ft_keys[ft]:
                        vals = [
                            d.get(s, 0)
                            for d, f in zip(per_spectrum_dicts, frag_type_list)
                            if f == ft
                        ]
                        if vals:
                            arr = np.asarray(vals, dtype=float)
                            parts.append(
                                f"{s}: " + fmt_median.format(float(arr.mean()))
                                + "±" + fmt_median.format(float(arr.std()))
                            )
                    if parts:
                        ax.text(
                            group_centers[g_idx],
                            summary_y,
                            "\n".join(parts),
                            ha="center",
                            va="top",
                            fontsize=6,
                            color="#444",
                        )
                ax.set_ylim(ymin - 0.20 * (ymax - ymin), ymax)
            else:
                ax.text(
                    0.5, 0.5, "No data",
                    ha="center", va="center", transform=ax.transAxes,
                )
            ax.set_title(title)

        # --- Panel A: TOTAL fragment ion count per spectrum, by frag_type ---
        # This is the quantity the quality gate is defined on (sum across
        # all ion series), so every spectrum in a gated run MUST be above
        # min_fragment_groups. A per-series breakdown (b vs y) can show 0
        # for one series when the other carries the total, which misleads
        # the reader into thinking the gate was violated — the per-series
        # stats are relegated to subtitle text so the gate invariant is
        # clear on the main plot.
        ax = axes[0, 0]
        totals_by_ft: Dict[str, List[int]] = {ft: [] for ft in frag_types}
        for total, f in zip(n_groups_total_raw, frag_type_raw):
            if f in totals_by_ft:
                totals_by_ft[f].append(int(total))
        box_data_A = [totals_by_ft[ft] for ft in frag_types]
        has_data_A = [len(v) > 0 for v in box_data_A]
        plot_fts = [ft for ft, ok in zip(frag_types, has_data_A) if ok]
        plot_data = [v for v, ok in zip(box_data_A, has_data_A) if ok]
        if plot_data:
            bp = ax.boxplot(
                plot_data,
                labels=plot_fts,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker=".", markersize=2, alpha=0.3),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor("#4c72b0")
                patch.set_alpha(0.6)
            # Annotate each box with median on top, mean ± std under label.
            for i, (ft, vals) in enumerate(zip(plot_fts, plot_data), start=1):
                arr = np.asarray(vals, dtype=float)
                ax.text(
                    i, float(np.median(arr)),
                    f"{int(np.median(arr))}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                )
            # Horizontal reference line at the gate threshold so the reader
            # can confirm visually that no box extends below it.
            min_grp = int(getattr(self, "min_fragment_groups", 7))
            ax.axhline(
                min_grp, color="red", linestyle="--", linewidth=1.2,
                label=f"gate threshold (≥ {min_grp})",
            )
            ax.legend(fontsize=8, loc="upper right")
            ax.set_ylabel("Total fragment groups / spectrum")
            ax.grid(axis="y", alpha=0.3)
            # Sub-line: per-series mean ± std for context. The pair
            # shown is chosen per frag_type (b/y for collisional
            # methods, c/z for ETD/ECD) so electron-driven runs
            # don't report 0/0 under their boxes.
            ymin, ymax = ax.get_ylim()
            summary_y = ymin - 0.06 * (ymax - ymin)
            for i, ft in enumerate(plot_fts, start=1):
                parts = []
                for s in self._primary_pair_for_mode(ft):
                    vs = [
                        d.get(s, 0)
                        for d, f in zip(n_by_series_raw, frag_type_raw)
                        if f == ft and isinstance(d, dict)
                    ]
                    if vs:
                        a = np.asarray(vs, dtype=float)
                        parts.append(
                            f"{s}: {a.mean():.1f}±{a.std():.1f}"
                        )
                if parts:
                    ax.text(
                        i, summary_y, "\n".join(parts),
                        ha="center", va="top", fontsize=6, color="#444",
                    )
            ax.set_ylim(ymin - 0.20 * (ymax - ymin), ymax)
        else:
            ax.text(
                0.5, 0.5, "No data",
                ha="center", va="center", transform=ax.transAxes,
            )
        ax.set_title("A. Total Fragment Groups per Spectrum (gate threshold shown)")

        # --- Panel B: Charge State Multiplicity (grouped bar) ---
        ax = axes[0, 1]
        if frag_types and by_frag_type:
            x_pos = np.arange(len(frag_types))
            bar_width = 0.25
            mult_colors = ["#1f77b4", "#ff7f0e", "#d62728"]
            mult_labels = ["1x", "2x", "3+x"]

            for k, (mult_key, color, label) in enumerate(
                zip(["1", "2", "3+"], mult_colors, mult_labels)
            ):
                vals = [
                    by_frag_type[ft]["charge_multiplicity"].get(mult_key, 0) * 100
                    for ft in frag_types
                ]
                bars = ax.bar(
                    x_pos + k * bar_width - bar_width,
                    vals,
                    bar_width,
                    color=color,
                    alpha=0.7,
                    label=label,
                )
                # Annotate with percentages
                for bar, val in zip(bars, vals):
                    if val > 3:  # Only annotate if visible
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            bar.get_height() + 0.5,
                            f"{val:.0f}%",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                        )
            ax.set_xticks(x_pos)
            ax.set_xticklabels(frag_types)
            ax.set_ylabel("% of Fragment Groups")
            ax.legend(fontsize=8, title="Charge States")
            ax.grid(axis="y", alpha=0.3)
        else:
            ax.text(
                0.5, 0.5, "No multiplicity data",
                ha="center", va="center", transform=ax.transAxes,
            )
        ax.set_title("B. Charge State Multiplicity")

        # --- Panel C: Backbone Cleavage Coverage by Ion Series (grouped box) ---
        # Each frag_type shows its native N/C base ion pair: b/y for
        # collisional activation (HCD/HCID/CID), c/z for electron-driven
        # methods (ETD/ECD). A fixed ("b","y") display made ETD always
        # read 0% since the theoretical generator emits c/z there.
        _grouped_box_panel(
            ax=axes[1, 0],
            frag_types=frag_types,
            per_spectrum_dicts=cov_by_series_raw,
            frag_type_list=frag_type_raw,
            series_keys_for_ft=self._primary_pair_for_mode,
            ylabel="Backbone Cleavage Coverage",
            title="C. Backbone Cleavage Coverage by Ion Series",
            fmt_median="{:.0%}",
            ylim=(-0.05, 1.05),
        )

        # --- Panel D: Fragment Charge vs Size by Precursor Charge (line plot) ---
        ax = axes[1, 1]
        if frag_obs:
            # Group by precursor charge
            prec_groups = {"z=2": [], "z=3": [], "z=4+": []}
            for rel_pos, charge, prec_charge, _ in frag_obs:
                if prec_charge == 2:
                    prec_groups["z=2"].append((rel_pos, charge))
                elif prec_charge == 3:
                    prec_groups["z=3"].append((rel_pos, charge))
                elif prec_charge >= 4:
                    prec_groups["z=4+"].append((rel_pos, charge))

            n_bins = 8
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            prec_colors = {"z=2": "#1f77b4", "z=3": "#ff7f0e", "z=4+": "#d62728"}
            prec_markers = {"z=2": "o", "z=3": "s", "z=4+": "^"}

            for label, obs_list in prec_groups.items():
                if not obs_list:
                    continue
                positions = np.array([o[0] for o in obs_list])
                charges = np.array([o[1] for o in obs_list])
                bin_idx = np.digitize(positions, bin_edges) - 1
                bin_idx = np.clip(bin_idx, 0, n_bins - 1)

                mean_charges = []
                valid_centers = []
                for b in range(n_bins):
                    mask = bin_idx == b
                    if mask.sum() >= 5:  # require minimum observations
                        mean_charges.append(float(charges[mask].mean()))
                        valid_centers.append(bin_centers[b])

                if valid_centers:
                    ax.plot(
                        valid_centers,
                        mean_charges,
                        color=prec_colors[label],
                        marker=prec_markers[label],
                        linewidth=2,
                        markersize=6,
                        label=label,
                    )

            ax.set_xlabel("Relative Fragment Position")
            ax.set_ylabel("Mean Fragment Charge State")
            ax.set_xlim(-0.05, 1.05)
            ax.legend(fontsize=8, title="Precursor Charge")
            ax.grid(alpha=0.3)
        else:
            ax.text(
                0.5, 0.5, "No fragment observations",
                ha="center", va="center", transform=ax.transAxes,
            )
        ax.set_title("D. Fragment Charge vs Size by Precursor Charge")

        # --- Panel E: Complementary Ion Pair Fraction (histogram) ---
        ax = axes[2, 0]
        if comp_raw:
            comp_arr = np.array(comp_raw)
            ax.hist(
                comp_arr, bins=30, color="steelblue", alpha=0.7, edgecolor="black",
            )
            mean_comp = float(np.mean(comp_arr))
            ax.axvline(
                mean_comp, color="red", linestyle="--", linewidth=2,
                label=f"Mean: {mean_comp:.2f}",
            )
            ax.set_xlabel("Complementary Pair Fraction")
            ax.set_ylabel("Count")
            ax.set_xlim(0, 1.0)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
        else:
            ax.text(
                0.5, 0.5, "No complementary pair data",
                ha="center", va="center", transform=ax.transAxes,
            )
        ax.set_title("E. Complementary b/y Ion Pairs")

        # --- Panel F: Complementary fraction by precursor charge ---
        ax = axes[2, 1]
        per_spectrum = self.results.get("per_spectrum", [])
        valid_ps = [
            r for r in per_spectrum if r.get("sequence_available", False)
        ]
        if comp_raw and len(comp_raw) == len(valid_ps):
            charge_groups: Dict[str, List[float]] = {}
            for i, r in enumerate(valid_ps):
                pc = r.get("precursor_charge", 0)
                if pc >= 4:
                    key = "4+"
                elif pc >= 1:
                    key = str(pc)
                else:
                    continue
                charge_groups.setdefault(key, []).append(comp_raw[i])
            sorted_keys = sorted(
                charge_groups.keys(), key=lambda x: (x.endswith("+"), x)
            )
            if sorted_keys:
                box_data = [charge_groups[k] for k in sorted_keys]
                box_labels = [f"z={k}\n(n={len(charge_groups[k]):,d})" for k in sorted_keys]
                bp = ax.boxplot(
                    box_data, labels=box_labels, patch_artist=True,
                    showfliers=False, widths=0.5,
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor("steelblue")
                    patch.set_alpha(0.5)
                ax.set_ylabel("Complementary Pair Fraction")
                ax.grid(axis="y", alpha=0.3)
            else:
                ax.text(
                    0.5, 0.5, "No charge data",
                    ha="center", va="center", transform=ax.transAxes,
                )
        else:
            ax.text(
                0.5, 0.5, "No complementary pair data",
                ha="center", va="center", transform=ax.transAxes,
            )
        ax.set_title("F. Complementary Pairs by Precursor Charge")

        plt.tight_layout()
        output_path = self.output_dir / "fragment_group_analysis.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.debug(f"Saved: {output_path}")
        plt.close(fig)
        return fig

    # -- Ladder length visualizations ------------------------------------------

    def visualize_ladder_analysis(self) -> Figure:
        """Ion ladder length analysis (2x2 grid, 14x10, 300 DPI).

        Panels:
          A — Ladder Length Distribution (histogram, log y-axis)
          B — Max Ladder Length by Fragmentation Type (box plot, excl. ETD)
          C — Group m/z Span per Fragment Ion Group (histogram, base+isotope)
          D — Cumulative Coverage by Span Size (CDF with p75/p90 lines)
        """
        logger.debug("Generating ladder length analysis figure...")

        fga = self.results.get("fragment_group_analysis", {})
        ladder = fga.get("ladder_analysis", {})
        if not ladder:
            logger.warning("No ladder analysis data for visualization")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Fragment Ion Ladder Length Analysis",
            fontsize=16,
            fontweight="bold",
        )

        ion_colors = {
            "b": ION_TYPE_COLORS["b"],
            "y": ION_TYPE_COLORS["y"],
        }

        # --- Panel A: Ladder length distribution (log y-axis) ---
        ax = axes[0, 0]
        all_lengths = fga.get("_all_ladder_lengths", {})
        max_len = 1
        for s in ("b", "y"):
            lengths = all_lengths.get(s, [])
            if lengths:
                max_len = max(max_len, max(lengths))

        bins = np.arange(0.5, max_len + 1.5, 1)
        for s in ("b", "y"):
            lengths = all_lengths.get(s, [])
            if lengths:
                ax.hist(
                    lengths,
                    bins=bins,
                    alpha=0.6,
                    color=ion_colors[s],
                    label=f"{s}-ion (n={len(lengths):,d})",
                    edgecolor="white",
                    linewidth=0.5,
                )
                med = np.median(lengths)
                ax.axvline(
                    med, color=ion_colors[s], linestyle="--", linewidth=1.5,
                    label=f"{s} median={med:.1f}",
                )
        ax.set_yscale("log")
        ax.set_xlabel("Ladder Length (consecutive positions)")
        ax.set_ylabel("Count")
        ax.set_title("A. Ladder Length Distribution", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # --- Panel B: Max ladder by fragmentation type (exclude ETD) ---
        ax = axes[0, 1]
        max_ladder_data = fga.get("_per_spectrum_max_ladder", {})
        frag_types_raw = fga.get("_per_spectrum_frag_type", [])
        # Filter out ETD (zero b/y signal)
        frag_types_unique = sorted(
            ft for ft in set(frag_types_raw) if ft.upper() != "ETD"
        )

        box_width = 0.35
        gap = 1.0
        box_data = []
        positions = []
        colors = []
        group_centers = []

        for g_idx, ft in enumerate(frag_types_unique):
            group_left = g_idx * (2 * box_width + gap)
            for s_idx, s in enumerate(("b", "y")):
                vals = [
                    ml
                    for ml, f in zip(max_ladder_data.get(s, []), frag_types_raw)
                    if f == ft
                ]
                if vals:
                    pos = group_left + s_idx * box_width
                    box_data.append(vals)
                    positions.append(pos)
                    colors.append(ion_colors[s])
            group_centers.append(group_left + box_width / 2)

        if box_data:
            bp = ax.boxplot(
                box_data,
                positions=positions,
                widths=box_width * 0.8,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker=".", markersize=2, alpha=0.3),
            )
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.set_xticks(group_centers)
            ax.set_xticklabels(frag_types_unique)
            from matplotlib.patches import Patch
            ax.legend(
                handles=[
                    Patch(facecolor=ion_colors[s], alpha=0.6, label=f"{s}-ion")
                    for s in ("b", "y")
                ],
                fontsize=8,
            )
        ax.set_ylabel("Max Ladder Length")
        ax.set_title("B. Max Ladder by Fragmentation Type", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # --- Panel C: Group m/z span (base + isotope peaks per group) ---
        ax = axes[1, 0]
        group_spans = fga.get("_all_group_peak_counts", {})
        span_bins = np.arange(0.5, 8.5, 1)  # cap x-axis at 8
        for s in ("b", "y"):
            spans = group_spans.get(s, [])
            if spans:
                ax.hist(
                    spans,
                    bins=span_bins,
                    alpha=0.6,
                    color=ion_colors[s],
                    label=(
                        f"{s}-ion (median={np.median(spans):.0f}, "
                        f"p75={np.percentile(spans, 75):.0f}, "
                        f"n={len(spans):,d})"
                    ),
                    edgecolor="white",
                    linewidth=0.5,
                )
        ax.set_xlabel("Group Size (base + isotope peaks)")
        ax.set_ylabel("Count (groups)")
        ax.set_title(
            "C. Fragment Ion Group Size",
            fontweight="bold",
        )
        ax.set_xlim(0, 8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # --- Panel D: CDF of group m/z span ---
        ax = axes[1, 1]
        for s in ("b", "y"):
            spans = group_spans.get(s, [])
            if spans:
                sorted_spans = np.sort(spans)
                cdf = np.arange(1, len(sorted_spans) + 1) / len(sorted_spans)
                ax.step(
                    sorted_spans, cdf * 100,
                    color=ion_colors[s],
                    linewidth=2,
                    label=f"{s}-ion (n={len(spans):,d})",
                    where="post",
                )
        # Reference lines at 75% and 90%
        ax.axhline(75, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(90, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax.text(0.3, 76, "75% → span_min", fontsize=8, color="gray")
        ax.text(0.3, 91, "90% → span_max", fontsize=8, color="gray")
        ax.set_xlabel("Group Size (base + isotope peaks)")
        ax.set_ylabel("Cumulative Coverage (%)")
        ax.set_title(
            "D. Cumulative Coverage by Group Size",
            fontweight="bold",
        )
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 102)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "ladder_length_analysis.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.debug(f"Saved: {output_path}")
        plt.close(fig)
        return fig

    # -- Quality gate visualizations ------------------------------------------

    def visualize_quality_gate_overview(self) -> Figure:
        """Quality gate overview (2x2 grid).

        Panels:
          A — Histogram of backbone_coverage with threshold line
          B — Rejection rate by frag_type (horizontal bar)
          C — Rejection rate by project (top 10 + OTHER)
          D — Scatter: coverage vs fragment groups with threshold lines
        """
        logger.debug("Generating quality gate overview figure...")
        qga = self.results.get("quality_gate_analysis", {})
        if not qga:
            logger.warning("No quality gate analysis data")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            plt.close(fig)
            return fig

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Quality Gate Overview", fontsize=16, fontweight="bold")

        coverage_arr = qga.get("_coverage_arr", np.array([]))
        n_groups_arr = qga.get("_n_groups_arr", np.array([]))
        below_mask = qga.get("_below_mask", np.array([]))
        overall_rate = qga["overall_rejection_rate"]
        min_cov = qga["min_backbone_coverage"]
        min_grp = qga["min_fragment_groups"]

        # --- Panel A: Histogram of backbone_coverage ---
        ax = axes[0, 0]
        if len(coverage_arr) > 0:
            bins = np.linspace(0, min(1.0, coverage_arr.max() + 0.05), 50)
            ax.hist(
                coverage_arr[below_mask],
                bins=bins,
                color="red",
                alpha=0.3,
                label=f"Rejected ({int(below_mask.sum()):,d})",
            )
            ax.hist(
                coverage_arr[~below_mask],
                bins=bins,
                color="green",
                alpha=0.3,
                label=f"Accepted ({int((~below_mask).sum()):,d})",
            )
            ax.axvline(
                min_cov, color="red", linestyle="--", linewidth=2,
                label=f"Threshold = {min_cov}",
            )
            ax.set_xlabel("Backbone coverage")
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)
        ax.set_title("A. Coverage Distribution")

        # --- Panel B: Rejection rate by frag_type ---
        ax = axes[0, 1]
        frag_data = qga.get("rejection_by_dimension", {}).get("frag_type", {})
        self._draw_rejection_bar(ax, frag_data, overall_rate, "B. Rejection by Fragmentation Type")

        # --- Panel C: Rejection rate by project (top 15 by n_total) ---
        ax = axes[1, 0]
        project_data = qga.get("rejection_by_dimension", {}).get("search_project", {})
        if len(project_data) > 15:
            sorted_items = sorted(
                project_data.items(), key=lambda x: x[1]["n_total"], reverse=True
            )
            project_data = dict(sorted_items[:15])
        self._draw_rejection_bar(ax, project_data, overall_rate, "C. Rejection by Project")

        # --- Panel D: Scatter — coverage vs fragment groups ---
        ax = axes[1, 1]
        if len(coverage_arr) > 0 and len(n_groups_arr) > 0:
            ax.scatter(
                n_groups_arr[~below_mask],
                coverage_arr[~below_mask],
                s=8, alpha=0.3, color="green", label="Accepted",
                rasterized=True,
            )
            ax.scatter(
                n_groups_arr[below_mask],
                coverage_arr[below_mask],
                s=8, alpha=0.3, color="red", label="Rejected",
                rasterized=True,
            )
            ax.axhline(
                min_cov, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
            )
            ax.axvline(
                min_grp, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
            )
            ax.set_xlabel("Fragment groups")
            ax.set_ylabel("Backbone coverage")
            ax.legend(fontsize=8, markerscale=3)
            ax.grid(alpha=0.3)
        else:
            ax.text(
                0.5, 0.5, "No data",
                ha="center", va="center", transform=ax.transAxes,
            )
        ax.set_title("D. Coverage vs Fragment Groups")

        plt.tight_layout()
        output_path = self.output_dir / "quality_gate_overview.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.debug(f"Saved: {output_path}")
        plt.close(fig)
        return fig

    def visualize_quality_gate_diagnostics(self) -> Figure:
        """Quality gate diagnostics (2x2 grid).

        Panels:
          A — Grouped box plots of coverage by frag_type (rejected vs accepted)
          B — Grouped box plots of fragment groups by frag_type
          C — Line plot — rejection rate vs sequence length bucket
          D — Heatmap — rejection rate by frag_type x charge
        """
        logger.debug("Generating quality gate diagnostics figure...")
        qga = self.results.get("quality_gate_analysis", {})
        if not qga:
            logger.warning("No quality gate analysis data")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            plt.close(fig)
            return fig

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Quality Gate Diagnostics", fontsize=16, fontweight="bold")

        coverage_arr = qga.get("_coverage_arr", np.array([]))
        n_groups_arr = qga.get("_n_groups_arr", np.array([]))
        below_mask = qga.get("_below_mask", np.array([]))

        # Per-spectrum frag_type must come from the FULL-population FGA
        # (cached as _full_fragment_group_analysis) because coverage_arr /
        # n_groups_arr / below_mask are full-pop arrays. After Phase C the
        # visible `fragment_group_analysis` slot holds the gated-pop arrays,
        # which have a shorter length and would cause these panels to silently
        # skip ("No data").
        fga = self.results.get(
            "_full_fragment_group_analysis",
            self.results.get("fragment_group_analysis", {}),
        )
        frag_types = fga.get("_per_spectrum_frag_type", [])

        # --- Panel A: Coverage by frag_type (rejected vs accepted) ---
        ax = axes[0, 0]
        if len(coverage_arr) > 0 and len(frag_types) == len(coverage_arr):
            unique_fts = sorted(set(frag_types))
            box_data = []
            box_labels = []
            box_colors = []
            for ft in unique_fts:
                ft_mask = np.array([f == ft for f in frag_types])
                below_vals = coverage_arr[ft_mask & below_mask]
                above_vals = coverage_arr[ft_mask & ~below_mask]
                if len(below_vals) > 0:
                    box_data.append(below_vals)
                    box_labels.append(f"{ft}\nRej")
                    box_colors.append("red")
                if len(above_vals) > 0:
                    box_data.append(above_vals)
                    box_labels.append(f"{ft}\nAcc")
                    box_colors.append("green")
            if box_data:
                bp = ax.boxplot(
                    box_data, labels=box_labels, patch_artist=True,
                    showfliers=False, widths=0.6,
                )
                for patch, color in zip(bp["boxes"], box_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.3)
                ax.tick_params(axis="x", labelsize=8)
            ax.set_ylabel("Backbone coverage")
        ax.set_title("A. Coverage by Frag Type")

        # --- Panel B: Fragment groups by frag_type ---
        ax = axes[0, 1]
        if len(n_groups_arr) > 0 and len(frag_types) == len(n_groups_arr):
            unique_fts = sorted(set(frag_types))
            box_data = []
            box_labels = []
            box_colors = []
            for ft in unique_fts:
                ft_mask = np.array([f == ft for f in frag_types])
                below_vals = n_groups_arr[ft_mask & below_mask]
                above_vals = n_groups_arr[ft_mask & ~below_mask]
                if len(below_vals) > 0:
                    box_data.append(below_vals)
                    box_labels.append(f"{ft}\nRej")
                    box_colors.append("red")
                if len(above_vals) > 0:
                    box_data.append(above_vals)
                    box_labels.append(f"{ft}\nAcc")
                    box_colors.append("green")
            if box_data:
                bp = ax.boxplot(
                    box_data, labels=box_labels, patch_artist=True,
                    showfliers=False, widths=0.6,
                )
                for patch, color in zip(bp["boxes"], box_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.3)
                ax.tick_params(axis="x", labelsize=8)
            ax.set_ylabel("Fragment groups")
        ax.set_title("B. Fragment Groups by Frag Type")

        # --- Panel C: Rejection rate vs sequence length ---
        ax = axes[1, 0]
        seq_data = qga.get("seq_len_analysis", {})
        seq_bins = seq_data.get("bins", [])
        if seq_bins:
            bin_labels = [b["bin"] for b in seq_bins]
            rates = [b["rejection_rate"] for b in seq_bins]
            sizes = [max(np.log10(max(b["n_total"], 1)) * 30, 20) for b in seq_bins]
            x = np.arange(len(bin_labels))
            ax.plot(x, rates, "o-", color="steelblue", linewidth=2)
            ax.scatter(x, rates, s=sizes, color="steelblue", zorder=5)
            ax.set_xticks(x)
            ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Rejection rate")
            ax.set_xlabel("Sequence length bucket")
            rho = seq_data.get("spearman_rho")
            if rho is not None:
                ax.text(
                    0.95, 0.95,
                    f"Spearman r={rho:.3f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )
        ax.set_title("C. Rejection Rate vs Sequence Length")

        # --- Panel D: Heatmap — frag_type x charge ---
        ax = axes[1, 1]
        cross_tab = qga.get("cross_tabulation", {}).get("frag_type_x_charge", {})
        if cross_tab:
            frag_type_list = sorted(cross_tab.keys())
            all_charges = sorted(
                {c for ft in cross_tab.values() for c in ft.keys()},
                key=lambda x: (x.isdigit(), x),
            )
            matrix = np.full((len(frag_type_list), len(all_charges)), np.nan)
            for i, ft in enumerate(frag_type_list):
                for j, ch in enumerate(all_charges):
                    entry = cross_tab[ft].get(ch)
                    if entry and entry["n_total"] >= 5:
                        matrix[i, j] = entry["rejection_rate"]

            im = ax.imshow(
                matrix, cmap="RdYlGn_r", aspect="auto",
                vmin=0, vmax=min(1.0, np.nanmax(matrix) * 1.2) if np.any(np.isfinite(matrix)) else 1.0,
            )
            ax.set_xticks(np.arange(len(all_charges)))
            ax.set_xticklabels(all_charges, fontsize=9)
            ax.set_xlabel("Precursor charge")
            ax.set_yticks(np.arange(len(frag_type_list)))
            ax.set_yticklabels(frag_type_list, fontsize=9)
            plt.colorbar(im, ax=ax, label="Rejection rate", shrink=0.8)
            # Annotate
            for i in range(len(frag_type_list)):
                for j in range(len(all_charges)):
                    val = matrix[i, j]
                    if np.isfinite(val):
                        n = cross_tab[frag_type_list[i]].get(all_charges[j], {}).get("n_total", 0)
                        ax.text(
                            j, i, f"{val:.0%}\n(n={n})",
                            ha="center", va="center", fontsize=7,
                            color="white" if val > 0.5 else "black",
                        )
        else:
            ax.text(
                0.5, 0.5, "No cross-tab data",
                ha="center", va="center", transform=ax.transAxes,
            )
        ax.set_title("D. Rejection Rate: Frag Type x Charge")

        plt.tight_layout()
        output_path = self.output_dir / "quality_gate_diagnostics.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.debug(f"Saved: {output_path}")
        plt.close(fig)
        return fig

    # -- Quality gate visualization helpers ------------------------------------

    def _draw_rejection_bar(
        self,
        ax: plt.Axes,
        data: Dict[str, Dict],
        overall_rate: float,
        title: str,
    ) -> None:
        """Draw horizontal bar chart of rejection rates per group."""
        if not data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            return

        # Sort by rejection rate descending
        items = sorted(data.items(), key=lambda x: x[1]["rejection_rate"], reverse=True)
        labels = [k for k, _ in items]
        rates = [v["rejection_rate"] for _, v in items]
        n_totals = [v["n_total"] for _, v in items]
        n_belows = [v["n_below"] for _, v in items]

        cmap = plt.cm.RdYlGn_r
        colors = [cmap(r) for r in rates]

        y = np.arange(len(labels))
        ax.barh(y, rates, color=colors, edgecolor="gray", linewidth=0.5)
        ax.axvline(overall_rate, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Rejection rate")
        ax.set_xlim(0, min(1.0, max(rates) * 1.2) if rates else 1.0)
        ax.invert_yaxis()

        # Annotate with n/N
        for i, (nb, nt) in enumerate(zip(n_belows, n_totals)):
            ax.text(
                rates[i] + 0.01, i, f"{nb}/{nt}",
                va="center", fontsize=7, color="gray",
            )

        ax.set_title(title)

    # =========================================================================
    # Complementary Pair Visualization
    # =========================================================================

    def visualize_complementary_pair_analysis(self) -> Figure:
        """Complementary b/y pair analysis (2x2 grid).

        Panels:
          A - Histogram of per-spectrum complementary pair fraction
          B - Pair deviation distribution (Da) with expected=0 line
          C - Pair rate by relative cleavage position (bar chart, deciles)
          D - Pair fraction by fragmentation type (box plot)
        """
        cpa = self.results.get("complementary_pair_analysis", {})
        if not cpa:
            logger.warning("No complementary pair data for visualization")
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.text(0.5, 0.5, "No complementary pair data", ha="center", va="center")
            return fig

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Complementary b/y Pair Analysis", fontsize=14, fontweight="bold")

        # Panel A: Histogram of per-spectrum pair fraction
        ax = axes[0, 0]
        fracs = cpa.get("_per_spectrum_pair_fraction", [])
        if fracs:
            ax.hist(fracs, bins=30, color="#1f77b4", edgecolor="white", alpha=0.8)
            mean_frac = cpa["pair_fraction"]["mean"]
            ax.axvline(mean_frac, color="red", linestyle="--", linewidth=1.5,
                       label=f"mean = {mean_frac:.3f}")
            ax.legend(fontsize=9)
        ax.set_xlabel("Complementary pair fraction")
        ax.set_ylabel("Number of spectra")
        ax.set_title("A) Per-spectrum pair fraction")

        # Panel B: Pair deviation distribution (Da)
        ax = axes[0, 1]
        devs = cpa.get("_pair_deviations_da", [])
        if devs:
            devs_arr = np.array(devs)
            # Extreme outliers (precursor-m/z errors, wrong charge
            # states, contaminants) can stretch the x-axis to hundreds of
            # Daltons and crush the informative central distribution.
            # The central mass of the distribution has |dev| ≤ tens of
            # ppm, i.e. a few ×10⁻³ Da. Use the MAD (median absolute
            # deviation) × 10 to auto-scale to that central range, clamp
            # to ±0.05 Da minimum so cleanly-calibrated data still shows
            # a visible spread, and cap at ±2 Da so even pathological
            # datasets don't blow up the axis. Outliers outside the
            # displayed window are counted explicitly in the legend so
            # nothing is hidden silently.
            abs_devs = np.abs(devs_arr)
            mad = float(np.median(abs_devs))
            x_range = float(np.clip(mad * 10.0, 0.05, 2.0))
            in_range = np.abs(devs_arr) <= x_range
            display_devs = devs_arr[in_range]
            n_clipped = int((~in_range).sum())
            ax.hist(display_devs, bins=50, color="#2ca02c", edgecolor="white", alpha=0.8)
            ax.axvline(0, color="red", linestyle="-", linewidth=1.5, label="Expected (0)")
            med = float(np.median(devs_arr))
            mean = float(np.mean(devs_arr))
            std = float(np.std(devs_arr))
            ax.axvline(med, color="orange", linestyle="--", linewidth=1,
                       label=f"median = {med:.4f} Da")
            legend_lines = []
            if n_clipped > 0:
                legend_lines.append(
                    f"{n_clipped:,d}/{len(devs_arr):,d} pairs outside |{x_range:.3f}| Da (hidden)"
                )
            legend_lines.append(f"mean ± std = {mean:.4f} ± {std:.4f} Da")
            ax.text(
                0.02, 0.97, "\n".join(legend_lines),
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
            )
            ax.set_xlim(-x_range, x_range)
            ax.legend(fontsize=9, loc="upper right")
        ax.set_xlabel("Deviation from expected sum (Da)")
        ax.set_ylabel("Number of pairs")
        ax.set_title("B) b+y mass sum deviation")

        # Panel C: Pair rate by relative position (deciles)
        ax = axes[1, 0]
        pos_cov = cpa.get("positional_coverage", {})
        labels = pos_cov.get("bin_labels", [])
        counts = pos_cov.get("bin_counts", [])
        total = pos_cov.get("total_pairs", 1)
        if labels and counts and total > 0:
            rates = [c / total for c in counts]
            x_pos = range(len(labels))
            ax.bar(x_pos, rates, color="#ff7f0e", edgecolor="white", alpha=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Relative cleavage position")
        ax.set_ylabel("Fraction of pairs")
        ax.set_title("C) Pair distribution by position")

        # Panel D: Pair fraction by fragmentation type (box plot)
        ax = axes[1, 1]
        frag_types_list = cpa.get("_per_spectrum_frag_type", [])
        if fracs and frag_types_list:
            unique_ft = sorted(set(frag_types_list))
            data_by_ft = [
                [f for f, ft in zip(fracs, frag_types_list) if ft == uft]
                for uft in unique_ft
            ]
            # Filter empty groups
            non_empty = [(uft, d) for uft, d in zip(unique_ft, data_by_ft) if d]
            if non_empty:
                ft_labels, ft_data = zip(*non_empty)
                bp = ax.boxplot(ft_data, labels=ft_labels, patch_artist=True)
                for patch, ft in zip(bp["boxes"], ft_labels):
                    patch.set_facecolor(FRAG_TYPE_COLORS.get(ft, "#7f7f7f"))
                    patch.set_alpha(0.7)
        ax.set_xlabel("Fragmentation type")
        ax.set_ylabel("Complementary pair fraction")
        ax.set_title("D) Pair fraction by fragmentation type")

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        output_path = self.output_dir / "complementary_pair_analysis.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.debug(f"Saved: {output_path}")
        plt.close(fig)
        return fig

    # =========================================================================
    # Mass Gap Validation Visualization
    # =========================================================================

    def visualize_mass_gap_analysis(self) -> Figure:
        """Mass gap validation against amino acid masses (2x3 grid).

        Panels:
          A - Histogram of per-spectrum gap-to-AA match rate
          B - Distribution of match errors (PPM) for valid gaps
          C - Ambiguity histogram (how many AAs match per gap)
          D - Sequence recovery rate by fragmentation type (bar chart)
          E - Two-sided constraint success rate distribution
          F - Gap match rate by m/z range
        """
        mga = self.results.get("mass_gap_analysis", {})
        if not mga:
            logger.warning("No mass gap data for visualization")
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.text(0.5, 0.5, "No mass gap data", ha="center", va="center")
            return fig

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            f"Mass Gap Validation Against Amino Acid Masses "
            f"(tol={mga.get('ppm_tolerance', 20):.0f} ppm)",
            fontsize=14, fontweight="bold",
        )

        # Panel A: Per-spectrum match rate histogram
        ax = axes[0, 0]
        match_rates = mga.get("_per_spectrum_match_rate", [])
        if match_rates:
            ax.hist(match_rates, bins=30, color="#1f77b4", edgecolor="white", alpha=0.8)
            mean_mr = mga["per_spectrum_match_rate"]["mean"]
            ax.axvline(mean_mr, color="red", linestyle="--", linewidth=1.5,
                       label=f"mean = {mean_mr:.3f}")
            ax.legend(fontsize=9)
        ax.set_xlabel("Fraction of gaps matching a valid AA mass")
        ax.set_ylabel("Number of spectra")
        ax.set_title("A) Per-spectrum gap match rate")

        # Panel B: Match error distribution (PPM)
        ax = axes[0, 1]
        errors_ppm = mga.get("_match_errors_ppm", [])
        if errors_ppm:
            ax.hist(errors_ppm, bins=50, color="#2ca02c", edgecolor="white", alpha=0.8)
            ax.axvline(0, color="red", linestyle="-", linewidth=1)
            med_err = float(np.median(errors_ppm))
            ax.axvline(med_err, color="orange", linestyle="--", linewidth=1,
                       label=f"median = {med_err:.2f} ppm")
            ax.legend(fontsize=9)
        ax.set_xlabel("Match error (PPM)")
        ax.set_ylabel("Number of gaps")
        ax.set_title("B) Mass error of valid gap matches")

        # Panel C: Ambiguity histogram
        ax = axes[0, 2]
        ambig = mga.get("ambiguity_distribution", {})
        if ambig:
            x_vals = sorted(ambig.keys())
            y_vals = [ambig[x] for x in x_vals]
            colors = ["#2ca02c" if x == 1 else "#ff7f0e" if x == 2 else "#d62728"
                      for x in x_vals]
            ax.bar([str(x) for x in x_vals], y_vals, color=colors, edgecolor="white")
            # Annotate with percentages
            total_gaps = sum(y_vals)
            for i, (x, y) in enumerate(zip(x_vals, y_vals)):
                pct = y / max(total_gaps, 1) * 100
                ax.text(i, y + total_gaps * 0.01, f"{pct:.1f}%",
                        ha="center", fontsize=8)
        ax.set_xlabel("Number of AA masses matching")
        ax.set_ylabel("Number of gaps")
        ax.set_title("C) Ambiguity (AAs within tolerance)")

        # Panel D: Sequence recovery rate by fragmentation type
        ax = axes[1, 0]
        by_ft = mga.get("by_frag_type", {})
        if by_ft:
            ft_names = sorted(by_ft.keys())
            match_vals = [by_ft[ft]["match_rate"]["mean"] for ft in ft_names]
            correct_vals = [by_ft[ft]["correct_rate"]["mean"] for ft in ft_names]
            x_pos = np.arange(len(ft_names))
            w = 0.35
            ax.bar(x_pos - w / 2, match_vals, w,
                   label="Valid AA match", color="#1f77b4", alpha=0.8)
            ax.bar(x_pos + w / 2, correct_vals, w,
                   label="Correct AA", color="#2ca02c", alpha=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(ft_names)
            ax.legend(fontsize=9)
            # Annotate with n_spectra
            for i, ft in enumerate(ft_names):
                n = by_ft[ft]["n_spectra"]
                ax.text(i, max(match_vals[i], correct_vals[i]) + 0.02,
                        f"n={n}", ha="center", fontsize=7, color="gray")
        ax.set_xlabel("Fragmentation type")
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1.15)
        ax.set_title("D) Match & recovery rate by frag type")

        # Panel E: Two-sided constraint rate distribution
        ax = axes[1, 1]
        two_sided = mga.get("_per_spectrum_two_sided_rate", [])
        if two_sided:
            ax.hist(two_sided, bins=30, color="#9467bd", edgecolor="white", alpha=0.8)
            mean_ts = mga["per_spectrum_two_sided_rate"]["mean"]
            ax.axvline(mean_ts, color="red", linestyle="--", linewidth=1.5,
                       label=f"mean = {mean_ts:.3f}")
            ax.legend(fontsize=9)
        ax.set_xlabel("Two-sided constraint success rate")
        ax.set_ylabel("Number of spectra")
        ax.set_title("E) Two-sided constraint (both neighbours valid)")

        # Panel F: Gap match rate by m/z range
        ax = axes[1, 2]
        by_mz = mga.get("by_mz_range", {})
        if by_mz:
            range_order = [r for r in self.mz_range_order if r in by_mz]
            match_vals = [by_mz[r]["match_rate"] for r in range_order]
            correct_vals = [by_mz[r]["correct_rate"] for r in range_order]
            n_vals = [by_mz[r]["n_gaps"] for r in range_order]
            x_pos = np.arange(len(range_order))
            w = 0.35
            ax.bar(x_pos - w / 2, match_vals, w,
                   label="Valid AA match", color="#1f77b4", alpha=0.8)
            ax.bar(x_pos + w / 2, correct_vals, w,
                   label="Correct AA", color="#2ca02c", alpha=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                [r.replace("_", "\n") for r in range_order], fontsize=8
            )
            ax.legend(fontsize=9)
            for i, n in enumerate(n_vals):
                ax.text(i, max(match_vals[i], correct_vals[i]) + 0.02,
                        f"n={n:,}", ha="center", fontsize=7, color="gray")
        ax.set_xlabel("m/z range")
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1.15)
        ax.set_title("F) Gap match rate by m/z range")

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        output_path = self.output_dir / "mass_gap_analysis.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.debug(f"Saved: {output_path}")
        plt.close(fig)
        return fig
