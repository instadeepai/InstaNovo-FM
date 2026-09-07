#!/usr/bin/env python
"""
Masking Strategy Comparison Analyser for Foundation Model Training.

Runs all configured masking strategies on the same preprocessed spectra,
producing cross-strategy comparison metrics and visualizations.

Output goes to ``masking_analysis/`` with per-strategy subdirectories.

Usage:
    Integrated with SpectrumAnalyser — not meant to be called directly.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from instanovo.__init__ import console
from instanovo_fm.data.masking import get_mask_function
from instanovo_fm.data.masking_gap_analyser import MaskingGapAnalyser
from instanovo_fm.data.theoretical_analyser import TheoreticalAnalyser
from instanovo_fm.utils.naming import sanitize_filename
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger


class MaskingAnalyser:
    """Compare masking strategies on the same preprocessed spectra.

    Follows the BinningAnalyser 4-method pattern:
    ``analyze_spectrum`` → ``aggregate_results`` → ``generate_visualizations``
    → ``save_results`` / ``print_summary``.

    For each configured strategy the analyser:
    1. Applies the masking function directly (from ``masking.py``).
    2. Delegates gap analysis to a per-strategy ``MaskingGapAnalyser``.
    3. Calls ``TheoreticalAnalyser._analyze_masking_effect`` when theoretical
       annotation is available.
    """

    # Annotation labels to exclude from analysis (not informative for masking study)
    _EXCLUDED_LABELS = {"precursor", "precursor-isotope"}

    def __init__(self, config: DictConfig, output_dir: Optional[Path] = None):
        self.config = config

        if output_dir is None:
            self.output_dir = Path("analysis_output") / "masking_analysis"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        analysis_config = config.get("analysis", {})

        # Number of stochastic repeats per strategy per spectrum
        self.n_repeats: int = int(analysis_config.get("masking_analysis_n_repeats", 1))

        # Model params needed for normalisation inversion
        self.max_mz: float = config.model.get("max_mz", 2500.0)
        self.normalize_mz: bool = config.model.get("normalize_mz", True)

        # Quality-gate thresholds: kept in sync with the theoretical
        # analyser so masking metrics describe the same population
        # (gated-only) that the theoretical report already uses. Gate
        # parameters live under ``task_configs.theoretical`` alongside
        # ``apply_quality_gate_filter`` so the two analysers stay
        # aligned without duplicating the values.
        task_configs = analysis_config.get("task_configs", {})
        theo_task = task_configs.get("theoretical", {})
        self.apply_quality_gate_filter: bool = bool(
            theo_task.get("theoretical_analysis", {}).get(
                "apply_quality_gate_filter", True
            )
        )
        self.min_backbone_coverage: float = float(
            theo_task.get("min_backbone_coverage", 0.33)
        )
        self.min_fragment_groups: int = int(
            theo_task.get("min_fragment_groups", 7)
        )

        # Strategy configs
        self.strategies: Dict[str, Dict[str, Any]] = self._get_masking_strategies_from_config()

        # Publication (clean paper) figure mode for the per-spectrum comparison:
        # drop the in-figure super-title, relabel internal strategy codenames to
        # readable names, optionally subset strategies, and also emit SVG + PDF.
        _mask_task = task_configs.get("masking", {})
        self.publication_figure: bool = bool(_mask_task.get("publication_figure", False))
        _pub_strats = _mask_task.get("publication_strategies", None)
        self.publication_strategies = list(_pub_strats) if _pub_strats else None
        self.publication_labels: Dict[str, str] = {
            "unif_gp30": "Uniform",
            "thom_gp30": "TS − isotope",
            "ts34_gp30_noiso": "TS − isotope",
            "ts34_gp30_iso": "TS + isotope",
            "ts34_gp25_iso": "TS + isotope (25%)",
            "sigaw_gp30": "Annotation oracle",
            **dict(_mask_task.get("publication_labels", {}) or {}),
        }

        # Create one MaskingGapAnalyser per strategy
        self.gap_analysers: Dict[str, MaskingGapAnalyser] = {}
        for name in self.strategies:
            strategy_dir = self.output_dir / name
            strategy_dir.mkdir(parents=True, exist_ok=True)
            self.gap_analysers[name] = MaskingGapAnalyser(config, strategy_dir)

        # Storage
        self._per_spectrum: List[Dict[str, Dict]] = []
        self.results: Dict[str, Any] = {}

        # Count of spectra skipped by the gate (for aggregation summary)
        self._n_gate_rejected: int = 0

        logger.info(
            f"MaskingAnalyser initialised with {len(self.strategies)} strategies: "
            f"{list(self.strategies.keys())}, n_repeats={self.n_repeats}, "
            f"quality_gate={'on' if self.apply_quality_gate_filter else 'off'} "
            f"(min_backbone_coverage={self.min_backbone_coverage}, "
            f"min_fragment_groups={self.min_fragment_groups})"
        )

    # ------------------------------------------------------------------
    # Quality gate helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _passes_quality_gate(
        theoretical_analysis: Dict[str, Any],
        min_backbone_coverage: float,
        min_fragment_groups: int,
    ) -> bool:
        """Backbone-coverage + fragment-group gate, shared with
        :class:`SpectrumAnalyser`. Duplicated here (rather than
        imported) so MaskingAnalyser has no runtime dependency on the
        orchestrator and can be invoked standalone in tests.
        """
        n_terminal_ions = {"b", "a", "c"}
        c_terminal_ions = {"y", "x", "z"}
        feature_types = theoretical_analysis.get("feature_types")
        annotations = theoretical_analysis.get("theo_annotations")
        if not feature_types or not annotations:
            return False
        seq = theoretical_analysis.get("clean_sequence", "") or ""
        seq_len = len(seq)
        groups: set = set()
        for ft, ann in zip(feature_types, annotations):
            if ft != "base" or not ann:
                continue
            ion_type = TheoreticalAnalyser._extract_ion_type(ann)
            position = TheoreticalAnalyser._extract_fragment_position(ann)
            if position < 1 or ion_type == "unknown":
                continue
            groups.add((ion_type, position))
        n_groups = len(groups)
        if seq_len > 1:
            max_sites = seq_len - 1
            cleavage_sites: set = set()
            for (ion_type, position) in groups:
                if ion_type in n_terminal_ions:
                    cleavage_sites.add(position)
                elif ion_type in c_terminal_ions:
                    cleavage_sites.add(seq_len - position)
            coverage = len(cleavage_sites) / max_sites
        else:
            coverage = 0.0
        return coverage >= min_backbone_coverage and n_groups >= min_fragment_groups

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _get_masking_strategies_from_config(self) -> Dict[str, Dict[str, Any]]:
        """Read masking strategy definitions from the analysis config.

        Checks three locations in priority order:
        1. ``config.analysis.masking_strategies`` (legacy flat key)
        2. ``config.analysis.task_configs.masking.strategies`` (new structured config)
        3. Fallback: build from ``config.model.masking`` defaults
        """
        analysis_config = self.config.get("analysis", {})

        # 1. Legacy flat key
        raw = analysis_config.get("masking_strategies", None)

        # 2. New structured config path
        if raw is None:
            task_configs = analysis_config.get("task_configs", {})
            masking_task = task_configs.get("masking", {})
            raw = masking_task.get("strategies", None)

        if raw is not None:
            # Config-supplied dict (OmegaConf → plain dict)
            from omegaconf import OmegaConf

            strategies = {}
            for name, entry in raw.items():
                if hasattr(entry, "_metadata"):
                    strategies[str(name)] = dict(OmegaConf.to_container(entry, resolve=True))
                else:
                    strategies[str(name)] = dict(entry)
            return strategies

        # 3. Fallback: one entry per registered strategy with model-level defaults
        masking_config = self.config.model.get("masking", {})
        mask_portion = masking_config.get("mask_portion", 0.30)
        return {
            "uniform": {"type": "uniform", "mask_portion": mask_portion},
            "thompson": {
                "type": "thompson",
                "mask_portion": mask_portion,
                "alpha": masking_config.get("alpha", 0.5),
                "beta": masking_config.get("beta", 0.5),
                "kappa": masking_config.get("kappa", 4.0),
                "gamma": masking_config.get("gamma", 0.7),
            },
            "thompson_span": {
                "type": "thompson_span",
                "mask_portion": mask_portion,
                "span_min": masking_config.get("span_min", 4),
                "span_max": masking_config.get("span_max", 7),
                "alpha": masking_config.get("alpha", 0.5),
                "beta": masking_config.get("beta", 0.5),
                "kappa": masking_config.get("kappa", 4.0),
                "gamma": masking_config.get("gamma", 0.7),
                "bidirectional": masking_config.get("bidirectional", True),
                "include_isotopes": masking_config.get("include_isotopes", False),
                "max_total_mask_ratio": masking_config.get("max_total_mask_ratio", 0.35),
            },
            "signal_aware_fragment": {
                "type": "signal_aware_fragment",
                "mask_portion": mask_portion,
                "min_backbone_coverage": masking_config.get("signal_min_backbone_coverage", 0.15),
                "min_fragment_groups": masking_config.get("signal_min_fragment_groups", 3),
                "annotation_ppm": masking_config.get("signal_ppm", 20.0),
                "annotation_cid_da_tol": masking_config.get("signal_cid_da_tol", 0.2),
                "annotation_ion_types": tuple(
                    masking_config.get("signal_ion_types", ["b", "y"])
                ),
                "max_total_mask_ratio": masking_config.get("max_total_mask_ratio", 0.35),
            },
        }

    # ------------------------------------------------------------------
    # Per-spectrum analysis
    # ------------------------------------------------------------------

    def _apply_strategy(
        self,
        name: str,
        strategy_config: Dict[str, Any],
        spectra: torch.Tensor,
        spectra_mask: torch.Tensor,
        charges: Optional[torch.Tensor] = None,
        peptides: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "tuple[torch.Tensor, Optional[bool]]":
        """Apply a single masking strategy and return the peak mask.

        Args:
            name: Strategy name (for logging).
            strategy_config: Dict with ``type`` and strategy-specific params.
            spectra: ``[1, L, 2]`` preprocessed spectra tensor.
            spectra_mask: ``[1, L]`` padding mask (True = pad).
            charges: ``[1]`` precursor charges (optional).
            peptides: ``[1]`` peptide sequences (optional).
            metadata: Spectrum metadata dict (used for frag_type in signal-aware).

        Returns:
            ``(peak_mask, fallback_used)`` — the ``[1, L]`` bool mask and
            a per-strategy fallback flag. ``fallback_used`` is ``None``
            for strategies that have no fallback path; it is ``True`` /
            ``False`` for ``signal_aware_fragment`` depending on whether
            the annotation-driven path or the ``thompson_span`` fallback
            produced the mask.
        """
        strategy_type = strategy_config["type"]
        mask_fn = get_mask_function(strategy_type)

        mz = spectra[:, :, 0]
        intensity = spectra[:, :, 1]
        mask_portion = strategy_config.get("mask_portion", 0.30)

        if strategy_type == "uniform":
            return mask_fn(
                intensity=intensity,
                spectra_mask=spectra_mask,
                mask_portion=mask_portion,
            ), None

        if strategy_type == "thompson":
            return mask_fn(
                intensity=intensity,
                spectra_mask=spectra_mask,
                mask_portion=mask_portion,
                alpha=strategy_config.get("alpha", 0.5),
                beta=strategy_config.get("beta", 0.5),
                kappa=strategy_config.get("kappa", 4.0),
                gamma=strategy_config.get("gamma", 0.7),
            ), None

        if strategy_type == "thompson_span":
            return mask_fn(
                intensity=intensity,
                spectra_mask=spectra_mask,
                mask_portion=mask_portion,
                span_min=strategy_config.get("span_min", 4),
                span_max=strategy_config.get("span_max", 7),
                alpha=strategy_config.get("alpha", 0.5),
                beta=strategy_config.get("beta", 0.5),
                kappa=strategy_config.get("kappa", 4.0),
                gamma=strategy_config.get("gamma", 0.7),
                bidirectional=strategy_config.get("bidirectional", True),
                mz=mz,
                charges=charges if strategy_config.get("include_isotopes", False) else None,
                include_isotopes=strategy_config.get("include_isotopes", False),
                isotope_ppm=strategy_config.get("isotope_ppm", 25.0),
                isotope_da_floor=strategy_config.get("isotope_da_floor", 0.02),
                isotope_max_charge=strategy_config.get("isotope_max_charge", 3),
                isotope_max_order=strategy_config.get("isotope_max_order", 2),
                max_total_mask_ratio=strategy_config.get("max_total_mask_ratio", 0.40),
                max_mz=self.max_mz,
                normalize_mz=self.normalize_mz,
            ), None

        if strategy_type == "signal_aware_fragment":
            frag_type = None
            if metadata is not None:
                frag_type = metadata.get("frag_type")
            frag_types = [frag_type] if frag_type else None

            peak_mask, fallback_flags = mask_fn(
                intensity=intensity,
                spectra_mask=spectra_mask,
                mz=mz,
                charges=charges,
                peptides=peptides,
                mask_portion=mask_portion,
                min_backbone_coverage=strategy_config.get("min_backbone_coverage", 0.15),
                min_fragment_groups=strategy_config.get("min_fragment_groups", 3),
                annotation_ppm=strategy_config.get("annotation_ppm", 20.0),
                annotation_cid_da_tol=strategy_config.get("annotation_cid_da_tol", 0.2),
                annotation_ion_types=tuple(
                    strategy_config.get("annotation_ion_types", ("b", "y"))
                ),
                frag_types=frag_types,
                max_mz=self.max_mz,
                normalize_mz=self.normalize_mz,
                max_total_mask_ratio=strategy_config.get("max_total_mask_ratio", 0.40),
                # Sequential to avoid nested multiprocessing
                num_workers=1,
                # Surface the annotation-vs-fallback decision so per-strategy
                # stats can split the two populations. Batch size here is 1
                # (see analyze_spectrum) so we read the single flag.
                return_fallback_mask=True,
            )
            fallback_used = bool(fallback_flags[0]) if fallback_flags else None
            return peak_mask, fallback_used

        raise ValueError(f"Unknown masking strategy type: {strategy_type}")

    def analyze_spectrum(
        self,
        spectra: torch.Tensor,
        spectra_mask: torch.Tensor,
        precursor_charges: Optional[torch.Tensor],
        valid_mz: np.ndarray,
        valid_intensity: np.ndarray,
        metadata: Dict[str, Any],
        theoretical_analysis: Optional[Dict[str, Any]] = None,
        matched_annotations: Optional[List[str]] = None,
        feature_types: Optional[List[Optional[str]]] = None,
        parent_annotations: Optional[List[Optional[str]]] = None,
        peptides: Optional[List[str]] = None,
        custom_ion_peak_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Run all strategies on one preprocessed spectrum.

        Args:
            spectra: ``[1, L, 2]`` preprocessed spectrum (m/z normalised, intensity).
            spectra_mask: ``[1, L]`` padding mask (True = pad).
            precursor_charges: ``[1]`` charge tensor (or None).
            valid_mz: 1-D numpy array of non-padded m/z values (Da).
            valid_intensity: 1-D numpy array of non-padded intensities.
            metadata: Spectrum metadata dict (frag_type, etc.).
            theoretical_analysis: Output of TheoreticalAnalyser for this spectrum.
            matched_annotations: Per-peak annotation strings.
            feature_types: Per-peak feature types.
            parent_annotations: Per-peak parent annotations.
            peptides: List with single peptide string (or None).

        Returns:
            ``{strategy_name: {gap_result, masking_effect, mask_ratio, ...}}``
        """
        valid_peaks = ~spectra_mask[0]  # [L] bool
        results: Dict[str, Dict[str, Any]] = {}

        # Gate: skip spectra that fail the theoretical backbone-coverage +
        # fragment-group gate so masking metrics describe the same
        # population reported by the theoretical analyser. Unannotated
        # spectra (no sequence / no theoretical) always pass through
        # since the gate is undefined for them — they never contribute
        # to masking_effect or leakage stats anyway.
        if (
            self.apply_quality_gate_filter
            and theoretical_analysis is not None
            and theoretical_analysis.get("sequence_available", False)
        ):
            if not self._passes_quality_gate(
                theoretical_analysis,
                self.min_backbone_coverage,
                self.min_fragment_groups,
            ):
                self._n_gate_rejected += 1
                # Empty dict is falsy so the orchestrator's
                # ``if mr: acc_masking_result.append(mr)`` filter
                # drops this spectrum without extra plumbing; the
                # per-analyser counter retains the rejection tally
                # for the aggregation summary.
                return {}

        for name, scfg in self.strategies.items():
            strategy_results: Dict[str, Any] = {"repeats": []}

            for _ in range(self.n_repeats):
                peak_mask, fallback_used = self._apply_strategy(
                    name, scfg, spectra, spectra_mask, precursor_charges, peptides, metadata
                )  # mask [1, L]; fallback_used bool|None

                mlm_mask_valid = peak_mask[0][valid_peaks].cpu().numpy()
                n_valid = int(valid_peaks.sum().item())
                mask_ratio = float(mlm_mask_valid.sum()) / max(n_valid, 1)

                # Gap analysis
                gap_result = self.gap_analysers[name].analyze_spectrum(
                    valid_mz=valid_mz,
                    mlm_mask_valid=mlm_mask_valid,
                    metadata=metadata,
                )

                # Theoretical masking interaction (if available)
                masking_effect: Dict[str, Any] = {}
                if (
                    theoretical_analysis is not None
                    and theoretical_analysis.get("sequence_available", False)
                    and matched_annotations is not None
                    and feature_types is not None
                    and parent_annotations is not None
                ):
                    # Thread frag_type through so the masking-effect
                    # computation can restrict its "primary" fragment-group
                    # scope to the method's canonical N/C ion pair (b/y for
                    # HCD/HCID/CID, c/z for ETD/ECD). Without this the
                    # denominator included a-ions that signal_aware never
                    # targets, inflating the "unmasked groups" count.
                    masking_effect = TheoreticalAnalyser._analyze_masking_effect(
                        theoretical_analysis=theoretical_analysis,
                        valid_intensity=valid_intensity,
                        mlm_mask=mlm_mask_valid,
                        matched_annotations=matched_annotations,
                        feature_types=feature_types,
                        parent_annotations=parent_annotations,
                        custom_ion_peak_mask=custom_ion_peak_mask,
                        frag_type=metadata.get("frag_type") if metadata else None,
                    )

                # Intensity stats for masked peaks
                masked_intensity = valid_intensity[mlm_mask_valid] if mlm_mask_valid.any() else np.array([])
                unmasked_intensity = valid_intensity[~mlm_mask_valid] if (~mlm_mask_valid).any() else np.array([])

                # Information leakage metric
                leakage: Dict[str, Any] = {}
                if (
                    theoretical_analysis is not None
                    and theoretical_analysis.get("sequence_available", False)
                    and feature_types is not None
                    and parent_annotations is not None
                    and matched_annotations is not None
                ):
                    leakage = MaskingAnalyser._compute_information_leakage(
                        mlm_mask_valid=mlm_mask_valid,
                        feature_types=feature_types,
                        parent_annotations=parent_annotations,
                        matched_annotations=matched_annotations,
                    )

                repeat_result = {
                    "mask_ratio": mask_ratio,
                    "gap_result": gap_result,
                    "masking_effect": masking_effect,
                    "leakage": leakage,
                    "mlm_mask_valid": mlm_mask_valid,
                    "masked_intensity_mean": float(masked_intensity.mean()) if len(masked_intensity) > 0 else 0.0,
                    "unmasked_intensity_mean": float(unmasked_intensity.mean()) if len(unmasked_intensity) > 0 else 0.0,
                    # None for strategies without a fallback path; True/False
                    # only for signal_aware_fragment. Lets aggregation separate
                    # annotation-driven masks from thompson_span fallbacks
                    # within the same strategy.
                    "fallback_used": fallback_used,
                }
                strategy_results["repeats"].append(repeat_result)

            # Average across repeats
            strategy_results["mask_ratio"] = float(
                np.mean([r["mask_ratio"] for r in strategy_results["repeats"]])
            )
            results[name] = strategy_results

        # Store visualization data (once per spectrum, shared across strategies)
        n_valid = int(valid_peaks.sum().item())
        viz_data: Dict[str, Any] = {
            "valid_mz": valid_mz,
            "valid_intensity": valid_intensity,
            "n_valid": n_valid,
            "metadata": metadata,
        }
        if (
            theoretical_analysis is not None
            and theoretical_analysis.get("sequence_available", False)
            and matched_annotations is not None
        ):
            viz_data["annotated_mask"] = np.array(
                theoretical_analysis.get("annotated_mask", [])
            )
            viz_data["theo_annotations"] = matched_annotations
            viz_data["feature_types"] = feature_types
            viz_data["parent_annotations"] = parent_annotations
            viz_data["custom_ion_data"] = theoretical_analysis.get("custom_ion_data", {})
        results["_viz_data"] = viz_data

        return results

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_results(
        self, per_spectrum_results: List[Dict[str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Aggregate per-spectrum results across all strategies.

        Produces two parallel per-strategy views:

        * ``per_strategy`` — aggregated over every spectrum that entered the
          pipeline (matches the deployed training behaviour, including
          ``signal_aware_fragment``'s thompson_span fallback).
        * ``per_strategy_annotation_driven`` — restricted to spectra where
          ``signal_aware_fragment`` used its annotation-driven path
          (``fallback_used == False``). This is the "gold-standard" view
          used to judge signal_aware at its intended operating point
          without fallback noise.

        The subset view is skipped (and logged) when
        ``signal_aware_fragment`` is not configured — there is no reference
        strategy to define the subset.

        Args:
            per_spectrum_results: List of dicts returned by ``analyze_spectrum``.

        Returns:
            Aggregated results dict with both per-strategy views plus
            ``comparison`` / ``stratified`` / ``quality_gate`` metadata.
        """
        if not per_spectrum_results:
            logger.warning("No masking analysis data to aggregate")
            self.results = {}
            return self.results

        strategy_names = list(self.strategies.keys())

        # Pre-compute spatial m/z bin edges (shared across strategies and views)
        min_mz = self.config.model.get("min_mz", 50.0)
        max_mz = self.max_mz
        n_spatial_bins = 25
        spatial_bin_edges = np.linspace(min_mz, max_mz, n_spatial_bins + 1)
        spatial_bin_centers = (
            (spatial_bin_edges[:-1] + spatial_bin_edges[1:]) / 2
        ).tolist()

        per_strategy = self._aggregate_per_strategy(
            per_spectrum_results,
            strategy_names,
            spatial_bin_edges,
            spatial_bin_centers,
            log_fallback=True,
        )

        # Build comparison DataFrames
        comparison = self._build_comparison_tables(per_strategy, strategy_names)

        # Build stratified masking summary (by frag_type, precursor_charge, instrument)
        stratified = self._build_stratified_masking_summary(
            per_spectrum_results, strategy_names
        )

        self.results = {
            "per_strategy": per_strategy,
            "comparison": comparison,
            "stratified": stratified,
            "n_spectra": len(per_spectrum_results),
            "n_strategies": len(strategy_names),
            # Quality-gate bookkeeping so the report makes the
            # (potentially different) denominator vs. raw input visible.
            "quality_gate": {
                "applied": self.apply_quality_gate_filter,
                "min_backbone_coverage": self.min_backbone_coverage,
                "min_fragment_groups": self.min_fragment_groups,
                "n_rejected": self._n_gate_rejected,
                "n_analyzed": len(per_spectrum_results),
            },
        }
        if self.apply_quality_gate_filter and self._n_gate_rejected > 0:
            logger.info(
                f"MaskingAnalyser quality gate: {self._n_gate_rejected} spectra rejected "
                f"(coverage<{self.min_backbone_coverage} or groups<{self.min_fragment_groups}); "
                f"{len(per_spectrum_results)} analysed"
            )

        # Annotation-driven subset: only defined when a strategy of type
        # ``signal_aware_fragment`` is configured (otherwise there's no
        # reference to pick the subset from). We match on ``type`` rather
        # than the config key so users can rename the strategy (e.g.
        # ``sigaw_mp30_cap40`` in default.yaml) without breaking this
        # plumbing. The subset applies to every strategy's column so the
        # comparison stays apples-to-apples — all strategies run on the
        # same spectrum list.
        ref = next(
            (
                name
                for name, cfg in self.strategies.items()
                if cfg.get("type") == "signal_aware_fragment"
            ),
            None,
        )
        if ref is not None:
            subset = [
                sr
                for sr in per_spectrum_results
                if self._repeat_fallback_used(sr.get(ref, {})) is False
            ]
            n_excluded = len(per_spectrum_results) - len(subset)
            if subset:
                per_strategy_ad = self._aggregate_per_strategy(
                    subset,
                    strategy_names,
                    spatial_bin_edges,
                    spatial_bin_centers,
                    log_fallback=False,
                )
                self.results["per_strategy_annotation_driven"] = per_strategy_ad
                self.results["annotation_driven_meta"] = {
                    "n_spectra": len(subset),
                    "n_fallback_excluded": n_excluded,
                    "reference_strategy": ref,
                }
                logger.info(
                    f"Annotation-driven subset: {len(subset)} spectra "
                    f"(excluded {n_excluded} where {ref} used fallback)"
                )
            else:
                logger.warning(
                    f"Annotation-driven subset is empty — "
                    f"{ref} fell back on every spectrum. "
                    "Skipping gold-standard view."
                )
        else:
            logger.info(
                "No signal_aware_fragment strategy configured; skipping "
                "annotation-driven (gold-standard) subset view."
            )

        return self.results

    @staticmethod
    def _repeat_fallback_used(strategy_data: Dict[str, Any]) -> Optional[bool]:
        """Return the first repeat's ``fallback_used`` flag, if present.

        Used to decide subset membership for the annotation-driven view.
        Returns None if the strategy did not run on this spectrum or its
        repeats lack the flag (e.g. uniform/thompson where fallback is
        undefined).
        """
        repeats = strategy_data.get("repeats", [])
        if not repeats:
            return None
        return repeats[0].get("fallback_used")

    def _aggregate_per_strategy(
        self,
        per_spectrum_subset: List[Dict[str, Dict[str, Any]]],
        strategy_names: List[str],
        spatial_bin_edges: np.ndarray,
        spatial_bin_centers: List[float],
        log_fallback: bool,
    ) -> Dict[str, Dict[str, Any]]:
        """Run the per-strategy aggregation over an arbitrary spectrum subset.

        Factored out of ``aggregate_results`` so the same logic can produce
        both the full-population and the annotation-driven views without
        drift. ``log_fallback`` suppresses the info line on the second pass
        (already logged during the first).
        """
        per_strategy: Dict[str, Dict[str, Any]] = {}
        for name in strategy_names:
            all_gap_results: List[Dict] = []
            all_masking_effects: List[Dict] = []
            all_mask_ratios: List[float] = []
            all_leakage: List[Dict] = []
            all_run_lengths: List[int] = []
            intensity_mask_curves: List[np.ndarray] = []
            spatial_mask_curves: List[np.ndarray] = []
            # Fallback split (signal_aware_fragment only): track repeat-level
            # mask ratios separately for annotation-driven vs thompson_span
            # fallback so the aggregate doesn't blur the two regimes.
            n_fallback = 0
            n_annotation_driven = 0
            mask_ratios_ann: List[float] = []
            mask_ratios_fb: List[float] = []

            for spectrum_result in per_spectrum_subset:
                if name not in spectrum_result:
                    continue
                viz_data = spectrum_result.get("_viz_data", {})
                valid_intensity = viz_data.get("valid_intensity")
                valid_mz = viz_data.get("valid_mz")

                strat = spectrum_result[name]
                for repeat in strat.get("repeats", []):
                    all_mask_ratios.append(repeat["mask_ratio"])
                    fb = repeat.get("fallback_used")
                    if fb is True:
                        n_fallback += 1
                        mask_ratios_fb.append(repeat["mask_ratio"])
                    elif fb is False:
                        n_annotation_driven += 1
                        mask_ratios_ann.append(repeat["mask_ratio"])
                    if repeat.get("gap_result"):
                        all_gap_results.append(repeat["gap_result"])
                    if repeat.get("masking_effect"):
                        all_masking_effects.append(repeat["masking_effect"])
                    if repeat.get("leakage"):
                        all_leakage.append(repeat["leakage"])

                    mlm_mask = repeat.get("mlm_mask_valid")
                    if mlm_mask is not None:
                        mlm_mask = np.asarray(mlm_mask, dtype=bool)
                        runs = self._compute_run_lengths(mlm_mask)
                        if len(runs) > 0:
                            all_run_lengths.extend(runs.tolist())
                        if (
                            valid_intensity is not None
                            and len(valid_intensity) == len(mlm_mask)
                        ):
                            curve = self._compute_intensity_mask_curve(
                                valid_intensity, mlm_mask
                            )
                            if curve is not None:
                                intensity_mask_curves.append(curve)
                        if (
                            valid_mz is not None
                            and len(valid_mz) == len(mlm_mask)
                        ):
                            curve = self._compute_spatial_mask_curve(
                                valid_mz, mlm_mask, spatial_bin_edges
                            )
                            if curve is not None:
                                spatial_mask_curves.append(curve)

            gap_analysis: Dict[str, Any] = {}
            if all_gap_results:
                # MaskingGapAnalyser keeps state between calls, so the
                # full-population pass runs first and the annotation-driven
                # pass will overwrite it — acceptable because the persisted
                # figures/CSVs are generated from the view the analyser saved
                # last. For JSON consumers the per-view gap summary is
                # attached below as ``gap_analysis``.
                gap_analysis = self.gap_analysers[name].aggregate_results(all_gap_results)

            masking_effect_agg = self._aggregate_masking_effects(all_masking_effects)

            mask_ratio_arr = np.array(all_mask_ratios) if all_mask_ratios else np.array([0.0])
            mask_ratio_stats = {
                "mean": float(mask_ratio_arr.mean()),
                "std": float(mask_ratio_arr.std()),
                "median": float(np.median(mask_ratio_arr)),
            }

            # Leakage stats: two complementary views, both retained so
            # consumers can pick the right one.
            #
            # * ``avg_leakage_ratio`` = mean over per-spectrum
            #   ``leakage_ratio``. Dispersion-aware but is NOT a simple
            #   decomposition into loss/isotope (so the pooled stack in
            #   the summary figure uses the other view).
            # * ``overall_leakage_ratio`` = total_with_leakage /
            #   total_masked_base (pooled across all spectra). Equal to
            #   ``leakage_by_type["loss"] + leakage_by_type["isotope"]``
            #   divided by total_masked_base, so it reconciles with the
            #   stacked bar chart.
            leakage_agg: Dict[str, Any] = {}
            if all_leakage:
                leakage_ratios = np.array([l["leakage_ratio"] for l in all_leakage])
                total_masked_base = sum(l["n_masked_base"] for l in all_leakage)
                total_with_leakage = sum(l["n_masked_base_with_leakage"] for l in all_leakage)
                total_loss_leak = sum(l["leakage_by_type"]["loss"] for l in all_leakage)
                total_isotope_leak = sum(l["leakage_by_type"]["isotope"] for l in all_leakage)
                leakage_agg = {
                    "avg_leakage_ratio": float(leakage_ratios.mean()),
                    "std_leakage_ratio": float(leakage_ratios.std()),
                    "total_masked_base": int(total_masked_base),
                    "total_with_leakage": int(total_with_leakage),
                    "overall_leakage_ratio": float(total_with_leakage) / max(total_masked_base, 1),
                    "leakage_by_type": {
                        "loss": int(total_loss_leak),
                        "isotope": int(total_isotope_leak),
                    },
                    "pooled_loss_fraction": (
                        float(total_loss_leak) / max(total_masked_base, 1)
                    ),
                    "pooled_isotope_fraction": (
                        float(total_isotope_leak) / max(total_masked_base, 1)
                    ),
                    "n_spectra_leakage_defined": int(len(all_leakage)),
                }

            behavior: Dict[str, Any] = {}
            if all_run_lengths:
                behavior["run_lengths"] = np.array(all_run_lengths)
            if intensity_mask_curves:
                stacked = np.vstack(intensity_mask_curves)
                behavior["intensity_mask_curve"] = {
                    "mean_rates": stacked.mean(axis=0).tolist(),
                    "std_rates": stacked.std(axis=0).tolist(),
                    "n_spectra": len(intensity_mask_curves),
                }
            if spatial_mask_curves:
                stacked = np.vstack(spatial_mask_curves)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    behavior["spatial_mask_curve"] = {
                        "mean_rates": np.nanmean(stacked, axis=0).tolist(),
                        "std_rates": np.nanstd(stacked, axis=0).tolist(),
                        "mz_bin_centers": spatial_bin_centers,
                        "n_spectra": len(spatial_mask_curves),
                    }

            fallback_summary: Dict[str, Any] = {}
            total_split = n_fallback + n_annotation_driven
            if total_split > 0:
                fallback_summary = {
                    "n_annotation_driven": n_annotation_driven,
                    "n_fallback": n_fallback,
                    "fallback_rate": n_fallback / total_split,
                    "mask_ratio_annotation_driven": (
                        float(np.mean(mask_ratios_ann)) if mask_ratios_ann else 0.0
                    ),
                    "mask_ratio_fallback": (
                        float(np.mean(mask_ratios_fb)) if mask_ratios_fb else 0.0
                    ),
                }

            per_strategy[name] = {
                "gap_analysis": gap_analysis,
                "masking_effect": masking_effect_agg,
                "leakage": leakage_agg,
                "mask_ratio_stats": mask_ratio_stats,
                "behavior": behavior,
                "fallback": fallback_summary,
                "n_spectra": len([
                    sr for sr in per_spectrum_subset if name in sr
                ]),
                "n_gap_records": len(all_gap_results),
                "n_masking_effects": len(all_masking_effects),
            }
            if (
                log_fallback
                and fallback_summary
                and fallback_summary["n_fallback"] > 0
            ):
                logger.info(
                    f"  [{name}] fallback used: {fallback_summary['n_fallback']}/"
                    f"{total_split} ({fallback_summary['fallback_rate']*100:.1f}%)"
                )

        return per_strategy

    @staticmethod
    def _compute_information_leakage(
        mlm_mask_valid: np.ndarray,
        feature_types: List[Optional[str]],
        parent_annotations: List[Optional[str]],
        matched_annotations: List[Optional[str]],
    ) -> Dict[str, Any]:
        """Compute information leakage: masked base ions with unmasked related peaks.

        A masked base ion "leaks" if any of its children (losses or isotopes
        sharing the same parent annotation) are unmasked, making the prediction
        trivially solvable (e.g., m/z of b3+ = m/z of b3-H2O+ + 18.01 Da).

        Args:
            mlm_mask_valid: ``[n_valid]`` bool — True for masked peaks.
            feature_types: Per-peak feature type (base/loss/isotope/precursor/None).
            parent_annotations: Per-peak parent annotation (set for losses/isotopes).
            matched_annotations: Per-peak annotation strings.

        Returns:
            Dict with ``n_masked_base``, ``n_masked_base_with_leakage``,
            ``leakage_ratio``, ``leakage_by_type``.
        """
        n_valid = len(mlm_mask_valid)

        # Build base_annotation → child indices mapping
        base_to_children: Dict[str, List[int]] = {}
        base_indices: Dict[str, int] = {}

        for i in range(n_valid):
            ft = feature_types[i] if i < len(feature_types) else None
            ann = matched_annotations[i] if i < len(matched_annotations) else None
            parent = parent_annotations[i] if i < len(parent_annotations) else None

            # Skip precursor-related peaks
            if ft == "precursor":
                continue

            if ft == "base" and ann:
                base_indices[ann] = i
                if ann not in base_to_children:
                    base_to_children[ann] = []
            elif ft in ("loss", "isotope") and parent:
                if parent not in base_to_children:
                    base_to_children[parent] = []
                base_to_children[parent].append(i)

        # Count masked base ions with unmasked children
        n_masked_base = 0
        n_masked_base_with_leakage = 0
        n_leakage_by_type: Dict[str, int] = {"loss": 0, "isotope": 0}

        for base_ann, base_idx in base_indices.items():
            if not mlm_mask_valid[base_idx]:
                continue  # Base is unmasked — no leakage concern
            n_masked_base += 1

            children = base_to_children.get(base_ann, [])
            has_loss_leak = False
            has_isotope_leak = False

            for child_idx in children:
                if not mlm_mask_valid[child_idx]:
                    # Child is unmasked → information leakage
                    child_ft = feature_types[child_idx] if child_idx < len(feature_types) else None
                    if child_ft == "loss":
                        has_loss_leak = True
                    elif child_ft == "isotope":
                        has_isotope_leak = True

            if has_loss_leak or has_isotope_leak:
                n_masked_base_with_leakage += 1
            if has_loss_leak:
                n_leakage_by_type["loss"] += 1
            if has_isotope_leak:
                n_leakage_by_type["isotope"] += 1

        leakage_ratio = float(n_masked_base_with_leakage) / max(n_masked_base, 1)

        return {
            "n_masked_base": n_masked_base,
            "n_masked_base_with_leakage": n_masked_base_with_leakage,
            "leakage_ratio": leakage_ratio,
            "leakage_by_type": n_leakage_by_type,
        }

    @staticmethod
    def _aggregate_masking_effects(effects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Average masking-effect dicts across spectra."""
        if not effects:
            return {}

        float_keys = [
            "annotated_mask_ratio",
            "unannotated_mask_ratio",
            "overall_mask_ratio",
            "annotated_preservation_ratio",
            "annotated_fraction_of_masked_peaks",
            "unannotated_fraction_of_masked_peaks",
            "annotated_intensity_fraction_of_masked",
            "unannotated_intensity_fraction_of_masked",
            "annotated_intensity_masked_fraction_of_annotated",
            "unannotated_intensity_masked_fraction_of_unannotated",
            # Primary view (N/C pair per frag_type, e.g. b/y for HCD).
            "fragment_group_full_mask_ratio",
            "fragment_group_avg_mask_fraction",
            "fragment_group_weighted_mask_fraction",
            # "All ion types" diagnostic view — includes a-ions for
            # collisional activation and every non-primary series that
            # the theoretical generator produced. Lets the report
            # show the delta between "what signal_aware could target"
            # and "every annotated group the matcher saw".
            "fragment_group_full_mask_ratio_all",
            "fragment_group_avg_mask_fraction_all",
            "fragment_group_weighted_mask_fraction_all",
            "precursor_group_mask_fraction",
            "precursor_mask_budget_fraction",
            "precursor_intensity_budget_fraction",
            "precursor_fraction_of_annotated_masked",
            "precursor_intensity_fraction_of_annotated_masked",
            "custom_mask_ratio",
            "custom_mask_budget_fraction",
            "custom_intensity_budget_fraction",
        ]

        agg: Dict[str, Any] = {}
        for key in float_keys:
            vals = [e.get(key, 0.0) for e in effects if key in e]
            if vals:
                agg[f"avg_{key}"] = float(np.mean(vals))
                agg[f"std_{key}"] = float(np.std(vals))

        # Aggregate integer count keys (sum and mean)
        int_count_keys = [
            # Primary view
            "n_fragment_groups_total",
            "n_fragment_groups_fully_masked",
            "n_fragment_groups_partially_masked",
            "n_fragment_groups_unmasked",
            "n_fragment_groups_any_masked",
            # All-types view
            "n_fragment_groups_total_all",
            "n_fragment_groups_fully_masked_all",
            "n_fragment_groups_partially_masked_all",
            "n_fragment_groups_unmasked_all",
            "precursor_n_peaks",
            "precursor_n_masked",
            "custom_n_peaks",
            "custom_n_masked",
        ]
        for key in int_count_keys:
            vals = [e.get(key, 0) for e in effects if key in e]
            if vals:
                agg[f"total_{key}"] = int(sum(vals))
                agg[f"avg_{key}"] = float(np.mean(vals))

        # Aggregate boolean frequency keys
        bool_keys = [
            "precursor_base_masked",
            "precursor_fully_masked",
            "precursor_has_leakage",
        ]
        for key in bool_keys:
            vals = [e.get(key, False) for e in effects if key in e]
            if vals:
                agg[f"freq_{key}"] = float(sum(1 for v in vals if v) / len(vals))

        # Aggregate per-ion-series group masking
        series_total_agg: Dict[str, int] = {}
        series_fully_agg: Dict[str, int] = {}
        series_partially_agg: Dict[str, int] = {}
        series_unmasked_agg: Dict[str, int] = {}
        for e in effects:
            for series, count in e.get("series_group_total", {}).items():
                series_total_agg[series] = series_total_agg.get(series, 0) + count
            for series, count in e.get("series_group_fully_masked", {}).items():
                series_fully_agg[series] = series_fully_agg.get(series, 0) + count
            for series, count in e.get("series_group_partially_masked", {}).items():
                series_partially_agg[series] = series_partially_agg.get(series, 0) + count
            for series, count in e.get("series_group_unmasked", {}).items():
                series_unmasked_agg[series] = series_unmasked_agg.get(series, 0) + count
        series_group_full_mask_ratio = {
            series: series_fully_agg.get(series, 0) / max(total, 1)
            for series, total in series_total_agg.items()
        }
        agg["series_group_total"] = series_total_agg
        agg["series_group_fully_masked"] = series_fully_agg
        agg["series_group_partially_masked"] = series_partially_agg
        agg["series_group_unmasked"] = series_unmasked_agg
        agg["series_group_full_mask_ratio"] = series_group_full_mask_ratio

        # Distributional data for fragment group masking status
        # Per-spectrum fully-masked group counts (for box/violin plots)
        per_spectrum_fully_masked = [
            e.get("n_fragment_groups_fully_masked", 0) for e in effects
        ]
        per_spectrum_partially_masked = [
            e.get("n_fragment_groups_partially_masked", 0) for e in effects
        ]
        per_spectrum_unmasked = [
            e.get("n_fragment_groups_unmasked", 0) for e in effects
        ]
        agg["dist_n_fully_masked"] = per_spectrum_fully_masked
        agg["dist_n_partially_masked"] = per_spectrum_partially_masked
        agg["dist_n_unmasked"] = per_spectrum_unmasked

        # Per-group mask fractions for partially-masked groups (for histogram)
        all_partial_fractions: List[float] = []
        for e in effects:
            for frac in e.get("group_mask_fractions", []):
                if 0.0 < frac < 1.0:
                    all_partial_fractions.append(frac)
        agg["dist_partial_group_fractions"] = all_partial_fractions

        # Aggregate per-type totals (excluding precursor-related labels)
        type_total: Dict[str, int] = {}
        type_masked: Dict[str, int] = {}
        for e in effects:
            for label, count in e.get("annotated_type_total", {}).items():
                if label in MaskingAnalyser._EXCLUDED_LABELS:
                    continue
                type_total[label] = type_total.get(label, 0) + count
            for label, count in e.get("annotated_type_masked", {}).items():
                if label in MaskingAnalyser._EXCLUDED_LABELS:
                    continue
                type_masked[label] = type_masked.get(label, 0) + count

        type_mask_ratio = {
            label: masked / max(type_total.get(label, 0), 1)
            for label, masked in type_masked.items()
        }
        agg["annotated_type_total"] = type_total
        agg["annotated_type_masked"] = type_masked
        agg["annotated_type_mask_ratio"] = type_mask_ratio

        # Aggregate per-type intensity (excluding precursor-related labels)
        type_intensity_total: Dict[str, float] = {}
        type_intensity_masked: Dict[str, float] = {}
        for e in effects:
            for label, intensity in e.get("annotated_type_intensity_total", {}).items():
                if label in MaskingAnalyser._EXCLUDED_LABELS:
                    continue
                type_intensity_total[label] = type_intensity_total.get(label, 0.0) + intensity
            for label, intensity in e.get("annotated_type_intensity_masked", {}).items():
                if label in MaskingAnalyser._EXCLUDED_LABELS:
                    continue
                type_intensity_masked[label] = type_intensity_masked.get(label, 0.0) + intensity
        type_intensity_mask_ratio = {
            label: type_intensity_masked.get(label, 0.0) / max(total, 1e-12)
            for label, total in type_intensity_total.items()
        }
        agg["annotated_type_intensity_total"] = type_intensity_total
        agg["annotated_type_intensity_masked"] = type_intensity_masked
        agg["annotated_type_intensity_mask_ratio"] = type_intensity_mask_ratio

        return agg

    @staticmethod
    def _compute_run_lengths(mask: np.ndarray) -> np.ndarray:
        """Compute lengths of consecutive True runs in a boolean mask."""
        mask = np.asarray(mask, dtype=bool)
        if len(mask) == 0 or not mask.any():
            return np.array([], dtype=int)
        padded = np.concatenate([[False], mask.astype(bool), [False]])
        diffs = np.diff(padded.astype(int))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        return ends - starts

    @staticmethod
    def _compute_intensity_mask_curve(
        intensity: np.ndarray, mask: np.ndarray, n_bins: int = 10
    ) -> Optional[np.ndarray]:
        """Compute mask rate by intensity percentile bin.

        Sorts peaks by intensity and splits into ``n_bins`` equal-count
        groups, returning the mask rate in each group.
        """
        n = len(intensity)
        if n < n_bins:
            return None
        order = np.argsort(intensity)
        sorted_mask = mask[order]
        bins = np.array_split(sorted_mask, n_bins)
        return np.array([b.mean() for b in bins])

    @staticmethod
    def _compute_spatial_mask_curve(
        mz: np.ndarray, mask: np.ndarray, bin_edges: np.ndarray
    ) -> Optional[np.ndarray]:
        """Compute mask rate by m/z bin.

        Returns an array of length ``len(bin_edges) - 1``, one mask rate
        per m/z bin.  Bins with no peaks are set to NaN.
        """
        n_bins = len(bin_edges) - 1
        if len(mz) == 0:
            return None
        bin_idx = np.clip(np.digitize(mz, bin_edges) - 1, 0, n_bins - 1)
        rates = np.full(n_bins, np.nan)
        for b in range(n_bins):
            in_bin = bin_idx == b
            if in_bin.any():
                rates[b] = mask[in_bin].mean()
        return rates

    @staticmethod
    def _build_comparison_tables(
        per_strategy: Dict[str, Dict[str, Any]],
        strategy_names: List[str],
    ) -> Dict[str, Any]:
        """Build cross-strategy comparison DataFrames."""
        # Mask ratio comparison
        mask_rows = []
        for name in strategy_names:
            stats = per_strategy.get(name, {}).get("mask_ratio_stats", {})
            mask_rows.append({"strategy": name, **stats})
        mask_ratio_df = pd.DataFrame(mask_rows).set_index("strategy") if mask_rows else pd.DataFrame()

        # Gap metrics comparison
        gap_rows = []
        for name in strategy_names:
            gap = per_strategy.get(name, {}).get("gap_analysis", {})
            summary = gap.get("summary", {})
            nearest = summary.get("nearest_distance_da", {})
            total = summary.get("total_gap_da", {})
            coverage = gap.get("group_coverage", {})
            gap_rows.append({
                "strategy": name,
                "nearest_da_mean": nearest.get("mean", np.nan),
                "nearest_da_median": nearest.get("median", np.nan),
                "total_gap_da_mean": total.get("mean", np.nan),
                "gap_within_3_groups": coverage.get("within_3_groups", np.nan),
                "gap_within_10_groups": coverage.get("within_10_groups", np.nan),
            })
        gap_df = pd.DataFrame(gap_rows).set_index("strategy") if gap_rows else pd.DataFrame()

        # Annotated interaction comparison
        ann_rows = []
        for name in strategy_names:
            eff = per_strategy.get(name, {}).get("masking_effect", {})
            ann_rows.append({
                "strategy": name,
                "annotated_mask_ratio": eff.get("avg_annotated_mask_ratio", np.nan),
                "unannotated_mask_ratio": eff.get("avg_unannotated_mask_ratio", np.nan),
                "annotated_preservation": eff.get("avg_annotated_preservation_ratio", np.nan),
                "frag_group_full_mask": eff.get("avg_fragment_group_full_mask_ratio", np.nan),
                "avg_frag_groups": eff.get("avg_n_fragment_groups_total", np.nan),
                "frag_groups_fully_masked": eff.get("avg_n_fragment_groups_fully_masked", np.nan),
                "frag_groups_partially_masked": eff.get("avg_n_fragment_groups_partially_masked", np.nan),
                "frag_groups_unmasked": eff.get("avg_n_fragment_groups_unmasked", np.nan),
            })
        ann_df = pd.DataFrame(ann_rows).set_index("strategy") if ann_rows else pd.DataFrame()

        # Training signal comparison (key metrics for strategy selection)
        signal_rows = []
        for name in strategy_names:
            eff = per_strategy.get(name, {}).get("masking_effect", {})
            leak = per_strategy.get(name, {}).get("leakage", {})
            mr = per_strategy.get(name, {}).get("mask_ratio_stats", {})
            signal_rows.append({
                "strategy": name,
                "mask_ratio": mr.get("mean", np.nan),
                "annotated_fraction_of_masked": eff.get(
                    "avg_annotated_fraction_of_masked_peaks", np.nan
                ),
                "annotated_preservation": eff.get(
                    "avg_annotated_preservation_ratio", np.nan
                ),
                # Per-spectrum mean (sensitive to dispersion). For the
                # pooled total_leaks/total_masked_base ratio see
                # ``leakage_ratio_pooled``.
                "leakage_ratio_per_spectrum_mean": leak.get(
                    "avg_leakage_ratio", np.nan
                ),
                "leakage_ratio_pooled": leak.get(
                    "overall_leakage_ratio", np.nan
                ),
                "precursor_mask_freq": eff.get("freq_precursor_base_masked", np.nan),
                "precursor_budget_frac": eff.get("avg_precursor_mask_budget_fraction", np.nan),
                "precursor_int_budget_frac": eff.get("avg_precursor_intensity_budget_fraction", np.nan),
                "precursor_frac_of_ann_masked": eff.get("avg_precursor_fraction_of_annotated_masked", np.nan),
                "precursor_int_frac_of_ann_masked": eff.get("avg_precursor_intensity_fraction_of_annotated_masked", np.nan),
                "custom_budget_frac": eff.get("avg_custom_mask_budget_fraction", np.nan),
                "custom_int_budget_frac": eff.get("avg_custom_intensity_budget_fraction", np.nan),
                "precursor_leakage_freq": eff.get("freq_precursor_has_leakage", np.nan),
            })
        signal_df = (
            pd.DataFrame(signal_rows).set_index("strategy")
            if signal_rows
            else pd.DataFrame()
        )

        return {
            "mask_ratio": mask_ratio_df,
            "gap_metrics": gap_df,
            "annotated_interaction": ann_df,
            "training_signal": signal_df,
        }

    def _build_stratified_masking_summary(
        self,
        per_spectrum_results: List[Dict[str, Dict[str, Any]]],
        strategy_names: List[str],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Build stratified masking summary by metadata keys.

        For each metadata key (frag_type, precursor_charge, search_instrument)
        and each group value, aggregate masking-effect metrics across
        spectra for **every** strategy. Previously only the first
        strategy was reported, which hid cross-strategy differences
        within a stratum (e.g. "does thompson_span behave differently
        from signal_aware on ETD?").

        Returns
        -------
        ``{metadata_key: {group_value: {
                "n_total": int,                # spectra in this stratum
                "by_strategy": {
                    <strategy_name>: {
                        "n": int,              # spectra with masking_effect
                        "avg_annotated_mask_ratio": float,
                        "avg_unannotated_mask_ratio": float,
                        "avg_annotated_intensity_masked_frac": float,
                    }, ...
                },
            }, ...}}``

        The top-level shape stays the same; the per-group dict now has
        ``by_strategy`` instead of flat metric keys.
        """
        strat_keys = ["frag_type", "precursor_charge", "search_instrument"]
        if not strategy_names:
            return {}

        # grouped[metadata_key][group_value][strategy_name] = [effect dicts]
        grouped: Dict[str, Dict[str, Dict[str, List[Dict]]]] = {
            k: {} for k in strat_keys
        }
        total_counts: Dict[str, Dict[str, int]] = {k: {} for k in strat_keys}

        for spectrum_result in per_spectrum_results:
            viz = spectrum_result.get("_viz_data", {})
            metadata = viz.get("metadata", {})

            # Only count the spectrum once per stratum key, regardless
            # of how many strategies it ran through.
            any_strategy_ran = any(
                spectrum_result.get(n, {}).get("repeats") for n in strategy_names
            )
            if not any_strategy_ran:
                continue
            for key in strat_keys:
                val = str(metadata.get(key, "unknown") or "unknown")
                total_counts[key][val] = total_counts[key].get(val, 0) + 1

            for strategy in strategy_names:
                strat_data = spectrum_result.get(strategy, {})
                repeats = strat_data.get("repeats", [])
                if not repeats:
                    continue
                effect = repeats[0].get("masking_effect", {})
                if not effect:
                    continue
                for key in strat_keys:
                    val = str(metadata.get(key, "unknown") or "unknown")
                    grouped[key].setdefault(val, {}).setdefault(strategy, []).append(
                        effect
                    )

        # Log coverage per stratum (total vs annotated, summed across strategies
        # for the headline number — individual strategy n appears in the JSON).
        for key in strat_keys:
            all_vals = sorted(total_counts[key].items(), key=lambda kv: kv[1], reverse=True)
            logger.info(
                f"Stratified '{key}': total spectra by group: "
                f"{', '.join(f'{v}={c}' for v, c in all_vals)}"
            )

        result: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for key in strat_keys:
            key_result: Dict[str, Dict[str, Any]] = {}
            for group_val, total_n in total_counts[key].items():
                if total_n < 5:
                    continue
                strat_dict = grouped[key].get(group_val, {})
                by_strategy: Dict[str, Dict[str, Any]] = {}
                for strategy, effects in strat_dict.items():
                    if not effects:
                        continue
                    ann_mr = [e.get("annotated_mask_ratio", 0) for e in effects]
                    unann_mr = [e.get("unannotated_mask_ratio", 0) for e in effects]
                    ann_int_frac = [
                        e.get("annotated_intensity_masked_fraction_of_annotated", 0)
                        for e in effects
                    ]
                    by_strategy[strategy] = {
                        "n": len(effects),
                        "avg_annotated_mask_ratio": float(np.mean(ann_mr)),
                        "avg_unannotated_mask_ratio": float(np.mean(unann_mr)),
                        "avg_annotated_intensity_masked_frac": float(np.mean(ann_int_frac)),
                    }
                if by_strategy:
                    key_result[group_val] = {
                        "n_total": total_n,
                        "by_strategy": by_strategy,
                    }
            result[key] = key_result

        return result

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def generate_visualizations(self) -> None:
        """Generate all comparison and per-strategy visualisations."""
        if not self.results:
            return

        self._generate_masking_summary()
        # Gold-standard companion view: only runs if aggregate_results
        # produced the annotation-driven subset (requires
        # signal_aware_fragment in the strategy list with at least one
        # non-fallback spectrum).
        if self.results.get("per_strategy_annotation_driven"):
            self._generate_masking_summary_annotation_driven()
        self._generate_fragment_ion_analysis()
        self._generate_fragment_group_status()
        self._generate_training_signal_analysis()
        self._generate_prediction_difficulty()
        self._generate_fragment_group_analysis()
        self._generate_precursor_analysis()
        self._generate_stratified_masking_summary()

        # Per-strategy detail figures
        self._generate_per_strategy_behavior()
        for name, analyser in self.gap_analysers.items():
            if analyser.results:
                analyser.generate_group_size_sweep()

    # ------------------------------------------------------------------
    # Individual spectrum visualisation
    # ------------------------------------------------------------------

    # 4-color scheme for masking × annotation
    _VIZ_COLORS = {
        "unmasked_annotated": "#2980b9",   # Blue — learnable context
        "unmasked_unannotated": "#95a5a6", # Gray — unannotated context
        "masked_annotated": "#c0392b",     # Red — learnable target
        "masked_unannotated": "#e67e22",   # Orange — unannotated target
    }

    def generate_individual_spectrum_visualizations(
        self,
        per_spectrum_results: List[Dict[str, Dict[str, Any]]],
        max_viz: int = 30,
    ) -> None:
        """Generate per-spectrum masking comparison and detail figures.

        Produces:
        1. Multi-row comparison figures (one per spectrum, all strategies).
        2. Per-strategy detail figures (one per spectrum per strategy).

        Args:
            per_spectrum_results: Raw per-spectrum dicts from ``analyze_spectrum``.
            max_viz: Maximum number of spectra to visualise.
        """
        # Select spectra with annotation data
        candidates = [
            r for r in per_spectrum_results
            if "_viz_data" in r and "annotated_mask" in r.get("_viz_data", {})
        ]
        if not candidates:
            logger.info("No annotated spectra available for individual masking visualisation")
            return

        n_viz = min(max_viz, len(candidates))
        indices = np.linspace(0, len(candidates) - 1, n_viz, dtype=int)
        selected = [candidates[i] for i in indices]
        strategy_names = list(self.strategies.keys())

        # Create comparison dir
        comparison_dir = self.output_dir / "individual_spectra_masking_strategy_comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        # Per-strategy dirs
        for name in strategy_names:
            (self.output_dir / name / "individual_spectra").mkdir(parents=True, exist_ok=True)

        for idx, spectrum_result in enumerate(selected):
            viz = spectrum_result["_viz_data"]
            metadata = viz.get("metadata", {})
            seq = metadata.get("clean_sequence") or metadata.get("sequence") or "unknown"
            seq_slug = sanitize_filename(seq)[:30]

            try:
                self._create_comparison_figure(
                    idx, seq_slug, spectrum_result, strategy_names, comparison_dir
                )
            except Exception as e:
                logger.warning(f"Comparison figure failed for spectrum {idx}: {e}")

            for name in strategy_names:
                try:
                    self._create_per_strategy_figure(
                        idx, seq_slug, name, spectrum_result,
                        self.output_dir / name / "individual_spectra",
                    )
                except Exception as e:
                    logger.warning(f"Per-strategy figure failed ({name}, spectrum {idx}): {e}")

        logger.info(
            f"Individual masking visualisations saved: {n_viz} comparison + "
            f"{n_viz * len(strategy_names)} per-strategy figures"
        )

    # Ion-type colour scheme — matches spectrum_analyser.py category_colors exactly.
    # Format: category → (hex_colour, alpha)
    _CATEGORY_COLORS = {
        "unannotated": ("#BDBDBD", 0.5),
        "B-ion": ("#1f77b4", 1.0),
        "Y-ion": ("#d62728", 1.0),
        "A-ion": ("#2ca02c", 0.9),
        "C-ion": ("#9467bd", 0.9),
        "X-ion": ("#8c564b", 0.9),
        "Z-ion": ("#e377c2", 0.9),
        "Precursor": ("#000000", 1.0),
        "Precursor (Isotope)": ("#4a4a4a", 0.8),
        "B-ion (Loss)": ("#1f77b4", 0.6),
        "Y-ion (Loss)": ("#d62728", 0.6),
        "A-ion (Loss)": ("#2ca02c", 0.6),
        "C-ion (Loss)": ("#9467bd", 0.6),
        "X-ion (Loss)": ("#8c564b", 0.6),
        "Z-ion (Loss)": ("#e377c2", 0.6),
        "Loss": ("#2ca02c", 0.6),
        "B-ion (Isotope)": ("#1f77b4", 0.5),
        "Y-ion (Isotope)": ("#d62728", 0.5),
        "A-ion (Isotope)": ("#2ca02c", 0.5),
        "C-ion (Isotope)": ("#9467bd", 0.5),
        "X-ion (Isotope)": ("#8c564b", 0.5),
        "Z-ion (Isotope)": ("#e377c2", 0.5),
        "Isotope": ("#7f7f7f", 0.5),
        "Immonium": ("#8c564b", 0.9),
        "Glycan": ("#e377c2", 0.9),
        "Phospho": ("#ff7f0e", 0.9),
        "Reporter": ("#17becf", 0.9),
        "Sulfate": ("#ff7f0e", 0.8),
        "Custom": ("#9467bd", 0.8),
        "Other": ("#bcbd22", 0.7),
    }

    # Darkened text colours for annotation labels — matches spectrum_analyser.py text_colors.
    _TEXT_COLORS = {
        "unannotated": "olive",
        "B-ion": "darkblue", "B-ion (Loss)": "darkblue", "B-ion (Isotope)": "darkblue",
        "Y-ion": "darkred", "Y-ion (Loss)": "darkred", "Y-ion (Isotope)": "darkred",
        "A-ion": "darkgreen", "A-ion (Loss)": "darkgreen", "A-ion (Isotope)": "darkgreen",
        "C-ion": "darkviolet", "C-ion (Loss)": "darkviolet", "C-ion (Isotope)": "darkviolet",
        "X-ion": "saddlebrown", "X-ion (Loss)": "saddlebrown", "X-ion (Isotope)": "saddlebrown",
        "Z-ion": "deeppink", "Z-ion (Loss)": "deeppink", "Z-ion (Isotope)": "deeppink",
        "Precursor": "black", "Precursor (Isotope)": "#4a4a4a",
        "Loss": "darkgreen", "Isotope": "dimgray",
        "Immonium": "saddlebrown", "Glycan": "deeppink",
        "Phospho": "darkorange", "Sulfate": "darkorange",
        "Reporter": "darkcyan", "Custom": "darkviolet", "Other": "olive",
    }

    @staticmethod
    def _classify_peak_colour(
        annotation: Optional[str],
        feature_type: Optional[str],
    ) -> str:
        """Classify a peak into a colour category based on annotation and feature type."""
        if not annotation:
            return "unannotated"

        ion_char = annotation[0].lower() if annotation else ""
        is_child = feature_type in ("loss", "isotope")

        if ion_char == "b":
            return "b_child" if is_child else "b_base"
        if ion_char == "y":
            return "y_child" if is_child else "y_base"
        # Other annotated ions (a, c, x, z, etc.)
        if annotation:
            return "other_child" if is_child else "other_ann"
        return "unannotated"

    @staticmethod
    def _categorize_ion(annotation: str) -> str:
        """Categorize ion annotation into a colour category.

        Replicates ``spectrum_analyser.categorize_ion()`` so that both
        analysers produce identical colour schemes.
        """
        if not annotation:
            return "unannotated"

        # Precursor ions (PSI mzPAF p^z or p-loss^z format)
        if annotation.startswith("p^") or annotation.startswith("p-"):
            if "[+" in annotation and "]" in annotation:
                return "Precursor (Isotope)"
            return "Precursor"

        # Isotope notation from conditional annotation: "b3+[+1]", "y5++[+2]"
        if "[+" in annotation and "]" in annotation:
            base_ion = annotation.split("[")[0]
            if base_ion and base_ion[0].lower() in "byacxz":
                return base_ion[0].upper() + "-ion (Isotope)"
            return "Isotope"

        # Custom ions: "custom:glycan_HexNAc@204.0867"
        if annotation.startswith("custom:"):
            parts = annotation.split(":")
            if len(parts) > 1:
                ion_name = parts[1].split("@")[0]
                if "glycan" in ion_name:
                    return "Glycan"
                elif "phospho" in ion_name or "PO3" in ion_name:
                    return "Phospho"
                elif "immonium" in ion_name:
                    return "Immonium"
                elif "TMT" in ion_name or "iTRAQ" in ion_name:
                    return "Reporter"
                elif "sulfate" in ion_name:
                    return "Sulfate"
                else:
                    return "Custom"

        # Neutral losses
        has_loss = False
        if "-" in annotation:
            parts = annotation.split("-")
            if len(parts) == 2:
                loss_part = parts[1].split("+")[0].split("++")[0]
                if any(x in loss_part.upper() for x in ["H2O", "NH3", "CO", "H3PO4", "H3PO3"]):
                    has_loss = True
                    base_ion = parts[0]
                    if base_ion and base_ion[0].lower() in "byacxz":
                        return base_ion[0].upper() + "-ion (Loss)"

        # Standard fragment ions (b, y, a, c, x, z)
        if annotation[0].lower() in "byacxz":
            ion_type = annotation[0].upper() + "-ion"
            if has_loss:
                return ion_type + " (Loss)"
            return ion_type

        return "Other"

    @staticmethod
    def _merge_custom_ion_annotations(
        valid_mz: np.ndarray,
        annotations: List[str],
        custom_ion_data: Dict[str, Any],
        ppm_tol: float = 10.0,
    ) -> List[str]:
        """Merge custom ion detections into the annotations list.

        Unannotated peaks that match a custom ion m/z (within tolerance)
        receive a ``custom:<group>@<mz>`` label.  Already-annotated peaks
        are left untouched.  Replicates the merging logic from
        ``spectrum_analyser.py``.
        """
        annotations = list(annotations)  # defensive copy
        custom_detection = custom_ion_data.get("detection", {})
        if not custom_detection or len(valid_mz) == 0:
            return annotations

        def _annotate_custom_peak(mz_val: float, label: str) -> None:
            if len(valid_mz) == 0:
                return
            idx = int(np.argmin(np.abs(valid_mz - mz_val)))
            if abs(valid_mz[idx] - mz_val) / max(mz_val, 1e-12) * 1e6 < ppm_tol * 2:
                if idx < len(annotations) and annotations[idx]:
                    return  # already annotated
                while len(annotations) <= idx:
                    annotations.append("")
                annotations[idx] = label

        for group_name, group_data in custom_detection.items():
            if not group_data.get("found"):
                continue
            for matched_mz_val in group_data.get("matched_mz", []):
                _annotate_custom_peak(matched_mz_val, f"custom:{group_name}@{matched_mz_val:.4f}")
            for iso_match in group_data.get("isotope_matches", []):
                iso_mz = iso_match["matched_mz"]
                iso_num = iso_match["isotope_num"]
                _annotate_custom_peak(iso_mz, f"custom:{group_name}[+{iso_num}]@{iso_mz:.4f}")

        return annotations

    @staticmethod
    def _format_custom_ion_label(annotation: str) -> str:
        """Format a ``custom:…`` annotation for display.

        ``"custom:glycan_HexNAc@204.0867"`` → ``"HexNAc"``
        """
        parts = annotation.split(":")
        if len(parts) > 1:
            ion_name = parts[1].split("@")[0]
            return ion_name.split("_", 1)[1] if "_" in ion_name else ion_name
        return annotation

    def _create_comparison_figure(
        self,
        idx: int,
        seq_slug: str,
        spectrum_result: Dict[str, Any],
        strategy_names: List[str],
        output_dir: Path,
    ) -> None:
        """Multi-row comparison using index-based patterns for easy visual comparison.

        Row 0: Annotated reference (index-based, colour-coded by ion type).
        Rows 1+: Per-strategy masking pattern (index-based, masked vs unmasked).
        """
        viz = spectrum_result["_viz_data"]
        valid_mz = np.asarray(viz["valid_mz"], dtype=np.float64)
        valid_intensity = np.asarray(viz["valid_intensity"], dtype=np.float64)
        annotated_mask = viz.get("annotated_mask", np.zeros(len(valid_mz), dtype=bool))
        if not isinstance(annotated_mask, np.ndarray):
            annotated_mask = np.asarray(annotated_mask, dtype=bool)
        annotations = list(viz.get("theo_annotations", []))
        feature_types = viz.get("feature_types", [])
        custom_ion_data = viz.get("custom_ion_data", {})

        # Merge custom ion detections into annotations
        annotations = self._merge_custom_ion_annotations(
            valid_mz, annotations, custom_ion_data,
        )

        # Publication mode: optionally restrict to a readable subset of strategies.
        if self.publication_figure and self.publication_strategies:
            strategy_names = [s for s in self.publication_strategies if s in spectrum_result]

        n_valid = len(valid_mz)
        n_strategies = len(strategy_names)
        n_rows = 1 + n_strategies
        fig, axes = plt.subplots(
            n_rows, 1,
            figsize=(18, 2.5 + 2.8 * n_strategies),
            sharex=True,
        )
        if n_rows == 1:
            axes = [axes]

        metadata = viz.get("metadata", {})
        seq = metadata.get("clean_sequence") or metadata.get("sequence") or "unknown"
        if not self.publication_figure:
            fig.suptitle(f"Masking Strategy Comparison: {seq}", fontsize=14, fontweight="bold")

        x_pos = np.arange(n_valid)

        # Row 0: Annotated reference (index-based, colour-coded)
        ax = axes[0]
        # Group peaks by colour category for batch plotting
        colour_groups: Dict[str, List[int]] = {}
        for i in range(n_valid):
            ann = annotations[i] if i < len(annotations) else ""
            cat = self._categorize_ion(ann) if ann else "unannotated"
            colour_groups.setdefault(cat, []).append(i)

        # Plot order: unannotated first (background), then sorted annotated categories
        if "unannotated" in colour_groups:
            unannotated_indices = colour_groups.pop("unannotated")
            idx_arr = np.array(unannotated_indices)
            color, alpha = self._CATEGORY_COLORS["unannotated"]
            ax.bar(x_pos[idx_arr], valid_intensity[idx_arr],
                   color=color, alpha=alpha,
                   width=1.0, linewidth=0, label="Unannotated")

        for cat in sorted(colour_groups.keys()):
            indices = colour_groups[cat]
            if not indices:
                continue
            idx_arr = np.array(indices)
            color, alpha = self._CATEGORY_COLORS.get(cat, ("#9467bd", 0.8))
            ax.bar(x_pos[idx_arr], valid_intensity[idx_arr],
                   color=color, alpha=alpha,
                   width=1.0, linewidth=0, label=cat)

        # Add annotations for all annotated peaks
        for i in range(n_valid):
            ann = annotations[i] if i < len(annotations) else ""
            if not ann:
                continue
            cat = self._categorize_ion(ann)
            if cat == "unannotated":
                continue
            display_ann = self._format_custom_ion_label(ann) if ann.startswith("custom:") else ann
            ax.annotate(
                display_ann, xy=(i, valid_intensity[i]), xytext=(0, 3),
                textcoords="offset points", ha="center", fontsize=5,
                rotation=90, alpha=0.9, fontweight="bold",
                color=self._TEXT_COLORS.get(cat, "black"),
            )

        if self.publication_figure:
            ax.set_ylabel("Annotated\nreference", fontsize=12, fontweight="bold")
        else:
            ax.set_title("Annotated Reference Spectrum (index-based)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Intensity")
        ax.legend(fontsize=(12 if self.publication_figure else 7), loc="upper right", ncol=2)
        ax.grid(True, alpha=0.3, axis="y")

        # Strategy rows: index-based masking pattern
        for row_i, name in enumerate(strategy_names, start=1):
            ax = axes[row_i]
            strat_data = spectrum_result.get(name, {})
            repeats = strat_data.get("repeats", [])
            if not repeats:
                ax.text(0.5, 0.5, f"{name}: no data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            rep = repeats[0]
            mlm_mask = rep.get("mlm_mask_valid")
            if mlm_mask is None:
                ax.text(0.5, 0.5, f"{name}: no mask data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            mlm_mask = np.asarray(mlm_mask, dtype=bool)
            unmasked_idx = ~mlm_mask
            masked_idx = mlm_mask
            if unmasked_idx.any():
                ax.bar(x_pos[unmasked_idx], valid_intensity[unmasked_idx],
                       color="#90CAF9", alpha=0.8, width=1.0, linewidth=0,
                       label="Unmasked")
            if masked_idx.any():
                ax.bar(x_pos[masked_idx], valid_intensity[masked_idx],
                       color="#E53935", alpha=0.8, width=1.0, linewidth=0,
                       label="Masked")

            if self.publication_figure:
                ax.set_ylabel(self.publication_labels.get(name, name), fontsize=11, fontweight="bold")
            else:
                ax.set_title(name, fontsize=11, fontweight="bold", loc="left")
                ax.set_ylabel("Intensity")
            ax.legend(fontsize=(12 if self.publication_figure else 7), loc="upper right")
            ax.grid(True, alpha=0.3, axis="y")

            # Stats inset
            effect = rep.get("masking_effect", {})
            leakage = rep.get("leakage", {})
            gap = rep.get("gap_result", {})
            nearest_da = gap.get("nearest_distance_da_mean", gap.get("nearest_distance_da", 0.0))
            if isinstance(nearest_da, dict):
                nearest_da = nearest_da.get("mean", 0.0)

            stats_text = (
                f"Mask: {rep.get('mask_ratio', 0):.1%}"
                f"  Ann.masked: {effect.get('annotated_fraction_of_masked_peaks', 0):.1%}"
                f"  Preserved: {effect.get('annotated_preservation_ratio', 0):.1%}"
                f"  Leak: {leakage.get('leakage_ratio', 0):.1%}"
                f"  Near: {nearest_da:.1f} Da"
            )
            ax.text(
                0.5, 0.98, stats_text, transform=ax.transAxes,
                fontsize=(11.5 if self.publication_figure else 7), fontfamily="monospace", va="top", ha="center",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                          alpha=0.9, edgecolor="gray", linewidth=0.8),
            )

        axes[-1].set_xlabel("Peak Index", fontsize=11)
        plt.tight_layout()
        path = output_dir / f"spectrum_{idx:04d}_{seq_slug}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        if self.publication_figure:
            for ext in ("svg", "pdf"):
                fig.savefig(output_dir / f"spectrum_{idx:04d}_{seq_slug}.{ext}",
                            bbox_inches="tight", metadata={"Title": "Masking strategy comparison"})
        plt.close(fig)

    def _create_per_strategy_figure(
        self,
        idx: int,
        seq_slug: str,
        strategy_name: str,
        spectrum_result: Dict[str, Any],
        output_dir: Path,
    ) -> None:
        """Per-strategy detail figure: stats, index reference, masked/unmasked, 4-color."""
        viz = spectrum_result["_viz_data"]
        valid_mz = np.asarray(viz["valid_mz"], dtype=np.float64)
        valid_intensity = np.asarray(viz["valid_intensity"], dtype=np.float64)
        annotated_mask = viz.get("annotated_mask", np.zeros(len(valid_mz), dtype=bool))
        if not isinstance(annotated_mask, np.ndarray):
            annotated_mask = np.asarray(annotated_mask, dtype=bool)
        annotations = list(viz.get("theo_annotations", []))
        feature_types = viz.get("feature_types", [])
        custom_ion_data = viz.get("custom_ion_data", {})

        # Merge custom ion detections into annotations
        annotations = self._merge_custom_ion_annotations(
            valid_mz, annotations, custom_ion_data,
        )

        strat_data = spectrum_result.get(strategy_name, {})
        repeats = strat_data.get("repeats", [])
        if not repeats:
            return
        rep = repeats[0]
        mlm_mask = rep.get("mlm_mask_valid")
        if mlm_mask is None:
            return
        mlm_mask = np.asarray(mlm_mask, dtype=bool)

        n_valid = len(valid_mz)
        n_rows = 4  # stats, index reference, masked/unmasked, 4-color
        fig, axes = plt.subplots(n_rows, 1, figsize=(18, 3.0 + 3.5 * 3),
                                 gridspec_kw={"height_ratios": [0.6, 1, 1, 1]})

        metadata = viz.get("metadata", {})
        seq = metadata.get("clean_sequence") or metadata.get("sequence") or "unknown"
        fig.suptitle(f"{strategy_name}: {seq}", fontsize=14, fontweight="bold")

        # --- Row 0: Stats box (with metadata) ---
        ax_stats = axes[0]
        ax_stats.axis("off")

        effect = rep.get("masking_effect", {})
        leakage = rep.get("leakage", {})
        gap = rep.get("gap_result", {})
        nearest_da = gap.get("nearest_distance_da_mean", gap.get("nearest_distance_da", 0.0))
        if isinstance(nearest_da, dict):
            nearest_da = nearest_da.get("mean", 0.0)

        n_masked = int(mlm_mask.sum())
        n_ann = int(annotated_mask.sum())
        n_ann_masked = int((mlm_mask & annotated_mask).sum())
        n_unann_masked = int((mlm_mask & ~annotated_mask).sum())
        n_ann_unmasked = int((~mlm_mask & annotated_mask).sum())
        n_unann_unmasked = int((~mlm_mask & ~annotated_mask).sum())

        # Build metadata line from available fields
        meta_parts = []
        charge = metadata.get("precursor_charge")
        if charge and charge != "unknown":
            meta_parts.append(f"Charge: {charge}")
        frag = metadata.get("frag_type")
        if frag and frag != "unknown":
            meta_parts.append(f"Frag: {frag}")
        instrument = metadata.get("search_instrument")
        if instrument and instrument != "unknown":
            meta_parts.append(f"Instrument: {instrument}")
        mz_range = f"m/z: [{valid_mz.min():.1f}, {valid_mz.max():.1f}]" if n_valid > 0 else ""
        if mz_range:
            meta_parts.append(mz_range)
        meta_line = " | ".join(meta_parts) if meta_parts else "No metadata"

        stats_lines = [
            meta_line,
            f"Peaks: {n_valid} valid | {n_ann} annotated | {n_valid - n_ann} unannotated",
            f"Masked: {n_masked} ({rep.get('mask_ratio', 0):.1%})"
            f" = {n_ann_masked} annotated + {n_unann_masked} unannotated"
            f" | Unmasked: {n_valid - n_masked}"
            f" = {n_ann_unmasked} annotated + {n_unann_unmasked} unannotated",
            f"Signal: {effect.get('annotated_fraction_of_masked_peaks', 0):.1%} of masked are annotated"
            f" | Preserved: {effect.get('annotated_preservation_ratio', 0):.1%}"
            f" | Leakage: {leakage.get('leakage_ratio', 0):.1%}"
            f" (loss={leakage.get('leakage_by_type', {}).get('loss', 0)},"
            f" isotope={leakage.get('leakage_by_type', {}).get('isotope', 0)})"
            f" | Nearest: {nearest_da:.1f} Da",
        ]
        ax_stats.text(
            0.02, 0.5, "\n".join(stats_lines), transform=ax_stats.transAxes,
            fontsize=10, va="center", ha="left", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="lightgray",
                      alpha=0.9, edgecolor="gray", linewidth=1.5),
        )

        # --- Row 1: Index-based annotated reference (ion-type coloring) ---
        ax_ref = axes[1]
        x_pos = np.arange(n_valid)
        colour_groups: Dict[str, List[int]] = {}
        for i in range(n_valid):
            ann = annotations[i] if i < len(annotations) else ""
            cat = self._categorize_ion(ann) if ann else "unannotated"
            colour_groups.setdefault(cat, []).append(i)

        # Plot unannotated first (background), then sorted annotated categories
        if "unannotated" in colour_groups:
            unannotated_indices = colour_groups.pop("unannotated")
            idx_arr = np.array(unannotated_indices)
            color, alpha = self._CATEGORY_COLORS["unannotated"]
            ax_ref.bar(x_pos[idx_arr], valid_intensity[idx_arr],
                       color=color, alpha=alpha,
                       width=1.0, linewidth=0, label="Unannotated")

        for cat in sorted(colour_groups.keys()):
            cat_indices = colour_groups[cat]
            if not cat_indices:
                continue
            idx_arr = np.array(cat_indices)
            color, alpha = self._CATEGORY_COLORS.get(cat, ("#9467bd", 0.8))
            ax_ref.bar(x_pos[idx_arr], valid_intensity[idx_arr],
                       color=color, alpha=alpha,
                       width=1.0, linewidth=0, label=cat)

        # Add annotations for annotated peaks
        for i in range(n_valid):
            ann = annotations[i] if i < len(annotations) else ""
            if not ann:
                continue
            cat = self._categorize_ion(ann)
            if cat == "unannotated":
                continue
            display_ann = self._format_custom_ion_label(ann) if ann.startswith("custom:") else ann
            ax_ref.annotate(
                display_ann, xy=(i, valid_intensity[i]),
                xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=5, rotation=90,
                alpha=0.9, fontweight="bold",
                color=self._TEXT_COLORS.get(cat, "black"),
            )

        ax_ref.set_ylabel("Intensity", fontsize=10)
        ax_ref.set_title("Annotated Reference (index-based)", fontsize=11, fontweight="bold")
        ax_ref.legend(fontsize=7, loc="upper right", ncol=2)
        ax_ref.grid(True, alpha=0.3, axis="y")

        # --- Row 2: Index-based masked vs unmasked (simple red/blue) ---
        ax_simple = axes[2]
        unmasked_idx = ~mlm_mask
        masked_idx = mlm_mask
        if unmasked_idx.any():
            ax_simple.bar(x_pos[unmasked_idx], valid_intensity[unmasked_idx],
                          color="#90CAF9", alpha=0.9, width=1.0, linewidth=0,
                          label="Unmasked")
        if masked_idx.any():
            ax_simple.bar(x_pos[masked_idx], valid_intensity[masked_idx],
                          color="#E53935", alpha=0.9, width=1.0, linewidth=0,
                          label="Masked")

        ax_simple.set_ylabel("Intensity", fontsize=10)
        ax_simple.set_title("Masking Pattern (index-based)", fontsize=11, fontweight="bold")
        ax_simple.legend(fontsize=7, loc="upper right")
        ax_simple.grid(True, alpha=0.3, axis="y")

        # --- Row 3: Index-based 4-color (masked/unmasked × annotated/unannotated) ---
        ax_detail = axes[3]
        ua = ~mlm_mask & annotated_mask
        uu = ~mlm_mask & ~annotated_mask
        ma = mlm_mask & annotated_mask
        mu = mlm_mask & ~annotated_mask

        C = self._VIZ_COLORS
        for mask_arr, key, label in [
            (uu, "unmasked_unannotated", "Unmasked Unannotated"),
            (ua, "unmasked_annotated", "Unmasked Annotated"),
            (mu, "masked_unannotated", "Masked Unannotated"),
            (ma, "masked_annotated", "Masked Annotated"),
        ]:
            if mask_arr.any():
                ax_detail.bar(x_pos[mask_arr], valid_intensity[mask_arr],
                              color=C[key], alpha=0.9, width=1.0, linewidth=0,
                              label=label)

        ax_detail.set_xlabel("Peak Index", fontsize=10)
        ax_detail.set_ylabel("Intensity", fontsize=10)
        ax_detail.set_title("Masking × Annotation Detail (index-based)", fontsize=11, fontweight="bold")
        ax_detail.legend(fontsize=7, loc="upper right", ncol=2)
        ax_detail.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path = output_dir / f"spectrum_{idx:04d}_{seq_slug}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Shared plotting helpers
    # ------------------------------------------------------------------

    def _plot_4color_spectrum(
        self,
        ax: plt.Axes,
        valid_mz: np.ndarray,
        valid_intensity: np.ndarray,
        mlm_mask: np.ndarray,
        annotated_mask: np.ndarray,
    ) -> None:
        """Plot a spectrum with the 4-color masking × annotation scheme."""
        C = self._VIZ_COLORS

        # Compute 4 categories
        ua = ~mlm_mask & annotated_mask     # Unmasked Annotated (blue)
        uu = ~mlm_mask & ~annotated_mask    # Unmasked Unannotated (gray)
        ma = mlm_mask & annotated_mask      # Masked Annotated (red)
        mu = mlm_mask & ~annotated_mask     # Masked Unannotated (orange)

        for mask, key, label in [
            (uu, "unmasked_unannotated", "Unmasked Unannotated"),
            (ua, "unmasked_annotated", "Unmasked Annotated"),
            (mu, "masked_unannotated", "Masked Unannotated"),
            (ma, "masked_annotated", "Masked Annotated"),
        ]:
            if mask.any():
                markerline, stemlines, baseline = ax.stem(
                    valid_mz[mask], valid_intensity[mask],
                    linefmt="-", markerfmt="o", basefmt=" ", label=label,
                )
                plt.setp(markerline, color=C[key], markersize=3, alpha=0.9)
                plt.setp(stemlines, color=C[key], alpha=0.8)

        ax.legend(fontsize=8, loc="upper right")

    @classmethod
    def _plot_annotated_reference(
        cls,
        ax: plt.Axes,
        valid_mz: np.ndarray,
        valid_intensity: np.ndarray,
        annotated_mask: np.ndarray,
        annotations: List[str],
    ) -> None:
        """Plot the reference spectrum with ion-type coloring and annotations."""
        # Categorize all peaks
        peak_categories: Dict[int, str] = {}
        for i in range(len(valid_mz)):
            if annotated_mask[i]:
                ann = annotations[i] if i < len(annotations) else ""
                peak_categories[i] = cls._categorize_ion(ann)
            else:
                peak_categories[i] = "unannotated"

        # Plot unannotated first
        unannotated_indices = [i for i, cat in peak_categories.items() if cat == "unannotated"]
        if unannotated_indices:
            color, alpha = cls._CATEGORY_COLORS["unannotated"]
            ml, sl, bl = ax.stem(
                valid_mz[unannotated_indices], valid_intensity[unannotated_indices],
                linefmt="-", markerfmt="o", basefmt=" ", label="Unannotated")
            plt.setp(ml, color=color, markersize=2, alpha=alpha)
            plt.setp(sl, color=color, alpha=alpha)

        # Plot annotated peaks by category
        for category in sorted(set(peak_categories.values()) - {"unannotated"}):
            cat_indices = [i for i, cat in peak_categories.items() if cat == category]
            if cat_indices:
                color, alpha = cls._CATEGORY_COLORS.get(category, ("#9467bd", 0.8))
                ml, sl, bl = ax.stem(
                    valid_mz[cat_indices], valid_intensity[cat_indices],
                    linefmt="-", markerfmt="o", basefmt=" ", label=category)
                plt.setp(ml, color=color, markersize=3, alpha=alpha)
                plt.setp(sl, color=color, alpha=alpha)

        # Add text annotations for annotated peaks
        for i in range(len(valid_mz)):
            if not annotated_mask[i]:
                continue
            ann = annotations[i] if i < len(annotations) else ""
            if not ann:
                continue
            cat = cls._categorize_ion(ann)
            if cat == "unannotated":
                continue
            display_ann = cls._format_custom_ion_label(ann) if ann.startswith("custom:") else ann
            ax.annotate(
                display_ann, xy=(valid_mz[i], valid_intensity[i]), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=6,
                rotation=90, alpha=0.9, fontweight="bold",
                color=cls._TEXT_COLORS.get(cat, "black"),
            )

        ax.legend(fontsize=8, loc="upper right")

    def _generate_masking_summary(self) -> None:
        """2x2 executive overview: budget composition, intensity-weighted, efficiency, mask rates."""
        self._render_masking_summary(
            per_strategy=self.results.get("per_strategy", {}),
            title="Masking Analysis Summary",
            output_path=self.output_dir / "masking_summary.png",
        )

    def _generate_masking_summary_annotation_driven(self) -> None:
        """Gold-standard companion to ``_generate_masking_summary``.

        Restricted to spectra where ``signal_aware_fragment`` used the
        annotation-driven path (``fallback_used == False``). Isolates
        signal_aware at its intended operating point while showing the
        same comparison strategies evaluated on the same spectrum subset,
        so the chart answers "when signal_aware is on its home turf, how
        does it compare to the others?" without fallback noise.
        """
        per_strategy = self.results.get("per_strategy_annotation_driven", {})
        if not per_strategy:
            return
        meta = self.results.get("annotation_driven_meta", {})
        n_subset = meta.get("n_spectra", 0)
        n_total = meta.get("n_spectra", 0) + meta.get("n_fallback_excluded", 0)
        ref = meta.get("reference_strategy", "signal_aware")
        # Caption the reader with the filter + the selection-bias warning
        # inline so the figure cannot be read out of context.
        title = (
            "Masking Analysis Summary — annotation-driven subset\n"
            f"({ref} fallback=False, n={n_subset} of {n_total}; "
            "biased toward spectra with rich annotation)"
        )
        self._render_masking_summary(
            per_strategy=per_strategy,
            title=title,
            output_path=self.output_dir / "masking_summary_annotation_driven.png",
        )

    def _render_masking_summary(
        self,
        per_strategy: Dict[str, Dict[str, Any]],
        title: str,
        output_path: Path,
    ) -> None:
        """Render the 2×2 masking summary from an arbitrary per-strategy dict.

        Factored out of :meth:`_generate_masking_summary` so the same
        panels drive both the full-population and annotation-driven
        views. Panels: masked peak composition (count), masked peak
        composition (intensity-weighted), training-signal efficiency,
        annotated vs unannotated mask rates.
        """
        names = list(per_strategy.keys())
        if not names:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight="bold")
        x = np.arange(len(names))

        # ── [0,0] Masked Peak Composition (Count-Based) ──────────────
        ax = axes[0, 0]
        ann_frac = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_annotated_fraction_of_masked_peaks", 0.0
            )
            for n in names
        ]
        prec_budget = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_precursor_mask_budget_fraction", 0.0
            )
            for n in names
        ]
        custom_budget = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_custom_mask_budget_fraction", 0.0
            )
            for n in names
        ]
        unann_frac = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_unannotated_fraction_of_masked_peaks", 0.0
            )
            for n in names
        ]
        # Fragment = annotated - precursor (custom is a subset of unannotated, not annotated)
        frag_frac = [max(a - p, 0.0) for a, p in zip(ann_frac, prec_budget)]
        # Unannotated adjusted: total unannotated minus custom (custom came from unannotated)
        unann_adj = [max(u - c, 0.0) for u, c in zip(unann_frac, custom_budget)]

        if any(v > 0 for v in ann_frac) or any(v > 0 for v in unann_frac):
            bottom = [0.0] * len(names)
            segments = [
                (frag_frac, "Fragment ions", "#3498db"),
                (prec_budget, "Precursor ions", "#e74c3c"),
                (custom_budget, "Custom ions", "#2ecc71"),
                (unann_adj, "Unannotated", "#95a5a6"),
            ]
            for vals, label, color in segments:
                ax.bar(x, vals, bottom=bottom, label=label,
                       color=color, alpha=0.85, edgecolor="black", linewidth=0.5)
                # Labels at segment midpoints (skip if <2%)
                for i_bar in range(len(names)):
                    if vals[i_bar] >= 0.02:
                        ax.text(
                            i_bar, bottom[i_bar] + vals[i_bar] / 2,
                            f"{vals[i_bar]:.1%}",
                            ha="center", va="center", fontsize=8, fontweight="bold",
                        )
                bottom = [b + v for b, v in zip(bottom, vals)]
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=20, ha="right")
            ax.set_ylabel("Fraction of Masked Peaks")
            ax.set_ylim(0, 1.05)
            ax.set_title("Masked Peak Composition (Count-Based)")
            ax.legend(fontsize=9, loc="upper right")
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.set_title("Masked Peak Composition — Count (no data)")
            ax.axis("off")

        # ── [0,1] Masked Peak Composition (Intensity-Weighted) ───────
        ax = axes[0, 1]
        ann_int_frac = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_annotated_intensity_fraction_of_masked", 0.0
            )
            for n in names
        ]
        prec_int_budget = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_precursor_intensity_budget_fraction", 0.0
            )
            for n in names
        ]
        custom_int_budget = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_custom_intensity_budget_fraction", 0.0
            )
            for n in names
        ]
        unann_int_frac = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_unannotated_intensity_fraction_of_masked", 0.0
            )
            for n in names
        ]
        frag_int_frac = [max(a - p, 0.0) for a, p in zip(ann_int_frac, prec_int_budget)]
        unann_int_adj = [max(u - c, 0.0) for u, c in zip(unann_int_frac, custom_int_budget)]

        if any(v > 0 for v in ann_int_frac) or any(v > 0 for v in unann_int_frac):
            bottom = [0.0] * len(names)
            segments = [
                (frag_int_frac, "Fragment ions", "#3498db"),
                (prec_int_budget, "Precursor ions", "#e74c3c"),
                (custom_int_budget, "Custom ions", "#2ecc71"),
                (unann_int_adj, "Unannotated", "#95a5a6"),
            ]
            for vals, label, color in segments:
                ax.bar(x, vals, bottom=bottom, label=label,
                       color=color, alpha=0.85, edgecolor="black", linewidth=0.5)
                for i_bar in range(len(names)):
                    if vals[i_bar] >= 0.02:
                        ax.text(
                            i_bar, bottom[i_bar] + vals[i_bar] / 2,
                            f"{vals[i_bar]:.1%}",
                            ha="center", va="center", fontsize=8, fontweight="bold",
                        )
                bottom = [b + v for b, v in zip(bottom, vals)]
            # Amplification annotations for precursor and custom
            for i_bar in range(len(names)):
                if prec_budget[i_bar] > 0.001 and prec_int_budget[i_bar] > 0.001:
                    amp = prec_int_budget[i_bar] / prec_budget[i_bar]
                    ax.annotate(
                        f"prec {amp:.1f}x",
                        xy=(i_bar, frag_int_frac[i_bar] + prec_int_budget[i_bar]),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=7, fontstyle="italic", color="#c0392b",
                    )
                if custom_budget[i_bar] > 0.001 and custom_int_budget[i_bar] > 0.001:
                    amp = custom_int_budget[i_bar] / custom_budget[i_bar]
                    ax.annotate(
                        f"cust {amp:.1f}x",
                        xy=(i_bar, frag_int_frac[i_bar] + prec_int_budget[i_bar] + custom_int_budget[i_bar]),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=7, fontstyle="italic", color="#27ae60",
                    )
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=20, ha="right")
            ax.set_ylabel("Fraction of Masked Intensity")
            ax.set_ylim(0, 1.05)
            ax.set_title("Masked Peak Composition (Intensity-Weighted)")
            ax.legend(fontsize=9, loc="upper right")
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.set_title("Masked Peak Composition — Intensity (no data)")
            ax.axis("off")

        # ── [1,0] Training Signal Efficiency ─────────────────────────
        ax = axes[1, 0]
        colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
        eff_vals = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_annotated_fraction_of_masked_peaks", 0.0
            )
            for n in names
        ]
        if any(v > 0 for v in eff_vals):
            bars = ax.bar(x, eff_vals,
                          color=colors[:len(names)], alpha=0.85,
                          edgecolor="black", linewidth=0.5)
            for bar, val in zip(bars, eff_vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=20, ha="right")
            ax.set_ylabel("Fraction")
            ax.set_title("Training Signal Efficiency\n(% of masked peaks that are annotated)")
            max_bar = max(eff_vals)
            ax.set_ylim(0, min(max_bar * 1.4, 1.05))
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.set_title("Training Signal Efficiency (no data)")
            ax.axis("off")

        # ── [1,1] Annotated vs Unannotated Mask Rates ───────────────
        ax = axes[1, 1]
        ann_mr = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_annotated_mask_ratio", 0.0
            )
            for n in names
        ]
        unann_mr = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_unannotated_mask_ratio", 0.0
            )
            for n in names
        ]
        if any(v > 0 for v in ann_mr) or any(v > 0 for v in unann_mr):
            width = 0.35
            ax.bar(x - width / 2, ann_mr, width, label="Annotated",
                   color="#3498db", alpha=0.85, edgecolor="black", linewidth=0.5)
            ax.bar(x + width / 2, unann_mr, width, label="Unannotated",
                   color="#e74c3c", alpha=0.85, edgecolor="black", linewidth=0.5)
            for i_bar, (a, u) in enumerate(zip(ann_mr, unann_mr)):
                ax.text(i_bar - width / 2, a + 0.005, f"{a:.1%}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
                ax.text(i_bar + width / 2, u + 0.005, f"{u:.1%}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=20, ha="right")
            ax.set_ylabel("Mask Rate")
            ax.set_title("Annotated vs Unannotated Mask Rates")
            ax.legend(fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.set_title("Annotated vs Unannotated Mask Rates (no data)")
            ax.axis("off")

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Masking summary figure saved to: {output_path}")

    def _generate_fragment_ion_analysis(self) -> None:
        """2x2 fragment ion detail: sub-type composition, intensity, series groups, ion rates."""
        per_strategy = self.results.get("per_strategy", {})
        names = list(per_strategy.keys())
        if not names:
            return

        # Check if any strategy has type data
        has_data = any(
            per_strategy[n].get("masking_effect", {}).get("annotated_type_masked")
            for n in names
        )
        if not has_data:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Fragment Ion Masking Analysis", fontsize=16, fontweight="bold")
        x = np.arange(len(names))

        def _classify_type_counts(type_dict: Dict[str, int]) -> Dict[str, int]:
            """Classify type counts into base/loss/isotope categories."""
            base = 0
            loss = 0
            isotope = 0
            for label, count in type_dict.items():
                if label in self._EXCLUDED_LABELS:
                    continue
                if label.endswith("-isotope"):
                    isotope += count
                elif label.endswith("-loss"):
                    loss += count
                else:
                    base += count
            return {"base": base, "loss": loss, "isotope": isotope}

        def _classify_type_floats(type_dict: Dict[str, float]) -> Dict[str, float]:
            """Classify type values into base/loss/isotope categories."""
            base = 0.0
            loss = 0.0
            isotope = 0.0
            for label, val in type_dict.items():
                if label in self._EXCLUDED_LABELS:
                    continue
                if label.endswith("-isotope"):
                    isotope += val
                elif label.endswith("-loss"):
                    loss += val
                else:
                    base += val
            return {"base": base, "loss": loss, "isotope": isotope}

        # ── [0,0] Fragment Sub-Type Composition (Count-Based) ────────
        ax = axes[0, 0]
        base_fracs = []
        loss_fracs = []
        iso_fracs = []
        for n in names:
            eff = per_strategy[n].get("masking_effect", {})
            classified = _classify_type_counts(eff.get("annotated_type_masked", {}))
            total = classified["base"] + classified["loss"] + classified["isotope"]
            if total > 0:
                base_fracs.append(classified["base"] / total)
                loss_fracs.append(classified["loss"] / total)
                iso_fracs.append(classified["isotope"] / total)
            else:
                base_fracs.append(0.0)
                loss_fracs.append(0.0)
                iso_fracs.append(0.0)

        if any(b > 0 or l > 0 or i > 0 for b, l, i in zip(base_fracs, loss_fracs, iso_fracs)):
            ax.bar(x, base_fracs, label="Base ions", color="#2ca02c", alpha=0.85,
                   edgecolor="black", linewidth=0.5)
            ax.bar(x, loss_fracs, bottom=base_fracs, label="Loss ions", color="#e67e22",
                   alpha=0.85, edgecolor="black", linewidth=0.5)
            bottoms_iso = [b + l for b, l in zip(base_fracs, loss_fracs)]
            ax.bar(x, iso_fracs, bottom=bottoms_iso, label="Isotope ions", color="#aec7e8",
                   alpha=0.85, edgecolor="black", linewidth=0.5)
            # Labels
            for i_bar in range(len(names)):
                if base_fracs[i_bar] >= 0.02:
                    ax.text(i_bar, base_fracs[i_bar] / 2, f"{base_fracs[i_bar]:.1%}",
                            ha="center", va="center", fontsize=8, fontweight="bold")
                if loss_fracs[i_bar] >= 0.02:
                    ax.text(i_bar, base_fracs[i_bar] + loss_fracs[i_bar] / 2,
                            f"{loss_fracs[i_bar]:.1%}",
                            ha="center", va="center", fontsize=8, fontweight="bold")
                if iso_fracs[i_bar] >= 0.02:
                    ax.text(i_bar, bottoms_iso[i_bar] + iso_fracs[i_bar] / 2,
                            f"{iso_fracs[i_bar]:.1%}",
                            ha="center", va="center", fontsize=8, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=20, ha="right")
            ax.set_ylabel("Fraction of Fragment Masked")
            ax.set_ylim(0, 1.05)
            ax.set_title("Fragment Sub-Type Composition (Count-Based)")
            ax.legend(fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.set_title("Fragment Sub-Type Composition — Count (no data)")
            ax.axis("off")

        # ── [0,1] Fragment Sub-Type Composition (Intensity-Weighted) ──
        ax = axes[0, 1]
        base_int_fracs = []
        loss_int_fracs = []
        iso_int_fracs = []
        for n in names:
            eff = per_strategy[n].get("masking_effect", {})
            classified = _classify_type_floats(eff.get("annotated_type_intensity_masked", {}))
            total = classified["base"] + classified["loss"] + classified["isotope"]
            if total > 0:
                base_int_fracs.append(classified["base"] / total)
                loss_int_fracs.append(classified["loss"] / total)
                iso_int_fracs.append(classified["isotope"] / total)
            else:
                base_int_fracs.append(0.0)
                loss_int_fracs.append(0.0)
                iso_int_fracs.append(0.0)

        if any(b > 0 or l > 0 or i > 0 for b, l, i in zip(base_int_fracs, loss_int_fracs, iso_int_fracs)):
            ax.bar(x, base_int_fracs, label="Base ions", color="#2ca02c", alpha=0.85,
                   edgecolor="black", linewidth=0.5)
            ax.bar(x, loss_int_fracs, bottom=base_int_fracs, label="Loss ions", color="#e67e22",
                   alpha=0.85, edgecolor="black", linewidth=0.5)
            bottoms_iso_int = [b + l for b, l in zip(base_int_fracs, loss_int_fracs)]
            ax.bar(x, iso_int_fracs, bottom=bottoms_iso_int, label="Isotope ions", color="#aec7e8",
                   alpha=0.85, edgecolor="black", linewidth=0.5)
            for i_bar in range(len(names)):
                if base_int_fracs[i_bar] >= 0.02:
                    ax.text(i_bar, base_int_fracs[i_bar] / 2, f"{base_int_fracs[i_bar]:.1%}",
                            ha="center", va="center", fontsize=8, fontweight="bold")
                if loss_int_fracs[i_bar] >= 0.02:
                    ax.text(i_bar, base_int_fracs[i_bar] + loss_int_fracs[i_bar] / 2,
                            f"{loss_int_fracs[i_bar]:.1%}",
                            ha="center", va="center", fontsize=8, fontweight="bold")
                if iso_int_fracs[i_bar] >= 0.02:
                    ax.text(i_bar, bottoms_iso_int[i_bar] + iso_int_fracs[i_bar] / 2,
                            f"{iso_int_fracs[i_bar]:.1%}",
                            ha="center", va="center", fontsize=8, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=20, ha="right")
            ax.set_ylabel("Fraction of Fragment Masked Intensity")
            ax.set_ylim(0, 1.05)
            ax.set_title("Fragment Sub-Type Composition (Intensity-Weighted)")
            ax.legend(fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.set_title("Fragment Sub-Type Composition — Intensity (no data)")
            ax.axis("off")

        # ── [1,0] Per-Series Group Full-Mask Ratio ──────────────────
        ax = axes[1, 0]
        series_colors = {"b": "#1f77b4", "y": "#d62728", "a": "#2ca02c"}
        all_series: set = set()
        for n in names:
            eff = per_strategy[n].get("masking_effect", {})
            series_ratio = eff.get("series_group_full_mask_ratio", {})
            all_series.update(s for s, v in series_ratio.items() if v > 0)
        sorted_series = sorted(s for s in all_series if s in series_colors)

        if sorted_series:
            n_series = len(sorted_series)
            n_strat = len(names)
            width = 0.8 / max(n_strat, 1)
            for i, n in enumerate(names):
                eff = per_strategy[n].get("masking_effect", {})
                series_ratio = eff.get("series_group_full_mask_ratio", {})
                vals = [series_ratio.get(s, 0.0) for s in sorted_series]
                xpos = np.arange(n_series) + i * width - (n_strat - 1) * width / 2
                ax.bar(xpos, vals, width, label=n, alpha=0.85,
                       edgecolor="black", linewidth=0.5)
            ax.set_xticks(np.arange(n_series))
            ax.set_xticklabels([f"{s}-series" for s in sorted_series])
            ax.set_ylabel("Full Mask Ratio")
            ax.set_title("Per-Series Group Full-Mask Ratio\n(fraction of groups fully masked)")
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.set_title("Per-Series Group Full-Mask Ratio (no data)")
            ax.axis("off")

        # ── [1,1] b/y Series Ion Mask Rate ──────────────────────────
        ax = axes[1, 1]
        target_types = ["b-ion", "y-ion", "b-loss", "y-loss"]
        available_types = []
        for t in target_types:
            for n in names:
                eff = per_strategy[n].get("masking_effect", {})
                if t in eff.get("annotated_type_mask_ratio", {}):
                    available_types.append(t)
                    break

        if available_types:
            n_types = len(available_types)
            n_strat = len(names)
            width = 0.8 / max(n_strat, 1)
            for i, n in enumerate(names):
                eff = per_strategy[n].get("masking_effect", {})
                ratios = eff.get("annotated_type_mask_ratio", {})
                vals = [ratios.get(t, 0.0) for t in available_types]
                xpos = np.arange(n_types) + i * width - (n_strat - 1) * width / 2
                ax.bar(xpos, vals, width, label=n, alpha=0.85,
                       edgecolor="black", linewidth=0.5)
            ax.set_xticks(np.arange(n_types))
            ax.set_xticklabels(available_types, rotation=20, ha="right")
            ax.set_ylabel("Mask Rate")
            ax.set_title("b/y Series Ion Mask Rate\n(per-ion-type mask rate)")
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.set_title("b/y Series Ion Mask Rate (no data)")
            ax.axis("off")

        plt.tight_layout()
        path = self.output_dir / "masking_fragment_ion_analysis.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Fragment ion analysis figure saved to: {path}")

    def _generate_training_signal_analysis(self) -> None:
        """2x2 training signal analysis: efficiency, ion-type heatmap, mask rates, intensity."""
        per_strategy = self.results.get("per_strategy", {})
        names = list(per_strategy.keys())
        if not names:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Training Signal Analysis", fontsize=16, fontweight="bold")
        x = np.arange(len(names))
        colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

        # [0,0] Training Signal Efficiency — annotated fraction of masked peaks
        ax = axes[0, 0]
        ann_frac = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_annotated_fraction_of_masked_peaks", 0.0
            )
            for n in names
        ]

        if any(v > 0 for v in ann_frac):
            bars = ax.bar(x, ann_frac,
                          color=colors[:len(names)], alpha=0.85,
                          edgecolor="black", linewidth=0.5)
            for bar, val in zip(bars, ann_frac):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=20, ha="right")
            ax.set_ylabel("Fraction")
            ax.set_title("Training Signal Efficiency\n(% of masked peaks that are annotated)")
            max_bar = max(ann_frac)
            ax.set_ylim(0, min(max_bar * 1.4, 1.05))
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.set_title("Training Signal Efficiency (no data)")
            ax.axis("off")

        # [0,1] Per-Ion-Type Mask Rates (heatmap, data-driven scale)
        ax = axes[0, 1]
        all_types: set = set()
        for n in names:
            eff = per_strategy[n].get("masking_effect", {})
            all_types.update(
                k for k in eff.get("annotated_type_mask_ratio", {}).keys()
                if k not in self._EXCLUDED_LABELS
            )
        sorted_types = sorted(all_types)[:12]
        if sorted_types:
            heatmap_data = np.zeros((len(names), len(sorted_types)))
            for i, n in enumerate(names):
                ratios = per_strategy[n].get("masking_effect", {}).get(
                    "annotated_type_mask_ratio", {}
                )
                for j, t in enumerate(sorted_types):
                    heatmap_data[i, j] = ratios.get(t, 0.0)
            data_max = heatmap_data.max() if heatmap_data.max() > 0 else 1.0
            data_min = heatmap_data.min()
            vmax = min(np.ceil(data_max * 20) / 20, 1.0)
            vmin = max(np.floor(data_min * 20) / 20, 0.0)
            im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
            ax.set_xticks(np.arange(len(sorted_types)))
            ax.set_xticklabels(sorted_types, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(np.arange(len(names)))
            ax.set_yticklabels(names)
            ax.set_title("Per-Ion-Type Mask Rates")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.set_title("Per-Ion-Type Mask Rates (no data)")
            ax.axis("off")

        # [1,0] Annotated vs Unannotated Mask Rates
        ax = axes[1, 0]
        ann_mr = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_annotated_mask_ratio", 0.0
            )
            for n in names
        ]
        unann_mr = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_unannotated_mask_ratio", 0.0
            )
            for n in names
        ]
        if any(v > 0 for v in ann_mr) or any(v > 0 for v in unann_mr):
            width = 0.35
            ax.bar(x - width / 2, ann_mr, width, label="Annotated",
                   color="#3498db", alpha=0.85, edgecolor="black", linewidth=0.5)
            ax.bar(x + width / 2, unann_mr, width, label="Unannotated",
                   color="#e74c3c", alpha=0.85, edgecolor="black", linewidth=0.5)
            for i_bar, (a, u) in enumerate(zip(ann_mr, unann_mr)):
                ax.text(i_bar - width / 2, a + 0.005, f"{a:.1%}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
                ax.text(i_bar + width / 2, u + 0.005, f"{u:.1%}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=20, ha="right")
            ax.set_ylabel("Mask Rate")
            ax.set_title("Annotated vs Unannotated Mask Rates\n(fraction of each type that is masked)")
            ax.legend(fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.set_title("Annotated vs Unannotated Mask Rates (no data)")
            ax.axis("off")

        # [1,1] Intensity-Weighted Signal
        ax = axes[1, 1]
        ann_int = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_annotated_intensity_fraction_of_masked", 0.0
            )
            for n in names
        ]
        unann_int = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_unannotated_intensity_fraction_of_masked", 0.0
            )
            for n in names
        ]
        if any(v > 0 for v in ann_int) or any(v > 0 for v in unann_int):
            width = 0.35
            ax.bar(x - width / 2, ann_int, width, label="Annotated Intensity",
                   color="#3498db", alpha=0.85, edgecolor="black", linewidth=0.5)
            ax.bar(x + width / 2, unann_int, width, label="Unannotated Intensity",
                   color="#e74c3c", alpha=0.85, edgecolor="black", linewidth=0.5)
            for i_bar, (a, u) in enumerate(zip(ann_int, unann_int)):
                ax.text(i_bar - width / 2, a + 0.005, f"{a:.1%}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
                ax.text(i_bar + width / 2, u + 0.005, f"{u:.1%}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=20, ha="right")
            ax.set_ylabel("Fraction of Masked Intensity")
            ax.set_title("Intensity-Weighted Signal\n(intensity share of masked peaks)")
            ax.legend(fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.set_title("Intensity-Weighted Signal (no data)")
            ax.axis("off")

        plt.tight_layout()
        path = self.output_dir / "masking_training_signal_analysis.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Training signal analysis figure saved to: {path}")

    def _generate_prediction_difficulty(self) -> None:
        """2x2 prediction difficulty analysis across strategies."""
        per_strategy = self.results.get("per_strategy", {})
        names = list(per_strategy.keys())
        if not names:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Prediction Difficulty Analysis", fontsize=16, fontweight="bold")

        # [0,0] Overlaid histograms of total gap (log-scaled x-axis)
        ax = axes[0, 0]
        colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
        has_hist_data = False
        for i, n in enumerate(names):
            gap = per_strategy[n].get("gap_analysis", {})
            raw = gap.get("raw_data", None)
            if raw is not None and isinstance(raw, pd.DataFrame) and "total_gap_da" in raw.columns:
                vals = raw["total_gap_da"].dropna()
                vals = vals[vals > 0]  # log scale needs positive values
                if len(vals) > 0:
                    has_hist_data = True
                    # Log-spaced bins for better dynamic range
                    log_bins = np.logspace(
                        np.log10(max(vals.min(), 0.01)), np.log10(vals.max()), 50
                    )
                    ax.hist(vals, bins=log_bins, alpha=0.4, color=colors[i],
                            label=n, edgecolor="none")
        if has_hist_data:
            ax.set_xscale("log")
        ax.set_xlabel("Total Gap Between Nearest Unmasked Neighbors (Da)")
        ax.set_ylabel("Count")
        ax.set_title("Unmasked Gap Distribution")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # [0,1] Mean candidate groups in gap per strategy × group_size
        # Shows how many groups the model must choose from when predicting
        # the group for a masked peak (ceil(total_gap / group_width)).
        ax = axes[0, 1]
        candidate_group_sizes = [25, 50, 100, 200, 500]
        bin_size = self.gap_analysers[names[0]].bin_size if names else 0.02
        has_sweep_data = False
        n_gs = len(candidate_group_sizes)
        n_strat = len(names)
        bar_width = 0.8 / max(n_strat, 1)
        for i, n in enumerate(names):
            gap = per_strategy[n].get("gap_analysis", {})
            raw = gap.get("raw_data", None)
            if raw is not None and isinstance(raw, pd.DataFrame) and "total_gap_da" in raw.columns:
                total_gap = raw["total_gap_da"].dropna().values
                if len(total_gap) == 0:
                    continue
                has_sweep_data = True
                means = []
                for gs in candidate_group_sizes:
                    group_width = bin_size * gs
                    n_groups = np.ceil(total_gap / group_width)
                    means.append(float(n_groups.mean()))
                xpos = np.arange(n_gs) + i * bar_width - (n_strat - 1) * bar_width / 2
                ax.bar(xpos, means, bar_width, label=n, color=colors[i], alpha=0.8)
        if has_sweep_data:
            ax.set_xticks(np.arange(n_gs))
            ax.set_xticklabels([f"gs={gs}\n({bin_size * gs:.1f} Da)" for gs in candidate_group_sizes],
                               fontsize=8)
            ax.set_ylabel("Mean Candidate Groups in Gap")
            ax.set_title("Group-Level Prediction Difficulty\n(groups between unmasked neighbors)")
            ax.legend(fontsize=7)
            ax.grid(True, axis="y", alpha=0.3)

        # [1,0] Per-region grouped bars (mean total gap)
        ax = axes[1, 0]
        region_order = ["immonium_internal", "core_fragment", "extended_fragment", "high_mass_fragment"]
        region_short = {"immonium_internal": "<200", "core_fragment": "200-800",
                        "extended_fragment": "800-1500", "high_mass_fragment": "1500+"}
        n_strategies = len(names)
        width = 0.8 / max(n_strategies, 1)
        for i, n in enumerate(names):
            gap = per_strategy[n].get("gap_analysis", {})
            strat = gap.get("stratified_by_mz_range", {})
            means = []
            for r in region_order:
                rdata = strat.get(r, {}).get("total_gap_da", {})
                means.append(rdata.get("mean", 0.0))
            xpos = np.arange(len(region_order)) + i * width - (n_strategies - 1) * width / 2
            ax.bar(xpos, means, width, label=n, color=colors[i], alpha=0.8)
        ax.set_xticks(np.arange(len(region_order)))
        ax.set_xticklabels([region_short.get(r, r) for r in region_order])
        ax.set_xlabel("m/z Region")
        ax.set_ylabel("Mean Total Gap (Da)")
        ax.set_title("Unmasked Gap by m/z Region")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

        # [1,1] CDF of candidate groups for the configured group_size.
        # Shows the cumulative distribution of how many groups the model
        # must choose from — steep curve = mostly easy, gradual = spread.
        ax = axes[1, 1]
        cfg_group_size = self.gap_analysers[names[0]].bin_group_size if names else 50
        cfg_group_width = self.gap_analysers[names[0]].group_width_da if names else 1.0
        has_cdf_data = False
        all_n_groups: list[np.ndarray] = []
        for i, n in enumerate(names):
            gap = per_strategy[n].get("gap_analysis", {})
            raw = gap.get("raw_data", None)
            if raw is not None and isinstance(raw, pd.DataFrame) and "total_gap_da" in raw.columns:
                total_gap = raw["total_gap_da"].dropna().values
                if len(total_gap) == 0:
                    continue
                has_cdf_data = True
                n_groups = np.ceil(total_gap / cfg_group_width)
                all_n_groups.append(n_groups)
                sorted_vals = np.sort(n_groups)
                cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) * 100
                ax.plot(sorted_vals, cdf, label=n, color=colors[i], linewidth=1.5)
        if has_cdf_data:
            ax.set_xlabel("Candidate Groups in Gap")
            ax.set_ylabel("Cumulative % of Masked Peaks")
            ax.set_title(
                f"Difficulty CDF (group_size={cfg_group_size}, {cfg_group_width:.1f} Da)"
            )
            # Limit x-axis to 99th percentile across all strategies
            ax.set_xlim(left=0)
            if all_n_groups:
                p99 = float(np.percentile(np.concatenate(all_n_groups), 99))
                ax.set_xlim(right=max(p99, 10))
            ax.set_ylim(0, 102)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.output_dir / "masking_prediction_difficulty.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Prediction difficulty figure saved to: {path}")

    def _generate_fragment_group_status(self) -> None:
        """2x2 fragment group masking status figure.

        Panels:
        - [0,0] Group masking status (stacked bar: fully/partially/unmasked)
        - [0,1] Group masking status by ion series
        - [1,0] Per-spectrum distribution of fully masked groups (box plot)
        - [1,1] Partial masking depth distribution (histogram)
        """
        per_strategy = self.results.get("per_strategy", {})
        names = list(per_strategy.keys())
        if not names:
            return

        # Check if any strategy has group data
        has_data = any(
            per_strategy[n].get("masking_effect", {}).get(
                "total_n_fragment_groups_total", 0
            ) > 0
            for n in names
        )
        if not has_data:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        # Surface the primary-view restriction in the suptitle so readers
        # know the fragment-group denominator is the frag_type's N/C pair
        # (b/y for collisional, c/z for ETD) rather than every annotated
        # ion the matcher saw. The all-types numbers remain accessible via
        # the ``avg_*_all`` keys in the JSON / comparison CSV.
        fig.suptitle(
            "Fragment Group Masking Status\n"
            "(primary view: b/y for HCD·HCID·CID, c/z for ETD·ECD)",
            fontsize=15, fontweight="bold",
        )
        x = np.arange(len(names))

        # ── [0,0] Group Masking Status (stacked bar, fraction) ───────
        ax = axes[0, 0]
        full_fracs = []
        partial_fracs = []
        unmask_fracs = []
        for n in names:
            eff = per_strategy[n].get("masking_effect", {})
            total = eff.get("total_n_fragment_groups_total", 0)
            if total > 0:
                full_fracs.append(
                    eff.get("total_n_fragment_groups_fully_masked", 0) / total
                )
                partial_fracs.append(
                    eff.get("total_n_fragment_groups_partially_masked", 0) / total
                )
                unmask_fracs.append(
                    eff.get("total_n_fragment_groups_unmasked", 0) / total
                )
            else:
                full_fracs.append(0.0)
                partial_fracs.append(0.0)
                unmask_fracs.append(0.0)

        ax.bar(
            x, full_fracs, label="Fully masked",
            color="#9b59b6", alpha=0.85, edgecolor="black", linewidth=0.5,
        )
        ax.bar(
            x, partial_fracs, bottom=full_fracs, label="Partially masked",
            color="#f39c12", alpha=0.85, edgecolor="black", linewidth=0.5,
        )
        bottoms_unm = [f + p for f, p in zip(full_fracs, partial_fracs)]
        ax.bar(
            x, unmask_fracs, bottom=bottoms_unm, label="Unmasked",
            color="#bdc3c7", alpha=0.85, edgecolor="black", linewidth=0.5,
        )
        # Labels at segment midpoints (skip if <2%)
        for i_bar in range(len(names)):
            if full_fracs[i_bar] >= 0.02:
                ax.text(
                    i_bar, full_fracs[i_bar] / 2,
                    f"{full_fracs[i_bar]:.1%}",
                    ha="center", va="center", fontsize=8, fontweight="bold",
                )
            if partial_fracs[i_bar] >= 0.02:
                ax.text(
                    i_bar, full_fracs[i_bar] + partial_fracs[i_bar] / 2,
                    f"{partial_fracs[i_bar]:.1%}",
                    ha="center", va="center", fontsize=8, fontweight="bold",
                )
            if unmask_fracs[i_bar] >= 0.02:
                ax.text(
                    i_bar, bottoms_unm[i_bar] + unmask_fracs[i_bar] / 2,
                    f"{unmask_fracs[i_bar]:.1%}",
                    ha="center", va="center", fontsize=8, fontweight="bold",
                )
        # Annotate average group count
        avg_total = np.mean([
            per_strategy[n].get("masking_effect", {}).get(
                "avg_n_fragment_groups_total", 0
            )
            for n in names
        ])
        ax.set_title(
            f"Group Masking Status\n(avg {avg_total:.0f} groups/spectrum)"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel("Fraction of Fragment Groups")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

        # ── [0,1] Group Masking Status by Ion Series ─────────────────
        ax = axes[0, 1]
        all_series: set = set()
        for n in names:
            eff = per_strategy[n].get("masking_effect", {})
            all_series.update(eff.get("series_group_total", {}).keys())
        sorted_series = sorted(
            s for s in all_series if s in ("b", "y", "a")
        )

        if sorted_series:
            # One stacked bar per series; segments = fully-masked fraction per strategy
            # Show only the "fully masked" fraction — the actionable metric — grouped
            # by series, with one bar per strategy.
            n_series = len(sorted_series)
            n_strat = len(names)
            width = 0.7 / max(n_strat, 1)
            strategy_colors = plt.cm.tab10(np.linspace(0, 1, max(n_strat, 1)))

            for i, n in enumerate(names):
                eff = per_strategy[n].get("masking_effect", {})
                s_total = eff.get("series_group_total", {})
                s_full = eff.get("series_group_fully_masked", {})
                s_partial = eff.get("series_group_partially_masked", {})

                full_vals = []
                partial_vals = []
                for s in sorted_series:
                    t = s_total.get(s, 0)
                    if t > 0:
                        full_vals.append(s_full.get(s, 0) / t)
                        partial_vals.append(s_partial.get(s, 0) / t)
                    else:
                        full_vals.append(0.0)
                        partial_vals.append(0.0)

                xpos = np.arange(n_series) + i * width - (n_strat - 1) * width / 2
                ax.bar(
                    xpos, full_vals, width,
                    color=strategy_colors[i], alpha=0.85,
                    edgecolor="black", linewidth=0.5,
                    label=n,
                )
                # Hatch overlay for partial fraction
                ax.bar(
                    xpos, partial_vals, width, bottom=full_vals,
                    color=strategy_colors[i], alpha=0.35,
                    edgecolor="black", linewidth=0.3,
                    hatch="//",
                )

            ax.set_xticks(np.arange(n_series))
            ax.set_xticklabels(
                [f"{s}-series" for s in sorted_series], fontsize=10,
            )
            ax.set_ylabel("Fraction of Groups")
            ax.set_title(
                "Group Status by Ion Series\n"
                "(solid = fully masked, hatched = partially masked)"
            )
            ax.legend(fontsize=8, title="Strategy", title_fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.set_title("Group Status by Ion Series (no data)")
            ax.axis("off")

        # ── [1,0] Per-Spectrum Distribution of Fully Masked Groups ───
        ax = axes[1, 0]
        box_data = []
        box_labels = []
        for n in names:
            eff = per_strategy[n].get("masking_effect", {})
            dist = eff.get("dist_n_fully_masked", [])
            if dist:
                box_data.append(dist)
                box_labels.append(n)

        if box_data:
            bp = ax.boxplot(
                box_data, tick_labels=box_labels, patch_artist=True,
                showfliers=False, widths=0.6,
                medianprops=dict(color="black", linewidth=1.5),
            )
            colors = plt.cm.tab10(np.linspace(0, 1, len(box_labels)))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            # Annotate mean (diamond) and value above box
            for i, (dist_vals, label) in enumerate(zip(box_data, box_labels)):
                arr = np.array(dist_vals)
                mean = float(np.mean(arr))
                q3 = float(np.percentile(arr, 75))
                ax.scatter(
                    [i + 1], [mean], marker="D", color="red",
                    s=30, zorder=5, label="Mean" if i == 0 else None,
                )
                ax.text(
                    i + 1, q3 + 0.5, f"μ={mean:.1f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                )
            ax.set_ylabel("# Fully Masked Groups per Spectrum")
            ax.set_title(
                "Distribution of Fully Masked Groups\n(per spectrum)"
            )
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_xticklabels(box_labels, rotation=20, ha="right")
        else:
            ax.set_title("Distribution of Fully Masked Groups (no data)")
            ax.axis("off")

        # ── [1,1] Partial Masking Depth CDF ──────────────────────────
        # CDF of group mask fraction for partially-masked groups.
        # X-axis: fraction of group that is masked (0 = almost all visible,
        # 1 = almost all masked). Higher curves = more groups at lower
        # mask fractions = more leakage exposure.
        ax = axes[1, 1]
        has_partial = False
        for idx_n, n in enumerate(names):
            eff = per_strategy[n].get("masking_effect", {})
            fracs = eff.get("dist_partial_group_fractions", [])
            if fracs:
                has_partial = True
                sorted_vals = np.sort(fracs)
                cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) * 100
                ax.plot(
                    sorted_vals, cdf, linewidth=2,
                    color=colors[idx_n],
                    label=f"{n} (n={len(fracs):,d})",
                )
        if has_partial:
            ax.set_xlabel("Group Mask Fraction")
            ax.set_ylabel("Cumulative % of Partially-Masked Groups")
            ax.set_title(
                "Partial Masking Depth (CDF)\n"
                "(low fraction = most members visible = high leakage)"
            )
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 102)
        else:
            ax.set_title("Partial Masking Depth (no data)")
            ax.axis("off")

        plt.tight_layout()
        path = self.output_dir / "masking_fragment_group_status.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Fragment group status figure saved to: {path}")

    def _generate_fragment_group_analysis(self) -> None:
        """2x2 fragment group integrity: context, leakage, completeness, weighted."""
        per_strategy = self.results.get("per_strategy", {})
        names = list(per_strategy.keys())

        # Check if any strategy has masking effect data
        has_data = any(
            per_strategy[n].get("masking_effect", {}).get("avg_annotated_mask_ratio", 0) > 0
            for n in names
        )
        if not has_data:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Fragment Group Integrity\n"
            "(primary view: b/y for HCD·HCID·CID, c/z for ETD·ECD)",
            fontsize=15, fontweight="bold", y=0.99,
        )
        x = np.arange(len(names))
        colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

        # [0,0] Context Quality — annotated preservation ratio
        ax = axes[0, 0]
        pres = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_annotated_preservation_ratio", 0.0
            )
            for n in names
        ]

        if any(v > 0 for v in pres):
            bars = ax.bar(x, pres,
                          color=colors[:len(names)], alpha=0.85,
                          edgecolor="black", linewidth=0.5)
            for bar, val in zip(bars, pres):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=20, ha="right")
            ax.set_ylabel("Preservation Ratio")
            ax.set_title("Context Quality\n(fraction of annotated peaks preserved as context)")
            max_bar = max(pres)
            ax.set_ylim(0, min(max_bar * 1.3, 1.05))
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.set_title("Context Quality (no data)")
            ax.axis("off")

        # [0,1] Information Leakage — stacked pooled bars with per-spectrum
        # dispersion marker.
        #
        # Two different leakage quantities were previously labelled
        # interchangeably:
        #   - pooled  = total_loss / total_masked_base (pools every masked
        #     base across every spectrum; insensitive to per-spectrum
        #     variance but exactly equal to loss% + isotope% summed).
        #   - per-spectrum mean = mean over per-spectrum leakage_ratio
        #     (dispersion-aware, but does NOT decompose into loss/isotope
        #     fractions that sum to the stack).
        # The stack uses pooled so loss% + isotope% adds up cleanly; the
        # per-spectrum mean ± std is drawn as a black marker on top so
        # the reader can see whether the pooled rate misrepresents the
        # typical spectrum.
        ax = axes[0, 1]
        leak_loss = []
        leak_isotope = []
        leak_per_spectrum_mean = []
        leak_per_spectrum_std = []
        for n in names:
            leak = per_strategy[n].get("leakage", {})
            total_base = max(leak.get("total_masked_base", 0), 1)
            by_type = leak.get("leakage_by_type", {})
            leak_loss.append(by_type.get("loss", 0) / total_base)
            leak_isotope.append(by_type.get("isotope", 0) / total_base)
            leak_per_spectrum_mean.append(leak.get("avg_leakage_ratio", 0.0))
            leak_per_spectrum_std.append(leak.get("std_leakage_ratio", 0.0))
        if any(v > 0 for v in leak_loss) or any(v > 0 for v in leak_isotope):
            ax.bar(x, leak_loss, label="Loss leakage (pooled)", color="#e67e22",
                   alpha=0.85, edgecolor="black", linewidth=0.5)
            ax.bar(x, leak_isotope, bottom=leak_loss,
                   label="Isotope leakage (pooled)",
                   color="#f1c40f", alpha=0.85, edgecolor="black", linewidth=0.5)
            for i_bar in range(len(names)):
                ll = leak_loss[i_bar]
                li = leak_isotope[i_bar]
                if ll > 0.01:
                    ax.text(i_bar, ll / 2, f"{ll:.1%}",
                            ha="center", va="center", fontsize=8, fontweight="bold")
                if li > 0.01:
                    ax.text(i_bar, ll + li / 2, f"{li:.1%}",
                            ha="center", va="center", fontsize=8, fontweight="bold")
            # Per-spectrum mean ± std overlay (diamond + error bar). If
            # this marker disagrees with the top of the stack, spectra
            # are heterogeneous — some leak a lot, most don't.
            ax.errorbar(
                x,
                leak_per_spectrum_mean,
                yerr=leak_per_spectrum_std,
                fmt="D",
                color="#2c3e50",
                ecolor="#2c3e50",
                elinewidth=1.2,
                capsize=3,
                markersize=5,
                label="Per-spectrum mean ± std",
                zorder=5,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=20, ha="right")
            ax.set_ylabel("Leakage Ratio")
            ax.set_title(
                "Information Leakage\n"
                "(masked base ions with unmasked losses/isotopes)"
            )
            pooled_max = max(l + i for l, i in zip(leak_loss, leak_isotope))
            marker_max = max(
                (m + s for m, s in zip(leak_per_spectrum_mean, leak_per_spectrum_std)),
                default=0.0,
            )
            max_bar = max(pooled_max, marker_max)
            ax.set_ylim(0, min(max_bar * 1.5 + 0.05, 1.05))
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.set_title("Information Leakage (no data)")
            ax.axis("off")

        # [1,0] Fragment Group Completeness
        ax = axes[1, 0]
        full_mask = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_fragment_group_full_mask_ratio", 0.0
            )
            for n in names
        ]
        avg_mask = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_fragment_group_avg_mask_fraction", 0.0
            )
            for n in names
        ]
        if any(v > 0 for v in full_mask) or any(v > 0 for v in avg_mask):
            width = 0.35
            ax.bar(x - width / 2, full_mask, width, label="Fully Masked Groups",
                   color="#9b59b6", alpha=0.85, edgecolor="black", linewidth=0.5)
            ax.bar(x + width / 2, avg_mask, width, label="Avg Group Mask Fraction",
                   color="#f39c12", alpha=0.85, edgecolor="black", linewidth=0.5)
            for i_bar, (f, a) in enumerate(zip(full_mask, avg_mask)):
                ax.text(i_bar - width / 2, f + 0.005, f"{f:.1%}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
                ax.text(i_bar + width / 2, a + 0.005, f"{a:.1%}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=20, ha="right")
            ax.set_ylabel("Fraction")
            ax.set_title("Fragment Group Completeness\n(how completely groups are masked)")
            ax.legend(fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.set_title("Fragment Group Completeness (no data)")
            ax.axis("off")

        # [1,1] Weighted vs Unweighted Group Completeness
        ax = axes[1, 1]
        weighted_mask = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_fragment_group_weighted_mask_fraction", 0.0
            )
            for n in names
        ]
        if any(v > 0 for v in weighted_mask) or any(v > 0 for v in avg_mask):
            width = 0.35
            ax.bar(x - width / 2, avg_mask, width, label="Unweighted",
                   color="#3498db", alpha=0.85, edgecolor="black", linewidth=0.5)
            ax.bar(x + width / 2, weighted_mask, width, label="Intensity-Weighted",
                   color="#e74c3c", alpha=0.85, edgecolor="black", linewidth=0.5)
            for i_bar, (uw, w) in enumerate(zip(avg_mask, weighted_mask)):
                ax.text(i_bar - width / 2, uw + 0.005, f"{uw:.1%}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
                ax.text(i_bar + width / 2, w + 0.005, f"{w:.1%}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=20, ha="right")
            ax.set_ylabel("Group Mask Fraction")
            ax.set_title("Weighted vs Unweighted Group Completeness\n(are high-intensity group members preferentially masked?)")
            ax.legend(fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.set_title("Weighted vs Unweighted (no data)")
            ax.axis("off")

        plt.tight_layout()
        path = self.output_dir / "masking_fragment_group_analysis.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Fragment group analysis figure saved to: {path}")

    def _generate_precursor_analysis(self) -> None:
        """2x2 dedicated precursor ion masking analysis figure.

        Panels:
        - [0,0] Masked peak composition (count-based): stacked bar
        - [0,1] Masked peak composition (intensity-weighted): stacked bar
        - [1,0] Precursor masking frequency: grouped bar
        - [1,1] Precursor share of annotated signal: grouped bar with error bars
        """
        per_strategy = self.results.get("per_strategy", {})
        names = list(per_strategy.keys())

        # Check if any strategy has precursor masking data
        has_data = any(
            per_strategy[n].get("masking_effect", {}).get(
                "freq_precursor_base_masked", 0
            ) > 0
            or per_strategy[n].get("masking_effect", {}).get(
                "avg_precursor_mask_budget_fraction", 0
            ) > 0
            for n in names
        )
        if not has_data:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Precursor Ion Masking Analysis",
            fontsize=16, fontweight="bold", y=0.98,
        )
        x = np.arange(len(names))

        # ── [0,0] Masked Peak Composition (Count-Based) ──────────────
        ax = axes[0, 0]
        # fragment = annotated_fraction - precursor_budget_fraction
        ann_frac = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_annotated_fraction_of_masked_peaks", 0.0
            )
            for n in names
        ]
        prec_budget = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_precursor_mask_budget_fraction", 0.0
            )
            for n in names
        ]
        unann_frac = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_unannotated_fraction_of_masked_peaks", 0.0
            )
            for n in names
        ]
        frag_frac = [max(a - p, 0.0) for a, p in zip(ann_frac, prec_budget)]

        if any(v > 0 for v in ann_frac) or any(v > 0 for v in unann_frac):
            # Stacked bar: fragment (bottom), precursor (middle), unannotated (top)
            bars_frag = ax.bar(
                x, frag_frac, label="Fragment ions",
                color="#3498db", alpha=0.85, edgecolor="black", linewidth=0.5,
            )
            bars_prec = ax.bar(
                x, prec_budget, bottom=frag_frac, label="Precursor ions",
                color="#e74c3c", alpha=0.85, edgecolor="black", linewidth=0.5,
            )
            bottoms_unann = [f + p for f, p in zip(frag_frac, prec_budget)]
            bars_unann = ax.bar(
                x, unann_frac, bottom=bottoms_unann, label="Unannotated",
                color="#95a5a6", alpha=0.85, edgecolor="black", linewidth=0.5,
            )
            # Labels at segment midpoints (skip if <2%)
            for i_bar in range(len(names)):
                # Fragment
                if frag_frac[i_bar] >= 0.02:
                    ax.text(
                        i_bar, frag_frac[i_bar] / 2,
                        f"{frag_frac[i_bar]:.1%}",
                        ha="center", va="center", fontsize=8, fontweight="bold",
                    )
                # Precursor
                if prec_budget[i_bar] >= 0.02:
                    ax.text(
                        i_bar, frag_frac[i_bar] + prec_budget[i_bar] / 2,
                        f"{prec_budget[i_bar]:.1%}",
                        ha="center", va="center", fontsize=8, fontweight="bold",
                        color="white",
                    )
                # Unannotated
                if unann_frac[i_bar] >= 0.02:
                    ax.text(
                        i_bar, bottoms_unann[i_bar] + unann_frac[i_bar] / 2,
                        f"{unann_frac[i_bar]:.1%}",
                        ha="center", va="center", fontsize=8, fontweight="bold",
                    )
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=20, ha="right")
            ax.set_ylabel("Fraction of Masked Peaks")
            ax.set_ylim(0, 1.05)
            ax.set_title(
                "Masked Peak Composition (Count-Based)\n"
                "(what type of peaks consume the masking budget?)"
            )
            ax.legend(fontsize=9, loc="upper right")
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.set_title("Masked Peak Composition — Count (no data)")
            ax.axis("off")

        # ── [0,1] Masked Peak Composition (Intensity-Weighted) ───────
        ax = axes[0, 1]
        ann_int_frac = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_annotated_intensity_fraction_of_masked", 0.0
            )
            for n in names
        ]
        prec_int_budget = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_precursor_intensity_budget_fraction", 0.0
            )
            for n in names
        ]
        unann_int_frac = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_unannotated_intensity_fraction_of_masked", 0.0
            )
            for n in names
        ]
        frag_int_frac = [max(a - p, 0.0) for a, p in zip(ann_int_frac, prec_int_budget)]

        if any(v > 0 for v in ann_int_frac) or any(v > 0 for v in unann_int_frac):
            bars_frag = ax.bar(
                x, frag_int_frac, label="Fragment ions",
                color="#3498db", alpha=0.85, edgecolor="black", linewidth=0.5,
            )
            bars_prec = ax.bar(
                x, prec_int_budget, bottom=frag_int_frac, label="Precursor ions",
                color="#e74c3c", alpha=0.85, edgecolor="black", linewidth=0.5,
            )
            bottoms_unann_int = [f + p for f, p in zip(frag_int_frac, prec_int_budget)]
            bars_unann = ax.bar(
                x, unann_int_frac, bottom=bottoms_unann_int, label="Unannotated",
                color="#95a5a6", alpha=0.85, edgecolor="black", linewidth=0.5,
            )
            # Labels at segment midpoints
            for i_bar in range(len(names)):
                if frag_int_frac[i_bar] >= 0.02:
                    ax.text(
                        i_bar, frag_int_frac[i_bar] / 2,
                        f"{frag_int_frac[i_bar]:.1%}",
                        ha="center", va="center", fontsize=8, fontweight="bold",
                    )
                if prec_int_budget[i_bar] >= 0.02:
                    ax.text(
                        i_bar, frag_int_frac[i_bar] + prec_int_budget[i_bar] / 2,
                        f"{prec_int_budget[i_bar]:.1%}",
                        ha="center", va="center", fontsize=8, fontweight="bold",
                        color="white",
                    )
                if unann_int_frac[i_bar] >= 0.02:
                    ax.text(
                        i_bar, bottoms_unann_int[i_bar] + unann_int_frac[i_bar] / 2,
                        f"{unann_int_frac[i_bar]:.1%}",
                        ha="center", va="center", fontsize=8, fontweight="bold",
                    )
            # Intensity amplification annotation
            for i_bar in range(len(names)):
                if prec_budget[i_bar] > 0.001 and prec_int_budget[i_bar] > 0.001:
                    amp = prec_int_budget[i_bar] / prec_budget[i_bar]
                    ax.annotate(
                        f"{amp:.1f}x",
                        xy=(i_bar, frag_int_frac[i_bar] + prec_int_budget[i_bar]),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=7, fontstyle="italic", color="#c0392b",
                    )
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=20, ha="right")
            ax.set_ylabel("Fraction of Masked Intensity")
            ax.set_ylim(0, 1.05)
            ax.set_title(
                "Masked Peak Composition (Intensity-Weighted)\n"
                "(precursor amplification from high intensity)"
            )
            ax.legend(fontsize=9, loc="upper right")
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.set_title("Masked Peak Composition — Intensity (no data)")
            ax.axis("off")

        # ── [1,0] Precursor Masking Frequency ────────────────────────
        ax = axes[1, 0]
        prec_base_freq = [
            per_strategy[n].get("masking_effect", {}).get(
                "freq_precursor_base_masked", 0.0
            )
            for n in names
        ]
        prec_full_freq = [
            per_strategy[n].get("masking_effect", {}).get(
                "freq_precursor_fully_masked", 0.0
            )
            for n in names
        ]
        prec_leakage = [
            per_strategy[n].get("masking_effect", {}).get(
                "freq_precursor_has_leakage", 0.0
            )
            for n in names
        ]
        has_freq = any(
            v > 0 for lst in (prec_base_freq, prec_full_freq, prec_leakage) for v in lst
        )
        if has_freq:
            width = 0.25
            ax.bar(
                x - width, prec_base_freq, width, label="Base masked freq",
                color="#3498db", alpha=0.85, edgecolor="black", linewidth=0.5,
            )
            ax.bar(
                x, prec_full_freq, width, label="Fully masked freq",
                color="#9b59b6", alpha=0.85, edgecolor="black", linewidth=0.5,
            )
            ax.bar(
                x + width, prec_leakage, width, label="Leakage freq",
                color="#e74c3c", alpha=0.85, edgecolor="black", linewidth=0.5,
            )
            # Value labels
            for i_bar in range(len(names)):
                for offset, vals in [
                    (-width, prec_base_freq),
                    (0, prec_full_freq),
                    (width, prec_leakage),
                ]:
                    if vals[i_bar] > 0.005:
                        ax.text(
                            i_bar + offset, vals[i_bar] + 0.005,
                            f"{vals[i_bar]:.0%}",
                            ha="center", va="bottom", fontsize=8, fontweight="bold",
                        )
            # Reference line at overall mask ratio
            mask_ratios = [
                per_strategy[n].get("masking_effect", {}).get(
                    "avg_overall_mask_ratio", 0.0
                )
                for n in names
            ]
            avg_mask_ratio = float(np.mean([r for r in mask_ratios if r > 0])) if any(
                r > 0 for r in mask_ratios
            ) else 0.3
            ax.axhline(
                avg_mask_ratio, color="gray", linestyle="--", linewidth=1, alpha=0.7,
                label=f"Avg mask ratio ({avg_mask_ratio:.0%})",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=20, ha="right")
            ax.set_ylabel("Frequency")
            ax.set_title(
                "Precursor Masking Frequency\n"
                "(how often masking interacts with precursor peaks)"
            )
            max_val = max(max(prec_base_freq), max(prec_full_freq), max(prec_leakage))
            ax.set_ylim(0, min(max_val * 1.3 + 0.05, 1.05))
            ax.legend(fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.set_title("Precursor Masking Frequency (no data)")
            ax.axis("off")

        # ── [1,1] Precursor Share of Annotated Signal ────────────────
        ax = axes[1, 1]
        prec_ann_frac = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_precursor_fraction_of_annotated_masked", 0.0
            )
            for n in names
        ]
        prec_ann_frac_std = [
            per_strategy[n].get("masking_effect", {}).get(
                "std_precursor_fraction_of_annotated_masked", 0.0
            )
            for n in names
        ]
        prec_ann_int_frac = [
            per_strategy[n].get("masking_effect", {}).get(
                "avg_precursor_intensity_fraction_of_annotated_masked", 0.0
            )
            for n in names
        ]
        prec_ann_int_frac_std = [
            per_strategy[n].get("masking_effect", {}).get(
                "std_precursor_intensity_fraction_of_annotated_masked", 0.0
            )
            for n in names
        ]
        has_share = any(v > 0 for v in prec_ann_frac) or any(v > 0 for v in prec_ann_int_frac)
        if has_share:
            width = 0.35
            ax.bar(
                x - width / 2, prec_ann_frac, width,
                yerr=prec_ann_frac_std, capsize=3,
                label="Count-based", color="#3498db", alpha=0.85,
                edgecolor="black", linewidth=0.5,
            )
            ax.bar(
                x + width / 2, prec_ann_int_frac, width,
                yerr=prec_ann_int_frac_std, capsize=3,
                label="Intensity-based", color="#e74c3c", alpha=0.85,
                edgecolor="black", linewidth=0.5,
            )
            for i_bar in range(len(names)):
                cf = prec_ann_frac[i_bar]
                intf = prec_ann_int_frac[i_bar]
                if cf > 0.005:
                    ax.text(
                        i_bar - width / 2, cf + prec_ann_frac_std[i_bar] + 0.005,
                        f"{cf:.1%}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold",
                    )
                if intf > 0.005:
                    ax.text(
                        i_bar + width / 2, intf + prec_ann_int_frac_std[i_bar] + 0.005,
                        f"{intf:.1%}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold",
                    )
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=20, ha="right")
            ax.set_ylabel("Fraction of Annotated Masked")
            ax.set_title(
                "Precursor Share of Annotated Masked Peaks\n"
                "(precursor ions as fraction of all annotated masked peaks)"
            )
            max_val = max(
                max(a + s for a, s in zip(prec_ann_frac, prec_ann_frac_std)),
                max(a + s for a, s in zip(prec_ann_int_frac, prec_ann_int_frac_std)),
            )
            ax.set_ylim(0, min(max_val * 1.4 + 0.02, 1.05))
            ax.legend(fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.set_title("Precursor Share of Annotated Signal (no data)")
            ax.axis("off")

        plt.tight_layout()
        path = self.output_dir / "masking_precursor_analysis.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Precursor analysis figure saved to: {path}")

    def _generate_per_strategy_behavior(self) -> None:
        """Generate per-strategy masking behavior figures.

        For each strategy, produces a 2x2 figure:
        - [0,0] Mask contiguity — run length distribution
        - [0,1] Mask rate vs peak intensity — intensity bias
        - [1,0] Spatial masking density — mask rate by m/z region
        - [1,1] Nearest unmasked distance — detailed single-strategy histogram
        """
        per_strategy = self.results.get("per_strategy", {})

        for name, data in per_strategy.items():
            behavior = data.get("behavior", {})
            if not behavior:
                continue

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(
                f"Masking Behavior: {name}", fontsize=16, fontweight="bold"
            )

            # [0,0] Mask Contiguity (run length distribution)
            ax = axes[0, 0]
            run_lengths = behavior.get("run_lengths")
            if run_lengths is not None and len(run_lengths) > 0:
                max_rl = min(int(np.percentile(run_lengths, 99)), 30)
                bins = np.arange(0.5, max_rl + 1.5, 1)
                ax.hist(
                    run_lengths, bins=bins, alpha=0.7,
                    color="steelblue", edgecolor="black", linewidth=0.5,
                )
                mean_rl = float(run_lengths.mean())
                median_rl = float(np.median(run_lengths))
                ax.axvline(
                    mean_rl, color="red", linestyle="--",
                    label=f"Mean: {mean_rl:.1f}",
                )
                ax.axvline(
                    median_rl, color="green", linestyle="--",
                    label=f"Median: {median_rl:.1f}",
                )
                frac_isolated = float((run_lengths == 1).sum()) / len(
                    run_lengths
                ) * 100
                ax.text(
                    0.95, 0.95,
                    f"Isolated (len=1): {frac_isolated:.0f}%",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round", facecolor="wheat", alpha=0.5
                    ),
                )
                ax.set_xlabel("Consecutive Masked Run Length")
                ax.set_ylabel("Count")
                ax.set_title(
                    "Mask Contiguity\n"
                    "(distribution of consecutive masked spans)"
                )
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
            else:
                ax.set_title("Mask Contiguity (no data)")
                ax.axis("off")

            # [0,1] Mask Rate vs Peak Intensity
            ax = axes[0, 1]
            imc = behavior.get("intensity_mask_curve", {})
            mean_rates = imc.get("mean_rates")
            if mean_rates is not None:
                n_bins = len(mean_rates)
                pct_centers = [
                    (i + 0.5) * (100 / n_bins) for i in range(n_bins)
                ]
                ax.plot(
                    pct_centers, mean_rates, "o-",
                    color="#e74c3c", linewidth=2, markersize=6,
                )
                std_rates = imc.get("std_rates")
                if std_rates is not None:
                    lower = np.array(mean_rates) - np.array(std_rates)
                    upper = np.array(mean_rates) + np.array(std_rates)
                    ax.fill_between(
                        pct_centers, lower, upper,
                        alpha=0.2, color="#e74c3c",
                    )
                overall_mr = data.get("mask_ratio_stats", {}).get("mean")
                if overall_mr:
                    ax.axhline(
                        overall_mr, color="gray", linestyle=":",
                        label=f"Overall: {overall_mr:.1%}",
                    )
                ax.set_xlabel("Intensity Percentile")
                ax.set_ylabel("Mask Rate")
                ax.set_title(
                    "Mask Rate vs Peak Intensity\n"
                    "(flat = uniform, rising = intensity-biased)"
                )
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 100)
            else:
                ax.set_title("Mask Rate vs Intensity (no data)")
                ax.axis("off")

            # [1,0] Spatial Masking Density
            ax = axes[1, 0]
            smc = behavior.get("spatial_mask_curve", {})
            smr = smc.get("mean_rates")
            bin_centers = smc.get("mz_bin_centers")
            if smr is not None and bin_centers is not None:
                smr_arr = np.array(smr)
                bc_arr = np.array(bin_centers)
                valid = ~np.isnan(smr_arr)
                if valid.any():
                    bar_width = (bc_arr[1] - bc_arr[0]) * 0.9 if len(bc_arr) > 1 else 50
                    ax.bar(
                        bc_arr[valid], smr_arr[valid], width=bar_width,
                        alpha=0.7, color="#2ecc71",
                        edgecolor="black", linewidth=0.5,
                    )
                    overall_mr = data.get("mask_ratio_stats", {}).get("mean")
                    if overall_mr:
                        ax.axhline(
                            overall_mr, color="red", linestyle="--",
                            label=f"Overall: {overall_mr:.1%}",
                        )
                    ax.set_xlabel("m/z (Da)")
                    ax.set_ylabel("Mask Rate")
                    ax.set_title(
                        "Spatial Masking Density\n"
                        "(mask rate by m/z region)"
                    )
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.set_title("Spatial Masking Density (no data)")
                    ax.axis("off")
            else:
                ax.set_title("Spatial Masking Density (no data)")
                ax.axis("off")

            # [1,1] Nearest Unmasked Distance Distribution
            ax = axes[1, 1]
            gap = data.get("gap_analysis", {})
            raw = gap.get("raw_data")
            if (
                raw is not None
                and isinstance(raw, pd.DataFrame)
                and "nearest_distance_da" in raw.columns
            ):
                vals = raw["nearest_distance_da"].dropna()
                pos_vals = vals[vals > 0]
                if len(pos_vals) > 0:
                    log_bins = np.logspace(
                        np.log10(max(pos_vals.min(), 1e-3)),
                        np.log10(pos_vals.max()),
                        50,
                    )
                    ax.hist(
                        pos_vals, bins=log_bins, alpha=0.7,
                        color="darkorange", edgecolor="black", linewidth=0.5,
                    )
                    ax.set_xscale("log")
                    mean_val = pos_vals.mean()
                    median_val = pos_vals.median()
                    ax.axvline(
                        mean_val, color="red", linestyle="--",
                        label=f"Mean: {mean_val:.3f} Da",
                    )
                    ax.axvline(
                        median_val, color="green", linestyle="--",
                        label=f"Median: {median_val:.3f} Da",
                    )
                    ax.set_xlabel("Nearest Unmasked Distance (Da)")
                    ax.set_ylabel("Count")
                    ax.set_title(
                        "Nearest Unmasked Neighbor Distance"
                    )
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.set_title("Nearest Distance (no data)")
                    ax.axis("off")
            else:
                ax.set_title("Nearest Distance (no data)")
                ax.axis("off")

            plt.tight_layout()
            output_dir = self.output_dir / name
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / "masking_behavior.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Masking behavior figure saved to: {path}")

    def _generate_stratified_masking_summary(self) -> None:
        """Generate stratified masking summary figure.

        Each row = one metric (annotated mask ratio, unannotated mask
        ratio, intensity-weighted annotated mask fraction). Each column =
        one stratum (frag_type, precursor_charge, instrument). Within a
        cell, x-axis = group value (e.g. HCD/CID/ETD), bar colour =
        strategy. This lets the reader answer "how does strategy X
        behave on ETD vs HCD?" at a glance — previously only the first
        strategy was plotted.
        """
        stratified = self.results.get("stratified", {})
        if not stratified:
            return

        strat_panels = [
            ("frag_type", "By Fragmentation Type"),
            ("precursor_charge", "By Precursor Charge"),
            ("search_instrument", "By Instrument"),
        ]
        active_panels = [
            (key, title) for key, title in strat_panels
            if stratified.get(key)
        ]
        if not active_panels:
            return

        metrics = [
            ("avg_annotated_mask_ratio", "Annotated mask %"),
            ("avg_unannotated_mask_ratio", "Unannotated mask %"),
            ("avg_annotated_intensity_masked_frac", "Ann. intensity masked %"),
        ]

        strategy_names = list(self.results.get("per_strategy", {}).keys())
        if not strategy_names:
            return

        # Stable colour per strategy (tab10 wraps for >10 strategies).
        import matplotlib as mpl
        cmap = mpl.colormaps.get_cmap("tab10")
        strategy_colors = {
            s: cmap(i % cmap.N) for i, s in enumerate(strategy_names)
        }

        n_rows = len(metrics)
        n_cols = len(active_panels)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(6 * n_cols, 4.2 * n_rows),
            squeeze=False,
        )
        fig.suptitle(
            "Stratified Masking Summary (per strategy)",
            fontsize=14, fontweight="bold", y=1.0,
        )

        for col, (key, title) in enumerate(active_panels):
            groups = stratified[key]
            if key == "precursor_charge":
                def _charge_sort_key(kv: tuple) -> tuple:
                    try:
                        return (0, int(kv[0]))
                    except (ValueError, TypeError):
                        return (1, kv[0])
                items = sorted(groups.items(), key=_charge_sort_key)
            else:
                items = sorted(
                    groups.items(),
                    key=lambda kv: kv[1].get("n_total", 0),
                    reverse=True,
                )
            if not items:
                for row in range(n_rows):
                    axes[row, col].axis("off")
                continue

            labels = [f"{name}\n(n={m.get('n_total', 0)})" for name, m in items]
            x = np.arange(len(labels))
            n_strat = len(strategy_names)
            bar_w = 0.8 / max(n_strat, 1)

            for row, (metric_key, metric_label) in enumerate(metrics):
                ax = axes[row, col]
                any_bars = False
                for s_idx, strategy in enumerate(strategy_names):
                    vals = []
                    for _, m in items:
                        bs = m.get("by_strategy", {}).get(strategy, {})
                        vals.append(bs.get(metric_key, 0) * 100)
                    if any(v > 0 for v in vals):
                        any_bars = True
                    offset = (s_idx - (n_strat - 1) / 2) * bar_w
                    ax.bar(
                        x + offset, vals, width=bar_w,
                        label=strategy,
                        color=strategy_colors[strategy],
                        alpha=0.85,
                        edgecolor="black", linewidth=0.3,
                    )
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
                ax.set_ylabel("Percent")
                if row == 0:
                    ax.set_title(title, fontsize=11, fontweight="bold")
                # Per-row label on the leftmost column for clarity.
                if col == 0:
                    ax.text(
                        -0.17, 0.5, metric_label,
                        transform=ax.transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=10, fontweight="bold",
                    )
                ax.grid(True, axis="y", alpha=0.3)
                if not any_bars:
                    ax.text(
                        0.5, 0.5, "no data",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=10, color="#888",
                    )

        # One legend for the whole figure (strategy → colour).
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=strategy_colors[s], alpha=0.85)
            for s in strategy_names
        ]
        fig.legend(
            handles, strategy_names,
            loc="lower center",
            ncol=min(len(strategy_names), 5),
            fontsize=9,
            bbox_to_anchor=(0.5, -0.02),
            title="Strategy",
        )
        plt.tight_layout(rect=(0.03, 0.04, 1, 0.97))
        path = self.output_dir / "stratified_masking_summary.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Stratified masking summary saved to: {path}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(self) -> None:
        """Save comparison CSVs and per-strategy JSON/CSV."""
        if not self.results:
            return

        comparison = self.results.get("comparison", {})

        # Comparison CSVs
        for key, df in comparison.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                csv_path = self.output_dir / f"masking_comparison_{key}.csv"
                df.to_csv(csv_path, float_format="%.6f")
                logger.info(f"Comparison CSV saved: {csv_path}")

        # Per-strategy results
        for name, analyser in self.gap_analysers.items():
            if analyser.results:
                analyser.save_results()

        # Summary JSON
        summary_data = self._prepare_json_summary()
        summary_path = self.output_dir / "masking_comparison_summary.json"
        with open(summary_path, "w") as f:
            json.dump(
                summary_data, f, indent=2,
                default=lambda obj: float(obj) if isinstance(obj, (np.floating, np.integer)) else str(obj),
            )
        logger.info(f"Summary JSON saved: {summary_path}")

    def _prepare_json_summary(self) -> Dict[str, Any]:
        """Prepare a JSON-serialisable summary (exclude raw DataFrames)."""
        if not self.results:
            return {}

        out: Dict[str, Any] = {
            "n_spectra": self.results.get("n_spectra", 0),
            "n_strategies": self.results.get("n_strategies", 0),
            "per_strategy": self._serialise_per_strategy(
                self.results.get("per_strategy", {})
            ),
            "quality_gate": self.results.get("quality_gate", {}),
            "stratified": self.results.get("stratified", {}),
        }

        # Gold-standard (annotation-driven) companion view; only emitted
        # when aggregate_results actually built it (signal_aware_fragment
        # is configured and at least one spectrum used the annotation
        # path).
        ad = self.results.get("per_strategy_annotation_driven")
        if ad:
            out["per_strategy_annotation_driven"] = self._serialise_per_strategy(ad)
            out["annotation_driven_meta"] = self.results.get(
                "annotation_driven_meta", {}
            )

        return out

    @staticmethod
    def _serialise_per_strategy(
        per_strategy: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """JSON-safe per-strategy dict — strips raw DataFrames/arrays.

        Shared by the default view and the annotation-driven companion so
        both receive identical shape.
        """
        serialised: Dict[str, Dict[str, Any]] = {}
        for name, data in per_strategy.items():
            entry: Dict[str, Any] = {
                "mask_ratio_stats": data.get("mask_ratio_stats", {}),
                "masking_effect": data.get("masking_effect", {}),
                "leakage": data.get("leakage", {}),
                "fallback": data.get("fallback", {}),
                "n_spectra": data.get("n_spectra", 0),
            }
            gap = data.get("gap_analysis", {})
            entry["gap_summary"] = {
                k: v for k, v in gap.items() if k != "raw_data"
            }
            behavior = data.get("behavior", {})
            if behavior:
                behavior_out: Dict[str, Any] = {}
                rl = behavior.get("run_lengths")
                if rl is not None and len(rl) > 0:
                    behavior_out["run_length_stats"] = {
                        "mean": float(rl.mean()),
                        "median": float(np.median(rl)),
                        "max": int(rl.max()),
                        "frac_isolated": float((rl == 1).sum() / len(rl)),
                        "n_runs": len(rl),
                    }
                if "intensity_mask_curve" in behavior:
                    behavior_out["intensity_mask_curve"] = (
                        behavior["intensity_mask_curve"]
                    )
                if "spatial_mask_curve" in behavior:
                    behavior_out["spatial_mask_curve"] = (
                        behavior["spatial_mask_curve"]
                    )
                entry["behavior"] = behavior_out
            serialised[name] = entry
        return serialised

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print cross-strategy comparison summary."""
        if not self.results:
            return

        logger.info("=" * 80)
        logger.info("MASKING STRATEGY COMPARISON SUMMARY")
        logger.info("=" * 80)

        n_spectra = self.results.get("n_spectra", 0)
        logger.info(f"Total spectra analyzed: {n_spectra:,d}")
        logger.info(f"Strategies compared: {self.results.get('n_strategies', 0)}")

        per_strategy = self.results.get("per_strategy", {})

        # Mask ratio table
        logger.info(
            f"\n{'Strategy':>25} {'Mask Ratio':>12} {'Nearest(Da)':>14} "
            f"{'Ann.Frac':>10} {'Preserv.':>10} {'Leakage':>10}"
        )
        logger.info("  " + "-" * 81)

        for name, data in per_strategy.items():
            mr = data.get("mask_ratio_stats", {})
            gap = data.get("gap_analysis", {}).get("summary", {}).get("nearest_distance_da", {})
            eff = data.get("masking_effect", {})
            leak = data.get("leakage", {})
            logger.info(
                f"  {name:>23} {mr.get('mean', 0):.4f}+/-{mr.get('std', 0):.4f}"
                f" {gap.get('mean', 0):>12.4f}"
                f" {eff.get('avg_annotated_fraction_of_masked_peaks', 0):>10.4f}"
                f" {eff.get('avg_annotated_preservation_ratio', 0):>10.4f}"
                f" {leak.get('avg_leakage_ratio', 0):>10.4f}"
            )

        # Fragment group & precursor summary table
        has_frag_data = any(
            data.get("masking_effect", {}).get("avg_n_fragment_groups_total", 0) > 0
            for data in per_strategy.values()
        )
        if has_frag_data:
            logger.info(
                f"\n{'Strategy':>25} {'FGrp Avg':>10} {'Full%':>8} "
                f"{'Part%':>8} {'Unmsk%':>8} {'Prec.Mask':>10} "
                f"{'Prec.Ann%':>10} {'Prec.IntAnn%':>13}"
            )
            logger.info("  " + "-" * 97)
            for name, data in per_strategy.items():
                eff = data.get("masking_effect", {})
                avg_total = eff.get("avg_n_fragment_groups_total", 0)
                avg_full = eff.get("avg_n_fragment_groups_fully_masked", 0)
                avg_part = eff.get("avg_n_fragment_groups_partially_masked", 0)
                avg_unmsk = eff.get("avg_n_fragment_groups_unmasked", 0)
                pct_full = avg_full / max(avg_total, 1e-9) * 100
                pct_part = avg_part / max(avg_total, 1e-9) * 100
                pct_unmsk = avg_unmsk / max(avg_total, 1e-9) * 100
                prec_mask = eff.get("freq_precursor_base_masked", 0)
                prec_ann = eff.get(
                    "avg_precursor_fraction_of_annotated_masked", 0
                )
                prec_int_ann = eff.get(
                    "avg_precursor_intensity_fraction_of_annotated_masked", 0
                )
                logger.info(
                    f"  {name:>23} {avg_total:>10.1f} {pct_full:>7.1f}%"
                    f" {pct_part:>7.1f}% {pct_unmsk:>7.1f}%"
                    f" {prec_mask:>10.1%} {prec_ann:>10.1%}"
                    f" {prec_int_ann:>13.1%}"
                )

        # Per-strategy gap summary
        for name in per_strategy:
            analyser = self.gap_analysers.get(name)
            if analyser and analyser.results:
                logger.info(f"\n--- {name} gap summary ---")
                analyser.print_summary()

        logger.info("=" * 80)
