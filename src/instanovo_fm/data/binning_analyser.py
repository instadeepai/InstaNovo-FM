#!/usr/bin/env python
"""Binning Strategy Analyzer for Foundation Model Training.

Performs analysis of m/z binning strategies including:
1. Part A: Binning simulation (bin error rates from mass measurement error)
2. Part B: Bin-jump analysis (label noise from repeated observations)
3. Part B2: Intra-spectrum collision analysis
4. Resolution-noise tradeoff (combines jump rate + collision)
5. Stratified analysis (per-fragmentation-type jump rates)

This module is extracted from SpectrumAnalyser to improve code organization
and maintainability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from instanovo.__init__ import console
from instanovo_fm.trainer.binning import (
    AdaptiveBinning,
    FixedDaBinning,
    FixedPpmBinning,
)
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger


class BinningAnalyser:
    """Analyzer for m/z binning strategies.

    Analyzes:
    - Part A: Binning simulation (theo/exp bin mismatches)
    - Part B: Bin-jump rates (label noise from repeated observations)
    - Part B2: Intra-spectrum collision (information loss from coarse binning)
    - Resolution-noise tradeoff
    - Stratified jump rate analysis (per fragmentation type)
    """

    def __init__(self, config: DictConfig, output_dir: Optional[Path] = None) -> None:
        """Initialize the binning analyzer.

        Args:
            config: Hydra configuration
            output_dir: Output directory for results and visualizations
        """
        self.config = config

        # Set output directory
        if output_dir is None:
            self.output_dir = Path("analysis_output") / "binning_analysis"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model configuration
        self.max_mz = config.model.get("max_mz", 2500.0)
        self.min_mz = config.model.get("min_mz", 50.0)

        # Analysis configuration — supports both task_configs and legacy flat
        analysis_config = config.get("analysis", {})

        # Resolve task_configs for binning-specific settings
        task_configs_raw = analysis_config.get("task_configs", {})
        if hasattr(task_configs_raw, "_metadata"):
            binning_tc = OmegaConf.to_container(task_configs_raw.get("binning", {}), resolve=True)
        elif hasattr(task_configs_raw, "get"):
            binning_tc = dict(task_configs_raw.get("binning", {}))
        else:
            binning_tc = {}

        def _tc(key: str, default: Any = None) -> Any:
            """Read from task_configs.binning[key], fallback to analysis_config[key]."""
            if binning_tc and key in binning_tc:
                return binning_tc[key]
            return analysis_config.get(key, default)

        self.ppm_tol = _tc("ppm_tol", 10.0)

        # Bin-aligned m/z region definitions
        self.mz_range_boundaries = _tc(
            "mz_range_boundaries",
            {
                "immonium_internal": (0, 200),
                "core_fragment": (200, 800),
                "extended_fragment": (800, 1500),
                "high_mass_fragment": (1500, float("inf")),
            },
        )
        self.mz_range_order = ["immonium_internal", "core_fragment", "extended_fragment", "high_mass_fragment"]

        # Analysis feature toggles
        self.enable_bin_jump_analysis = _tc("enable_bin_jump_analysis", True)
        self.min_ion_observations = _tc("min_ion_observations", 2)
        self.bin_jump_threshold = _tc("bin_jump_threshold", 0.10)

        self.enable_intra_collision_analysis = _tc("enable_intra_collision_analysis", True)
        self.intra_collision_weight = _tc("intra_collision_weight", 0.5)

        self.enable_stratified_analysis = _tc("enable_stratified_analysis", True)

        self.enable_cid_analysis = _tc("enable_cid_analysis", True)
        self.cid_error_sweep_da = list(
            _tc(
                "cid_error_sweep_da",
                [
                    0.005,
                    0.01,
                    0.02,
                    0.03,
                    0.05,
                    0.08,
                    0.1,
                    0.15,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                ],
            )
        )
        self.cid_literature_errors_da = list(_tc("cid_literature_errors_da", [0.3, 0.4, 0.5]))
        self.orbitrap_reference_errors_ppm = list(_tc("orbitrap_reference_errors_ppm", [5.0, 10.0]))

        self.enable_resolution_information = _tc("enable_resolution_information", True)
        self.resolution_info_window_size = _tc("resolution_info_window_size", 100.0)
        self.resolution_info_min_peaks_per_window = _tc("resolution_info_min_peaks_per_window", 30)

        # Group size sensitivity analysis (Part C2)
        self.enable_group_size_sensitivity = _tc("enable_group_size_sensitivity", True)
        self.group_size_candidates = list(_tc("group_size_candidates", [25, 50, 100, 200, 500]))
        self.group_size_max_offset_utilization = _tc("group_size_max_offset_utilization", 0.85)
        self.group_size_min_offset_headroom = _tc("group_size_min_offset_headroom", 1.0)
        self.group_size_max_n_groups = _tc("group_size_max_n_groups", 10000)

        # Binning strategies from config (task_configs.binning.strategies or legacy flat)
        strategies = _tc("strategies", None)
        if strategies is None:
            strategies = analysis_config.get("binning_strategies", None)
        self.binning_strategies = strategies

        # Results storage
        self.results: dict[str, Any] = {}

        logger.info(f"Binning analyzer initialized. Output directory: {self.output_dir}")

    def analyze(
        self,
        mass_error_df: pd.DataFrame,
        coverage_stats: Dict[str, Any],
        unmatched_theo_data: List[Dict],
        per_spectrum_mz: Optional[List[np.ndarray]] = None,
        per_spectrum_frag_type: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run binning strategy analysis.

        Args:
            mass_error_df: DataFrame with columns [theo_mz, exp_mz, delta_mz_da,
                          delta_mz_ppm, ion_type, charge, mz_range, frag_type,
                          feature_type, peptide, position]
            coverage_stats: Theoretical ion coverage statistics
            unmatched_theo_data: List of unmatched theoretical ions
            per_spectrum_mz: Optional list of numpy arrays, each containing
                            all valid m/z values for one spectrum
            per_spectrum_frag_type: Optional list of fragmentation types per spectrum,
                                  aligned with per_spectrum_mz

        Returns:
            Dict containing all analysis results
        """
        logger.info("=" * 80)
        logger.info("BINNING STRATEGY ANALYSIS")
        logger.info("=" * 80)

        if len(mass_error_df) == 0:
            logger.warning("No mass error data provided. Skipping binning analysis.")
            return {"error": "No data"}

        # Calculate overall and range-stratified statistics
        overall_stats = self._calculate_overall_stats(mass_error_df)
        range_stats = self._calculate_range_stats(mass_error_df)

        # Part A: Binning simulation (bin error rates)
        logger.info("Part A: Simulating binning strategies...")
        binning_sim = self._simulate_binning_strategies(mass_error_df)

        # Part B: Bin-jump statistics (label noise)
        bin_jump_analysis = None
        if self.enable_bin_jump_analysis:
            logger.info("Part B: Calculating bin-jump statistics...")
            bin_jump_analysis = self._calculate_bin_jump_statistics(mass_error_df)

        # Part B2: Intra-spectrum collision analysis
        intra_spectrum_collision = None
        if self.enable_intra_collision_analysis and per_spectrum_mz is not None and len(per_spectrum_mz) > 0:
            logger.info("Part B2: Calculating intra-spectrum collision statistics...")
            intra_spectrum_collision = self._calculate_intra_spectrum_collision_statistics(per_spectrum_mz, per_spectrum_frag_type)

        # Resolution information analysis (Part C) — computed before tradeoff
        # so prediction_entropy is available for effective information rate
        resolution_information = None
        if self.enable_resolution_information:
            logger.info("Calculating resolution information analysis...")
            resolution_information = self._calculate_resolution_information_analysis(mass_error_df, per_spectrum_mz)

        # Resolution-noise tradeoff (combines jump rate + intra-spectrum collision + entropy)
        resolution_noise_tradeoff = None
        if bin_jump_analysis and "per_strategy" in bin_jump_analysis and intra_spectrum_collision:
            prediction_entropy = resolution_information.get("prediction_entropy") if resolution_information else None
            resolution_noise_tradeoff = self._compute_resolution_noise_tradeoff(
                bin_jump_analysis,
                intra_spectrum_collision,
                prediction_entropy=prediction_entropy,
            )

        # Stratified analysis (per fragmentation type)
        stratified_analysis = None
        if self.enable_bin_jump_analysis and self.enable_stratified_analysis and "frag_type" in mass_error_df.columns:
            strategies = self._get_binning_strategies_from_config()
            logger.info("Calculating stratified jump rate analysis...")
            stratified_analysis = self._calculate_stratified_bin_analysis(mass_error_df, strategies)

        # Error model fit
        error_model_fit = self._fit_error_model(mass_error_df)

        # CID simulated analysis
        cid_analysis = None
        if self.enable_cid_analysis and per_spectrum_mz and per_spectrum_frag_type:
            logger.info("Calculating CID simulated analysis...")
            cid_analysis = self._calculate_cid_simulated_analysis(
                per_spectrum_mz,
                per_spectrum_frag_type,
                mass_error_df=mass_error_df,
            )

        # Store all results
        self.results = {
            "overall_stats": overall_stats,
            "range_stats": range_stats,
            "coverage_stats": coverage_stats,
            "binning_simulation": binning_sim,
            "raw_data": mass_error_df,
            "unmatched_theo_data": unmatched_theo_data,
            "bin_jump_analysis": bin_jump_analysis,
            "intra_spectrum_collision": intra_spectrum_collision,
            "resolution_noise_tradeoff": resolution_noise_tradeoff,
            "stratified_analysis": stratified_analysis,
            "resolution_information": resolution_information,
            "error_model_fit": error_model_fit,
            "cid_analysis": cid_analysis,
        }

        logger.info("Binning analysis complete.")
        return self.results

    def generate_visualizations(self) -> None:
        """Generate all binning-related visualizations."""
        logger.info("Generating binning visualizations...")

        if not self.results or "raw_data" not in self.results:
            logger.warning("No results available for visualization")
            return

        # Figure 1: Strategy comparison (2x2)
        self._generate_strategy_comparison_visualization()

        # Figure 2: Strategy shapes (3x1)
        self._generate_strategy_shapes_visualization()

        # Figure 3: Resolution-noise tradeoff (1x2)
        if self.results.get("resolution_noise_tradeoff"):
            self._generate_resolution_noise_tradeoff_visualization()

        # Figure 4: Group/offset balance (1x2)
        self._generate_group_offset_balance_visualization()

        # Figure 5: Stratified jump rates (only if multiple frag types)
        if self.results.get("stratified_analysis"):
            stratified = self.results["stratified_analysis"]
            if stratified.get("n_frag_types", 0) > 1:
                self._generate_stratified_jump_rates_visualization()

        # Figure 6: Resolution information (entropy, resolvability, local entropy)
        if self.results.get("resolution_information"):
            self._generate_resolution_information_visualization()

        # Figure 7: Group size sensitivity (2x2)
        if (self.results.get("resolution_information") or {}).get("group_size_sensitivity"):
            self._generate_group_size_sensitivity_visualization()

        # Figure 8: CID analysis (2x2)
        if self.results.get("cid_analysis"):
            self._generate_cid_analysis_visualization()

        logger.info("Binning visualizations complete")

    def save_results(self) -> None:
        """Save analysis results to files."""
        logger.info("Saving binning analysis results...")

        if not self.results:
            logger.warning("No results to save")
            return

        # Export per-peak mass error data (essential columns only to reduce file size)
        if "raw_data" in self.results:
            mass_error_df = self.results["raw_data"]
            save_cols = [
                c
                for c in [
                    "theo_mz",
                    "exp_mz",
                    "signed_ppm",
                    "ion_type",
                    "charge",
                    "frag_type",
                    "feature_type",
                ]
                if c in mass_error_df.columns
            ]
            mass_error_csv = self.output_dir / "mass_error_per_peak.csv"
            mass_error_df[save_cols].to_csv(mass_error_csv, index=False, float_format="%.6f")
            logger.info(f"Per-peak mass error data saved to: {mass_error_csv} ({len(mass_error_df):,d} peaks, {len(save_cols)} columns)")

        # Export per-ion bin-jump data if available
        if self.enable_bin_jump_analysis:
            bin_jump_analysis = self.results.get("bin_jump_analysis")
            if bin_jump_analysis and "per_ion_data" in bin_jump_analysis:
                bin_jump_df = bin_jump_analysis["per_ion_data"]
                bin_jump_csv = self.output_dir / "bin_jump_per_ion.csv"
                bin_jump_df.to_csv(bin_jump_csv, index=False, float_format="%.6f")
                logger.info(f"Per-ion bin-jump data saved to: {bin_jump_csv} ({len(bin_jump_df):,d} ions)")

        # Export resolution-noise tradeoff table
        tradeoff = self.results.get("resolution_noise_tradeoff")
        if tradeoff and "tradeoff_table" in tradeoff:
            tradeoff_df = pd.DataFrame(tradeoff["tradeoff_table"])
            tradeoff_csv = self.output_dir / "resolution_noise_tradeoff.csv"
            tradeoff_df.to_csv(tradeoff_csv, index=False, float_format="%.6f")
            logger.info(f"Resolution-noise tradeoff saved to: {tradeoff_csv} ({len(tradeoff_df)} strategies)")

        # Export resolution information CSVs
        res_info = self.results.get("resolution_information")
        if res_info:
            # 1. Prediction entropy table
            pred_ent = res_info.get("prediction_entropy", {})
            if pred_ent:
                rows: list[Any] = []
                for name, metrics in pred_ent.items():
                    rows.append(
                        {
                            "strategy": name,
                            "total_entropy_bits": metrics["total_entropy"],
                            "group_entropy_bits": metrics["group_entropy"],
                            "offset_entropy_bits": metrics["offset_entropy"],
                            "max_entropy_bits": metrics["max_entropy"],
                            "efficiency": metrics["efficiency"],
                            "effective_classes": metrics["effective_classes"],
                            "occupancy_rate": metrics["occupancy_rate"],
                            "n_bins": metrics["n_bins"],
                            "n_groups": metrics["n_groups"],
                            "bin_group_size": metrics["bin_group_size"],
                            "n_peaks": metrics["n_peaks"],
                        }
                    )
                ent_df = pd.DataFrame(rows)
                ent_csv = self.output_dir / "resolution_prediction_entropy.csv"
                ent_df.to_csv(ent_csv, index=False, float_format="%.6f")
                logger.info(f"Prediction entropy saved to: {ent_csv} ({len(ent_df)} strategies)")

            # 2. Physical resolvability table
            phys = res_info.get("physical_resolvability", {})
            if phys and "per_strategy" in phys:
                rows = []
                for strat_name, diffs in phys["per_strategy"].items():
                    for diff_name, entry in diffs.items():
                        rows.append(
                            {
                                "strategy": strat_name,
                                "mass_difference": diff_name,
                                "delta_da": phys["mass_differences"][diff_name],
                                "min_bins": entry["min_bins"],
                                "classification": entry["classification"],
                            }
                        )
                resolv_df = pd.DataFrame(rows)
                resolv_csv = self.output_dir / "resolution_physical_resolvability.csv"
                resolv_df.to_csv(resolv_csv, index=False, float_format="%.6f")
                logger.info(f"Physical resolvability saved to: {resolv_csv} ({len(resolv_df)} entries)")

            # 3. Local entropy table
            local = res_info.get("local_entropy", {})
            if local and "per_strategy" in local:
                window_centers = local["window_centers"]
                rows = []
                for strat_name, data in local["per_strategy"].items():
                    for i, (ent, count) in enumerate(zip(data["entropies"], data["peak_counts"], strict=False)):
                        rows.append(
                            {
                                "strategy": strat_name,
                                "mz_window_center": window_centers[i],
                                "entropy_bits": ent,
                                "peak_count": count,
                            }
                        )
                local_df = pd.DataFrame(rows)
                local_csv = self.output_dir / "resolution_local_entropy.csv"
                local_df.to_csv(local_csv, index=False, float_format="%.6f")
                logger.info(f"Local entropy saved to: {local_csv} ({len(local_df)} entries)")

            # 4. Group size sensitivity table
            gs_sens = res_info.get("group_size_sensitivity")
            if gs_sens and "per_strategy" in gs_sens:
                rows = []
                for strat_name, gs_results in gs_sens["per_strategy"].items():
                    for gs, metrics in sorted(gs_results.items()):
                        rows.append(
                            {
                                "strategy": strat_name,
                                "group_size": gs,
                                "n_groups": metrics["n_groups"],
                                "n_offset_classes": metrics["n_offset_classes"],
                                "h_bin_bits": metrics["h_bin"],
                                "h_group_bits": metrics["h_group"],
                                "h_offset_given_group_bits": metrics["h_offset_given_group"],
                                "max_h_offset_bits": metrics["max_h_offset"],
                                "offset_utilization": metrics["offset_utilization"],
                                "offset_headroom_bits": metrics["offset_headroom"],
                                "head_balance": metrics["head_balance"],
                                "group_width_da_mean": metrics["group_width_da_mean"],
                                "group_width_da_min": metrics["group_width_da_min"],
                                "group_width_da_max": metrics["group_width_da_max"],
                                "recommended": metrics["recommended"],
                            }
                        )
                gs_df = pd.DataFrame(rows)
                gs_csv = self.output_dir / "resolution_group_size_sensitivity.csv"
                gs_df.to_csv(gs_csv, index=False, float_format="%.6f")
                logger.info(f"Group size sensitivity saved to: {gs_csv} ({len(gs_df)} entries)")

        # Export error model fit data
        error_model = self.results.get("error_model_fit")
        if error_model and error_model.get("per_bin_data"):
            rows = []
            for bin_data in error_model["per_bin_data"]:
                row = dict(bin_data)
                # Add per-strategy bin width and safety margin at this m/z
                mz_center = bin_data["mz_bin_center"]
                for strat_name, margins in error_model.get("safety_margins", {}).items():
                    # Find closest reference m/z
                    ref_mzs = sorted(margins.keys())
                    closest_ref = min(ref_mzs, key=lambda r: abs(r - mz_center))
                    m = margins[closest_ref]
                    row[f"bin_width_{strat_name}"] = m["bin_width"]
                    row[f"safety_margin_{strat_name}"] = m["safety_margin"]
                rows.append(row)
            fit_df = pd.DataFrame(rows)
            fit_csv = self.output_dir / "error_model_fit.csv"
            fit_df.to_csv(fit_csv, index=False, float_format="%.6f")
            logger.info(f"Error model fit saved to: {fit_csv} ({len(fit_df)} bins)")

        # Export CID simulated mismatch data
        cid = self.results.get("cid_analysis")
        if cid and "simulated_bin_mismatch" in cid:
            rows = []
            for strat_name, sim in cid["simulated_bin_mismatch"].items():
                for delta_da, data in sim.get("sweep", {}).items():
                    row = {
                        "strategy": strat_name,
                        "error_type": "Da_sweep",
                        "error_value": delta_da,
                        "error_unit": "Da",
                        "is_literature": data.get("is_literature", False),
                        "mismatch_rate": data["mismatch_rate"],
                        "n_peaks": data["n_peaks"],
                    }
                    for rn, rd in data.get("per_mz_range", {}).items():
                        row[f"mismatch_{rn}"] = rd["mismatch_rate"]
                    rows.append(row)
                for ppm_val, data in sim.get("orbitrap_errors", {}).items():
                    row = {
                        "strategy": strat_name,
                        "error_type": "Orbitrap",
                        "error_value": ppm_val,
                        "error_unit": "PPM",
                        "is_literature": False,
                        "mismatch_rate": data["mismatch_rate"],
                        "n_peaks": data["n_peaks"],
                    }
                    for rn, rd in data.get("per_mz_range", {}).items():
                        row[f"mismatch_{rn}"] = rd["mismatch_rate"]
                    rows.append(row)
            if rows:
                cid_df = pd.DataFrame(rows)
                cid_csv = self.output_dir / "cid_simulated_mismatch.csv"
                cid_df.to_csv(cid_csv, index=False, float_format="%.6f")
                logger.info(f"CID simulated mismatch saved to: {cid_csv} ({len(cid_df)} rows)")

            # Export observed CID errors
            obs = cid.get("observed_cid_errors")
            if obs:
                obs_rows = [{k: v for k, v in obs.items() if k != "note"}]
                obs_df = pd.DataFrame(obs_rows)
                obs_csv = self.output_dir / "observed_cid_errors.csv"
                obs_df.to_csv(obs_csv, index=False, float_format="%.6f")
                logger.info(f"Observed CID errors saved to: {obs_csv}")

        logger.info("Binning analysis results saved")

    def print_summary(self) -> None:
        """Print analysis summary to console."""
        if not self.results:
            logger.warning("No results to summarize")
            return

        logger.info("=" * 80)
        logger.info("BINNING ANALYSIS SUMMARY")
        logger.info("=" * 80)

        # Overall statistics
        if "overall_stats" in self.results:
            stats = self.results["overall_stats"]
            logger.info(f"Overall mass error: {stats['mean_delta_ppm']:.2f} +/- {stats['std_delta_ppm']:.2f} ppm")

        # Bin-jump analysis
        if self.results.get("bin_jump_analysis"):
            jump_analysis = self.results["bin_jump_analysis"]
            if "summary" in jump_analysis:
                summary = jump_analysis["summary"]
                logger.info(f"Best strategy (bin-jump): {summary['best_strategy']}")
                logger.info(f"Jump rate: {summary['best_mean_jump_rate'] * 100:.2f}%")

        # Intra-spectrum collision analysis
        if self.results.get("intra_spectrum_collision"):
            intra = self.results["intra_spectrum_collision"]
            best = intra["best_strategy"]
            best_cr = intra["per_strategy"][best]["mean_collision_rate"]
            logger.info(f"Best strategy (intra-collision): {best} ({best_cr * 100:.2f}%)")
            worst = intra["worst_strategy"]
            worst_cr = intra["per_strategy"][worst]["mean_collision_rate"]
            logger.info(f"Worst strategy (intra-collision): {worst} ({worst_cr * 100:.2f}%)")

        # Resolution-noise tradeoff
        if self.results.get("resolution_noise_tradeoff"):
            tradeoff = self.results["resolution_noise_tradeoff"]
            pareto = tradeoff.get("pareto_optimal_strategies", [])
            if pareto:
                logger.info(f"Pareto-optimal strategies: {', '.join(pareto)}")

        # Resolution information
        res_info = self.results.get("resolution_information")
        if res_info and "summary" in res_info:
            summary = res_info["summary"]
            logger.info("Resolution information:")
            mi = summary["most_informative_strategy"]
            mi_h = summary["most_informative_entropy"]
            mi_eff = summary["most_informative_efficiency"]
            mi_ec = summary["most_informative_effective_classes"]
            logger.info(f"  Most informative: {mi} (H={mi_h:.1f} bits, efficiency={mi_eff * 100:.1f}%, {mi_ec:,.0f} effective classes)")
            me = summary["most_efficient_strategy"]
            me_h = summary["most_efficient_entropy"]
            me_eff = summary["most_efficient_efficiency"]
            logger.info(f"  Most efficient: {me} (H={me_h:.1f} bits, efficiency={me_eff * 100:.1f}%)")

            # Report resolvability issues
            phys = res_info.get("physical_resolvability", {})
            if phys and "per_strategy" in phys:
                issues: list[Any] = []
                for diff_name in phys["mass_differences"]:
                    failing = [s for s, diffs in phys["per_strategy"].items() if diffs[diff_name]["classification"] == "unresolvable"]
                    if failing:
                        issues.append(f"{diff_name} unresolvable by {', '.join(failing)}")
                if issues:
                    logger.info(f"  Resolvability issues: {'; '.join(issues)}")

            # Group size sensitivity
            gs_sens = res_info.get("group_size_sensitivity")
            if gs_sens and "recommendations" in gs_sens:
                logger.info("  Group size sensitivity:")
                for strat_name in sorted(gs_sens["recommendations"]):
                    rec = gs_sens["recommendations"][strat_name]
                    strat_data = gs_sens["per_strategy"][strat_name]
                    if rec:
                        # Find group_size with head_balance closest to 0.5
                        best_gs = min(
                            rec,
                            key=lambda g: abs(strat_data[g]["head_balance"] - 0.5),
                        )
                        bal = strat_data[best_gs]["head_balance"]
                        logger.info(
                            f"    {strat_name}: recommended {rec} "
                            f"(best balance at gs={best_gs}: "
                            f"{bal * 100:.0f}% group / {(1 - bal) * 100:.0f}% offset)"
                        )
                    else:
                        logger.info(f"    {strat_name}: NO recommended group_size (offset saturated at all candidates)")

        # Error model fit
        error_model = self.results.get("error_model_fit")
        if error_model and error_model.get("fitted_da_floor") is not None:
            logger.info(
                f"Error model fit (Orbitrap): da_floor={error_model['fitted_da_floor']:.6f} Da, "
                f"ppm_equiv={error_model['fitted_ppm_equiv']:.2f} PPM, "
                f"R^2={error_model['r_squared']:.4f}"
            )

        # CID analysis
        cid = self.results.get("cid_analysis")
        if cid and "composition" in cid:
            comp_str = ", ".join(f"{ft}: {d['n_spectra']} spectra/{d['n_peaks']} peaks" for ft, d in sorted(cid["composition"].items()))
            logger.info(f"CID composition: {comp_str} (strict CID: {cid.get('n_cid_spectra', '?')} spectra, {cid.get('n_cid_peaks', '?'):,d} peaks)")
            obs = cid.get("observed_cid_errors")
            if obs:
                logger.info(
                    f"Observed CID errors ({obs['n_matched_peaks']} matched): "
                    f"P50={obs['p50_da']:.4f}, P95={obs['p95_da']:.4f}, "
                    f"max={obs['max_da']:.4f} Da"
                )
            breaking = cid.get("breaking_points", {})
            if breaking:
                bp_strs: list[Any] = []
                for sn in sorted(breaking.keys()):
                    bp = breaking[sn]["error_da_at_50pct_mismatch"]
                    bp_strs.append(f"{sn}={bp:.3f}" if bp else f"{sn}=>0.5")
                logger.info(f"Breaking points (50% mismatch): {', '.join(bp_strs)}")
            gap = cid.get("cid_vs_orbitrap_gap", {})
            if gap:
                dominant = max(gap.items(), key=lambda x: x[1].get("gap_ratio", 0))
                logger.info(
                    f"CID vs Orbitrap gap (worst): {dominant[0]} "
                    f"(ratio={dominant[1]['gap_ratio']:.1f}x, "
                    f"CID={dominant[1]['cid_mismatch'] * 100:.1f}%, "
                    f"Orbitrap={dominant[1]['orbitrap_mismatch'] * 100:.1f}%)"
                )

        # Effective information rate
        tradeoff = self.results.get("resolution_noise_tradeoff")
        if tradeoff and "tradeoff_table" in tradeoff:
            table = tradeoff["tradeoff_table"]
            info_rates = [(r["strategy"], r.get("effective_info_rate_bits", 0)) for r in table if r.get("effective_info_rate_bits", 0) > 0]
            if info_rates:
                info_rates.sort(key=lambda x: x[1], reverse=True)
                top3 = info_rates[:3]
                top_str = ", ".join(f"{n} ({r:.2f} bits)" for n, r in top3)
                logger.info(f"Top strategies by effective info rate: {top_str}")

        logger.info("=" * 80)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_binning_strategies_from_config(self) -> Dict[str, Dict]:
        """Get binning strategies from analysis config.

        Returns:
            Dictionary mapping strategy names to parameter dictionaries
        """
        if self.binning_strategies:
            # Handle both OmegaConf and plain dict (from task_configs resolution)
            if hasattr(self.binning_strategies, "_metadata"):
                return OmegaConf.to_container(self.binning_strategies, resolve=True)  # type: ignore[no-any-return]
            return dict(self.binning_strategies)
        else:
            return {
                "fixed_da_0.05": {"type": "fixed_da", "bin_width": 0.05},
                "fixed_da_0.02": {"type": "fixed_da", "bin_width": 0.02},
                "fixed_da_0.01": {"type": "fixed_da", "bin_width": 0.01},
                "fixed_ppm_10": {"type": "fixed_ppm", "ppm": 10},
                "fixed_ppm_20": {"type": "fixed_ppm", "ppm": 20},
                "fixed_ppm_50": {"type": "fixed_ppm", "ppm": 50},
                "adaptive_fine": {
                    "type": "adaptive",
                    "function": "hyperbolic",
                    "da_floor": 0.02,
                    "ppm_asymptote": 15.0,
                    "min_da": 0.005,
                    "max_da": 0.12,
                },
                "adaptive_coarse": {
                    "type": "adaptive",
                    "function": "hyperbolic",
                    "da_floor": 0.025,
                    "ppm_asymptote": 50.0,
                    "min_da": 0.005,
                    "max_da": 0.15,
                },
            }

    def _build_ion_id(self, df: pd.DataFrame) -> pd.Series:
        """Build unique ion identifiers from DataFrame columns.

        Format: PEPTIDE_b10+1_base@105123
          - peptide: the peptide sequence
          - ion_type + position: e.g., b10, y5
          - charge: e.g., +1, +2
          - feature_type: base, isotope, or loss
          - theo_mz_bin: theoretical m/z rounded to 0.01 Da (as integer centidaltions)

        The 0.01 Da resolution is necessary to distinguish isotopic peaks of
        multiply-charged fragments. E.g., charge-2 isotope spacing is ~0.5 Da,
        so 1 Da rounding would alias M+1 and M+2 isotopes into the same ion_id,
        creating phantom "distant" bin jumps.

        Args:
            df: DataFrame with columns [peptide, ion_type, position, charge, theo_mz]
                and optionally [feature_type]

        Returns:
            Series of ion_id strings
        """
        theo_mz_bin = (df["theo_mz"] * 100).round().astype(int)

        if "feature_type" in df.columns:
            feature_types = df["feature_type"].fillna("base").astype(str)
        else:
            feature_types = "base"

        return (
            df["peptide"].astype(str)
            + "_"
            + df["ion_type"].astype(str)
            + df["position"].astype(str)
            + "+"
            + df["charge"].astype(str)
            + "_"
            + feature_types
            + "@"
            + theo_mz_bin.astype(str)
        )

    def _compute_jump_rates(
        self,
        multi_obs_df: pd.DataFrame,
        strategies: Dict[str, Dict],
        ion_total_counts: pd.Series,
    ) -> Dict[str, Dict[str, Any]]:
        """Compute jump rates per strategy for multi-observation ions.

        Core logic shared between Part B and stratified analysis.

        Args:
            multi_obs_df: DataFrame with pre-calculated bin columns (bin_{strategy_name})
                         and ion_id column
            strategies: Dict of strategy names (must match bin column names)
            ion_total_counts: Series mapping ion_id to total observation count

        Returns:
            Dict mapping strategy_name to jump rate results including raw Series
        """
        per_strategy: dict[str, Any] = {}

        for strategy_name in strategies:
            bin_col = f"bin_{strategy_name}"

            # Group by (ion_id, bin) and count occurrences
            bin_counts = multi_obs_df.groupby(["ion_id", bin_col], observed=True).size().reset_index(name="count")

            # Find modal bin count for each ion
            modal_counts = bin_counts.groupby("ion_id", observed=True)["count"].max()

            # Count unique bins per ion
            n_unique_bins = bin_counts.groupby("ion_id", observed=True).size()

            # Calculate jump rates: 1 - (modal_count / total_count)
            jump_rates = 1.0 - (modal_counts / ion_total_counts)

            per_strategy[strategy_name] = {
                "mean_jump_rate": float(np.mean(jump_rates)),
                "median_jump_rate": float(np.median(jump_rates)),
                "n_ions": len(jump_rates),
                "jump_rates_series": jump_rates,
                "n_unique_bins_series": n_unique_bins,
            }

        return per_strategy

    def _classify_mz_range(self, mz: float) -> str:
        """Classify m/z value into bin-aligned regions."""
        for range_name in self.mz_range_order:
            low, high = self.mz_range_boundaries[range_name]
            if low <= mz < high:
                return range_name
        return self.mz_range_order[-1]

    def _calculate_overall_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall mass error statistics."""
        ppm_values = df["delta_mz_ppm"].values
        percentiles_ppm: dict[str, Any] = {
            "p50": float(np.percentile(np.abs(ppm_values), 50)),
            "p90": float(np.percentile(np.abs(ppm_values), 90)),
            "p95": float(np.percentile(np.abs(ppm_values), 95)),
            "p99": float(np.percentile(np.abs(ppm_values), 99)),
        }

        return {
            "n_peaks": len(df),
            "mean_delta_da": float(df["delta_mz_da"].mean()),
            "std_delta_da": float(df["delta_mz_da"].std()),
            "mean_delta_ppm": float(df["delta_mz_ppm"].mean()),
            "std_delta_ppm": float(df["delta_mz_ppm"].std()),
            "percentiles_ppm": percentiles_ppm,
        }

    def _calculate_range_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Calculate mass error statistics stratified by m/z range."""
        range_stats: dict[str, Any] = {}

        for mz_range in self.mz_range_order:
            range_df = df[df["mz_range"] == mz_range]
            if len(range_df) > 0:
                mean_da = float(range_df["delta_mz_da"].mean())
                std_da = float(range_df["delta_mz_da"].std())
                mean_ppm = float(range_df["delta_mz_ppm"].mean())
                std_ppm = float(range_df["delta_mz_ppm"].std())

                range_stats[mz_range] = {
                    "n_peaks": len(range_df),
                    "mean_theo_mz": float(range_df["theo_mz"].mean()),
                    "mean_delta_da": mean_da,
                    "std_delta_da": std_da,
                    "mean_delta_ppm": mean_ppm,
                    "std_delta_ppm": std_ppm,
                    "cv_delta_da": float(std_da / mean_da) if mean_da > 0 else 0.0,
                    "cv_delta_ppm": float(std_ppm / mean_ppm) if mean_ppm > 0 else 0.0,
                }

        return range_stats

    def _create_binning_strategy(self, params: Dict[str, Any]) -> Any:
        """Create a binning strategy object from parameters.

        Args:
            params: Binning strategy parameters

        Returns:
            BinningStrategy instance (FixedDaBinning, FixedPpmBinning, or AdaptiveBinning)
        """
        min_mz = self.min_mz
        max_mz = self.max_mz

        bin_group_size = params.get("bin_group_size")
        if bin_group_size is None:
            bin_group_size = self.config.model.get("mz_head", {}).get("bin_group_size", 100)

        if params["type"] == "fixed_da":
            bin_width = params.get("bin_width")
            if bin_width is None:
                raise ValueError("fixed_da strategy requires 'bin_width' parameter")
            return FixedDaBinning(min_mz, max_mz, bin_width, bin_group_size)

        elif params["type"] == "fixed_ppm":
            ppm_target = params.get("ppm_target") or params.get("ppm")
            if ppm_target is None:
                raise ValueError("fixed_ppm strategy requires 'ppm_target' or 'ppm' parameter")
            return FixedPpmBinning(min_mz, max_mz, ppm_target, bin_group_size)

        elif params["type"] == "adaptive":
            return AdaptiveBinning(
                min_mz=min_mz,
                max_mz=max_mz,
                function=params.get("function", "hyperbolic"),
                da_floor=params.get("da_floor", 0.01),
                ppm_asymptote=params.get("ppm_asymptote", 10.0),
                ppm_slope=params.get("ppm_slope", 10.0),
                scale=params.get("scale", 0.001),
                exponent=params.get("exponent", 0.5),
                min_da=params.get("min_da", 0.005),
                max_da=params.get("max_da", 0.12),
                bin_group_size=bin_group_size,
            )

        else:
            raise ValueError(f"Unknown binning type: {params['type']}")

    def _calculate_bins(self, mz_array: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Calculate bin indices for m/z array given binning strategy.

        Uses the same binning implementation as training to ensure consistency.

        Args:
            mz_array: Array of m/z values
            params: Binning strategy parameters

        Returns:
            Array of bin indices
        """
        strategy = self._create_binning_strategy(params)
        mz_tensor = torch.from_numpy(mz_array).float()
        bin_indices = strategy.mz_to_bin(mz_tensor)
        return bin_indices.cpu().numpy()

    # =========================================================================
    # Part A: Binning Simulation Methods
    # =========================================================================

    def _calculate_within_bin_residuals(self, mz_array: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate within-bin residual metrics for quantization error analysis.

        For each m/z value, compute how far it falls from the center of its assigned bin.
        Residuals are normalized by bin width, giving values in [-0.5, +0.5]:
        - 0.0 = exactly at bin center (optimal quantization)
        - +/- 0.5 = at bin edge (maximum quantization error)

        Args:
            mz_array: Array of m/z values
            params: Binning strategy parameters

        Returns:
            Dict with residual statistics
        """
        strategy = self._create_binning_strategy(params)
        mz_tensor = torch.from_numpy(mz_array).float()

        bin_indices = strategy.mz_to_bin(mz_tensor)
        bin_centers = strategy.bin_to_mz(bin_indices)

        bin_edges = strategy.bin_edges
        left_edges = bin_edges[bin_indices]
        right_edges = bin_edges[bin_indices + 1]
        bin_widths = right_edges - left_edges

        residuals = (mz_tensor - bin_centers) / bin_widths
        residuals_np = residuals.cpu().numpy()
        abs_residuals = np.abs(residuals_np)

        return {
            "mean_abs_residual": float(np.mean(abs_residuals)),
            "std_residual": float(np.std(residuals_np)),
            "max_abs_residual": float(np.max(abs_residuals)),
            "fraction_near_edge": float(np.mean(abs_residuals > 0.4)),
            "fraction_near_center": float(np.mean(abs_residuals < 0.1)),
            "percentiles": {
                "p50": float(np.percentile(abs_residuals, 50)),
                "p90": float(np.percentile(abs_residuals, 90)),
                "p95": float(np.percentile(abs_residuals, 95)),
                "p99": float(np.percentile(abs_residuals, 99)),
            },
        }

    def _simulate_binning_strategies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Simulate different binning strategies and measure bin error rates.

        Measures how often a peak's experimental m/z falls into a different bin
        than its theoretical m/z. This is NOT the same as bin-jump rate (Part B),
        which measures reproducibility across multiple observations.

        Args:
            df: DataFrame with per-peak error data (theo_mz, exp_mz)

        Returns:
            Dict mapping strategy names to simulation results
        """
        strategies = self._get_binning_strategies_from_config()

        results: dict[str, Any] = {}
        for strategy_name, params in strategies.items():
            theo_bins = self._calculate_bins(df["theo_mz"].values, params)
            exp_bins = self._calculate_bins(df["exp_mz"].values, params)

            bin_errors = int(np.sum(theo_bins != exp_bins))
            bin_error_rate = float(bin_errors / len(df))

            within_bin_residuals = self._calculate_within_bin_residuals(df["exp_mz"].values, params)

            results[strategy_name] = {
                "bin_errors": bin_errors,
                "bin_error_rate": bin_error_rate,
                "total_peaks": len(df),
                "params": params,
                "within_bin_residuals": within_bin_residuals,
            }

        return results

    # =========================================================================
    # Part B: Bin-Jump Analysis Methods
    # =========================================================================

    def _calculate_bin_jump_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate bin-jump rate statistics for fragment ions observed multiple times.

        Measures label noise as experienced by the classification head: when the same
        physical fragment ion (peptide + ion_type + position + charge) falls into
        different bins across multiple observations.

        Args:
            df: DataFrame with columns [peptide, ion_type, position, charge, exp_mz, theo_mz]

        Returns:
            Dict with per_strategy results, summary statistics, and per_ion data
        """
        logger.info("Calculating bin-jump rate statistics...")

        # Keep only the columns needed for bin-jump analysis to reduce memory
        needed_cols = ["peptide", "ion_type", "position", "charge", "exp_mz", "theo_mz"]
        if "feature_type" in df.columns:
            needed_cols.append("feature_type")
        valid_df = df.loc[df["position"] >= 1, needed_cols].copy()

        if len(valid_df) == 0:
            logger.warning("No valid fragment ions with positions found")
            return {"error": "No valid fragment ions"}

        # Create ion identifier using shared helper
        valid_df["ion_id"] = self._build_ion_id(valid_df)

        # Filter to ions observed multiple times
        ion_counts = valid_df.groupby("ion_id").size()
        multi_obs_ion_ids = ion_counts[ion_counts >= self.min_ion_observations].index

        logger.info(f"Found {len(multi_obs_ion_ids):,d} ions observed >= {self.min_ion_observations} times")

        if len(multi_obs_ion_ids) == 0:
            logger.warning("No ions observed multiple times")
            return {"error": "No multi-observation ions"}

        # Select only columns needed downstream before copying to reduce
        # memory — the full mass_error_df may carry many extra object-dtype
        # columns whose consolidation caused OOM on large datasets.
        keep_cols = ["ion_id", "exp_mz", "theo_mz", "peptide", "ion_type", "position", "charge"]
        if "feature_type" in valid_df.columns:
            keep_cols.append("feature_type")
        multi_obs_df = valid_df.loc[valid_df["ion_id"].isin(multi_obs_ion_ids), keep_cols].copy()
        del valid_df  # free memory early

        strategies = self._get_binning_strategies_from_config()

        # Pre-calculate bins for all strategies (vectorized)
        logger.info("Pre-calculating bins for all strategies...")
        for strategy_name, params in strategies.items():
            bins = self._calculate_bins(multi_obs_df["exp_mz"].values, params)
            multi_obs_df[f"bin_{strategy_name}"] = bins

        # Calculate per-ion aggregates once
        logger.info("Calculating per-ion statistics...")
        ion_aggregates = multi_obs_df.groupby("ion_id", observed=True).agg(
            {
                "exp_mz": ["mean", "count"],
                "theo_mz": "mean",
                "peptide": "first",
                "ion_type": "first",
                "position": "first",
                "charge": "first",
            }
        )
        ion_aggregates.columns = ["_".join(col).strip("_") for col in ion_aggregates.columns]

        ion_total_counts = ion_aggregates["exp_mz_count"]

        n_total_valid_peaks = len(multi_obs_df) + int((df["position"] >= 1).sum() - len(multi_obs_df))

        # Use shared jump rate computation
        computed = self._compute_jump_rates(multi_obs_df, strategies, ion_total_counts)

        # Build per-strategy results without copying ion_aggregates.
        # Previously, ion_aggregates.copy() was called once per strategy
        # (7+ times), each producing a ~47 MB DataFrame for large datasets.
        # All copies were retained in strategy_combined_data but only the
        # best strategy was ever used later.  Instead, read directly from
        # ion_aggregates and the per-strategy Series.
        per_strategy_results: dict[str, Any] = {}
        # Store only the lightweight Series per strategy for later use
        strategy_jump_data: Dict[str, Dict[str, pd.Series]] = {}

        # Pre-extract arrays that are constant across strategies
        _exp_mz_count = ion_aggregates["exp_mz_count"].values
        _exp_mz_mean = ion_aggregates["exp_mz_mean"].values
        _theo_mz_mean = ion_aggregates["theo_mz_mean"].values
        _ion_ids = ion_aggregates.index

        for strategy_name, comp in computed.items():
            jump_rates_series = comp["jump_rates_series"]
            n_unique_bins_series = comp["n_unique_bins_series"]

            strategy_jump_data[strategy_name] = {
                "jump_rates_series": jump_rates_series,
                "n_unique_bins_series": n_unique_bins_series,
            }

            # Vectorized per-ion statistics — no DataFrame copy needed
            jump_rates_arr = jump_rates_series.reindex(_ion_ids).values
            n_unique_arr = n_unique_bins_series.reindex(_ion_ids).values

            ion_jump_stats = pd.DataFrame(
                {
                    "ion_id": _ion_ids,
                    "jump_rate": jump_rates_arr,
                    "n_obs": _exp_mz_count.astype(int),
                    "n_unique_bins": n_unique_arr.astype(int),
                    "mean_exp_mz": _exp_mz_mean,
                    "mean_theo_mz": _theo_mz_mean,
                }
            ).to_dict("records")

            per_strategy_results[strategy_name] = {
                "mean_jump_rate": comp["mean_jump_rate"],
                "median_jump_rate": comp["median_jump_rate"],
                "std_jump_rate": float(np.std(jump_rates_arr)),
                "n_ions": comp["n_ions"],
                "percent_below_5pct": float(np.sum(jump_rates_arr < 0.05) / len(jump_rates_arr) * 100),
                "percent_below_10pct": float(np.sum(jump_rates_arr < 0.10) / len(jump_rates_arr) * 100),
                "per_ion_data": ion_jump_stats,
            }

        # Summary: best strategy
        strategy_rankings = sorted(per_strategy_results.items(), key=lambda x: x[1]["mean_jump_rate"])

        best_strategy_name = strategy_rankings[0][0]

        summary: dict[str, Any] = {
            "best_strategy": best_strategy_name,
            "best_mean_jump_rate": strategy_rankings[0][1]["mean_jump_rate"],
            "n_multi_obs_ions": len(multi_obs_ion_ids),
            "n_total_valid_peaks": n_total_valid_peaks,
            "strategy_rankings": [(s, r["mean_jump_rate"]) for s, r in strategy_rankings],
        }

        # Jump distance analysis for all strategies
        jump_distance_results: dict[str, Any] = {}
        for strategy_name, params in strategies.items():
            jump_dist = self._analyze_jump_distances(multi_obs_df, strategy_name, params)
            jump_distance_results[strategy_name] = jump_dist

        # Export per-ion records for the best strategy (vectorized).
        # Only build the combined view for the single best strategy.
        best_jump = strategy_jump_data[best_strategy_name]
        best_jump_rates = best_jump["jump_rates_series"].reindex(_ion_ids).values

        ion_ids = pd.Series(_ion_ids, index=_ion_ids)
        # Parse ion_id format: PEPTIDE_b10+1_base@1051
        at_parts = ion_ids.str.rsplit("@", n=1)
        prefixes = at_parts.str[0]
        prefix_parts = prefixes.str.rsplit("_", n=1)
        feature_types = prefix_parts.str[1].fillna("base")
        prefix2s = prefix_parts.str[0]
        prefix2_parts = prefix2s.str.rsplit("_", n=1)
        fragments = prefix2_parts.str[1].fillna("")
        peptides = prefix2_parts.str[0]

        per_ion_df = pd.DataFrame(
            {
                "ion_id": ion_ids.values,
                "peptide": peptides.values,
                "fragment": fragments.values,
                "feature_type": feature_types.values,
                "n_observations": _exp_mz_count.astype(int),
                "jump_rate": best_jump_rates,
                "mean_exp_mz": _exp_mz_mean,
                "mean_theo_mz": _theo_mz_mean,
                "strategy": best_strategy_name,
            }
        )

        logger.info(f"Bin-jump rate analysis complete. Best strategy: {summary['best_strategy']}")

        return {
            "per_strategy": per_strategy_results,
            "summary": summary,
            "per_ion_data": per_ion_df,
            "jump_distance_analysis": jump_distance_results,
        }

    def _analyze_jump_distances(self, multi_obs_df: pd.DataFrame, strategy_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how far ions jump when they jump bins.

        Classifies jumps as adjacent (1 bin), near (2 bins), or distant (>2 bins).

        Args:
            multi_obs_df: DataFrame with multi-observation ions
            strategy_name: Name of binning strategy
            params: Binning strategy parameters

        Returns:
            Dict with jump distance statistics
        """
        if multi_obs_df is None or len(multi_obs_df) == 0:
            return {"no_data": True}

        bin_col = f"bin_{strategy_name}"

        if bin_col not in multi_obs_df.columns:
            bins = self._calculate_bins(multi_obs_df["exp_mz"].values, params)
            multi_obs_df = multi_obs_df.copy()
            multi_obs_df[bin_col] = bins

        if "ion_id" not in multi_obs_df.columns:
            return {"no_data": True, "reason": "No ion_id column"}

        grouped = multi_obs_df.groupby("ion_id")[bin_col]
        modal_bins = grouped.agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])

        unique_bins_per_ion = grouped.nunique()
        jumping_ions = unique_bins_per_ion[unique_bins_per_ion > 1].index

        if len(jumping_ions) == 0:
            return {"no_jumps": True, "fraction_adjacent": 0.0, "fraction_within_2": 0.0, "fraction_distant": 0.0, "n_jumps": 0}

        jump_distances: list[Any] = []
        for ion_id in jumping_ions:
            ion_bins = grouped.get_group(ion_id).values
            modal_bin = modal_bins[ion_id]
            distances = np.abs(ion_bins - modal_bin)
            jump_distances.extend(distances[distances > 0])

        if not jump_distances:
            return {"no_jumps": True, "fraction_adjacent": 0.0, "fraction_within_2": 0.0, "fraction_distant": 0.0, "n_jumps": 0}

        jump_distances = np.array(jump_distances)

        return {
            "mean_jump_distance": float(np.mean(jump_distances)),
            "median_jump_distance": float(np.median(jump_distances)),
            "fraction_adjacent": float(np.mean(jump_distances == 1)),
            "fraction_within_2": float(np.mean(jump_distances <= 2)),
            "fraction_distant": float(np.mean(jump_distances > 2)),
            "n_jumps": len(jump_distances),
        }

    # =========================================================================
    # Part B2: Intra-Spectrum Collision Analysis
    # =========================================================================

    def _calculate_intra_spectrum_collision_statistics(
        self,
        per_spectrum_mz: List[np.ndarray],
        per_spectrum_frag_type: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Calculate intra-spectrum peak collision statistics for each binning strategy.

        When multiple peaks within a single spectrum map to the same bin,
        information is destroyed:
        - If one colliding peak is masked and one visible: task leakage
        - If both are masked: model predicts the same bin twice

        This metric penalizes coarse binning, complementing jump rate which
        penalizes fine binning.

        Args:
            per_spectrum_mz: List of numpy arrays, each containing all valid
                            m/z values for one spectrum
            per_spectrum_frag_type: Optional list of fragmentation types per spectrum

        Returns:
            Dict with per-strategy collision statistics and summary
        """
        strategies = self._get_binning_strategies_from_config()

        spectrum_lengths = np.array([len(a) for a in per_spectrum_mz], dtype=np.int64)
        n_spectra = len(per_spectrum_mz)

        if n_spectra == 0:
            return {"per_strategy": {}, "best_strategy": None, "worst_strategy": None, "strategy_rankings": []}

        all_mz_flat = np.concatenate(per_spectrum_mz)
        all_mz_tensor = torch.from_numpy(all_mz_flat).float()

        split_points = np.cumsum(spectrum_lengths[:-1])

        # Vectorized m/z range classification
        range_boundaries_upper = np.array([self.mz_range_boundaries[r][1] for r in self.mz_range_order], dtype=np.float64)
        range_boundaries_upper = np.where(np.isinf(range_boundaries_upper), 1e12, range_boundaries_upper)
        all_range_indices = np.searchsorted(range_boundaries_upper, all_mz_flat, side="left")
        all_range_indices = np.clip(all_range_indices, 0, len(self.mz_range_order) - 1)

        per_strategy_results = {}

        for strategy_name, params in strategies.items():
            strategy = self._create_binning_strategy(params)
            n_bins = strategy.n_bins

            all_bins_flat = strategy.mz_to_bin(all_mz_tensor).numpy()
            per_spectrum_bins = np.split(all_bins_flat, split_points)

            collision_rates = np.empty(n_spectra, dtype=np.float64)
            unique_bin_ratios = np.empty(n_spectra, dtype=np.float64)
            peaks_per_occupied_bin_arr = np.empty(n_spectra, dtype=np.float64)
            max_peaks_per_bin_arr = np.ones(n_spectra, dtype=np.int64)

            for i, bins in enumerate(per_spectrum_bins):
                n_peaks = len(bins)
                if n_peaks < 2:
                    collision_rates[i] = 0.0
                    unique_bin_ratios[i] = 1.0
                    peaks_per_occupied_bin_arr[i] = 1.0
                    max_peaks_per_bin_arr[i] = 1
                    continue

                unique_bins, counts = np.unique(bins, return_counts=True)
                n_unique = len(unique_bins)

                peaks_in_collision = int(np.sum(counts[counts > 1]))
                collision_rates[i] = peaks_in_collision / n_peaks
                unique_bin_ratios[i] = n_unique / n_peaks
                peaks_per_occupied_bin_arr[i] = counts.mean()
                max_peaks_per_bin_arr[i] = counts.max()

            collision_by_mz_range: dict[str, Any] = {}
            for range_idx, range_name in enumerate(self.mz_range_order):
                mask = all_range_indices == range_idx
                n_peaks_in_range = int(mask.sum())
                if n_peaks_in_range < 2:
                    collision_by_mz_range[range_name] = {
                        "n_peaks": n_peaks_in_range,
                        "collision_rate": 0.0,
                    }
                    continue
                bins_in_range = all_bins_flat[mask]
                _, cnts = np.unique(bins_in_range, return_counts=True)
                colliding = int(np.sum(cnts[cnts > 1]))
                collision_by_mz_range[range_name] = {
                    "n_peaks": n_peaks_in_range,
                    "collision_rate": float(colliding / n_peaks_in_range),
                }

            # Fragmentation-type stratified collision rates
            collision_by_frag_type: dict[str, Any] = {}
            if per_spectrum_frag_type and len(per_spectrum_frag_type) == n_spectra:
                frag_types_arr = np.array(per_spectrum_frag_type, dtype=object)
                unique_frag_types = np.unique(frag_types_arr)
                for ft in unique_frag_types:
                    ft_mask = frag_types_arr == ft
                    n_ft = int(ft_mask.sum())
                    if n_ft < 10:
                        continue
                    ft_collision_rates = collision_rates[ft_mask]
                    collision_by_frag_type[str(ft)] = {
                        "n_spectra": n_ft,
                        "mean_collision_rate": float(ft_collision_rates.mean()),
                        "median_collision_rate": float(np.median(ft_collision_rates)),
                        "p95_collision_rate": float(np.percentile(ft_collision_rates, 95)),
                    }

            per_strategy_results[strategy_name] = {
                "mean_collision_rate": float(collision_rates.mean()),
                "median_collision_rate": float(np.median(collision_rates)),
                "std_collision_rate": float(collision_rates.std()),
                "p95_collision_rate": float(np.percentile(collision_rates, 95)),
                "mean_unique_bin_ratio": float(unique_bin_ratios.mean()),
                "mean_peaks_per_occupied_bin": float(peaks_per_occupied_bin_arr.mean()),
                "max_peaks_per_bin": int(max_peaks_per_bin_arr.max()),
                "n_spectra": n_spectra,
                "n_bins": n_bins,
                "collision_by_mz_range": collision_by_mz_range,
                "collision_by_frag_type": collision_by_frag_type,
            }

        # Summary
        strategy_rankings = sorted(per_strategy_results.items(), key=lambda x: x[1]["mean_collision_rate"])
        best_strategy = strategy_rankings[0][0]
        worst_strategy = strategy_rankings[-1][0]

        logger.info(
            f"Intra-spectrum collision analysis complete. "
            f"Best: {best_strategy} ({per_strategy_results[best_strategy]['mean_collision_rate'] * 100:.2f}% collision rate), "
            f"Worst: {worst_strategy} ({per_strategy_results[worst_strategy]['mean_collision_rate'] * 100:.2f}%)"
        )

        return {
            "per_strategy": per_strategy_results,
            "best_strategy": best_strategy,
            "worst_strategy": worst_strategy,
            "strategy_rankings": [(s, r["mean_collision_rate"]) for s, r in strategy_rankings],
        }

    # =========================================================================
    # Resolution-Noise Tradeoff
    # =========================================================================

    def _compute_resolution_noise_tradeoff(
        self,
        bin_jump_analysis: Dict[str, Any],
        intra_spectrum_collision: Dict[str, Any],
        prediction_entropy: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Combine jump rate and intra-spectrum collision into a resolution-noise tradeoff.

        Jump rate penalizes fine-grained binning (more bins -> label noise).
        Collision rate penalizes coarse binning (fewer bins -> information loss).

        The combined score balances both:
            score = (1 - w) * jump_rate + w * collision_rate

        where w = self.intra_collision_weight.

        Args:
            bin_jump_analysis: Results from Part B bin-jump analysis
            intra_spectrum_collision: Results from Part B2 intra-collision analysis
            prediction_entropy: Optional prediction entropy per strategy from
                               resolution information analysis

        Returns:
            Dict with per-strategy tradeoff data and Pareto frontier
        """
        w = self.intra_collision_weight
        jump_per_strategy = bin_jump_analysis["per_strategy"]
        collision_per_strategy = intra_spectrum_collision["per_strategy"]

        # Only include strategies present in both analyses
        common_strategies = sorted(set(jump_per_strategy.keys()) & set(collision_per_strategy.keys()))

        if not common_strategies:
            logger.warning("No common strategies between jump rate and collision analyses")
            return {"error": "No common strategies"}

        tradeoff_table: list[Any] = []
        jump_rates: list[Any] = []
        collision_rates: list[Any] = []

        for name in common_strategies:
            jr = jump_per_strategy[name]["mean_jump_rate"]
            cr = collision_per_strategy[name]["mean_collision_rate"]
            n_bins = collision_per_strategy[name]["n_bins"]
            combined = (1 - w) * jr + w * cr

            # Effective information rate: entropy * (1 - jump_rate)
            h_bin = 0.0
            if prediction_entropy and name in prediction_entropy:
                h_bin = prediction_entropy[name]["total_entropy"]
            effective_info_rate = h_bin * (1.0 - jr)

            tradeoff_table.append(
                {
                    "strategy": name,
                    "jump_rate": jr,
                    "collision_rate": cr,
                    "combined_score": combined,
                    "n_bins": n_bins,
                    "h_bin_bits": h_bin,
                    "effective_info_rate_bits": effective_info_rate,
                }
            )
            jump_rates.append(jr)
            collision_rates.append(cr)

        # Identify Pareto-optimal strategies
        pareto_mask = self._identify_pareto_optimal(jump_rates, collision_rates)

        for i, row in enumerate(tradeoff_table):
            row["pareto_optimal"] = pareto_mask[i]

        pareto_strategies = [t["strategy"] for t in tradeoff_table if t["pareto_optimal"]]

        logger.info(f"Resolution-noise tradeoff complete. Pareto-optimal: {', '.join(pareto_strategies)} (w={w})")

        return {
            "tradeoff_table": tradeoff_table,
            "weight": w,
            "pareto_optimal_strategies": pareto_strategies,
        }

    # =========================================================================
    # Stratified Analysis (Per Fragmentation Type)
    # =========================================================================

    def _calculate_stratified_bin_analysis(self, df: pd.DataFrame, strategies: Dict[str, Any]) -> Dict[str, Any]:
        """Run jump rate analysis stratified by fragmentation type.

        Reveals whether different instrument types need different binning.

        Args:
            df: DataFrame with mass error data including frag_type column
            strategies: Dict of binning strategy configurations

        Returns:
            Dict with per-fragmentation-type jump rate results
        """
        logger.info("Running stratified analysis by fragmentation type...")

        if "frag_type" not in df.columns:
            logger.warning("No frag_type column in data, skipping stratified analysis")
            return {"error": "No frag_type column"}

        frag_types = df["frag_type"].dropna().unique()
        logger.info(f"Found fragmentation types: {list(frag_types)}")

        stratified_results: dict[str, Any] = {}

        for frag_type in frag_types:
            frag_df = df[df["frag_type"] == frag_type].copy()
            n_peaks = len(frag_df)

            if n_peaks < 100:
                logger.warning(f"Skipping {frag_type}: only {n_peaks} peaks")
                continue

            logger.info(f"Analyzing {frag_type}: {n_peaks:,d} peaks ({n_peaks / len(df) * 100:.1f}%)")

            jump_results = self._calculate_stratified_jump_rates(frag_df, strategies)

            stratified_results[str(frag_type)] = {
                "n_peaks": n_peaks,
                "fraction": n_peaks / len(df),
                "jump_analysis": jump_results,
            }

        return {
            "per_frag_type": stratified_results,
            "n_frag_types": len(stratified_results),
        }

    def _calculate_stratified_jump_rates(self, df: pd.DataFrame, strategies: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate jump rates for a stratified subset of data.

        Uses shared _compute_jump_rates helper for deduplication.
        """
        needed_cols = ["peptide", "ion_type", "position", "charge", "exp_mz", "theo_mz"]
        if "feature_type" in df.columns:
            needed_cols.append("feature_type")
        valid_df = df.loc[df["position"] >= 1, needed_cols].copy()

        if len(valid_df) == 0:
            return {"error": "No valid fragment ions"}

        # Use shared ion ID builder
        valid_df["ion_id"] = self._build_ion_id(valid_df)

        ion_counts = valid_df.groupby("ion_id").size()
        multi_obs_ion_ids = ion_counts[ion_counts >= self.min_ion_observations].index

        if len(multi_obs_ion_ids) == 0:
            return {"error": "No multi-observation ions"}

        keep_cols = ["ion_id", "exp_mz", "theo_mz", "peptide", "ion_type", "position", "charge"]
        if "feature_type" in valid_df.columns:
            keep_cols.append("feature_type")
        multi_obs_df = valid_df.loc[valid_df["ion_id"].isin(multi_obs_ion_ids), keep_cols].copy()
        del valid_df

        # Pre-calculate bins
        for strategy_name, params in strategies.items():
            bins = self._calculate_bins(multi_obs_df["exp_mz"].values, params)
            multi_obs_df[f"bin_{strategy_name}"] = bins

        # Calculate per-ion total counts
        ion_total_counts = multi_obs_df.groupby("ion_id", observed=True)["exp_mz"].count()

        # Use shared helper
        computed = self._compute_jump_rates(multi_obs_df, strategies, ion_total_counts)

        per_strategy_results = {
            name: {
                "mean_jump_rate": comp["mean_jump_rate"],
                "median_jump_rate": comp["median_jump_rate"],
                "n_ions": comp["n_ions"],
            }
            for name, comp in computed.items()
        }

        best_strategy = min(per_strategy_results.keys(), key=lambda k: per_strategy_results[k]["mean_jump_rate"])

        return {
            "per_strategy": per_strategy_results,
            "best_strategy": best_strategy,
            "best_jump_rate": per_strategy_results[best_strategy]["mean_jump_rate"],
            "n_multi_obs_ions": len(multi_obs_ion_ids),
        }

    # =========================================================================
    # Pareto Helper
    # =========================================================================

    def _identify_pareto_optimal(self, jump_rates: List[float], collision_rates: List[float]) -> List[bool]:
        """Identify Pareto-optimal strategies (lower is better for both metrics).

        Args:
            jump_rates: List of jump rates
            collision_rates: List of collision rates

        Returns:
            List of booleans indicating Pareto-optimal status
        """
        n = len(jump_rates)
        pareto_mask = [True] * n

        for i in range(n):
            for j in range(n):
                if i != j:
                    # j dominates i if j is <= in both and < in at least one
                    if (
                        jump_rates[j] <= jump_rates[i]
                        and collision_rates[j] <= collision_rates[i]
                        and (jump_rates[j] < jump_rates[i] or collision_rates[j] < collision_rates[i])
                    ):
                        pareto_mask[i] = False
                        break

        return pareto_mask

    # =========================================================================
    # Visualization Methods
    # =========================================================================

    def _generate_strategy_comparison_visualization(self) -> None:
        """Generate Figure 1: strategy_comparison.png (2x2).

        Panels:
        - (0,0): Bin error rate -- horizontal bar chart, sorted
        - (0,1): Mean bin-jump rate -- horizontal bar chart
        - (1,0): Vocabulary size & efficiency -- grouped bar (n_bins vs effective classes)
        - (1,1): P90 rolling jump rate vs m/z -- line plot, all strategies overlaid
        """
        binning_sim = self.results.get("binning_simulation", {})
        bin_jump = self.results.get("bin_jump_analysis")

        if not binning_sim:
            logger.warning("No binning simulation results for strategy comparison")
            return

        logger.info("Generating Figure 1: Strategy comparison...")

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        # ---- (0,0): Bin error rate ----
        ax = axes[0, 0]
        strategies_sorted = sorted(binning_sim.items(), key=lambda x: x[1]["bin_error_rate"])
        names = [s[0] for s in strategies_sorted]
        rates = [s[1]["bin_error_rate"] * 100 for s in strategies_sorted]

        colors: list[Any] = []
        for rate in rates:
            if rate < 5:
                colors.append("lightgreen")
            elif rate < 10:
                colors.append("gold")
            else:
                colors.append("salmon")

        y_pos = np.arange(len(names))
        ax.barh(y_pos, rates, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Bin Error Rate (%)")
        ax.set_title("Theo-Exp Bin Mismatch Rate")
        ax.axvline(x=5, color="green", linestyle="--", alpha=0.7, linewidth=1.5, label="5%")
        ax.axvline(x=10, color="orange", linestyle="--", alpha=0.7, linewidth=1.5, label="10%")
        for i, rate in enumerate(rates):
            ax.text(rate + 0.3, i, f"{rate:.1f}%", va="center", fontsize=8)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3, axis="x")

        # ---- (0,1): Mean bin-jump rate ----
        ax = axes[0, 1]
        if bin_jump and "per_strategy" in bin_jump:
            per_strat = bin_jump["per_strategy"]
            jump_sorted = sorted(per_strat.items(), key=lambda x: x[1]["mean_jump_rate"])
            j_names = [s[0] for s in jump_sorted]
            j_means = [s[1]["mean_jump_rate"] * 100 for s in jump_sorted]
            [s[1]["std_jump_rate"] * 100 for s in jump_sorted]

            j_colors: list[Any] = []
            for rate in j_means:
                if rate < 5:
                    j_colors.append("lightgreen")
                elif rate < 10:
                    j_colors.append("gold")
                else:
                    j_colors.append("salmon")

            y_pos = np.arange(len(j_names))
            ax.barh(y_pos, j_means, color=j_colors, alpha=0.8, edgecolor="black", linewidth=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(j_names, fontsize=9)
            ax.set_xlabel("Mean Jump Rate (%)")
            ax.set_title("Bin-Jump Rate (Label Noise)")
            ax.axvline(x=5, color="green", linestyle="--", alpha=0.7, linewidth=1.5, label="5%")
            ax.axvline(x=10, color="orange", linestyle="--", alpha=0.7, linewidth=1.5, label="10%")
            for i, mean in enumerate(j_means):
                ax.text(mean + 0.3, i, f"{mean:.1f}%", va="center", fontsize=8)
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(True, alpha=0.3, axis="x")
        else:
            ax.text(0.5, 0.5, "Bin-jump analysis not available", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Bin-Jump Rate (Label Noise)")

        # ---- (1,0): Vocabulary size & efficiency ----
        ax = axes[1, 0]
        res_info = self.results.get("resolution_information", {})
        pred_entropy = res_info.get("prediction_entropy", {})
        tradeoff = self.results.get("resolution_noise_tradeoff") or {}
        tradeoff_table = tradeoff.get("tradeoff_table", [])

        # Build n_bins lookup from tradeoff table or prediction entropy
        vocab_data: dict[str, Any] = {}
        for row in tradeoff_table:
            vocab_data[row["strategy"]] = {"n_bins": row["n_bins"]}
        for strat_name, metrics in pred_entropy.items():
            if strat_name not in vocab_data:
                vocab_data[strat_name] = {}
            vocab_data[strat_name]["n_bins"] = metrics.get("n_bins", vocab_data.get(strat_name, {}).get("n_bins", 0))
            vocab_data[strat_name]["effective_classes"] = metrics.get("effective_classes", 0)
            vocab_data[strat_name]["occupancy_rate"] = metrics.get("occupancy_rate", 0)

        if vocab_data:
            # Sort by n_bins ascending
            sorted_strats = sorted(vocab_data.items(), key=lambda x: x[1].get("n_bins", 0))
            strat_names = [s[0] for s in sorted_strats]
            n_bins_vals = [s[1].get("n_bins", 0) for s in sorted_strats]
            eff_classes = [s[1].get("effective_classes", 0) for s in sorted_strats]
            occupancy = [s[1].get("occupancy_rate", 0) for s in sorted_strats]

            x_pos = np.arange(len(strat_names))
            width = 0.35

            ax.bar(x_pos - width / 2, [n / 1000 for n in n_bins_vals], width, label="Total bins (k)", color="steelblue", alpha=0.8)
            ax.bar(x_pos + width / 2, [e / 1000 for e in eff_classes], width, label="Effective classes (k)", color="seagreen", alpha=0.8)

            # Annotate occupancy rate on top of each pair
            for i, (nb, ec, occ) in enumerate(zip(n_bins_vals, eff_classes, occupancy, strict=False)):
                bar_top = max(nb, ec) / 1000
                ax.text(i, bar_top + 5, f"{occ:.0%}", ha="center", va="bottom", fontsize=7, fontweight="bold", color="#555")

            ax.set_xticks(x_pos)
            ax.set_xticklabels(strat_names, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Count (thousands)")
            ax.set_title("Vocabulary Size & Efficiency")
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(True, alpha=0.3, axis="y")
        else:
            ax.text(0.5, 0.5, "No vocabulary data available", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Vocabulary Size & Efficiency")

        # ---- (1,1): P90 rolling jump rate vs m/z ----
        ax = axes[1, 1]
        if bin_jump and "per_strategy" in bin_jump:
            per_strat = bin_jump["per_strategy"]
            colors_map = plt.cm.tab10(np.linspace(0, 1, len(per_strat)))

            for strategy_idx, (strategy_name, strategy_data) in enumerate(per_strat.items()):
                per_ion_data = strategy_data["per_ion_data"]
                if not per_ion_data:
                    continue

                df_ions = pd.DataFrame([{"mean_exp_mz": ion["mean_exp_mz"], "jump_rate": ion["jump_rate"]} for ion in per_ion_data])

                mz_min = df_ions["mean_exp_mz"].min()
                mz_max = df_ions["mean_exp_mz"].max()
                window_size = 100.0

                mz_bins = np.arange(
                    np.floor(mz_min / window_size) * window_size, np.ceil(mz_max / window_size) * window_size + window_size, window_size
                )

                if len(mz_bins) < 2:
                    continue

                df_ions["mz_window"] = pd.cut(df_ions["mean_exp_mz"], bins=mz_bins, labels=mz_bins[:-1], include_lowest=True)

                window_p90 = df_ions.groupby("mz_window", observed=True)["jump_rate"].quantile(0.90)
                if len(window_p90) == 0:
                    continue

                mz_centers = window_p90.index.astype(float).values + (window_size / 2.0)

                ax.plot(
                    mz_centers, window_p90.values, "o-", label=strategy_name, color=colors_map[strategy_idx], alpha=0.7, linewidth=2, markersize=4
                )

            ax.axhline(y=0.05, color="orange", linestyle="--", alpha=0.7, linewidth=1.5, label="5%")
            ax.axhline(y=0.10, color="red", linestyle="--", alpha=0.7, linewidth=1.5, label="10%")
            ax.set_xlabel("m/z (Da)")
            ax.set_ylabel("P90 Jump Rate")
            ax.set_title("P90 Rolling Jump Rate vs m/z (100 Da windows)")
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 0.45)
        else:
            ax.text(0.5, 0.5, "Bin-jump analysis not available", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("P90 Rolling Jump Rate vs m/z")

        fig.suptitle("Binning Strategy Comparison", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig_path = self.output_dir / "strategy_comparison.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Strategy comparison visualization saved to: {fig_path}")

    def _generate_strategy_shapes_visualization(self) -> None:
        """Generate Figure 2: strategy_shapes.png (4x1).

        Panels:
        - Bin width (Da) vs m/z across strategies
        - Effective resolution (PPM) vs m/z across strategies
        - Cumulative bin count vs m/z (where budget is spent)
        - Bin Width Safety Margin vs m/z (bin_width / 2*P95_error)
        """
        logger.info("Generating Figure 2: Strategy shapes...")

        strategies = self._get_binning_strategies_from_config()

        error_model = self.results.get("error_model_fit")
        has_error_model = (
            error_model is not None and error_model.get("fitted_da_floor") is not None and error_model.get("fitted_ppm_equiv") is not None
        )
        n_panels = 4 if has_error_model else 3

        fig, axes_list = plt.subplots(n_panels, 1, figsize=(18, 5 * n_panels + 1))
        if n_panels == 3:
            ax1, ax2, ax3 = axes_list
            ax4 = None
        else:
            ax1, ax2, ax3, ax4 = axes_list

        colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))

        for (strategy_name, params), color in zip(strategies.items(), colors, strict=False):
            try:
                strategy = self._create_binning_strategy(params)
                bin_edges = strategy.bin_edges.cpu().numpy()
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
                bin_widths = np.diff(bin_edges)

                # Panel 1: Bin width
                ax1.plot(bin_centers, bin_widths, "-", label=strategy_name, color=color, linewidth=2, alpha=0.8)

                if params["type"] == "adaptive":
                    min_da = params.get("min_da", 0.005)
                    max_da = params.get("max_da", 0.12)
                    ax1.axhline(y=min_da, color="gray", linestyle=":", alpha=0.4)
                    ax1.axhline(y=max_da, color="gray", linestyle=":", alpha=0.4)

                # Panel 2: Effective PPM
                effective_ppm = 1e6 * bin_widths / bin_centers
                ax2.plot(bin_centers, effective_ppm, "-", label=strategy_name, color=color, linewidth=2, alpha=0.8)

                # Panel 3: Cumulative bins
                cumulative_fraction = np.cumsum(np.ones_like(bin_centers)) / len(bin_centers)
                ax3.plot(bin_centers, cumulative_fraction, "-", label=strategy_name, color=color, linewidth=2, alpha=0.8)

            except Exception as e:
                logger.warning(f"Failed to process strategy {strategy_name}: {e}")
                continue

        ax1.set_xlabel("m/z (Da)")
        ax1.set_ylabel("Bin Width (Da)")
        ax1.set_title("Bin Width vs m/z")
        ax1.legend(fontsize=9, loc="best")
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("m/z (Da)")
        ax2.set_ylabel("Effective PPM Resolution")
        ax2.set_title("Effective Resolution (PPM) vs m/z")
        ax2.axhline(y=10, color="green", linestyle="--", alpha=0.5, linewidth=1, label="10 PPM")
        ax2.axhline(y=20, color="orange", linestyle="--", alpha=0.5, linewidth=1, label="20 PPM")
        ax2.axhline(y=50, color="red", linestyle="--", alpha=0.5, linewidth=1, label="50 PPM")
        ax2.legend(fontsize=9, loc="best")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, min(250, ax2.get_ylim()[1]))

        ax3.set_xlabel("m/z (Da)")
        ax3.set_ylabel("Cumulative Fraction of Bins")
        ax3.set_title("Cumulative Bin Count")
        ax3.legend(fontsize=9, loc="best")
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.0)

        # ---- Panel 4: Safety Margin (only if error model is fitted) ----
        if ax4 is not None and has_error_model:
            fitted_a = error_model["fitted_da_floor"]  # type: ignore[index]
            fitted_b = error_model["fitted_ppm_equiv"] / 1e6  # type: ignore[index]  # convert back to raw

            for (strategy_name, params), color in zip(strategies.items(), colors, strict=False):
                try:
                    strategy = self._create_binning_strategy(params)
                    bin_edges = strategy.bin_edges.cpu().numpy()
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
                    bin_widths = np.diff(bin_edges)

                    p95_errors = np.sqrt(fitted_a**2 + (fitted_b * bin_centers) ** 2)
                    safety = bin_widths / (2 * p95_errors)

                    ax4.plot(bin_centers, safety, "-", label=strategy_name, color=color, linewidth=2, alpha=0.8)
                except Exception as e:
                    logger.warning(f"Failed to plot safety margin for {strategy_name}: {e}")

            ax4.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, linewidth=2, label="Margin = 1 (bin = 2x error)")
            ax4.axhline(y=2.0, color="orange", linestyle=":", alpha=0.5, linewidth=1.5, label="Margin = 2 (comfortable)")
            ax4.set_xlabel("m/z (Da)")
            ax4.set_ylabel("Safety Margin (bin_width / 2*P95)")
            ax4.set_title("Bin Width Safety Margin vs Fitted P95 Error (Orbitrap)")
            ax4.legend(fontsize=8, loc="best")
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, min(20, ax4.get_ylim()[1]))

        fig.suptitle("Binning Strategy Shapes", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig_path = self.output_dir / "strategy_shapes.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Strategy shapes visualization saved to: {fig_path}")

    def _generate_resolution_noise_tradeoff_visualization(self) -> None:
        """Generate Figure 3: resolution_noise_tradeoff.png (1x2).

        Panels:
        - (0): Pareto frontier scatter -- jump rate vs collision rate.
               Size proportional to n_bins. Color = Pareto-optimal or dominated.
        - (1): Jump distance distribution -- grouped bar per strategy.
               Fraction adjacent (1 bin), near (2 bins), distant (>2 bins).
        """
        tradeoff = self.results.get("resolution_noise_tradeoff")
        bin_jump = self.results.get("bin_jump_analysis")

        if not tradeoff or "tradeoff_table" not in tradeoff:
            return

        logger.info("Generating Figure 3: Resolution-noise tradeoff...")

        table = tradeoff["tradeoff_table"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # ---- (0): Pareto frontier scatter ----
        jump_rates = [r["jump_rate"] * 100 for r in table]
        collision_rates = [r["collision_rate"] * 100 for r in table]
        n_bins_list = [r["n_bins"] for r in table]
        names = [r["strategy"] for r in table]
        pareto = [r["pareto_optimal"] for r in table]

        # Normalize point sizes
        n_bins_arr = np.array(n_bins_list, dtype=float)
        if n_bins_arr.max() > n_bins_arr.min():
            sizes = 80 + 320 * (n_bins_arr - n_bins_arr.min()) / (n_bins_arr.max() - n_bins_arr.min())
        else:
            sizes = np.full_like(n_bins_arr, 200)

        for i in range(len(table)):
            color = "#2ecc71" if pareto[i] else "lightgray"
            edge_color = "#27ae60" if pareto[i] else "gray"
            ax1.scatter(
                jump_rates[i], collision_rates[i], s=sizes[i], c=color, edgecolors=edge_color, linewidths=1.5, zorder=3 if pareto[i] else 2, alpha=0.8
            )

        # Draw Pareto frontier
        pareto_points = [(jump_rates[i], collision_rates[i]) for i in range(len(table)) if pareto[i]]
        if len(pareto_points) >= 2:
            pareto_points.sort(key=lambda p: p[0])
            px, py = zip(*pareto_points, strict=False)
            ax1.plot(px, py, "--", color="#27ae60", linewidth=1.5, alpha=0.6, label="Pareto frontier")

        # Annotate
        for i in range(len(table)):
            label = f"{names[i]}\n({n_bins_list[i]:,d} bins)"
            offset = (8, 8) if i % 2 == 0 else (8, -12)
            ax1.annotate(
                label,
                (jump_rates[i], collision_rates[i]),
                xytext=offset,
                textcoords="offset points",
                fontsize=7,
                alpha=0.8,
                arrowprops={"arrowstyle": "-", "alpha": 0.3, "lw": 0.5},
            )

        ax1.axvline(x=5, color="red", linestyle=":", alpha=0.4, label="5% threshold")
        ax1.axhline(y=5, color="red", linestyle=":", alpha=0.4)

        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markeredgecolor="#27ae60", markersize=10, label="Pareto-optimal"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgray", markeredgecolor="gray", markersize=10, label="Dominated"),
            Line2D([0], [0], linestyle="--", color="#27ae60", label="Pareto frontier"),
            Line2D([0], [0], linestyle=":", color="red", alpha=0.4, label="5% threshold"),
        ]
        ax1.legend(handles=legend_elements, loc="upper right", fontsize=8)

        ax1.set_xlabel("Jump Rate (%) -- label noise")
        ax1.set_ylabel("Intra-Spectrum Collision Rate (%) -- information loss")
        ax1.set_title("Pareto Frontier: Jump Rate vs Collision Rate")
        ax1.grid(True, alpha=0.3)

        # ---- (1): Jump distance distribution ----
        jump_dist = bin_jump.get("jump_distance_analysis", {}) if bin_jump else {}

        if jump_dist:
            # Filter to strategies with actual data
            strat_names = [s for s in jump_dist if not jump_dist[s].get("no_data") and not jump_dist[s].get("no_jumps")]

            if strat_names:
                frac_adjacent = [jump_dist[s].get("fraction_adjacent", 0) * 100 for s in strat_names]
                frac_near = [(jump_dist[s].get("fraction_within_2", 0) - jump_dist[s].get("fraction_adjacent", 0)) * 100 for s in strat_names]
                frac_distant = [jump_dist[s].get("fraction_distant", 0) * 100 for s in strat_names]

                x_pos = np.arange(len(strat_names))
                width = 0.25

                ax2.bar(x_pos - width, frac_adjacent, width, label="Adjacent (1 bin)", color="steelblue", alpha=0.8)
                ax2.bar(x_pos, frac_near, width, label="Near (2 bins)", color="gold", alpha=0.8)
                ax2.bar(x_pos + width, frac_distant, width, label="Distant (>2 bins)", color="salmon", alpha=0.8)

                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(strat_names, rotation=45, ha="right", fontsize=8)
                ax2.set_ylabel("Fraction of Jumps (%)")
                ax2.set_title("Jump Distance Distribution")
                ax2.legend(fontsize=9)
                ax2.grid(True, alpha=0.3, axis="y")
            else:
                ax2.text(0.5, 0.5, "No jump distance data available", ha="center", va="center", transform=ax2.transAxes)
                ax2.set_title("Jump Distance Distribution")
        else:
            ax2.text(0.5, 0.5, "No jump distance data available", ha="center", va="center", transform=ax2.transAxes)
            ax2.set_title("Jump Distance Distribution")

        fig.suptitle("Resolution-Noise Tradeoff", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig_path = self.output_dir / "resolution_noise_tradeoff.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Resolution-noise tradeoff visualization saved to: {fig_path}")

    def _generate_group_offset_balance_visualization(self) -> None:
        """Generate Figure 4: group_offset_balance.png (1x2).

        Panels:
        - Rank-frequency curve for group labels
        - Offset usage distribution
        """
        if "raw_data" not in self.results:
            logger.warning("No raw data for group/offset balance")
            return

        logger.info("Generating Figure 4: Group/offset balance...")

        df = self.results["raw_data"]
        if len(df) == 0:
            return

        strategies = self._get_binning_strategies_from_config()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))

        for (strategy_name, params), color in zip(strategies.items(), colors, strict=False):
            try:
                strategy = self._create_binning_strategy(params)
                mz_tensor = torch.from_numpy(df["exp_mz"].values).float()
                group_indices, offset_indices = strategy.mz_to_bin_groups(mz_tensor)

                group_indices_np = group_indices.cpu().numpy()
                offset_indices_np = offset_indices.cpu().numpy()

                # Panel 1: Group frequency
                group_counts = pd.Series(group_indices_np).value_counts().sort_values(ascending=False)
                ranks = np.arange(1, len(group_counts) + 1)
                ax1.plot(ranks, group_counts.values, "o-", label=strategy_name, color=color, alpha=0.7, linewidth=2, markersize=4)

                # Panel 2: Offset distribution
                offset_counts = pd.Series(offset_indices_np).value_counts().sort_index()
                offset_pct = 100.0 * offset_counts / offset_counts.sum()
                ax2.bar(offset_pct.index, offset_pct.values, alpha=0.5, label=strategy_name, color=color, width=0.8)

            except Exception as e:
                logger.warning(f"Failed to process strategy {strategy_name}: {e}")
                continue

        ax1.set_yscale("log")
        ax1.set_xlabel("Group Rank")
        ax1.set_ylabel("Frequency (log scale)")
        ax1.set_title("Group Label Distribution (Rank-Frequency)")
        ax1.legend(fontsize=9, loc="best")
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Offset Index")
        ax2.set_ylabel("Frequency (%)")
        ax2.set_title("Offset Usage Distribution")
        ax2.legend(fontsize=9, loc="best")
        ax2.grid(True, alpha=0.3, axis="y")

        fig.suptitle("Hierarchical Classification Balance", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig_path = self.output_dir / "group_offset_balance.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Group/offset balance visualization saved to: {fig_path}")

    def _generate_stratified_jump_rates_visualization(self) -> None:
        """Generate Figure 5: stratified_jump_rates.png (grouped bar).

        Shows mean jump rate across ALL strategies per fragmentation type.
        Only generated if multiple fragmentation types exist in data.
        """
        stratified = self.results.get("stratified_analysis")
        if not stratified or "per_frag_type" not in stratified:
            return

        per_frag = stratified["per_frag_type"]
        if len(per_frag) < 2:
            return

        logger.info("Generating Figure 5: Stratified jump rates...")

        # Collect all strategies across frag types
        all_strategies = set()
        for frag_data in per_frag.values():
            jump_analysis = frag_data.get("jump_analysis", {})
            if "per_strategy" in jump_analysis:
                all_strategies.update(jump_analysis["per_strategy"].keys())

        all_strategies = sorted(all_strategies)
        frag_types = list(per_frag.keys())

        if not all_strategies:
            return

        fig, ax = plt.subplots(1, 1, figsize=(max(10, len(all_strategies) * 2), 8))

        x_pos = np.arange(len(frag_types))
        n_strats = len(all_strategies)
        bar_width = 0.8 / n_strats

        colors = plt.cm.tab10(np.linspace(0, 1, n_strats))

        for strat_idx, strategy_name in enumerate(all_strategies):
            rates: list[Any] = []
            for frag_type in frag_types:
                jump_analysis = per_frag[frag_type].get("jump_analysis", {})
                per_strategy = jump_analysis.get("per_strategy", {})
                if strategy_name in per_strategy:
                    rates.append(per_strategy[strategy_name]["mean_jump_rate"] * 100)
                else:
                    rates.append(0)

            offset = (strat_idx - n_strats / 2 + 0.5) * bar_width
            ax.bar(x_pos + offset, rates, bar_width, label=strategy_name, color=colors[strat_idx], alpha=0.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(frag_types, fontsize=10)
        ax.set_xlabel("Fragmentation Type")
        ax.set_ylabel("Mean Jump Rate (%)")
        ax.set_title("Jump Rate by Fragmentation Type and Strategy")
        ax.axhline(y=5, color="green", linestyle="--", alpha=0.7, linewidth=1.5, label="5%")
        ax.axhline(y=10, color="orange", linestyle="--", alpha=0.7, linewidth=1.5, label="10%")
        ax.legend(fontsize=8, loc="best", ncol=2)
        ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle("Stratified Jump Rate Analysis", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig_path = self.output_dir / "stratified_jump_rates.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Stratified jump rates visualization saved to: {fig_path}")

    # =========================================================================
    # Part C: Resolution Information Analysis
    # =========================================================================

    @staticmethod
    def _shannon_entropy_bits(counts: np.ndarray) -> float:
        """Compute Shannon entropy in bits from a count array.

        Args:
            counts: Array of non-negative integer counts.

        Returns:
            Entropy in bits. Returns 0.0 if all counts are zero.
        """
        counts = counts[counts > 0]
        if len(counts) == 0:
            return 0.0
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log2(probs)))

    def _calculate_resolution_information_analysis(
        self,
        mass_error_df: pd.DataFrame,
        per_spectrum_mz: Optional[List[np.ndarray]],
    ) -> Dict[str, Any]:
        """Orchestrate resolution information analysis.

        Computes prediction entropy, physical resolvability, and local entropy
        for all configured binning strategies.

        Args:
            mass_error_df: DataFrame with matched peak data (exp_mz column).
            per_spectrum_mz: Optional list of per-spectrum m/z arrays (all peaks).

        Returns:
            Dict with prediction_entropy, physical_resolvability, local_entropy,
            and summary sub-dicts.
        """
        strategies = self._get_binning_strategies_from_config()

        # Build flat m/z array — prefer all peaks (per_spectrum_mz) over matched only
        if per_spectrum_mz is not None and len(per_spectrum_mz) > 0:
            all_mz = np.concatenate(per_spectrum_mz)
            data_source = "per_spectrum_mz (all peaks)"
        else:
            all_mz = mass_error_df["exp_mz"].values
            data_source = "mass_error_df (matched peaks only)"

        logger.info(f"Resolution information: {len(all_mz):,d} peaks from {data_source}")

        prediction_entropy = self._compute_prediction_entropy(all_mz, strategies)
        physical_resolvability = self._compute_physical_resolvability(strategies)
        local_entropy = self._compute_local_entropy(all_mz, strategies)

        # Group size sensitivity (Part C2)
        group_size_sensitivity = None
        if self.enable_group_size_sensitivity:
            logger.info("Computing group size sensitivity analysis...")
            group_size_sensitivity = self._compute_group_size_sensitivity(all_mz, strategies)

        # Summary
        most_informative = max(
            prediction_entropy.items(),
            key=lambda x: x[1]["total_entropy"],
        )
        most_efficient = max(
            prediction_entropy.items(),
            key=lambda x: x[1]["efficiency"],
        )

        summary = {
            "data_source": data_source,
            "n_peaks_total": len(all_mz),
            "most_informative_strategy": most_informative[0],
            "most_informative_entropy": most_informative[1]["total_entropy"],
            "most_informative_efficiency": most_informative[1]["efficiency"],
            "most_informative_effective_classes": most_informative[1]["effective_classes"],
            "most_efficient_strategy": most_efficient[0],
            "most_efficient_entropy": most_efficient[1]["total_entropy"],
            "most_efficient_efficiency": most_efficient[1]["efficiency"],
        }

        return {
            "prediction_entropy": prediction_entropy,
            "physical_resolvability": physical_resolvability,
            "local_entropy": local_entropy,
            "group_size_sensitivity": group_size_sensitivity,
            "summary": summary,
        }

    def _compute_prediction_entropy(
        self,
        all_mz: np.ndarray,
        strategies: Dict[str, Dict],
    ) -> Dict[str, Dict[str, Any]]:
        """Compute prediction entropy metrics for each binning strategy.

        For each strategy computes:
        - H(bin): Shannon entropy of bin index distribution
        - H(group): Shannon entropy of group index distribution
        - H(offset|group): H(bin) - H(group) (chain rule)
        - Derived: efficiency, effective_classes, occupancy_rate

        Args:
            all_mz: Flat array of all m/z values.
            strategies: Dict mapping strategy names to parameter dicts.

        Returns:
            Dict mapping strategy_name to entropy metrics dict.
        """
        mz_tensor = torch.from_numpy(all_mz).float()
        results = {}

        for strategy_name, params in strategies.items():
            strategy = self._create_binning_strategy(params)
            bin_indices = strategy.mz_to_bin(mz_tensor).numpy()
            group_indices, _ = strategy.mz_to_bin_groups(mz_tensor)
            group_indices = group_indices.numpy()

            n_bins = strategy.n_bins
            n_groups = strategy.n_groups
            bin_group_size = strategy.bin_group_size

            # Bin-level entropy
            bin_counts = np.bincount(bin_indices, minlength=n_bins)
            h_bin = self._shannon_entropy_bits(bin_counts)
            max_h_bin = np.log2(n_bins) if n_bins > 1 else 0.0

            # Group-level entropy
            group_counts = np.bincount(group_indices, minlength=n_groups)
            h_group = self._shannon_entropy_bits(group_counts)

            # Offset entropy (chain rule: H(bin) = H(group) + H(offset|group))
            h_offset = h_bin - h_group
            max_h_offset = np.log2(bin_group_size) if bin_group_size > 1 else 0.0

            # Derived metrics
            efficiency = h_bin / max_h_bin if max_h_bin > 0 else 0.0
            effective_classes = 2.0**h_bin
            n_occupied = int(np.sum(bin_counts > 0))
            occupancy_rate = n_occupied / n_bins if n_bins > 0 else 0.0

            results[strategy_name] = {
                "total_entropy": h_bin,
                "group_entropy": h_group,
                "offset_entropy": h_offset,
                "max_entropy": max_h_bin,
                "max_offset_entropy": max_h_offset,
                "efficiency": efficiency,
                "effective_classes": effective_classes,
                "occupancy_rate": occupancy_rate,
                "n_bins": n_bins,
                "n_groups": n_groups,
                "bin_group_size": bin_group_size,
                "n_peaks": len(all_mz),
            }

        return results

    def _compute_physical_resolvability(
        self,
        strategies: Dict[str, Dict],
    ) -> Dict[str, Any]:
        """Compute analytical resolvability for key mass differences.

        For each (strategy, mass_diff) pair, computes min_bins = delta / max(bin_width)
        which is the worst-case number of bins separating two peaks differing by delta Da.

        Args:
            strategies: Dict mapping strategy names to parameter dicts.

        Returns:
            Dict with mass_differences and per_strategy resolvability matrices.
        """
        mass_differences: dict[str, Any] = {
            # Custom ion mass differences (low m/z region)
            "TMT N/C channel": 0.0063,  # TMT_127N vs TMT_127C (isobaric reporter)
            "immonium Arg vs TMT_129N": 0.018,  # Arg immonium vs TMT reporter overlap
            "Gln-Lys": 0.036,  # Also immonium_Gln vs immonium_Lys
            "glycan_126 vs TMT_126": 0.0727,  # Glycan fragment vs TMT reporter
            # Backbone fragment mass differences
            "Isotope (z=3)": 0.334,
            "Isotope (z=2)": 0.502,
            "H2O-NH3 loss": 0.984,
            "Isotope (z=1)": 1.003,
            "Gly-Ala": 14.016,
        }

        per_strategy = {}
        for strategy_name, params in strategies.items():
            strategy = self._create_binning_strategy(params)
            bin_edges = strategy.bin_edges.cpu().numpy()
            bin_widths = np.diff(bin_edges)
            max_bin_width = float(bin_widths.max())

            per_diff: dict[str, Any] = {}
            for diff_name, delta in mass_differences.items():
                min_bins = delta / max_bin_width if max_bin_width > 0 else float("inf")
                if min_bins >= 2.0:
                    classification = "resolvable"
                elif min_bins >= 1.0:
                    classification = "marginal"
                else:
                    classification = "unresolvable"

                per_diff[diff_name] = {
                    "min_bins": min_bins,
                    "classification": classification,
                }

            per_strategy[strategy_name] = per_diff

        return {
            "mass_differences": mass_differences,
            "per_strategy": per_strategy,
        }

    def _compute_local_entropy(
        self,
        all_mz: np.ndarray,
        strategies: Dict[str, Dict],
    ) -> Dict[str, Any]:
        """Compute per-window entropy profiles across the m/z range.

        Divides the m/z range into windows of size `resolution_info_window_size`,
        then computes bin-level entropy within each window for each strategy.

        Args:
            all_mz: Flat array of all m/z values.
            strategies: Dict mapping strategy names to parameter dicts.

        Returns:
            Dict with window_size, window_centers, and per-strategy entropy profiles.
        """
        window_size = self.resolution_info_window_size
        min_peaks = self.resolution_info_min_peaks_per_window

        mz_min = float(all_mz.min())
        mz_max = float(all_mz.max())

        # Create windows
        window_starts = np.arange(
            np.floor(mz_min / window_size) * window_size,
            np.ceil(mz_max / window_size) * window_size,
            window_size,
        )
        window_centers = window_starts + window_size / 2.0
        n_windows = len(window_starts)

        # Pre-sort peaks into windows using searchsorted
        sort_idx = np.argsort(all_mz)
        sorted_mz = all_mz[sort_idx]
        # Find split points for each window boundary
        window_edges = np.append(window_starts, window_starts[-1] + window_size)
        split_indices = np.searchsorted(sorted_mz, window_edges)

        # Pre-compute bin indices for all strategies
        mz_tensor = torch.from_numpy(all_mz).float()
        strategy_bin_indices: dict[str, Any] = {}
        for strategy_name, params in strategies.items():
            strategy = self._create_binning_strategy(params)
            bins = strategy.mz_to_bin(mz_tensor).numpy()
            # Reorder by sorted_mz order
            strategy_bin_indices[strategy_name] = bins[sort_idx]

        per_strategy = {}
        for strategy_name in strategies:
            bins_sorted = strategy_bin_indices[strategy_name]
            entropies: list[Any] = []
            peak_counts: list[Any] = []

            for w in range(n_windows):
                start_idx = split_indices[w]
                end_idx = split_indices[w + 1]
                n_peaks_in_window = end_idx - start_idx

                peak_counts.append(n_peaks_in_window)

                if n_peaks_in_window < min_peaks:
                    entropies.append(None)
                    continue

                window_bins = bins_sorted[start_idx:end_idx]
                counts = np.bincount(window_bins)
                entropies.append(self._shannon_entropy_bits(counts))

            per_strategy[strategy_name] = {
                "entropies": entropies,
                "peak_counts": peak_counts,
            }

        return {
            "window_size": window_size,
            "min_peaks_per_window": min_peaks,
            "window_centers": window_centers.tolist(),
            "per_strategy": per_strategy,
        }

    # =========================================================================
    # Part C2: Group Size Sensitivity Analysis
    # =========================================================================

    def _compute_group_size_sensitivity(
        self,
        all_mz: np.ndarray,
        strategies: Dict[str, Dict],
    ) -> Dict[str, Any]:
        """Compute how entropy splits between group/offset heads at different group sizes.

        H(bin) is constant for a given strategy — it depends only on how peaks
        distribute across bins. Changing group_size only changes how that fixed
        entropy is split between H(group) and H(offset|group). Everything is
        computed from bin_counts alone via efficient array reshaping.

        Args:
            all_mz: Flat array of all m/z values.
            strategies: Dict mapping strategy names to parameter dicts.

        Returns:
            Dict with group_size_candidates, thresholds, per_strategy metrics,
            and recommendations.
        """
        candidates = self.group_size_candidates
        max_offset_util = self.group_size_max_offset_utilization
        min_headroom = self.group_size_min_offset_headroom
        max_n_groups = self.group_size_max_n_groups

        mz_tensor = torch.from_numpy(all_mz).float()
        per_strategy = {}
        recommendations: dict[str, Any] = {}

        for strategy_name, params in strategies.items():
            strategy = self._create_binning_strategy(params)
            bin_indices = strategy.mz_to_bin(mz_tensor).numpy()
            n_bins = strategy.n_bins
            bin_counts = np.bincount(bin_indices, minlength=n_bins)
            h_bin = self._shannon_entropy_bits(bin_counts)
            bin_edges = strategy.bin_edges.cpu().numpy()

            strategy_results: dict[str, Any] = {}
            recommended_sizes: list[Any] = []

            for gs in candidates:
                n_groups = (n_bins + gs - 1) // gs

                # Efficient aggregation: pad + reshape + sum
                padded = np.zeros(n_groups * gs, dtype=bin_counts.dtype)
                padded[:n_bins] = bin_counts
                group_counts = padded.reshape(n_groups, gs).sum(axis=1)

                h_group = self._shannon_entropy_bits(group_counts)
                h_offset_given_group = h_bin - h_group

                max_h_offset = np.log2(gs) if gs > 1 else 0.0
                offset_utilization = h_offset_given_group / max_h_offset if max_h_offset > 0 else 0.0
                offset_headroom = max_h_offset - h_offset_given_group
                head_balance = h_group / h_bin if h_bin > 0 else 0.0

                # Physical group widths
                group_starts = np.arange(n_groups) * gs
                group_ends = np.minimum(group_starts + gs, n_bins)
                group_widths = bin_edges[group_ends] - bin_edges[group_starts]

                recommended = offset_utilization < max_offset_util and offset_headroom >= min_headroom and n_groups <= max_n_groups
                if recommended:
                    recommended_sizes.append(gs)

                strategy_results[gs] = {
                    "n_groups": n_groups,
                    "n_offset_classes": gs,
                    "h_bin": h_bin,
                    "h_group": h_group,
                    "h_offset_given_group": h_offset_given_group,
                    "max_h_offset": max_h_offset,
                    "offset_utilization": offset_utilization,
                    "offset_headroom": offset_headroom,
                    "head_balance": head_balance,
                    "group_width_da_mean": float(group_widths.mean()),
                    "group_width_da_min": float(group_widths.min()),
                    "group_width_da_max": float(group_widths.max()),
                    "recommended": recommended,
                }

            per_strategy[strategy_name] = strategy_results
            recommendations[strategy_name] = recommended_sizes

        return {
            "group_size_candidates": candidates,
            "thresholds": {
                "max_offset_utilization": max_offset_util,
                "min_offset_headroom": min_headroom,
                "max_n_groups": max_n_groups,
            },
            "per_strategy": per_strategy,
            "recommendations": recommendations,
        }

    # =========================================================================
    # Figure 6: Resolution Information Visualization
    # =========================================================================

    def _generate_resolution_information_visualization(self) -> None:
        """Generate Figure 6: resolution_information.png (2x2).

        Panels:
        - (0,0): Prediction entropy — horizontal stacked bar (group + offset)
        - (0,1): Physical resolvability — heatmap
        - (1,0): Local entropy vs m/z — line plot per strategy
        - (1,1): Vocabulary utilization — grouped bar (occupancy + efficiency)
        """
        res_info = self.results.get("resolution_information")
        if not res_info:
            return

        logger.info("Generating Figure 6: Resolution information...")

        pred_entropy = res_info["prediction_entropy"]
        phys_resolv = res_info["physical_resolvability"]
        local_ent = res_info["local_entropy"]

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        # ---- (0,0): Prediction Entropy — stacked horizontal bar ----
        ax = axes[0, 0]
        sorted_strategies = sorted(
            pred_entropy.items(),
            key=lambda x: x[1]["total_entropy"],
        )
        s_names = [s[0] for s in sorted_strategies]
        group_h = [s[1]["group_entropy"] for s in sorted_strategies]
        offset_h = [s[1]["offset_entropy"] for s in sorted_strategies]
        total_h = [s[1]["total_entropy"] for s in sorted_strategies]

        y_pos = np.arange(len(s_names))
        ax.barh(y_pos, group_h, color="steelblue", alpha=0.85, label="H(group)", edgecolor="black", linewidth=0.5)
        ax.barh(y_pos, offset_h, left=group_h, color="coral", alpha=0.85, label="H(offset|group)", edgecolor="black", linewidth=0.5)

        # Reference line: log2(bin_group_size)
        if sorted_strategies:
            ref_offset = sorted_strategies[0][1]["max_offset_entropy"]
            ax.axvline(x=ref_offset, color="gray", linestyle=":", alpha=0.6, label=f"max H(offset) = {ref_offset:.2f} bits")

        for i, h in enumerate(total_h):
            ax.text(h + 0.1, i, f"{h:.1f} bits", va="center", fontsize=8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(s_names, fontsize=9)
        ax.set_xlabel("Entropy (bits)")
        ax.set_title("Prediction Entropy (group + offset)")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3, axis="x")

        # ---- (0,1): Physical Resolvability — heatmap ----
        ax = axes[0, 1]
        mass_diffs = phys_resolv["mass_differences"]
        per_strat = phys_resolv["per_strategy"]
        diff_names = sorted(mass_diffs.keys(), key=lambda k: mass_diffs[k])
        strat_names = sorted(per_strat.keys())

        n_diffs = len(diff_names)
        n_strats = len(strat_names)

        heatmap_data = np.zeros((n_diffs, n_strats))
        annotations = np.empty((n_diffs, n_strats), dtype=object)
        cell_colors = np.empty((n_diffs, n_strats), dtype=object)

        for i, diff_name in enumerate(diff_names):
            for j, strat_name in enumerate(strat_names):
                entry = per_strat[strat_name][diff_name]
                mb = entry["min_bins"]
                heatmap_data[i, j] = mb
                annotations[i, j] = f"{mb:.1f}"
                if entry["classification"] == "resolvable":
                    cell_colors[i, j] = "#2ecc71"
                elif entry["classification"] == "marginal":
                    cell_colors[i, j] = "#f1c40f"
                else:
                    cell_colors[i, j] = "#e74c3c"

        # Draw colored cells manually
        for i in range(n_diffs):
            for j in range(n_strats):
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor=cell_colors[i, j],
                        alpha=0.6,
                        edgecolor="white",
                        linewidth=1,
                    )
                )
                ax.text(j, i, annotations[i, j], ha="center", va="center", fontsize=8, fontweight="bold")

        ax.set_xlim(-0.5, n_strats - 0.5)
        ax.set_ylim(-0.5, n_diffs - 0.5)
        ax.set_xticks(range(n_strats))
        ax.set_xticklabels(strat_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_diffs))
        diff_labels = [f"{d}\n({mass_diffs[d]:.3f} Da)" for d in diff_names]
        ax.set_yticklabels(diff_labels, fontsize=8)
        ax.set_title("Physical Resolvability (min bins)")
        ax.invert_yaxis()

        # Legend patches
        from matplotlib.patches import Patch

        legend_patches = [
            Patch(facecolor="#2ecc71", alpha=0.6, label="Resolvable (>=2)"),
            Patch(facecolor="#f1c40f", alpha=0.6, label="Marginal ([1,2))"),
            Patch(facecolor="#e74c3c", alpha=0.6, label="Unresolvable (<1)"),
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=7)

        # ---- (1,0): Local Entropy vs m/z — line plot ----
        ax = axes[1, 0]
        window_centers = local_ent["window_centers"]
        colors_map = plt.cm.tab10(np.linspace(0, 1, len(local_ent["per_strategy"])))

        for idx, (strategy_name, data) in enumerate(local_ent["per_strategy"].items()):
            entropies = data["entropies"]
            valid_x: list[Any] = []
            valid_y: list[Any] = []
            for i, e in enumerate(entropies):
                if e is not None:
                    valid_x.append(window_centers[i])
                    valid_y.append(e)
            if valid_x:
                ax.plot(valid_x, valid_y, "o-", label=strategy_name, color=colors_map[idx], alpha=0.7, linewidth=2, markersize=3)

        ax.set_xlabel("m/z (Da)")
        ax.set_ylabel("Entropy (bits)")
        ax.set_title(f"Local Entropy vs m/z ({local_ent['window_size']:.0f} Da windows)")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

        # ---- (1,1): Vocabulary Utilization — grouped bar ----
        ax = axes[1, 1]
        strat_names_sorted = sorted(pred_entropy.keys())
        occupancy = [pred_entropy[s]["occupancy_rate"] * 100 for s in strat_names_sorted]
        efficiency = [pred_entropy[s]["efficiency"] * 100 for s in strat_names_sorted]

        x_pos = np.arange(len(strat_names_sorted))
        width = 0.35

        ax.bar(x_pos - width / 2, occupancy, width, label="Occupancy Rate (%)", color="steelblue", alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.bar(x_pos + width / 2, efficiency, width, label="Efficiency (%)", color="coral", alpha=0.8, edgecolor="black", linewidth=0.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(strat_names_sorted, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Vocabulary Utilization")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 105)

        fig.suptitle("Resolution Information Analysis", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig_path = self.output_dir / "resolution_information.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Resolution information visualization saved to: {fig_path}")

    # =========================================================================
    # Figure 7: Group Size Sensitivity Visualization
    # =========================================================================

    def _generate_group_size_sensitivity_visualization(self) -> None:
        """Generate Figure 7: group_size_sensitivity.png (2x2).

        Panels:
        - (0,0): Entropy decomposition vs group size — line plot
        - (0,1): Offset utilization vs group size — line plot
        - (1,0): Head balance vs group size — line plot
        - (1,1): Recommendation heatmap — colored grid
        """
        gs_sens = self.results.get("resolution_information", {}).get("group_size_sensitivity")
        if not gs_sens:
            return

        logger.info("Generating Figure 7: Group size sensitivity...")

        candidates = gs_sens["group_size_candidates"]
        per_strategy = gs_sens["per_strategy"]
        thresholds = gs_sens["thresholds"]
        strat_names = sorted(per_strategy.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(strat_names)))

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        # ---- (0,0): Entropy Decomposition vs Group Size ----
        ax = axes[0, 0]
        for idx, strat_name in enumerate(strat_names):
            data = per_strategy[strat_name]
            h_group = [data[gs]["h_group"] for gs in candidates]
            h_offset = [data[gs]["h_offset_given_group"] for gs in candidates]
            h_bin = data[candidates[0]]["h_bin"]
            color = colors[idx]

            ax.plot(
                candidates,
                h_group,
                "o-",
                color=color,
                alpha=0.8,
                linewidth=2,
                markersize=5,
                label=f"{strat_name} H(group)",
            )
            ax.plot(
                candidates,
                h_offset,
                "s--",
                color=color,
                alpha=0.6,
                linewidth=1.5,
                markersize=4,
                label=f"{strat_name} H(offset|group)",
            )
            ax.axhline(
                y=h_bin,
                color=color,
                linestyle=":",
                alpha=0.3,
                linewidth=1,
            )

        ax.set_xscale("log")
        ax.set_xticks(candidates)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Group Size")
        ax.set_ylabel("Entropy (bits)")
        ax.set_title("Entropy Decomposition vs Group Size")
        ax.legend(fontsize=6, loc="best", ncol=2)
        ax.grid(True, alpha=0.3)

        # ---- (0,1): Offset Utilization vs Group Size ----
        ax = axes[0, 1]
        for idx, strat_name in enumerate(strat_names):
            data = per_strategy[strat_name]
            util_pct = [data[gs]["offset_utilization"] * 100 for gs in candidates]
            ax.plot(
                candidates,
                util_pct,
                "o-",
                color=colors[idx],
                alpha=0.8,
                linewidth=2,
                markersize=5,
                label=strat_name,
            )
        ax.axhline(
            y=thresholds["max_offset_utilization"] * 100,
            color="red",
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
            label=f"Threshold ({thresholds['max_offset_utilization'] * 100:.0f}%)",
        )
        ax.set_xscale("log")
        ax.set_xticks(candidates)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Group Size")
        ax.set_ylabel("Offset Utilization (%)")
        ax.set_title("Offset Utilization vs Group Size")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

        # ---- (1,0): Head Balance vs Group Size ----
        ax = axes[1, 0]
        for idx, strat_name in enumerate(strat_names):
            data = per_strategy[strat_name]
            balance_pct = [data[gs]["head_balance"] * 100 for gs in candidates]
            ax.plot(
                candidates,
                balance_pct,
                "o-",
                color=colors[idx],
                alpha=0.8,
                linewidth=2,
                markersize=5,
                label=strat_name,
            )
        ax.axhline(
            y=50,
            color="gray",
            linestyle="--",
            alpha=0.5,
            linewidth=1.5,
            label="50% (balanced)",
        )
        ax.set_xscale("log")
        ax.set_xticks(candidates)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Group Size")
        ax.set_ylabel("Head Balance (% info in group head)")
        ax.set_title("Head Balance vs Group Size")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

        # ---- (1,1): Recommendation Heatmap ----
        ax = axes[1, 1]
        n_strats = len(strat_names)
        n_candidates = len(candidates)

        for i, strat_name in enumerate(strat_names):
            data = per_strategy[strat_name]
            for j, gs in enumerate(candidates):
                metrics = data[gs]
                is_rec = metrics["recommended"]
                color = "#2ecc71" if is_rec else "#e74c3c"
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor=color,
                        alpha=0.6,
                        edgecolor="white",
                        linewidth=1,
                    )
                )
                ax.text(
                    j,
                    i,
                    f"{metrics['offset_utilization'] * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )

        ax.set_xlim(-0.5, n_candidates - 0.5)
        ax.set_ylim(-0.5, n_strats - 0.5)
        ax.set_xticks(range(n_candidates))
        ax.set_xticklabels([str(gs) for gs in candidates], fontsize=9)
        ax.set_xlabel("Group Size")
        ax.set_yticks(range(n_strats))
        ax.set_yticklabels(strat_names, fontsize=8)
        ax.set_title("Recommendation (cell = offset utilization %)")
        ax.invert_yaxis()

        from matplotlib.patches import Patch

        legend_patches = [
            Patch(facecolor="#2ecc71", alpha=0.6, label="Recommended"),
            Patch(facecolor="#e74c3c", alpha=0.6, label="Not recommended"),
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

        fig.suptitle(
            "Group Size Sensitivity Analysis",
            fontsize=16,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig_path = self.output_dir / "group_size_sensitivity.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Group size sensitivity visualization saved to: {fig_path}")

    # =========================================================================
    # Error Model Fit
    # =========================================================================

    def _fit_error_model(self, mass_error_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Fit hyperbolic error model P95_error(mz) = sqrt(a^2 + (b*mz)^2) to observed data.

        The matched peak data is Orbitrap-dominated (CID peaks are largely
        filtered out by the 10 PPM quality gate). The fitted model therefore
        represents Orbitrap error characteristics, not CID.

        Args:
            mass_error_df: DataFrame with theo_mz and delta_mz_da columns.

        Returns:
            Dict with fitted parameters, per-bin statistics, and safety margins,
            or None if fitting fails.
        """
        if len(mass_error_df) < 100:
            logger.warning("Too few peaks for error model fit")
            return None

        logger.info("Fitting data-driven error model...")

        theo_mz = mass_error_df["theo_mz"].values
        abs_error_da = np.abs(mass_error_df["delta_mz_da"].values)

        # Bin by theo_mz into ~20 bins
        mz_min, mz_max = float(theo_mz.min()), float(theo_mz.max())
        n_fit_bins = min(25, max(10, len(mass_error_df) // 500))
        bin_edges = np.linspace(mz_min, mz_max, n_fit_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        bin_indices = np.digitize(theo_mz, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_fit_bins - 1)

        p50_errors = np.zeros(n_fit_bins)
        p95_errors = np.zeros(n_fit_bins)
        p99_errors = np.zeros(n_fit_bins)
        bin_counts = np.zeros(n_fit_bins, dtype=int)

        for i in range(n_fit_bins):
            mask = bin_indices == i
            n_in_bin = int(mask.sum())
            bin_counts[i] = n_in_bin
            if n_in_bin < 5:
                p50_errors[i] = np.nan
                p95_errors[i] = np.nan
                p99_errors[i] = np.nan
            else:
                errors_in_bin = abs_error_da[mask]
                p50_errors[i] = float(np.percentile(errors_in_bin, 50))
                p95_errors[i] = float(np.percentile(errors_in_bin, 95))
                p99_errors[i] = float(np.percentile(errors_in_bin, 99))

        # Fit hyperbolic: P95(mz) = sqrt(a^2 + (b*mz)^2)
        valid_mask = ~np.isnan(p95_errors)
        valid_centers = bin_centers[valid_mask]
        valid_p95 = p95_errors[valid_mask]

        fitted_a = None
        fitted_b = None
        r_squared = None

        if len(valid_centers) >= 3:
            try:
                from scipy.optimize import curve_fit

                def hyperbolic_model(mz: Any, a: Any, b: Any) -> Any:
                    """Hyperbolic model."""
                    return np.sqrt(a**2 + (b * mz) ** 2)

                popt, _ = curve_fit(
                    hyperbolic_model,
                    valid_centers,
                    valid_p95,
                    p0=[0.002, 5e-6],
                    bounds=([0, 0], [1.0, 1e-3]),
                    maxfev=5000,
                )
                fitted_a, fitted_b = float(popt[0]), float(popt[1])

                # R^2
                predicted = hyperbolic_model(valid_centers, fitted_a, fitted_b)
                ss_res = np.sum((valid_p95 - predicted) ** 2)
                ss_tot = np.sum((valid_p95 - np.mean(valid_p95)) ** 2)
                r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

                logger.info(f"Error model fit: da_floor={fitted_a:.6f}, ppm_equiv={fitted_b * 1e6:.2f}, R^2={r_squared:.4f}")

            except Exception as e:
                logger.warning(f"Curve fit failed, falling back to grid search: {e}")
                # Grid search fallback
                best_loss = float("inf")
                for a_try in np.linspace(0.0005, 0.01, 20):
                    for b_try in np.linspace(1e-7, 5e-5, 20):
                        pred = np.sqrt(a_try**2 + (b_try * valid_centers) ** 2)
                        loss = np.sum((valid_p95 - pred) ** 2)
                        if loss < best_loss:
                            best_loss = loss
                            fitted_a, fitted_b = float(a_try), float(b_try)

                predicted = np.sqrt(fitted_a**2 + (fitted_b * valid_centers) ** 2)  # type: ignore[operator]
                ss_res = np.sum((valid_p95 - predicted) ** 2)
                ss_tot = np.sum((valid_p95 - np.mean(valid_p95)) ** 2)
                r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
                logger.info(f"Grid search fit: da_floor={fitted_a:.6f}, ppm_equiv={fitted_b * 1e6:.2f}, R^2={r_squared:.4f}")  # type: ignore[operator]

        # Safety margin: bin_width / (2 * P95_error) at reference m/z points
        strategies = self._get_binning_strategies_from_config()
        reference_mz = np.array([100, 200, 500, 800, 1000, 1500, 2000], dtype=np.float64)
        safety_margins: dict[str, Any] = {}

        for strategy_name, params in strategies.items():
            strategy = self._create_binning_strategy(params)
            bin_edges_np = strategy.bin_edges.cpu().numpy()
            bin_widths_np = np.diff(bin_edges_np)
            bin_centers_strat = (bin_edges_np[:-1] + bin_edges_np[1:]) / 2.0

            margins: dict[str, Any] = {}
            for ref_mz in reference_mz:
                # Find closest bin center
                idx = np.searchsorted(bin_centers_strat, ref_mz)
                idx = min(idx, len(bin_widths_np) - 1)
                bw = float(bin_widths_np[idx])
                if fitted_a is not None and fitted_b is not None:
                    p95_at_mz = np.sqrt(fitted_a**2 + (fitted_b * ref_mz) ** 2)
                    margin = bw / (2 * p95_at_mz) if p95_at_mz > 0 else float("inf")
                else:
                    p95_at_mz = 0.0
                    margin = float("inf")
                margins[float(ref_mz)] = {  # type: ignore[index]
                    "bin_width": bw,
                    "p95_error": float(p95_at_mz),
                    "safety_margin": float(margin),
                }
            safety_margins[strategy_name] = margins

        # Per-bin data for CSV export
        per_bin_data: list[Any] = []
        for i in range(n_fit_bins):
            row: dict[str, Any] = {
                "mz_bin_center": float(bin_centers[i]),
                "n_peaks": int(bin_counts[i]),
                "p50_error_da": float(p50_errors[i]) if not np.isnan(p50_errors[i]) else None,
                "p95_error_da": float(p95_errors[i]) if not np.isnan(p95_errors[i]) else None,
                "p99_error_da": float(p99_errors[i]) if not np.isnan(p99_errors[i]) else None,
            }
            if fitted_a is not None and fitted_b is not None:
                row["fitted_error_da"] = float(np.sqrt(fitted_a**2 + (fitted_b * bin_centers[i]) ** 2))
            per_bin_data.append(row)

        return {
            "fitted_da_floor": fitted_a,
            "fitted_ppm_equiv": fitted_b * 1e6 if fitted_b is not None else None,
            "r_squared": r_squared,
            "n_fit_bins": n_fit_bins,
            "per_bin_data": per_bin_data,
            "safety_margins": safety_margins,
            "note": "Fitted to Orbitrap-dominated matched peaks. CID peaks are underrepresented due to quality gate filtering at 10 PPM.",
        }

    # =========================================================================
    # CID Simulated Analysis
    # =========================================================================

    @staticmethod
    def _is_cid_frag_type(ft: str) -> bool:
        """Check if fragmentation type is CID (not HCID/EThcD etc).

        Uses word-boundary matching: "CID" must appear as a standalone token
        or at the start of the string, not as a suffix of another mode.
        """
        ft_upper = str(ft).upper().strip()
        # Exact match
        if ft_upper == "CID":
            return True
        # Starts with CID (e.g. "CID-IT")
        if ft_upper.startswith("CID") and (len(ft_upper) == 3 or not ft_upper[3].isalpha()):
            return True
        # Ion trap keywords
        if ft_upper in ("IT", "ION TRAP"):
            return True
        return False

    def _calculate_cid_simulated_analysis(
        self,
        per_spectrum_mz: List[np.ndarray],
        per_spectrum_frag_type: List[str],
        mass_error_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Simulate errors on real spectra to assess binning robustness.

        Runs a fine error sweep (0.005-0.5 Da) to show the full mismatch curve,
        revealing where each strategy transitions from robust to broken.
        Also extracts observed CID errors from matched peaks when available.

        Args:
            per_spectrum_mz: List of m/z arrays per spectrum.
            per_spectrum_frag_type: Fragmentation type per spectrum.
            mass_error_df: Optional DataFrame with theo_mz, delta_mz_da, frag_type
                          for extracting observed CID errors.

        Returns:
            Dict with composition, error sweep, observed errors, and Orbitrap comparison.
        """
        # 1. Partition by frag_type (exact match)
        frag_types_arr = np.array(per_spectrum_frag_type, dtype=object)
        unique_types = np.unique(frag_types_arr)

        composition: dict[str, Any] = {}
        for ft in unique_types:
            ft_mask = frag_types_arr == ft
            n_spectra = int(ft_mask.sum())
            n_peaks = sum(len(per_spectrum_mz[i]) for i in np.where(ft_mask)[0])
            composition[str(ft)] = {"n_spectra": n_spectra, "n_peaks": n_peaks}

        logger.info(f"CID analysis composition: {composition}")

        # 2. Extract CID m/z peaks (strict matching — excludes HCID, EThcD etc.)
        cid_mask = np.array([self._is_cid_frag_type(ft) for ft in per_spectrum_frag_type], dtype=bool)

        n_cid_spectra = int(cid_mask.sum())
        use_synthetic = False
        if n_cid_spectra == 0:
            logger.info("No CID spectra found. Using all peaks for simulation.")
            all_mz_flat = np.concatenate(per_spectrum_mz) if per_spectrum_mz else np.array([])
            use_synthetic = True
        else:
            cid_indices = np.where(cid_mask)[0]
            all_mz_flat = np.concatenate([per_spectrum_mz[i] for i in cid_indices])

        if len(all_mz_flat) == 0:
            return {"error": "No peaks available for CID analysis"}

        logger.info(
            f"CID peaks for simulation: {len(all_mz_flat):,d} from "
            f"{n_cid_spectra} CID spectra"
            f"{' (synthetic — using all peaks)' if use_synthetic else ''}"
        )

        # 3. Observed CID errors from mass_error_df
        observed_cid_errors = None
        if mass_error_df is not None and "frag_type" in mass_error_df.columns:
            cid_error_mask = mass_error_df["frag_type"].apply(self._is_cid_frag_type)
            n_cid_matched = int(cid_error_mask.sum())
            if n_cid_matched >= 10:
                cid_abs_da = np.abs(mass_error_df.loc[cid_error_mask, "delta_mz_da"].values)
                observed_cid_errors = {
                    "n_matched_peaks": n_cid_matched,
                    "p50_da": float(np.percentile(cid_abs_da, 50)),
                    "p75_da": float(np.percentile(cid_abs_da, 75)),
                    "p90_da": float(np.percentile(cid_abs_da, 90)),
                    "p95_da": float(np.percentile(cid_abs_da, 95)),
                    "p99_da": float(np.percentile(cid_abs_da, 99)),
                    "max_da": float(cid_abs_da.max()),
                    "mean_da": float(cid_abs_da.mean()),
                    "note": (
                        "These are from CID peaks that passed the quality gate "
                        "(~10 PPM tolerance). The true CID error distribution is "
                        "likely broader — these represent the best-matched tail."
                    ),
                }
                logger.info(
                    f"Observed CID errors (from {n_cid_matched} matched peaks): "
                    f"P50={observed_cid_errors['p50_da']:.4f} Da, "
                    f"P95={observed_cid_errors['p95_da']:.4f} Da, "
                    f"max={observed_cid_errors['max_da']:.4f} Da"
                )
            else:
                logger.info(f"Only {n_cid_matched} CID peaks passed quality gate — insufficient for observed error statistics")

        strategies = self._get_binning_strategies_from_config()

        # 4. Error sweep: fine-grained from sub-bin to CID-level
        sweep_errors = sorted(self.cid_error_sweep_da)
        literature_errors = set(self.cid_literature_errors_da)

        simulated_results: dict[str, Any] = {}
        for strategy_name, params in strategies.items():
            strategy = self._create_binning_strategy(params)
            mz_tensor = torch.from_numpy(all_mz_flat).float()
            base_bins = strategy.mz_to_bin(mz_tensor).numpy()

            # Sweep: constant Da errors
            sweep_results: dict[str, Any] = {}
            for delta_da in sweep_errors:
                plus_bins = strategy.mz_to_bin(mz_tensor + delta_da).numpy()
                minus_bins = strategy.mz_to_bin(mz_tensor - delta_da).numpy()
                mismatches = (base_bins != plus_bins) | (base_bins != minus_bins)
                mismatch_rate = float(mismatches.mean())

                # Stratify by m/z range
                per_range: dict[str, Any] = {}
                for range_name in self.mz_range_order:
                    low, high = self.mz_range_boundaries[range_name]
                    high = min(high, 1e12)
                    range_mask = (all_mz_flat >= low) & (all_mz_flat < high)
                    n_in_range = int(range_mask.sum())
                    if n_in_range > 0:
                        per_range[range_name] = {
                            "n_peaks": n_in_range,
                            "mismatch_rate": float(mismatches[range_mask].mean()),
                        }

                sweep_results[delta_da] = {
                    "mismatch_rate": mismatch_rate,
                    "n_peaks": len(all_mz_flat),
                    "is_literature": delta_da in literature_errors,
                    "per_mz_range": per_range,
                }

            # Orbitrap reference (PPM-based error)
            orbitrap_results: dict[str, Any] = {}
            for ppm in self.orbitrap_reference_errors_ppm:
                delta_ppm = all_mz_flat * ppm / 1e6
                plus_bins = strategy.mz_to_bin(torch.from_numpy(all_mz_flat + delta_ppm).float()).numpy()
                minus_bins = strategy.mz_to_bin(torch.from_numpy(all_mz_flat - delta_ppm).float()).numpy()
                mismatches = (base_bins != plus_bins) | (base_bins != minus_bins)
                mismatch_rate = float(mismatches.mean())

                per_range = {}
                for range_name in self.mz_range_order:
                    low, high = self.mz_range_boundaries[range_name]
                    high = min(high, 1e12)
                    range_mask = (all_mz_flat >= low) & (all_mz_flat < high)
                    n_in_range = int(range_mask.sum())
                    if n_in_range > 0:
                        per_range[range_name] = {
                            "n_peaks": n_in_range,
                            "mismatch_rate": float(mismatches[range_mask].mean()),
                        }

                orbitrap_results[ppm] = {
                    "mismatch_rate": mismatch_rate,
                    "n_peaks": len(all_mz_flat),
                    "per_mz_range": per_range,
                }

            simulated_results[strategy_name] = {
                "sweep": sweep_results,
                "orbitrap_errors": orbitrap_results,
            }

        # 5. Breaking point: find error Da where mismatch first exceeds 50%
        breaking_points: dict[str, Any] = {}
        for strategy_name, sim in simulated_results.items():
            bp = None
            for delta_da in sweep_errors:
                rate = sim["sweep"][delta_da]["mismatch_rate"]
                if rate >= 0.5:
                    bp = delta_da
                    break
            breaking_points[strategy_name] = {
                "error_da_at_50pct_mismatch": bp,
                "note": "First sweep error (Da) where mismatch >= 50%",
            }

        # 6. CID vs Orbitrap gap (at literature midpoint)
        cid_vs_orbitrap_gap: dict[str, Any] = {}
        mid_cid = self.cid_literature_errors_da[len(self.cid_literature_errors_da) // 2]
        mid_ppm = self.orbitrap_reference_errors_ppm[0]
        for strategy_name, sim in simulated_results.items():
            cid_rate = sim["sweep"].get(mid_cid, {}).get("mismatch_rate", 0)
            orb_rate = sim["orbitrap_errors"].get(mid_ppm, {}).get("mismatch_rate", 0)
            gap_ratio = cid_rate / orb_rate if orb_rate > 0 else float("inf")
            cid_vs_orbitrap_gap[strategy_name] = {
                "cid_error_da": mid_cid,
                "orbitrap_ppm": mid_ppm,
                "cid_mismatch": cid_rate,
                "orbitrap_mismatch": orb_rate,
                "gap_ratio": gap_ratio,
            }

        return {
            "composition": composition,
            "use_synthetic": use_synthetic,
            "n_cid_spectra": n_cid_spectra,
            "n_cid_peaks": len(all_mz_flat),
            "observed_cid_errors": observed_cid_errors,
            "sweep_errors_da": sweep_errors,
            "simulated_bin_mismatch": simulated_results,
            "breaking_points": breaking_points,
            "cid_vs_orbitrap_gap": cid_vs_orbitrap_gap,
        }

    # =========================================================================
    # Figure 8: CID Analysis Visualization
    # =========================================================================

    def _generate_cid_analysis_visualization(self) -> None:
        """Generate Figure 8: cid_analysis.png (2x2).

        Panels:
        - (0,0): Data composition bar chart (spectra + peaks by frag_type)
        - (0,1): Mismatch curve — mismatch rate vs error Da per strategy (line plot)
        - (1,0): Mismatch by m/z range at a representative error level
        - (1,1): Observed CID error distribution + breaking points table
        """
        cid = self.results.get("cid_analysis")
        if not cid:
            return

        logger.info("Generating Figure 8: CID analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        # ---- (0,0): Data composition ----
        ax = axes[0, 0]
        composition = cid["composition"]
        if composition:
            ft_names = sorted(composition.keys())
            spectra_counts = [composition[ft]["n_spectra"] for ft in ft_names]
            peak_counts = [composition[ft]["n_peaks"] for ft in ft_names]

            x_pos = np.arange(len(ft_names))
            width = 0.35
            ax.bar(x_pos - width / 2, spectra_counts, width, label="Spectra", color="steelblue", alpha=0.8)
            ax.bar(x_pos + width / 2, [p / 1000 for p in peak_counts], width, label="Peaks (k)", color="coral", alpha=0.8)

            # Annotate CID bar
            for i, ft in enumerate(ft_names):
                if self._is_cid_frag_type(ft):
                    ax.annotate(
                        "CID",
                        xy=(x_pos[i] - width / 2, spectra_counts[i]),
                        fontsize=9,
                        fontweight="bold",
                        color="firebrick",
                        ha="center",
                        va="bottom",
                    )

            ax.set_xticks(x_pos)
            ax.set_xticklabels(ft_names, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Count")
            ax.set_title(f"Data Composition ({cid.get('n_cid_spectra', '?')} CID spectra, {cid.get('n_cid_peaks', '?'):,d} CID peaks)")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")
        else:
            ax.text(0.5, 0.5, "No composition data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Data Composition")

        # ---- (0,1): Mismatch curve — error sweep ----
        ax = axes[0, 1]
        sim = cid.get("simulated_bin_mismatch", {})
        sweep_errors = cid.get("sweep_errors_da", [])
        if sim and sweep_errors:
            strategy_names = sorted(sim.keys())
            colors = plt.cm.tab10(np.linspace(0, 1, len(strategy_names)))

            for si, sn in enumerate(strategy_names):
                rates = [sim[sn]["sweep"].get(d, {}).get("mismatch_rate", 0) * 100 for d in sweep_errors]
                ax.plot(sweep_errors, rates, "o-", color=colors[si], linewidth=2, markersize=4, alpha=0.8, label=sn)

            # Reference lines: literature CID errors
            for lit_da in self.cid_literature_errors_da:
                ax.axvline(x=lit_da, color="firebrick", linestyle=":", alpha=0.4, linewidth=1.5)
            # Label the rightmost literature line
            if self.cid_literature_errors_da:
                ax.text(
                    self.cid_literature_errors_da[0],
                    55,
                    f"CID literature\n({self.cid_literature_errors_da[0]} Da)",
                    fontsize=7,
                    color="firebrick",
                    alpha=0.7,
                    ha="center",
                    va="bottom",
                )

            # Observed CID P95 marker
            obs = cid.get("observed_cid_errors")
            if obs:
                p95 = obs["p95_da"]
                ax.axvline(x=p95, color="darkgreen", linestyle="--", alpha=0.7, linewidth=2)
                ax.text(p95, 45, f"Observed CID\nP95={p95:.3f} Da", fontsize=7, color="darkgreen", fontweight="bold", ha="center", va="bottom")

            # 50% mismatch threshold
            ax.axhline(y=50, color="gray", linestyle="--", alpha=0.4, linewidth=1)
            ax.text(sweep_errors[0], 51, "50% mismatch", fontsize=7, color="gray", alpha=0.7)

            ax.set_xlabel("Error (Da)")
            ax.set_ylabel("Mismatch Rate (%)")
            ax.set_title("Mismatch Curve: Error Sweep (constant Da)")
            ax.legend(fontsize=6, loc="lower right", ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=0)
            ax.set_ylim(0, 105)
        else:
            ax.text(0.5, 0.5, "No sweep data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Mismatch Curve")

        # ---- (1,0): Mismatch by m/z range at a representative error ----
        ax = axes[1, 0]
        if sim and sweep_errors:
            # Pick adaptive_fine or first adaptive strategy
            target_strat = None
            for sn in strategy_names:
                if "fine" in sn:
                    target_strat = sn
                    break
            if target_strat is None:
                for sn in strategy_names:
                    if "adaptive" in sn.lower():
                        target_strat = sn
                        break
            if target_strat is None:
                target_strat = strategy_names[0]

            # Show multiple error levels across m/z ranges
            error_levels_to_show = [0.02, 0.05, 0.1, 0.2, 0.3]
            error_levels_to_show = [e for e in error_levels_to_show if e in sweep_errors]
            if not error_levels_to_show:
                error_levels_to_show = sweep_errors[:: max(1, len(sweep_errors) // 5)]

            orb_ppm = self.orbitrap_reference_errors_ppm[0]
            orb_per_range = sim[target_strat]["orbitrap_errors"].get(orb_ppm, {}).get("per_mz_range", {})

            range_names = [
                r
                for r in self.mz_range_order
                if any(sim[target_strat]["sweep"].get(e, {}).get("per_mz_range", {}).get(r) for e in error_levels_to_show)
            ]

            if range_names:
                sweep_colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(error_levels_to_show)))
                x_pos = np.arange(len(range_names))

                for ei, delta_da in enumerate(error_levels_to_show):
                    per_range = sim[target_strat]["sweep"].get(delta_da, {}).get("per_mz_range", {})
                    rates = [per_range.get(r, {}).get("mismatch_rate", 0) * 100 for r in range_names]
                    ax.plot(x_pos, rates, "o-", color=sweep_colors[ei], linewidth=2, markersize=6, label=f"{delta_da} Da")

                # Orbitrap reference
                orb_rates = [orb_per_range.get(r, {}).get("mismatch_rate", 0) * 100 for r in range_names]
                ax.plot(x_pos, orb_rates, "s--", color="steelblue", linewidth=2.5, markersize=7, label=f"Orbitrap {orb_ppm} PPM")

                ax.set_xticks(x_pos)
                ax.set_xticklabels(range_names, fontsize=9)
                ax.set_ylabel("Mismatch Rate (%)")
                ax.set_title(f"Mismatch by m/z Range ({target_strat})")
                ax.legend(fontsize=7, loc="best")
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, "No per-range data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title("Mismatch by m/z Range")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Mismatch by m/z Range")

        # ---- (1,1): Observed CID errors + breaking points ----
        ax = axes[1, 1]
        obs = cid.get("observed_cid_errors")
        breaking = cid.get("breaking_points", {})

        # Build a text-based summary table
        lines: list[Any] = []
        if obs:
            lines.append("Observed CID Errors (quality-gate survivors)")
            lines.append(f"  Matched peaks: {obs['n_matched_peaks']:,d}")
            lines.append(f"  P50: {obs['p50_da']:.4f} Da")
            lines.append(f"  P75: {obs['p75_da']:.4f} Da")
            lines.append(f"  P90: {obs['p90_da']:.4f} Da")
            lines.append(f"  P95: {obs['p95_da']:.4f} Da")
            lines.append(f"  P99: {obs['p99_da']:.4f} Da")
            lines.append(f"  Max: {obs['max_da']:.4f} Da")
            lines.append("")
            lines.append("NOTE: These are best-case CID errors")
            lines.append("(passed 10 PPM quality gate). True CID")
            lines.append("distribution is likely much broader.")
        else:
            lines.append("No observed CID errors available")
            lines.append("(too few CID peaks passed quality gate)")

        lines.append("")
        lines.append("Breaking Points (error Da -> 50% mismatch)")
        lines.append("-" * 44)
        for sn in sorted(breaking.keys()):
            bp = breaking[sn]["error_da_at_50pct_mismatch"]
            bp_str = f"{bp:.3f} Da" if bp is not None else "> 0.5 Da"
            lines.append(f"  {sn:30s} {bp_str}")

        ax.text(
            0.05,
            0.95,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightyellow", "alpha": 0.8},
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Observed CID Errors & Strategy Breaking Points")

        fig.suptitle("CID-Aware Analysis", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig_path = self.output_dir / "cid_analysis.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"CID analysis visualization saved to: {fig_path}")
