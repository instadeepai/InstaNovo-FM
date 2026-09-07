#!/usr/bin/env python
"""Comprehensive Intensity Distribution Analysis for Foundation Model Training.

This module analyzes intensity distributions across spectra, providing:
- Overall intensity statistics (percentiles, CV, dynamic range)
- Stratified analysis by fragmentation type, instrument, charge, m/z ranges
- Cross-comparison heatmaps
- Rich visualizations

All statistics are computed on sqrt-transformed, L2-normalized intensity values
(the model-space representation used during training). This means raw ion
currents have been square-root compressed and then normalized so that each
spectrum's intensity vector has unit L2 norm.

Usage:
    Integrated with SpectrumAnalyser - not meant to be called directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from instanovo.__init__ import console
from instanovo.utils.colorlogging import ColorLog

# Optional (for skewness/kurtosis)
try:
    from scipy.stats import kurtosis, skew

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = ColorLog(console, __name__).logger


class IntensityAnalyser:
    """Comprehensive analyzer for intensity distributions and statistics.

    All intensity values analyzed here are in **model space**: sqrt-transformed
    and L2-normalized per spectrum.  This is the representation seen by the
    foundation model during training.  Raw ion currents are *not* used.

    Provides:
    - Overall intensity distributions across all peaks
    - Percentile statistics (5th, 10th, 25th, 50th, 75th, 90th, 95th, 99th)
    - Coefficient of variation and dynamic range analysis
    - Stratified analysis by: fragmentation type, instrument, charge, m/z ranges
    - Advanced distribution metrics: skewness, kurtosis, intensity concentration
    - Cross-comparison heatmaps (frag_type x mz_range, charge x mz_range)
    """

    def __init__(self, config: DictConfig, output_dir: Optional[Path] = None) -> None:
        """Initialize the intensity analyzer.

        Args:
            config: Hydra configuration
            output_dir: Output directory for results and visualizations
        """
        self.config = config

        # Set output directory
        if output_dir is None:
            self.output_dir = Path("analysis_output") / "intensity_analysis"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model configuration
        self.max_mz = config.model.get("max_mz", 2500.0)
        self.min_mz = config.model.get("min_mz", 50.0)
        self.min_intensity = config.model.get("min_intensity", 0.01)

        # Analysis configuration
        analysis_config = config.get("analysis", {})

        # Intensity analysis specific config
        self.enable_intensity_analysis = analysis_config.get("enable_intensity_analysis", True)
        self.intensity_bins = analysis_config.get("intensity_bins", 100)
        self.intensity_log_scale = analysis_config.get("intensity_log_scale", True)
        self.compute_skewness_kurtosis = analysis_config.get("compute_skewness_kurtosis", True) and SCIPY_AVAILABLE

        # m/z range boundaries for stratification (align with BinningAnalyser)
        mz_range_boundaries = analysis_config.get("mz_range_boundaries", None)
        if mz_range_boundaries is None:
            # Default boundaries — use max_mz instead of inf for consistency
            self.mz_range_boundaries = {
                "immonium_internal": (0, 200),
                "core_fragment": (200, 800),
                "extended_fragment": (800, 1500),
                "high_mass_fragment": (1500, self.max_mz),
            }
        else:
            # Convert from config format [low, high] to tuple (low, high)
            self.mz_range_boundaries = {name: tuple(bounds) if isinstance(bounds, list) else bounds for name, bounds in mz_range_boundaries.items()}

        self.mz_range_order = ["immonium_internal", "core_fragment", "extended_fragment", "high_mass_fragment"]

        # Results storage structure matching existing pattern
        self.results: dict[str, Any] = {
            "summary": {},
            "per_spectrum": [],
            "overall_intensity_stats": {},
            "stratified_intensity": {},
            "raw_data": None,  # Will be DataFrame
        }

        logger.info(f"Intensity analyzer initialized. Output directory: {self.output_dir}")

    def _classify_mz_range(self, mz: float) -> str:
        """Classify m/z value into bin-aligned regions (matches BinningAnalyser).

        Args:
            mz: m/z value

        Returns:
            Range name (e.g., "core_fragment")
        """
        for range_name in self.mz_range_order:
            low, high = self.mz_range_boundaries[range_name]
            if low <= mz < high:
                return range_name
        return self.mz_range_order[-1]  # Default to last range

    def _classify_mz_range_vectorized(self, mz_values: np.ndarray) -> np.ndarray:
        """Vectorized m/z range classification.

        Args:
            mz_values: Array of m/z values.

        Returns:
            Array of range name strings.
        """
        result = np.full(len(mz_values), self.mz_range_order[-1], dtype=object)
        for range_name in self.mz_range_order:
            low, high = self.mz_range_boundaries[range_name]
            mask = (mz_values >= low) & (mz_values < high)
            result[mask] = range_name
        return result

    def analyze_spectrum(
        self,
        valid_mz: np.ndarray,
        valid_intensity: np.ndarray,
        metadata: Dict[str, Any],
        annotations: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Analyze intensity distribution for a single spectrum.

        This method is called by SpectrumAnalyser.analyze_single_spectrum() to
        collect per-spectrum intensity metrics that will be aggregated later.

        Intensities are expected in model space (sqrt + L2-normalized).

        Args:
            valid_mz: Filtered m/z array (peaks within valid range)
            valid_intensity: Filtered intensity array (sqrt + L2-normalized, 0-1 scale)
            metadata: Spectrum metadata dict containing:
                - frag_type: Fragmentation type (HCD, CID, etc.)
                - search_instrument: Instrument name
                - precursor_charge: Precursor charge state
            annotations: Optional list of matched ion annotations (for annotated vs unannotated)

        Returns:
            Dictionary containing lightweight per-spectrum intensity metrics.
            ``raw_intensity_records`` is a dict-of-arrays (columnar) for memory efficiency.
        """
        n_peaks = len(valid_intensity)

        if n_peaks == 0:
            return {"n_peaks": 0, "error": "No valid peaks"}

        # Basic statistics
        total_intensity = float(valid_intensity.sum())
        mean_intensity = float(valid_intensity.mean())
        median_intensity = float(np.median(valid_intensity))
        std_intensity = float(valid_intensity.std())
        min_intensity_val = float(valid_intensity.min())
        max_intensity_val = float(valid_intensity.max())

        # Derived metrics
        dynamic_range = max_intensity_val / max(min_intensity_val, 1e-10)
        coefficient_of_variation = std_intensity / max(mean_intensity, 1e-10)

        # Per-peak data as dict-of-arrays (columnar) for memory efficiency
        mz_ranges = self._classify_mz_range_vectorized(valid_mz)

        raw_intensity_records: Dict[str, np.ndarray] = {
            "intensity": valid_intensity.astype(np.float64),
            "mz": valid_mz.astype(np.float64),
            "mz_range": mz_ranges,
            "frag_type": np.full(n_peaks, metadata.get("frag_type", "unknown"), dtype=object),
            "instrument": np.full(n_peaks, metadata.get("search_instrument", "unknown"), dtype=object),
            "charge": np.full(n_peaks, metadata.get("precursor_charge", None), dtype=object),
        }

        # Add annotation info if available
        if annotations:
            is_annotated = np.zeros(n_peaks, dtype=bool)
            n_ann = min(len(annotations), n_peaks)
            for i in range(n_ann):
                is_annotated[i] = bool(annotations[i])
            raw_intensity_records["is_annotated"] = is_annotated

        return {
            "n_peaks": n_peaks,
            "total_intensity": total_intensity,
            "mean_intensity": mean_intensity,
            "median_intensity": median_intensity,
            "std_intensity": std_intensity,
            "min_intensity": min_intensity_val,
            "max_intensity": max_intensity_val,
            "dynamic_range": dynamic_range,
            "intensity_cv": coefficient_of_variation,
            "raw_intensity_records": raw_intensity_records,
            "metadata": metadata,
        }

    def _calculate_overall_intensity_stats(self, intensity_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive overall intensity statistics.

        Values are in model space (sqrt + L2-normalized).

        Args:
            intensity_df: DataFrame with intensity and metadata columns

        Returns:
            Dictionary with overall statistics
        """
        intensities = intensity_df["intensity"].values

        # Basic statistics
        stats: dict[str, Any] = {
            "n_peaks": len(intensities),
            "mean": float(np.mean(intensities)),
            "median": float(np.median(intensities)),
            "std": float(np.std(intensities)),
            "min": float(np.min(intensities)),
            "max": float(np.max(intensities)),
        }

        # Percentiles (5th, 25th, 50th, 75th, 95th, 99th)
        percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
        stats["percentiles"] = {f"p{p}": float(np.percentile(intensities, p)) for p in percentiles}

        # Derived metrics
        stats["dynamic_range"] = stats["max"] / max(stats["min"], 1e-10)
        stats["coefficient_of_variation"] = stats["std"] / max(stats["mean"], 1e-10)
        stats["iqr"] = stats["percentiles"]["p75"] - stats["percentiles"]["p25"]

        # Advanced statistics (if enabled)
        if self.compute_skewness_kurtosis:
            stats["skewness"] = float(skew(intensities))
            stats["kurtosis"] = float(kurtosis(intensities))

        # Intensity concentration (what fraction of total intensity is in top X%)
        sorted_intensities = np.sort(intensities)[::-1]
        cumsum = np.cumsum(sorted_intensities)
        total = cumsum[-1]

        stats["intensity_concentration"] = {
            "top_1_percent_fraction": float(cumsum[max(1, len(intensities) // 100)] / total) if len(intensities) >= 100 else float(cumsum[0] / total),
            "top_5_percent_fraction": float(cumsum[max(1, len(intensities) // 20)] / total) if len(intensities) >= 20 else float(cumsum[0] / total),
            "top_10_percent_fraction": float(cumsum[max(1, len(intensities) // 10)] / total) if len(intensities) >= 10 else float(cumsum[0] / total),
        }

        return stats

    def _calculate_stratified_stats(self, intensity_df: pd.DataFrame, stratify_by: str) -> Dict[str, Dict[str, Any]]:
        """Calculate intensity statistics stratified by a categorical variable.

        Args:
            intensity_df: DataFrame with intensity and metadata
            stratify_by: Column name to stratify by

        Returns:
            Dictionary mapping group values to statistics
        """
        stratified: dict[str, Any] = {}

        if stratify_by not in intensity_df.columns:
            return stratified

        for group_value in intensity_df[stratify_by].unique():
            if pd.isna(group_value) or group_value == "unknown":
                continue

            group_df = intensity_df[intensity_df[stratify_by] == group_value]
            intensities = group_df["intensity"].values

            if len(intensities) < 10:  # Skip small groups
                continue

            stratified[str(group_value)] = {
                "n_peaks": len(intensities),
                "mean": float(np.mean(intensities)),
                "median": float(np.median(intensities)),
                "std": float(np.std(intensities)),
                "cv": float(np.std(intensities) / max(np.mean(intensities), 1e-10)),
                "percentiles": {
                    "p5": float(np.percentile(intensities, 5)),
                    "p25": float(np.percentile(intensities, 25)),
                    "p50": float(np.percentile(intensities, 50)),
                    "p75": float(np.percentile(intensities, 75)),
                    "p95": float(np.percentile(intensities, 95)),
                },
            }

        return stratified

    def _calculate_cross_stratified_stats(self, intensity_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate cross-stratified statistics (e.g., frag_type x mz_range).

        Args:
            intensity_df: DataFrame with intensity and metadata

        Returns:
            Dictionary with cross-stratified statistics
        """
        cross_stats = {}

        # frag_type x mz_range
        if "frag_type" in intensity_df.columns and "mz_range" in intensity_df.columns:
            frag_mz_stats: dict[str, Any] = {}
            for frag_type in intensity_df["frag_type"].unique():
                if pd.isna(frag_type) or frag_type == "unknown":
                    continue

                frag_mz_stats[str(frag_type)] = {}
                frag_df = intensity_df[intensity_df["frag_type"] == frag_type]

                for mz_range in self.mz_range_order:
                    mz_df = frag_df[frag_df["mz_range"] == mz_range]
                    if len(mz_df) < 10:
                        continue

                    intensities = mz_df["intensity"].values
                    frag_mz_stats[str(frag_type)][mz_range] = {
                        "n_peaks": len(intensities),
                        "mean": float(np.mean(intensities)),
                        "median": float(np.median(intensities)),
                        "cv": float(np.std(intensities) / max(np.mean(intensities), 1e-10)),
                    }

            cross_stats["frag_type_x_mz_range"] = frag_mz_stats

        # instrument x charge
        if "instrument" in intensity_df.columns and "charge" in intensity_df.columns:
            inst_charge_stats: dict[str, Any] = {}
            for instrument in intensity_df["instrument"].unique():
                if pd.isna(instrument) or instrument == "unknown":
                    continue

                inst_charge_stats[str(instrument)] = {}
                inst_df = intensity_df[intensity_df["instrument"] == instrument]

                for charge in sorted(inst_df["charge"].dropna().unique()):
                    if pd.isna(charge):
                        continue

                    charge_df = inst_df[inst_df["charge"] == charge]
                    if len(charge_df) < 10:
                        continue

                    intensities = charge_df["intensity"].values
                    inst_charge_stats[str(instrument)][str(int(charge))] = {
                        "n_peaks": len(intensities),
                        "mean": float(np.mean(intensities)),
                        "median": float(np.median(intensities)),
                    }

            cross_stats["instrument_x_charge"] = inst_charge_stats

        # charge x mz_range (for heatmap visualization)
        if "charge" in intensity_df.columns and "mz_range" in intensity_df.columns:
            charge_mz_stats: dict[str, Any] = {}
            for charge in sorted(intensity_df["charge"].dropna().unique()):
                if pd.isna(charge):
                    continue
                charge_str = str(int(charge))
                charge_mz_stats[charge_str] = {}
                charge_df = intensity_df[intensity_df["charge"] == charge]

                for mz_range in self.mz_range_order:
                    mz_df = charge_df[charge_df["mz_range"] == mz_range]
                    if len(mz_df) < 10:
                        continue
                    intensities = mz_df["intensity"].values
                    charge_mz_stats[charge_str][mz_range] = {
                        "n_peaks": len(intensities),
                        "mean": float(np.mean(intensities)),
                        "median": float(np.median(intensities)),
                    }

            cross_stats["charge_x_mz_range"] = charge_mz_stats

        return cross_stats

    def aggregate_results(self, per_spectrum_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate intensity statistics across all analyzed spectra.

        Args:
            per_spectrum_results: List of results from analyze_spectrum() calls

        Returns:
            Dictionary containing aggregated results
        """
        logger.info("Aggregating intensity results...")

        # Collect all raw intensity records (dict-of-arrays format)
        arrays: Dict[str, list] = {}
        for result in per_spectrum_results:
            if "raw_intensity_records" not in result:
                continue
            records = result["raw_intensity_records"]
            if isinstance(records, dict):
                # New dict-of-arrays format
                for key, arr in records.items():
                    arrays.setdefault(key, []).append(np.asarray(arr))
            else:
                # Legacy list-of-dicts format (backward compatibility)
                for rec in records:
                    for key, val in rec.items():
                        arrays.setdefault(key, []).append(val)

        if not arrays or "intensity" not in arrays:
            logger.warning("No intensity data to aggregate")
            return {"error": "No data"}

        # Concatenate arrays or build from scalars
        combined: Dict[str, np.ndarray] = {}
        for key, parts in arrays.items():
            if isinstance(parts[0], np.ndarray):
                combined[key] = np.concatenate(parts)
            else:
                combined[key] = np.array(parts)

        # Convert to DataFrame for efficient aggregation
        intensity_df = pd.DataFrame(combined)

        logger.info(f"Aggregating {len(intensity_df):,d} peaks from {len(per_spectrum_results):,d} spectra")

        # 1. Overall intensity statistics
        overall_stats = self._calculate_overall_intensity_stats(intensity_df)

        # 2. Stratified analysis
        stratified_stats = {
            "by_frag_type": self._calculate_stratified_stats(intensity_df, "frag_type"),
            "by_instrument": self._calculate_stratified_stats(intensity_df, "instrument"),
            "by_charge": self._calculate_stratified_stats(intensity_df, "charge"),
            "by_mz_range": self._calculate_stratified_stats(intensity_df, "mz_range"),
        }

        # Add annotation stratification if available
        if "is_annotated" in intensity_df.columns:
            stratified_stats["by_annotation"] = self._calculate_stratified_stats(intensity_df, "is_annotated")

        # 3. Cross-stratification (e.g., frag_type x mz_range)
        cross_stratified = self._calculate_cross_stratified_stats(intensity_df)

        # Store results
        self.results = {
            "summary": overall_stats,
            "per_spectrum": per_spectrum_results,
            "overall_intensity_stats": overall_stats,
            "stratified_intensity": stratified_stats,
            "cross_stratified_intensity": cross_stratified,
            "raw_data": intensity_df,
        }

        logger.info("Intensity aggregation complete")
        return self.results

    def generate_visualizations(self) -> None:
        """Generate all intensity-related visualizations."""
        logger.info("Generating intensity visualizations...")

        if not self.results or self.results.get("raw_data") is None:
            logger.warning("No results available for visualization")
            return

        # Generate comprehensive intensity distribution plots
        self._generate_overall_intensity_distribution_plot()

        # Generate stratified comparison plots
        self._generate_stratified_intensity_plots()

        # Generate cross-comparison plots
        self._generate_cross_comparison_plots()

        logger.info("Intensity visualizations complete")

    def _generate_overall_intensity_distribution_plot(self) -> None:
        """Generate comprehensive intensity distribution visualization.

        Creates 3x3 plot grid with overall intensity characteristics.
        All values are in sqrt + L2-normalized model space.
        """
        if "raw_data" not in self.results or self.results["raw_data"] is None:
            return

        intensity_df = self.results["raw_data"]
        overall_stats = self.results["overall_intensity_stats"]

        fig = plt.figure(figsize=(24, 24))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        # PLOT 1: Linear histogram
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(intensity_df["intensity"], bins=self.intensity_bins, alpha=0.7, color="steelblue", edgecolor="black")
        ax1.set_xlabel("Intensity (normalized)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Count", fontsize=12, fontweight="bold")
        ax1.set_title("Overall Intensity Distribution\n(Linear Scale)", fontsize=14, fontweight="bold")
        ax1.axvline(overall_stats["mean"], color="red", linestyle="--", linewidth=2, label=f"Mean: {overall_stats['mean']:.3f}")
        ax1.axvline(overall_stats["median"], color="green", linestyle="--", linewidth=2, label=f"Median: {overall_stats['median']:.3f}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # PLOT 2: Log histogram
        ax2 = fig.add_subplot(gs[0, 1])
        min_intensity = max(intensity_df["intensity"].min(), 1e-6)
        max_intensity = intensity_df["intensity"].max()
        if min_intensity < max_intensity:
            ax2.hist(
                intensity_df["intensity"],
                bins=np.logspace(np.log10(min_intensity), np.log10(max_intensity), self.intensity_bins),
                alpha=0.7,
                color="darkorange",
                edgecolor="black",
            )
            ax2.set_xscale("log")
        else:
            ax2.hist(intensity_df["intensity"], bins=self.intensity_bins, alpha=0.7, color="darkorange", edgecolor="black")
        ax2.set_xlabel("Intensity (normalized, log scale)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Count", fontsize=12, fontweight="bold")
        ax2.set_title("Overall Intensity Distribution\n(Log Scale)", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # PLOT 3: Percentile plot
        ax3 = fig.add_subplot(gs[0, 2])
        percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = [overall_stats["percentiles"][f"p{p}"] for p in percentiles]
        ax3.plot(percentiles, percentile_values, marker="o", linewidth=2, markersize=8, color="darkgreen")
        ax3.set_xlabel("Percentile", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Intensity", fontsize=12, fontweight="bold")
        ax3.set_title("Intensity Percentiles", fontsize=14, fontweight="bold")
        ax3.grid(True, alpha=0.3)

        # Add value labels
        for p, val in zip(percentiles, percentile_values, strict=False):
            ax3.text(p, val, f"{val:.3f}", fontsize=8, ha="right", va="bottom")

        # PLOT 4: Cumulative distribution
        ax4 = fig.add_subplot(gs[1, 0])
        sorted_intensities = np.sort(intensity_df["intensity"])
        cumulative = np.arange(1, len(sorted_intensities) + 1) / len(sorted_intensities)
        ax4.plot(sorted_intensities, cumulative, linewidth=2, color="purple")
        ax4.set_xlabel("Intensity", fontsize=12, fontweight="bold")
        ax4.set_ylabel("Cumulative Probability", fontsize=12, fontweight="bold")
        ax4.set_title("Cumulative Distribution Function", fontsize=14, fontweight="bold")
        ax4.grid(True, alpha=0.3)

        # PLOT 5: Dynamic range per spectrum (sqrt + L2-normalized space)
        ax5 = fig.add_subplot(gs[1, 1])
        per_spectrum = self.results.get("per_spectrum", [])
        dynamic_ranges = [s["dynamic_range"] for s in per_spectrum if "dynamic_range" in s and s["dynamic_range"] > 0]
        if dynamic_ranges:
            ax5.hist(np.log10(dynamic_ranges), bins=50, alpha=0.7, color="coral", edgecolor="black")
            ax5.set_xlabel("log10(Dynamic Range)", fontsize=12, fontweight="bold")
            ax5.set_ylabel("Count", fontsize=12, fontweight="bold")
            ax5.set_title("Dynamic Range Distribution\n(per spectrum, sqrt + L2-normalized space)", fontsize=14, fontweight="bold")
            ax5.axvline(
                np.log10(np.median(dynamic_ranges)), color="red", linestyle="--", linewidth=2, label=f"Median: {np.median(dynamic_ranges):.1f}"
            )
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax5.transAxes)

        # PLOT 6: Coefficient of variation (sqrt + L2-normalized space)
        ax6 = fig.add_subplot(gs[1, 2])
        cv_values = [s["intensity_cv"] for s in per_spectrum if "intensity_cv" in s]
        if cv_values:
            ax6.hist(cv_values, bins=50, alpha=0.7, color="teal", edgecolor="black")
            ax6.set_xlabel("Coefficient of Variation", fontsize=12, fontweight="bold")
            ax6.set_ylabel("Count", fontsize=12, fontweight="bold")
            ax6.set_title("Coefficient of Variation\n(per spectrum, sqrt + L2-normalized space)", fontsize=14, fontweight="bold")
            ax6.axvline(np.median(cv_values), color="red", linestyle="--", linewidth=2, label=f"Median: {np.median(cv_values):.2f}")
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax6.transAxes)

        # PLOT 7: Intensity vs m/z hexbin density
        ax7 = fig.add_subplot(gs[2, 0])
        if "mz" in intensity_df.columns and len(intensity_df) > 0:
            hb = ax7.hexbin(
                intensity_df["mz"],
                intensity_df["intensity"],
                gridsize=50,
                cmap="inferno",
                mincnt=1,
            )
            ax7.set_xlabel("m/z (Da)", fontsize=12, fontweight="bold")
            ax7.set_ylabel("Intensity", fontsize=12, fontweight="bold")
            ax7.set_title("Intensity vs m/z\n(hexbin density)", fontsize=14, fontweight="bold")
            plt.colorbar(hb, ax=ax7, label="Count")
            ax7.grid(True, alpha=0.3)
        else:
            ax7.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax7.transAxes)

        # PLOT 8: Cumulative intensity by peak rank
        ax8 = fig.add_subplot(gs[2, 1])
        sorted_desc = np.sort(intensity_df["intensity"].values)[::-1]
        cum_frac = np.cumsum(sorted_desc) / sorted_desc.sum()
        ranks = np.arange(1, len(cum_frac) + 1)
        # Subsample for plotting if too many points
        if len(ranks) > 2000:
            idx = np.linspace(0, len(ranks) - 1, 2000, dtype=int)
            ranks_plot, cum_plot = ranks[idx], cum_frac[idx]
        else:
            ranks_plot, cum_plot = ranks, cum_frac
        ax8.plot(ranks_plot, cum_plot, linewidth=2, color="darkblue")
        ax8.set_xlabel("Peak Rank (sorted by intensity)", fontsize=12, fontweight="bold")
        ax8.set_ylabel("Cumulative Intensity Fraction", fontsize=12, fontweight="bold")
        ax8.set_title("Cumulative Intensity by Peak Rank", fontsize=14, fontweight="bold")
        ax8.grid(True, alpha=0.3)
        # Mark 50% and 90% thresholds
        for threshold in [0.5, 0.9]:
            idx_t = np.searchsorted(cum_frac, threshold)
            if idx_t < len(ranks):
                ax8.axhline(threshold, color="gray", linestyle=":", alpha=0.5)
                ax8.axvline(ranks[idx_t], color="red", linestyle="--", alpha=0.5, label=f"{threshold * 100:.0f}% at rank {ranks[idx_t]:,d}")
        ax8.legend(fontsize=9)

        # PLOT 9: Per-spectrum peak count histogram
        ax9 = fig.add_subplot(gs[2, 2])
        peak_counts = [s["n_peaks"] for s in per_spectrum if "n_peaks" in s and s["n_peaks"] > 0]
        if peak_counts:
            ax9.hist(peak_counts, bins=50, alpha=0.7, color="mediumpurple", edgecolor="black")
            ax9.set_xlabel("Number of Peaks", fontsize=12, fontweight="bold")
            ax9.set_ylabel("Number of Spectra", fontsize=12, fontweight="bold")
            ax9.set_title("Peak Count Distribution\n(per spectrum)", fontsize=14, fontweight="bold")
            ax9.axvline(np.median(peak_counts), color="red", linestyle="--", linewidth=2, label=f"Median: {np.median(peak_counts):.0f}")
            ax9.legend()
            ax9.grid(True, alpha=0.3)
        else:
            ax9.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax9.transAxes)

        # Save figure
        output_path = self.output_dir / "intensity_distribution_overall.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved overall intensity distribution plot: {output_path}")

    def _generate_stratified_intensity_plots(self) -> None:
        """Generate stratified intensity comparison visualizations.

        Creates 3x3 plot grid with stratified comparisons.
        """
        if "raw_data" not in self.results or self.results["raw_data"] is None:
            return

        intensity_df = self.results["raw_data"]
        stratified = self.results.get("stratified_intensity", {})

        fig = plt.figure(figsize=(24, 24))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        # PLOT 1: Intensity by fragmentation type (box plot)
        ax1 = fig.add_subplot(gs[0, 0])
        if "frag_type" in intensity_df.columns and "by_frag_type" in stratified and stratified["by_frag_type"]:
            frag_types = sorted([ft for ft in intensity_df["frag_type"].unique() if ft != "unknown" and not pd.isna(ft)])
            if frag_types:
                data_by_frag = [intensity_df[intensity_df["frag_type"] == ft]["intensity"].values for ft in frag_types]

                bp = ax1.boxplot(data_by_frag, tick_labels=frag_types, patch_artist=True, showfliers=False)  # Hide outliers for clarity
                for patch in bp["boxes"]:
                    patch.set_facecolor("lightblue")

                ax1.set_xlabel("Fragmentation Type", fontsize=12, fontweight="bold")
                ax1.set_ylabel("Intensity", fontsize=12, fontweight="bold")
                ax1.set_title("Intensity Distribution by Fragmentation Type", fontsize=14, fontweight="bold")
                ax1.grid(True, alpha=0.3, axis="y")
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
            else:
                ax1.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax1.transAxes)
        else:
            ax1.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax1.transAxes)

        # PLOT 2: Intensity by instrument (violin plot)
        ax2 = fig.add_subplot(gs[0, 1])
        if "instrument" in intensity_df.columns and "by_instrument" in stratified and stratified["by_instrument"]:
            instruments = sorted([inst for inst in intensity_df["instrument"].unique() if inst != "unknown" and not pd.isna(inst)])[
                :10
            ]  # Limit to top 10

            if instruments:
                data_by_inst = [intensity_df[intensity_df["instrument"] == inst]["intensity"].values for inst in instruments]

                ax2.violinplot(data_by_inst, positions=range(len(instruments)), showmeans=True, showmedians=True)

                ax2.set_xticks(range(len(instruments)))
                ax2.set_xticklabels(instruments, rotation=45, ha="right")
                ax2.set_xlabel("Instrument", fontsize=12, fontweight="bold")
                ax2.set_ylabel("Intensity", fontsize=12, fontweight="bold")
                ax2.set_title("Intensity Distribution by Instrument", fontsize=14, fontweight="bold")
                ax2.grid(True, alpha=0.3, axis="y")
            else:
                ax2.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax2.transAxes)
        else:
            ax2.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax2.transAxes)

        # PLOT 3: Intensity by charge state
        ax3 = fig.add_subplot(gs[0, 2])
        if "charge" in intensity_df.columns and "by_charge" in stratified and stratified["by_charge"]:
            charges = sorted([ch for ch in intensity_df["charge"].unique() if not pd.isna(ch)])
            if charges:
                data_by_charge = [intensity_df[intensity_df["charge"] == ch]["intensity"].values for ch in charges]

                bp = ax3.boxplot(data_by_charge, tick_labels=[f"{int(ch)}+" for ch in charges], patch_artist=True, showfliers=False)
                for patch in bp["boxes"]:
                    patch.set_facecolor("lightgreen")

                ax3.set_xlabel("Precursor Charge", fontsize=12, fontweight="bold")
                ax3.set_ylabel("Intensity", fontsize=12, fontweight="bold")
                ax3.set_title("Intensity Distribution by Charge State", fontsize=14, fontweight="bold")
                ax3.grid(True, alpha=0.3, axis="y")
            else:
                ax3.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax3.transAxes)

        # PLOT 4: Intensity by m/z range
        ax4 = fig.add_subplot(gs[1, 0])
        if "mz_range" in intensity_df.columns:
            data_by_mz = [
                intensity_df[intensity_df["mz_range"] == mz_range]["intensity"].values
                for mz_range in self.mz_range_order
                if mz_range in intensity_df["mz_range"].values
            ]
            labels = [mz_range for mz_range in self.mz_range_order if mz_range in intensity_df["mz_range"].values]

            if data_by_mz:
                bp = ax4.boxplot(data_by_mz, tick_labels=labels, patch_artist=True, showfliers=False)
                for patch in bp["boxes"]:
                    patch.set_facecolor("lightyellow")

                ax4.set_xlabel("m/z Range", fontsize=12, fontweight="bold")
                ax4.set_ylabel("Intensity", fontsize=12, fontweight="bold")
                ax4.set_title("Intensity Distribution by m/z Range", fontsize=14, fontweight="bold")
                ax4.grid(True, alpha=0.3, axis="y")
                plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")
            else:
                ax4.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax4.transAxes)
        else:
            ax4.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax4.transAxes)

        # PLOT 5: CV comparison across frag types
        ax5 = fig.add_subplot(gs[1, 1])
        if "by_frag_type" in stratified and stratified["by_frag_type"]:
            frag_type_stats = stratified["by_frag_type"]
            frag_types = sorted(frag_type_stats.keys())
            cv_values = [frag_type_stats[ft]["cv"] for ft in frag_types]

            bars = ax5.bar(range(len(frag_types)), cv_values, color="steelblue", alpha=0.7, edgecolor="black")
            ax5.set_xticks(range(len(frag_types)))
            ax5.set_xticklabels(frag_types, rotation=45, ha="right")
            ax5.set_xlabel("Fragmentation Type", fontsize=12, fontweight="bold")
            ax5.set_ylabel("Coefficient of Variation", fontsize=12, fontweight="bold")
            ax5.set_title("Intensity CV by Fragmentation Type", fontsize=14, fontweight="bold")
            ax5.grid(True, alpha=0.3, axis="y")

            # Add value labels
            for i, (_bar, cv) in enumerate(zip(bars, cv_values, strict=False)):
                ax5.text(i, cv, f"{cv:.3f}", ha="center", va="bottom", fontsize=9)
        else:
            ax5.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax5.transAxes)

        # PLOT 6: CV comparison across instruments
        ax6 = fig.add_subplot(gs[1, 2])
        if "by_instrument" in stratified and stratified["by_instrument"]:
            inst_stats = stratified["by_instrument"]
            # Sort by n_peaks and take top 10
            instruments_sorted = sorted(inst_stats.items(), key=lambda x: x[1]["n_peaks"], reverse=True)[:10]
            instruments = [inst for inst, _ in instruments_sorted]
            cv_values = [inst_stats[inst]["cv"] for inst in instruments]

            bars = ax6.bar(range(len(instruments)), cv_values, color="darkorange", alpha=0.7, edgecolor="black")
            ax6.set_xticks(range(len(instruments)))
            ax6.set_xticklabels(instruments, rotation=45, ha="right")
            ax6.set_xlabel("Instrument", fontsize=12, fontweight="bold")
            ax6.set_ylabel("Coefficient of Variation", fontsize=12, fontweight="bold")
            ax6.set_title("Intensity CV by Instrument", fontsize=14, fontweight="bold")
            ax6.grid(True, alpha=0.3, axis="y")
        else:
            ax6.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax6.transAxes)

        # PLOT 7: Annotated vs unannotated intensity boxplot
        ax7 = fig.add_subplot(gs[2, 0])
        if "is_annotated" in intensity_df.columns and "by_annotation" in stratified and stratified["by_annotation"]:
            ann_groups = sorted(intensity_df["is_annotated"].unique())
            data_by_ann = [intensity_df[intensity_df["is_annotated"] == g]["intensity"].values for g in ann_groups]
            labels = ["Unannotated" if not g else "Annotated" for g in ann_groups]
            if data_by_ann and all(len(d) > 0 for d in data_by_ann):
                bp = ax7.boxplot(data_by_ann, tick_labels=labels, patch_artist=True, showfliers=False)
                colors = ["salmon", "lightgreen"]
                for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])], strict=False):
                    patch.set_facecolor(color)
                ax7.set_ylabel("Intensity", fontsize=12, fontweight="bold")
                ax7.set_title("Annotated vs Unannotated\nIntensity Distribution", fontsize=14, fontweight="bold")
                ax7.grid(True, alpha=0.3, axis="y")
            else:
                ax7.text(0.5, 0.5, "Insufficient annotation data", ha="center", va="center", transform=ax7.transAxes)
        else:
            ax7.text(0.5, 0.5, "No annotation data", ha="center", va="center", transform=ax7.transAxes)

        # PLOT 8: Intensity by peak rank position (bar + error)
        ax8 = fig.add_subplot(gs[2, 1])
        per_spectrum = self.results.get("per_spectrum", [])
        # Compute mean intensity at each rank across spectra
        max_rank = 20  # Show top 20 ranks
        rank_means = []
        rank_stds = []
        for rank in range(max_rank):
            rank_vals = []
            for s in per_spectrum:
                recs = s.get("raw_intensity_records", {})
                if isinstance(recs, dict) and "intensity" in recs:
                    intensities = recs["intensity"]
                    sorted_idx = np.argsort(intensities)[::-1]
                    if rank < len(sorted_idx):
                        rank_vals.append(intensities[sorted_idx[rank]])
            if rank_vals:
                rank_means.append(np.mean(rank_vals))
                rank_stds.append(np.std(rank_vals))
            else:
                break

        if rank_means:
            positions = np.arange(1, len(rank_means) + 1)
            ax8.bar(positions, rank_means, yerr=rank_stds, color="teal", alpha=0.7, edgecolor="black", capsize=2)
            ax8.set_xlabel("Peak Rank (by intensity, descending)", fontsize=12, fontweight="bold")
            ax8.set_ylabel("Mean Intensity", fontsize=12, fontweight="bold")
            ax8.set_title("Intensity by Peak Rank\n(mean +/- std across spectra)", fontsize=14, fontweight="bold")
            ax8.grid(True, alpha=0.3, axis="y")
        else:
            ax8.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax8.transAxes)

        # PLOT 9: Mean intensity by m/z range (bar + error bars)
        ax9 = fig.add_subplot(gs[2, 2])
        if "by_mz_range" in stratified and stratified["by_mz_range"]:
            mz_stats = stratified["by_mz_range"]
            present_ranges = [r for r in self.mz_range_order if r in mz_stats]
            if present_ranges:
                means = [mz_stats[r]["mean"] for r in present_ranges]
                stds = [mz_stats[r]["std"] for r in present_ranges]
                positions = np.arange(len(present_ranges))

                ax9.bar(positions, means, yerr=stds, color="goldenrod", alpha=0.7, edgecolor="black", capsize=3)
                ax9.set_xticks(positions)
                ax9.set_xticklabels(present_ranges, rotation=45, ha="right")
                ax9.set_ylabel("Mean Intensity", fontsize=12, fontweight="bold")
                ax9.set_title("Mean Intensity by m/z Range\n(+/- std)", fontsize=14, fontweight="bold")
                ax9.grid(True, alpha=0.3, axis="y")
            else:
                ax9.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax9.transAxes)
        else:
            ax9.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax9.transAxes)

        # Save figure
        output_path = self.output_dir / "intensity_stratified_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved stratified intensity comparison plot: {output_path}")

    def _generate_cross_comparison_plots(self) -> None:
        """Generate cross-comparison heatmaps.

        Creates 2x2 plot grid with cross-comparisons.
        """
        if "cross_stratified_intensity" not in self.results:
            return

        cross_strat = self.results["cross_stratified_intensity"]

        if not cross_strat:
            logger.info("No cross-stratified data available for visualization")
            return

        fig = plt.figure(figsize=(20, 20))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # PLOT 1: Mean intensity (frag_type x mz_range)
        ax1 = fig.add_subplot(gs[0, 0])
        if "frag_type_x_mz_range" in cross_strat and cross_strat["frag_type_x_mz_range"]:
            data = cross_strat["frag_type_x_mz_range"]

            # Build matrix
            frag_types = sorted(data.keys())
            matrix = []
            for ft in frag_types:
                row = [data[ft].get(mz_range, {}).get("mean", np.nan) for mz_range in self.mz_range_order]
                matrix.append(row)

            matrix = np.array(matrix)

            # Plot heatmap
            im = ax1.imshow(matrix, aspect="auto", cmap="YlOrRd")
            ax1.set_xticks(range(len(self.mz_range_order)))
            ax1.set_xticklabels(self.mz_range_order, rotation=45, ha="right")
            ax1.set_yticks(range(len(frag_types)))
            ax1.set_yticklabels(frag_types)
            ax1.set_xlabel("m/z Range", fontsize=12, fontweight="bold")
            ax1.set_ylabel("Fragmentation Type", fontsize=12, fontweight="bold")
            ax1.set_title("Mean Intensity\n(Frag Type x m/z Range)", fontsize=14, fontweight="bold")

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax1)
            cbar.set_label("Mean Intensity", fontsize=10)

            # Add value annotations
            for i in range(len(frag_types)):
                for j in range(len(self.mz_range_order)):
                    if not np.isnan(matrix[i, j]):
                        ax1.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
        else:
            ax1.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax1.transAxes)

        # PLOT 2: CV (frag_type x mz_range)
        ax2 = fig.add_subplot(gs[0, 1])
        if "frag_type_x_mz_range" in cross_strat and cross_strat["frag_type_x_mz_range"]:
            data = cross_strat["frag_type_x_mz_range"]
            frag_types = sorted(data.keys())

            # Build matrix
            matrix = []
            for ft in frag_types:
                row = [data[ft].get(mz_range, {}).get("cv", np.nan) for mz_range in self.mz_range_order]
                matrix.append(row)

            matrix = np.array(matrix)

            # Plot heatmap
            im = ax2.imshow(matrix, aspect="auto", cmap="RdYlGn_r")
            ax2.set_xticks(range(len(self.mz_range_order)))
            ax2.set_xticklabels(self.mz_range_order, rotation=45, ha="right")
            ax2.set_yticks(range(len(frag_types)))
            ax2.set_yticklabels(frag_types)
            ax2.set_xlabel("m/z Range", fontsize=12, fontweight="bold")
            ax2.set_ylabel("Fragmentation Type", fontsize=12, fontweight="bold")
            ax2.set_title("Coefficient of Variation\n(Frag Type x m/z Range)", fontsize=14, fontweight="bold")

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2)
            cbar.set_label("CV", fontsize=10)

            # Add value annotations
            for i in range(len(frag_types)):
                for j in range(len(self.mz_range_order)):
                    if not np.isnan(matrix[i, j]):
                        ax2.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
        else:
            ax2.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax2.transAxes)

        # PLOT 3: Median intensity heatmap (charge x mz_range)
        ax3 = fig.add_subplot(gs[1, 0])
        if "charge_x_mz_range" in cross_strat and cross_strat["charge_x_mz_range"]:
            data = cross_strat["charge_x_mz_range"]
            charges = sorted(data.keys(), key=lambda x: int(x))

            matrix = []
            for ch in charges:
                row = [data[ch].get(mz_range, {}).get("median", np.nan) for mz_range in self.mz_range_order]
                matrix.append(row)

            matrix = np.array(matrix)

            im = ax3.imshow(matrix, aspect="auto", cmap="YlGnBu")
            ax3.set_xticks(range(len(self.mz_range_order)))
            ax3.set_xticklabels(self.mz_range_order, rotation=45, ha="right")
            ax3.set_yticks(range(len(charges)))
            ax3.set_yticklabels([f"{ch}+" for ch in charges])
            ax3.set_xlabel("m/z Range", fontsize=12, fontweight="bold")
            ax3.set_ylabel("Charge State", fontsize=12, fontweight="bold")
            ax3.set_title("Median Intensity\n(Charge x m/z Range)", fontsize=14, fontweight="bold")

            cbar = plt.colorbar(im, ax=ax3)
            cbar.set_label("Median Intensity", fontsize=10)

            for i in range(len(charges)):
                for j in range(len(self.mz_range_order)):
                    if not np.isnan(matrix[i, j]):
                        ax3.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
        else:
            ax3.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax3.transAxes)

        # PLOT 4: Peak count heatmap (frag_type x mz_range)
        ax4 = fig.add_subplot(gs[1, 1])
        if "frag_type_x_mz_range" in cross_strat and cross_strat["frag_type_x_mz_range"]:
            data = cross_strat["frag_type_x_mz_range"]
            frag_types = sorted(data.keys())

            matrix = []
            for ft in frag_types:
                row = [data[ft].get(mz_range, {}).get("n_peaks", 0) for mz_range in self.mz_range_order]
                matrix.append(row)

            matrix = np.array(matrix, dtype=float)
            # Use log scale for better visibility
            with np.errstate(divide="ignore"):
                matrix_log = np.where(matrix > 0, np.log10(matrix), np.nan)

            im = ax4.imshow(matrix_log, aspect="auto", cmap="viridis")
            ax4.set_xticks(range(len(self.mz_range_order)))
            ax4.set_xticklabels(self.mz_range_order, rotation=45, ha="right")
            ax4.set_yticks(range(len(frag_types)))
            ax4.set_yticklabels(frag_types)
            ax4.set_xlabel("m/z Range", fontsize=12, fontweight="bold")
            ax4.set_ylabel("Fragmentation Type", fontsize=12, fontweight="bold")
            ax4.set_title("Peak Count (log10)\n(Frag Type x m/z Range)", fontsize=14, fontweight="bold")

            cbar = plt.colorbar(im, ax=ax4)
            cbar.set_label("log10(Peak Count)", fontsize=10)

            for i in range(len(frag_types)):
                for j in range(len(self.mz_range_order)):
                    val = int(matrix[i, j])
                    if val > 0:
                        ax4.text(j, i, f"{val:,d}", ha="center", va="center", fontsize=7)
        else:
            ax4.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax4.transAxes)

        # Save figure
        output_path = self.output_dir / "intensity_cross_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved cross-comparison plot: {output_path}")

    def save_results(self) -> None:
        """Save intensity analysis results to files."""
        logger.info("Saving intensity analysis results...")

        if not self.results:
            logger.warning("No results to save")
            return

        # Save overall statistics as JSON
        summary_path = self.output_dir / "intensity_analysis_summary.json"
        summary_data = {
            "overall_intensity_stats": self.results.get("overall_intensity_stats", {}),
            "stratified_intensity": self.results.get("stratified_intensity", {}),
            "cross_stratified_intensity": self.results.get("cross_stratified_intensity", {}),
        }

        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        logger.info(f"Saved intensity summary: {summary_path}")

        # Save per-peak intensity data as CSV (only essential columns to limit file size)
        if "raw_data" in self.results and self.results["raw_data"] is not None:
            intensity_df = self.results["raw_data"]
            save_cols = [c for c in ["intensity", "mz", "is_annotated"] if c in intensity_df.columns]
            csv_path = self.output_dir / "intensity_per_peak.csv"
            intensity_df[save_cols].to_csv(csv_path, index=False, float_format="%.6f")
            logger.info(f"Saved per-peak intensity data: {csv_path} ({len(intensity_df):,d} peaks, columns: {save_cols})")

        # Save per-spectrum summary as CSV
        if "per_spectrum" in self.results:
            per_spectrum_records = []
            for result in self.results["per_spectrum"]:
                if "error" not in result:
                    per_spectrum_records.append(
                        {
                            "n_peaks": result.get("n_peaks", 0),
                            "total_intensity": result.get("total_intensity", 0),
                            "mean_intensity": result.get("mean_intensity", 0),
                            "median_intensity": result.get("median_intensity", 0),
                            "std_intensity": result.get("std_intensity", 0),
                            "dynamic_range": result.get("dynamic_range", 0),
                            "intensity_cv": result.get("intensity_cv", 0),
                            "frag_type": result.get("metadata", {}).get("frag_type", "unknown"),
                            "instrument": result.get("metadata", {}).get("search_instrument", "unknown"),
                        }
                    )

            if per_spectrum_records:
                per_spec_df = pd.DataFrame(per_spectrum_records)
                csv_path = self.output_dir / "intensity_per_spectrum.csv"
                per_spec_df.to_csv(csv_path, index=False, float_format="%.6f")
                logger.info(f"Saved per-spectrum intensity data: {csv_path} ({len(per_spec_df):,d} spectra)")

        logger.info("Intensity analysis results saved")

    def print_summary(self) -> None:
        """Print analysis summary to console."""
        if not self.results:
            logger.warning("No results to summarize")
            return

        logger.info("=" * 80)
        logger.info("INTENSITY ANALYSIS SUMMARY")
        logger.info("=" * 80)

        overall = self.results.get("overall_intensity_stats", {})

        if overall:
            logger.info("Overall Statistics:")
            logger.info(f"  N peaks: {overall['n_peaks']:,d}")
            logger.info(f"  Mean intensity: {overall['mean']:.4f}")
            logger.info(f"  Median intensity: {overall['median']:.4f}")
            logger.info(f"  Std intensity: {overall['std']:.4f}")
            logger.info(f"  CV: {overall['coefficient_of_variation']:.4f}")
            logger.info(f"  Dynamic range: {overall['dynamic_range']:.1f}x")
            logger.info(f"  P95: {overall['percentiles']['p95']:.4f}")

        stratified = self.results.get("stratified_intensity", {})

        if "by_frag_type" in stratified and stratified["by_frag_type"]:
            logger.info("")
            logger.info("By Fragmentation Type:")
            for ft, stats in sorted(stratified["by_frag_type"].items()):
                logger.info(f"  {ft}: mean={stats['mean']:.4f}, cv={stats['cv']:.3f}, n={stats['n_peaks']:,d}")

        if "by_instrument" in stratified and stratified["by_instrument"]:
            logger.info("")
            logger.info("By Instrument (top 5):")
            top_instruments = sorted(stratified["by_instrument"].items(), key=lambda x: x[1]["n_peaks"], reverse=True)[:5]
            for inst, stats in top_instruments:
                logger.info(f"  {inst}: mean={stats['mean']:.4f}, cv={stats['cv']:.3f}, n={stats['n_peaks']:,d}")

        logger.info("=" * 80)
