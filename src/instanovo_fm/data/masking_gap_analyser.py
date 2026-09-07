#!/usr/bin/env python
"""Masking Gap Analysis for Foundation Model Training.

Analyzes the m/z distance from each masked peak to its nearest unmasked
neighbors. This reveals how much local context the model has when predicting
masked peaks, and how the classification head's binning (group + offset)
relates to these distances.

Usage:
    Integrated with SpectrumAnalyser - not meant to be called directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from instanovo.__init__ import console
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger


class MaskingGapAnalyser:
    """Analyzes m/z distances from masked peaks to nearest unmasked neighbors.

    For each masked peak, computes:
    - Distance to nearest unmasked peak on left (lower m/z)
    - Distance to nearest unmasked peak on right (higher m/z)
    - Nearest distance (min of left, right)
    - Total gap (left + right)
    - Distances expressed in bins and groups (relates to classification head)

    Results are stratified by m/z region and related to the binning strategy.
    """

    def __init__(self, config: DictConfig, output_dir: Optional[Path] = None) -> None:
        """Initialise the input."""
        self.config = config

        if output_dir is None:
            self.output_dir = Path("analysis_output") / "masking_gap_analysis"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Binning parameters
        mz_head = config.model.get("mz_head", {})
        binning_config = mz_head.get("binning", {})
        self.bin_size = binning_config.get("bin_size", 0.02)
        self.bin_group_size = mz_head.get("bin_group_size", 50)
        self.group_width_da = self.bin_size * self.bin_group_size

        # m/z range boundaries (shared with other analysers)
        analysis_config = config.get("analysis", {})
        mz_range_boundaries = analysis_config.get("mz_range_boundaries", None)
        if mz_range_boundaries is None:
            self.mz_range_boundaries = {
                "immonium_internal": (0, 200),
                "core_fragment": (200, 800),
                "extended_fragment": (800, 1500),
                "high_mass_fragment": (1500, float("inf")),
            }
        else:
            self.mz_range_boundaries = {name: tuple(bounds) if isinstance(bounds, list) else bounds for name, bounds in mz_range_boundaries.items()}

        self.mz_range_order = [
            "immonium_internal",
            "core_fragment",
            "extended_fragment",
            "high_mass_fragment",
        ]

        self.results: Dict[str, Any] = {}

        logger.info(
            f"Masking gap analyser initialized. "
            f"bin_size={self.bin_size} Da, group_size={self.bin_group_size}, "
            f"group_width={self.group_width_da:.2f} Da"
        )

    def _classify_mz_range(self, mz: float) -> str:
        """Classify m/z value into bin-aligned regions."""
        for range_name in self.mz_range_order:
            low, high = self.mz_range_boundaries[range_name]
            if low <= mz < high:
                return range_name
        return self.mz_range_order[-1]

    def analyze_spectrum(
        self,
        valid_mz: np.ndarray,
        mlm_mask_valid: np.ndarray,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze masking gaps for a single spectrum.

        Args:
            valid_mz: Sorted m/z values of valid (non-padded) peaks.
            mlm_mask_valid: Boolean mask over valid peaks (True = masked).
            metadata: Spectrum metadata (frag_type, etc.).

        Returns:
            Dict with gap_records list and summary counts.
        """
        masked_indices = np.where(mlm_mask_valid)[0]
        unmasked_indices = np.where(~mlm_mask_valid)[0]

        n_masked = len(masked_indices)
        n_unmasked = len(unmasked_indices)

        if n_masked == 0:
            return {"n_masked": 0, "skip": True}

        if n_unmasked == 0:
            return {"n_masked": n_masked, "n_unmasked": 0, "all_masked": True, "skip": False}

        masked_mz = valid_mz[masked_indices]
        unmasked_mz = valid_mz[unmasked_indices]  # sorted since valid_mz is sorted

        # Vectorized nearest-neighbor search via searchsorted
        insert_pos = np.searchsorted(unmasked_mz, masked_mz)

        # Left neighbor: insert_pos - 1
        left_idx = insert_pos - 1
        has_left = left_idx >= 0
        left_distance = np.full(n_masked, np.nan)
        left_distance[has_left] = masked_mz[has_left] - unmasked_mz[left_idx[has_left]]

        # Right neighbor: insert_pos
        right_idx = insert_pos
        has_right = right_idx < n_unmasked
        right_distance = np.full(n_masked, np.nan)
        right_distance[has_right] = unmasked_mz[right_idx[has_right]] - masked_mz[has_right]

        # Nearest and total gap
        nearest_distance = np.fmin(left_distance, right_distance)
        total_gap = left_distance + right_distance

        # Build per-peak records
        bin_size = self.bin_size
        group_width = self.group_width_da
        frag_type = metadata.get("frag_type", "unknown")

        gap_records = []
        for i in range(n_masked):
            nearest_da = nearest_distance[i]
            total_da = total_gap[i]
            gap_records.append(
                {
                    "masked_mz": float(masked_mz[i]),
                    "left_distance_da": float(left_distance[i]),
                    "right_distance_da": float(right_distance[i]),
                    "nearest_distance_da": float(nearest_da),
                    "total_gap_da": float(total_da),
                    "nearest_in_bins": float(nearest_da / bin_size) if not np.isnan(nearest_da) else np.nan,
                    "nearest_in_groups": float(nearest_da / group_width) if not np.isnan(nearest_da) else np.nan,
                    "total_gap_in_groups": float(total_da / group_width) if not np.isnan(total_da) else np.nan,
                    "mz_range": self._classify_mz_range(float(masked_mz[i])),
                    "has_left": bool(has_left[i]),
                    "has_right": bool(has_right[i]),
                    "frag_type": frag_type,
                }
            )

        return {
            "n_masked": n_masked,
            "n_unmasked": n_unmasked,
            "skip": False,
            "all_masked": False,
            "gap_records": gap_records,
        }

    def aggregate_results(self, per_spectrum_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate gap records from all spectra into summary statistics.

        Args:
            per_spectrum_results: List of per-spectrum results from analyze_spectrum().

        Returns:
            Aggregated results dict.
        """
        all_records: List[Dict] = []
        for result in per_spectrum_results:
            if result.get("skip") or result.get("all_masked"):
                continue
            all_records.extend(result.get("gap_records", []))

        if not all_records:
            logger.warning("No masking gap data to aggregate")
            self.results = {}
            return self.results

        gap_df = pd.DataFrame(all_records)

        # Overall statistics
        metrics = [
            "left_distance_da",
            "right_distance_da",
            "nearest_distance_da",
            "total_gap_da",
            "nearest_in_bins",
            "nearest_in_groups",
            "total_gap_in_groups",
        ]
        overall: Dict[str, Dict[str, float]] = {}
        for col in metrics:
            vals = gap_df[col].dropna()
            if len(vals) > 0:
                overall[col] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "median": float(vals.median()),
                    "p25": float(vals.quantile(0.25)),
                    "p75": float(vals.quantile(0.75)),
                    "p95": float(vals.quantile(0.95)),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                }

        # Stratified by m/z region
        stratified: Dict[str, Dict] = {}
        for region in self.mz_range_order:
            region_df = gap_df[gap_df["mz_range"] == region]
            if len(region_df) == 0:
                continue
            region_stats: Dict[str, Dict[str, float]] = {}
            for col in ["nearest_distance_da", "total_gap_da", "nearest_in_groups"]:
                vals = region_df[col].dropna()
                if len(vals) > 0:
                    region_stats[col] = {
                        "mean": float(vals.mean()),
                        "std": float(vals.std()),
                        "median": float(vals.median()),
                        "n": len(vals),
                    }
            stratified[region] = region_stats

        # Group coverage: fraction of masked peaks whose total gap (distance
        # between the two adjacent unmasked peaks) fits within N groups.
        # This measures the prediction search space, not just the nearest
        # neighbor distance.
        total_groups = gap_df["total_gap_in_groups"].dropna()
        group_thresholds = [1, 2, 3, 5, 10]
        group_coverage: Dict[str, float] = {}
        for g in group_thresholds:
            frac = float((total_groups <= g).mean()) if len(total_groups) > 0 else 0.0
            group_coverage[f"within_{g}_groups"] = frac

        # Boundary statistics
        n_total = len(gap_df)
        n_no_left = int((~gap_df["has_left"]).sum())
        n_no_right = int((~gap_df["has_right"]).sum())

        # Group size sweep: evaluate different bin_group_size values
        nearest_da = gap_df["nearest_distance_da"].dropna().values
        n_bins = int((self.config.model.get("max_mz", 2500.0) - self.config.model.get("min_mz", 50.0)) / self.bin_size)
        group_size_sweep = self._compute_group_size_sweep(nearest_da, n_bins)

        self.results = {
            "summary": overall,
            "stratified_by_mz_range": stratified,
            "group_coverage": group_coverage,
            "group_size_sweep": group_size_sweep,
            "boundary_stats": {
                "n_no_left_neighbor": n_no_left,
                "n_no_right_neighbor": n_no_right,
                "frac_no_left": n_no_left / max(n_total, 1),
                "frac_no_right": n_no_right / max(n_total, 1),
            },
            "n_total_masked_peaks": n_total,
            "raw_data": gap_df,
        }
        return self.results

    def _compute_group_size_sweep(self, nearest_da: np.ndarray, n_bins: int) -> List[Dict[str, Any]]:
        """Evaluate group coverage at different bin_group_size values.

        For each candidate group_size, computes:
        - group_width_da: how wide each group is in Daltons
        - n_groups: number of group classes the model must predict
        - n_offset_classes: number of offset classes within each group
        - frac_within_1_group: fraction of masked peaks with nearest unmasked
          within 1 group_width (i.e., group prediction is trivially constrained)
        - frac_within_2_groups: fraction within 2 group_widths

        Args:
            nearest_da: Array of nearest-unmasked distances in Da.
            n_bins: Total number of bins across the m/z range.

        Returns:
            List of dicts, one per candidate group_size.
        """
        candidates = [10, 25, 50, 75, 100, 150, 200, 500, 1000]
        sweep_results = []

        for gs in candidates:
            group_width = self.bin_size * gs
            n_groups = int(np.ceil(n_bins / gs))

            within_1 = float((nearest_da <= group_width).mean()) if len(nearest_da) > 0 else 0.0
            within_2 = float((nearest_da <= 2 * group_width).mean()) if len(nearest_da) > 0 else 0.0
            within_3 = float((nearest_da <= 3 * group_width).mean()) if len(nearest_da) > 0 else 0.0

            sweep_results.append(
                {
                    "bin_group_size": gs,
                    "group_width_da": group_width,
                    "n_groups": n_groups,
                    "n_offset_classes": gs,
                    "frac_within_1_group": within_1,
                    "frac_within_2_groups": within_2,
                    "frac_within_3_groups": within_3,
                }
            )

        return sweep_results

    def generate_visualizations(self) -> None:
        """Generate per-strategy gap visualisations (group size sweep only)."""
        if not self.results or "raw_data" not in self.results:
            return
        self.generate_group_size_sweep()

    def generate_group_size_sweep(self) -> None:
        """Generate group size trade-off figure.

        Shows how different bin_group_size values affect:
        - The fraction of masked peaks where the group prediction is trivial
          (nearest unmasked within 1 group_width)
        - The number of group vs offset classes
        """
        sweep = self.results.get("group_size_sweep", [])
        if not sweep:
            return

        group_sizes = [s["bin_group_size"] for s in sweep]
        group_widths = [s["group_width_da"] for s in sweep]
        n_groups_list = [s["n_groups"] for s in sweep]
        frac_1 = [s["frac_within_1_group"] * 100 for s in sweep]
        frac_2 = [s["frac_within_2_groups"] * 100 for s in sweep]
        frac_3 = [s["frac_within_3_groups"] * 100 for s in sweep]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Group Size Trade-off Analysis", fontsize=16, fontweight="bold")

        # Left panel: coverage vs group_width
        ax = axes[0]
        ax.plot(group_widths, frac_1, "o-", color="#2ecc71", linewidth=2, markersize=7, label="Within 1 group")
        ax.plot(group_widths, frac_2, "s-", color="#3498db", linewidth=2, markersize=7, label="Within 2 groups")
        ax.plot(group_widths, frac_3, "^-", color="#9b59b6", linewidth=2, markersize=7, label="Within 3 groups")

        # Mark current config
        current_width = self.group_width_da
        current_idx = None
        for i, gw in enumerate(group_widths):
            if abs(gw - current_width) < 0.001:
                current_idx = i
                break
        if current_idx is not None:
            ax.axvline(current_width, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
            ax.annotate(
                f"current\n({self.bin_group_size})",
                xy=(current_width, frac_1[current_idx]),
                xytext=(current_width + max(group_widths) * 0.05, frac_1[current_idx] + 5),
                fontsize=9,
                color="red",
                arrowprops={"arrowstyle": "->", "color": "red", "lw": 1.2},
            )

        # Annotate each point with group_size
        for i, gs in enumerate(group_sizes):
            ax.annotate(
                f"gs={gs}",
                xy=(group_widths[i], frac_1[i]),
                xytext=(0, -14),
                textcoords="offset points",
                fontsize=7,
                ha="center",
                color="gray",
            )

        ax.set_xlabel("Group Width (Da)")
        ax.set_ylabel("Fraction of Masked Peaks (%)")
        ax.set_title("Coverage: Peaks with Nearest Unmasked Within N Groups")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

        # Right panel: number of classes trade-off
        ax = axes[1]
        ax2 = ax.twinx()

        color_group = "#e74c3c"
        color_offset = "#3498db"

        l1 = ax.plot(group_widths, n_groups_list, "o-", color=color_group, linewidth=2, markersize=7, label="Group classes")
        l2 = ax2.plot(group_widths, group_sizes, "s-", color=color_offset, linewidth=2, markersize=7, label="Offset classes")

        ax.set_xlabel("Group Width (Da)")
        ax.set_ylabel("Number of Group Classes", color=color_group)
        ax.tick_params(axis="y", labelcolor=color_group)
        ax2.set_ylabel("Number of Offset Classes (= group_size)", color=color_offset)
        ax2.tick_params(axis="y", labelcolor=color_offset)

        # Mark current config
        if current_idx is not None:
            ax.axvline(current_width, color="red", linestyle=":", linewidth=1.5, alpha=0.7)

        ax.set_title("Classification Complexity: Group vs Offset Classes")
        lines = l1 + l2
        labels = [l.get_label() for l in lines]  # noqa: E741
        ax.legend(lines, labels, fontsize=9, loc="center right")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax2.set_yscale("log")

        plt.tight_layout()
        viz_path = self.output_dir / "group_size_tradeoff.png"
        fig.savefig(viz_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Group size trade-off visualization saved to: {viz_path}")

    def save_results(self) -> None:
        """No-op: gap results are saved via MaskingAnalyser's summary JSON."""

    def print_summary(self) -> None:
        """Print masking gap analysis summary to console."""
        if not self.results:
            return

        logger.info("=" * 80)
        logger.info("MASKING GAP ANALYSIS SUMMARY")
        logger.info("=" * 80)

        n_total = self.results.get("n_total_masked_peaks", 0)
        logger.info(f"Total masked peaks analyzed: {n_total:,d}")
        logger.info(f"Binning: bin_size={self.bin_size} Da, group_size={self.bin_group_size}, group_width={self.group_width_da:.1f} Da")

        overall = self.results.get("summary", {})

        if "nearest_distance_da" in overall:
            s = overall["nearest_distance_da"]
            logger.info(
                f"\nNearest Unmasked Distance:"
                f"\n  Mean: {s['mean']:.4f} Da | Median: {s['median']:.4f} Da | Std: {s['std']:.4f} Da"
                f"\n  P25: {s['p25']:.4f} Da | P75: {s['p75']:.4f} Da | P95: {s['p95']:.4f} Da"
            )

        if "total_gap_da" in overall:
            s = overall["total_gap_da"]
            logger.info(f"\nTotal Gap (Left + Right):\n  Mean: {s['mean']:.4f} Da | Median: {s['median']:.4f} Da | Std: {s['std']:.4f} Da")

        if "nearest_in_groups" in overall:
            s = overall["nearest_in_groups"]
            logger.info(
                f"\nNearest Distance in Groups ({self.group_width_da:.1f} Da each):"
                f"\n  Mean: {s['mean']:.2f} groups | Median: {s['median']:.2f} groups"
            )

        gc = self.results.get("group_coverage", {})
        if gc:
            logger.info("\nGroup Coverage (total gap between adjacent unmasked peaks):")
            for k in sorted(gc.keys(), key=lambda k: int(k.split("_")[1])):
                logger.info(f"  {k}: {gc[k] * 100:.1f}%")

        # Per-region breakdown
        stratified = self.results.get("stratified_by_mz_range", {})
        if stratified:
            logger.info("\nNearest Distance by m/z Region:")
            for region in self.mz_range_order:
                if region in stratified and "nearest_distance_da" in stratified[region]:
                    s = stratified[region]["nearest_distance_da"]
                    logger.info(f"  {region}: mean={s['mean']:.4f} Da, std={s['std']:.4f} Da, median={s['median']:.4f} Da (n={s['n']:,d})")

        boundary = self.results.get("boundary_stats", {})
        if boundary:
            logger.info(
                f"\nBoundary Peaks:"
                f"\n  No left neighbor: {boundary['n_no_left_neighbor']} ({boundary['frac_no_left'] * 100:.1f}%)"
                f"\n  No right neighbor: {boundary['n_no_right_neighbor']} ({boundary['frac_no_right'] * 100:.1f}%)"
            )

        # Group size sweep
        sweep = self.results.get("group_size_sweep", [])
        if sweep:
            logger.info("\nGroup Size Sweep (trade-off analysis):")
            logger.info(f"  {'group_size':>10} {'width(Da)':>10} {'n_groups':>10} {'<=1 grp':>10} {'<=2 grp':>10} {'<=3 grp':>10}")
            logger.info("  %s", "-" * 62)
            for s in sweep:
                marker = " <-- current" if s["bin_group_size"] == self.bin_group_size else ""
                logger.info(
                    f"  {s['bin_group_size']:>10} {s['group_width_da']:>10.2f} {s['n_groups']:>10} "
                    f"{s['frac_within_1_group'] * 100:>9.1f}% {s['frac_within_2_groups'] * 100:>9.1f}% "
                    f"{s['frac_within_3_groups'] * 100:>9.1f}%{marker}"
                )

        logger.info("=" * 80)
