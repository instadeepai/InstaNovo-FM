#!/usr/bin/env python
"""Metadata Analyzer for Foundation Model Training Data.

Analyzes dataset metadata and search data to understand:
1. Categorical variable distributions (acquisition type, fragmentation, etc.)
2. Numerical variable distributions (precursor m/z, charge, RT, etc.)
3. Cross-tabulations and correlations
4. Data coverage and completeness

Outputs separate visualizations and statistics for metadata analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from instanovo.__init__ import console
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger


class MetadataAnalyser:
    """Analyzer for dataset metadata and search data."""

    # Columns to completely exclude from analysis (spectrum data, not metadata)
    EXCLUDE_COLUMNS = {
        "intensity_array",
        "mz_array",
        "scan",
        "header",
        "sequence",
        "unmodified_peptide",
    }

    # High-value columns that should always be visualized (if categorical)
    HIGH_VALUE_CATEGORICAL = {
        "acquisition",
        "frag_type",
        "search_acquisition",
        "search_detector",
        "search_fragmentation",
        "search_instrument",
        "search_enzyme",
        "search_organism",
        "search_quant",
        "search_modifications",
        "search_project",
    }

    # Columns to exclude from detailed JSON export (too verbose/not actionable)
    EXCLUDE_FROM_JSON = {
        "intensity_array",
        "mz_array",
        "scan",
        "header",
        "sequence",
        "unmodified_peptide",
        "experiment_name",
        "usi",
        "search_file path",
        "search_workflow",
        "protein",
    }

    # Cardinality thresholds for visualization
    MAX_UNIQUE_FOR_PLOT = 50  # Don't plot if more unique values
    MAX_UNIQUE_FOR_DETAILED_STATS = 100  # Only basic stats if more

    # High-value numerical columns (core MS/MS properties) - detailed analysis
    HIGH_VALUE_NUMERICAL = {
        "precursor_charge",
        "precursor_mz",
        "precursor_mass",
        "peptide_observed_mz",
        "peptide_calc_mz",
        "delta_mass",
        "collision_energy",
        "retention_time",
    }

    # Note: precursor_mass is already in HIGH_VALUE_NUMERICAL - it's a core MS/MS property

    # Medium-value numerical columns (search metrics) - standard analysis
    MEDIUM_VALUE_NUMERICAL = {
        "hyperscore",
        "nextscore",
        "expectation",
        "probability",
    }

    # Low-value numerical columns (technical parameters) - minimal analysis
    LOW_VALUE_NUMERICAL = {
        "isolation_target",
        "upper_offset",
        "lower_offset",
        "precursor_intensity",
        "auc_intensity",
        "scale_factor",
    }

    # Exclude from numerical analysis (not metadata)
    EXCLUDE_NUMERICAL = {
        "index",  # Row identifier, not metadata
    }

    # QC thresholds for warnings
    QC_THRESHOLDS: dict[str, Any] = {
        "delta_mass": {"abs_mean": 0.5, "std": 1.0},  # Mass error should be small
        "precursor_charge": {"min": 0, "max": 6},  # DIA may have charge=0
        "probability": {"min": 0.8},  # Should be high confidence
    }

    def __init__(self, output_dir: Path) -> None:
        """Initialize metadata analyzer.

        Args:
            output_dir: Directory to save metadata analysis results
        """
        self.output_dir = output_dir / "metadata_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_results: dict[str, Any] = {
            "categorical_summary": {},
            "numerical_summary": {},
            "high_cardinality_summary": {},  # For columns with too many unique values
            "qc_warnings": [],  # Quality control warnings
        }

        # Set plotting style
        plt.style.use("default")
        sns.set_palette("husl")

    def analyze_metadata(
        self,
        metadata_dict: Dict[str, List[Any]],
        metadata_columns: List[str],
    ) -> Dict[str, Any]:
        """Analyze metadata columns from dataset with intelligent filtering.

        Args:
            metadata_dict: Dictionary with metadata column names as keys and lists as values
            metadata_columns: List of metadata column names to analyze

        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing {len(metadata_columns)} metadata columns...")

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(metadata_dict)

        # Filter out excluded columns
        filtered_columns = [col for col in metadata_columns if col not in self.EXCLUDE_COLUMNS]
        excluded_count = len(metadata_columns) - len(filtered_columns)
        if excluded_count > 0:
            logger.info(f"Excluded {excluded_count} columns (spectrum data/sequences): {self.EXCLUDE_COLUMNS & set(metadata_columns)}")

        # Separate categorical and numerical columns
        categorical_cols = []
        numerical_cols = []

        for col in filtered_columns:
            if col not in df.columns:
                continue

            # Infer type
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue

            # Check if numerical - try converting and see if it succeeds
            try:
                # Attempt to convert to numeric, with errors='coerce' to handle mixed types
                numeric_converted = pd.to_numeric(non_null, errors="coerce")
                # If most values (>80%) successfully converted to numeric, treat as numerical
                # Use high threshold to avoid treating mixed-type columns as numerical
                non_na_after_conversion = numeric_converted.notna().sum()
                conversion_success_rate = non_na_after_conversion / len(non_null)

                if conversion_success_rate > 0.8 and col not in self.EXCLUDE_NUMERICAL:
                    numerical_cols.append(col)
                else:
                    categorical_cols.append(col)
            except (ValueError, TypeError):
                categorical_cols.append(col)

        logger.info(f"Found {len(categorical_cols)} categorical and {len(numerical_cols)} numerical columns")

        # Analyze categorical columns with cardinality awareness
        for col in categorical_cols:
            self._analyze_categorical(df, col)

        # Analyze numerical columns
        for col in numerical_cols:
            self._analyze_numerical(df, col)

        # Log summary of categorization
        if self.metadata_results["high_cardinality_summary"]:
            logger.info(f"Identified {len(self.metadata_results['high_cardinality_summary'])} high-cardinality columns (stats only, no plots)")

        return self.metadata_results

    def _analyze_categorical(self, df: pd.DataFrame, col: str) -> None:
        """Analyze single categorical column with cardinality-aware handling."""
        logger.debug(f"Analyzing categorical column: {col}")

        # Convert column to string to handle unhashable types (lists, dicts, etc.)
        # This prevents pandas value_counts() errors with complex data types
        col_data = df[col].apply(lambda x: str(x) if x is not None and not pd.isna(x) else x)

        # Count nulls before filling
        total_count = len(df)
        null_count = col_data.isna().sum()

        # For high-value categorical columns, convert null/None/empty to "unknown"
        # so they appear as a visible category in distributions and plots
        is_high_value = col in self.HIGH_VALUE_CATEGORICAL
        if is_high_value or null_count > 0:
            col_data = col_data.fillna("unknown")
            # Also treat empty strings and "None" literals as unknown
            col_data = col_data.replace({"": "unknown", "None": "unknown", "none": "unknown"})

        # Get value counts (nulls are now labelled as "unknown")
        value_counts = col_data.value_counts()
        unique_count = len(value_counts)

        # Determine if this is high or low cardinality
        is_high_cardinality = unique_count > self.MAX_UNIQUE_FOR_PLOT

        # Basic stats for all categorical columns
        base_stats: dict[str, Any] = {
            "unique_values": unique_count,
            "total_count": total_count,
            "null_count": int(null_count),
            "null_percentage": float(null_count / total_count * 100),
        }

        # Decide on categorization
        if is_high_cardinality and not is_high_value:
            # High cardinality, low value - minimal stats only
            logger.debug(f"  → High cardinality ({unique_count} unique) - stats only")
            self.metadata_results["high_cardinality_summary"][col] = {
                **base_stats,
                "top_values": value_counts.head(5).index.tolist(),
                "top_counts": value_counts.head(5).values.tolist(),
                "cardinality": "high",
                "visualization": "none",
            }
        else:
            # Low-medium cardinality OR high-value column - full analysis
            logger.debug(f"  → {'High-value' if is_high_value else 'Low cardinality'} ({unique_count} unique) - full analysis")

            # Store top N values (more for low cardinality, fewer for high)
            top_n = min(20, unique_count) if unique_count <= self.MAX_UNIQUE_FOR_PLOT else 15

            self.metadata_results["categorical_summary"][col] = {
                **base_stats,
                "top_values": value_counts.head(top_n).index.tolist(),
                "top_counts": value_counts.head(top_n).values.tolist(),
                "most_common": value_counts.index[0] if len(value_counts) > 0 else None,
                "most_common_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "cardinality": "low" if unique_count <= 20 else "medium",
                "visualization": "enabled",
                "is_high_value": is_high_value,
            }

    def _analyze_numerical(self, df: pd.DataFrame, col: str) -> None:
        """Analyze single numerical column with proteomics-aware categorization."""
        logger.debug(f"Analyzing numerical column: {col}")

        # Convert to numeric, coercing errors to NaN
        numeric_data = pd.to_numeric(df[col], errors="coerce")

        # Check if we have enough valid data
        valid_count = numeric_data.notna().sum()
        if valid_count == 0:
            logger.warning(f"Column {col} has no valid numerical data after conversion, skipping")
            return

        # Drop NaN and ensure float64 dtype to avoid pandas treating as categorical
        numeric_clean = numeric_data.dropna().astype(float)

        # Calculate statistics on clean numeric data
        stats = numeric_clean.describe()

        # Determine value category
        if col in self.HIGH_VALUE_NUMERICAL:
            value_category = "high"
            visualization = "detailed"
        elif col in self.MEDIUM_VALUE_NUMERICAL:
            value_category = "medium"
            visualization = "standard"
        elif col in self.LOW_VALUE_NUMERICAL:
            value_category = "low"
            visualization = "minimal"
        else:
            value_category = "unknown"
            visualization = "standard"

        logger.debug(f"  → {value_category.upper()} value ({visualization} visualization)")

        # Store summary - safely access stats with .get() and defaults
        summary: dict[str, Any] = {
            "count": int(stats.get("count", 0)),
            "mean": float(stats.get("mean", 0.0)),
            "std": float(stats.get("std", 0.0)),
            "min": float(stats.get("min", 0.0)),
            "q25": float(stats.get("25%", 0.0)),
            "median": float(stats.get("50%", 0.0)),
            "q75": float(stats.get("75%", 0.0)),
            "max": float(stats.get("max", 0.0)),
            "null_count": int(numeric_data.isna().sum()),
            "null_percentage": float(numeric_data.isna().sum() / len(df) * 100),
            "value_category": value_category,
            "visualization": visualization,
        }

        # Store actual data for histogram generation (only for visualized columns)
        if visualization in ["detailed", "standard"]:
            summary["histogram_data"] = numeric_data.dropna().values.tolist()

        self.metadata_results["numerical_summary"][col] = summary

        # QC checks for specific columns (only if we have valid stats)
        if valid_count > 0:
            self._check_qc(col, numeric_data, stats)

    def _check_qc(self, col: str, data: pd.Series, stats: pd.Series) -> None:
        """Perform QC checks on numerical columns and generate warnings."""
        if col not in self.QC_THRESHOLDS:
            return

        thresholds = self.QC_THRESHOLDS[col]

        # Check delta_mass (mass accuracy)
        if col == "delta_mass":
            abs_mean = abs(stats["mean"])
            if abs_mean > thresholds.get("abs_mean", float("inf")):
                self.metadata_results["qc_warnings"].append(
                    {
                        "column": col,
                        "type": "mass_accuracy",
                        "severity": "warning",
                        "message": f"High mean mass error: {stats['mean']:.3f} Da (threshold: ±{thresholds['abs_mean']} Da)",
                        "value": float(stats["mean"]),
                    }
                )

            if stats["std"] > thresholds.get("std", float("inf")):
                self.metadata_results["qc_warnings"].append(
                    {
                        "column": col,
                        "type": "mass_precision",
                        "severity": "warning",
                        "message": f"High mass error variability: {stats['std']:.3f} Da (threshold: {thresholds['std']} Da)",
                        "value": float(stats["std"]),
                    }
                )

        # Check precursor_charge
        elif col == "precursor_charge":
            if stats["min"] < thresholds.get("min", 0):
                self.metadata_results["qc_warnings"].append(
                    {
                        "column": col,
                        "type": "charge_range",
                        "severity": "error",
                        "message": f"Unusual minimum charge: {int(stats['min'])} (expected: {thresholds['min']}+)",
                        "value": int(stats["min"]),
                    }
                )

            if stats["max"] > thresholds.get("max", float("inf")):
                self.metadata_results["qc_warnings"].append(
                    {
                        "column": col,
                        "type": "charge_range",
                        "severity": "warning",
                        "message": f"High maximum charge: {int(stats['max'])} (typical: <{thresholds['max']})",
                        "value": int(stats["max"]),
                    }
                )

        # Check probability
        elif col == "probability":
            if stats["min"] < thresholds.get("min", 0):
                self.metadata_results["qc_warnings"].append(
                    {
                        "column": col,
                        "type": "confidence",
                        "severity": "info",
                        "message": f"Low minimum probability: {stats['min']:.3f} (threshold: {thresholds['min']})",
                        "value": float(stats["min"]),
                    }
                )

    def generate_visualizations(self) -> None:
        """Generate comprehensive metadata visualizations."""
        logger.info("Generating metadata visualizations...")

        if self.metadata_results["categorical_summary"]:
            self._generate_categorical_visualizations()

        if self.metadata_results["numerical_summary"]:
            self._generate_numerical_visualizations()

        self._generate_summary_visualization()

    def _generate_categorical_visualizations(self) -> None:
        """Generate visualizations for categorical variables."""
        cat_summary = self.metadata_results["categorical_summary"]

        # Determine grid size
        n_cols = len(cat_summary)
        if n_cols == 0:
            return

        n_rows = (n_cols + 2) // 3  # 3 columns per row
        n_rows = max(n_rows, 1)

        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows), squeeze=False)

        fig.suptitle("Categorical Metadata Distribution", fontsize=16, y=0.995)

        for idx, (col_name, stats) in enumerate(cat_summary.items()):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]

            # Plot top values
            top_values = stats["top_values"][:10]  # Top 10
            top_counts = stats["top_counts"][:10]

            if top_values and top_counts:
                # Horizontal bar chart
                y_pos = np.arange(len(top_values))
                ax.barh(y_pos, top_counts, alpha=0.7)
                ax.set_yticks(y_pos)
                ax.set_yticklabels([str(v)[:30] for v in top_values])  # Truncate long labels
                ax.set_xlabel("Count")
                ax.set_title(f"{col_name}\n({stats['unique_values']} unique values)")
                ax.grid(True, alpha=0.3, axis="x")
                ax.invert_yaxis()  # Highest count at top
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(col_name)

        # Hide empty subplots
        for idx in range(n_cols, n_rows * 3):
            row = idx // 3
            col = idx % 3
            axes[row, col].axis("off")

        plt.tight_layout()

        # Save
        viz_path = self.output_dir / "categorical_distributions.png"
        fig.savefig(viz_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Categorical visualizations saved to: {viz_path}")

    def _generate_numerical_visualizations(self) -> None:
        """Generate histogram visualizations for numerical variables (category-aware)."""
        num_summary = self.metadata_results["numerical_summary"]

        # Filter for high and medium value columns only (exclude low-value technical params)
        visualize_cols = {col: stats for col, stats in num_summary.items() if stats.get("visualization") in ["detailed", "standard"]}

        if not visualize_cols:
            logger.info("No numerical columns selected for visualization")
            return

        n_cols = len(visualize_cols)
        logger.info(f"Generating histogram visualizations for {n_cols} numerical columns (high/medium value)")

        # We need the actual data to create histograms - store it during analysis
        # For now, create informative box plots as placeholders
        # TODO: In next iteration, pass actual data arrays to enable true histograms

        # Determine grid size
        n_rows = (n_cols + 2) // 3
        n_rows = max(n_rows, 1)

        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows), squeeze=False)

        fig.suptitle("Numerical Metadata Distribution (High & Medium Value)", fontsize=16, y=0.995)

        for idx, (col_name, stats) in enumerate(visualize_cols.items()):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]

            # Color by value category
            if stats.get("value_category") == "high":
                color = "#ff6b6b"  # Red for high-value (no emoji in plots)
                edge_color = "#c92a2a"
                category_label = "Core MS/MS"
            else:
                color = "#4dabf7"  # Blue for medium-value
                edge_color = "#1971c2"
                category_label = "Search Metric"

            # Get histogram data from stored values if available
            if "histogram_data" in stats:
                # Use actual histogram data
                hist_data = stats["histogram_data"]
                ax.hist(hist_data, bins=30, color=color, alpha=0.7, edgecolor=edge_color, linewidth=0.5)

                # Add KDE overlay if enough data points
                if len(hist_data) > 10:
                    from scipy import stats as scipy_stats

                    try:
                        kde = scipy_stats.gaussian_kde(hist_data)
                        x_range = np.linspace(stats["min"], stats["max"], 200)
                        kde_values = kde(x_range)
                        # Scale KDE to histogram height
                        kde_values * (len(hist_data) * (stats["max"] - stats["min"]) / 30)
                        ax2 = ax.twinx()
                        ax2.plot(
                            x_range, kde_values * len(hist_data) * (stats["max"] - stats["min"]) / 30, color=edge_color, linewidth=2, label="KDE"
                        )
                        ax2.set_yticks([])
                    except Exception:
                        pass  # Skip KDE if it fails
            else:
                # Fallback: Create approximate histogram from quantiles
                # This is a limitation - we'll improve this next
                [stats["min"], stats["q25"], stats["median"], stats["q75"], stats["max"]]
                # Estimate counts (assuming normal distribution)
                counts = [stats["count"] * 0.25, stats["count"] * 0.25, stats["count"] * 0.25, stats["count"] * 0.25]
                ax.bar(range(4), counts, color=color, alpha=0.7, edgecolor=edge_color, width=0.8)
                ax.set_xticks(range(4))
                ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"], fontsize=8)
                ax.set_ylabel("Approx. Count", fontsize=9)

                # Add note about approximation
                ax.text(0.5, 0.95, "(Approximate dist.)", transform=ax.transAxes, ha="center", va="top", fontsize=7, style="italic", color="gray")

            # Title with category and stats
            title_str = f"{col_name} [{category_label}]\n"
            title_str += f"μ={stats['mean']:.2f}, σ={stats['std']:.2f}"
            if col_name == "delta_mass":
                title_str += f" | Median={stats['median']:.3f}"  # Important for mass accuracy
            ax.set_title(title_str, fontsize=9)
            ax.set_xlabel("Value", fontsize=9)
            if "histogram_data" not in stats:
                ax.set_ylabel("Approx. Count", fontsize=9)
            else:
                ax.set_ylabel("Count", fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")

            # Add reference line for delta_mass (should be near 0)
            if col_name == "delta_mass":
                ax.axvline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Target (0 Da)")
                ax.legend(fontsize=7)

        # Hide empty subplots
        for idx in range(n_cols, n_rows * 3):
            row = idx // 3
            col = idx % 3
            axes[row, col].axis("off")

        plt.tight_layout()

        # Save
        viz_path = self.output_dir / "numerical_distributions.png"
        fig.savefig(viz_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Numerical histogram visualizations saved to: {viz_path}")

    def _generate_summary_visualization(self) -> None:
        """Generate summary statistics visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Metadata Analysis Summary", fontsize=16)

        # 1. Categorical variable summary
        ax1 = axes[0]
        cat_summary = self.metadata_results["categorical_summary"]
        if cat_summary:
            col_names = list(cat_summary.keys())[:10]  # Top 10
            unique_counts = [cat_summary[col]["unique_values"] for col in col_names]

            y_pos = np.arange(len(col_names))
            ax1.barh(y_pos, unique_counts, alpha=0.7, color="skyblue")
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels([str(c)[:20] for c in col_names])
            ax1.set_xlabel("Unique Values")
            ax1.set_title("Categorical Variables\n(Cardinality)")
            ax1.grid(True, alpha=0.3, axis="x")
            ax1.invert_yaxis()
        else:
            ax1.text(0.5, 0.5, "No categorical data", ha="center", va="center", transform=ax1.transAxes)
            ax1.set_title("Categorical Variables")

        # 2. Numerical variable summary
        ax2 = axes[1]
        num_summary = self.metadata_results["numerical_summary"]
        if num_summary:
            col_names = list(num_summary.keys())
            means = [num_summary[col]["mean"] for col in col_names]
            stds = [num_summary[col]["std"] for col in col_names]

            x_pos = np.arange(len(col_names))
            ax2.bar(x_pos, means, yerr=stds, alpha=0.7, color="lightcoral", capsize=5)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([str(c)[:15] for c in col_names], rotation=45, ha="right")
            ax2.set_ylabel("Value")
            ax2.set_title("Numerical Variables\n(Mean ± Std)")
            ax2.grid(True, alpha=0.3, axis="y")
        else:
            ax2.text(0.5, 0.5, "No numerical data", ha="center", va="center", transform=ax2.transAxes)
            ax2.set_title("Numerical Variables")

        # 3. Data completeness
        ax3 = axes[2]
        all_cols: dict[str, Any] = {}
        all_cols.update(cat_summary)
        all_cols.update(num_summary)

        if all_cols:
            completeness = []
            labels = []
            for col_name, stats in all_cols.items():
                if "null_percentage" in stats:
                    completeness.append(100 - stats["null_percentage"])
                    labels.append(str(col_name)[:20])

            if completeness:
                # Sort by completeness
                sorted_data = sorted(zip(completeness, labels, strict=False))
                completeness, labels = zip(*sorted_data, strict=False)  # type: ignore[assignment]

                y_pos = np.arange(len(labels))
                colors = ["green" if c >= 90 else "orange" if c >= 50 else "red" for c in completeness]
                ax3.barh(y_pos, completeness, alpha=0.7, color=colors)
                ax3.set_yticks(y_pos)
                ax3.set_yticklabels(labels)
                ax3.set_xlabel("Completeness (%)")
                ax3.set_title("Data Completeness")
                ax3.set_xlim([0, 100])
                ax3.axvline(90, color="green", linestyle="--", alpha=0.5, label="90%")
                ax3.axvline(50, color="orange", linestyle="--", alpha=0.5, label="50%")
                ax3.grid(True, alpha=0.3, axis="x")
                ax3.legend()
                ax3.invert_yaxis()
        else:
            ax3.text(0.5, 0.5, "No completeness data", ha="center", va="center", transform=ax3.transAxes)
            ax3.set_title("Data Completeness")

        plt.tight_layout()

        # Save
        viz_path = self.output_dir / "metadata_summary.png"
        fig.savefig(viz_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Summary visualization saved to: {viz_path}")

    def save_results(self) -> None:
        """Save metadata analysis results to JSON (filtered for relevant data only)."""
        json_path = self.output_dir / "metadata_analysis.json"

        # Create filtered version for JSON export (exclude bloated columns)
        filtered_results: dict[str, Any] = {
            "categorical_summary": {
                col: stats for col, stats in self.metadata_results["categorical_summary"].items() if col not in self.EXCLUDE_FROM_JSON
            },
            "numerical_summary": {
                col: {k: v for k, v in stats.items() if k != "histogram_data"}  # Exclude raw histogram data
                for col, stats in self.metadata_results["numerical_summary"].items()
            },
            "high_cardinality_summary": {
                col: stats for col, stats in self.metadata_results["high_cardinality_summary"].items() if col not in self.EXCLUDE_FROM_JSON
            },
            "qc_warnings": self.metadata_results["qc_warnings"].copy(),
        }

        # Log what was excluded
        excluded_cat = len(self.metadata_results["categorical_summary"]) - len(filtered_results["categorical_summary"])
        excluded_high = len(self.metadata_results["high_cardinality_summary"]) - len(filtered_results["high_cardinality_summary"])
        if excluded_cat + excluded_high > 0:
            logger.info(f"Excluded {excluded_cat + excluded_high} high-verbosity columns from JSON export")

        with open(json_path, "w") as f:
            json.dump(filtered_results, f, indent=2)

        logger.info(f"Metadata analysis results saved to: {json_path}")

        # Also save as CSV for easy inspection
        self._save_as_csv()

    def _save_as_csv(self) -> None:
        """Save metadata summaries as CSV files."""
        # Categorical summary (low-medium cardinality)
        if self.metadata_results["categorical_summary"]:
            cat_records = []
            for col_name, stats in self.metadata_results["categorical_summary"].items():
                cat_records.append(
                    {
                        "column": col_name,
                        "cardinality": stats.get("cardinality", "unknown"),
                        "unique_values": stats["unique_values"],
                        "total_count": stats["total_count"],
                        "null_count": stats["null_count"],
                        "null_percentage": stats["null_percentage"],
                        "most_common": stats.get("most_common"),
                        "most_common_count": stats.get("most_common_count", 0),
                        "is_high_value": stats.get("is_high_value", False),
                    }
                )

            df_cat = pd.DataFrame(cat_records)
            csv_path = self.output_dir / "categorical_summary.csv"
            df_cat.to_csv(csv_path, index=False)
            logger.info(f"Categorical summary saved to: {csv_path}")

        # High cardinality summary (stats only, no plots)
        if self.metadata_results["high_cardinality_summary"]:
            high_card_records = []
            for col_name, stats in self.metadata_results["high_cardinality_summary"].items():
                # Get top value info
                top_val = stats["top_values"][0] if stats["top_values"] else None
                top_count = int(stats["top_counts"][0]) if stats["top_counts"] else 0

                high_card_records.append(
                    {
                        "column": col_name,
                        "cardinality": "high",
                        "unique_values": stats["unique_values"],
                        "total_count": stats["total_count"],
                        "null_count": stats["null_count"],
                        "null_percentage": stats["null_percentage"],
                        "most_common": top_val,
                        "most_common_count": top_count,
                        "visualization": "disabled",
                    }
                )

            df_high = pd.DataFrame(high_card_records)
            csv_path = self.output_dir / "high_cardinality_summary.csv"
            df_high.to_csv(csv_path, index=False)
            logger.info(f"High cardinality summary saved to: {csv_path}")

        # Numerical summary
        if self.metadata_results["numerical_summary"]:
            num_records = []
            for col_name, stats in self.metadata_results["numerical_summary"].items():
                num_records.append(
                    {
                        "column": col_name,
                        "value_category": stats.get("value_category", "unknown"),
                        "visualization": stats.get("visualization", "standard"),
                        "count": stats["count"],
                        "mean": stats["mean"],
                        "std": stats["std"],
                        "min": stats["min"],
                        "q25": stats["q25"],
                        "median": stats["median"],
                        "q75": stats["q75"],
                        "max": stats["max"],
                        "null_count": stats["null_count"],
                        "null_percentage": stats["null_percentage"],
                    }
                )

            df_num = pd.DataFrame(num_records)
            csv_path = self.output_dir / "numerical_summary.csv"
            df_num.to_csv(csv_path, index=False)
            logger.info(f"Numerical summary saved to: {csv_path}")

        # QC warnings
        if self.metadata_results["qc_warnings"]:
            qc_records = []
            for warning in self.metadata_results["qc_warnings"]:
                qc_records.append(
                    {
                        "column": warning["column"],
                        "type": warning["type"],
                        "severity": warning["severity"],
                        "message": warning["message"],
                        "value": warning["value"],
                    }
                )

            df_qc = pd.DataFrame(qc_records)
            csv_path = self.output_dir / "qc_warnings.csv"
            df_qc.to_csv(csv_path, index=False)
            logger.info(f"QC warnings saved to: {csv_path}")

    def print_summary(self) -> None:
        """Print summary to console."""
        print("\n" + "=" * 80)  # noqa: T201
        print("METADATA ANALYSIS SUMMARY")  # noqa: T201
        print("=" * 80)  # noqa: T201

        cat_summary = self.metadata_results["categorical_summary"]
        num_summary = self.metadata_results["numerical_summary"]
        high_card_summary = self.metadata_results["high_cardinality_summary"]

        if cat_summary:
            # Separate high-value from regular categorical
            high_value = {k: v for k, v in cat_summary.items() if v.get("is_high_value", False)}
            regular = {k: v for k, v in cat_summary.items() if not v.get("is_high_value", False)}

            if high_value:
                print(f"\n🎯 High-Value Categorical Variables ({len(high_value)}) - Plotted:")  # noqa: T201
                for col_name, stats in list(high_value.items())[:10]:
                    print(f"  • {col_name}:")  # noqa: T201
                    print(f"      Unique values: {stats['unique_values']} ({stats.get('cardinality', 'unknown')} cardinality)")  # noqa: T201
                    print(f"      Most common: {stats.get('most_common')} ({stats.get('most_common_count', 0)} occurrences)")  # noqa: T201
                    print(f"      Completeness: {100 - stats['null_percentage']:.1f}%")  # noqa: T201

            if regular:
                print(f"\nCategorical Variables ({len(regular)}) - Plotted:")  # noqa: T201
                for col_name, stats in list(regular.items())[:8]:
                    print(f"  • {col_name}:")  # noqa: T201
                    print(f"      Unique values: {stats['unique_values']}")  # noqa: T201
                    print(f"      Most common: {stats.get('most_common')} ({stats.get('most_common_count', 0)} occurrences)")  # noqa: T201

        if high_card_summary:
            print(f"\n📊 High-Cardinality Variables ({len(high_card_summary)}) - Stats Only:")  # noqa: T201
            for col_name, stats in list(high_card_summary.items())[:5]:
                print(f"  • {col_name}: {stats['unique_values']:,} unique values")  # noqa: T201

        if num_summary:
            # Categorize numerical columns
            high_value_num = {k: v for k, v in num_summary.items() if v.get("value_category") == "high"}
            medium_value_num = {k: v for k, v in num_summary.items() if v.get("value_category") == "medium"}
            low_value_num = {k: v for k, v in num_summary.items() if v.get("value_category") == "low"}

            if high_value_num:
                print(f"\n🔬 High-Value Numerical (Core MS/MS) ({len(high_value_num)}) - Plotted:")  # noqa: T201
                for col_name, stats in list(high_value_num.items())[:7]:
                    print(f"  • {col_name}:")  # noqa: T201
                    print(f"      Mean ± Std: {stats['mean']:.2f} ± {stats['std']:.2f}")  # noqa: T201
                    print(f"      Range: [{stats['min']:.2f}, {stats['max']:.2f}]")  # noqa: T201

            if medium_value_num:
                print(f"\n📊 Medium-Value Numerical (Search Metrics) ({len(medium_value_num)}) - Plotted:")  # noqa: T201
                for col_name, stats in list(medium_value_num.items())[:5]:
                    print(f"  • {col_name}: μ={stats['mean']:.2f}, σ={stats['std']:.2f}")  # noqa: T201

            if low_value_num:
                print(f"\n🔧 Low-Value Numerical (Technical) ({len(low_value_num)}) - Stats Only:")  # noqa: T201
                print(f"    {', '.join(list(low_value_num.keys())[:10])}")  # noqa: T201

        # QC Warnings
        if self.metadata_results["qc_warnings"]:
            print(f"\n⚠️  QC Warnings ({len(self.metadata_results['qc_warnings'])}):")  # noqa: T201
            for warning in self.metadata_results["qc_warnings"][:5]:
                severity_icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(warning["severity"], "•")
                print(f"  {severity_icon} {warning['message']}")  # noqa: T201

        print(f"\nResults saved to: {self.output_dir}")  # noqa: T201
        print("=" * 80)  # noqa: T201
