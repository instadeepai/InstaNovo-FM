"""Confidence-Signal Analysis Task for Foundation Model Embeddings.

This task evaluates how well the model's per-peak confidence scores align with
theoretical signal peaks in mass spectra. It helps answer:
1. Does the model assign higher confidence to real signal peaks vs. noise?
2. Is the model learning meaningful spectral structure?
3. Can confidence scores be used for quality control and explainability?

The task:
- Generates theoretical fragment ions for each peptide using PyOpenMS
- Matches experimental peaks to theoretical ions
- Compares confidence scores for theoretical vs. non-theoretical peaks
- Produces separation metrics (AUROC, precision-recall, KS statistic)
- Creates visualizations (distributions, ROC curves, calibration plots)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from tqdm import tqdm

from instanovo.__init__ import console
from instanovo_fm.eval.embed_eval_tasks import BaseTask
from instanovo_fm.utils.ion_visualization import (
    CATEGORY_COLORS,
    TEXT_COLORS,
    categorize_ion,
    format_annotation_display,
)
from instanovo_fm.utils.peak_classification import (
    extract_fragment_position,
    extract_ion_type,
)
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None

# Optional imports for metrics
try:
    from scipy.stats import ks_2samp, spearmanr
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ConfidenceSignalAnalysisTask(BaseTask):
    """Analyze alignment between model confidence and theoretical signal peaks.

    This task evaluates whether the foundation model has learned to recognize
    real peptide fragment signals by comparing per-peak confidence scores against
    theoretical ion annotations.

    Key metrics:
    - AUROC: How well confidence separates signal from noise
    - Average Precision: Precision-recall performance
    - KS statistic: Distribution separation between signal/noise confidence
    - Mean confidence gap: Average difference in confidence (signal - noise)

    Outputs:
    - JSON summary with all metrics
    - Confidence distribution plots (signal vs. noise)
    - ROC and precision-recall curves
    - Optional calibration/reliability plots
    """

    def __init__(
        self,
        output_dir: str = "./confidence_signal_analysis",
        max_samples: Optional[int] = None,
        create_plots: bool = True,
        max_individual_plots: int = 30,  # Maximum number of individual spectrum plots to create
        min_backbone_coverage: float = 0.0,  # Backbone coverage quality gate (0 = disabled)
        min_fragment_groups: int = 0,  # Fragment group quality gate (0 = disabled)
        **kwargs: Any,
    ) -> None:
        """Initialize the Confidence-Signal Analysis Task.

        Note: Theoretical spectrum generation parameters are now configured globally
        in the evaluation config (theoretical_spectrum section) and precomputed
        during embedding generation.

        Args:
            output_dir: Directory to save results
            max_samples: Maximum number of spectra to analyze (None = all)
            create_plots: Whether to create visualization plots
            max_individual_plots: Maximum number of individual spectrum plots to create
            min_backbone_coverage: Minimum backbone cleavage coverage (0-1, 0 = disabled)
            min_fragment_groups: Minimum unique fragment ion groups (0 = disabled)
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(output_dir=output_dir, **kwargs)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_samples = max_samples
        self.create_plots = create_plots
        self.max_individual_plots = max_individual_plots
        self.min_backbone_coverage = min_backbone_coverage
        self.min_fragment_groups = min_fragment_groups

        # Store max_mz for denormalization (will be updated from config if available)
        self.max_mz = 2500.0  # Default value

    def run(  # type: ignore[override]  # base class run() signature differs across tasks
        self,
        embeddings: np.ndarray,
        metadata: Dict[str, np.ndarray],
        faiss_index: Any,
    ) -> Dict[str, Any]:
        """Run confidence-signal analysis.

        Args:
            embeddings: Embedding vectors (N, D) - not used directly
            metadata: Metadata dictionary with sequences, spectra, and confidence
            faiss_index: FAISS index - not used

        Returns:
            Dictionary with analysis results and metrics
        """
        # Check dependencies
        if not SKLEARN_AVAILABLE:
            logger.error("sklearn is required for this task. Install with: pip install scikit-learn")
            return {"error": "sklearn not available", "success": False}

        # Check if theoretical spectra are precomputed
        has_theoretical = all(k in metadata for k in ["theoretical_mz", "theoretical_annotations", "theoretical_match_mask"])

        # Extract max_mz from metadata if available (stored during theoretical generation)
        if "theoretical_max_mz" in metadata:
            self.max_mz = float(metadata["theoretical_max_mz"])

        if not has_theoretical:
            logger.error("Theoretical spectra not found in metadata!")
            logger.error("Enable in evaluation config: theoretical_spectrum.enabled: True")
            return {"error": "Theoretical spectra not precomputed", "success": False}

        # Extract and validate required data
        data = self._extract_data(metadata)

        if data is None:
            return {"error": "Failed to extract required data", "success": False}

        sequences, spectra, precursor_charges = data
        n_spectra = len(sequences)

        # Extract per-peak confidence from metadata if available (before limiting samples)
        per_peak_confidence = metadata.get("per_peak_confidence", None)
        if per_peak_confidence is None:
            logger.warning("Per-peak confidence not found in metadata - task requires real confidence scores")

        # Extract decomposed confidence components (group and offset)
        per_peak_conf_group = metadata.get("per_peak_conf_group", None)
        per_peak_conf_offset = metadata.get("per_peak_conf_offset", None)

        # Extract per-spectrum confidence from metadata if available
        spectrum_confidences = metadata.get("spectrum_confidence", None)

        # Extract fragmentation types from metadata if available (needed for limiting samples)
        frag_types = metadata.get("frag_type", None)

        # Extract precomputed theoretical data before limiting samples
        theoretical_match_masks = metadata.get("theoretical_match_mask", None)

        # Extract matched_annotation list (per-peak annotations aligned with valid peaks)
        matched_annotations = metadata.get("matched_annotation", None)

        # Limit samples if requested
        # Use sequential indices (0, 1, 2, ..., max_samples-1) to match head_analysis task
        # This ensures spectrum_0000, spectrum_0001, etc. refer to the same sequences in both tasks
        if self.max_samples is not None and n_spectra > self.max_samples:
            indices = np.arange(self.max_samples)  # Sequential indices instead of random sampling
            sequences = sequences[indices]
            spectra = spectra[indices]
            precursor_charges = precursor_charges[indices]
            if frag_types is not None:
                frag_types = frag_types[indices]
            if per_peak_confidence is not None:
                per_peak_confidence = per_peak_confidence[indices]
            if per_peak_conf_group is not None:
                per_peak_conf_group = per_peak_conf_group[indices]
            if per_peak_conf_offset is not None:
                per_peak_conf_offset = per_peak_conf_offset[indices]
            if spectrum_confidences is not None:
                spectrum_confidences = spectrum_confidences[indices]
            if theoretical_match_masks is not None:
                theoretical_match_masks = theoretical_match_masks[indices]
            if matched_annotations is not None:
                matched_annotations = matched_annotations[indices]
            n_spectra = self.max_samples

        # Extract spectrum quality metrics (if available from embedding_io)
        spectrum_quality_arr = metadata.get("spectrum_quality", None)
        if spectrum_quality_arr is not None and self.max_samples is not None and n_spectra == self.max_samples:
            # indices were sequential [0..max_samples-1], slice accordingly
            spectrum_quality_arr = spectrum_quality_arr[:n_spectra]

        # Collect per-peak confidence and theoretical labels
        results = self._collect_peak_data(
            sequences,
            spectra,
            precursor_charges,
            per_peak_confidence,
            theoretical_match_masks,
            spectrum_quality=spectrum_quality_arr,
            per_peak_conf_group=per_peak_conf_group,
            per_peak_conf_offset=per_peak_conf_offset,
            matched_annotations=matched_annotations,
        )

        if results is None:
            return {"error": "Failed to collect peak data", "success": False}

        (
            confidence_scores,
            is_theoretical,
            n_analyzed,
            conf_group_scores,
            conf_offset_scores,
            spectrum_boundaries,
            peak_annotations,
            intensity_scores,
        ) = results

        n_peaks = len(confidence_scores)
        pct_ann = is_theoretical.mean() * 100
        logger.info(f"  {n_analyzed}/{n_spectra} spectra, {n_peaks:,} peaks ({pct_ann:.1f}% annotated)")

        # Compute separation metrics
        metrics = self._compute_metrics(confidence_scores, is_theoretical)

        # Intensity-confidence analysis (compute first to include in headline)
        intensity_analysis = self._compute_intensity_confidence_analysis(
            confidence_scores,
            intensity_scores,
            is_theoretical,
        )
        metrics["intensity_analysis"] = {k: v for k, v in intensity_analysis.items() if k != "stratified_bins"}
        metrics["intensity_analysis"]["stratified_bins"] = intensity_analysis["stratified_bins"]

        # Log headline: confidence vs intensity baseline + residual
        delta = intensity_analysis["confidence_delta_over_intensity"]
        residual = intensity_analysis["residual_confidence_auroc"]
        logger.info(
            f"  AUROC={metrics['auroc']:.4f}  Intensity baseline={intensity_analysis['intensity_auroc']:.4f}  "
            f"Delta={delta:+.4f}  Residual={residual:.4f}"
        )

        # Decomposed confidence analysis (group vs offset vs joint)
        if conf_group_scores is not None and conf_offset_scores is not None:
            decomposed = self._compute_decomposed_metrics(
                confidence_scores,
                conf_group_scores,
                conf_offset_scores,
                is_theoretical,
            )
            metrics["decomposed"] = decomposed

        # Per-spectrum AUROC distribution
        per_spectrum_results = self._compute_per_spectrum_metrics(
            confidence_scores,
            is_theoretical,
            spectrum_boundaries,
        )
        per_spectrum_aurocs = per_spectrum_results.pop("per_spectrum_aurocs")
        metrics["per_spectrum_auroc"] = per_spectrum_results
        if len(per_spectrum_aurocs) > 0:
            logger.info(
                f"  Per-spectrum AUROC: mean={per_spectrum_results['mean']:.4f}  "
                f"median={per_spectrum_results['median']:.4f}  "
                f"(n={per_spectrum_results['n_spectra_evaluated']})"
            )
        else:
            logger.warning("  No spectra had enough peaks for per-spectrum AUROC")

        # Confidence-quality correlation: does mean per-spectrum confidence
        # correlate with theoretical annotation quality?
        quality_corr = self._compute_quality_correlation(
            confidence_scores,
            is_theoretical,
            spectrum_boundaries,
            spectrum_quality_arr,
        )
        if quality_corr:
            metrics["quality_correlation"] = quality_corr

        # Ion-type confidence breakdown
        has_annotations = any(ann != "" for ann in peak_annotations)
        if has_annotations:
            ion_breakdown = self._compute_ion_type_breakdown(
                confidence_scores,
                peak_annotations,
                conf_group_scores=conf_group_scores,
                conf_offset_scores=conf_offset_scores,
            )
            metrics["ion_type_breakdown"] = ion_breakdown

            # Raw per-peak (category, confidence) dump so downstream figures can show
            # individual scatter points, not just per-type summary statistics.
            try:
                import polars as _pl

                _cats = [categorize_ion(a) for a in peak_annotations]
                _pl.DataFrame(
                    {
                        "ion_category": _cats,
                        "confidence": np.asarray(confidence_scores, dtype=float),
                    }
                ).write_parquet(str(Path(self.output_dir) / "per_peak_confidence_by_type.parquet"))
                logger.info(f"Wrote per-peak confidence dump ({len(_cats):,} peaks)")
            except Exception as _e:  # never break the task on the dump
                logger.warning(f"per-peak confidence dump failed: {_e}")

        # Fragment position confidence analysis (b/y ions by position)
        fragment_position_data = None
        if has_annotations:
            fragment_position_data = self._compute_fragment_position_confidence(
                confidence_scores,
                peak_annotations,
                spectrum_boundaries,
                sequences,
            )
            if fragment_position_data is not None:
                metrics["fragment_position_confidence"] = fragment_position_data

        # Create visualizations
        if self.create_plots and PLOTTING_AVAILABLE:
            self._create_plots(confidence_scores, is_theoretical, metrics, per_spectrum_aurocs)

            if has_annotations:
                self._create_ion_type_plot(
                    confidence_scores,
                    peak_annotations,
                    conf_group_scores=conf_group_scores,
                    conf_offset_scores=conf_offset_scores,
                )

            if fragment_position_data is not None:
                unannotated_mean = float(np.mean(confidence_scores[~is_theoretical]))
                self._create_fragment_position_plot(
                    fragment_position_data,
                    unannotated_mean,
                    max_position=15,
                    max_peptide_length=20,
                )

            self._create_intensity_confidence_plot(
                confidence_scores,
                intensity_scores,
                is_theoretical,
                intensity_analysis,
            )

            # Create individual spectrum plots
            if self.max_individual_plots > 0 and per_peak_confidence is not None:
                # Extract precomputed theoretical data
                theoretical_mz_list = metadata.get("theoretical_mz", None)
                theoretical_annotations_list = metadata.get("theoretical_annotations", None)
                theoretical_match_masks = metadata.get("theoretical_match_mask", None)
                theoretical_match_idx_list = metadata.get("theoretical_match_idx", None)
                frag_types = metadata.get("frag_type", None)

                # Re-apply sample limiting to these arrays
                if self.max_samples is not None and theoretical_mz_list is not None:
                    indices = np.arange(min(self.max_samples, len(theoretical_mz_list)))
                    if theoretical_mz_list is not None:
                        theoretical_mz_list = theoretical_mz_list[indices]
                    if theoretical_annotations_list is not None:
                        theoretical_annotations_list = theoretical_annotations_list[indices]
                    if theoretical_match_masks is not None:
                        theoretical_match_masks = theoretical_match_masks[indices]
                    if theoretical_match_idx_list is not None:
                        theoretical_match_idx_list = theoretical_match_idx_list[indices]
                    if frag_types is not None:
                        frag_types = frag_types[indices]

                self._create_individual_spectrum_plots(
                    sequences=sequences,
                    spectra=spectra,
                    precursor_charges=precursor_charges,
                    per_peak_confidence=per_peak_confidence,
                    theoretical_mz_list=theoretical_mz_list,
                    theoretical_annotations_list=theoretical_annotations_list,
                    theoretical_match_masks=theoretical_match_masks,
                    theoretical_match_idx_list=theoretical_match_idx_list,
                    frag_types=frag_types,
                    spectrum_confidences=spectrum_confidences,
                    matched_annotations=matched_annotations,
                    dataset_metrics=metrics,
                    spectrum_quality=metadata.get("spectrum_quality", None),
                )
        elif self.create_plots:
            logger.warning("Plotting libraries not available. Skipping visualizations.")

        # Save results
        self._save_results(metrics, n_analyzed, n_spectra)

        # Strip large curve arrays before returning — they were only needed for plotting
        metrics.pop("roc_curve", None)
        metrics.pop("pr_curve", None)

        return {
            "success": True,
            "metrics": metrics,
            "n_spectra_analyzed": n_analyzed,
            "n_spectra_total": n_spectra,
            "n_peaks_total": len(confidence_scores),
            "n_theoretical_peaks": int(is_theoretical.sum()),
            "fraction_theoretical": float(is_theoretical.mean()),
        }

    def _extract_data(self, metadata: Dict[str, np.ndarray]) -> Optional[tuple]:
        """Extract required data from metadata.

        Returns:
            Tuple of (sequences, spectra, charges) or None
        """
        # Try different sequence keys
        sequence_keys = ["sequence", "peptides", "peptide", "seq"]
        sequences = None
        for key in sequence_keys:
            if key in metadata:
                sequences = metadata[key]
                break

        if sequences is None:
            logger.error(f"No sequence data found. Tried keys: {sequence_keys}")
            return None

        # Get spectra (should be in metadata as 'spectra')
        if "spectra" not in metadata:
            logger.error("'spectra' not found in metadata")
            return None

        spectra = metadata["spectra"]

        # Get precursor charges
        charge_keys = ["precursor_charge", "charge", "precursor_charge_id"]
        precursor_charges = None
        for key in charge_keys:
            if key in metadata:
                precursor_charges = metadata[key]
                break

        if precursor_charges is None:
            logger.warning("No precursor charge data found. Using default charge=2")
            precursor_charges = np.full(len(sequences), 2, dtype=np.int32)

        return sequences, spectra, precursor_charges

    def _collect_peak_data(
        self,
        sequences: np.ndarray,
        spectra: np.ndarray,
        precursor_charges: np.ndarray,
        per_peak_confidence: Optional[np.ndarray],
        theoretical_match_masks: np.ndarray,
        spectrum_quality: Optional[np.ndarray] = None,
        per_peak_conf_group: Optional[np.ndarray] = None,
        per_peak_conf_offset: Optional[np.ndarray] = None,
        matched_annotations: Optional[np.ndarray] = None,
    ) -> Optional[tuple]:
        """Collect per-peak confidence scores and theoretical labels from precomputed data.

        Args:
            sequences: Peptide sequences
            spectra: Spectra arrays (N, L, 2)
            precursor_charges: Precursor charge states
            per_peak_confidence: Per-peak joint confidence (conf_group * conf_offset)
            theoretical_match_masks: Precomputed theoretical match masks (object array)
            spectrum_quality: Optional pre-computed quality metrics
            per_peak_conf_group: Optional per-peak group confidence (P(top1_group))
            per_peak_conf_offset: Optional per-peak offset confidence (P(top1_offset))
            matched_annotations: Optional per-peak annotation strings (object array of lists)

        Returns:
            Tuple of (confidence_scores, is_theoretical, n_analyzed,
                       conf_group_scores, conf_offset_scores,
                       spectrum_boundaries, all_peak_annotations,
                       intensity_scores) or None.
            conf_group_scores and conf_offset_scores may be None if not available.
            spectrum_boundaries is a numpy array of CSR-style boundaries (spectrum i
            covers peaks [boundaries[i]:boundaries[i+1]]).
            all_peak_annotations is a flat list of per-peak annotation strings.
            intensity_scores is a flat numpy array of per-peak normalized intensities.
        """
        all_confidence: list[Any] = []
        all_intensity: list[Any] = []
        all_conf_group: list[Any] | None = [] if per_peak_conf_group is not None else None
        all_conf_offset: list[Any] | None = [] if per_peak_conf_offset is not None else None
        all_is_theoretical: list[Any] = []
        spectrum_boundaries = [0]
        all_peak_annotations: list[str] = []
        n_analyzed = 0
        n_no_confidence = 0
        n_no_theoretical = 0
        n_skipped_invalid = 0
        n_skipped_low_quality = 0

        # Check if per-peak confidence is available
        has_confidence = per_peak_confidence is not None
        if not has_confidence:
            logger.warning("Per-peak confidence not available in metadata.")
            logger.warning("To enable, set compute_spectrum_confidence: True in evaluation config.")

        for idx in tqdm(range(len(sequences)), desc="Analyzing spectra", disable=True):
            seq = sequences[idx]

            # Skip invalid sequences
            if seq is None or not isinstance(seq, str) or len(seq.strip()) == 0:
                n_skipped_invalid += 1
                continue

            # Get precomputed theoretical match mask
            theo_mask = theoretical_match_masks[idx]
            if theo_mask is None:
                n_no_theoretical += 1
                continue

            # Apply backbone coverage quality gate
            if spectrum_quality is not None and idx < len(spectrum_quality) and spectrum_quality[idx] is not None:
                sq = spectrum_quality[idx]
                bc = sq.get("backbone_coverage", 0.0)
                ng = sq.get("n_fragment_groups", 0)
                if bc < self.min_backbone_coverage or ng < self.min_fragment_groups:
                    n_skipped_low_quality += 1
                    continue

            # Get spectrum data
            if spectra.ndim == 3:  # (N, L, 2) format
                spectrum = spectra[idx]  # (L, 2)
                mz = spectrum[:, 0]
                intensity = spectrum[:, 1]
            else:
                logger.warning(f"Cannot extract spectrum data for index {idx}")
                n_skipped_invalid += 1
                continue

            # Filter out zero/padding peaks (match what was done during theoretical generation)
            valid_mask = (mz > 0) & (intensity > 0)
            mz = mz[valid_mask]
            intensity = intensity[valid_mask]

            # The theoretical mask should already be aligned with valid peaks
            # (it was computed on the same valid peaks during generation)
            is_theoretical_peaks = theo_mask

            # Get confidence scores for these peaks
            if has_confidence and per_peak_confidence is not None and idx < len(per_peak_confidence):
                spectrum_conf = per_peak_confidence[idx]  # (L,) array

                # Match confidence to valid peaks (filter out padding)
                if len(spectrum_conf) >= len(mz):
                    peak_confidences = spectrum_conf[valid_mask]
                else:
                    peak_confidences = spectrum_conf[: len(mz)]

                # Ensure lengths match
                if len(peak_confidences) != len(mz):
                    logger.warning(f"Confidence length mismatch for spectrum {idx}: {len(peak_confidences)} vs {len(mz)}")
                    n_no_confidence += 1
                    continue

                # Extract decomposed confidence using the same valid_mask
                peak_conf_group = None
                peak_conf_offset = None
                if per_peak_conf_group is not None and idx < len(per_peak_conf_group):
                    g = per_peak_conf_group[idx]
                    peak_conf_group = g[valid_mask] if len(g) >= len(mz) else g[: len(mz)]
                if per_peak_conf_offset is not None and idx < len(per_peak_conf_offset):
                    o = per_peak_conf_offset[idx]
                    peak_conf_offset = o[valid_mask] if len(o) >= len(mz) else o[: len(mz)]
            else:
                # No confidence available — skip this spectrum entirely
                n_no_confidence += 1
                continue

            all_confidence.extend(peak_confidences)
            all_intensity.extend(intensity)
            all_is_theoretical.extend(is_theoretical_peaks)
            if all_conf_group is not None and peak_conf_group is not None:
                all_conf_group.extend(peak_conf_group)
            if all_conf_offset is not None and peak_conf_offset is not None:
                all_conf_offset.extend(peak_conf_offset)

            # Track spectrum boundaries (CSR-style) and per-peak annotations
            spectrum_boundaries.append(len(all_confidence))
            if matched_annotations is not None and idx < len(matched_annotations):
                spec_anns = matched_annotations[idx]
                if spec_anns is not None:
                    # Annotations are aligned with valid peaks
                    for i in range(len(peak_confidences)):
                        ann = spec_anns[i] if i < len(spec_anns) else None
                        all_peak_annotations.append(str(ann) if ann is not None and str(ann) != "None" else "")
                else:
                    all_peak_annotations.extend([""] * len(peak_confidences))
            else:
                all_peak_annotations.extend([""] * len(peak_confidences))

            n_analyzed += 1

        # Log skip warnings (only when something was actually skipped)
        n_skipped_total = n_skipped_invalid + n_no_theoretical + n_skipped_low_quality + n_no_confidence
        if n_skipped_total > 0:
            parts: list[Any] = []
            if n_skipped_invalid > 0:
                parts.append(f"invalid={n_skipped_invalid}")
            if n_no_theoretical > 0:
                parts.append(f"no_theo={n_no_theoretical}")
            if n_skipped_low_quality > 0:
                parts.append(f"low_quality={n_skipped_low_quality}")
            if n_no_confidence > 0:
                parts.append(f"no_conf={n_no_confidence}")
            logger.warning(f"  Skipped {n_skipped_total} spectra ({', '.join(parts)})")

        if len(all_confidence) == 0:
            logger.error("No valid spectra analyzed.")
            logger.error("Possible reasons:")
            logger.error("  1. All sequences failed cleaning or theoretical generation")
            logger.error("  2. All spectra had too few peaks")
            logger.error("  3. Per-peak confidence not available in metadata")
            return None

        conf_group_arr = np.array(all_conf_group) if all_conf_group else None
        conf_offset_arr = np.array(all_conf_offset) if all_conf_offset else None

        return (
            np.array(all_confidence),
            np.array(all_is_theoretical, dtype=bool),
            n_analyzed,
            conf_group_arr,
            conf_offset_arr,
            np.array(spectrum_boundaries),
            all_peak_annotations,
            np.array(all_intensity),
        )

    def _compute_metrics(self, confidence_scores: np.ndarray, is_theoretical: np.ndarray) -> Dict[str, Any]:
        """Compute separation metrics between signal and noise confidence.

        Args:
            confidence_scores: Per-peak confidence values
            is_theoretical: Boolean mask (True = theoretical peak, False = noise)

        Returns:
            Dictionary of metrics
        """
        # Binary labels for classification metrics
        labels = is_theoretical.astype(int)

        # AUROC - How well confidence separates signal from noise
        auroc = roc_auc_score(labels, confidence_scores)

        # Average Precision - Precision-recall performance
        avg_precision = average_precision_score(labels, confidence_scores)

        # ROC curve for plotting (stripped before return from run())
        fpr, tpr, roc_thresholds = roc_curve(labels, confidence_scores)

        # Precision-Recall curve for plotting (stripped before return from run())
        precision, recall, pr_thresholds = precision_recall_curve(labels, confidence_scores)

        # KS statistic - Distribution separation
        signal_conf = confidence_scores[is_theoretical]
        noise_conf = confidence_scores[~is_theoretical]
        ks_stat, ks_pvalue = ks_2samp(signal_conf, noise_conf)

        # Cohen's d — standardized effect size (weighted pooled SD for unequal groups)
        mean_signal_conf = float(np.mean(signal_conf))
        mean_noise_conf = float(np.mean(noise_conf))
        n1, n2 = len(signal_conf), len(noise_conf)
        var_signal = float(np.var(signal_conf, ddof=1)) if n1 > 1 else 0.0
        var_noise = float(np.var(noise_conf, ddof=1)) if n2 > 1 else 0.0
        denom = n1 + n2 - 2
        pooled_std = np.sqrt(((n1 - 1) * var_signal + (n2 - 1) * var_noise) / denom) if denom > 0 else 0.0
        cohens_d = (mean_signal_conf - mean_noise_conf) / pooled_std if pooled_std > 0 else 0.0

        # Class imbalance ratio
        n_signal = int(is_theoretical.sum())
        n_noise = int((~is_theoretical).sum())
        class_imbalance_ratio = n_noise / n_signal if n_signal > 0 else float("inf")

        return {
            "auroc": float(auroc),
            "average_precision": float(avg_precision),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "cohens_d": float(cohens_d),
            "class_imbalance_ratio": class_imbalance_ratio,
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": roc_thresholds.tolist(),
            },
            "pr_curve": {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": pr_thresholds.tolist(),
            },
        }

    def _compute_decomposed_metrics(
        self,
        conf_joint: np.ndarray,
        conf_group: np.ndarray,
        conf_offset: np.ndarray,
        is_theoretical: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Compute discrimination metrics for each confidence component.

        This reveals whether group (which m/z bin group) or offset (which bin
        within group) is more informative for separating annotated vs
        unannotated peaks.

        Returns:
            Dict with keys "joint", "group", "offset", each containing
            auroc, average_precision, cohens_d.
        """
        labels = is_theoretical.astype(int)
        result: dict[str, Any] = {}

        for name, scores in [("joint", conf_joint), ("group", conf_group), ("offset", conf_offset)]:
            signal = scores[is_theoretical]
            noise = scores[~is_theoretical]

            auroc = float(roc_auc_score(labels, scores))
            ap = float(average_precision_score(labels, scores))

            mean_sig = float(np.mean(signal))
            mean_noi = float(np.mean(noise))

            n1, n2 = len(signal), len(noise)
            var1 = float(np.var(signal, ddof=1)) if n1 > 1 else 0.0
            var2 = float(np.var(noise, ddof=1)) if n2 > 1 else 0.0
            denom = n1 + n2 - 2
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / denom) if denom > 0 else 0.0
            d = (mean_sig - mean_noi) / pooled_std if pooled_std > 0 else 0.0

            result[name] = {
                "auroc": auroc,
                "average_precision": ap,
                "cohens_d": d,
            }

        return result

    def _compute_per_spectrum_metrics(
        self,
        confidence_scores: np.ndarray,
        is_theoretical: np.ndarray,
        spectrum_boundaries: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute AUROC for each spectrum individually.

        Spectra with fewer than 2 annotated or 2 unannotated peaks are skipped
        (AUROC is undefined).

        Args:
            confidence_scores: Flat array of per-peak confidence values.
            is_theoretical: Flat boolean array (True = annotated peak).
            spectrum_boundaries: CSR-style boundaries — spectrum i covers peaks
                [boundaries[i]:boundaries[i+1]].

        Returns:
            Dict with 'per_spectrum_aurocs' (np.ndarray) and summary statistics.
        """
        n_spectra = len(spectrum_boundaries) - 1
        aurocs: list[Any] = []

        for i in range(n_spectra):
            start, end = spectrum_boundaries[i], spectrum_boundaries[i + 1]
            conf = confidence_scores[start:end]
            labels = is_theoretical[start:end]

            n_pos = int(labels.sum())
            n_neg = int((~labels).sum())
            if n_pos < 2 or n_neg < 2:
                continue

            try:
                auroc = float(roc_auc_score(labels.astype(int), conf))
                aurocs.append(auroc)
            except ValueError:
                continue

        auroc_arr = np.array(aurocs) if aurocs else np.array([])

        summary: Dict[str, Any] = {
            "per_spectrum_aurocs": auroc_arr,
            "n_spectra_evaluated": len(auroc_arr),
            "n_spectra_skipped": n_spectra - len(auroc_arr),
        }
        if len(auroc_arr) > 0:
            summary.update(
                {
                    "mean": float(np.mean(auroc_arr)),
                    "median": float(np.median(auroc_arr)),
                    "std": float(np.std(auroc_arr)),
                    "q25": float(np.percentile(auroc_arr, 25)),
                    "q75": float(np.percentile(auroc_arr, 75)),
                }
            )

        return summary

    def _compute_ion_type_breakdown(
        self,
        confidence_scores: np.ndarray,
        peak_annotations: list[str],
        conf_group_scores: Optional[np.ndarray] = None,
        conf_offset_scores: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compute confidence statistics broken down by ion type.

        Uses ``categorize_ion()`` to classify each peak annotation, then
        computes count, mean/median/std of joint confidence (and optionally
        group and offset confidence) per ion type.

        Args:
            confidence_scores: Flat per-peak joint confidence values.
            peak_annotations: Flat list of per-peak annotation strings
                (empty string for unannotated).
            conf_group_scores: Optional flat per-peak group confidence.
            conf_offset_scores: Optional flat per-peak offset confidence.

        Returns:
            Dict mapping ion-type category to statistics dict.
        """
        from collections import defaultdict

        buckets: dict[str, list[int]] = defaultdict(list)
        for i, ann in enumerate(peak_annotations):
            cat = categorize_ion(ann)
            buckets[cat].append(i)

        # Merge rare categories (< 50 peaks) into "Other"
        MIN_CATEGORY_SIZE = 500  # noqa: N806
        rare_cats = [c for c, idxs in buckets.items() if len(idxs) < MIN_CATEGORY_SIZE and c != "unannotated"]
        if rare_cats:
            other_idxs: list[int] = []
            for c in rare_cats:
                other_idxs.extend(buckets.pop(c))
            if other_idxs:
                buckets["Other"].extend(other_idxs)

        breakdown: Dict[str, Dict[str, float]] = {}
        for cat in sorted(buckets.keys()):
            idxs = np.array(buckets[cat])
            conf = confidence_scores[idxs]
            entry: Dict[str, float] = {
                "count": len(idxs),
                "mean_conf": float(np.mean(conf)),
                "median_conf": float(np.median(conf)),
                "std_conf": float(np.std(conf)),
            }
            if conf_group_scores is not None:
                g = conf_group_scores[idxs]
                entry["mean_conf_group"] = float(np.mean(g))
                entry["median_conf_group"] = float(np.median(g))
            if conf_offset_scores is not None:
                o = conf_offset_scores[idxs]
                entry["mean_conf_offset"] = float(np.mean(o))
                entry["median_conf_offset"] = float(np.median(o))
            breakdown[cat] = entry

        return breakdown

    def _compute_fragment_position_confidence(
        self,
        confidence_scores: np.ndarray,
        peak_annotations: list[str],
        spectrum_boundaries: np.ndarray,
        sequences: np.ndarray,
        max_position: int = 15,
        max_peptide_length: int = 20,
    ) -> Optional[Dict[str, Any]]:
        """Compute confidence statistics by fragment ion position.

        For b-ions and y-ions, groups peaks by (ion_series, position, subtype)
        where subtype is one of {base, isotope, loss}. Filters to peptides
        with length < max_peptide_length and positions <= max_position.

        Args:
            confidence_scores: Flat per-peak joint confidence values.
            peak_annotations: Flat list of per-peak annotation strings.
            spectrum_boundaries: CSR-style boundaries for per-spectrum slicing.
            sequences: Array of peptide sequences (used for length filtering).
            max_position: Maximum fragment position to include (e.g. 15).
            max_peptide_length: Maximum peptide length to include (e.g. 20).

        Returns:
            Dict with per-position stats for b and y ions, or None if
            insufficient data.
        """
        # Build per-spectrum peptide length lookup
        n_spectra = len(spectrum_boundaries) - 1

        # Collect confidence by (ion_series, position, subtype)
        # subtype: "base", "isotope", "loss"
        from collections import defaultdict

        data: dict[str, dict[int, dict[str, list[float]]]] = {
            "b": defaultdict(lambda: defaultdict(list)),
            "y": defaultdict(lambda: defaultdict(list)),
        }

        for spec_i in range(n_spectra):
            # Filter by peptide length
            seq = sequences[spec_i] if spec_i < len(sequences) else None
            if seq is None or not isinstance(seq, str):
                continue
            # Strip modifications for length count (e.g. "[UNIMOD:35]")
            clean_seq = re.sub(r"\[.*?\]", "", seq)
            if len(clean_seq) > max_peptide_length:
                continue

            start, end = spectrum_boundaries[spec_i], spectrum_boundaries[spec_i + 1]

            for peak_i in range(start, end):
                ann = peak_annotations[peak_i]
                if not ann:
                    continue

                ion_type = extract_ion_type(ann)
                if ion_type not in ("b", "y"):
                    continue

                position = extract_fragment_position(ann)
                if position < 1 or position > max_position:
                    continue

                # Determine subtype from the annotation
                cat = categorize_ion(ann)
                if "Isotope" in cat:
                    subtype = "isotope"
                elif "Loss" in cat:
                    subtype = "loss"
                else:
                    subtype = "base"

                data[ion_type][position][subtype].append(confidence_scores[peak_i])

        # Compute stats
        result: Dict[str, Any] = {}
        for ion_series in ("b", "y"):
            positions_data: Dict[int, Dict[str, Dict[str, float]]] = {}
            for pos in sorted(data[ion_series].keys()):
                pos_stats: Dict[str, Dict[str, float]] = {}
                for subtype in ("base", "isotope", "loss"):
                    values = data[ion_series][pos].get(subtype, [])
                    if len(values) > 0:
                        arr = np.array(values)
                        pos_stats[subtype] = {
                            "count": len(values),
                            "mean": float(np.mean(arr)),
                            "std": float(np.std(arr)),
                        }
                if pos_stats:
                    positions_data[pos] = pos_stats
            result[ion_series] = positions_data

        if not result.get("b") and not result.get("y"):
            return None

        return result

    def _create_fragment_position_plot(
        self,
        fragment_position_data: Dict[str, Any],
        unannotated_mean_conf: float,
        max_position: int = 15,
        max_peptide_length: int = 20,
    ) -> None:
        """Create the fragment ion position confidence figure.

        Two-panel figure: y-ions (top), b-ions (bottom).
        At each position, 3 grouped bars (base, isotope, loss) with error bars.

        Args:
            fragment_position_data: Output of _compute_fragment_position_confidence.
            unannotated_mean_conf: Mean confidence of unannotated peaks (reference line).
            max_position: Maximum fragment position shown (for title).
            max_peptide_length: Maximum peptide length filter applied (for title).
        """
        if not PLOTTING_AVAILABLE:
            return

        sns.set_style("whitegrid")
        filter_text = f"positions 1–{max_position}, peptide length ≤ {max_peptide_length}"
        fig, (ax_y, ax_b) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
        fig.suptitle(
            f"Fragment Ion Confidence by Position ({filter_text})",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )

        subtypes = ["base", "isotope", "loss"]
        subtype_colors: dict[str, Any] = {"base": "#2ca02c", "isotope": "#1f77b4", "loss": "#ff7f0e"}
        subtype_labels: dict[str, Any] = {"base": "Base ion", "isotope": "Isotope", "loss": "Neutral loss"}
        bar_width = 0.25

        for ax, ion_series, title in [
            (ax_y, "y", "Y-ion"),
            (ax_b, "b", "B-ion"),
        ]:
            pos_data = fragment_position_data.get(ion_series, {})
            if not pos_data:
                ax.text(0.5, 0.5, f"No {title} data", transform=ax.transAxes, ha="center", va="center", fontsize=12, color="gray")
                ax.set_title(title, fontsize=13, fontweight="bold")
                continue

            positions = sorted(int(p) for p in pos_data.keys())
            x = np.arange(len(positions))

            for i, subtype in enumerate(subtypes):
                means: list[Any] = []
                stds: list[Any] = []
                counts: list[Any] = []
                for pos in positions:
                    stats = pos_data.get(pos, {}).get(subtype, {})
                    if stats:
                        means.append(stats["mean"])
                        stds.append(stats["std"])
                        counts.append(stats["count"])
                    else:
                        means.append(0)
                        stds.append(0)
                        counts.append(0)

                means_arr = np.array(means)
                stds_arr = np.array(stds)
                counts_arr = np.array(counts)

                # Only plot bars where we have data
                mask = counts_arr > 0
                if not mask.any():
                    continue

                offset = (i - 1) * bar_width
                ax.bar(
                    x[mask] + offset,
                    means_arr[mask],
                    bar_width,
                    yerr=stds_arr[mask],
                    capsize=2,
                    label=subtype_labels[subtype],
                    color=subtype_colors[subtype],
                    alpha=0.8,
                    edgecolor="white",
                    error_kw={"linewidth": 0.8, "alpha": 0.6},
                )

                # (counts available in JSON output)

            # Reference line: unannotated mean confidence
            ax.axhline(
                unannotated_mean_conf, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label=f"Unannotated mean ({unannotated_mean_conf:.3f})"
            )

            ax.set_xticks(x)
            ax.set_xticklabels([f"{ion_series}{p}" for p in positions], fontsize=10)
            ax.set_ylabel("Mean Confidence", fontsize=11)
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.legend(fontsize=9, loc="upper right")
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        save_path = Path(self.output_dir) / "confidence_fragment_position.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _compute_intensity_confidence_analysis(
        self,
        confidence_scores: np.ndarray,
        intensity_scores: np.ndarray,
        is_theoretical: np.ndarray,
        n_bins: int = 5,
    ) -> Dict[str, Any]:
        """Analyze the relationship between peak intensity and model confidence.

        Four analyses:
        1. Intensity AUROC — how well does intensity alone separate signal from noise?
           This is the "free baseline" that any model should beat.
        2. Residual confidence AUROC — confidence after regressing out intensity
           (rank-based). Isolates what the model learns beyond "bright peaks are real."
        3. Intensity-stratified confidence AUROC — within each intensity quintile,
           how well does confidence discriminate? Reveals whether the model adds
           value at all intensity levels or only among bright peaks.
        4. Spearman correlation — how tightly does confidence track intensity,
           split by annotation status.

        Args:
            confidence_scores: Flat per-peak joint confidence values.
            intensity_scores: Flat per-peak normalized intensities.
            is_theoretical: Flat boolean array (True = annotated peak).
            n_bins: Number of intensity bins for stratified analysis.

        Returns:
            Dict with intensity_auroc, residual_confidence_auroc, spearman
            correlations, and intensity_stratified results.
        """
        from scipy.stats import rankdata

        labels = is_theoretical.astype(int)

        # 1. Intensity AUROC — the free baseline
        intensity_auroc = float(roc_auc_score(labels, intensity_scores))

        # 2. Residual confidence AUROC — regress out intensity via ranks
        # Rank-based to avoid distributional assumptions
        conf_ranks = rankdata(confidence_scores)
        intensity_ranks = rankdata(intensity_scores)
        residual_scores = conf_ranks - intensity_ranks
        residual_auroc = float(roc_auc_score(labels, residual_scores))

        confidence_auroc = float(roc_auc_score(labels, confidence_scores))

        # 3. Spearman correlations
        rho_overall, _ = spearmanr(intensity_scores, confidence_scores)
        ann_mask = is_theoretical
        rho_ann, _ = spearmanr(intensity_scores[ann_mask], confidence_scores[ann_mask])
        rho_unann, _ = spearmanr(intensity_scores[~ann_mask], confidence_scores[~ann_mask])

        # 4. Intensity-stratified confidence AUROC
        # Use quantile boundaries so each bin has roughly equal peak count
        bin_edges = np.quantile(intensity_scores, np.linspace(0, 1, n_bins + 1))
        # Ensure unique edges (can happen if many peaks share the same intensity)
        bin_edges = np.unique(bin_edges)
        actual_bins = len(bin_edges) - 1

        stratified: list[Any] = []
        for b in range(actual_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            if b < actual_bins - 1:
                mask = (intensity_scores >= lo) & (intensity_scores < hi)
            else:
                mask = (intensity_scores >= lo) & (intensity_scores <= hi)

            bin_labels = labels[mask]
            bin_conf = confidence_scores[mask]
            n_pos = int(bin_labels.sum())
            n_neg = int((~is_theoretical[mask]).sum())

            entry: Dict[str, Any] = {
                "intensity_lo": float(lo),
                "intensity_hi": float(hi),
                "n_peaks": int(mask.sum()),
                "n_annotated": n_pos,
                "pct_annotated": float(n_pos / mask.sum() * 100) if mask.sum() > 0 else 0.0,
            }

            if n_pos >= 2 and n_neg >= 2:
                entry["confidence_auroc"] = float(roc_auc_score(bin_labels, bin_conf))
            else:
                entry["confidence_auroc"] = None

            stratified.append(entry)

        return {
            "intensity_auroc": intensity_auroc,
            "confidence_auroc": confidence_auroc,
            "confidence_delta_over_intensity": confidence_auroc - intensity_auroc,
            "residual_confidence_auroc": residual_auroc,
            "spearman_overall": float(rho_overall),
            "spearman_annotated": float(rho_ann),
            "spearman_unannotated": float(rho_unann),
            "stratified_bins": stratified,
        }

    def _create_intensity_confidence_plot(
        self,
        confidence_scores: np.ndarray,
        intensity_scores: np.ndarray,
        is_theoretical: np.ndarray,
        analysis: Dict[str, Any],
        n_curve_bins: int = 30,
    ) -> None:
        """Create intensity-confidence relationship visualization.

        Panel A: Intensity-stratified confidence AUROC bar chart.
        Panel B: Mean confidence vs intensity curve, split by annotation status.

        Args:
            confidence_scores: Flat per-peak confidence values.
            intensity_scores: Flat per-peak normalized intensities.
            is_theoretical: Flat boolean array.
            analysis: Results from _compute_intensity_confidence_analysis().
            n_curve_bins: Number of bins for the Panel B curve.
        """
        if not PLOTTING_AVAILABLE:
            return

        sns.set_style("whitegrid")
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5.5))

        stratified = analysis["stratified_bins"]
        intensity_auroc = analysis["intensity_auroc"]
        confidence_auroc = analysis["confidence_auroc"]

        # === Panel A: Intensity-stratified confidence AUROC ===
        bin_labels: list[Any] = []
        auroc_vals: list[Any] = []
        pct_ann_vals: list[Any] = []
        for i, s in enumerate(stratified):
            label = f"Q{i + 1}\n({s['intensity_lo']:.2f}-{s['intensity_hi']:.2f})"
            bin_labels.append(label)
            auroc_vals.append(s["confidence_auroc"])
            pct_ann_vals.append(s["pct_annotated"])

        x_pos = np.arange(len(bin_labels))
        valid_aurocs = [(i, v) for i, v in enumerate(auroc_vals) if v is not None]

        if valid_aurocs:
            bar_x = [i for i, _ in valid_aurocs]
            bar_v = [v for _, v in valid_aurocs]
            # Bars start from 0.5 baseline — height represents AUROC above chance
            bar_heights = [v - 0.5 for v in bar_v]
            colors = ["#2ca02c" if h > 0 else "#d62728" for h in bar_heights]
            ax_a.bar(bar_x, bar_heights, bottom=0.5, color=colors, alpha=0.7, width=0.6, edgecolor="black", linewidth=0.5)

        # Reference lines
        ax_a.axhline(0.5, color="gray", linestyle=":", linewidth=1.2, alpha=0.7, label="Random (0.5)")
        ax_a.axhline(
            confidence_auroc, color="#1f77b4", linestyle="--", linewidth=1.5, alpha=0.8, label=f"Overall conf AUROC ({confidence_auroc:.3f})"
        )
        ax_a.axhline(intensity_auroc, color="#ff7f0e", linestyle="--", linewidth=1.5, alpha=0.8, label=f"Intensity AUROC ({intensity_auroc:.3f})")

        ax_a.set_xticks(x_pos)
        ax_a.set_xticklabels(bin_labels, fontsize=8)
        ax_a.set_ylabel("Confidence AUROC", fontsize=11)
        ax_a.set_xlabel("Intensity Quintile", fontsize=11)
        ax_a.set_title("Confidence Discrimination by Intensity", fontsize=13, fontweight="bold")
        all_ref = [confidence_auroc, intensity_auroc] + [v for v in auroc_vals if v is not None]
        ax_a.set_ylim(0.45, max(0.75, max(all_ref, default=0.6) + 0.05))
        ax_a.legend(fontsize=8, loc="upper left")
        ax_a.grid(axis="y", alpha=0.3)

        # Annotated % as secondary annotation
        for i, s in enumerate(stratified):
            if s["confidence_auroc"] is not None:
                ax_a.text(i, s["confidence_auroc"] + 0.01, f"{s['pct_annotated']:.0f}%", ha="center", fontsize=7, color="#555555")

        # === Panel B: Mean confidence vs intensity percentile, by annotation status ===
        # Use quantile-based bins and plot on percentile x-axis so each bin has
        # equal data support (intensity is heavily right-skewed, raw values
        # compress most data into a tiny range and balloon noise at the tail).
        quantile_fracs = np.linspace(0, 1, n_curve_bins + 1)
        bin_edges = np.quantile(intensity_scores, quantile_fracs)
        bin_edges = np.unique(bin_edges)
        n_actual = len(bin_edges) - 1

        ann_means: list[Any] = []
        unann_means: list[Any] = []
        ann_stds: list[Any] = []
        unann_stds: list[Any] = []
        # Evenly-spaced percentile centers for x-axis
        pct_centers: list[Any] = []

        for b in range(n_actual):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            if b < n_actual - 1:
                mask = (intensity_scores >= lo) & (intensity_scores < hi)
            else:
                mask = (intensity_scores >= lo) & (intensity_scores <= hi)

            pct_center = (b + 0.5) / n_actual * 100  # 0–100 %
            pct_centers.append(pct_center)

            ann_in_bin = confidence_scores[mask & is_theoretical]
            unann_in_bin = confidence_scores[mask & ~is_theoretical]

            ann_means.append(float(np.mean(ann_in_bin)) if len(ann_in_bin) > 0 else np.nan)
            unann_means.append(float(np.mean(unann_in_bin)) if len(unann_in_bin) > 0 else np.nan)
            ann_stds.append(float(np.std(ann_in_bin)) if len(ann_in_bin) > 1 else 0.0)
            unann_stds.append(float(np.std(unann_in_bin)) if len(unann_in_bin) > 1 else 0.0)

        pct_centers = np.array(pct_centers)
        ann_means = np.array(ann_means)
        unann_means = np.array(unann_means)
        ann_stds = np.array(ann_stds)
        unann_stds = np.array(unann_stds)

        # Plot annotated line
        valid_ann = ~np.isnan(ann_means)
        if valid_ann.any():
            ax_b.plot(pct_centers[valid_ann], ann_means[valid_ann], color="#2ca02c", linewidth=2, label="Annotated", zorder=3)
            ax_b.fill_between(
                pct_centers[valid_ann],
                ann_means[valid_ann] - ann_stds[valid_ann],
                ann_means[valid_ann] + ann_stds[valid_ann],
                color="#2ca02c",
                alpha=0.15,
            )

        # Plot unannotated line
        valid_unann = ~np.isnan(unann_means)
        if valid_unann.any():
            ax_b.plot(pct_centers[valid_unann], unann_means[valid_unann], color="#d62728", linewidth=2, label="Unannotated", zorder=3)
            ax_b.fill_between(
                pct_centers[valid_unann],
                unann_means[valid_unann] - unann_stds[valid_unann],
                unann_means[valid_unann] + unann_stds[valid_unann],
                color="#d62728",
                alpha=0.15,
            )

        # Annotate Spearman correlations
        textstr = (
            f"Spearman(conf, int):\n"
            f"  overall = {analysis['spearman_overall']:.3f}\n"
            f"  annotated = {analysis['spearman_annotated']:.3f}\n"
            f"  unannotated = {analysis['spearman_unannotated']:.3f}"
        )
        ax_b.text(
            0.97,
            0.03,
            textstr,
            transform=ax_b.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "wheat", "alpha": 0.8},
        )

        ax_b.set_xlabel("Intensity Percentile", fontsize=11)
        ax_b.set_ylabel("Mean Confidence", fontsize=11)
        ax_b.set_title("Confidence vs Intensity by Annotation", fontsize=13, fontweight="bold")
        ax_b.set_xlim(0, 100)
        ax_b.legend(fontsize=9, loc="upper left")
        ax_b.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            Path(self.output_dir) / "confidence_intensity_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_ion_type_plot(
        self,
        confidence_scores: np.ndarray,
        peak_annotations: list[str],
        conf_group_scores: Optional[np.ndarray] = None,
        conf_offset_scores: Optional[np.ndarray] = None,
        min_peaks: int = 500,
    ) -> None:
        """Create ion-type confidence breakdown visualization.

        Panel A (always): Horizontal violin plot showing the confidence
        distribution per ion type, sorted by median confidence descending.
        The unannotated baseline is shown at the bottom for reference.

        Panel B (conditional — only when decomposed confidence is available):
        Dot plot showing mean group vs offset confidence per ion type,
        revealing whether discrimination comes from group or offset.

        Args:
            confidence_scores: Flat per-peak joint confidence values.
            peak_annotations: Flat list of per-peak annotation strings.
            conf_group_scores: Optional flat per-peak group confidence.
            conf_offset_scores: Optional flat per-peak offset confidence.
            min_peaks: Minimum peaks in a category to show it (avoids noisy violins).
        """
        from collections import defaultdict

        if not PLOTTING_AVAILABLE:
            return

        # --- Group peaks by ion type ---
        buckets: dict[str, list[int]] = defaultdict(list)
        for i, ann in enumerate(peak_annotations):
            cat = categorize_ion(ann)
            buckets[cat].append(i)

        # Filter to categories with enough peaks
        categories = [cat for cat, idxs in buckets.items() if len(idxs) >= min_peaks]
        if len(categories) < 2:
            logger.warning("Not enough ion-type categories with sufficient peaks for visualization")
            return

        # Sort by median confidence (descending), but keep unannotated last
        def sort_key(cat: str) -> tuple[int, float]:
            """Sort key."""
            idxs = np.array(buckets[cat])
            median = float(np.median(confidence_scores[idxs]))
            # unannotated sorts last (1, ...), everything else first (0, ...)
            return (1 if cat == "unannotated" else 0, -median)

        categories.sort(key=sort_key)

        # Compute unannotated median for reference line
        unannotated_median = None
        if "unannotated" in buckets and len(buckets["unannotated"]) >= min_peaks:
            unannotated_median = float(np.median(confidence_scores[np.array(buckets["unannotated"])]))

        has_decomposed = conf_group_scores is not None and conf_offset_scores is not None

        # --- Figure layout ---
        if has_decomposed:
            fig, (ax_a, ax_b) = plt.subplots(
                1,
                2,
                figsize=(16, max(5, 0.6 * len(categories))),
                gridspec_kw={"width_ratios": [3, 2], "wspace": 0.35},
            )
        else:
            fig, ax_a = plt.subplots(figsize=(10, max(5, 0.6 * len(categories))))

        sns.set_style("whitegrid")

        # === Panel A: Horizontal violin plot ===
        # Build data for seaborn
        violin_data: list[Any] = []
        violin_cats: list[Any] = []
        for cat in categories:
            idxs = np.array(buckets[cat])
            conf = confidence_scores[idxs]
            violin_data.append(conf)
            violin_cats.append(cat)

        # Collect colors and counts for labeling
        cat_colors: list[Any] = []
        cat_counts: list[Any] = []
        for cat in categories:
            color, _ = CATEGORY_COLORS.get(cat, ("#9467bd", 0.8))
            cat_colors.append(color)
            cat_counts.append(len(buckets[cat]))

        # Plot violins using matplotlib directly for color control
        parts = ax_a.violinplot(
            violin_data,
            positions=range(len(categories)),
            vert=False,
            showmedians=False,
            showextrema=False,
        )

        # Color each violin body
        for i, body in enumerate(parts["bodies"]):
            color, alpha = CATEGORY_COLORS.get(categories[i], ("#9467bd", 0.8))
            body.set_facecolor(color)
            body.set_alpha(max(alpha, 0.6))
            body.set_edgecolor("black")
            body.set_linewidth(0.5)

        # Overlay median (white dot) and mean (small tick) per category
        for i, cat in enumerate(categories):
            idxs = np.array(buckets[cat])
            conf = confidence_scores[idxs]
            median_val = float(np.median(conf))
            mean_val = float(np.mean(conf))
            q25 = float(np.percentile(conf, 25))
            q75 = float(np.percentile(conf, 75))

            # IQR whisker
            ax_a.hlines(i, q25, q75, color="black", linewidth=1.5, zorder=3)
            # Median dot
            ax_a.scatter([median_val], [i], color="white", edgecolor="black", s=40, zorder=4, linewidths=0.8)
            # Mean tick
            ax_a.scatter([mean_val], [i], color="black", marker="|", s=60, zorder=4, linewidths=1.2)

        # Unannotated median reference line
        if unannotated_median is not None:
            ax_a.axvline(
                unannotated_median,
                color="#BDBDBD",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
                zorder=1,
                label=f"Unannotated median ({unannotated_median:.3f})",
            )
            ax_a.legend(fontsize=8, loc="upper right")

        # Y-axis: category names
        ax_a.set_yticks(range(len(categories)))
        ax_a.set_yticklabels(categories, fontsize=9)
        ax_a.invert_yaxis()

        # Count labels on the right margin
        ax_a_twin = ax_a.twinx()
        ax_a_twin.set_ylim(ax_a.get_ylim())
        ax_a_twin.set_yticks(range(len(categories)))
        ax_a_twin.set_yticklabels(
            [f"n={cat_counts[i]:,}" for i in range(len(categories))],
            fontsize=8,
            color="#555555",
        )
        ax_a_twin.tick_params(axis="y", length=0)

        ax_a.set_xlabel("Joint Confidence", fontsize=11)
        ax_a.set_title("Ion-Type Confidence Distribution", fontsize=13, fontweight="bold")
        ax_a.set_xlim(-0.02, 1.02)
        ax_a.grid(axis="x", alpha=0.3)

        # === Panel B: Group vs Offset dot plot (conditional) ===
        if has_decomposed and conf_group_scores is not None and conf_offset_scores is not None:
            for i, cat in enumerate(categories):
                idxs = np.array(buckets[cat])
                mean_g = float(np.mean(conf_group_scores[idxs]))
                mean_o = float(np.mean(conf_offset_scores[idxs]))
                color, _ = CATEGORY_COLORS.get(cat, ("#9467bd", 0.8))

                # Connecting line
                ax_b.plot([mean_g, mean_o], [i, i], color=color, linewidth=1.2, alpha=0.6)
                # Group confidence (circle)
                ax_b.scatter([mean_g], [i], color=color, marker="o", s=50, edgecolor="black", linewidths=0.5, zorder=3)
                # Offset confidence (triangle)
                ax_b.scatter([mean_o], [i], color=color, marker="^", s=50, edgecolor="black", linewidths=0.5, zorder=3)

            # Legend for markers
            ax_b.scatter([], [], color="gray", marker="o", s=50, edgecolor="black", linewidths=0.5, label="Group (P(top1_group))")
            ax_b.scatter([], [], color="gray", marker="^", s=50, edgecolor="black", linewidths=0.5, label="Offset (P(top1_offset))")
            ax_b.legend(fontsize=8, loc="upper right")

            ax_b.set_yticks(range(len(categories)))
            ax_b.set_yticklabels([])  # shared with Panel A
            ax_b.invert_yaxis()
            ax_b.set_xlabel("Mean Confidence", fontsize=11)
            ax_b.set_title("Group vs Offset Confidence", fontsize=13, fontweight="bold")
            ax_b.set_xlim(-0.02, 1.02)
            ax_b.grid(axis="x", alpha=0.3)

        plt.savefig(
            Path(self.output_dir) / "confidence_ion_type_breakdown.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_plots(
        self,
        confidence_scores: np.ndarray,
        is_theoretical: np.ndarray,
        metrics: Dict[str, Any],
        per_spectrum_aurocs: Optional[np.ndarray] = None,
    ) -> None:
        """Create 4-panel dataset-level dashboard.

        Args:
            confidence_scores: Per-peak confidence values
            is_theoretical: Boolean mask for theoretical peaks
            metrics: Computed metrics dictionary
            per_spectrum_aurocs: Optional array of per-spectrum AUROC values for Panel D
        """
        output_dir = Path(self.output_dir)

        sns.set_style("whitegrid")

        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.30)

        ann_conf = confidence_scores[is_theoretical]
        unann_conf = confidence_scores[~is_theoretical]

        # --- Panel A: Confidence Distributions ---
        ax_a = fig.add_subplot(gs[0, 0])

        ax_a.hist(unann_conf, bins=50, alpha=0.5, label=f"Unannotated (n={len(unann_conf):,})", color="#d62728", density=True)
        ax_a.hist(ann_conf, bins=50, alpha=0.5, label=f"Annotated (n={len(ann_conf):,})", color="#2ca02c", density=True)

        # KDE curves
        from scipy.stats import gaussian_kde

        if len(ann_conf) > 1 and len(unann_conf) > 1:
            x_grid = np.linspace(0, 1, 200)
            try:
                kde_ann = gaussian_kde(ann_conf)
                kde_unann = gaussian_kde(unann_conf)
                ax_a.plot(x_grid, kde_ann(x_grid), color="#2ca02c", linewidth=2)
                ax_a.plot(x_grid, kde_unann(x_grid), color="#d62728", linewidth=2)
            except np.linalg.LinAlgError:
                pass  # KDE can fail with degenerate data

        mean_ann = float(np.mean(ann_conf))
        mean_unann = float(np.mean(unann_conf))
        ax_a.axvline(mean_ann, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Mean Annotated ({mean_ann:.3f})")
        ax_a.axvline(mean_unann, color="#d62728", linestyle="--", linewidth=1.5, label=f"Mean Unannotated ({mean_unann:.3f})")

        # Text box with summary stats
        textstr = f"Cohen's d = {metrics['cohens_d']:.3f}\nKS stat = {metrics['ks_statistic']:.3f}"
        ax_a.text(
            0.97,
            0.97,
            textstr,
            transform=ax_a.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "wheat", "alpha": 0.8},
        )

        ax_a.set_xlabel("Confidence Score", fontsize=11)
        ax_a.set_ylabel("Density", fontsize=11)
        ax_a.set_title("A: Confidence Distributions", fontsize=13, fontweight="bold")
        ax_a.legend(fontsize=9, loc="upper left")
        ax_a.grid(alpha=0.3)

        # --- Panel B: ROC Curve ---
        ax_b = fig.add_subplot(gs[0, 1])

        fpr = np.array(metrics["roc_curve"]["fpr"])
        tpr = np.array(metrics["roc_curve"]["tpr"])

        ax_b.plot(fpr, tpr, linewidth=2, color="#1f77b4", label=f"ROC (AUC = {metrics['auroc']:.4f})")
        ax_b.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")

        # Mark optimal threshold point (Youden's J)
        youdens_j = tpr - fpr
        best_roc_idx = int(np.argmax(youdens_j))
        ax_b.scatter(
            [fpr[best_roc_idx]],
            [tpr[best_roc_idx]],
            s=80,
            color="red",
            zorder=5,
            label=f"Youden J ({fpr[best_roc_idx]:.2f}, {tpr[best_roc_idx]:.2f})",
        )

        ax_b.set_xlabel("False Positive Rate", fontsize=11)
        ax_b.set_ylabel("True Positive Rate", fontsize=11)
        ax_b.set_title("B: ROC Curve", fontsize=13, fontweight="bold")
        ax_b.legend(fontsize=9)
        ax_b.grid(alpha=0.3)

        # --- Panel C: Precision-Recall ---
        ax_c = fig.add_subplot(gs[1, 0])

        pr_precision = np.array(metrics["pr_curve"]["precision"])
        pr_recall = np.array(metrics["pr_curve"]["recall"])

        ax_c.plot(pr_recall, pr_precision, linewidth=2, color="#ff7f0e", label=f"PR (AP = {metrics['average_precision']:.4f})")

        # Prevalence baseline
        prevalence = float(is_theoretical.mean())
        ax_c.axhline(prevalence, color="gray", linestyle="--", linewidth=1, alpha=0.7, label=f"Prevalence ({prevalence:.3f})")

        ax_c.set_xlabel("Recall", fontsize=11)
        ax_c.set_ylabel("Precision", fontsize=11)
        ax_c.set_title("C: Precision-Recall Curve", fontsize=13, fontweight="bold")
        ax_c.legend(fontsize=9)
        ax_c.grid(alpha=0.3)

        # --- Panel D: Per-Spectrum AUROC Histogram ---
        ax_d = fig.add_subplot(gs[1, 1])

        if per_spectrum_aurocs is not None and len(per_spectrum_aurocs) > 0:
            ax_d.hist(per_spectrum_aurocs, bins=50, alpha=0.7, color="#1f77b4", edgecolor="white", linewidth=0.5)

            mean_auroc = float(np.mean(per_spectrum_aurocs))
            median_auroc = float(np.median(per_spectrum_aurocs))

            ax_d.axvline(mean_auroc, color="red", linestyle="--", linewidth=1.5, label=f"Mean ({mean_auroc:.3f})")
            ax_d.axvline(median_auroc, color="orange", linestyle="--", linewidth=1.5, label=f"Median ({median_auroc:.3f})")
            ax_d.axvline(0.5, color="gray", linestyle=":", linewidth=1.2, alpha=0.7, label="Random (0.5)")

            # Stats text box
            std_auroc = float(np.std(per_spectrum_aurocs))
            q25 = float(np.percentile(per_spectrum_aurocs, 25))
            q75 = float(np.percentile(per_spectrum_aurocs, 75))
            textstr = f"n = {len(per_spectrum_aurocs):,}\nstd = {std_auroc:.3f}\nQ25 = {q25:.3f}\nQ75 = {q75:.3f}"
            ax_d.text(
                0.03,
                0.97,
                textstr,
                transform=ax_d.transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="left",
                bbox={"boxstyle": "round,pad=0.4", "facecolor": "wheat", "alpha": 0.8},
            )

            ax_d.set_xlabel("Per-Spectrum AUROC", fontsize=11)
            ax_d.set_ylabel("Count", fontsize=11)
            ax_d.set_title("D: Per-Spectrum AUROC Distribution", fontsize=13, fontweight="bold")
            ax_d.legend(fontsize=9)
        else:
            ax_d.text(0.5, 0.5, "Per-spectrum AUROC\nnot available", transform=ax_d.transAxes, ha="center", va="center", fontsize=12)
            ax_d.set_title("D: Per-Spectrum AUROC Distribution", fontsize=13, fontweight="bold")

        ax_d.grid(alpha=0.3)

        plt.savefig(output_dir / "confidence_signal_dashboard.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _create_individual_spectrum_plots(
        self,
        sequences: np.ndarray,
        spectra: np.ndarray,
        precursor_charges: np.ndarray,
        per_peak_confidence: np.ndarray,
        theoretical_mz_list: Optional[np.ndarray],
        theoretical_annotations_list: Optional[np.ndarray],
        theoretical_match_masks: Optional[np.ndarray],
        theoretical_match_idx_list: Optional[np.ndarray],
        frag_types: Optional[np.ndarray] = None,
        spectrum_confidences: Optional[np.ndarray] = None,
        matched_annotations: Optional[np.ndarray] = None,
        dataset_metrics: Optional[Dict[str, Any]] = None,
        spectrum_quality: Optional[np.ndarray] = None,
    ) -> None:
        """Create individual spectrum plots with confidence scores.

        Args:
            sequences: Array of peptide sequences
            spectra: Array of spectra (N, L, 2) with normalized m/z and intensity
            precursor_charges: Array of precursor charges
            per_peak_confidence: Array of per-peak confidence scores (N, L)
            theoretical_mz_list: Precomputed theoretical m/z arrays (object array)
            theoretical_annotations_list: Precomputed theoretical annotations (object array)
            theoretical_match_masks: Precomputed theoretical match masks (object array)
            theoretical_match_idx_list: Precomputed theoretical match indices (object array)
            frag_types: Optional array of fragmentation types
            spectrum_confidences: Optional array of per-spectrum confidence scores (N,)
            matched_annotations: Optional array of per-peak annotation lists (object array)
            dataset_metrics: Optional dataset-level metrics dict (for optimal_threshold line)
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available, skipping individual spectrum plots")
            return

        # Create subdirectory for individual plots
        plot_dir = Path(self.output_dir) / "individual_spectra"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Build candidate indices, filtered by the quality gate when available
        gate_active = (self.min_backbone_coverage > 0 or self.min_fragment_groups > 0) and spectrum_quality is not None
        if gate_active and spectrum_quality is not None:
            passing: list[int] = []
            for i in range(len(sequences)):
                sq = spectrum_quality[i] if i < len(spectrum_quality) else None
                if not isinstance(sq, dict):
                    continue
                bc = sq.get("backbone_coverage", 0.0)
                ng = sq.get("n_fragment_groups", 0)
                if bc >= self.min_backbone_coverage and ng >= self.min_fragment_groups:
                    passing.append(i)
                if len(passing) >= self.max_individual_plots:
                    break
            if not passing:
                logger.warning(
                    "No spectra pass the quality gate "
                    f"(min_backbone_coverage={self.min_backbone_coverage}, "
                    f"min_fragment_groups={self.min_fragment_groups}); skipping individual plots"
                )
                return
            indices = np.array(passing, dtype=np.int64)
        else:
            n_spectra = min(len(sequences), self.max_individual_plots)
            if n_spectra == 0:
                logger.warning("No spectra available for individual plotting")
                return
            indices = np.arange(n_spectra)

        for _plot_idx, spec_idx in enumerate(tqdm(indices, desc="Creating individual plots", disable=True)):
            try:
                # Extract data for this spectrum
                sequence = sequences[spec_idx]
                spectrum = spectra[spec_idx]  # (L, 2)
                precursor_charge = int(precursor_charges[spec_idx])
                confidence = per_peak_confidence[spec_idx]  # (L,)
                frag_type = frag_types[spec_idx] if frag_types is not None else None
                spectrum_conf = spectrum_confidences[spec_idx] if spectrum_confidences is not None else None

                # Get precomputed theoretical data
                theo_mz = theoretical_mz_list[spec_idx] if theoretical_mz_list is not None else None
                theo_annotations = theoretical_annotations_list[spec_idx] if theoretical_annotations_list is not None else None
                signal_mask = theoretical_match_masks[spec_idx] if theoretical_match_masks is not None else None
                match_idx = theoretical_match_idx_list[spec_idx] if theoretical_match_idx_list is not None else None

                # Get per-peak annotations: prefer matched_annotations, fallback to reconstruction
                peak_annotations = None
                if matched_annotations is not None and spec_idx < len(matched_annotations):
                    peak_annotations = matched_annotations[spec_idx]

                # Filter valid peaks (non-zero m/z)
                valid_mask = spectrum[:, 0] > 0
                mz = spectrum[valid_mask, 0] * self.max_mz  # Denormalize
                intensity = spectrum[valid_mask, 1]
                peak_confidence = confidence[valid_mask]

                # Filter annotations to valid peaks
                if peak_annotations is not None and len(peak_annotations) >= len(mz):
                    # peak_annotations is aligned with valid peaks already from embedding_io
                    # But if it was stored for full spectrum, filter
                    if len(peak_annotations) > len(mz):
                        peak_annotations = (
                            [peak_annotations[i] for i in range(len(peak_annotations)) if valid_mask[i]]
                            if len(peak_annotations) == len(valid_mask)
                            else peak_annotations[: len(mz)]
                        )
                    else:
                        peak_annotations = list(peak_annotations[: len(mz)])
                elif peak_annotations is not None:
                    peak_annotations = list(peak_annotations)

                # Fallback: reconstruct annotations from match_idx + theo_annotations
                if peak_annotations is None and match_idx is not None and theo_annotations is not None and signal_mask is not None:
                    peak_annotations = [None] * len(mz)
                    for i in range(len(mz)):
                        if signal_mask[i] and match_idx[i] >= 0 and match_idx[i] < len(theo_annotations):
                            peak_annotations[i] = theo_annotations[match_idx[i]]

                # Skip if too few peaks
                if len(mz) < 5:
                    continue

                seq_display = str(sequence)[:30] if sequence else "unknown"
                self._plot_single_spectrum_with_confidence(
                    mz=mz,
                    intensity=intensity,
                    confidence=peak_confidence,
                    signal_mask=signal_mask,
                    match_idx=match_idx,
                    sequence=seq_display,
                    precursor_charge=precursor_charge,
                    frag_type=frag_type,
                    theo_mz=theo_mz,
                    theo_annotations=theo_annotations,
                    spectrum_confidence=spectrum_conf,
                    save_path=plot_dir / f"spectrum_{spec_idx:04d}_{seq_display[:20].replace('/', '_').replace('[', '(').replace(']', ')')}.png",
                    matched_annotations=peak_annotations,
                    dataset_metrics=dataset_metrics,
                )

            except Exception as e:
                logger.warning(f"Failed to create plot for spectrum {spec_idx}: {e}")
                continue

    def _plot_single_spectrum_with_confidence(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        confidence: np.ndarray,
        signal_mask: Optional[np.ndarray],
        match_idx: Optional[np.ndarray],
        sequence: str,
        precursor_charge: int,
        frag_type: Optional[str],
        theo_mz: Optional[np.ndarray],
        theo_annotations: Optional[list],
        spectrum_confidence: Optional[float],
        save_path: Path,
        matched_annotations: Optional[list] = None,
        dataset_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Plot a 3-panel spectrum figure: m/z view, index view, confidence panel.

        Args:
            mz: M/z values (denormalized)
            intensity: Intensity values (normalized)
            confidence: Per-peak confidence scores
            signal_mask: Boolean mask for theoretical peaks
            match_idx: Precomputed match indices
            sequence: Peptide sequence
            precursor_charge: Precursor charge state
            frag_type: Fragmentation type
            theo_mz: Theoretical m/z values
            theo_annotations: Theoretical ion annotations
            spectrum_confidence: Per-spectrum confidence score
            save_path: Path to save the figure
            matched_annotations: Per-peak annotation strings (aligned with mz)
            dataset_metrics: Dataset-level metrics (for optimal_threshold line)
        """
        n_valid = len(mz)

        # --- Build per-peak categories and annotation lists ---
        peak_categories: list[Any] = []
        peak_ann_display = []  # formatted annotations for display

        if matched_annotations is not None:
            for i in range(n_valid):
                ann = matched_annotations[i] if i < len(matched_annotations) else None
                ann_str = str(ann) if ann is not None and str(ann) != "None" else ""
                cat = categorize_ion(ann_str)
                peak_categories.append(cat)
                peak_ann_display.append(format_annotation_display(ann_str) if ann_str else "")
        elif signal_mask is not None and match_idx is not None and theo_annotations is not None:
            # Fallback: reconstruct from match_idx
            for i in range(n_valid):
                if signal_mask[i] and match_idx[i] >= 0 and match_idx[i] < len(theo_annotations):
                    ann = theo_annotations[match_idx[i]]
                    ann_str = str(ann) if ann else ""
                    peak_categories.append(categorize_ion(ann_str))
                    peak_ann_display.append(format_annotation_display(ann_str) if ann_str else "")
                else:
                    peak_categories.append("unannotated")
                    peak_ann_display.append("")
        else:
            peak_categories = ["unannotated"] * n_valid
            peak_ann_display = [""] * n_valid

        # Normalise intensity to [0, 1] for consistent display
        max_int = intensity.max() if n_valid > 0 else 1.0
        norm_int = intensity / max_int if max_int > 0 else intensity

        # --- Suptitle ---
        suptitle_parts = [sequence]
        if precursor_charge:
            suptitle_parts.append(f"z={precursor_charge}")
        if frag_type and str(frag_type) != "None":
            suptitle_parts.append(str(frag_type))
        if spectrum_confidence is not None:
            suptitle_parts.append(f"Spectrum Conf: {spectrum_confidence:.4f}")

        # --- Create 3-panel figure ---
        # Use a 3-column grid: [main axes | thin colorbar gap | colorbar]
        # so that panels 2 and 3 share the exact same main-axes width.
        fig = plt.figure(figsize=(20, 18))
        gs = fig.add_gridspec(
            3,
            2,
            height_ratios=[3, 4, 4],
            width_ratios=[1, 0.02],  # narrow column for colorbar
            hspace=0.30,
            wspace=0.02,
            top=0.96,  # reduce gap between suptitle and first panel
        )
        fig.suptitle(" | ".join(suptitle_parts), fontsize=13, fontweight="bold", y=0.985)

        # ===== Panel 1: m/z Spectrum with Ion-Type Coloring =====
        ax1 = fig.add_subplot(gs[0, :])  # span both columns (no colorbar)
        self._draw_mz_panel(ax1, mz, norm_int, peak_categories, peak_ann_display, n_valid)

        # ===== Panel 2: Index-Based View with Ion-Type Coloring =====
        ax2 = fig.add_subplot(gs[1, 0])
        self._draw_index_panel(ax2, mz, norm_int, peak_categories, peak_ann_display, n_valid)
        # Hide the spare cell next to panel 2
        ax2_spare = fig.add_subplot(gs[1, 1])
        ax2_spare.axis("off")

        # ===== Panel 3: Index-Based Confidence Coloring =====
        ax3 = fig.add_subplot(gs[2, 0])
        ax_cbar = fig.add_subplot(gs[2, 1])  # dedicated colorbar axes
        self._draw_confidence_panel(
            ax3,
            ax_cbar,
            mz,
            norm_int,
            confidence,
            peak_categories,
            peak_ann_display,
            n_valid,
            dataset_metrics,
        )

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    # -----------------------------------------------------------------
    #  Panel drawing helpers
    # -----------------------------------------------------------------

    def _draw_mz_panel(
        self,
        ax: Any,
        mz: np.ndarray,
        norm_int: np.ndarray,
        peak_categories: list,
        peak_ann_display: list,
        n_valid: int,
    ) -> None:
        """Panel 1: m/z spectrum with ion-type coloring (stem plot)."""
        # Plot unannotated first (background), then annotated on top
        for layer in ("unannotated", "annotated"):
            for i in range(n_valid):
                cat = peak_categories[i]
                if layer == "unannotated" and cat != "unannotated":
                    continue
                if layer == "annotated" and cat == "unannotated":
                    continue
                color, alpha = CATEGORY_COLORS.get(cat, ("#9467bd", 0.8))
                ax.plot([mz[i], mz[i]], [0, norm_int[i]], color=color, linewidth=1.2, alpha=alpha, zorder=2 if cat == "unannotated" else 3)
                ax.scatter([mz[i]], [norm_int[i]], color=color, s=20, alpha=alpha, zorder=2 if cat == "unannotated" else 3)

        # Annotation labels — with intensity threshold + m/z anti-collision
        max_int = norm_int.max() if n_valid > 0 else 1.0
        min_label_intensity = max_int * 0.03
        min_mz_gap = 20.0

        placed_mz: list[float] = []
        # Annotated peaks sorted by intensity descending
        annotated = [
            (i, mz[i], norm_int[i], peak_ann_display[i]) for i in range(n_valid) if peak_categories[i] != "unannotated" and peak_ann_display[i]
        ]
        annotated.sort(key=lambda t: t[2], reverse=True)

        for idx, m, inten, display in annotated:
            if inten < min_label_intensity:
                continue
            if any(abs(m - p) < min_mz_gap for p in placed_mz):
                continue
            placed_mz.append(m)
            cat = peak_categories[idx]
            text_color = TEXT_COLORS.get(cat, "black")
            ax.annotate(
                display,
                xy=(m, inten),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=text_color,
                rotation=90,
                alpha=0.9,
                fontweight="bold",
            )

        # Top-20 unannotated peaks labeled with m/z in gray italic
        unannotated_indices = [i for i in range(n_valid) if peak_categories[i] == "unannotated"]
        if unannotated_indices:
            unannotated_arr = np.array(unannotated_indices)
            top_k = min(20, len(unannotated_arr))
            top_unann = unannotated_arr[np.argsort(norm_int[unannotated_arr])[-top_k:]]
            for i in top_unann:
                if norm_int[i] < min_label_intensity:
                    continue
                if any(abs(mz[i] - p) < min_mz_gap for p in placed_mz):
                    continue
                placed_mz.append(mz[i])
                ax.annotate(
                    f"{mz[i]:.1f}",
                    xy=(mz[i], norm_int[i]),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=6,
                    color="#555555",
                    rotation=90,
                    alpha=0.7,
                    fontstyle="italic",
                )

        ax.set_xlabel("m/z", fontsize=11, fontweight="bold")
        ax.set_ylabel("Normalized Intensity", fontsize=11, fontweight="bold")
        ax.set_title("m/z Spectrum with Ion-Type Coloring", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(50, self.max_mz)
        ax.set_ylim(0, max_int * 1.45)

    def _draw_index_panel(
        self,
        ax: Any,
        mz: np.ndarray,
        norm_int: np.ndarray,
        peak_categories: list,
        peak_ann_display: list,
        n_valid: int,
    ) -> None:
        """Panel 2: index-based bar plot with ion-type coloring."""
        x_pos = np.arange(n_valid)

        # Group by category for efficient plotting
        category_groups: dict[str, list[int]] = {}
        for i in range(n_valid):
            cat = peak_categories[i]
            category_groups.setdefault(cat, []).append(i)

        # Plot unannotated first
        unann = category_groups.get("unannotated", [])
        if unann:
            arr = np.array(unann)
            color, alpha = CATEGORY_COLORS["unannotated"]
            ax.bar(x_pos[arr], norm_int[arr], color=color, alpha=alpha, width=1.0, linewidth=0, label="unannotated")

        # Then annotated categories
        for cat in sorted(set(peak_categories) - {"unannotated"}):
            idxs = category_groups.get(cat, [])
            if not idxs:
                continue
            arr = np.array(idxs)
            color, alpha = CATEGORY_COLORS.get(cat, ("#9467bd", 0.8))
            ax.bar(x_pos[arr], norm_int[arr], color=color, alpha=alpha, width=1.0, linewidth=0, label=cat)

        # Label every annotated peak
        for i in range(n_valid):
            if peak_categories[i] != "unannotated" and peak_ann_display[i]:
                ax.annotate(
                    peak_ann_display[i],
                    xy=(i, norm_int[i]),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    rotation=90,
                    alpha=0.9,
                    fontweight="bold",
                    color="black",
                )

        # Top-20 unannotated peaks labeled with m/z
        if unann:
            unannotated_arr = np.array(unann)
            top_k = min(20, len(unannotated_arr))
            top_unann = unannotated_arr[np.argsort(norm_int[unannotated_arr])[-top_k:]]
            for i in top_unann:
                ax.annotate(
                    f"{mz[i]:.2f}",
                    xy=(i, norm_int[i]),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=6,
                    rotation=90,
                    alpha=0.7,
                    fontstyle="italic",
                    color="#555555",
                )

        # X-tick labels at regular intervals
        tick_step = max(1, n_valid // 20)
        tick_pos = np.arange(0, n_valid, tick_step)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([f"{mz[i]:.0f}" for i in tick_pos], fontsize=8, rotation=45, ha="right")
        ax.set_xlim(-1, n_valid)
        max_int = norm_int.max() if n_valid > 0 else 1.0
        ax.set_ylim(0, max_int * 1.3)

        ax.set_xlabel("m/z (at peak index)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Normalized Intensity", fontsize=11, fontweight="bold")
        ax.set_title("Index-Based View with Ion-Type Coloring", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

    def _draw_confidence_panel(
        self,
        ax: Any,
        ax_cbar: Any,
        mz: np.ndarray,
        norm_int: np.ndarray,
        confidence: np.ndarray,
        peak_categories: list,
        peak_ann_display: list,
        n_valid: int,
        dataset_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Panel 3: index-based view with bars colored by confidence score.

        Same spectrum as Panel 2 (bar height = intensity) but bar colour encodes
        the per-peak confidence via a viridis colormap.  Annotated peaks get a
        coloured edge so they are distinguishable at a glance.

        Args:
            ax: Main axes for the bar plot.
            ax_cbar: Dedicated axes for the colorbar (keeps main axes width
                     identical to Panel 2).
        """
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        x_pos = np.arange(n_valid)

        # Map confidence to colormap
        norm = Normalize(vmin=confidence.min(), vmax=confidence.max())
        cmap = plt.cm.viridis

        # Determine annotated / unannotated indices
        is_annotated = np.array([c != "unannotated" for c in peak_categories])
        ann_idx = np.where(is_annotated)[0]
        unann_idx = np.where(~is_annotated)[0]

        # Draw bars: unannotated first (background), then annotated on top
        for i in unann_idx:
            ax.bar(x_pos[i], norm_int[i], color=cmap(norm(confidence[i])), alpha=0.85, width=1.0, linewidth=0)
        for i in ann_idx:
            ax.bar(x_pos[i], norm_int[i], color=cmap(norm(confidence[i])), alpha=0.85, width=1.0, edgecolor="red", linewidth=0.8)

        # Annotation labels — same as Panel 2
        for i in range(n_valid):
            if peak_categories[i] != "unannotated" and peak_ann_display[i]:
                ax.annotate(
                    peak_ann_display[i],
                    xy=(i, norm_int[i]),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    rotation=90,
                    alpha=0.9,
                    fontweight="bold",
                    color="black",
                )

        # Top-20 unannotated peaks labeled with m/z
        if len(unann_idx) > 0:
            top_k = min(20, len(unann_idx))
            top_unann = unann_idx[np.argsort(norm_int[unann_idx])[-top_k:]]
            for i in top_unann:
                ax.annotate(
                    f"{mz[i]:.2f}",
                    xy=(i, norm_int[i]),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=6,
                    rotation=90,
                    alpha=0.7,
                    fontstyle="italic",
                    color="#555555",
                )

        # Mean confidence
        mean_ann = float(np.mean(confidence[ann_idx])) if len(ann_idx) > 0 else 0.0
        mean_unann = float(np.mean(confidence[unann_idx])) if len(unann_idx) > 0 else 0.0

        # Colorbar in dedicated axes (does not steal width from main axes)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, cax=ax_cbar)
        ax_cbar.set_ylabel("Confidence", fontsize=9, rotation=270, labelpad=12)

        # Legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=cmap(0.7), edgecolor="red", linewidth=1.2, label=f"Annotated (mean conf {mean_ann:.3f})"),
            Patch(facecolor=cmap(0.3), alpha=0.85, label=f"Unannotated (mean conf {mean_unann:.3f})"),
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

        # X-tick labels at regular intervals
        tick_step = max(1, n_valid // 20)
        tick_pos = np.arange(0, n_valid, tick_step)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([f"{mz[i]:.0f}" for i in tick_pos], fontsize=8, rotation=45, ha="right")
        ax.set_xlim(-1, n_valid)
        max_int = norm_int.max() if n_valid > 0 else 1.0
        ax.set_ylim(0, max_int * 1.3)

        gap = mean_ann - mean_unann
        ax.set_title(
            f"Confidence Coloring | Gap: {gap:.4f} | Annotated Mean: {mean_ann:.4f} | Unannotated Mean: {mean_unann:.4f}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("m/z (at peak index)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Normalized Intensity", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

    def _save_results(
        self,
        metrics: Dict[str, float],
        n_analyzed: int,
        n_total: int,
    ) -> None:
        """Save results to JSON file.

        Args:
            metrics: Computed metrics
            n_analyzed: Number of spectra successfully analyzed
            n_total: Total number of spectra
        """
        output_path = Path(self.output_dir) / "confidence_signal_metrics.json"

        results: dict[str, Any] = {
            "task": "ConfidenceSignalAnalysisTask",
            "description": "Analysis of model confidence alignment with theoretical signal peaks",
            "n_spectra_analyzed": n_analyzed,
            "n_spectra_total": n_total,
            "metrics": {
                k: v
                for k, v in metrics.items()
                if k not in ["roc_curve", "pr_curve"]  # Exclude curves from summary
            },
            "note": "Theoretical spectra are precomputed during embedding generation (see evaluation config: theoretical_spectrum)",
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    def _compute_quality_correlation(
        self,
        confidence_scores: np.ndarray,
        is_theoretical: np.ndarray,
        spectrum_boundaries: np.ndarray,
        spectrum_quality: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Compute correlation between per-spectrum mean confidence and annotation quality.

        Tests whether the model is more confident on well-structured spectra
        (high annotation ratio) — validating confidence as an unsupervised
        quality metric.

        Args:
            confidence_scores: Flat array of per-peak confidence scores
            is_theoretical: Flat boolean array (True = annotated peak)
            spectrum_boundaries: CSR-style boundaries for per-spectrum slicing
            spectrum_quality: Per-spectrum quality dicts (not used directly —
                index alignment with boundaries is unreliable after filtering)

        Returns:
            Dictionary with Spearman correlations, or empty dict if insufficient data.
        """
        from scipy.stats import spearmanr

        n_spectra = len(spectrum_boundaries) - 1
        mean_confidences: list[Any] = []
        annotation_ratios: list[Any] = []
        n_peaks_per_spectrum: list[Any] = []

        for i in range(n_spectra):
            start = spectrum_boundaries[i]
            end = spectrum_boundaries[i + 1]
            if end <= start:
                continue

            spec_conf = confidence_scores[start:end]
            spec_theo = is_theoretical[start:end]
            mean_conf = float(np.mean(spec_conf))
            ann_ratio = float(spec_theo.mean())

            mean_confidences.append(mean_conf)
            annotation_ratios.append(ann_ratio)
            n_peaks_per_spectrum.append(end - start)

        mean_confidences = np.array(mean_confidences)
        annotation_ratios = np.array(annotation_ratios)
        n_peaks_arr = np.array(n_peaks_per_spectrum)

        if len(mean_confidences) < 20:
            logger.info(f"  Quality correlation: too few spectra ({len(mean_confidences)}) — skipping")
            return {}

        results: Dict[str, Any] = {"n_spectra": len(mean_confidences)}

        # Correlation: confidence vs annotation ratio (fraction of peaks that
        # are theoretically matched — computed directly from is_theoretical,
        # no index alignment issues)
        rho_ann, p_ann = spearmanr(mean_confidences, annotation_ratios)
        results["confidence_vs_annotation_ratio"] = {
            "spearman_rho": float(rho_ann),
            "p_value": float(p_ann),
        }

        # Correlation: confidence vs number of peaks (controls for spectrum
        # complexity — more peaks = more fragment ions = potentially higher quality)
        rho_npeaks, p_npeaks = spearmanr(mean_confidences, n_peaks_arr)
        results["confidence_vs_n_peaks"] = {
            "spearman_rho": float(rho_npeaks),
            "p_value": float(p_npeaks),
        }

        logger.info(f"  Quality correlation: conf↔annotation_ratio ρ={rho_ann:.4f}  conf↔n_peaks ρ={rho_npeaks:.4f}  (n={len(mean_confidences)})")

        return results

    def get_loggable_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extract metrics suitable for logging to MLflow.

        Returns 4 metrics for ablation comparison:
        - auroc: pooled signal/noise discrimination
        - per_spectrum_auroc_mean: per-spectrum discrimination (avoids Simpson's paradox)
        - intensity_auroc: baseline — how well intensity alone separates signal/noise
        - residual_auroc: confidence discrimination after regressing out intensity

        All other metrics (decomposed, ion-type breakdown, fragment position,
        Cohen's d, AP, Spearman correlations) are in the full JSON output.

        Args:
            results: Task results dictionary

        Returns:
            Dictionary of scalar metrics for logging
        """
        if "error" in results:
            return {}

        metrics = results.get("metrics", {})

        loggable = {
            "auroc": metrics.get("auroc", 0.0),
        }

        # Per-spectrum AUROC (avoids pooling bias)
        per_spec = metrics.get("per_spectrum_auroc", {})
        if per_spec:
            loggable["per_spectrum_auroc_mean"] = per_spec.get("mean", 0.0)

        # Intensity baseline and residual AUROC
        intensity = metrics.get("intensity_analysis", {})
        if intensity:
            loggable["intensity_auroc"] = intensity.get("intensity_auroc", 0.0)
            loggable["residual_auroc"] = intensity.get("residual_confidence_auroc", 0.0)

        return loggable
