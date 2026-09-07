"""Peak Type Classification Task for Foundation Model Evaluation.

This task evaluates whether peak-level embeddings capture information about
peak identity (fragment ion type, neutral losses, isotopes, etc.) by training
linear classifiers on peak embeddings to predict peak types.

Two classification tasks:
1. Binary: annotated vs unannotated peaks (derived from multiclass)
2. Multi-class: b-ion, y-ion, precursor, unannotated
   (4-class taxonomy grouping base ions, isotopes, and losses by parent series)

Key features:
- Uses peak-level embeddings (not spectrum-level)
- Handles severe class imbalance with balanced class weights
- Reports macro-F1, per-class metrics, and confusion matrices
- Filters out padded peaks using spectra_mask
- Filters low-quality spectra via backbone coverage quality gate
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from instanovo.__init__ import console
from instanovo_fm.eval.embed_eval_tasks import BaseTask
from instanovo_fm.utils.modifications import clean_peptide_sequence
from instanovo_fm.utils.peak_classification import (
    extract_charge_from_annotation,
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


class LinearClassifier(nn.Module):
    """Simple linear classifier for peak type classification.

    This is essentially logistic regression implemented in PyTorch,
    which allows us to leverage GPU acceleration.
    """

    def __init__(self, input_dim: int, num_classes: int):
        """Initialize the linear classifier.

        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            Logits (batch_size, num_classes)
        """
        return self.linear(x)


class PeakTypeClassificationTask(BaseTask):
    """Peak type classification task using peak-level embeddings.

    This task trains linear classifiers on peak embeddings to predict:
    1. Binary: Whether a peak is annotated (derived from multiclass)
    2. Multi-class: b-ion, y-ion, precursor, unannotated (4-class taxonomy
       grouping base ions, isotopes, and losses by parent ion series)

    Requires:
    - peak_embeddings in metadata (extracted via store_peak_embeddings=True)
    - theoretical matching results (feature_type, matched_annotation)
    - spectra_mask to filter padded peaks
    """

    name = "peaktypeclassificationtask"
    description = "Classify peak types using peak-level embeddings"
    requires_metadata = True
    requires_faiss = False
    requires_model = False  # Uses pre-computed peak embeddings

    def __init__(
        self,
        output_dir: str = "./peak_type_classification",
        max_samples: int = 10_000,
        test_size: float = 0.2,
        random_state: int = 42,
        use_grid_search: bool = False,  # Fixed L2 is fast; enable for ablation-quality runs
        l2_reg_values: Optional[List[float]] = None,
        max_epochs: int = 100,
        batch_size: int = 4096,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
        device: Optional[str] = None,
        create_plots: bool = True,
        min_backbone_coverage: float = 0.0,  # Backbone coverage quality gate (0 = disabled)
        min_fragment_groups: int = 0,  # Fragment group quality gate (0 = disabled)
        enable_classification: bool = True,  # Train linear probe classifiers
        enable_baselines: bool = False,  # Enable baseline comparisons (slow, for ablation-quality runs)
        enable_random_baseline: bool = False,  # Enable random embeddings baseline (memory intensive)
        max_prediction_plots: int = 5,  # Max spectra to plot with predictions
        enable_prediction_plots: bool = False,  # Per-spectrum prediction PNG plots (local debugging)
        enable_peak_umap: bool = True,  # Create peak-level UMAP scatter plots
        max_umap_samples: int = 75_000,  # Max peaks for UMAP (subsampled by spectrum)
        umap_n_neighbors: int = 30,  # UMAP n_neighbors
        umap_min_dist: float = 0.1,  # UMAP min_dist
        umap_metric: str = "cosine",  # UMAP distance metric
        umap_point_size: float = 4.0,  # Scatter marker size for UMAP plots
        umap_dpi: int = 300,  # DPI for UMAP plots
        enable_pairwise_similarity: bool = True,  # Pairwise cosine similarity analysis
        max_pairs_per_group: int = 10_000,  # Max pairs to sample per relation group
        enable_pretransformer_probe: bool = True,  # Pre- vs post-transformer comparison
        enable_embedding_diagnostics: bool = True,  # Embedding quality diagnostics (intrinsic)
        enable_diagnostics_classifier: bool = False,  # Also train classifier for confidence/entropy/margin
        enable_cross_spectrum_identity: bool = True,  # Cross-spectrum ion identity analysis
        cross_spectrum_max_pairs: int = 10_000,  # Max pairs for cross-spectrum similarity
        cross_spectrum_mz_tolerance: float = 0.5,  # Da tolerance for m/z-matched control group
        max_fragment_position: int = 30,  # Cap fragment ladder positions (higher positions are rare)
        enable_fragment_umap: bool = True,  # Additional UMAP using only fragment ions (b/y)
        fragment_umap_frag_types: Optional[List[str]] = None,  # Filter to these frag types (e.g., ["HCD"])
        fragment_umap_detectors: Optional[List[str]] = None,  # Filter to these detectors (e.g., ["Orbitrap"])
        fragment_umap_instruments: Optional[List[str]] = None,  # Filter to these instruments (e.g., ["Q Exactive HF"])
        enable_chemistry_probes: bool = True,  # Enable chemistry understanding probes
        **kwargs,
    ):
        """Initialize the peak type classification task.

        Args:
            output_dir: Directory to save results
            max_samples: Maximum number of spectra to use (peaks will be much more)
            test_size: Fraction of data for test set
            random_state: Random seed for reproducibility
            use_grid_search: Whether to use grid search for hyperparameter tuning
            l2_reg_values: L2 regularization strengths to try (None = default)
            max_epochs: Maximum training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            early_stopping_patience: Number of epochs without improvement before stopping
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            create_plots: Whether to create visualization plots
            min_backbone_coverage: Minimum backbone cleavage coverage (0-1, 0 = disabled)
            min_fragment_groups: Minimum unique fragment ion groups (0 = disabled)
        """
        super().__init__(output_dir=output_dir, **kwargs)
        self.output_dir = Path(output_dir)
        self.max_samples = max_samples
        self.test_size = test_size
        self.random_state = random_state
        self.use_grid_search = use_grid_search
        self.l2_reg_values = l2_reg_values or [0.0001, 0.001, 0.01, 0.1, 1.0]
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.create_plots = create_plots
        self.min_backbone_coverage = min_backbone_coverage
        self.min_fragment_groups = min_fragment_groups
        self.enable_classification = enable_classification
        self.enable_baselines = enable_baselines
        self.enable_random_baseline = enable_random_baseline
        self.max_prediction_plots = max_prediction_plots
        self.enable_prediction_plots = enable_prediction_plots
        self.enable_peak_umap = enable_peak_umap
        self.max_umap_samples = max_umap_samples
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.umap_metric = umap_metric
        self.umap_point_size = umap_point_size
        self.umap_dpi = umap_dpi
        self.enable_pairwise_similarity = enable_pairwise_similarity
        self.max_pairs_per_group = max_pairs_per_group
        self.enable_pretransformer_probe = enable_pretransformer_probe
        self.enable_embedding_diagnostics = enable_embedding_diagnostics
        self.enable_diagnostics_classifier = enable_diagnostics_classifier
        self.enable_cross_spectrum_identity = enable_cross_spectrum_identity
        self.cross_spectrum_max_pairs = cross_spectrum_max_pairs
        self.cross_spectrum_mz_tolerance = cross_spectrum_mz_tolerance
        self.max_fragment_position = max_fragment_position
        self.enable_fragment_umap = enable_fragment_umap
        self.fragment_umap_frag_types = fragment_umap_frag_types
        self.fragment_umap_detectors = fragment_umap_detectors
        self.fragment_umap_instruments = fragment_umap_instruments
        self.enable_chemistry_probes = enable_chemistry_probes

        # Device for linear classifier training and inference.
        # _train_pytorch_classifier defaults to CPU/1-thread internally
        # (optimal for single-layer models), but self.device is used for
        # test-set inference and can be overridden via config.
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # No verbose __init__ logging — config is visible in the eval config dump

    def run(self, emb: np.ndarray, meta: Dict[str, np.ndarray], faiss_index: Any = None) -> Dict[str, Any]:
        """Run peak type classification evaluation.

        Args:
            emb: Spectrum-level embeddings (not used, we use peak embeddings from metadata)
            meta: Metadata dictionary containing peak_embeddings and theoretical matching results
            faiss_index: Not used

        Returns:
            Dictionary containing classification results
        """
        start_time = time.time()

        # Validate inputs
        self.validate_inputs(emb, meta, faiss_index)

        # Check for required metadata fields
        required_fields = ["peak_embeddings", "feature_type", "spectra_mask"]
        missing_fields = [f for f in required_fields if f not in meta]
        if missing_fields:
            error_msg = (
                f"Missing required metadata fields: {missing_fields}. "
                "Make sure to enable theoretical spectrum generation and peak embeddings storage."
            )
            logger.error(error_msg)
            return {"error": error_msg, "success": False}

        # Extract peak embeddings and labels
        peak_data = self._extract_peak_data(meta)

        if peak_data is None:
            return {"error": "Failed to extract peak data", "success": False}

        X = peak_data["embeddings"]
        binary_labels = peak_data["binary_labels"]
        multiclass_labels = peak_data["multiclass_labels"]

        # Classification (linear probe) — can be disabled to only run UMAP
        multiclass_results = {}
        binary_results = {}
        baseline_results = {}

        if self.enable_classification:
            # Build per-peak group IDs for splitting.
            # Group spectra by peptide sequence so that duplicate sequences never
            # leak between train/val/test splits.
            split_groups = self._build_peptide_split_groups(peak_data["spectrum_indices_per_peak"], meta)

            # Run baseline comparisons if enabled
            if self.enable_baselines:
                baseline_results = self._run_baseline_comparisons(peak_data, multiclass_labels, meta, split_groups)

            # Run multi-class classification with learned embeddings
            multiclass_results = self._run_multiclass_classification(X, multiclass_labels, split_groups)

            # Derive binary metrics from multiclass confusion matrix
            binary_results = self._derive_binary_from_multiclass(multiclass_results)

            # Create classification visualizations
            if self.create_plots and PLOTTING_AVAILABLE:
                self._create_visualizations(binary_results, multiclass_results)
                if self.enable_prediction_plots:
                    self._create_prediction_plots(peak_data, multiclass_results, meta)

            # Strip large per-peak arrays when prediction plots are disabled
            # (these are only consumed by _create_prediction_plots)
            if not self.enable_prediction_plots or not self.create_plots:
                for key in ["test_indices", "y_test_labels", "y_pred_labels", "y_pred_confidence"]:
                    multiclass_results.pop(key, None)
        else:
            logger.info("Classification disabled (enable_classification=false) — skipping linear probes")

        # Create peak-level UMAP visualizations
        umap_paths = []
        umap_diagnostics = {}
        if self.create_plots and PLOTTING_AVAILABLE and self.enable_peak_umap:
            umap_paths, umap_diagnostics = self._create_peak_umap_visualizations(peak_data)

        # Pairwise similarity + structural consistency analysis
        pairwise_results = {}
        structural_results = {}
        if self.enable_pairwise_similarity:
            pairwise_results, structural_results = self._run_pairwise_similarity_analysis(peak_data)

        # Cross-spectrum ion identity analysis
        cross_spectrum_results = {}
        if self.enable_cross_spectrum_identity:
            cross_spectrum_results = self._run_cross_spectrum_ion_identity(peak_data, meta)

            # Merge cross-spectrum and cross-charge groups into structural results
            for key in ("same_ion_cross_spectrum", "different_ion_similar_mz", "random_cross_spectrum"):
                stats = cross_spectrum_results.get(key, {})
                if isinstance(stats, dict) and stats.get("mean") is not None:
                    structural_results[key] = stats
            cc = cross_spectrum_results.get("cross_charge", {})
            cc_sim = cc.get("cross_charge_similarity", {})
            if cc_sim.get("mean") is not None:
                structural_results["cross_charge"] = cc_sim
            auroc = cross_spectrum_results.get("auroc_same_vs_different_mz")
            if auroc is not None:
                structural_results["_cross_spectrum_auroc"] = auroc

            # Re-render pairwise plot with all groups
            if self.create_plots and PLOTTING_AVAILABLE and self.enable_pairwise_similarity:
                relation_groups = pairwise_results.get("relation_groups", {})
                pairwise_plot = self._plot_pairwise_similarity(relation_groups, structural_results)
                pairwise_results["plot_path"] = pairwise_plot
                structural_results["plot_path"] = pairwise_plot

        # Chemistry understanding probes
        chemistry_probe_results = {}
        if self.enable_chemistry_probes:
            chemistry_probe_results = self._run_chemistry_probes(peak_data, meta, split_groups if self.enable_classification else None)

            # Add complementary pair stats to pairwise plot (re-render)
            comp = chemistry_probe_results.get("complementary_by_pairs", {})
            if comp and not comp.get("skipped") and not comp.get("error"):
                structural_results["complementary_by"] = {
                    "mean": comp["complementary_mean_similarity"],
                    "q25": comp["complementary_mean_similarity"],  # no IQR available
                    "q75": comp["complementary_mean_similarity"],
                    "n_pairs": comp["n_complementary_pairs"],
                }
                if self.create_plots and PLOTTING_AVAILABLE and self.enable_pairwise_similarity:
                    relation_groups = pairwise_results.get("relation_groups", {})
                    pairwise_plot = self._plot_pairwise_similarity(relation_groups, structural_results)
                    pairwise_results["plot_path"] = pairwise_plot
                    structural_results["plot_path"] = pairwise_plot

            # Create chemistry probes figure
            if self.create_plots and PLOTTING_AVAILABLE:
                self._create_chemistry_probes_figure(chemistry_probe_results)

        # Pre-transformer vs post-transformer comparison
        pretransformer_results = {}
        if self.enable_pretransformer_probe and self.enable_classification:
            pretransformer_results = self._run_pretransformer_probe(peak_data, split_groups, multiclass_results)

        # Free pre-transformer embeddings — no longer needed after UMAP
        # correction and pretransformer probe. Saves ~1.5 GB for 500k peaks.
        if peak_data.get("embeddings_pretransformer") is not None:
            del peak_data["embeddings_pretransformer"]

        # Embedding quality diagnostics
        diagnostics_results = {}
        if self.enable_embedding_diagnostics and self.enable_classification:
            diagnostics_results = self._run_embedding_diagnostics(peak_data, split_groups)

        execution_time = time.time() - start_time

        results = {
            "task_name": self.name,
            "num_spectra": int(len(meta["peak_embeddings"])),
            "num_peaks": int(len(X)),
            "peak_embedding_dim": int(X.shape[1]),
            "execution_time": float(execution_time),
            "binary_classification": binary_results,
            "multiclass_classification": multiclass_results,
            "baseline_comparisons": baseline_results if self.enable_baselines else None,
            "pairwise_similarity": pairwise_results,
            "structural_consistency": structural_results,
            "cross_spectrum_identity": cross_spectrum_results,
            "chemistry_probes": chemistry_probe_results,
            "pretransformer_probe": pretransformer_results,
            "umap_diagnostics": umap_diagnostics,
            "embedding_diagnostics": diagnostics_results,
            "config": {
                "max_samples": int(self.max_samples),
                "test_size": float(self.test_size),
                "random_state": int(self.random_state),
                "use_grid_search": bool(self.use_grid_search),
                "l2_reg_values": [float(l2) for l2 in self.l2_reg_values],
                "max_epochs": int(self.max_epochs),
                "batch_size": int(self.batch_size),
                "learning_rate": float(self.learning_rate),
                "early_stopping_patience": int(self.early_stopping_patience),
                "device": str(self.device),
                "min_backbone_coverage": float(self.min_backbone_coverage),
                "min_fragment_groups": int(self.min_fragment_groups),
                "enable_baselines": bool(self.enable_baselines),
                "enable_random_baseline": bool(self.enable_random_baseline),
                "max_prediction_plots": int(self.max_prediction_plots),
                "enable_prediction_plots": bool(self.enable_prediction_plots),
                "enable_peak_umap": bool(self.enable_peak_umap),
                "enable_pairwise_similarity": bool(self.enable_pairwise_similarity),
                "max_pairs_per_group": int(self.max_pairs_per_group),
                "enable_pretransformer_probe": bool(self.enable_pretransformer_probe),
                "enable_embedding_diagnostics": bool(self.enable_embedding_diagnostics),
                "enable_diagnostics_classifier": bool(self.enable_diagnostics_classifier),
                "enable_cross_spectrum_identity": bool(self.enable_cross_spectrum_identity),
                "cross_spectrum_max_pairs": int(self.cross_spectrum_max_pairs),
                "cross_spectrum_mz_tolerance": float(self.cross_spectrum_mz_tolerance),
                "enable_chemistry_probes": bool(self.enable_chemistry_probes),
            },
            "umap_paths": umap_paths,
            "success": True,
        }

        # Save results
        self._save_results(results)

        return results

    def _extract_peak_data(self, meta: Dict[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
        """Extract peak embeddings and labels from metadata.

        Args:
            meta: Metadata dictionary

        Returns:
            Dictionary with keys:
            - embeddings: Peak embeddings (N_peaks, D)
            - binary_labels: Binary labels (N_peaks,) - 0=unannotated, 1=annotated
            - multiclass_labels: Multi-class labels (N_peaks,) - string labels
        """
        peak_embeddings = meta["peak_embeddings"]  # List of (L, D) arrays
        peak_embeddings_pre = meta.get("peak_embeddings_pretransformer", None)  # Pre-transformer
        feature_types = meta["feature_type"]  # List of lists
        matched_annotations = meta.get("matched_annotation", None)  # List of lists
        parent_annotations = meta.get("parent_annotation", None)  # List of lists
        spectrum_quality = meta.get("spectrum_quality", None)  # List of quality dicts
        spectra_masks = meta["spectra_mask"]  # (N, L) boolean array
        spectra = meta.get("spectra", None)  # Optional: (N, L, 2) arrays
        max_mz = float(meta.get("theoretical_max_mz", 2500.0))

        # Spectrum-level metadata for instrument filtering
        frag_types_meta = meta.get("frag_type", None)  # Per-spectrum fragmentation type
        search_detectors_meta = meta.get("search_detector", None)  # Per-spectrum detector
        search_instruments_meta = meta.get("search_instrument", None)  # Per-spectrum instrument

        # Sample spectra if needed
        n_spectra = len(peak_embeddings)
        if self.max_samples is not None and n_spectra > self.max_samples:
            np.random.seed(self.random_state)
            indices = np.random.choice(n_spectra, self.max_samples, replace=False)
            peak_embeddings = [peak_embeddings[i] for i in indices]
            if peak_embeddings_pre is not None:
                peak_embeddings_pre = [peak_embeddings_pre[i] for i in indices]
            feature_types = [feature_types[i] for i in indices]
            if matched_annotations is not None:
                matched_annotations = [matched_annotations[i] for i in indices]
            if parent_annotations is not None:
                parent_annotations = [parent_annotations[i] for i in indices]
            if spectrum_quality is not None:
                spectrum_quality = [spectrum_quality[i] for i in indices]
            spectra_masks = spectra_masks[indices]
            if spectra is not None:
                spectra = spectra[indices]
            if frag_types_meta is not None:
                frag_types_meta = frag_types_meta[indices] if hasattr(frag_types_meta, "__getitem__") else frag_types_meta
            if search_detectors_meta is not None:
                search_detectors_meta = search_detectors_meta[indices] if hasattr(search_detectors_meta, "__getitem__") else search_detectors_meta
            if search_instruments_meta is not None:
                search_instruments_meta = (
                    search_instruments_meta[indices] if hasattr(search_instruments_meta, "__getitem__") else search_instruments_meta
                )

        # Flatten peaks across all spectra (with quality filtering)
        all_peak_embeddings = []
        all_peak_embeddings_pre = []
        all_binary_labels = []
        all_multiclass_labels = []
        spectrum_indices_kept = []
        all_spectrum_indices = []
        all_peak_indices = []
        all_mz_values = []
        all_intensity_values = []
        all_feature_types = []
        all_matched_annotations = []
        all_parent_annotations = []
        all_frag_types = []
        all_search_detectors = []
        all_search_instruments = []

        n_filtered = 0  # Count filtered spectra

        # Build mapping from sampled indices to original indices
        if self.max_samples is not None and n_spectra > self.max_samples:
            original_indices = indices
        else:
            original_indices = list(range(len(peak_embeddings)))

        for spec_idx, (peak_emb, feat_types, spec_mask) in enumerate(zip(peak_embeddings, feature_types, spectra_masks, strict=False)):
            # peak_emb: (L, D)
            # feat_types: list of length L with values: "base", "loss", "isotope", "precursor", None
            # spec_mask: (L,) boolean - True for padding

            # Skip spectra where theoretical generation failed (feat_types is None)
            if feat_types is None:
                n_filtered += 1
                continue

            # Filter out padded peaks
            valid_mask = ~spec_mask  # True for valid peaks

            # Extract valid peaks
            valid_peak_emb = peak_emb[valid_mask]  # (N_valid, D)
            valid_feat_types = [feat_types[i] for i in range(len(feat_types)) if valid_mask[i]]

            # Backbone coverage quality gate
            if spectrum_quality is not None and spec_idx < len(spectrum_quality) and spectrum_quality[spec_idx] is not None:
                sq = spectrum_quality[spec_idx]
                bc = sq.get("backbone_coverage", 0.0)
                ng = sq.get("n_fragment_groups", 0)
                if bc < self.min_backbone_coverage or ng < self.min_fragment_groups:
                    n_filtered += 1
                    continue

            # Track that we kept this spectrum (use original index)
            spectrum_indices_kept.append(original_indices[spec_idx])

            # Extract matched annotations for this spectrum
            valid_annotations = None
            if matched_annotations is not None and spec_idx < len(matched_annotations):
                spec_annotations = matched_annotations[spec_idx]
                if spec_annotations is not None:
                    valid_annotations = [spec_annotations[i] for i in range(len(spec_annotations)) if i < len(spec_annotations) and valid_mask[i]]

            # Extract parent annotations for this spectrum
            valid_parents = None
            if parent_annotations is not None and spec_idx < len(parent_annotations):
                spec_parents = parent_annotations[spec_idx]
                if spec_parents is not None:
                    valid_parents = [spec_parents[i] for i in range(len(spec_parents)) if valid_mask[i]]

            # Create binary labels (0=unannotated, 1=annotated)
            binary_labels = np.array([0 if ft is None else 1 for ft in valid_feat_types], dtype=np.int32)

            # Create multi-class labels with b/y separation
            multiclass_labels = []
            for i, ft in enumerate(valid_feat_types):
                annotation = valid_annotations[i] if valid_annotations else None
                parent = valid_parents[i] if valid_parents else None

                label = self._parse_peak_label(ft, annotation, parent)
                multiclass_labels.append(label)

            all_peak_embeddings.append(valid_peak_emb)
            if peak_embeddings_pre is not None and spec_idx < len(peak_embeddings_pre):
                all_peak_embeddings_pre.append(peak_embeddings_pre[spec_idx][valid_mask])
            all_binary_labels.extend(binary_labels)
            all_multiclass_labels.extend(multiclass_labels)
            all_feature_types.extend(valid_feat_types)
            all_matched_annotations.extend(valid_annotations if valid_annotations else [None] * len(valid_feat_types))
            all_parent_annotations.extend(valid_parents if valid_parents else [None] * len(valid_feat_types))

            n_valid = valid_peak_emb.shape[0]
            all_spectrum_indices.extend([original_indices[spec_idx]] * n_valid)
            all_peak_indices.extend([i for i, is_valid in enumerate(valid_mask) if is_valid])

            if spectra is not None and spec_idx < len(spectra):
                spec = spectra[spec_idx]
                if spec.ndim == 2 and spec.shape[1] >= 2:
                    spec_mz = spec[:, 0][valid_mask] * max_mz
                    spec_intensity = spec[:, 1][valid_mask]
                    all_mz_values.extend(spec_mz.tolist())
                    all_intensity_values.extend(spec_intensity.tolist())
                else:
                    all_mz_values.extend([np.nan] * n_valid)
                    all_intensity_values.extend([np.nan] * n_valid)
            else:
                all_mz_values.extend([np.nan] * n_valid)
                all_intensity_values.extend([np.nan] * n_valid)

            # Propagate spectrum-level metadata to per-peak arrays
            ft_val = None
            if frag_types_meta is not None and spec_idx < len(frag_types_meta):
                ft_val = frag_types_meta[spec_idx]
                if isinstance(ft_val, (bytes, np.bytes_)):
                    ft_val = ft_val.decode("utf-8")
            all_frag_types.extend([ft_val] * n_valid)

            det_val = None
            if search_detectors_meta is not None and spec_idx < len(search_detectors_meta):
                det_val = search_detectors_meta[spec_idx]
                if isinstance(det_val, (bytes, np.bytes_)):
                    det_val = det_val.decode("utf-8")
            all_search_detectors.extend([det_val] * n_valid)

            inst_val = None
            if search_instruments_meta is not None and spec_idx < len(search_instruments_meta):
                inst_val = search_instruments_meta[spec_idx]
                if isinstance(inst_val, (bytes, np.bytes_)):
                    inst_val = inst_val.decode("utf-8")
            all_search_instruments.extend([inst_val] * n_valid)

        # Concatenate all peaks
        X = np.concatenate(all_peak_embeddings, axis=0)
        binary_labels = np.array(all_binary_labels, dtype=np.int32)
        multiclass_labels = np.array(all_multiclass_labels, dtype=object)

        # Pre-transformer embeddings (if available)
        X_pre = None
        if all_peak_embeddings_pre and len(all_peak_embeddings_pre) == len(all_peak_embeddings):
            X_pre = np.concatenate(all_peak_embeddings_pre, axis=0)

        # Compact summary
        n_kept = n_spectra - n_filtered
        unique, counts = np.unique(multiclass_labels, return_counts=True)
        dist_str = ", ".join(
            f"{lbl}: {cnt:,} ({cnt / len(multiclass_labels) * 100:.1f}%)"
            for lbl, cnt in sorted(zip(unique, counts, strict=False), key=lambda x: -x[1])
        )
        logger.info(
            f"  {len(X):,} peaks from {n_kept:,}/{n_spectra} spectra "
            f"(filtered {n_filtered})"
            f"{' | pre-transformer available' if X_pre is not None else ''}"
        )
        logger.info(f"  Classes: {dist_str}")

        return {
            "embeddings": X,
            "embeddings_pretransformer": X_pre,
            "binary_labels": binary_labels,
            "multiclass_labels": multiclass_labels,
            "spectrum_indices": spectrum_indices_kept,
            "spectrum_indices_per_peak": np.array(all_spectrum_indices, dtype=np.int32),
            "peak_indices_per_spectrum": np.array(all_peak_indices, dtype=np.int32),
            "mz_values": np.array(all_mz_values, dtype=np.float64),
            "intensity_values": np.array(all_intensity_values, dtype=np.float64),
            "feature_types": np.array(all_feature_types, dtype=object),
            "matched_annotations": np.array(all_matched_annotations, dtype=object),
            "parent_annotations": np.array(all_parent_annotations, dtype=object),
            "frag_types": np.array(all_frag_types, dtype=object),
            "search_detectors": np.array(all_search_detectors, dtype=object),
            "search_instruments": np.array(all_search_instruments, dtype=object),
        }

    def _parse_peak_label(self, feature_type: Optional[str], annotation: Optional[str], parent_annotation: Optional[str]) -> str:
        """Parse peak annotation into a 4-class taxonomy for HCD/CID spectra.

        Groups base ions, isotopes, and neutral losses by their parent ion
        series.  This reflects the physical reality that isotopes and losses
        are structurally derived from their parent fragment ion and should
        cluster together in embedding space.

        Args:
            feature_type: Feature type ("base", "loss", "isotope", "precursor", None)
            annotation: Matched annotation (e.g., "b3+", "y5++", "b3-H2O+")
            parent_annotation: Parent annotation for losses/isotopes

        Returns:
            One of: "b-ion", "y-ion", "precursor", "unannotated"
        """
        # Unannotated peaks
        if feature_type is None:
            return "unannotated"

        # Precursor ions (base, isotope, loss all grouped)
        if feature_type == "precursor":
            return "precursor"

        # For base, loss, and isotope, parse ion type from annotation.
        # Prefer parent annotation for loss/isotope if available.
        if feature_type in ("loss", "isotope") and parent_annotation and parent_annotation != "":
            ion_annotation = parent_annotation
        elif annotation and annotation != "":
            ion_annotation = annotation
        else:
            return "unannotated"  # No annotation available — treat as unannotated

        # Parse ion type from annotation
        if not isinstance(ion_annotation, str) or len(ion_annotation) == 0:
            return "unannotated"

        # Handle precursor annotations (M+nH) for isotope/loss of precursor
        if ion_annotation.startswith("M+") and "H" in ion_annotation:
            return "precursor"

        # Extract first character (ion type: b, y, a, c, x, z)
        try:
            ion_type_char = str(ion_annotation)[0].lower()
        except (IndexError, AttributeError, TypeError):
            return "unannotated"

        # Map to b/y (the dominant series in HCD/CID)
        # Rare ion types (a, c, x, z) are folded into unannotated
        if ion_type_char == "b":
            return "b-ion"
        if ion_type_char == "y":
            return "y-ion"

        # Rare ion types (a, c, x, z) folded into unannotated.
        # a-ions (~5.8% of HCD fragments) are structurally related to b-ions
        # (a_i = b_i - CO) but too sparse for a separate class.  This means
        # the "unannotated" class contains some chemically structured peaks,
        # slightly inflating the annotated-vs-unannotated separability.
        return "unannotated"

    def _run_baseline_comparisons(
        self,
        peak_data: Dict[str, np.ndarray],
        multiclass_labels: np.ndarray,
        meta: Dict[str, np.ndarray],
        split_groups: np.ndarray,
        max_baseline_peaks: int = 1_000_000,
        max_baseline_epochs: int = 50,
    ) -> Dict[str, Any]:
        """Run baseline comparisons to validate that learned embeddings add value.

        Uses multiclass classification only (binary is derived from multiclass).
        Baselines use fixed L2 and capped epochs for speed.

        Args:
            peak_data: Dictionary with peak data
            multiclass_labels: Multi-class labels for all peaks
            meta: Full metadata dictionary
            split_groups: Per-peak group IDs for peptide-aware splitting
            max_baseline_peaks: Maximum peaks for baselines (subsample if needed)
            max_baseline_epochs: Max epochs for baseline classifiers (capped for speed)

        Returns:
            Dictionary with baseline results
        """
        results = {}
        spectrum_groups = peak_data["spectrum_indices_per_peak"]

        # Check if we need to subsample for baselines (subsample at spectrum level)
        n_peaks = len(multiclass_labels)
        if n_peaks > max_baseline_peaks:
            unique_spectra = np.unique(spectrum_groups)
            np.random.seed(self.random_state)
            np.random.shuffle(unique_spectra)

            selected_spectra = set()
            peak_count = 0
            for s in unique_spectra:
                s_peaks = np.sum(spectrum_groups == s)
                if peak_count + s_peaks > max_baseline_peaks:
                    break
                selected_spectra.add(s)
                peak_count += s_peaks

            subsample_mask = np.isin(spectrum_groups, list(selected_spectra))
            multiclass_labels_sub = multiclass_labels[subsample_mask]
            split_groups_sub = split_groups[subsample_mask]
            subsampled = True
        else:
            subsample_mask = None
            multiclass_labels_sub = multiclass_labels
            split_groups_sub = split_groups
            subsampled = False

        # Temporarily cap epochs and disable grid search for baselines
        orig_max_epochs = self.max_epochs
        orig_grid_search = self.use_grid_search
        self.max_epochs = max_baseline_epochs
        self.use_grid_search = False

        # 1. Raw Features Baseline
        raw_features = self._extract_raw_features(peak_data, meta, subsample_mask)

        raw_multiclass = self._run_multiclass_classification(raw_features, multiclass_labels_sub, split_groups_sub)
        # Baseline results are never used for prediction plots — strip large arrays
        for key in ["test_indices", "y_test_labels", "y_pred_labels", "y_pred_confidence"]:
            raw_multiclass.pop(key, None)

        results["raw_features"] = {
            "feature_dim": int(raw_features.shape[1]),
            "n_peaks_used": int(len(multiclass_labels_sub)),
            "subsampled": subsampled,
            "multiclass_classification": raw_multiclass,
        }

        # 2. Random Embeddings Baseline (optional, memory intensive)
        if self.enable_random_baseline:
            learned_emb_sub = peak_data["embeddings"][subsample_mask] if subsample_mask is not None else peak_data["embeddings"]
            random_embeddings = self._generate_random_embeddings(learned_emb_sub.shape)
            random_multiclass = self._run_multiclass_classification(random_embeddings, multiclass_labels_sub, split_groups_sub)
            for key in ["test_indices", "y_test_labels", "y_pred_labels", "y_pred_confidence"]:
                random_multiclass.pop(key, None)
            results["random_embeddings"] = {
                "embedding_dim": int(random_embeddings.shape[1]),
                "n_peaks_used": int(len(multiclass_labels_sub)),
                "subsampled": subsampled,
                "multiclass_classification": random_multiclass,
            }
        else:
            results["random_embeddings"] = None

        # Restore original settings
        self.max_epochs = orig_max_epochs
        self.use_grid_search = orig_grid_search

        # Summary
        logger.info(f"  Baselines: raw_features F1={raw_multiclass['macro_f1']:.4f}")
        if self.enable_random_baseline and results["random_embeddings"] is not None:
            rm = results["random_embeddings"]["multiclass_classification"]
            logger.info(f"             random_embeddings F1={rm['macro_f1']:.4f}")

        return results

    def _extract_raw_features(
        self, peak_data: Dict[str, np.ndarray], meta: Dict[str, np.ndarray], subsample_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Extract raw peak features for baseline comparison.

        Features per peak: [m/z, intensity, rank, local_density, precursor_mz, precursor_charge]

        Uses the same spectrum indices as peak_data to ensure consistency.

        Args:
            peak_data: Dictionary with peak data (contains spectrum_indices)
            meta: Full metadata dictionary
            subsample_mask: Optional boolean mask to subsample peaks (for memory efficiency)

        Returns:
            Raw features array (N_peaks, 6)
        """
        # Use the spectrum indices from peak_data to ensure we process the same spectra
        spectrum_indices = peak_data.get("spectrum_indices", None)

        if spectrum_indices is None:
            raise ValueError("peak_data must contain 'spectrum_indices' for raw feature extraction")

        spectra = meta["spectra"]  # (N, L, 2) - [m/z, intensity]
        spectra_masks = meta["spectra_mask"]

        # Get precursor info (per-spectrum scalars)
        precursor_mzs = meta.get("precursor_mz", None)
        precursor_charges = meta.get("precursor_charge", None)

        all_features = []

        for spec_idx in spectrum_indices:
            # Get spectrum data
            spectrum = spectra[spec_idx]  # (L, 2)
            mz_values = spectrum[:, 0]
            intensity_values = spectrum[:, 1]

            # Get valid mask
            valid_mask = ~spectra_masks[spec_idx]

            # Extract valid peaks
            valid_mz = mz_values[valid_mask]
            valid_intensity = intensity_values[valid_mask]
            n_valid = len(valid_mz)

            # Compute rank (higher intensity = lower rank number)
            intensity_ranks = np.argsort(np.argsort(-valid_intensity)) + 1
            normalized_ranks = intensity_ranks / n_valid

            # Compute local density using sorted array + searchsorted (O(n log n))
            sorted_mz = np.sort(valid_mz)
            right_idx = np.searchsorted(sorted_mz, valid_mz + 50, side="right")
            left_idx = np.searchsorted(sorted_mz, valid_mz - 50, side="left")
            local_density = (right_idx - left_idx - 1).astype(np.float64)
            normalized_density = local_density / n_valid

            # Precursor features (broadcast per-spectrum scalar to all peaks)
            prec_mz = float(precursor_mzs[spec_idx]) if precursor_mzs is not None else 0.0
            prec_charge = float(precursor_charges[spec_idx]) if precursor_charges is not None else 0.0

            # Stack features: [m/z, intensity, rank, local_density, precursor_mz, precursor_charge]
            features = np.column_stack(
                [
                    valid_mz,
                    valid_intensity,
                    normalized_ranks,
                    normalized_density,
                    np.full(n_valid, prec_mz),
                    np.full(n_valid, prec_charge),
                ]
            )

            all_features.append(features)

        # Concatenate all features
        X_raw = np.concatenate(all_features, axis=0)

        # Apply subsampling if requested
        if subsample_mask is not None:
            X_raw = X_raw[subsample_mask]

        return X_raw

    def _generate_random_embeddings(self, shape: Tuple[int, int], batch_size: int = 50000) -> np.ndarray:
        """Generate random embeddings with the same shape as learned embeddings.

        Uses Gaussian distribution N(0, 1) to match typical embedding distributions.
        Generates in batches to avoid memory issues with large datasets.

        Args:
            shape: Shape of embeddings (N_peaks, embedding_dim)
            batch_size: Number of embeddings to generate per batch

        Returns:
            Random embeddings array
        """
        np.random.seed(self.random_state)
        n_peaks, embed_dim = shape

        # Generate in batches to avoid memory issues
        random_emb_list = []
        for start_idx in range(0, n_peaks, batch_size):
            end_idx = min(start_idx + batch_size, n_peaks)
            batch_size_actual = end_idx - start_idx

            # Generate batch
            batch_emb = np.random.randn(batch_size_actual, embed_dim).astype(np.float32)
            # Normalize to unit norm (same as learned embeddings)
            batch_emb = batch_emb / (np.linalg.norm(batch_emb, axis=1, keepdims=True) + 1e-8)
            random_emb_list.append(batch_emb)

        return np.concatenate(random_emb_list, axis=0)

    def _train_pytorch_classifier(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_classes: int,
        l2_reg: float,
        class_weights: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[LinearClassifier, Dict[str, List[float]]]:
        """Train a PyTorch linear classifier.

        For a single-layer linear classifier, CPU is typically faster than GPU
        because the computation is too small to offset data transfer overhead.
        Callers can pass ``device=torch.device("cpu")`` to force CPU training.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            num_classes: Number of classes
            l2_reg: L2 regularization strength
            class_weights: Class weights for imbalanced data
            device: Device override. If None, uses self.device.

        Returns:
            Tuple of (trained model, training history)
        """
        # CPU with 1 thread is optimal for linear classifiers: the model is too
        # small for GPU transfer overhead or multi-thread sync to pay off.
        # Benchmarked: CPU/1-thread=2.2s, GPU=5.0s, CPU/14-thread=5.3s per epoch.
        device = device if device is not None else torch.device("cpu")
        prev_threads = torch.get_num_threads()
        if device.type == "cpu":
            torch.set_num_threads(1)
        use_pin_memory = device.type == "cuda"
        # Convert to PyTorch tensors
        X_train_t = torch.from_numpy(X_train).float()
        y_train_t = torch.from_numpy(y_train).long()
        X_val_t = torch.from_numpy(X_val).float()
        y_val_t = torch.from_numpy(y_val).long()

        # Create data loaders (data is already in-memory tensors, so num_workers=0
        # avoids unnecessary inter-process communication overhead)
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=use_pin_memory,
        )

        # Create validation loader for batched validation
        val_dataset = TensorDataset(X_val_t, y_val_t)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size * 4,  # Larger batches for validation (no gradients)
            shuffle=False,
            pin_memory=use_pin_memory,
        )

        # Initialize model
        model = LinearClassifier(X_train.shape[1], num_classes).to(device)

        # Loss function with class weights
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            criterion = nn.CrossEntropyLoss()

        # Optimizer with L2 regularization
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=l2_reg)

        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_f1_macro": [],
        }

        best_val_f1 = 0.0
        best_model_state = None
        patience_counter = 0

        # Training loop
        for epoch in range(self.max_epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_samples = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch_X)
                train_samples += len(batch_X)

            train_loss /= train_samples

            # Validation (batched for efficiency)
            model.eval()
            val_loss = 0.0
            val_samples = 0
            all_val_preds = []
            all_val_labels = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)

                    val_outputs = model(batch_X)
                    loss = criterion(val_outputs, batch_y)
                    val_loss += loss.item() * len(batch_X)
                    val_samples += len(batch_X)

                    # Get predictions
                    val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
                    all_val_preds.extend(val_preds)
                    all_val_labels.extend(batch_y.cpu().numpy())

            val_loss /= val_samples

            # Compute F1 on CPU
            all_val_preds = np.array(all_val_preds)
            all_val_labels = np.array(all_val_labels)
            val_f1_macro = f1_score(all_val_labels, all_val_preds, average="macro")

            # Record history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1_macro"].append(val_f1_macro)

            # Early stopping check
            if val_f1_macro > best_val_f1:
                best_val_f1 = val_f1_macro
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.early_stopping_patience:
                break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Restore thread count
        torch.set_num_threads(prev_threads)

        return model, history

    def _build_peptide_split_groups(
        self,
        spectrum_indices_per_peak: np.ndarray,
        meta: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Build per-peak group IDs that cluster spectra by peptide sequence.

        Spectra sharing the same peptide sequence are assigned the same group
        ID so they always land in the same train/val/test split.  If no
        peptide metadata is found, falls back to spectrum indices (one group
        per spectrum).

        Args:
            spectrum_indices_per_peak: Spectrum index for each peak (N_peaks,).
            meta: Evaluator-level metadata dict (contains spectrum-level
                sequences under one of ``sequence``, ``peptides``, ``peptide``,
                ``seq``).

        Returns:
            Integer group ID per peak (N_peaks,).
        """
        # Find the peptide sequence array in metadata
        peptides = None
        for key in ["sequence", "peptides", "peptide", "seq"]:
            if key in meta:
                peptides = meta[key]
                break

        if peptides is None:
            logger.warning("No peptide sequence metadata found — falling back to spectrum-level splitting (no duplicate-sequence protection)")
            return spectrum_indices_per_peak

        # Map each unique peptide string to a group ID
        peptide_to_group: Dict[str, int] = {}
        next_group = 0
        spectrum_to_group = np.empty(len(peptides), dtype=np.int32)

        for i, pep in enumerate(peptides):
            pep_str = str(pep).strip() if pep is not None else f"__none_{i}"
            if pep_str not in peptide_to_group:
                peptide_to_group[pep_str] = next_group
                next_group += 1
            spectrum_to_group[i] = peptide_to_group[pep_str]

        # Map each peak to its spectrum's peptide group
        return spectrum_to_group[spectrum_indices_per_peak]

    def _split_by_spectrum(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/val/test by group to prevent leakage.

        All peaks sharing the same group ID (peptide sequence) are placed in
        the same split so that neither same-spectrum nor duplicate-sequence
        information leaks between train, validation, and test sets.

        Args:
            X: Peak features (N_peaks, D)
            y: Labels (N_peaks,)
            groups: Group ID per peak (N_peaks,) — peaks with the same ID
                are never split across train/val/test.

        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test, test_peak_indices)
        """
        unique_groups = np.unique(groups)

        # Split groups into train+val / test
        grp_train_val, grp_test = train_test_split(
            unique_groups,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        # Split train+val groups into train / val
        grp_train, grp_val = train_test_split(
            grp_train_val,
            test_size=0.2,
            random_state=self.random_state,
        )

        train_set = set(grp_train.tolist())
        val_set = set(grp_val.tolist())
        test_set = set(grp_test.tolist())

        train_mask = np.isin(groups, list(train_set))
        val_mask = np.isin(groups, list(val_set))
        test_mask = np.isin(groups, list(test_set))

        test_peak_indices = np.where(test_mask)[0]

        return (
            X[train_mask],
            y[train_mask],
            X[val_mask],
            y[val_mask],
            X[test_mask],
            y[test_mask],
            test_peak_indices,
        )

    def _run_binary_classification(
        self,
        X: np.ndarray,
        y: np.ndarray,
        spectrum_groups: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run binary classification: annotated vs unannotated.

        Args:
            X: Peak embeddings (N, D)
            y: Binary labels (N,) - 0=unannotated, 1=annotated
            spectrum_groups: Spectrum index per peak for spectrum-level splitting.
                If None, falls back to peak-level splitting (not recommended).

        Returns:
            Dictionary with classification results
        """
        # Split data at the spectrum level to prevent information leakage
        if spectrum_groups is not None:
            X_train, y_train, X_val, y_val, X_test, y_test, _ = self._split_by_spectrum(X, y, spectrum_groups)
        else:
            logger.warning("No spectrum groups provided — falling back to peak-level split (risk of leakage)")
            X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.2, random_state=self.random_state, stratify=y_train_val
            )

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Binary classification has 2 classes
        num_classes = 2

        # Compute class weights for balanced training
        class_counts = np.bincount(y_train, minlength=num_classes)
        class_weights = np.zeros(num_classes, dtype=np.float32)
        for i in range(num_classes):
            if class_counts[i] > 0:
                class_weights[i] = len(y_train) / (num_classes * class_counts[i])
        class_weights = torch.from_numpy(class_weights).float()

        # Train classifier
        best_l2_reg = None
        best_val_f1 = 0.0
        best_model = None

        if self.use_grid_search:
            for l2_reg in self.l2_reg_values:
                model, history = self._train_pytorch_classifier(
                    X_train_scaled,
                    y_train,
                    X_val_scaled,
                    y_val,
                    num_classes=2,
                    l2_reg=l2_reg,
                    class_weights=class_weights,
                )

                val_f1 = max(history["val_f1_macro"])
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_l2_reg = l2_reg
                    best_model = model
        else:
            best_l2_reg = 0.01
            best_model, history = self._train_pytorch_classifier(
                X_train_scaled,
                y_train,
                X_val_scaled,
                y_val,
                num_classes=2,
                l2_reg=best_l2_reg,
                class_weights=class_weights,
            )

        # Evaluate on test set
        best_model.eval()
        with torch.no_grad():
            X_test_t = torch.from_numpy(X_test_scaled).float().to(self.device)
            test_outputs = best_model(X_test_t)
            y_pred = torch.argmax(test_outputs, dim=1).cpu().numpy()

        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", labels=np.arange(num_classes), zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", labels=np.arange(num_classes), zero_division=0)

        # Specify labels to ensure all classes are included, even if not in test set or predictions
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None, labels=np.arange(num_classes), zero_division=0)

        conf_matrix = confusion_matrix(y_test, y_pred, labels=np.arange(num_classes))

        class_names = ["unannotated", "annotated"]

        return {
            "accuracy": float(accuracy),
            "macro_f1": float(f1_macro),
            "weighted_f1": float(f1_weighted),
            "per_class_precision": precision.tolist(),
            "per_class_recall": recall.tolist(),
            "per_class_f1": f1.tolist(),
            "per_class_support": support.tolist(),
            "confusion_matrix": conf_matrix.tolist(),
            "class_names": class_names,
            "best_l2_reg": float(best_l2_reg),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
        }

    def _run_multiclass_classification(
        self,
        X: np.ndarray,
        y: np.ndarray,
        spectrum_groups: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run multi-class classification: b/y ions, losses, isotopes, precursor, unannotated.

        Args:
            X: Peak embeddings (N, D)
            y: Multi-class labels (N,) - string labels
            spectrum_groups: Spectrum index per peak for spectrum-level splitting.
                If None, falls back to peak-level splitting (not recommended).

        Returns:
            Dictionary with classification results
        """
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        class_names = label_encoder.classes_.tolist()
        num_classes = len(class_names)

        # Split data at the spectrum level to prevent information leakage
        if spectrum_groups is not None:
            X_train, y_train, X_val, y_val, X_test, y_test, test_peak_indices = self._split_by_spectrum(X, y_encoded, spectrum_groups)
        else:
            logger.warning("No spectrum groups provided — falling back to peak-level split (risk of leakage)")
            indices = np.arange(len(X))
            train_val_idx, test_peak_indices, y_train_val, y_test = train_test_split(
                indices, y_encoded, test_size=self.test_size, random_state=self.random_state, stratify=y_encoded
            )
            train_idx, val_idx, y_train, y_val = train_test_split(
                train_val_idx, y_train_val, test_size=0.2, random_state=self.random_state, stratify=y_train_val
            )
            X_train = X[train_idx]
            X_val = X[val_idx]
            X_test = X[test_peak_indices]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Compute class weights for balanced training
        class_counts = np.bincount(y_train, minlength=num_classes)
        class_weights = np.zeros(num_classes, dtype=np.float32)
        for i in range(num_classes):
            if class_counts[i] > 0:
                class_weights[i] = len(y_train) / (num_classes * class_counts[i])
            else:
                class_weights[i] = 0.0
        class_weights = torch.from_numpy(class_weights).float()

        # Train classifier
        best_l2_reg = None
        best_val_f1 = 0.0
        best_model = None

        if self.use_grid_search:
            for l2_reg in self.l2_reg_values:
                model, history = self._train_pytorch_classifier(
                    X_train_scaled,
                    y_train,
                    X_val_scaled,
                    y_val,
                    num_classes=num_classes,
                    l2_reg=l2_reg,
                    class_weights=class_weights,
                )

                val_f1 = max(history["val_f1_macro"])
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_l2_reg = l2_reg
                    best_model = model
        else:
            best_l2_reg = 0.01
            best_model, history = self._train_pytorch_classifier(
                X_train_scaled,
                y_train,
                X_val_scaled,
                y_val,
                num_classes=num_classes,
                l2_reg=best_l2_reg,
                class_weights=class_weights,
            )

        # Evaluate on test set
        best_model.eval()
        with torch.no_grad():
            X_test_t = torch.from_numpy(X_test_scaled).float().to(self.device)
            test_outputs = best_model(X_test_t)
            y_pred = torch.argmax(test_outputs, dim=1).cpu().numpy()
            y_proba = torch.softmax(test_outputs, dim=1).cpu().numpy()

        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", labels=np.arange(num_classes), zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", labels=np.arange(num_classes), zero_division=0)

        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None, labels=np.arange(num_classes), zero_division=0)

        conf_matrix = confusion_matrix(y_test, y_pred, labels=np.arange(num_classes))

        # Normalize confusion matrix by true labels (rows)
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        conf_matrix_normalized = conf_matrix.astype(float) / row_sums

        logger.info(f"  Accuracy={accuracy:.4f}  Macro-F1={f1_macro:.4f}  Weighted-F1={f1_weighted:.4f}")

        y_test_labels = label_encoder.inverse_transform(y_test)
        y_pred_labels = label_encoder.inverse_transform(y_pred)
        y_pred_confidence = y_proba[np.arange(len(y_pred)), y_pred]

        return {
            "accuracy": float(accuracy),
            "macro_f1": float(f1_macro),
            "weighted_f1": float(f1_weighted),
            "per_class_precision": precision.tolist(),
            "per_class_recall": recall.tolist(),
            "per_class_f1": f1.tolist(),
            "per_class_support": support.tolist(),
            "confusion_matrix": conf_matrix.tolist(),
            "confusion_matrix_normalized": conf_matrix_normalized.tolist(),
            "class_names": class_names,
            "test_indices": test_peak_indices.tolist(),
            "y_test_labels": y_test_labels.tolist(),
            "y_pred_labels": y_pred_labels.tolist(),
            "y_pred_confidence": y_pred_confidence.tolist(),
            "best_l2_reg": float(best_l2_reg),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
        }

    def _create_visualizations(self, binary_results: Dict[str, Any], multiclass_results: Dict[str, Any]) -> None:
        """Create visualization plots for classification results.

        Args:
            binary_results: Binary classification results
            multiclass_results: Multi-class classification results
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib/seaborn not available, skipping visualizations")
            return

        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle("Peak Type Classification Results", fontsize=16, fontweight="bold")

        # 1. Binary confusion matrix
        ax = axes[0, 0]
        conf_matrix = np.array(binary_results["confusion_matrix"])
        # Normalize by row (true labels)
        conf_matrix_norm = conf_matrix.astype(float) / conf_matrix.sum(axis=1, keepdims=True)

        sns.heatmap(
            conf_matrix_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=binary_results["class_names"],
            yticklabels=binary_results["class_names"],
            ax=ax,
            cbar_kws={"label": "Fraction"},
        )
        ax.set_title("Binary Classification\nConfusion Matrix (Normalized)", fontweight="bold")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

        # Add counts as text
        for i in range(len(conf_matrix)):
            for j in range(len(conf_matrix)):
                ax.text(j + 0.5, i + 0.7, f"n={conf_matrix[i, j]:,}", ha="center", va="center", fontsize=8, color="gray")

        # 2. Binary per-class metrics
        ax = axes[0, 1]
        class_names = binary_results["class_names"]
        precision = binary_results["per_class_precision"]
        recall = binary_results["per_class_recall"]
        f1 = binary_results["per_class_f1"]

        x = np.arange(len(class_names))
        width = 0.25

        ax.bar(x - width, precision, width, label="Precision", alpha=0.8)
        ax.bar(x, recall, width, label="Recall", alpha=0.8)
        ax.bar(x + width, f1, width, label="F1-Score", alpha=0.8)

        ax.set_ylabel("Score")
        ax.set_title("Binary Classification\nPer-Class Metrics", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0, 1.05])

        # Add macro-F1 as text
        macro_f1 = binary_results["macro_f1"]
        ax.text(
            0.98,
            0.98,
            f"Macro-F1: {macro_f1:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
            fontsize=10,
            fontweight="bold",
        )

        # 3. Multi-class confusion matrix
        ax = axes[1, 0]
        conf_matrix = np.array(multiclass_results["confusion_matrix_normalized"])
        class_names_mc = multiclass_results["class_names"]

        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names_mc,
            yticklabels=class_names_mc,
            ax=ax,
            cbar_kws={"label": "Fraction"},
        )
        ax.set_title("Multi-Class Classification\nConfusion Matrix (Normalized)", fontweight="bold")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

        # 4. Multi-class per-class metrics
        ax = axes[1, 1]
        precision = multiclass_results["per_class_precision"]
        recall = multiclass_results["per_class_recall"]
        f1 = multiclass_results["per_class_f1"]

        x = np.arange(len(class_names_mc))
        width = 0.25

        ax.bar(x - width, precision, width, label="Precision", alpha=0.8)
        ax.bar(x, recall, width, label="Recall", alpha=0.8)
        ax.bar(x + width, f1, width, label="F1-Score", alpha=0.8)

        ax.set_ylabel("Score")
        ax.set_title("Multi-Class Classification\nPer-Class Metrics", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names_mc, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0, 1.05])

        # Add macro-F1 as text
        macro_f1 = multiclass_results["macro_f1"]
        ax.text(
            0.98,
            0.98,
            f"Macro-F1: {macro_f1:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
            fontsize=10,
            fontweight="bold",
        )

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / "peak_type_classification.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _create_prediction_plots(
        self,
        peak_data: Dict[str, np.ndarray],
        multiclass_results: Dict[str, Any],
        meta: Dict[str, np.ndarray],
    ) -> None:
        """Create per-spectrum plots with ground-truth and predicted labels."""
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib/seaborn not available, skipping prediction plots")
            return

        required_keys = ["test_indices", "y_test_labels", "y_pred_labels", "y_pred_confidence"]
        if not all(key in multiclass_results for key in required_keys):
            logger.warning("Prediction outputs missing in multiclass results; skipping prediction plots")
            return

        if "mz_values" not in peak_data or "intensity_values" not in peak_data:
            logger.warning("Spectra not available in metadata; skipping prediction plots")
            return

        test_indices = np.asarray(multiclass_results["test_indices"], dtype=np.int64)
        true_labels = np.asarray(multiclass_results["y_test_labels"], dtype=object)
        pred_labels = np.asarray(multiclass_results["y_pred_labels"], dtype=object)
        pred_confidence = np.asarray(multiclass_results["y_pred_confidence"], dtype=np.float64)

        spectrum_indices = peak_data["spectrum_indices_per_peak"][test_indices]
        mz_values = peak_data["mz_values"][test_indices]
        intensity_values = peak_data["intensity_values"][test_indices]

        valid_plot_mask = np.isfinite(mz_values) & np.isfinite(intensity_values)
        if not np.any(valid_plot_mask):
            logger.warning("No valid m/z or intensity values available for plotting")
            return

        spectrum_indices = spectrum_indices[valid_plot_mask]
        mz_values = mz_values[valid_plot_mask]
        intensity_values = intensity_values[valid_plot_mask]
        true_labels = true_labels[valid_plot_mask]
        pred_labels = pred_labels[valid_plot_mask]
        pred_confidence = pred_confidence[valid_plot_mask]

        unique_spectra, counts = np.unique(spectrum_indices, return_counts=True)
        if len(unique_spectra) == 0:
            logger.warning("No spectra available for prediction plots")
            return

        # Pick spectra with most test peaks for clearer plots
        sorted_indices = np.argsort(counts)[::-1]
        selected_spectra = unique_spectra[sorted_indices][: self.max_prediction_plots]

        output_dir = self.output_dir / "prediction_spectra"
        output_dir.mkdir(parents=True, exist_ok=True)

        for spec_idx in selected_spectra:
            spec_mask = spectrum_indices == spec_idx
            spec_mz = mz_values[spec_mask]
            spec_intensity = intensity_values[spec_mask]
            spec_true = true_labels[spec_mask]
            spec_pred = pred_labels[spec_mask]
            spec_conf = pred_confidence[spec_mask]

            sequence = self._get_sequence_for_spectrum(meta, int(spec_idx))
            frag_type = self._get_meta_value(meta, "frag_type", int(spec_idx))
            precursor_charge = self._get_meta_value(meta, "precursor_charge", int(spec_idx))

            save_path = output_dir / f"spectrum_{int(spec_idx):04d}.png"
            self._plot_spectrum_with_predictions(
                mz=spec_mz,
                intensity=spec_intensity,
                true_labels=spec_true,
                pred_labels=spec_pred,
                pred_confidence=spec_conf,
                sequence=sequence,
                frag_type=frag_type,
                precursor_charge=precursor_charge,
                save_path=save_path,
            )

    @staticmethod
    def _get_meta_value(meta: Dict[str, np.ndarray], key: str, index: int) -> Optional[Any]:
        if key in meta and index < len(meta[key]):
            return meta[key][index]
        return None

    @staticmethod
    def _get_sequence_for_spectrum(meta: Dict[str, np.ndarray], index: int) -> str:
        for key in ["sequence", "peptides", "peptide", "seq"]:
            if key in meta and index < len(meta[key]):
                seq = meta[key][index]
                return str(seq) if seq is not None else "unknown"
        return "unknown"

    def _plot_spectrum_with_predictions(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        pred_confidence: np.ndarray,
        sequence: str,
        frag_type: Optional[str],
        precursor_charge: Optional[Any],
        save_path: Path,
    ) -> None:
        """Plot a single spectrum with true/predicted peak labels."""
        from matplotlib.lines import Line2D

        fig, ax = plt.subplots(figsize=(16, 8))

        # Background stems
        ax.vlines(mz, 0, intensity, color="lightgray", alpha=0.4, linewidth=1.0, zorder=1)

        # Color mapping for labels (4-class taxonomy)
        label_colors = {
            "b-ion": "tab:blue",
            "y-ion": "tab:orange",
            "precursor": "tab:brown",
            "unannotated": "tab:red",
        }

        # Plot peaks with true label fill and predicted label edge
        for i in range(len(mz)):
            true_label = str(true_labels[i])
            pred_label = str(pred_labels[i])
            color = label_colors.get(true_label, "tab:gray")
            edge_color = label_colors.get(pred_label, "tab:gray")
            size = 30 + 60 * float(pred_confidence[i])
            ax.scatter(
                mz[i],
                intensity[i],
                s=size,
                color=color,
                edgecolors=edge_color,
                linewidths=1.2,
                alpha=0.9,
                zorder=2,
            )

        # Title with metadata and accuracy
        accuracy = float(np.mean(true_labels == pred_labels)) if len(true_labels) > 0 else 0.0
        title_parts = [sequence]
        if precursor_charge is not None:
            title_parts.append(f"z={precursor_charge}")
        if frag_type is not None and str(frag_type) != "None":
            title_parts.append(str(frag_type))
        title_parts.append(f"Acc={accuracy * 100:.1f}%")
        ax.set_title(" | ".join(title_parts), fontsize=12, fontweight="bold")

        ax.set_xlabel("m/z", fontsize=11, fontweight="bold")
        ax.set_ylabel("Normalized Intensity", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if len(mz) > 0:
            ax.set_xlim(float(np.min(mz)) - 5, float(np.max(mz)) + 5)
        ax.set_ylim(0, float(np.max(intensity)) * 1.15 if len(intensity) > 0 else 1.0)

        # Legend: true label colors
        unique_true = sorted(set(true_labels.tolist()))
        true_handles = []
        for label in unique_true:
            color = label_colors.get(str(label), "tab:gray")
            true_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markersize=7,
                    label=f"true: {label}",
                )
            )

        # Legend: predicted label edge colors
        unique_pred = sorted(set(pred_labels.tolist()))
        pred_handles = []
        for label in unique_pred:
            color = label_colors.get(str(label), "tab:gray")
            pred_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="none",
                    markeredgecolor=color,
                    markersize=7,
                    label=f"pred: {label}",
                )
            )

        if true_handles:
            first_legend = ax.legend(handles=true_handles, loc="upper right", fontsize=8, title="True")
            ax.add_artist(first_legend)
        if pred_handles:
            ax.legend(handles=pred_handles, loc="upper left", fontsize=8, title="Predicted")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save results to JSON file.

        Args:
            results: Results dictionary
        """
        output_path = self.output_dir / "peak_type_classification_results.json"

        def _default(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=_default)

    # ------------------------------------------------------------------
    # Peak-level UMAP visualizations
    # ------------------------------------------------------------------

    def _create_peak_umap_visualizations(self, peak_data: Dict[str, np.ndarray]) -> List[str]:
        """Create peak-level UMAP visualizations.

        Produces three UMAP variants:
        1. All peaks (raw embeddings)
        2. All peaks (residual — pre-transformer regressed out)
        3. Fragment ions only (b/y, no noise/unannotated — cleaner structure)

        Returns:
            Tuple of (list of saved PNG paths, dict of UMAP diagnostics).
        """
        embeddings = peak_data["embeddings"]
        n_peaks = len(embeddings)
        diagnostics: Dict[str, float] = {}

        if n_peaks < 500:
            logger.warning(f"Too few peaks ({n_peaks}) for UMAP — skipping peak-level UMAP visualizations")
            return [], diagnostics

        # Subsample by spectrum to stay within budget
        sample_indices = self._subsample_peaks_by_spectrum(peak_data["spectrum_indices_per_peak"], self.max_umap_samples)

        # Slice all peak_data arrays to the sampled indices
        sampled = {}
        for k, v in peak_data.items():
            if isinstance(v, np.ndarray) and v.shape[0] == n_peaks:
                sampled[k] = v[sample_indices]
            else:
                sampled[k] = v

        emb_sampled = embeddings[sample_indices]

        paths: List[str] = []

        # Plot functions for all-peak UMAPs
        all_peak_plots = [
            (self._plot_umap_binary, "binary"),
            (self._plot_umap_ion_categories, "ion_categories"),
            (self._plot_umap_fragment_ladder, "fragment_ladder"),
            (self._plot_umap_mz_position, "mz_position"),
            (self._plot_umap_intensity, "intensity"),
            (self._plot_umap_spectrum_identity, "spectrum_identity"),
            (self._plot_umap_frag_type, "frag_type"),
            (self._plot_umap_instrument, "instrument"),
            (self._plot_umap_detector, "detector"),
        ]

        # Plot functions for fragment-only UMAPs (no binary — all are annotated)
        fragment_plots = [
            (self._plot_umap_ion_categories, "ion_categories"),
            (self._plot_umap_fragment_ladder, "fragment_ladder"),
            (self._plot_umap_mz_position, "mz_position"),
            (self._plot_umap_intensity, "intensity"),
            (self._plot_umap_spectrum_identity, "spectrum_identity"),
            (self._plot_umap_frag_type, "frag_type"),
            (self._plot_umap_instrument, "instrument"),
            (self._plot_umap_detector, "detector"),
        ]

        # --- 1) All-peak raw UMAP ---
        raw_dir = self.output_dir / "peak_umap"
        raw_dir.mkdir(parents=True, exist_ok=True)

        umap_2d = self._compute_peak_umap(emb_sampled)
        if umap_2d is None:
            return [], diagnostics

        rho = self._log_umap_mz_correlation(umap_2d, sampled, label="raw")
        if rho is not None:
            diagnostics["umap_mz_rho_raw"] = rho
        self._save_umap_coords(umap_2d, sampled, "raw", raw_dir)

        for plot_fn, name in all_peak_plots:
            try:
                path = plot_fn(umap_2d, sampled, raw_dir)
                if path is not None:
                    paths.append(path)
            except Exception as e:
                logger.warning(f"Failed to create UMAP plot '{name}': {e}")

        # --- 2) All-peak residual UMAP ---
        emb_pre = sampled.get("embeddings_pretransformer")
        if emb_pre is not None:
            corr_dir = self.output_dir / "peak_umap_residual"
            corr_dir.mkdir(parents=True, exist_ok=True)

            emb_corrected, r2 = self._correct_for_input_embedding(emb_sampled, emb_pre)
            diagnostics["pretransformer_r2"] = r2

            umap_2d_corr = self._compute_peak_umap(emb_corrected)
            if umap_2d_corr is not None:
                rho = self._log_umap_mz_correlation(umap_2d_corr, sampled, label="residual")
                if rho is not None:
                    diagnostics["umap_mz_rho_residual"] = rho
                self._save_umap_coords(umap_2d_corr, sampled, "residual", corr_dir)
                for plot_fn, name in all_peak_plots:
                    try:
                        path = plot_fn(umap_2d_corr, sampled, corr_dir, suffix="_residual")
                        if path is not None:
                            paths.append(path)
                    except Exception as e:
                        logger.warning(f"Failed to create residual UMAP plot '{name}': {e}")
        else:
            pass  # Pre-transformer embeddings not available — skip residual UMAPs

        # --- 3) Fragment-only UMAP (b/y ions only — removes noise/unannotated) ---
        if self.enable_fragment_umap:
            # Start with fragment ion mask
            frag_mask = np.isin(sampled["multiclass_labels"], ["b-ion", "y-ion"])

            # Optional instrument filtering (removes instrument-driven clusters)
            filter_parts = []
            if self.fragment_umap_frag_types and "frag_types" in sampled:
                ft_mask = np.isin(sampled["frag_types"], self.fragment_umap_frag_types)
                frag_mask &= ft_mask
                filter_parts.append(f"frag_type={self.fragment_umap_frag_types}")
            if self.fragment_umap_detectors and "search_detectors" in sampled:
                det_mask = np.isin(sampled["search_detectors"], self.fragment_umap_detectors)
                frag_mask &= det_mask
                filter_parts.append(f"detector={self.fragment_umap_detectors}")
            if self.fragment_umap_instruments and "search_instruments" in sampled:
                inst_mask = np.isin(sampled["search_instruments"], self.fragment_umap_instruments)
                frag_mask &= inst_mask
                filter_parts.append(f"instrument={self.fragment_umap_instruments}")

            n_frag = int(frag_mask.sum())

            if n_frag >= 500:
                frag_sampled = {}
                for k, v in sampled.items():
                    if isinstance(v, np.ndarray) and v.shape[0] == len(frag_mask):
                        frag_sampled[k] = v[frag_mask]
                    else:
                        frag_sampled[k] = v

                emb_frag = emb_sampled[frag_mask]

                frag_dir = self.output_dir / "peak_umap_fragments"
                frag_dir.mkdir(parents=True, exist_ok=True)

                umap_2d_frag = self._compute_peak_umap(emb_frag)
                if umap_2d_frag is not None:
                    self._log_umap_mz_correlation(umap_2d_frag, frag_sampled, label="fragments_raw")
                    for plot_fn, name in fragment_plots:
                        try:
                            path = plot_fn(umap_2d_frag, frag_sampled, frag_dir)
                            if path is not None:
                                paths.append(path)
                        except Exception as e:
                            logger.warning(f"Failed to create fragment UMAP plot '{name}': {e}")

                # Fragment-only residual
                if emb_pre is not None:
                    emb_pre_frag = emb_pre[frag_mask]
                    frag_res_dir = self.output_dir / "peak_umap_fragments_residual"
                    frag_res_dir.mkdir(parents=True, exist_ok=True)

                    emb_frag_corr, _ = self._correct_for_input_embedding(emb_frag, emb_pre_frag)
                    umap_2d_frag_corr = self._compute_peak_umap(emb_frag_corr)
                    if umap_2d_frag_corr is not None:
                        self._log_umap_mz_correlation(umap_2d_frag_corr, frag_sampled, label="fragments_residual")
                        for plot_fn, name in fragment_plots:
                            try:
                                path = plot_fn(
                                    umap_2d_frag_corr,
                                    frag_sampled,
                                    frag_res_dir,
                                    suffix="_residual",
                                )
                                if path is not None:
                                    paths.append(path)
                            except Exception as e:
                                logger.warning(f"Failed to create fragment residual UMAP plot '{name}': {e}")
            else:
                logger.warning(f"Too few fragment ions ({n_frag}) for fragment-only UMAP")

        return paths, diagnostics

    def _subsample_peaks_by_spectrum(
        self,
        spectrum_indices_per_peak: np.ndarray,
        max_peaks: int,
    ) -> np.ndarray:
        """Subsample peaks by randomly selecting whole spectra until budget is reached.

        Preserves within-spectrum structure so loss/isotope peaks stay
        grouped with their parent ions.

        Returns:
            Integer indices into the peak arrays.
        """
        if len(spectrum_indices_per_peak) <= max_peaks:
            return np.arange(len(spectrum_indices_per_peak))

        unique_spectra = np.unique(spectrum_indices_per_peak)
        rng = np.random.RandomState(self.random_state)
        rng.shuffle(unique_spectra)

        selected = set()
        peak_count = 0
        for s in unique_spectra:
            s_mask = spectrum_indices_per_peak == s
            s_n = int(s_mask.sum())
            if peak_count + s_n > max_peaks and peak_count > 0:
                break
            selected.add(int(s))
            peak_count += s_n

        mask = np.isin(spectrum_indices_per_peak, list(selected))
        return np.where(mask)[0]

    def _save_umap_coords(self, umap_2d: np.ndarray, sampled: Dict[str, np.ndarray], tag: str, out_dir: Path) -> None:
        """Persist the 2D UMAP coords + per-peak colour labels so the scatter can be
        re-rendered downstream (e.g. paper-style SVG) without recomputing embeddings.

        Defensive: only writes 1-D label arrays that match the coord length; never raises.
        """
        try:
            import pandas as pd

            n = len(umap_2d)
            cols: Dict[str, np.ndarray] = {"umap_x": umap_2d[:, 0], "umap_y": umap_2d[:, 1]}
            for key in (
                "multiclass_labels",
                "matched_annotations",
                "ion_category",
                "ion_categories",
                "mz",
                "intensity",
                "frag_type",
                "instrument",
                "detector",
                "fragment_ladder",
                "is_annotated",
                "peak_label",
                "spectrum_indices_per_peak",
            ):
                v = sampled.get(key)
                if isinstance(v, np.ndarray) and v.ndim == 1 and v.shape[0] == n:
                    cols[key] = v
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(cols).to_parquet(out_dir / f"peak_umap_coords_{tag}.parquet")
            logger.info(f"Saved peak-UMAP coords ({tag}, n={n}) -> {out_dir}")
        except Exception as e:  # never let a coord dump break the task
            logger.warning(f"peak-UMAP coord dump failed ({tag}): {e}")

    def _compute_peak_umap(self, embeddings: np.ndarray) -> Optional[np.ndarray]:
        """Compute 2-D UMAP projection of peak embeddings.

        Returns:
            Array of shape (N, 2) or None on failure.
        """
        try:
            import umap
        except ImportError:
            logger.warning("umap-learn not installed — skipping peak UMAP")
            return None

        logger.debug(
            f"UMAP: {len(embeddings):,} peaks, n_neighbors={self.umap_n_neighbors}, min_dist={self.umap_min_dist}, metric={self.umap_metric}"
        )
        reducer = umap.UMAP(
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            metric=self.umap_metric,
            random_state=self.random_state,
            low_memory=True,
        )
        umap_2d = reducer.fit_transform(embeddings)
        return np.asarray(umap_2d)

    @staticmethod
    def _log_umap_mz_correlation(
        umap_2d: np.ndarray,
        sampled: Dict[str, np.ndarray],
        label: str = "",
    ) -> Optional[float]:
        """Compute and log Spearman correlation between UMAP axes and m/z.

        Returns the absolute rho of the dominant axis, or None if unavailable.
        """
        from scipy.stats import spearmanr

        mz = sampled.get("mz_values")
        if mz is None:
            return None
        valid = np.isfinite(mz)
        if valid.sum() < 100:
            return None

        rho1, _ = spearmanr(umap_2d[valid, 0], mz[valid])
        rho2, _ = spearmanr(umap_2d[valid, 1], mz[valid])
        best_rho = rho1 if abs(rho1) >= abs(rho2) else rho2
        abs_rho = abs(best_rho)
        return abs_rho

    def _plot_umap_mz_position(
        self,
        umap_2d: np.ndarray,
        sampled: Dict[str, np.ndarray],
        save_dir: Path,
        suffix: str = "",
    ) -> Optional[str]:
        """UMAP colored by absolute m/z position.

        Tests whether embedding space is primarily organized by m/z.
        """
        mz = sampled.get("mz_values")
        if mz is None:
            return None
        valid = ~np.isnan(mz)
        if valid.sum() < 100:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(
            umap_2d[valid, 0],
            umap_2d[valid, 1],
            c=mz[valid],
            cmap="viridis",
            s=self.umap_point_size,
            alpha=0.5,
            rasterized=True,
            edgecolors="none",
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("m/z (Da)", fontsize=10)
        ax.set_title(f"Peak Embeddings: m/z Position{' (residual)' if suffix else ''}", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        save_path = save_dir / f"peak_umap_mz_position{suffix}.png"
        fig.savefig(save_path, dpi=self.umap_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"  Saved: {save_path}")
        return str(save_path)

    def _plot_umap_intensity(
        self,
        umap_2d: np.ndarray,
        sampled: Dict[str, np.ndarray],
        save_dir: Path,
        suffix: str = "",
    ) -> Optional[str]:
        """UMAP colored by peak intensity (percentile-ranked).

        Uses percentile ranking to spread the color range across the
        heavily right-skewed sqrt-normalized intensity distribution.
        """
        intensity = sampled.get("intensity_values")
        if intensity is None:
            return None
        valid = ~np.isnan(intensity)
        if valid.sum() < 100:
            return None

        # Percentile-rank to spread color across the skewed distribution
        from scipy.stats import rankdata

        ranked = rankdata(intensity[valid], method="average") / valid.sum()

        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(
            umap_2d[valid, 0],
            umap_2d[valid, 1],
            c=ranked,
            cmap="plasma",
            s=self.umap_point_size,
            alpha=0.5,
            rasterized=True,
            edgecolors="none",
            vmin=0,
            vmax=1,
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Intensity (percentile)", fontsize=10)
        ax.set_title(f"Peak Embeddings: Intensity{' (residual)' if suffix else ''}", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        save_path = save_dir / f"peak_umap_intensity{suffix}.png"
        fig.savefig(save_path, dpi=self.umap_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"  Saved: {save_path}")
        return str(save_path)

    def _plot_umap_spectrum_identity(
        self,
        umap_2d: np.ndarray,
        sampled: Dict[str, np.ndarray],
        save_dir: Path,
        suffix: str = "",
        n_top_spectra: int = 10,
    ) -> Optional[str]:
        """UMAP colored by spectrum identity (randomly selected spectra).

        Tests whether embeddings are spectrum-specific (peaks from same spectrum
        cluster tightly) or universal (peaks spread by ion type regardless of
        which spectrum they come from).

        Selects spectra randomly (with a minimum peak count of 50) to avoid
        bias toward fully-padded spectra that all have n_peaks=200.
        """
        spec_ids = sampled.get("spectrum_indices_per_peak")
        if spec_ids is None:
            return None

        # Select random spectra with enough peaks to be visible
        unique, counts = np.unique(spec_ids, return_counts=True)
        min_peaks = 50
        eligible = unique[counts >= min_peaks]
        if len(eligible) < n_top_spectra:
            eligible = unique[np.argsort(-counts)]  # fallback to largest
        rng = np.random.RandomState(self.random_state)
        chosen = rng.choice(eligible, min(n_top_spectra, len(eligible)), replace=False)
        top_spectra = set(chosen)

        fig, ax = plt.subplots(figsize=(12, 8))

        # Background: all other spectra
        bg_mask = np.array([s not in top_spectra for s in spec_ids])
        if bg_mask.any():
            ax.scatter(
                umap_2d[bg_mask, 0],
                umap_2d[bg_mask, 1],
                c="#E0E0E0",
                s=self.umap_point_size * 0.5,
                alpha=0.1,
                rasterized=True,
                edgecolors="none",
                label=f"Other spectra (n={bg_mask.sum():,})",
            )

        # Selected spectra with distinct colors
        cmap = plt.cm.tab10
        for rank, spec_global_idx in enumerate(chosen):
            mask = spec_ids == spec_global_idx
            n_peaks = mask.sum()
            color = cmap(rank / max(n_top_spectra - 1, 1))
            ax.scatter(
                umap_2d[mask, 0],
                umap_2d[mask, 1],
                c=[color],
                s=self.umap_point_size * 1.2,
                alpha=0.6,
                rasterized=True,
                edgecolors="none",
                label=f"Spectrum {int(spec_global_idx)} (n={n_peaks})",
            )

        ax.legend(
            fontsize=7,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            framealpha=0.9,
            markerscale=2,
        )
        ax.set_title(
            f"Peak Embeddings: Spectrum Identity (top {n_top_spectra}){' (residual)' if suffix else ''}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])

        save_path = save_dir / f"peak_umap_spectrum_identity{suffix}.png"
        fig.savefig(save_path, dpi=self.umap_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"  Saved: {save_path}")
        return str(save_path)

    def _plot_umap_metadata_categorical(
        self,
        umap_2d: np.ndarray,
        sampled: Dict[str, np.ndarray],
        save_dir: Path,
        meta_key: str,
        title: str,
        filename: str,
        suffix: str = "",
    ) -> Optional[str]:
        """Generic UMAP colored by a categorical metadata field (per-peak)."""
        values = sampled.get(meta_key)
        if values is None:
            return None

        # Convert to string, handle None
        str_values = np.array([str(v) if v is not None else "Unknown" for v in values])
        unique_vals, counts = np.unique(str_values, return_counts=True)

        if len(unique_vals) < 2:
            return None

        # Sort by count descending for consistent legend
        sort_idx = np.argsort(-counts)
        unique_vals = unique_vals[sort_idx]
        counts = counts[sort_idx]

        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.cm.tab10 if len(unique_vals) <= 10 else plt.cm.tab20

        for i, val in enumerate(unique_vals):
            mask = str_values == val
            color = cmap(i / max(len(unique_vals) - 1, 1))
            ax.scatter(
                umap_2d[mask, 0],
                umap_2d[mask, 1],
                s=self.umap_point_size,
                c=[color],
                alpha=0.5,
                edgecolors="none",
                rasterized=True,
                label=f"{val} (n={counts[i]:,})",
            )

        title_suffix = " (residual)" if suffix else ""
        ax.set_title(f"{title}{title_suffix}", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
            markerscale=3,
            frameon=True,
            framealpha=0.9,
        )

        save_path = save_dir / f"{filename}{suffix}.png"
        fig.savefig(save_path, dpi=self.umap_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"  Saved: {save_path}")
        return str(save_path)

    def _plot_umap_frag_type(
        self,
        umap_2d: np.ndarray,
        sampled: Dict[str, np.ndarray],
        save_dir: Path,
        suffix: str = "",
    ) -> Optional[str]:
        """UMAP colored by fragmentation type (HCD, CID, HCID, ETD)."""
        return self._plot_umap_metadata_categorical(
            umap_2d,
            sampled,
            save_dir,
            meta_key="frag_types",
            title="Peak Embeddings: Fragmentation Type",
            filename="peak_umap_frag_type",
            suffix=suffix,
        )

    def _plot_umap_instrument(
        self,
        umap_2d: np.ndarray,
        sampled: Dict[str, np.ndarray],
        save_dir: Path,
        suffix: str = "",
    ) -> Optional[str]:
        """UMAP colored by instrument type."""
        return self._plot_umap_metadata_categorical(
            umap_2d,
            sampled,
            save_dir,
            meta_key="search_instruments",
            title="Peak Embeddings: Instrument",
            filename="peak_umap_instrument",
            suffix=suffix,
        )

    def _plot_umap_detector(
        self,
        umap_2d: np.ndarray,
        sampled: Dict[str, np.ndarray],
        save_dir: Path,
        suffix: str = "",
    ) -> Optional[str]:
        """UMAP colored by detector type (Orbitrap, IonTrap, TOF)."""
        return self._plot_umap_metadata_categorical(
            umap_2d,
            sampled,
            save_dir,
            meta_key="search_detectors",
            title="Peak Embeddings: Detector",
            filename="peak_umap_detector",
            suffix=suffix,
        )

    @staticmethod
    def _correct_for_input_embedding(
        post_transformer: np.ndarray,
        pre_transformer: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Remove the pre-transformer embedding contribution via Ridge regression.

        Returns:
            Tuple of (residual embeddings (N, D), R² of the regression).
        """
        from sklearn.linear_model import Ridge

        reg = Ridge(alpha=1.0, fit_intercept=True)
        reg.fit(pre_transformer, post_transformer)
        predicted = reg.predict(pre_transformer)
        residuals = post_transformer - predicted

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((post_transformer - post_transformer.mean(axis=0)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return residuals, r2

    # -- UMAP 1: Binary -----------------------------------------------

    def _plot_umap_binary(
        self,
        umap_2d: np.ndarray,
        sampled: Dict[str, np.ndarray],
        save_dir: Path,
        suffix: str = "",
    ) -> Optional[str]:
        """UMAP colored by annotated vs unannotated."""
        binary = sampled["binary_labels"]
        fig, ax = plt.subplots(figsize=(10, 8))

        # Unannotated (background)
        mask_unann = binary == 0
        ax.scatter(
            umap_2d[mask_unann, 0],
            umap_2d[mask_unann, 1],
            s=self.umap_point_size,
            c="#BDBDBD",
            alpha=0.2,
            edgecolors="none",
            rasterized=True,
            label=f"Unannotated (n={mask_unann.sum():,})",
        )
        # Annotated (foreground)
        mask_ann = binary == 1
        ax.scatter(
            umap_2d[mask_ann, 0],
            umap_2d[mask_ann, 1],
            s=self.umap_point_size,
            c="#1976D2",
            alpha=0.5,
            edgecolors="none",
            rasterized=True,
            label=f"Annotated (n={mask_ann.sum():,})",
        )

        ax.set_title("Peak Embeddings: Annotated vs Unannotated", fontsize=14, fontweight="bold")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(loc="best", fontsize=9, markerscale=3)
        ax.set_xticks([])
        ax.set_yticks([])

        save_path = save_dir / f"peak_umap_binary{suffix}.png"
        fig.savefig(save_path, dpi=self.umap_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"  Saved: {save_path}")
        return str(save_path)

    # -- UMAP 2: Ion Origin Categories ---------------------------------

    @staticmethod
    def _classify_custom_ion(annotation: Optional[str]) -> str:
        """Classify a custom ion annotation into subcategory."""
        if not annotation or not isinstance(annotation, str):
            return "other"
        ann_lower = annotation.lower()
        if "immonium" in ann_lower:
            return "immonium"
        if "tmt" in ann_lower or "itraq" in ann_lower:
            return "reporter"
        if "glycan" in ann_lower:
            return "glycan"
        return "other"

    def _plot_umap_ion_categories(
        self,
        umap_2d: np.ndarray,
        sampled: Dict[str, np.ndarray],
        save_dir: Path,
        suffix: str = "",
    ) -> Optional[str]:
        """UMAP colored by ion origin category."""
        labels = sampled["multiclass_labels"]
        annotations = sampled.get("matched_annotations", np.array([None] * len(labels)))

        # Map 4-class labels to display categories
        label_to_cat = {
            "unannotated": "Unannotated",
            "b-ion": "b-series",
            "y-ion": "y-series",
            "precursor": "Precursor",
        }
        n = len(labels)
        categories = np.array([label_to_cat.get(str(labels[i]), "Unannotated") for i in range(n)], dtype=object)

        # Color map
        cat_colors = {
            "Unannotated": "#E0E0E0",
            "b-series": "#1976D2",
            "y-series": "#D32F2F",
            "Precursor": "#FF9800",
        }

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot order: unannotated first, then ascending category size
        unique_cats, cat_counts = np.unique(categories, return_counts=True)
        # Always plot unannotated first
        plot_order = ["Unannotated"]
        # Then remaining sorted by count descending (largest first → underneath)
        remaining = [(c, cnt) for c, cnt in zip(unique_cats, cat_counts, strict=False) if c != "Unannotated"]
        remaining.sort(key=lambda x: -x[1])
        plot_order.extend([c for c, _ in remaining])

        for cat in plot_order:
            mask = categories == cat
            if not mask.any():
                continue
            color = cat_colors.get(cat, "#795548")
            alpha = 0.15 if cat == "Unannotated" else 0.6
            ax.scatter(
                umap_2d[mask, 0],
                umap_2d[mask, 1],
                s=self.umap_point_size,
                c=color,
                alpha=alpha,
                edgecolors="none",
                rasterized=True,
                label=f"{cat} (n={mask.sum():,})",
            )

        ax.set_title("Peak Embeddings: Ion Origin Categories", fontsize=14, fontweight="bold")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
            markerscale=3,
            borderaxespad=0.0,
        )
        ax.set_xticks([])
        ax.set_yticks([])

        save_path = save_dir / f"peak_umap_ion_categories{suffix}.png"
        fig.savefig(save_path, dpi=self.umap_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"  Saved: {save_path}")
        return str(save_path)

    # -- UMAP 3: Fragment Ion Ladder -----------------------------------

    def _plot_umap_fragment_ladder(
        self,
        umap_2d: np.ndarray,
        sampled: Dict[str, np.ndarray],
        save_dir: Path,
        suffix: str = "",
    ) -> Optional[str]:
        """UMAP with b/y ion ladder gradient and marker shapes for base/loss/isotope.

        Positions are capped at ``self.max_fragment_position`` so that rare
        high-position ions don't compress the color scale.  b-ions use a blue
        gradient (``winter``), y-ions use an orange gradient (``YlOrRd``) for
        clear visual separation.
        """
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        from matplotlib.lines import Line2D

        labels = sampled["multiclass_labels"]
        feature_types = sampled.get("feature_types", np.array([None] * len(labels)))
        annotations = sampled.get("matched_annotations", np.array([None] * len(labels)))
        parent_anns = sampled.get("parent_annotations", np.array([None] * len(labels)))

        n = len(labels)
        max_pos = self.max_fragment_position

        # Pre-compute per-peak: ion_series ("b"/"y"/None), position, role
        ion_series = np.empty(n, dtype=object)
        positions = np.full(n, -1, dtype=np.int32)
        roles = np.empty(n, dtype=object)  # "base", "loss", "isotope"

        for i in range(n):
            lbl = str(labels[i])
            ft = feature_types[i]
            ann = annotations[i]
            par = parent_anns[i]

            if lbl.startswith("b-"):
                ion_series[i] = "b"
            elif lbl.startswith("y-"):
                ion_series[i] = "y"
            else:
                ion_series[i] = None
                roles[i] = None
                continue

            if ft in ("loss", "isotope") and par is not None:
                positions[i] = extract_fragment_position(par)
            else:
                positions[i] = extract_fragment_position(ann)

            if ft == "loss":
                roles[i] = "loss"
            elif ft == "isotope":
                roles[i] = "isotope"
            else:
                roles[i] = "base"

        # Identify fragment vs background peaks
        has_series = np.array([s is not None for s in ion_series], dtype=bool)
        is_fragment = has_series & (positions > 0)
        is_background = ~is_fragment

        frag_positions = positions[is_fragment]
        if len(frag_positions) == 0:
            logger.warning("No fragment ions found — skipping fragment ladder UMAP")
            return None

        # Cap positions at max_fragment_position
        n_capped = int((frag_positions > max_pos).sum())
        positions = np.clip(positions, -1, max_pos)
        pos_min = 1
        pos_max = max_pos

        # Discrete position bins for clear color steps
        n_bins = min(max_pos, 10)  # ~10 discrete color steps
        bin_size = max(1, max_pos // n_bins)
        boundaries = list(range(pos_min, pos_max + bin_size, bin_size))
        if boundaries[-1] < pos_max:
            boundaries.append(pos_max + 1)

        norm = Normalize(vmin=pos_min, vmax=pos_max)

        # Distinct colormaps: cool blue for b, warm orange for y
        # Sample from the "dark to mid" range (0.25–0.85) to avoid white/invisible
        cmap_b = plt.cm.winter  # cyan-blue → green
        cmap_y = plt.cm.YlOrRd  # yellow → orange → red

        def _pos_to_color(series: str, pos: int):
            t = norm(pos)
            # Map to 0.2–0.9 range to avoid extremes (too light or too dark)
            mapped = 0.2 + t * 0.7
            return cmap_b(mapped) if series == "b" else cmap_y(mapped)

        # Marker map
        marker_map = {"base": "o", "loss": "v", "isotope": "D"}
        size_map = {
            "base": self.umap_point_size * 1.2,
            "loss": self.umap_point_size * 1.2,
            "isotope": self.umap_point_size,
        }

        fig, ax = plt.subplots(figsize=(12, 8))

        # Layer 1: background (non-fragment)
        if is_background.any():
            ax.scatter(
                umap_2d[is_background, 0],
                umap_2d[is_background, 1],
                s=self.umap_point_size * 0.5,
                c="#E0E0E0",
                alpha=0.1,
                edgecolors="none",
                rasterized=True,
            )

        # Layer 2: fragment ions — group by (series, role) then plot by position
        for series in ("b", "y"):
            for role in ("base", "loss", "isotope"):
                mask = is_fragment & (ion_series == series) & (roles == role)
                if not mask.any():
                    continue
                idx = np.where(mask)[0]
                pos_vals = positions[idx]
                sort_order = np.argsort(pos_vals)
                idx = idx[sort_order]

                colors = np.array([_pos_to_color(series, int(positions[j])) for j in idx])
                ax.scatter(
                    umap_2d[idx, 0],
                    umap_2d[idx, 1],
                    s=size_map.get(role, self.umap_point_size),
                    c=colors,
                    marker=marker_map.get(role, "o"),
                    alpha=0.7,
                    edgecolors="none",
                    rasterized=True,
                )

        # --- Dual colorbars ---
        ax_pos = ax.get_position()
        norm_cbar = Normalize(vmin=pos_min, vmax=pos_max)
        cbar_ticks = sorted(
            {
                pos_min,
                pos_max,
                *(t for t in range(5, pos_max + 1, 5) if pos_min <= t <= pos_max),
            }
        )

        # b-series colorbar (winter: dark→light with position)
        cax_b = fig.add_axes(
            [
                ax_pos.x1 + 0.02,
                ax_pos.y0 + ax_pos.height * 0.55,
                0.015,
                ax_pos.height * 0.35,
            ]
        )
        # Build matching colorbar using the same 0.2–0.9 mapping
        from matplotlib.colors import LinearSegmentedColormap

        b_colors_list = [cmap_b(0.2 + i / 255 * 0.7) for i in range(256)]
        cmap_b_bar = LinearSegmentedColormap.from_list("b_bar", b_colors_list)
        sm_b = ScalarMappable(cmap=cmap_b_bar, norm=norm_cbar)
        sm_b.set_array([])
        cb_b = fig.colorbar(sm_b, cax=cax_b)
        cb_b.set_label("b-ion position", fontsize=8)
        cb_b.set_ticks(cbar_ticks)
        cb_b.ax.tick_params(labelsize=7)

        # y-series colorbar (YlOrRd: yellow→red with position)
        cax_y = fig.add_axes(
            [
                ax_pos.x1 + 0.02,
                ax_pos.y0 + ax_pos.height * 0.05,
                0.015,
                ax_pos.height * 0.35,
            ]
        )
        y_colors_list = [cmap_y(0.2 + i / 255 * 0.7) for i in range(256)]
        cmap_y_bar = LinearSegmentedColormap.from_list("y_bar", y_colors_list)
        sm_y = ScalarMappable(cmap=cmap_y_bar, norm=norm_cbar)
        sm_y.set_array([])
        cb_y = fig.colorbar(sm_y, cax=cax_y)
        cb_y.set_label("y-ion position", fontsize=8)
        cb_y.set_ticks(cbar_ticks)
        cb_y.ax.tick_params(labelsize=7)

        # Marker shape legend
        marker_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=7, label="Base ion"),
            Line2D([0], [0], marker="v", color="w", markerfacecolor="gray", markersize=7, label="Neutral loss"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="gray", markersize=6, label="Isotope"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#E0E0E0", markersize=6, label="Non-fragment"),
        ]
        ax.legend(
            handles=marker_handles,
            loc="upper left",
            fontsize=8,
            framealpha=0.8,
        )

        # Counts annotation
        n_b = int((ion_series == "b").sum())
        n_y = int((ion_series == "y").sum())
        n_bg = int(is_background.sum())
        cap_note = f"  |  capped>{max_pos}: {n_capped:,}" if n_capped > 0 else ""
        ax.annotate(
            f"b-ions: {n_b:,}  |  y-ions: {n_y:,}  |  background: {n_bg:,}{cap_note}",
            xy=(0.5, -0.06),
            xycoords="axes fraction",
            ha="center",
            fontsize=8,
            color="gray",
        )

        title_suffix = " (residual)" if suffix else ""
        ax.set_title(
            f"Peak Embeddings: Fragment Ion Ladder{title_suffix}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_xticks([])
        ax.set_yticks([])

        save_path = save_dir / f"peak_umap_fragment_ladder{suffix}.png"
        fig.savefig(save_path, dpi=self.umap_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"  Saved: {save_path}")
        return str(save_path)

    @staticmethod
    def _derive_binary_from_multiclass(multiclass_results: Dict[str, Any]) -> Dict[str, Any]:
        """Derive binary (annotated vs unannotated) metrics from multiclass results.

        Merges all annotated classes into one and computes binary precision,
        recall, F1 from the multiclass confusion matrix.

        Args:
            multiclass_results: Results dict from _run_multiclass_classification.

        Returns:
            Dict with binary accuracy, macro_f1, weighted_f1, and per-class metrics.
        """
        if not multiclass_results or "class_names" not in multiclass_results:
            return {}

        class_names = multiclass_results["class_names"]
        conf = np.array(multiclass_results["confusion_matrix"])

        # Find the unannotated class index
        try:
            unann_idx = class_names.index("unannotated")
        except ValueError:
            return {}  # No unannotated class found

        # Collapse to 2×2: unannotated vs annotated
        ann_idxs = [i for i in range(len(class_names)) if i != unann_idx]
        tp_unann = conf[unann_idx, unann_idx]
        fp_unann = sum(conf[i, unann_idx] for i in ann_idxs)
        fn_unann = sum(conf[unann_idx, j] for j in ann_idxs)
        tp_ann = sum(conf[i, j] for i in ann_idxs for j in ann_idxs)
        fp_ann = fn_unann  # unannotated predicted as annotated
        fn_ann = fp_unann  # annotated predicted as unannotated

        total = conf.sum()
        accuracy = (tp_unann + tp_ann) / total if total > 0 else 0.0

        def _prf(tp, fp, fn):
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            return float(p), float(r), float(f)

        p_u, r_u, f_u = _prf(tp_unann, fp_unann, fn_unann)
        p_a, r_a, f_a = _prf(tp_ann, fp_ann, fn_ann)

        macro_f1 = (f_u + f_a) / 2
        support_u = int(tp_unann + fn_unann)
        support_a = int(tp_ann + fn_ann)
        weighted_f1 = (f_u * support_u + f_a * support_a) / total if total > 0 else 0.0

        # Build 2×2 confusion matrix: rows=true, cols=predicted
        # Order: [unannotated, annotated]
        binary_conf = [
            [int(tp_unann), int(fn_unann)],  # true=unannotated
            [int(fp_unann), int(tp_ann)],  # true=annotated
        ]

        return {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "per_class_f1": [f_u, f_a],
            "per_class_precision": [p_u, p_a],
            "per_class_recall": [r_u, r_a],
            "per_class_support": [support_u, support_a],
            "confusion_matrix": binary_conf,
            "class_names": ["unannotated", "annotated"],
            "derived_from_multiclass": True,
        }

    # ------------------------------------------------------------------ #
    #  Pairwise Similarity + Structural Consistency (Features 1 + 4)      #
    # ------------------------------------------------------------------ #

    def _find_parent_child_pairs(
        self,
        peak_data: Dict[str, np.ndarray],
    ) -> Tuple[
        List[Tuple[int, int, str]],
        List[Tuple[int, int, str]],
        Dict[int, List[Tuple[int, int, str]]],
    ]:
        """Find parent-child peak pairs within each spectrum.

        For each peak with a non-None parent_annotation, finds the annotated
        peak(s) whose matched_annotation matches.

        Returns:
            canonical_pairs: One (parent_idx, child_idx, child_type) per child,
                selecting the parent closest by m/z for similarity summaries.
            all_pairs: All valid (parent_idx, child_idx, child_type) tuples
                (for retrieval evaluation where multiple candidates matter).
            per_spectrum_pairs: {spectrum_idx: [canonical pairs in that spectrum]}.
        """
        spec_indices = peak_data["spectrum_indices_per_peak"]
        matched_anns = peak_data["matched_annotations"]
        parent_anns = peak_data["parent_annotations"]
        feature_types = peak_data["feature_types"]
        mz_values = peak_data["mz_values"]

        # Group global peak indices by spectrum
        spectrum_to_peaks: Dict[int, List[int]] = defaultdict(list)
        for global_idx, spec_id in enumerate(spec_indices):
            spectrum_to_peaks[int(spec_id)].append(global_idx)

        canonical_pairs: List[Tuple[int, int, str]] = []
        all_pairs: List[Tuple[int, int, str]] = []
        per_spectrum_pairs: Dict[int, List[Tuple[int, int, str]]] = {}

        for spec_id, peak_idxs in spectrum_to_peaks.items():
            # Build annotation → [global indices] for annotated peaks
            ann_to_idxs: Dict[str, List[int]] = defaultdict(list)
            for gidx in peak_idxs:
                ann = matched_anns[gidx]
                if ann is not None and ann != "":
                    ann_to_idxs[str(ann)].append(gidx)

            spec_canonical = []
            for gidx in peak_idxs:
                par = parent_anns[gidx]
                if par is None or par == "":
                    continue
                ft = feature_types[gidx]
                if ft not in ("loss", "isotope"):
                    continue
                child_type = str(ft)  # "loss" or "isotope"

                candidates = ann_to_idxs.get(str(par), [])
                if not candidates:
                    continue

                # All pairs for retrieval
                for pidx in candidates:
                    all_pairs.append((pidx, gidx, child_type))

                # Canonical: closest parent by m/z
                child_mz = mz_values[gidx]
                best_parent = min(candidates, key=lambda p: abs(mz_values[p] - child_mz))
                pair = (best_parent, gidx, child_type)
                canonical_pairs.append(pair)
                spec_canonical.append(pair)

            if spec_canonical:
                per_spectrum_pairs[spec_id] = spec_canonical

        return canonical_pairs, all_pairs, per_spectrum_pairs

    def _sample_within_spectrum_pairs(
        self,
        peak_data: Dict[str, np.ndarray],
        mask_a: np.ndarray,
        mask_b: np.ndarray,
        max_pairs: int,
        same_set: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample peak pairs within each spectrum, then pool globally.

        Args:
            peak_data: Peak data dict with spectrum_indices_per_peak.
            mask_a: Boolean mask for set A peaks (N_peaks,).
            mask_b: Boolean mask for set B peaks (N_peaks,).
            max_pairs: Maximum total pairs to return.
            same_set: If True, mask_a == mask_b and we enforce i < j.

        Returns:
            (indices_a, indices_b) — global peak indices for each pair.
        """
        spec_indices = peak_data["spectrum_indices_per_peak"]
        unique_specs = np.unique(spec_indices)

        # Collect candidate pairs per spectrum
        all_a, all_b = [], []
        for spec_id in unique_specs:
            spec_mask = spec_indices == spec_id
            idxs_a = np.where(spec_mask & mask_a)[0]
            idxs_b = np.where(spec_mask & mask_b)[0]
            if len(idxs_a) == 0 or len(idxs_b) == 0:
                continue

            if same_set:
                # Sample pairs (i, j) with i < j from same set
                if len(idxs_a) < 2:
                    continue
                n_possible = len(idxs_a) * (len(idxs_a) - 1) // 2
                n_sample = min(n_possible, max(1, max_pairs // len(unique_specs)))
                for _ in range(n_sample):
                    i, j = np.random.choice(len(idxs_a), 2, replace=False)
                    if i > j:
                        i, j = j, i
                    all_a.append(idxs_a[i])
                    all_b.append(idxs_a[j])
            else:
                n_possible = len(idxs_a) * len(idxs_b)
                n_sample = min(n_possible, max(1, max_pairs // len(unique_specs)))
                sa = np.random.choice(len(idxs_a), n_sample, replace=True)
                sb = np.random.choice(len(idxs_b), n_sample, replace=True)
                all_a.extend(idxs_a[sa])
                all_b.extend(idxs_b[sb])

        if not all_a:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        all_a = np.array(all_a, dtype=np.int64)
        all_b = np.array(all_b, dtype=np.int64)

        # Subsample to max_pairs if needed
        if len(all_a) > max_pairs:
            sel = np.random.choice(len(all_a), max_pairs, replace=False)
            all_a, all_b = all_a[sel], all_b[sel]

        return all_a, all_b

    @staticmethod
    def _compute_group_stats(
        similarities: np.ndarray,
        per_spectrum_sims: Optional[Dict[int, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Compute summary statistics for a group of cosine similarities.

        Args:
            similarities: Array of cosine similarity values.
            per_spectrum_sims: Optional dict {spec_id: similarities} for
                spectrum-aware aggregation.

        Returns:
            Dict with mean, std, median, q25, q75, n_pairs, n_spectra,
            low_sample_warning, and optional per_spectrum_means.
        """
        n = len(similarities)
        result: Dict[str, Any] = {
            "n_pairs": int(n),
            "low_sample_warning": n < 30,
        }
        if n == 0:
            result.update({"mean": None, "std": None, "median": None, "q25": None, "q75": None})
        else:
            result.update(
                {
                    "mean": float(np.mean(similarities)),
                    "std": float(np.std(similarities)),
                    "median": float(np.median(similarities)),
                    "q25": float(np.percentile(similarities, 25)),
                    "q75": float(np.percentile(similarities, 75)),
                }
            )

        if per_spectrum_sims is not None:
            spec_means = np.array([float(np.mean(s)) for s in per_spectrum_sims.values() if len(s) > 0])
            result["n_spectra"] = int(len(spec_means))
            if len(spec_means) > 0:
                result["per_spectrum_means"] = {
                    "mean": float(np.mean(spec_means)),
                    "std": float(np.std(spec_means)),
                    "median": float(np.median(spec_means)),
                }
            else:
                result["per_spectrum_means"] = {"mean": None, "std": None, "median": None}
        else:
            result["n_spectra"] = None

        return result

    def _compute_parent_recovery(
        self,
        all_pairs: List[Tuple[int, int, str]],
        embeddings_norm: np.ndarray,
        peak_data: Dict[str, np.ndarray],
        k_values: Tuple[int, ...] = (1, 3),
    ) -> Dict[str, Any]:
        """Compute parent recovery: for each child, does the true parent rank in top-k?

        Candidates are restricted to annotated peaks in the same spectrum.

        Args:
            all_pairs: (parent_idx, child_idx, child_type) for all candidates.
            embeddings_norm: L2-normalized embeddings (N_peaks, D).
            peak_data: Peak data dict.
            k_values: Tuple of k values for top-k recovery.

        Returns:
            Dict with top-k recovery rates and n_children.
        """
        if not all_pairs:
            return {f"top{k}": None for k in k_values} | {"n_children": 0}

        spec_indices = peak_data["spectrum_indices_per_peak"]
        binary_labels = peak_data["binary_labels"]
        annotated_mask = binary_labels == 1

        # Group pairs by child
        child_to_parents: Dict[int, List[int]] = defaultdict(list)
        for pidx, cidx, _ in all_pairs:
            child_to_parents[cidx].append(pidx)

        recoveries = {k: [] for k in k_values}

        for cidx, parent_idxs in child_to_parents.items():
            spec_id = spec_indices[cidx]
            # All annotated peaks in same spectrum (candidates)
            spec_mask = (spec_indices == spec_id) & annotated_mask
            candidate_idxs = np.where(spec_mask)[0]
            # Exclude the child itself from candidates
            candidate_idxs = candidate_idxs[candidate_idxs != cidx]
            if len(candidate_idxs) == 0:
                continue

            # Cosine similarities to all candidates
            child_emb = embeddings_norm[cidx]  # (D,)
            cand_embs = embeddings_norm[candidate_idxs]  # (C, D)
            sims = cand_embs @ child_emb  # (C,)

            # Rank by descending similarity
            ranked_idxs = candidate_idxs[np.argsort(-sims)]
            parent_set = set(parent_idxs)

            for k in k_values:
                top_k = set(ranked_idxs[:k].tolist())
                hit = 1.0 if top_k & parent_set else 0.0
                recoveries[k].append(hit)

        result = {"n_children": int(len(child_to_parents))}
        for k in k_values:
            vals = recoveries[k]
            result[f"top{k}"] = float(np.mean(vals)) if vals else None
        return result

    def _run_pairwise_similarity_analysis(
        self,
        peak_data: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run pairwise cosine similarity analysis across peak relation groups.

        Returns:
            Tuple of (pairwise_results, structural_results).
        """
        np.random.seed(self.random_state)
        X = peak_data["embeddings"]
        # L2-normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        X_norm = X / norms

        binary_labels = peak_data["binary_labels"]
        multiclass_labels = peak_data["multiclass_labels"]
        spec_indices = peak_data["spectrum_indices_per_peak"]

        annotated_mask = binary_labels == 1
        unannotated_mask = binary_labels == 0

        # --- Within-spectrum baseline ---
        relation_groups: Dict[str, Dict[str, Any]] = {}

        all_mask = np.ones(len(binary_labels), dtype=bool)
        idx_a, idx_b = self._sample_within_spectrum_pairs(peak_data, all_mask, all_mask, self.max_pairs_per_group, same_set=True)
        if len(idx_a) > 0:
            sims = np.sum(X_norm[idx_a] * X_norm[idx_b], axis=1)
            relation_groups["random_within_spectrum"] = self._compute_group_stats(sims)
        else:
            relation_groups["random_within_spectrum"] = self._compute_group_stats(np.array([]))

        # --- Parent-child structural consistency ---
        canonical_pairs, all_retrieval_pairs, per_spec_pairs = self._find_parent_child_pairs(peak_data)

        structural: Dict[str, Any] = {}
        for child_type in ("isotope", "loss"):
            typed_pairs = [(p, c, t) for p, c, t in canonical_pairs if t == child_type]
            if typed_pairs:
                parent_idxs = np.array([p for p, c, t in typed_pairs])
                child_idxs = np.array([c for p, c, t in typed_pairs])
                sims = np.sum(X_norm[parent_idxs] * X_norm[child_idxs], axis=1)
                # Per-spectrum aggregation
                per_spec_pc: Dict[int, list] = defaultdict(list)
                for i, s in enumerate(sims):
                    per_spec_pc[int(spec_indices[parent_idxs[i]])].append(s)
                structural[f"{child_type}_parent"] = self._compute_group_stats(sims, {k: np.array(v) for k, v in per_spec_pc.items()})
            else:
                structural[f"{child_type}_parent"] = self._compute_group_stats(np.array([]))

        # Parent recovery retrieval metric
        structural["parent_recovery"] = self._compute_parent_recovery(all_retrieval_pairs, X_norm, peak_data)

        # --- Logging ---
        # Compact summary: key structural groups + parent recovery
        iso_p = structural.get("isotope_parent", {})
        loss_p = structural.get("loss_parent", {})
        rec = structural.get("parent_recovery", {})
        iso_m = f"{iso_p['mean']:.4f}" if iso_p.get("mean") is not None else "N/A"
        loss_m = f"{loss_p['mean']:.4f}" if loss_p.get("mean") is not None else "N/A"
        t1 = f"{rec['top1']:.4f}" if rec.get("top1") is not None else "N/A"
        logger.info(f"  Pairwise: isotope_parent={iso_m}  loss_parent={loss_m}  parent_recovery_top1={t1}")

        # --- Plots ---
        pairwise_plot = None
        if self.create_plots and PLOTTING_AVAILABLE:
            pairwise_plot = self._plot_pairwise_similarity(relation_groups, structural)

        pairwise_results = {
            "relation_groups": relation_groups,
            "plot_path": pairwise_plot,
        }
        structural_results = {
            **structural,
            "plot_path": pairwise_plot,  # shared plot
        }
        return pairwise_results, structural_results

    def _plot_pairwise_similarity(
        self,
        relation_groups: Dict[str, Dict[str, Any]],
        structural: Dict[str, Dict[str, Any]],
    ) -> Optional[str]:
        """Create a two-panel summary of peak embedding similarity.

        Top panel: within-spectrum structural relationships
        Bottom panel: cross-spectrum ion identity (if available)
        """
        save_dir = self.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        # --- Build panel data ---
        # Top panel: within-spectrum structural + baseline
        within_order = [
            "isotope_parent",
            "loss_parent",
            "complementary_by",
            "cross_charge",
            "random_within_spectrum",
        ]
        within_labels = {
            "isotope_parent": "Isotope → parent",
            "loss_parent": "Neutral loss → parent",
            "complementary_by": "Complementary b/y pair",
            "cross_charge": "Same fragment, different charge",
            "random_within_spectrum": "Random (within-spectrum)",
        }
        within_colors = {
            "isotope_parent": "#4CAF50",
            "loss_parent": "#66BB6A",
            "complementary_by": "#7E57C2",
            "cross_charge": "#FF9800",
            "random_within_spectrum": "#9E9E9E",
        }

        all_data = {**relation_groups}
        for k, v in structural.items():
            if k != "parent_recovery" and isinstance(v, dict) and "mean" in v:
                all_data[k] = v

        within_groups = [(k, all_data[k]) for k in within_order if k in all_data and all_data[k].get("mean") is not None]

        # Bottom panel: cross-spectrum identity
        cross_order = ["same_ion_cross_spectrum", "different_ion_similar_mz", "random_cross_spectrum"]
        cross_labels = {
            "same_ion_cross_spectrum": "Same ion (cross-spectrum)",
            "different_ion_similar_mz": "Different ion, similar m/z",
            "random_cross_spectrum": "Random (cross-spectrum)",
        }
        cross_colors = {
            "same_ion_cross_spectrum": "#2196F3",
            "different_ion_similar_mz": "#EF5350",
            "random_cross_spectrum": "#9E9E9E",
        }

        cross_groups = [(k, all_data[k]) for k in cross_order if k in all_data and all_data[k].get("mean") is not None]

        if not within_groups and not cross_groups:
            return None

        # --- Create figure ---
        n_within = len(within_groups)
        n_cross = len(cross_groups)
        n_panels = (1 if n_within else 0) + (1 if n_cross else 0)
        if n_panels == 0:
            return None

        height_ratios = []
        if n_within:
            height_ratios.append(max(2, n_within * 0.7))
        if n_cross:
            height_ratios.append(max(2, n_cross * 0.7))

        fig, axes = plt.subplots(
            n_panels,
            1,
            figsize=(10, sum(height_ratios) + 1.5),
            gridspec_kw={"height_ratios": height_ratios} if n_panels > 1 else None,
            squeeze=False,
        )
        axes = axes.flatten()

        def _draw_panel(ax, groups, labels, colors, title):
            names = [k for k, _ in groups]
            stats_list = [s for _, s in groups]
            y_pos = np.arange(len(names))

            for i, (key, stats) in enumerate(zip(names, stats_list, strict=False)):
                mean = stats["mean"]
                q25 = stats.get("q25", mean)
                q75 = stats.get("q75", mean)
                n = stats.get("n_pairs", 0)
                color = colors.get(key, "#9E9E9E")

                ax.barh(i, q75 - q25, left=q25, height=0.6, color=color, alpha=0.4)
                ax.plot(mean, i, "o", color=color, markersize=8, zorder=5)
                warn = " ⚠" if stats.get("low_sample_warning") else ""
                ax.annotate(f"  {mean:.3f} (n={n:,}){warn}", (q75, i), va="center", fontsize=8)

            ax.set_yticks(y_pos)
            ax.set_yticklabels([labels.get(k, k.replace("_", " ")) for k in names], fontsize=9)
            ax.set_xlabel("Cosine Similarity", fontsize=9)
            ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
            ax.invert_yaxis()
            ax.set_xlim(-0.05, 1.3)

        panel_idx = 0
        if n_within:
            rec = structural.get("parent_recovery", {})
            title = "Within-Spectrum Structural Relationships"
            if rec.get("top1") is not None:
                title += f"  (parent recovery: top1={rec['top1']:.3f})"
            _draw_panel(axes[panel_idx], within_groups, within_labels, within_colors, title)
            panel_idx += 1

        if n_cross:
            # Try to get AUROC from cross_spectrum_results (passed via structural)
            auroc = structural.get("_cross_spectrum_auroc")
            title = "Cross-Spectrum Ion Identity"
            if auroc is not None:
                title += f"  (AUROC={auroc:.3f})"
            _draw_panel(axes[panel_idx], cross_groups, cross_labels, cross_colors, title)

        fig.suptitle("Peak Embedding Similarity Analysis", fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()

        save_path = save_dir / "pairwise_similarity.png"
        fig.savefig(save_path, dpi=self.umap_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"  Saved: {save_path}")
        return str(save_path)

    # ------------------------------------------------------------------ #
    #  Cross-Spectrum Ion Identity Analysis                                #
    # ------------------------------------------------------------------ #

    def _run_cross_spectrum_ion_identity(
        self,
        peak_data: Dict[str, np.ndarray],
        meta: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Test whether the same fragment ion gets similar embeddings across spectra.

        Groups peaks by (peptide, ion_type, position, charge) and compares
        within-group (same ion, different spectra) cosine similarity against
        two controls: different-ion-at-similar-m/z and random cross-spectrum pairs.

        Falls back to position-only grouping (ion_type, position, charge) when
        fewer than 100 peptide-level groups have replicates.
        """
        X = peak_data["embeddings"]
        spec_per_peak = peak_data["spectrum_indices_per_peak"]
        feature_types = peak_data["feature_types"]
        annotations = peak_data["matched_annotations"]
        mz_values = peak_data["mz_values"]

        # L2-normalize embeddings in-place to avoid a full copy (~3GB for 1M×768)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        X_norm = np.empty_like(X)
        # Normalize in chunks to limit peak memory
        chunk = 50_000
        for start in range(0, len(X), chunk):
            end = min(start + chunk, len(X))
            X_norm[start:end] = X[start:end] / norms[start:end]
        del norms

        # ---- Step 1: build ion key per peak ----
        # Only base fragment ions (b/y/a) with valid annotations.
        # Use compact parallel arrays instead of per-peak Python objects.
        n_peaks = len(X)
        base_ion_mask = np.zeros(n_peaks, dtype=bool)
        ion_type_arr = np.empty(n_peaks, dtype="U1")  # single char
        position_arr = np.full(n_peaks, -1, dtype=np.int32)
        charge_arr = np.full(n_peaks, -1, dtype=np.int32)
        # Cache peptide per spectrum (not per peak) to avoid repeated lookups
        _peptide_cache: Dict[int, str] = {}
        peptide_spec_arr = np.empty(n_peaks, dtype=object)  # peptide string per peak

        for i in range(n_peaks):
            ft = feature_types[i]
            ann = annotations[i]
            if ft != "base" or ann is None or not isinstance(ann, str) or ann == "":
                continue
            ion_type = extract_ion_type(ann)
            if ion_type in ("unknown", "p"):
                continue
            position = extract_fragment_position(ann)
            if position < 1:
                continue
            charge = extract_charge_from_annotation(ann)
            spec_idx = int(spec_per_peak[i])
            if spec_idx not in _peptide_cache:
                _peptide_cache[spec_idx] = self._get_sequence_for_spectrum(meta, spec_idx)
            peptide = _peptide_cache[spec_idx]
            if peptide == "unknown":
                continue
            base_ion_mask[i] = True
            ion_type_arr[i] = ion_type
            position_arr[i] = position
            charge_arr[i] = charge
            peptide_spec_arr[i] = peptide

        # Build ion_keys only for base ions (sparse — avoids N-length Python list)
        base_indices = np.where(base_ion_mask)[0]
        ion_keys: Dict[int, tuple] = {}  # peak_idx → (peptide, ion_type, position, charge)
        peptide_per_peak: Dict[int, str] = {}
        for i in base_indices:
            pep = peptide_spec_arr[i]
            ion_keys[i] = (pep, ion_type_arr[i], int(position_arr[i]), int(charge_arr[i]))
            peptide_per_peak[i] = pep

        # ---- Step 2: group peaks by ion key ----
        ion_groups: Dict[tuple, List[int]] = defaultdict(list)
        for i, key in ion_keys.items():
            ion_groups[key].append(i)

        # Filter to groups with ≥2 observations from ≥2 distinct spectra
        cross_spectrum_groups: Dict[tuple, List[int]] = {}
        for key, indices in ion_groups.items():
            spec_ids = set(int(spec_per_peak[i]) for i in indices)
            if len(spec_ids) >= 2 and len(indices) >= 2:
                cross_spectrum_groups[key] = indices

        n_ion_groups = len(cross_spectrum_groups)
        n_peptides = len(set(k[0] for k in cross_spectrum_groups))

        if n_ion_groups == 0:
            logger.warning("  No cross-spectrum ion replicates found — skipping analysis")
            return {
                "n_unique_ions_with_replicates": 0,
                "n_peptides_with_replicates": 0,
                "same_ion_cross_spectrum": self._compute_group_stats(np.array([])),
                "different_ion_similar_mz": self._compute_group_stats(np.array([])),
                "random_cross_spectrum": self._compute_group_stats(np.array([])),
                "auroc_same_vs_different_mz": None,
                "position_only_fallback": None,
                "plot_path": None,
            }

        # ---- Step 3: sample same-ion cross-spectrum pairs ----
        same_ion_sims = self._sample_cross_spectrum_same_ion_pairs(cross_spectrum_groups, spec_per_peak, X_norm)

        # ---- Step 4: m/z-matched control pairs ----
        # base_ion_mask was computed in step 1
        base_mask = base_ion_mask
        mz_control_sims = self._sample_mz_matched_control_pairs(
            cross_spectrum_groups,
            ion_keys,
            peptide_per_peak,
            base_mask,
            mz_values,
            spec_per_peak,
            X_norm,
        )

        # ---- Step 5: random cross-spectrum pairs ----
        random_sims = self._sample_random_cross_spectrum_pairs(base_mask, spec_per_peak, X_norm)

        # ---- Step 6: statistics ----
        same_stats = self._compute_group_stats(same_ion_sims)
        mz_stats = self._compute_group_stats(mz_control_sims)
        random_stats = self._compute_group_stats(random_sims)

        # ---- Step 7: AUROC ----
        auroc = None
        if len(same_ion_sims) >= 10 and len(mz_control_sims) >= 10:
            labels = np.concatenate(
                [
                    np.ones(len(same_ion_sims)),
                    np.zeros(len(mz_control_sims)),
                ]
            )
            scores = np.concatenate([same_ion_sims, mz_control_sims])
            try:
                auroc = float(roc_auc_score(labels, scores))
            except ValueError:
                auroc = None

        # Persist the pair-level similarities so the Figure 4c ROC curve can be drawn without
        # re-running the eval. The AUROC alone is not enough to plot a curve.
        self._save_cross_spectrum_pairs(same_ion_sims, mz_control_sims, random_sims, auroc)

        same_m = f"{same_stats['mean']:.4f}" if same_stats.get("mean") is not None else "N/A"
        mz_m = f"{mz_stats['mean']:.4f}" if mz_stats.get("mean") is not None else "N/A"
        auroc_s = f"{auroc:.4f}" if auroc is not None else "N/A"
        logger.info(f"  Cross-spectrum: same-ion={same_m}  mz-control={mz_m}  AUROC={auroc_s}")

        # ---- Step 8: position-only fallback ----
        position_fallback = None
        if n_ion_groups < 100:
            position_fallback = self._run_position_only_fallback(
                ion_keys,
                feature_types,
                annotations,
                spec_per_peak,
                mz_values,
                base_mask,
                X_norm,
            )

        # ---- Step 9: cross-charge similarity ----
        # Group by (peptide, ion_type, position) ignoring charge to find
        # the same fragment ion observed at different charge states.
        cross_charge_results = self._run_cross_charge_analysis(ion_keys, charge_arr, spec_per_peak, X_norm)

        return {
            "same_ion_cross_spectrum": same_stats,
            "different_ion_similar_mz": mz_stats,
            "random_cross_spectrum": random_stats,
            "auroc_same_vs_different_mz": auroc,
            "n_unique_ions_with_replicates": n_ion_groups,
            "n_peptides_with_replicates": n_peptides,
            "position_only_fallback": position_fallback,
            "cross_charge": cross_charge_results,
        }

    def _save_cross_spectrum_pairs(
        self,
        same_ion_sims: Optional[np.ndarray],
        mz_control_sims: Optional[np.ndarray],
        random_sims: Optional[np.ndarray],
        auroc: Optional[float],
    ) -> None:
        """Dump cross-spectrum pair similarities for the Figure 4c ROC curve.

        The task reports only the AUROC, which cannot be turned back into a curve. Writes one
        row per sampled pair with its relation and cosine similarity, plus the AUROC for
        cross-checking against the figure.
        """
        try:
            import polars as pl

            rows = []
            for name, sims in (
                ("same_ion_cross_spectrum", same_ion_sims),
                ("different_ion_similar_mz", mz_control_sims),
                ("random_cross_spectrum", random_sims),
            ):
                if sims is None or len(sims) == 0:
                    continue
                rows.append(
                    pl.DataFrame(
                        {
                            "relation": [name] * len(sims),
                            "cosine_similarity": np.asarray(sims, dtype=float),
                        }
                    )
                )
            if not rows:
                return
            out_dir = getattr(self, "output_dir", None) or Path(".")
            path = Path(out_dir) / "cross_spectrum_pair_similarities.parquet"
            df = pl.concat(rows)
            df.write_parquet(path)
            logger.info(f"  Saved {len(df):,} cross-spectrum pair similarities to {path} (AUROC {auroc if auroc is None else round(auroc, 4)})")
        except Exception as e:  # never let a diagnostic dump break the eval
            logger.warning(f"  Could not save cross-spectrum pair similarities: {e}")

    def _sample_cross_spectrum_same_ion_pairs(
        self,
        cross_spectrum_groups: Dict[tuple, List[int]],
        spec_per_peak: np.ndarray,
        X_norm: np.ndarray,
    ) -> np.ndarray:
        """Sample cosine similarities between same-ion peaks from different spectra."""
        rng = np.random.RandomState(self.random_state)
        all_sims: List[float] = []
        budget = self.cross_spectrum_max_pairs
        # Distribute budget roughly equally across groups
        per_group = max(1, budget // max(len(cross_spectrum_groups), 1))

        for indices in cross_spectrum_groups.values():
            # Build cross-spectrum pairs
            specs = np.array([int(spec_per_peak[i]) for i in indices])
            idx_arr = np.array(indices)
            pairs = []
            for j in range(len(idx_arr)):
                for k in range(j + 1, len(idx_arr)):
                    if specs[j] != specs[k]:
                        pairs.append((idx_arr[j], idx_arr[k]))
            if not pairs:
                continue
            # Subsample if needed
            if len(pairs) > per_group:
                chosen = rng.choice(len(pairs), per_group, replace=False)
                pairs = [pairs[c] for c in chosen]
            for a, b in pairs:
                all_sims.append(float(np.dot(X_norm[a], X_norm[b])))
            if len(all_sims) >= budget:
                break

        return np.array(all_sims[:budget], dtype=np.float64) if all_sims else np.array([], dtype=np.float64)

    def _sample_mz_matched_control_pairs(
        self,
        cross_spectrum_groups: Dict[tuple, List[int]],
        ion_keys: Dict[int, tuple],
        peptide_per_peak: Dict[int, str],
        base_mask: np.ndarray,
        mz_values: np.ndarray,
        spec_per_peak: np.ndarray,
        X_norm: np.ndarray,
    ) -> np.ndarray:
        """Sample cosine similarities between different ions at similar m/z across spectra."""
        rng = np.random.RandomState(self.random_state + 1)
        all_sims: List[float] = []
        budget = self.cross_spectrum_max_pairs
        tol = self.cross_spectrum_mz_tolerance

        # Precompute sorted m/z for efficient range queries
        base_indices = np.where(base_mask)[0]
        if len(base_indices) == 0:
            return np.array([], dtype=np.float64)
        base_mz = mz_values[base_indices]
        sort_order = np.argsort(base_mz)
        sorted_mz = base_mz[sort_order]
        sorted_indices = base_indices[sort_order]

        per_group = max(1, budget // max(len(cross_spectrum_groups), 1))

        for key, group_indices in cross_spectrum_groups.items():
            peptide = key[0]
            # Mean m/z of this ion group
            group_mz = np.mean([mz_values[i] for i in group_indices])

            # Find peaks within ±tol of group_mz from different peptides
            lo = np.searchsorted(sorted_mz, group_mz - tol, side="left")
            hi = np.searchsorted(sorted_mz, group_mz + tol, side="right")
            candidates = sorted_indices[lo:hi]

            # Filter to different peptides
            candidates = [c for c in candidates if c in peptide_per_peak and peptide_per_peak[c] != peptide]
            if not candidates or not group_indices:
                continue

            # Sample pairs: one from group, one from candidates (different spectra)
            n_sample = min(per_group, len(candidates) * len(group_indices))
            for _ in range(n_sample):
                a = group_indices[rng.randint(len(group_indices))]
                b = candidates[rng.randint(len(candidates))]
                if int(spec_per_peak[a]) != int(spec_per_peak[b]):
                    all_sims.append(float(np.dot(X_norm[a], X_norm[b])))
            if len(all_sims) >= budget:
                break

        return np.array(all_sims[:budget], dtype=np.float64) if all_sims else np.array([], dtype=np.float64)

    def _sample_random_cross_spectrum_pairs(
        self,
        base_mask: np.ndarray,
        spec_per_peak: np.ndarray,
        X_norm: np.ndarray,
    ) -> np.ndarray:
        """Sample random cross-spectrum base-ion pairs as baseline."""
        rng = np.random.RandomState(self.random_state + 2)
        base_indices = np.where(base_mask)[0]
        if len(base_indices) < 2:
            return np.array([], dtype=np.float64)

        sims: List[float] = []
        budget = self.cross_spectrum_max_pairs
        attempts = 0
        max_attempts = budget * 5

        while len(sims) < budget and attempts < max_attempts:
            a, b = rng.choice(base_indices, 2, replace=False)
            if int(spec_per_peak[a]) != int(spec_per_peak[b]):
                sims.append(float(np.dot(X_norm[a], X_norm[b])))
            attempts += 1

        return np.array(sims, dtype=np.float64)

    def _run_position_only_fallback(
        self,
        ion_keys: Dict[int, tuple],
        feature_types: np.ndarray,
        annotations: np.ndarray,
        spec_per_peak: np.ndarray,
        mz_values: np.ndarray,
        base_mask: np.ndarray,
        X_norm: np.ndarray,
    ) -> Dict[str, Any]:
        """Fallback: group by (ion_type, position, charge) ignoring peptide identity.

        Tests the weaker hypothesis that all ions of the same type/position
        share some universal structure across different peptides.
        """
        # Build position-only groups
        pos_groups: Dict[tuple, List[int]] = defaultdict(list)
        for i, key in ion_keys.items():
            # key = (peptide, ion_type, position, charge)
            pos_key = key[1:]  # (ion_type, position, charge)
            pos_groups[pos_key].append(i)

        # Filter to groups with ≥2 observations from ≥2 spectra
        cross_pos_groups: Dict[tuple, List[int]] = {}
        for key, indices in pos_groups.items():
            spec_ids = set(int(spec_per_peak[i]) for i in indices)
            if len(spec_ids) >= 2 and len(indices) >= 2:
                cross_pos_groups[key] = indices

        n_groups = len(cross_pos_groups)

        if n_groups == 0:
            return {"n_position_groups": 0, "same_position_cross_spectrum": self._compute_group_stats(np.array([]))}

        # Sample same-position cross-spectrum pairs
        same_sims = self._sample_cross_spectrum_same_ion_pairs(cross_pos_groups, spec_per_peak, X_norm)

        return {
            "n_position_groups": n_groups,
            "same_position_cross_spectrum": self._compute_group_stats(same_sims),
        }

    def _run_cross_charge_analysis(
        self,
        ion_keys: Dict[int, tuple],
        charge_arr: np.ndarray,
        spec_per_peak: np.ndarray,
        X_norm: np.ndarray,
    ) -> Dict[str, Any]:
        """Test whether the same fragment at different charge states gets similar embeddings.

        Groups by (peptide, ion_type, position) ignoring charge. Within each
        group, samples pairs of peaks with different charges and computes
        cosine similarity. Also computes a same-charge control within groups
        that have multiple charge observations.
        """
        # Group by (peptide, ion_type, position) — charge-agnostic key
        charge_agnostic_groups: Dict[tuple, List[int]] = defaultdict(list)
        for i, key in ion_keys.items():
            # key = (peptide, ion_type, position, charge)
            ca_key = key[:3]  # (peptide, ion_type, position)
            charge_agnostic_groups[ca_key].append(i)

        # Find groups with ≥2 distinct charge states
        multi_charge_groups: Dict[tuple, List[int]] = {}
        for key, indices in charge_agnostic_groups.items():
            charges = set(int(charge_arr[i]) for i in indices)
            if len(charges) >= 2:
                multi_charge_groups[key] = indices

        n_groups = len(multi_charge_groups)

        if n_groups == 0:
            return {
                "n_multi_charge_groups": 0,
                "cross_charge_similarity": self._compute_group_stats(np.array([])),
            }

        rng = np.random.RandomState(self.random_state + 3)
        cross_charge_sims: List[float] = []
        budget = self.cross_spectrum_max_pairs

        for indices in multi_charge_groups.values():
            charges = np.array([int(charge_arr[i]) for i in indices])
            specs = np.array([int(spec_per_peak[i]) for i in indices])
            idx_arr = np.array(indices)

            # Within-spectrum cross-charge pairs only
            for j in range(len(idx_arr)):
                for k in range(j + 1, len(idx_arr)):
                    if charges[j] != charges[k] and specs[j] == specs[k]:
                        cross_charge_sims.append(float(np.dot(X_norm[idx_arr[j]], X_norm[idx_arr[k]])))

            if len(cross_charge_sims) >= budget:
                break

        cc_arr = np.array(cross_charge_sims[:budget], dtype=np.float64) if cross_charge_sims else np.array([], dtype=np.float64)
        cc_stats = self._compute_group_stats(cc_arr)

        return {
            "n_multi_charge_groups": n_groups,
            "cross_charge_similarity": cc_stats,
        }

    # ------------------------------------------------------------------ #
    #  Pre-transformer Probe (Feature 2)                                  #
    # ------------------------------------------------------------------ #

    def _run_pretransformer_probe(
        self,
        peak_data: Dict[str, np.ndarray],
        split_groups: np.ndarray,
        post_multiclass: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare pre-transformer vs post-transformer embeddings via classification.

        Runs multiclass classification only (binary is derived from multiclass).
        Uses fixed L2 for speed — the delta is what matters, not absolute perf.

        Args:
            peak_data: Peak data dict (must contain embeddings_pretransformer).
            split_groups: Per-peak group IDs for splitting.
            post_multiclass: Multiclass classification results from post-transformer.

        Returns:
            Dict with pre-transformer results and deltas, or {} if unavailable.
        """
        X_pre = peak_data.get("embeddings_pretransformer")
        if X_pre is None:
            logger.warning("Pre-transformer embeddings not available — skipping probe")
            return {}

        multiclass_labels = peak_data["multiclass_labels"]

        # Use fixed L2 for speed (disable grid search temporarily)
        orig_grid_search = self.use_grid_search
        self.use_grid_search = False

        pre_multiclass = self._run_multiclass_classification(X_pre, multiclass_labels, split_groups)
        for key in ["test_indices", "y_test_labels", "y_pred_labels", "y_pred_confidence"]:
            pre_multiclass.pop(key, None)

        self.use_grid_search = orig_grid_search

        # Derive binary from multiclass for both pre and post
        pre_binary = self._derive_binary_from_multiclass(pre_multiclass)

        # Compute deltas
        delta_multiclass = {}
        for key in ["accuracy", "macro_f1", "weighted_f1"]:
            if key in post_multiclass and key in pre_multiclass:
                delta_multiclass[key] = float(post_multiclass[key] - pre_multiclass[key])

        # Log comparison
        pre_f1 = pre_multiclass.get("macro_f1", 0)
        post_f1 = post_multiclass.get("macro_f1", 0)
        delta_f1 = delta_multiclass.get("macro_f1", 0)
        logger.info(f"  Pre-transformer Macro-F1={pre_f1:.4f} → Post={post_f1:.4f} (delta=+{delta_f1:.4f})")

        return {
            "pretransformer_binary": pre_binary,
            "pretransformer_multiclass": pre_multiclass,
            "delta_binary": {},  # kept for backward compat, empty
            "delta_multiclass": delta_multiclass,
        }

    # ------------------------------------------------------------------ #
    #  Embedding Quality Diagnostics (Feature 3)                          #
    # ------------------------------------------------------------------ #

    def _run_embedding_diagnostics(
        self,
        peak_data: Dict[str, np.ndarray],
        split_groups: np.ndarray,
    ) -> Dict[str, Any]:
        """Analyze embedding quality to test for implicit denoising/confidence signals.

        Computes intrinsic metrics (L2 norm, centroid distance) for all peaks
        and classifier-based metrics (confidence, entropy, margin) on held-out
        test peaks only.

        Args:
            peak_data: Peak data dict.
            split_groups: Per-peak group IDs for splitting.

        Returns:
            Dict with per-class diagnostics, annotated-vs-unannotated comparison,
            and optional plot path.
        """
        X = peak_data["embeddings"]
        binary_labels = peak_data["binary_labels"]
        multiclass_labels = peak_data["multiclass_labels"]

        # --- Intrinsic metrics (all peaks) ---
        l2_norms = np.linalg.norm(X, axis=1)

        # L2-normalize in-place to avoid a ~1.5 GB copy
        norms_safe = np.maximum(l2_norms, 1e-8)
        X_norm = X.copy()  # We need a copy since we modify in-place, but only one
        X_norm /= norms_safe[:, np.newaxis]

        # Split for centroid computation (train centroids, measure all)
        X_train, y_train, X_val, y_val, X_test, y_test, test_indices = self._split_by_spectrum(X_norm, multiclass_labels, split_groups)
        # Encode labels
        le = LabelEncoder()
        le.fit(multiclass_labels)
        y_train_enc = le.transform(y_train)
        class_names = le.classes_.tolist()
        num_classes = len(class_names)

        # Compute centroids from training set
        dim = X_norm.shape[1]
        centroids = np.zeros((num_classes, dim), dtype=np.float64)
        for ci in range(num_classes):
            mask = y_train_enc == ci
            if mask.any():
                centroids[ci] = np.mean(X_train[mask], axis=0)
                cn = np.linalg.norm(centroids[ci])
                if cn > 1e-8:
                    centroids[ci] /= cn

        # Free the split views — we only need centroids and X_norm going forward
        del X_train, X_val, X_test

        # Cosine distance to own centroid — chunked to avoid (N, D) broadcast copy
        y_all_enc = le.transform(multiclass_labels)
        n = len(y_all_enc)
        cos_sims = np.empty(n, dtype=np.float64)
        chunk_size = 50_000
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_centroids = centroids[y_all_enc[start:end]]
            cos_sims[start:end] = np.sum(X_norm[start:end] * chunk_centroids, axis=1)
            del chunk_centroids
        centroid_dists = 1.0 - cos_sims

        del X_norm  # Free normalized copy

        # --- Classifier-based metrics (optional, held-out test set only) ---
        test_metrics: Dict[str, np.ndarray] = {}
        test_idxs = np.array([], dtype=np.int64)
        y_te_raw = np.array([], dtype=object)

        if self.enable_diagnostics_classifier:
            X_tr_raw, y_tr_raw, X_v_raw, y_v_raw, X_te_raw, y_te_raw, test_idxs = self._split_by_spectrum(
                peak_data["embeddings"], multiclass_labels, split_groups
            )
            y_tr_enc = le.transform(y_tr_raw)
            y_v_enc = le.transform(y_v_raw)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_raw)
            X_v_s = scaler.transform(X_v_raw)
            X_te_s = scaler.transform(X_te_raw)

            class_counts = np.bincount(y_tr_enc, minlength=num_classes)
            cw = np.zeros(num_classes, dtype=np.float32)
            for ci in range(num_classes):
                if class_counts[ci] > 0:
                    cw[ci] = len(y_tr_enc) / (num_classes * class_counts[ci])
            class_weights = torch.from_numpy(cw).float()

            model, _ = self._train_pytorch_classifier(
                X_tr_s,
                y_tr_enc,
                X_v_s,
                y_v_enc,
                num_classes=num_classes,
                l2_reg=0.01,
                class_weights=class_weights,
            )

            model.eval()
            with torch.no_grad():
                X_te_t = torch.from_numpy(X_te_s).float().to(self.device)
                logits = model(X_te_t)
                proba = torch.softmax(logits, dim=1).cpu().numpy()

            test_metrics["confidence"] = np.max(proba, axis=1)
            test_metrics["entropy"] = -np.sum(proba * np.log(proba + 1e-10), axis=1)
            sorted_proba = np.sort(proba, axis=1)[:, ::-1]
            test_metrics["margin"] = sorted_proba[:, 0] - sorted_proba[:, 1] if proba.shape[1] > 1 else sorted_proba[:, 0]

        # --- Aggregate diagnostics ---
        all_metrics: Dict[str, np.ndarray] = {
            "l2_norm": l2_norms,
            "centroid_dist": centroid_dists,
        }

        # Per-class diagnostics
        per_class: Dict[str, Dict[str, float]] = {}
        y_te_enc = le.transform(y_te_raw) if len(y_te_raw) > 0 else np.array([])
        for ci, cname in enumerate(class_names):
            all_mask = y_all_enc == ci
            stats: Dict[str, float] = {}
            for mname, vals in all_metrics.items():
                v = vals[all_mask]
                if len(v) > 0:
                    stats[f"{mname}_mean"] = float(np.mean(v))
                    stats[f"{mname}_median"] = float(np.median(v))
                    stats[f"{mname}_std"] = float(np.std(v))
            if test_metrics and len(y_te_enc) > 0:
                test_mask = y_te_enc == ci
                for mname, vals in test_metrics.items():
                    v = vals[test_mask]
                    if len(v) > 0:
                        stats[f"{mname}_mean"] = float(np.mean(v))
                        stats[f"{mname}_median"] = float(np.median(v))
                        stats[f"{mname}_std"] = float(np.std(v))
            stats["n_peaks_all"] = int(all_mask.sum())
            per_class[cname] = stats

        # Annotated vs unannotated
        ann_vs_unann: Dict[str, Dict[str, float]] = {}
        for group_name, bmask_val in [("annotated", 1), ("unannotated", 0)]:
            all_mask = binary_labels == bmask_val
            stats = {}
            for mname, vals in all_metrics.items():
                v = vals[all_mask]
                if len(v) > 0:
                    stats[f"{mname}_mean"] = float(np.mean(v))
                    stats[f"{mname}_median"] = float(np.median(v))
            if test_metrics and len(test_idxs) > 0:
                test_binary = binary_labels[test_idxs]
                t_mask = test_binary == bmask_val
                for mname, vals in test_metrics.items():
                    v = vals[t_mask]
                    if len(v) > 0:
                        stats[f"{mname}_mean"] = float(np.mean(v))
                        stats[f"{mname}_median"] = float(np.median(v))
            stats["n_peaks_all"] = int(all_mask.sum())
            ann_vs_unann[group_name] = stats

        # --- Plot ---
        plot_path = None
        if self.create_plots and PLOTTING_AVAILABLE:
            plot_path = self._plot_embedding_diagnostics(
                per_class,
                class_names,
                all_metrics,
                test_metrics,
                multiclass_labels,
                y_all_enc,
                le.transform(y_te_raw) if len(y_te_raw) > 0 else np.array([]),
                test_idxs,
            )

        return {
            "per_class_diagnostics": per_class,
            "annotated_vs_unannotated": ann_vs_unann,
            "n_test_peaks": int(len(test_idxs)),
            "class_names": class_names,
            "plot_path": plot_path,
        }

    def _plot_embedding_diagnostics(
        self,
        per_class: Dict[str, Dict[str, float]],
        class_names: List[str],
        all_metrics: Dict[str, np.ndarray],
        test_metrics: Dict[str, np.ndarray],
        multiclass_labels: np.ndarray,
        y_all_enc: np.ndarray,
        y_test_enc: np.ndarray,
        test_idxs: np.ndarray,
    ) -> Optional[str]:
        """Create diagnostic box plots showing metric distributions by peak class."""
        save_dir = self.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        metric_configs = [
            ("l2_norm", "L2 Norm", all_metrics.get("l2_norm"), y_all_enc, len(multiclass_labels)),
            ("centroid_dist", "Cosine Dist to Centroid", all_metrics.get("centroid_dist"), y_all_enc, len(multiclass_labels)),
            ("confidence", "Max Softmax (test)", test_metrics.get("confidence"), y_test_enc, len(test_idxs)),
            ("entropy", "Prediction Entropy (test)", test_metrics.get("entropy"), y_test_enc, len(test_idxs)),
            ("margin", "Margin top1-top2 (test)", test_metrics.get("margin"), y_test_enc, len(test_idxs)),
        ]

        # Filter to metrics that have data
        metric_configs = [(k, t, v, ye, n) for k, t, v, ye, n in metric_configs if v is not None and len(v) > 0]
        if not metric_configs:
            return None

        n_metrics = len(metric_configs)
        ncols = min(3, n_metrics)
        nrows = (n_metrics + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        if n_metrics == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for ax_idx, (key, title, values, y_enc, n_total) in enumerate(metric_configs):
            ax = axes[ax_idx]
            # Group values by class
            box_data = []
            box_labels = []
            for ci, cname in enumerate(class_names):
                mask = y_enc == ci
                v = values[mask]
                if len(v) > 0:
                    # Cap at 5000 points per class for box plot
                    if len(v) > 5000:
                        v = np.random.choice(v, 5000, replace=False)
                    box_data.append(v)
                    box_labels.append(cname)

            if box_data:
                bp = ax.boxplot(box_data, vert=True, patch_artist=True, showfliers=False)
                for patch in bp["boxes"]:
                    patch.set_facecolor("#B3D9FF")
                    patch.set_alpha(0.7)
                ax.set_xticklabels(box_labels, rotation=45, ha="right", fontsize=7)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)

        # Hide unused axes
        for ax_idx in range(len(metric_configs), len(axes)):
            axes[ax_idx].set_visible(False)

        fig.suptitle("Embedding Quality Diagnostics by Peak Class", fontsize=13, fontweight="bold")
        fig.tight_layout()

        save_path = save_dir / "embedding_diagnostics.png"
        fig.savefig(save_path, dpi=self.umap_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"  Saved: {save_path}")
        return str(save_path)

    # ------------------------------------------------------------------
    # Chemistry understanding probes
    # ------------------------------------------------------------------

    def _run_chemistry_probes(
        self,
        peak_data: Dict[str, np.ndarray],
        meta: Dict[str, np.ndarray],
        split_groups: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run chemistry understanding probes on peak embeddings.

        Four probes that test whether the model learned specific chemical
        principles of peptide fragmentation:

        1. **Fragment position regression** — can we linearly predict the
           backbone cleavage position (1–30) from peak embeddings?  Tests
           mass ladder / amino acid alphabet learning.

        2. **Complementary b/y pair similarity** — are b_i and y_{n-i}
           embeddings more similar than random b/y pairs?  Tests the
           b+y = M + H₂O conservation law.

        3. **Neutral loss type discrimination** — can we distinguish H₂O
           losses from NH₃ losses?  Tests amino-acid-level chemistry.

        4. **Charge state classification** — can we decode the fragment
           charge state from peak embeddings?

        Returns:
            Dictionary with results for each probe.
        """
        results: Dict[str, Any] = {}

        X = peak_data["embeddings"]
        annotations = peak_data["matched_annotations"]
        parent_anns = peak_data["parent_annotations"]
        feature_types = peak_data["feature_types"]
        spec_indices = peak_data["spectrum_indices_per_peak"]

        # --- Probe 1: Fragment position regression ---
        try:
            results["fragment_position_regression"] = self._probe_fragment_position(X, annotations, parent_anns, feature_types, split_groups)
        except Exception as e:
            logger.warning(f"  Fragment position probe failed: {e}")
            results["fragment_position_regression"] = {"error": str(e)}

        # --- Probe 2: Complementary b/y pair similarity ---
        # L2-normalize only for this probe (avoids keeping a full copy in memory)
        try:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            X_norm = X / norms
            results["complementary_by_pairs"] = self._probe_complementary_pairs(X_norm, annotations, parent_anns, feature_types, spec_indices, meta)
            del X_norm  # Free ~1.2 GB
        except Exception as e:
            logger.warning(f"  Complementary pair probe failed: {e}")
            results["complementary_by_pairs"] = {"error": str(e)}

        # --- Probe 3: Neutral loss type discrimination ---
        try:
            mz_values = peak_data.get("mz_values")
            results["neutral_loss_discrimination"] = self._probe_neutral_loss_type(X, annotations, feature_types, split_groups, mz_values=mz_values)
        except Exception as e:
            logger.warning(f"  Neutral loss probe failed: {e}")
            results["neutral_loss_discrimination"] = {"error": str(e)}

        # --- Probe 4: Charge state classification ---
        try:
            results["charge_state_classification"] = self._probe_charge_state(X, annotations, feature_types, split_groups, mz_values=mz_values)
        except Exception as e:
            logger.warning(f"  Charge state probe failed: {e}")
            results["charge_state_classification"] = {"error": str(e)}

        return results

    def _probe_fragment_position(
        self,
        X: np.ndarray,
        annotations: np.ndarray,
        parent_anns: np.ndarray,
        feature_types: np.ndarray,
        split_groups: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Probe 1: Fragment position regression.

        Train a Ridge regression to predict the backbone cleavage position
        (1–max_fragment_position) from peak embeddings.  Only uses base
        fragment ions (b/y) to get clean position labels.
        """
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, r2_score

        positions = np.full(len(X), -1, dtype=np.int32)
        ion_types = np.empty(len(X), dtype=object)

        for i in range(len(X)):
            ft = feature_types[i]
            if ft != "base":
                continue
            ann = annotations[i]
            it = extract_ion_type(ann)
            if it not in ("b", "y"):
                continue
            pos = extract_fragment_position(ann)
            if pos < 1 or pos > self.max_fragment_position:
                continue  # Exclude out-of-range positions rather than capping
            positions[i] = pos
            ion_types[i] = it

        valid = positions > 0
        n_valid = int(valid.sum())
        if n_valid < 100:
            return {"skipped": True, "n_valid": n_valid}

        X_pos = X[valid]
        y_pos = positions[valid].astype(np.float32)

        # Split (spectrum-aware if available)
        if split_groups is not None:
            groups_pos = split_groups[valid]
            unique_groups = np.unique(groups_pos)
            np.random.seed(self.random_state)
            np.random.shuffle(unique_groups)
            n_test = max(1, int(len(unique_groups) * self.test_size))
            test_groups = set(unique_groups[:n_test].tolist())
            test_mask = np.array([g in test_groups for g in groups_pos])
            train_mask = ~test_mask
        else:
            np.random.seed(self.random_state)
            perm = np.random.permutation(len(X_pos))
            n_test = max(1, int(len(X_pos) * self.test_size))
            test_mask = np.zeros(len(X_pos), dtype=bool)
            test_mask[perm[:n_test]] = True
            train_mask = ~test_mask

        X_train, y_train = X_pos[train_mask], y_pos[train_mask]
        X_test, y_test = X_pos[test_mask], y_pos[test_mask]

        if len(X_train) < 50 or len(X_test) < 20:
            return {"skipped": True, "n_train": len(X_train), "n_test": len(X_test)}

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))

        # Per-ion-type breakdown
        ion_types_test = ion_types[valid][test_mask]
        per_ion = {}
        for it in ("b", "y"):
            mask = ion_types_test == it
            if mask.sum() >= 10:
                per_ion[it] = {
                    "r2": float(r2_score(y_test[mask], y_pred[mask])),
                    "mae": float(mean_absolute_error(y_test[mask], y_pred[mask])),
                    "n": int(mask.sum()),
                }

        logger.info(f"  Fragment position regression: R²={r2:.4f}  MAE={mae:.2f}")

        return {
            "r2": r2,
            "mae": mae,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "per_ion_type": per_ion,
        }

    def _probe_complementary_pairs(
        self,
        X_norm: np.ndarray,
        annotations: np.ndarray,
        parent_anns: np.ndarray,
        feature_types: np.ndarray,
        spec_indices: np.ndarray,
        meta: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Probe 2: Complementary b/y pair embedding similarity.

        Tests whether complementary ion pairs (b_i, y_{n-i}) from the same
        backbone cleavage site receive more similar embeddings than random
        b/y pairs from the same spectrum.

        Note: this tests embedding *clustering*, not whether the model can
        *compute* the complementary mass relationship (b_i + y_{n-i} =
        M + H₂O).  A stronger test would train a linear head to predict
        one ion's m/z from the other's embedding + precursor mass.  The
        current similarity test is a necessary but not sufficient condition
        for learning the complementary constraint.
        """
        # Get peptide sequences for determining n (peptide length)
        sequences = None
        for key in ("peptides", "sequence", "peptide", "seq"):
            if key in meta:
                sequences = meta[key]
                break
        if sequences is None:
            return {"skipped": True, "reason": "no_sequences"}

        # Group base b/y ions by spectrum
        spectrum_to_peaks: Dict[int, Dict[str, Dict[int, int]]] = defaultdict(lambda: {"b": {}, "y": {}})
        for i in range(len(X_norm)):
            ft = feature_types[i]
            if ft != "base":
                continue
            ann = annotations[i]
            it = extract_ion_type(ann)
            if it not in ("b", "y"):
                continue
            pos = extract_fragment_position(ann)
            if pos < 1:
                continue
            spec_id = int(spec_indices[i])
            spectrum_to_peaks[spec_id][it][pos] = i

        # Find complementary pairs: b_i and y_{n-i} for same cleavage site
        comp_sims = []
        random_sims = []
        rng = np.random.RandomState(self.random_state)

        for spec_id, ions in spectrum_to_peaks.items():
            b_ions = ions["b"]  # position -> global_idx
            y_ions = ions["y"]
            if not b_ions or not y_ions:
                continue

            # Determine peptide length from sequence
            if spec_id >= len(sequences):
                continue
            seq = sequences[spec_id]
            if seq is None or not isinstance(seq, str):
                continue
            # Get amino acid count using proper tokenizer that handles
            # modifications, flanking dots, and N-terminal mods correctly
            clean_seq = clean_peptide_sequence(str(seq))
            n = len(clean_seq)
            if n < 3:
                continue

            # Find complementary pairs: b_i matches y_{n-i}
            for b_pos, b_idx in b_ions.items():
                y_comp_pos = n - b_pos
                if y_comp_pos in y_ions:
                    y_idx = y_ions[y_comp_pos]
                    sim = float(np.dot(X_norm[b_idx], X_norm[y_idx]))
                    comp_sims.append(sim)

            # Random b/y pairs from same spectrum (control)
            # Sample ~3 random pairs per spectrum to build a matched control set
            b_idxs = list(b_ions.values())
            y_idxs = list(y_ions.values())
            n_random = min(len(b_idxs) * len(y_idxs), 5)
            for _ in range(n_random):
                bi = rng.choice(b_idxs)
                yi = rng.choice(y_idxs)
                random_sims.append(float(np.dot(X_norm[bi], X_norm[yi])))

        comp_sims = np.array(comp_sims, dtype=np.float64)
        random_sims = np.array(random_sims, dtype=np.float64)

        if len(comp_sims) < 10:
            return {"skipped": True, "n_pairs": len(comp_sims)}

        comp_mean = float(comp_sims.mean())
        random_mean = float(random_sims.mean()) if len(random_sims) > 0 else 0.0
        delta = comp_mean - random_mean

        # AUROC: can we distinguish complementary from random?
        auroc = None
        if len(random_sims) >= 10:
            labels = np.concatenate([np.ones(len(comp_sims)), np.zeros(len(random_sims))])
            scores = np.concatenate([comp_sims, random_sims])
            auroc = float(roc_auc_score(labels, scores))

        return {
            "complementary_mean_similarity": comp_mean,
            "random_by_mean_similarity": random_mean,
            "delta": delta,
            "auroc": auroc,
            "n_complementary_pairs": len(comp_sims),
            "n_random_pairs": len(random_sims),
        }

    def _probe_neutral_loss_type(
        self,
        X: np.ndarray,
        annotations: np.ndarray,
        feature_types: np.ndarray,
        split_groups: Optional[np.ndarray],
        mz_values: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Probe 3: Neutral loss type discrimination (H₂O vs NH₃).

        Binary classification to test if peak embeddings can distinguish
        water losses (-18.01 Da) from ammonia losses (-17.03 Da).

        Includes baselines to control for data leakage:
        - **m/z-only baseline**: classifier using only the peak's m/z value.
          High m/z AUROC means the embedding's m/z encoding alone can separate
          the classes, inflating the headline result.
        - **Same-parent-type control**: restricted to losses from the same
          parent ion series (e.g., only b-ion losses), removing the confound
          that H₂O losses predominantly come from b-ions and NH₃ from y-ions.
          This is the cleanest test of amino-acid-level chemistry.
        """
        from sklearn.linear_model import LogisticRegression

        # Parse loss type and parent ion type from annotation strings
        loss_labels = np.empty(len(X), dtype=object)
        parent_ion_types = np.empty(len(X), dtype=object)
        loss_mask = np.zeros(len(X), dtype=bool)

        for i in range(len(X)):
            if feature_types[i] != "loss":
                continue
            ann = str(annotations[i]) if annotations[i] is not None else ""
            # Determine loss type
            if "-H2O" in ann or "-H₂O" in ann:
                loss_labels[i] = "H2O"
                loss_mask[i] = True
            elif "-NH3" in ann or "-NH₃" in ann:
                loss_labels[i] = "NH3"
                loss_mask[i] = True
            else:
                continue
            # Determine parent ion type (b or y)
            it = extract_ion_type(ann)
            parent_ion_types[i] = it if it in ("b", "y") else "other"

        n_h2o = int((loss_labels == "H2O").sum())
        n_nh3 = int((loss_labels == "NH3").sum())

        if n_h2o < 50 or n_nh3 < 50:
            return {"skipped": True, "n_h2o": n_h2o, "n_nh3": n_nh3}

        X_loss = X[loss_mask]
        y_loss = loss_labels[loss_mask]
        parent_types = parent_ion_types[loss_mask]
        mz_loss = None
        if mz_values is not None:
            mz_arr = np.array(mz_values, dtype=np.float64)
            mz_loss = mz_arr[loss_mask]

        # Split
        if split_groups is not None:
            groups_loss = split_groups[loss_mask]
            unique_groups = np.unique(groups_loss)
            np.random.seed(self.random_state)
            np.random.shuffle(unique_groups)
            n_test = max(1, int(len(unique_groups) * self.test_size))
            test_groups = set(unique_groups[:n_test].tolist())
            test_mask = np.array([g in test_groups for g in groups_loss])
            train_mask = ~test_mask
        else:
            np.random.seed(self.random_state)
            perm = np.random.permutation(len(X_loss))
            n_test = max(1, int(len(X_loss) * self.test_size))
            test_mask = np.zeros(len(X_loss), dtype=bool)
            test_mask[perm[:n_test]] = True
            train_mask = ~test_mask

        le = LabelEncoder()
        y_enc = le.fit_transform(y_loss)

        X_train, y_train = X_loss[train_mask], y_enc[train_mask]
        X_test, y_test = X_loss[test_mask], y_enc[test_mask]

        if len(X_train) < 30 or len(X_test) < 10:
            return {"skipped": True}

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # --- Main classifier: full embeddings ---
        clf = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", solver="lbfgs")
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        y_proba = clf.predict_proba(X_test_s)[:, 1]

        accuracy = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="binary", zero_division=0))
        auroc = float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) == 2 else None

        class_names = le.classes_.tolist()
        logger.info(f"  Neutral loss AUROC={f'{auroc:.4f}' if auroc is not None else 'N/A'}")

        result: Dict[str, Any] = {
            "accuracy": accuracy,
            "f1": f1,
            "auroc": auroc,
            "class_names": class_names,
            "n_h2o": n_h2o,
            "n_nh3": n_nh3,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        # --- Baseline: m/z only ---
        # If m/z alone can discriminate, the embedding result may be inflated
        # by the model's m/z encoding rather than learned chemistry.
        if mz_loss is not None:
            mz_train = mz_loss[train_mask].reshape(-1, 1)
            mz_test = mz_loss[test_mask].reshape(-1, 1)
            if not np.any(np.isnan(mz_train)) and not np.any(np.isnan(mz_test)):
                scaler_mz = StandardScaler()
                mz_train_s = scaler_mz.fit_transform(mz_train)
                mz_test_s = scaler_mz.transform(mz_test)
                clf_mz = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", solver="lbfgs")
                clf_mz.fit(mz_train_s, y_train)
                mz_proba = clf_mz.predict_proba(mz_test_s)[:, 1]
                if len(np.unique(y_test)) == 2:
                    auroc_mz = float(roc_auc_score(y_test, mz_proba))
                    acc_mz = float(accuracy_score(y_test, clf_mz.predict(mz_test_s)))
                    result["mz_only_auroc"] = auroc_mz
                    result["mz_only_accuracy"] = acc_mz
                    delta_over_mz = (auroc or 0) - auroc_mz
                    result["delta_over_mz_baseline"] = delta_over_mz

        # --- Baseline: parent ion type confound ---
        # How much of the discrimination is explained by knowing the parent is
        # a b-ion vs y-ion?  H₂O losses are predominantly from b-ions, NH₃
        # from y-ions.  Quantify this confound.
        parent_test = parent_types[test_mask]
        b_mask_test = parent_test == "b"
        y_mask_test = parent_test == "y"

        # Cross-tabulation: loss type vs parent ion type
        n_b_h2o = int(((parent_types == "b") & (y_loss == "H2O")).sum())
        n_b_nh3 = int(((parent_types == "b") & (y_loss == "NH3")).sum())
        n_y_h2o = int(((parent_types == "y") & (y_loss == "H2O")).sum())
        n_y_nh3 = int(((parent_types == "y") & (y_loss == "NH3")).sum())
        result["parent_ion_crosstab"] = {
            "b_H2O": n_b_h2o,
            "b_NH3": n_b_nh3,
            "y_H2O": n_y_h2o,
            "y_NH3": n_y_nh3,
        }

        # --- Same-parent-type control ---
        # Restrict to losses from the SAME parent ion series to remove the
        # b/y confound.  If the model still discriminates, the signal is
        # genuinely from loss-type chemistry, not parent identity.
        for parent_type in ("b", "y"):
            pt_mask = parent_types == parent_type
            n_h2o_pt = int(((y_loss == "H2O") & pt_mask).sum())
            n_nh3_pt = int(((y_loss == "NH3") & pt_mask).sum())

            if n_h2o_pt < 30 or n_nh3_pt < 30:
                continue

            X_pt = X_loss[pt_mask]
            y_pt = y_enc[pt_mask]
            train_pt = train_mask[pt_mask]
            test_pt = test_mask[pt_mask]

            if train_pt.sum() < 20 or test_pt.sum() < 10:
                continue

            scaler_pt = StandardScaler()
            X_pt_train = scaler_pt.fit_transform(X_pt[train_pt])
            X_pt_test = scaler_pt.transform(X_pt[test_pt])
            y_pt_train = y_pt[train_pt]
            y_pt_test = y_pt[test_pt]

            clf_pt = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", solver="lbfgs")
            clf_pt.fit(X_pt_train, y_pt_train)
            y_pt_proba = clf_pt.predict_proba(X_pt_test)[:, 1]

            if len(np.unique(y_pt_test)) == 2:
                auroc_pt = float(roc_auc_score(y_pt_test, y_pt_proba))
                acc_pt = float(accuracy_score(y_pt_test, clf_pt.predict(X_pt_test)))
                result[f"same_parent_{parent_type}_auroc"] = auroc_pt
                result[f"same_parent_{parent_type}_accuracy"] = acc_pt
                result[f"same_parent_{parent_type}_n_test"] = int(test_pt.sum())

        return result

    def _probe_charge_state(
        self,
        X: np.ndarray,
        annotations: np.ndarray,
        feature_types: np.ndarray,
        split_groups: Optional[np.ndarray],
        mz_values: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Probe 4: Charge state classification.

        Classify fragment charge state (z=1 vs z≥2) from peak embeddings.
        Only uses base b/y ions for clean charge labels.

        Includes an m/z-only baseline since charge correlates strongly with
        m/z (higher m/z fragments are more likely singly charged).
        """
        charges = np.full(len(X), -1, dtype=np.int32)

        for i in range(len(X)):
            if feature_types[i] != "base":
                continue
            ann = annotations[i]
            it = extract_ion_type(ann)
            if it not in ("b", "y"):
                continue
            z = extract_charge_from_annotation(ann)
            charges[i] = z

        valid = charges > 0
        # Binary: z=1 vs z>=2
        charge_labels = np.where(charges == 1, "z1", "z2+")
        charge_labels[~valid] = ""

        n_z1 = int((charge_labels == "z1").sum())
        n_z2 = int((charge_labels == "z2+").sum())

        # Prepare m/z for baseline
        mz_chrg = None
        if mz_values is not None:
            mz_arr = np.array(mz_values, dtype=np.float64)
            mz_chrg = mz_arr[valid]

        if n_z1 < 50 or n_z2 < 50:
            return {"skipped": True, "n_z1": n_z1, "n_z2": n_z2}

        X_chrg = X[valid]
        y_chrg = charge_labels[valid]

        # Split
        if split_groups is not None:
            groups_chrg = split_groups[valid]
            unique_groups = np.unique(groups_chrg)
            np.random.seed(self.random_state)
            np.random.shuffle(unique_groups)
            n_test = max(1, int(len(unique_groups) * self.test_size))
            test_groups = set(unique_groups[:n_test].tolist())
            test_mask = np.array([g in test_groups for g in groups_chrg])
            train_mask = ~test_mask
        else:
            np.random.seed(self.random_state)
            perm = np.random.permutation(len(X_chrg))
            n_test = max(1, int(len(X_chrg) * self.test_size))
            test_mask = np.zeros(len(X_chrg), dtype=bool)
            test_mask[perm[:n_test]] = True
            train_mask = ~test_mask

        le = LabelEncoder()
        y_enc = le.fit_transform(y_chrg)

        X_train, y_train = X_chrg[train_mask], y_enc[train_mask]
        X_test, y_test = X_chrg[test_mask], y_enc[test_mask]

        if len(X_train) < 30 or len(X_test) < 10:
            return {"skipped": True}

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", solver="lbfgs")
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        y_proba = clf.predict_proba(X_test_s)

        accuracy = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        auroc = None
        if len(np.unique(y_test)) == 2:
            auroc = float(roc_auc_score(y_test, y_proba[:, 1]))

        class_names = le.classes_.tolist()
        logger.info(f"  Charge state AUROC={f'{auroc:.4f}' if auroc is not None else 'N/A'}")

        result = {
            "accuracy": accuracy,
            "f1": f1,
            "auroc": auroc,
            "class_names": class_names,
            "n_z1": n_z1,
            "n_z2_plus": n_z2,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        # --- Baseline: m/z only ---
        # Charge correlates strongly with m/z (higher m/z → more likely z=1).
        # Quantify how much of the charge discrimination is from m/z position.
        if mz_chrg is not None:
            mz_train_c = mz_chrg[train_mask].reshape(-1, 1)
            mz_test_c = mz_chrg[test_mask].reshape(-1, 1)
            if not np.any(np.isnan(mz_train_c)) and not np.any(np.isnan(mz_test_c)):
                from sklearn.linear_model import LogisticRegression as LR

                scaler_mz = StandardScaler()
                mz_train_cs = scaler_mz.fit_transform(mz_train_c)
                mz_test_cs = scaler_mz.transform(mz_test_c)
                clf_mz = LR(C=1.0, max_iter=500, class_weight="balanced", solver="lbfgs")
                clf_mz.fit(mz_train_cs, y_train)
                mz_proba_c = clf_mz.predict_proba(mz_test_cs)
                if len(np.unique(y_test)) == 2:
                    auroc_mz = float(roc_auc_score(y_test, mz_proba_c[:, 1]))
                    delta_over_mz = (auroc or 0) - auroc_mz
                    result["mz_only_auroc"] = auroc_mz
                    result["delta_over_mz_baseline"] = delta_over_mz

        return result

    def _create_chemistry_probes_figure(
        self,
        chemistry_results: Dict[str, Any],
    ) -> Optional[str]:
        """Create a summary figure for chemistry understanding probes.

        Layout: 2x2 grid
        - Top-left: Fragment position regression (per-ion R² bars)
        - Top-right: Neutral loss discrimination (H₂O vs NH₃ metrics)
        - Bottom-left: Charge state classification (z=1 vs z≥2 metrics)
        - Bottom-right: Summary scorecard of all probes
        """
        if not PLOTTING_AVAILABLE:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Chemistry Understanding Probes", fontsize=14, fontweight="bold")

        # --- Top-left: Fragment position regression ---
        ax = axes[0, 0]
        fpr = chemistry_results.get("fragment_position_regression", {})
        if fpr and not fpr.get("skipped") and not fpr.get("error"):
            per_ion = fpr.get("per_ion_type", {})
            labels = ["Overall"]
            r2_vals = [fpr["r2"]]
            mae_vals = [fpr["mae"]]
            colors = ["#5C6BC0"]
            for it, c in [("b", "#2196F3"), ("y", "#FF7043")]:
                if it in per_ion:
                    labels.append(f"{it}-ions")
                    r2_vals.append(per_ion[it]["r2"])
                    mae_vals.append(per_ion[it]["mae"])
                    colors.append(c)

            x = np.arange(len(labels))
            bars = ax.bar(x, r2_vals, color=colors, alpha=0.85, width=0.6)
            ax.set_ylabel("R²", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.set_title("Fragment Position Regression", fontweight="bold", fontsize=11)
            ax.grid(True, alpha=0.3, axis="y")

            # Add MAE labels on bars
            for bar, mae in zip(bars, mae_vals, strict=False):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"MAE={mae:.1f}", ha="center", va="bottom", fontsize=9)
        else:
            ax.text(0.5, 0.5, "Skipped / Error", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_title("Fragment Position Regression", fontweight="bold", fontsize=11)

        # --- Top-right: Neutral loss discrimination ---
        ax = axes[0, 1]
        nld = chemistry_results.get("neutral_loss_discrimination", {})
        if nld and not nld.get("skipped") and not nld.get("error"):
            metrics = ["Accuracy", "F1", "AUROC"]
            values = [nld["accuracy"], nld["f1"], nld.get("auroc", 0)]
            colors_nl = ["#66BB6A", "#43A047", "#2E7D32"]
            x = np.arange(len(metrics))
            bars = ax.bar(x, values, color=colors_nl, alpha=0.85, width=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score", fontsize=10)
            ax.set_title("Neutral Loss Type: H₂O vs NH₃", fontweight="bold", fontsize=11)
            ax.grid(True, alpha=0.3, axis="y")

            for bar, val in zip(bars, values, strict=False):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

            ax.text(
                0.98,
                0.02,
                f"n(H₂O)={nld['n_h2o']:,}  n(NH₃)={nld['n_nh3']:,}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="gray",
            )
        else:
            ax.text(0.5, 0.5, "Skipped / Error", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_title("Neutral Loss Type: H₂O vs NH₃", fontweight="bold", fontsize=11)

        # --- Bottom-left: Charge state classification ---
        ax = axes[1, 0]
        csc = chemistry_results.get("charge_state_classification", {})
        if csc and not csc.get("skipped") and not csc.get("error"):
            metrics = ["Accuracy", "F1", "AUROC"]
            values = [csc["accuracy"], csc["f1"], csc.get("auroc", 0)]
            colors_cs = ["#42A5F5", "#1E88E5", "#1565C0"]
            x = np.arange(len(metrics))
            bars = ax.bar(x, values, color=colors_cs, alpha=0.85, width=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score", fontsize=10)
            ax.set_title("Charge State: z=1 vs z≥2", fontweight="bold", fontsize=11)
            ax.grid(True, alpha=0.3, axis="y")

            for bar, val in zip(bars, values, strict=False):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

            ax.text(
                0.98,
                0.02,
                f"n(z=1)={csc['n_z1']:,}  n(z≥2)={csc['n_z2_plus']:,}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="gray",
            )
        else:
            ax.text(0.5, 0.5, "Skipped / Error", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_title("Charge State: z=1 vs z≥2", fontweight="bold", fontsize=11)

        # --- Bottom-right: Summary scorecard ---
        ax = axes[1, 1]
        ax.axis("off")

        rows = []
        # Fragment position
        if fpr and not fpr.get("skipped") and not fpr.get("error"):
            rows.append(("Fragment position (R²)", f"{fpr['r2']:.3f}", "Mass ladder"))
        else:
            rows.append(("Fragment position (R²)", "—", "Mass ladder"))
        # Complementary pairs
        comp = chemistry_results.get("complementary_by_pairs", {})
        if comp and not comp.get("skipped") and not comp.get("error"):
            auroc_val = comp.get("auroc")
            rows.append(("Complementary b/y (AUROC)", f"{auroc_val:.3f}" if auroc_val else "—", "b+y=M+H₂O"))
        else:
            rows.append(("Complementary b/y (AUROC)", "—", "b+y=M+H₂O"))
        # Neutral loss
        if nld and not nld.get("skipped") and not nld.get("error"):
            rows.append(("Loss type H₂O/NH₃ (AUROC)", f"{nld.get('auroc', 0):.3f}", "AA chemistry"))
        else:
            rows.append(("Loss type H₂O/NH₃ (AUROC)", "—", "AA chemistry"))
        # Charge state
        if csc and not csc.get("skipped") and not csc.get("error"):
            rows.append(("Charge state z1/z2+ (AUROC)", f"{csc.get('auroc', 0):.3f}", "Physical param"))
        else:
            rows.append(("Charge state z1/z2+ (AUROC)", "—", "Physical param"))

        table = ax.table(
            cellText=[[r[0], r[1], r[2]] for r in rows],
            colLabels=["Probe", "Score", "Tests"],
            cellLoc="center",
            loc="center",
            colWidths=[0.45, 0.2, 0.35],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.8)

        # Style header row
        for j in range(3):
            table[0, j].set_facecolor("#E0E0E0")
            table[0, j].set_text_props(fontweight="bold")

        # Color-code scores
        for i, row in enumerate(rows, start=1):
            score_str = row[1]
            if score_str != "—":
                score = float(score_str)
                if score >= 0.8:
                    table[i, 1].set_facecolor("#C8E6C9")
                elif score >= 0.5:
                    table[i, 1].set_facecolor("#FFF9C4")
                else:
                    table[i, 1].set_facecolor("#FFCDD2")

        ax.set_title("Chemistry Probes Summary", fontweight="bold", fontsize=11)

        plt.tight_layout()
        save_path = self.output_dir / "chemistry_probes.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return str(save_path)

    def get_loggable_metrics(self, task_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract 6 core metrics for MLflow logging and ablation comparison.

        - multiclass_macro_f1: Peak type discrimination (headline)
        - cross_spectrum_auroc: Ion identity consistency across spectra
        - isotope_parent_mean: Structural awareness (isotope proximity)
        - parent_recovery_top1: Nearest-neighbor parent retrieval
        - pretransformer_r2: How much input encoding explains (architecture)
        - umap_mz_rho_residual: Residual m/z signal after correction
        """
        if "error" in task_results:
            return {}

        metrics = {}

        # 1. Peak type classification
        mc = task_results.get("multiclass_classification", {})
        if mc and "macro_f1" in mc:
            metrics["multiclass_macro_f1"] = mc["macro_f1"]

        # 2. Cross-spectrum ion identity AUROC
        csi = task_results.get("cross_spectrum_identity", {})
        if csi:
            auroc = csi.get("auroc_same_vs_different_mz")
            if auroc is not None:
                metrics["cross_spectrum_auroc"] = auroc

        # 3. Isotope-parent similarity
        sc = task_results.get("structural_consistency", {})
        if sc:
            iso = sc.get("isotope_parent", {})
            if isinstance(iso, dict) and iso.get("mean") is not None:
                metrics["isotope_parent_mean"] = iso["mean"]
            recovery = sc.get("parent_recovery", {})
            if recovery and recovery.get("top1") is not None:
                metrics["parent_recovery_top1"] = recovery["top1"]

        # 4-5. UMAP diagnostics (R² and residual m/z correlation)
        ud = task_results.get("umap_diagnostics", {})
        if ud:
            if "pretransformer_r2" in ud:
                metrics["pretransformer_r2"] = ud["pretransformer_r2"]
            if "umap_mz_rho_residual" in ud:
                metrics["umap_mz_rho_residual"] = ud["umap_mz_rho_residual"]

        return metrics
