"""UMAP Visualisation evaluation task.

Projects embeddings to 2D using UMAP and creates multiple PNG visualizations
colored by different metadata fields (metadata is extracted upstream in data processor).
Computes quantitative projection quality metrics (trustworthiness, kNN preservation).
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

from instanovo_fm.eval.embed_eval_tasks import BaseTask

logger = logging.getLogger(__name__)

# Fields for the summary panel (subset of visualization_configs)
_SUMMARY_PANEL_FIELDS = [
    "precursor_charge",
    "precursor_mz",
    "collision_energy",
    "hydrophobicity",
    "sequence_length",
    "n_peaks",
    "annotation_ratio",
    "backbone_coverage",
    "modification_types",
]


class UMAPVisualisationTask(BaseTask):
    """UMAP visualisation evaluation task.

    Projects embeddings to 2D using UMAP and creates PNG visualizations
    colored by various metadata fields (extracted upstream in data processor).
    Optionally computes trustworthiness and kNN preservation metrics.
    """

    name = "UMAP Visualisation"
    description = "Project embeddings to 2D with UMAP and create visualizations colored by metadata fields."
    requires_metadata = True
    requires_faiss = False

    def __init__(self, **kwargs: Any) -> None:
        """Initialise the input."""
        super().__init__(**kwargs)
        # Sampling and output
        self.max_samples: int = kwargs.get("max_samples", 10000)
        self.ckpt_step = kwargs.get("ckpt_step", "unknown")
        self.save_dir: Optional[str] = kwargs.get("save_dir", None)
        self.output_dir: Optional[str] = kwargs.get("output_dir", None)
        self.random_state: int = kwargs.get("random_state", 42)

        # Core UMAP parameters
        self.n_neighbors: int = kwargs.get("n_neighbors", 40)
        self.min_dist: float = kwargs.get("min_dist", 0.15)
        self.metric: str = kwargs.get("metric", "cosine")

        # Advanced UMAP parameters
        self.n_components: int = kwargs.get("n_components", 2)
        self.spread: float = kwargs.get("spread", 1.0)
        self.local_connectivity: float = kwargs.get("local_connectivity", 1.0)
        self.repulsion_strength: float = kwargs.get("repulsion_strength", 1.0)
        self.negative_sample_rate: int = kwargs.get("negative_sample_rate", 5)
        self.transform_queue_size: float = kwargs.get("transform_queue_size", 4.0)
        self.verbose: bool = kwargs.get("verbose", False)

        # Visualization parameters
        self.create_multiple_plots: bool = kwargs.get("create_multiple_plots", True)
        plot_size_raw = kwargs.get("plot_size", (10, 8))
        self.plot_size: Tuple[int, int] = tuple(plot_size_raw) if isinstance(plot_size_raw, (list, tuple)) else (10, 8)
        self.dpi: int = kwargs.get("dpi", 200)
        self.alpha: float = kwargs.get("alpha", 0.7)
        self.point_size: int = kwargs.get("point_size", 8)

        # Legend management
        self.max_categories: int = kwargs.get("max_categories", 15)
        self.collapse_rare_categories: bool = kwargs.get("collapse_rare_categories", True)
        self.modification_max_categories: int = kwargs.get("modification_max_categories", 10)
        self.modification_collapse_rare: bool = kwargs.get("modification_collapse_rare", True)

        # Quality metrics
        self.compute_quality_metrics: bool = kwargs.get("compute_quality_metrics", True)

        # Summary panel
        self.create_summary_panel: bool = kwargs.get("create_summary_panel", True)

        # Instrument-conditional UMAPs — filter by metadata and run separate UMAPs
        # Uses the same pattern as PeakTypeClassificationTask
        self.enable_conditional_umaps: bool = kwargs.get("enable_conditional_umaps", False)
        self.conditional_min_samples: int = kwargs.get("conditional_min_samples", 500)
        self.conditional_subsets: List[Dict[str, Any]] = kwargs.get(
            "conditional_subsets",
            [
                {
                    "name": "hcd_orbitrap",
                    "frag_types": ["HCD", "HCID"],
                    "detectors": ["Orbitrap"],
                    "instruments": None,
                },
            ],
        )

        # Visualization configurations using metadata extracted upstream
        self.visualization_configs: List[Dict[str, Any]] = kwargs.get(
            "visualization_configs",
            [
                # Core spectral metadata
                {"name": "precursor_charge", "key": "precursor_charge", "title": "Precursor Charge", "cmap": "viridis"},
                {"name": "spectrum_confidence", "key": "spectrum_confidence", "title": "Spectrum Confidence (Model)", "cmap": "RdYlGn"},
                {"name": "sequence_length", "key": "sequence_length", "title": "Peptide Sequence Length", "cmap": "viridis"},
                {"name": "hydrophobicity", "key": "hydrophobicity", "title": "Peptide Hydrophobicity", "cmap": "RdBu_r"},
                {"name": "retention_time", "key": "retention_time", "title": "Retention Time (s)", "cmap": "plasma"},
                {"name": "collision_energy", "key": "collision_energy", "title": "Collision Energy (V)", "cmap": "inferno", "vmin": 20, "vmax": 40},
                {"name": "precursor_mz", "key": "precursor_mz", "title": "Precursor m/z", "cmap": "magma"},
                {"name": "n_peaks", "key": "n_peaks", "title": "Number of Peaks", "cmap": "viridis"},
                {"name": "peak_center_of_mass", "key": "peak_center_of_mass", "title": "Peak Center of Mass (m/z)", "cmap": "magma"},
                {"name": "peak_spread", "key": "peak_spread", "title": "Peak Spread (std m/z)", "cmap": "plasma"},
                {"name": "hyperscore", "key": "hyperscore", "title": "Hyperscore", "cmap": "YlOrRd"},
                # PTM analysis
                {"name": "modification_types", "key": "modification_types", "title": "Modification Types", "cmap": "tab10"},
                {"name": "ptm_present", "key": "modification_types", "title": "PTM Presence", "cmap": "Set1"},
                # Categorical
                {"name": "frag_type", "key": "frag_type", "title": "Fragmentation Type", "cmap": "tab10"},
                # Search metadata
                {"name": "search_acquisition", "key": "search_acquisition", "title": "Search Acquisition", "cmap": "Set1"},
                {"name": "search_instrument", "key": "search_instrument", "title": "Search Instrument", "cmap": "tab20"},
                {"name": "search_detector", "key": "search_detector", "title": "Search Detector", "cmap": "tab20"},
                {"name": "search_organism", "key": "search_organism", "title": "Search Organism", "cmap": "Set2"},
                {"name": "search_enzyme", "key": "search_enzyme", "title": "Search Enzyme", "cmap": "tab10"},
                {"name": "search_project", "key": "search_project", "title": "Search Project", "cmap": "tab20"},
                {"name": "search_quant", "key": "search_quant", "title": "Search Quant", "cmap": "Set2"},
                {"name": "search_modifications", "key": "search_modifications", "title": "Search Modifications", "cmap": "Set1"},
                # Sequence clustering
                {"name": "seq_cluster_id", "key": "seq_cluster_id", "title": "Sequence Similarity Clusters", "cmap": "tab10"},
                # Duplicate peptide analysis
                {"name": "top_duplicate_peptides", "key": "top_duplicate_peptides", "title": "Top 10 Duplicate Peptides", "cmap": "tab10"},
                # Theoretical spectrum annotation metrics (requires theoretical_spectrum.enabled=True)
                {"name": "annotation_ratio", "key": "annotation_ratio", "title": "Annotation Ratio (matched/total peaks)", "cmap": "YlOrRd"},
                {"name": "backbone_coverage", "key": "backbone_coverage", "title": "Backbone Coverage", "cmap": "YlGn"},
                {"name": "signal_intensity_ratio", "key": "signal_intensity_ratio", "title": "Signal Intensity Ratio", "cmap": "RdYlGn"},
                {"name": "n_fragment_groups_metric", "key": "n_fragment_groups_metric", "title": "Number of Fragment Groups", "cmap": "viridis"},
                {"name": "median_ppm_error", "key": "median_ppm_error", "title": "Median Mass Error (ppm)", "cmap": "RdYlGn_r"},
            ],
        )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------

    def _create_unique_output_directory(self) -> Path:
        """Create and return the output directory for figures."""
        if self.save_dir:
            base_dir = Path(self.save_dir)
        elif self.output_dir:
            base_dir = Path(self.output_dir) / "figures"
        else:
            base_dir = Path("evaluation_results/umap_figs")
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    # ------------------------------------------------------------------
    # Legend management
    # ------------------------------------------------------------------

    def _manage_categorical_legend(
        self,
        unique_values: np.ndarray,
        counts: np.ndarray,
        max_categories: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Collapse rare categories and limit legend size.

        Returns (filtered_values, filtered_counts, labels).
        """
        max_cat = int(max_categories) if max_categories is not None else self.max_categories

        # Sort by count descending
        order = np.argsort(counts)[::-1]
        sorted_values = unique_values[order]
        sorted_counts = counts[order]

        if len(sorted_values) <= max_cat:
            labels = [f"{v} (n={c})" for v, c in zip(sorted_values, sorted_counts, strict=False)]
            return sorted_values, sorted_counts, labels

        if self.collapse_rare_categories:
            top_values = sorted_values[: max_cat - 1]
            top_counts = sorted_counts[: max_cat - 1]
            other_count = int(np.sum(sorted_counts[max_cat - 1 :]))
            other_values = sorted_values[max_cat - 1 :]

            filtered_values = np.concatenate([top_values, ["Other"]])
            filtered_counts = np.concatenate([top_counts, [other_count]])

            labels = [f"{v} (n={c})" for v, c in zip(top_values, top_counts, strict=False)]
            preview = ", ".join(str(v) for v in other_values[:3])
            suffix = "..." if len(other_values) > 3 else ""
            labels.append(f"Other ({preview}{suffix}) (n={other_count})")
        else:
            filtered_values = sorted_values[:max_cat]
            filtered_counts = sorted_counts[:max_cat]
            labels = [f"{v} (n={c})" for v, c in zip(filtered_values, filtered_counts, strict=False)]

        return filtered_values, filtered_counts, labels

    # ------------------------------------------------------------------
    # Spectral property computation (vectorized)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_spectral_properties(
        meta: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Compute n_peaks, peak_center_of_mass, peak_spread from spectra metadata.

        Operates on *already-subsampled* data.  Mutates ``meta`` in-place and
        returns a dict of the newly added keys for convenience.
        """
        added: Dict[str, np.ndarray] = {}

        # Sequence length from unmodified_peptide
        if "unmodified_peptide" in meta:
            try:
                seqs = meta["unmodified_peptide"]
                seq_lens = np.array(
                    [len(str(s).strip()) if isinstance(s, str) and s else 0 for s in seqs],
                    dtype=np.int32,
                )
                meta["sequence_length"] = seq_lens
                added["sequence_length"] = seq_lens
                logger.debug(
                    "Computed sequence lengths: min=%d, max=%d, mean=%.1f",
                    seq_lens.min(),
                    seq_lens.max(),
                    seq_lens.mean(),
                )
            except Exception as e:
                logger.warning("Failed to compute sequence length: %s", e)

        # Spectral properties from raw spectra + mask
        if "spectra" in meta and "spectra_mask" in meta:
            try:
                spectra_raw = meta["spectra"]  # (N, L, 2)
                spectra_mask = meta["spectra_mask"]  # (N, L) True=padded
                max_mz = float(meta.get("theoretical_max_mz", 2500.0))

                valid = ~spectra_mask  # True = real peak
                n_peaks = valid.sum(axis=1).astype(np.int32)

                mz = spectra_raw[:, :, 0] * max_mz
                intensities = spectra_raw[:, :, 1]

                mz_masked = np.where(valid, mz, 0.0)
                int_masked = np.where(valid, intensities, 0.0)

                total_int = int_masked.sum(axis=1)
                has_int = total_int > 0
                safe_n = np.maximum(n_peaks, 1).astype(np.float64)

                # Center of mass (intensity-weighted mean m/z)
                weighted_sum = (mz_masked * int_masked).sum(axis=1)
                simple_mean = mz_masked.sum(axis=1) / safe_n
                com = np.where(has_int, weighted_sum / np.maximum(total_int, 1e-12), simple_mean)

                # Peak spread (std of m/z)
                sq_diff = np.where(valid, (mz - simple_mean[:, None]) ** 2, 0.0)
                spread = np.sqrt(sq_diff.sum(axis=1) / np.maximum(safe_n - 1, 1))

                no_peaks = n_peaks == 0
                com[no_peaks] = np.nan
                spread[no_peaks] = np.nan

                meta["n_peaks"] = n_peaks
                meta["peak_center_of_mass"] = com.astype(np.float32)
                meta["peak_spread"] = spread.astype(np.float32)
                added.update({"n_peaks": n_peaks, "peak_center_of_mass": com, "peak_spread": spread})

                logger.debug(
                    "Spectral properties: n_peaks [%d-%d], CoM [%.1f-%.1f], spread [%.1f-%.1f]",
                    n_peaks.min(),
                    n_peaks.max(),
                    np.nanmin(com),
                    np.nanmax(com),
                    np.nanmin(spread),
                    np.nanmax(spread),
                )
            except Exception as e:
                logger.warning("Failed to compute spectral properties: %s", e)

        elif "spectra_mask" in meta:
            # Fallback: n_peaks from mask alone
            try:
                n_peaks = (~meta["spectra_mask"]).sum(axis=1).astype(np.int32)
                meta["n_peaks"] = n_peaks
                added["n_peaks"] = n_peaks
                logger.debug("n_peaks from mask: [%d-%d]", n_peaks.min(), n_peaks.max())
            except Exception as e:
                logger.warning("Failed to compute n_peaks from spectra_mask: %s", e)

        return added

    # ------------------------------------------------------------------
    # Annotation property computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_annotation_properties(
        meta: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Derive per-spectrum annotation metrics from theoretical matching data.

        Requires ``match_metrics`` and ``spectrum_quality`` in metadata
        (populated when ``theoretical_spectrum.enabled=True``).
        """
        added: Dict[str, np.ndarray] = {}

        if "match_metrics" not in meta or "spectrum_quality" not in meta:
            return added

        match_metrics = meta["match_metrics"]  # object array of dicts
        spectrum_quality = meta["spectrum_quality"]  # object array of dicts
        N = len(match_metrics)  # noqa: N806

        annotation_ratio = np.full(N, np.nan, dtype=np.float32)
        backbone_coverage = np.full(N, np.nan, dtype=np.float32)
        signal_intensity_ratio = np.full(N, np.nan, dtype=np.float32)
        n_fragment_groups = np.full(N, np.nan, dtype=np.float32)
        median_ppm_error = np.full(N, np.nan, dtype=np.float32)

        n_peaks = meta.get("n_peaks")

        for i in range(N):
            mm = match_metrics[i]
            sq = spectrum_quality[i]
            if isinstance(mm, dict):
                n_matched = mm.get("n_matched", 0)
                if n_peaks is not None and n_peaks[i] > 0:
                    annotation_ratio[i] = n_matched / float(n_peaks[i])
                frac = mm.get("frac_intensity")
                if frac is not None:
                    signal_intensity_ratio[i] = float(frac)
                ppm = mm.get("median_abs_ppm")
                if ppm is not None:
                    median_ppm_error[i] = float(ppm)
            if isinstance(sq, dict):
                bc = sq.get("backbone_coverage")
                if bc is not None:
                    backbone_coverage[i] = float(bc)
                fg = sq.get("n_fragment_groups")
                if fg is not None:
                    n_fragment_groups[i] = float(fg)

        meta["annotation_ratio"] = annotation_ratio
        meta["backbone_coverage"] = backbone_coverage
        meta["signal_intensity_ratio"] = signal_intensity_ratio
        meta["n_fragment_groups_metric"] = n_fragment_groups
        meta["median_ppm_error"] = median_ppm_error

        added: dict[str, Any] = {
            "annotation_ratio": annotation_ratio,
            "backbone_coverage": backbone_coverage,
            "signal_intensity_ratio": signal_intensity_ratio,
            "n_fragment_groups_metric": n_fragment_groups,
            "median_ppm_error": median_ppm_error,
        }

        n_valid = int(np.isfinite(annotation_ratio).sum())
        if n_valid > 0:
            logger.debug(
                "Annotation properties: %d/%d spectra, annotation_ratio [%.2f-%.2f]",
                n_valid,
                N,
                np.nanmin(annotation_ratio),
                np.nanmax(annotation_ratio),
            )
        else:
            logger.debug("No annotation data available")

        return added

    # ------------------------------------------------------------------
    # Duplicate peptide coloring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_top_duplicate_peptides(
        meta: Dict[str, np.ndarray],
        top_n: int = 10,
    ) -> Dict[str, np.ndarray]:
        """Identify the top-N most duplicated peptide sequences and create a categorical label.

        Each spectrum belonging to a top-N duplicate group gets a label like
        ``"Peptide 1 (ACDEF..., n=42)"``.  Everything else is labeled ``"Other"``.
        """
        added: Dict[str, np.ndarray] = {}

        # Try several possible sequence keys
        seq_key = None
        for candidate in ("peptides", "unmodified_peptide", "sequence"):
            if candidate in meta:
                seq_key = candidate
                break
        if seq_key is None:
            return added

        seqs = meta[seq_key]
        N = len(seqs)  # noqa: N806

        # Count occurrences
        from collections import Counter

        seq_counts: Counter = Counter()
        for s in seqs:
            s_str = str(s).strip()
            if s_str:
                seq_counts[s_str] += 1

        # Keep only duplicates (count >= 2), sorted by count descending
        duplicates = [(seq, cnt) for seq, cnt in seq_counts.most_common() if cnt >= 2]
        if not duplicates:
            logger.debug("No duplicate peptide sequences found for UMAP coloring")
            return added

        top_dups = duplicates[:top_n]

        # Build label array
        seq_to_label: Dict[str, int] = {}
        labels_list: list = []
        for rank, (seq, cnt) in enumerate(top_dups):
            display = seq if len(seq) <= 20 else f"{seq[:17]}..."
            labels_list.append(f"Peptide {rank + 1}: {display} (n={cnt})")
            seq_to_label[seq] = rank

        other_idx = len(labels_list)
        labels_list.append("Other")

        # Map each spectrum to its label index
        label_arr = np.full(N, other_idx, dtype=np.int32)
        for i, s in enumerate(seqs):
            s_str = str(s).strip()
            if s_str in seq_to_label:
                label_arr[i] = seq_to_label[s_str]

        meta["top_duplicate_peptides"] = label_arr
        meta["_top_duplicate_labels"] = np.array(labels_list, dtype=object)
        added["top_duplicate_peptides"] = label_arr

        n_in_top = int((label_arr != other_idx).sum())
        logger.debug(
            "Top %d duplicate peptides cover %d/%d spectra (%.1f%%)",
            len(top_dups),
            n_in_top,
            N,
            100.0 * n_in_top / N,
        )

        return added

    # ------------------------------------------------------------------
    # Subsampling
    # ------------------------------------------------------------------

    @staticmethod
    def _subsample(
        emb: np.ndarray,
        meta: Dict[str, Any],
        max_samples: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Subsample embeddings and metadata to *max_samples*."""
        if len(emb) <= max_samples:
            return emb, meta

        indices = rng.choice(len(emb), max_samples, replace=False)
        emb_sampled = emb[indices]

        meta_sampled: Dict[str, Any] = {}
        for k, v in meta.items():
            if not hasattr(v, "__getitem__") or not hasattr(v, "__len__"):
                continue
            if len(v) != len(emb):
                continue
            if isinstance(v, np.ndarray):
                meta_sampled[k] = v[indices]
            elif isinstance(v, list):
                meta_sampled[k] = [v[i] for i in indices]
            else:
                try:
                    meta_sampled[k] = np.asarray(v)[indices]
                except Exception:
                    continue

        return emb_sampled, meta_sampled

    # ------------------------------------------------------------------
    # UMAP quality metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_quality_metrics(
        emb_hd: np.ndarray,
        emb_2d: np.ndarray,
        k_values: Tuple[int, ...] = (5, 15),
    ) -> Dict[str, float]:
        """Compute trustworthiness and kNN preservation for the UMAP projection."""
        from sklearn.manifold import trustworthiness
        from sklearn.neighbors import NearestNeighbors

        metrics: Dict[str, float] = {}

        for k in k_values:
            if k >= len(emb_hd):
                continue
            tw = trustworthiness(emb_hd, emb_2d, n_neighbors=k, metric="cosine")
            metrics[f"trustworthiness_k{k}"] = float(tw)

        # kNN preservation at k=15 (or largest feasible k)
        k_knn = min(15, len(emb_hd) - 1)
        if k_knn >= 1:
            nn_hd = NearestNeighbors(n_neighbors=k_knn + 1, metric="cosine").fit(emb_hd)
            nn_2d = NearestNeighbors(n_neighbors=k_knn + 1, metric="euclidean").fit(emb_2d)
            idx_hd = nn_hd.kneighbors(emb_hd, return_distance=False)[:, 1:]
            idx_2d = nn_2d.kneighbors(emb_2d, return_distance=False)[:, 1:]
            overlaps = np.array([len(set(a) & set(b)) / k_knn for a, b in zip(idx_hd, idx_2d, strict=False)])
            metrics[f"knn_preservation_k{k_knn}"] = float(overlaps.mean())

        return metrics

    # ------------------------------------------------------------------
    # Coloring data extraction
    # ------------------------------------------------------------------

    def _get_coloring_data(
        self,
        meta: Dict[str, np.ndarray],
        key: str,
        name: str,
    ) -> Tuple[Optional[np.ndarray], Optional[str], bool, Optional[List[str]], Set[int]]:
        """Get coloring data for a metadata key.

        Returns:
            (numeric_data, label, is_categorical, category_labels, background_indices)

            ``background_indices`` is the set of label indices that should be
            rendered as a gray background layer (e.g. "Unmodified", "Other").
        """
        if key not in meta:
            return None, None, False, None, set()

        data = meta[key]

        # Unwrap nested object arrays
        if isinstance(data, np.ndarray) and data.dtype == object and len(data) > 0:
            if isinstance(data[0], np.ndarray):
                try:
                    data = np.array([item.item() if isinstance(item, np.ndarray) and item.size == 1 else item for item in data])
                except Exception:
                    return None, None, False, None, set()

        data = np.asarray(data)

        # Must be 1-D
        if data.ndim > 1:
            if key in ("spectra_mask", "dmz_labels", "intensity_labels", "precursors"):
                return None, None, False, None, set()
            if data.shape[1] == 1:
                data = data.flatten()
            else:
                return None, None, False, None, set()

        if len(data) == 0:
            return None, None, False, None, set()

        # ---- Special handlers ----

        if name == "sequence_length":
            if np.issubdtype(data.dtype, np.number):
                return data.astype(float), "Sequence Length (amino acids)", False, None, set()
            return None, None, False, None, set()

        if name == "hydrophobicity":
            if np.issubdtype(data.dtype, np.number):
                return data.astype(float), "Hydrophobicity (Kyte-Doolittle)", False, None, set()
            return None, None, False, None, set()

        if name == "modification_types":
            mod_types = np.asarray(data)
            unique_mods, counts = np.unique(mod_types, return_counts=True)
            if self.modification_collapse_rare:
                filt_mods, _, labels = self._manage_categorical_legend(unique_mods, counts, max_categories=self.modification_max_categories)
            else:
                filt_mods = unique_mods
                labels = [f"{v} (n={c})" for v, c in zip(filt_mods, counts, strict=False)]
            mod_to_idx = {m: i for i, m in enumerate(filt_mods)}
            numeric = np.array([mod_to_idx.get(m, len(filt_mods) - 1) for m in mod_types])
            bg = {i for i, lbl in enumerate(labels) if "Unmodified" in lbl}
            return numeric, "Modification Types", True, labels, bg

        if name == "ptm_present":
            if "modification_types" not in meta:
                return None, None, False, None, set()
            mod_types = np.asarray(meta["modification_types"])
            ptm = np.array([0 if str(m).strip() == "Unmodified" else 1 for m in mod_types], dtype=int)
            unique_vals, counts = np.unique(ptm, return_counts=True)
            labels = []
            bg = set()
            for val, cnt in zip(unique_vals, counts, strict=False):
                if val == 0:
                    labels.append(f"Unmodified (n={cnt})")
                    bg.add(len(labels) - 1)
                else:
                    labels.append(f"Modified (n={cnt})")
            return ptm, "PTM Presence", True, labels, bg

        if name == "top_duplicate_peptides":
            if not np.issubdtype(data.dtype, np.number):
                return None, None, False, None, set()
            # Labels were pre-computed by _compute_top_duplicate_peptides
            labels_arr = meta.get("_top_duplicate_labels")
            if labels_arr is None:
                return None, None, False, None, set()
            labels = list(labels_arr)
            other_idx = len(labels) - 1  # last entry is "Other"
            return data, "Top Duplicate Peptides", True, labels, {other_idx}

        if name == "seq_cluster_id":
            if not np.issubdtype(data.dtype, np.number):
                return None, None, False, None, set()
            valid_clusters = data[data >= 0]
            if len(valid_clusters) == 0:
                return None, None, False, None, set()
            unique_cl, cl_counts = np.unique(valid_clusters, return_counts=True)
            order = np.argsort(cl_counts)[::-1]
            top_n = 10
            top_clusters = unique_cl[order[:top_n]]
            top_counts = cl_counts[order[:top_n]]
            labels = [f"Cluster {i + 1} (n={c})" for i, (_, c) in enumerate(zip(top_clusters, top_counts, strict=False))]
            bg = set()
            if len(unique_cl) > top_n:
                other_cnt = int(np.sum(cl_counts[order[top_n:]]))
                labels.append(f"Other (n={other_cnt})")
                bg.add(len(labels) - 1)
            cl_to_idx = {cl: i for i, cl in enumerate(top_clusters)}
            other_idx = len(top_clusters)
            numeric = np.array([cl_to_idx.get(c, other_idx) if c >= 0 else -1 for c in data])
            return numeric, "Sequence Similarity Clusters", True, labels, bg

        # ---- Generic handlers ----

        if np.issubdtype(data.dtype, np.number):
            if np.all(np.isnan(data)):
                return None, None, False, None, set()
            return data.astype(float), key.replace("_", " ").title(), False, None, set()

        # Categorical / string — replace None with "unknown" to avoid comparison errors
        data = np.array(["unknown" if v is None else v for v in data], dtype=object)
        unique_vals, counts = np.unique(data, return_counts=True)
        filt_vals, _, labels = self._manage_categorical_legend(unique_vals, counts)
        val_to_idx = {v: i for i, v in enumerate(filt_vals)}
        numeric = np.array([val_to_idx.get(v, len(filt_vals) - 1) for v in data])
        return numeric, key.replace("_", " ").title(), True, labels, set()

    # ------------------------------------------------------------------
    # Categorical scatter plotting (unified)
    # ------------------------------------------------------------------

    def _plot_categorical(
        self,
        ax: plt.Axes,
        emb_2d: np.ndarray,
        color_data: np.ndarray,
        labels: List[str],
        cmap_name: str,
        background_indices: Set[int],
        skip_value: Optional[int] = None,
    ) -> List[plt.Artist]:
        """Plot categorical data with optional gray background layer.

        Args:
            ax: Matplotlib axes.
            emb_2d: 2-D UMAP coordinates (N, 2).
            color_data: Integer category index per point.
            labels: Legend label per category index.
            cmap_name: Matplotlib colormap name for non-background categories.
            background_indices: Category indices to render in gray.
            skip_value: If set, points with this ``color_data`` value are omitted.

        Returns:
            List of legend handles.
        """
        # Use full colormap (not truncated to len(labels)) so colors stay saturated
        n_fg = max(1, len(labels) - len(background_indices))
        cmap_obj = plt.cm.get_cmap(cmap_name, max(n_fg, 8))
        handles: List[plt.Artist] = []

        # Background layer first (gray, lower alpha)
        for idx in sorted(background_indices):
            if idx >= len(labels):
                continue
            mask = color_data == idx
            if skip_value is not None:
                mask &= color_data != skip_value
            if not mask.any():
                continue
            h = ax.scatter(
                emb_2d[mask, 0],
                emb_2d[mask, 1],
                c="gray",
                s=self.point_size,
                alpha=self.alpha * 0.6,
                edgecolors="none",
                label=labels[idx],
            )
            handles.append(h)

        # Foreground categories — assign colors from 0..n_fg-1 to keep them distinct
        fg_color_idx = 0
        for idx, lbl in enumerate(labels):
            if idx in background_indices:
                continue
            mask = color_data == idx
            if skip_value is not None:
                mask &= color_data != skip_value
            if not mask.any():
                fg_color_idx += 1
                continue
            h = ax.scatter(
                emb_2d[mask, 0],
                emb_2d[mask, 1],
                c=[cmap_obj(fg_color_idx % cmap_obj.N)],
                s=self.point_size,
                alpha=self.alpha,
                edgecolors="none",
                label=lbl,
            )
            handles.append(h)
            fg_color_idx += 1

        return handles

    # ------------------------------------------------------------------
    # Single visualization
    # ------------------------------------------------------------------

    def _create_single_visualization(
        self,
        emb_2d: np.ndarray,
        meta: Dict[str, np.ndarray],
        config: Dict[str, Any],
        save_dir: Path,
    ) -> Optional[str]:
        """Create a single UMAP visualization colored by a metadata field."""
        try:
            key = config["key"]
            title = config["title"]
            cmap = config["cmap"]
            name = config["name"]
            vmin = float(config["vmin"]) if config.get("vmin") is not None else None
            vmax = float(config["vmax"]) if config.get("vmax") is not None else None

            color_data, color_label, is_categorical, labels, bg_indices = self._get_coloring_data(meta, key, name)
            if color_data is None:
                return None

            fig, ax = plt.subplots(figsize=self.plot_size)

            if is_categorical and labels is not None:
                skip_val = -1 if name == "seq_cluster_id" else None
                handles = self._plot_categorical(ax, emb_2d, color_data, labels, cmap, bg_indices, skip_value=skip_val)
                ax.set_title(f"UMAP: {title}")
                ax.set_xlabel("UMAP-1")
                ax.set_ylabel("UMAP-2")
                if handles:
                    ax.legend(
                        handles=handles,
                        title=color_label,
                        loc="center left",
                        bbox_to_anchor=(1.02, 0.5),
                        borderaxespad=0.0,
                        fontsize="small",
                    )
            else:
                scatter = ax.scatter(
                    emb_2d[:, 0],
                    emb_2d[:, 1],
                    c=color_data,
                    cmap=cmap,
                    s=self.point_size,
                    alpha=self.alpha,
                    edgecolors="none",
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.set_title(f"UMAP: {title}")
                ax.set_xlabel("UMAP-1")
                ax.set_ylabel("UMAP-2")
                if color_label:
                    cbar = fig.colorbar(scatter, ax=ax)
                    cbar.set_label(color_label)

            save_path = save_dir / f"umap_{name}.png"
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            return str(save_path)

        except Exception as e:
            logger.warning("Failed to create visualization for %s: %s", config.get("name", "unknown"), e)
            return None

    # ------------------------------------------------------------------
    # Summary multi-panel figure
    # ------------------------------------------------------------------

    def _create_summary_panel(
        self,
        emb_2d: np.ndarray,
        meta: Dict[str, np.ndarray],
        save_dir: Path,
    ) -> Optional[str]:
        """Create a 2x3 summary panel of the most informative UMAP views."""
        # Resolve which panel fields are actually available
        config_by_name = {c["name"]: c for c in self.visualization_configs}
        panel_configs = [config_by_name[f] for f in _SUMMARY_PANEL_FIELDS if f in config_by_name]

        # Filter to those with actual data
        available = []
        for cfg in panel_configs:
            cd, cl, is_cat, labels, bg = self._get_coloring_data(meta, cfg["key"], cfg["name"])
            if cd is not None:
                available.append((cfg, cd, cl, is_cat, labels, bg))

        if not available:
            return None

        n_cols = 3
        n_rows = (len(available) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
        axes_flat = np.asarray(axes).flatten()

        for idx, (cfg, cd, _cl, is_cat, labels, bg) in enumerate(available):
            ax = axes_flat[idx]
            name = cfg["name"]
            title = cfg["title"]
            cmap = cfg["cmap"]

            if is_cat and labels is not None:
                skip_val = -1 if name == "seq_cluster_id" else None
                self._plot_categorical(ax, emb_2d, cd, labels, cmap, bg, skip_value=skip_val)
            else:
                vmin = float(cfg["vmin"]) if cfg.get("vmin") is not None else None
                vmax = float(cfg["vmax"]) if cfg.get("vmax") is not None else None
                sc = ax.scatter(
                    emb_2d[:, 0],
                    emb_2d[:, 1],
                    c=cd,
                    cmap=cmap,
                    s=max(1, self.point_size // 2),
                    alpha=self.alpha,
                    edgecolors="none",
                    vmin=vmin,
                    vmax=vmax,
                )
                fig.colorbar(sc, ax=ax, shrink=0.7)

            ax.set_title(title, fontsize=10)
            ax.set_xlabel("UMAP-1", fontsize=8)
            ax.set_ylabel("UMAP-2", fontsize=8)
            ax.tick_params(labelsize=7)

        # Hide unused axes
        for idx in range(len(available), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle("UMAP Summary Panel", fontsize=14, y=1.01)
        fig.tight_layout()

        save_path = save_dir / "umap_summary_panel.png"
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.debug("Saved summary panel: %s", save_path)
        return str(save_path)

    def _run_conditional_umaps(
        self,
        emb_sampled: np.ndarray,
        meta_sampled: Dict[str, np.ndarray],
        base_save_dir: Path,
        saved_paths: List[str],
        results: Dict[str, Any],
    ) -> None:
        """Run separate UMAPs for each conditional subset."""
        import umap

        conditional_results: Dict[str, Any] = {}

        for subset_cfg in self.conditional_subsets:
            subset_name = subset_cfg.get("name", "unnamed")
            logger.info("Conditional UMAP subset: %s", subset_name)

            emb_filt, meta_filt, desc = self.apply_conditional_filter(
                emb_sampled,
                meta_sampled,
                subset_cfg,
            )
            n_filt = len(emb_filt)
            logger.info(
                "  Filtered: %d / %d spectra (%.1f%%) — %s",
                n_filt,
                len(emb_sampled),
                100.0 * n_filt / max(len(emb_sampled), 1),
                desc,
            )

            if n_filt < self.conditional_min_samples:
                logger.warning(
                    "  Skipping subset '%s': only %d samples (min=%d)",
                    subset_name,
                    n_filt,
                    self.conditional_min_samples,
                )
                conditional_results[subset_name] = {
                    "skipped": True,
                    "reason": f"insufficient samples ({n_filt} < {self.conditional_min_samples})",
                    "n_samples": n_filt,
                    "filter": desc,
                }
                continue

            # Compute derived properties on the filtered subset
            self._compute_spectral_properties(meta_filt)
            self._compute_annotation_properties(meta_filt)
            self._compute_top_duplicate_peptides(meta_filt)

            # UMAP projection on filtered subset
            reducer = umap.UMAP(
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric=self.metric,
                random_state=self.random_state,
                n_components=self.n_components,
                spread=self.spread,
                local_connectivity=self.local_connectivity,
                repulsion_strength=self.repulsion_strength,
                negative_sample_rate=self.negative_sample_rate,
                transform_queue_size=self.transform_queue_size,
                verbose=self.verbose,
                densmap=False,
                low_memory=True,
            )
            emb_2d_filt = np.asarray(reducer.fit_transform(emb_filt))

            # Quality metrics for this subset
            subset_quality: Dict[str, float] = {}
            if self.compute_quality_metrics:
                try:
                    subset_quality = self._compute_quality_metrics(emb_filt, emb_2d_filt)
                except Exception as e:
                    logger.warning("  Failed to compute quality metrics: %s", e)

            # Create output subdirectory
            subset_dir = base_save_dir / f"conditional_{subset_name}"
            subset_dir.mkdir(parents=True, exist_ok=True)

            # Generate all visualizations for this subset
            subset_paths: List[str] = []
            if self.create_multiple_plots:
                for cfg in self.visualization_configs:
                    plot_path = self._create_single_visualization(
                        emb_2d_filt,
                        meta_filt,
                        cfg,
                        subset_dir,
                    )
                    if plot_path:
                        subset_paths.append(plot_path)

            # Summary panel
            if self.create_summary_panel:
                panel_path = self._create_summary_panel(emb_2d_filt, meta_filt, subset_dir)
                if panel_path:
                    subset_paths.append(panel_path)

            saved_paths.extend(subset_paths)
            conditional_results[subset_name] = {
                "skipped": False,
                "n_samples": n_filt,
                "filter": desc,
                "quality_metrics": subset_quality,
                "n_plots": len(subset_paths),
                "save_dir": str(subset_dir),
            }
            knn_pres = subset_quality.get("knn_preservation_k15", 0)
            logger.info(
                "  %s: %d spectra, knn_preservation=%.3f, %d plots saved",
                subset_name,
                n_filt,
                knn_pres,
                len(subset_paths),
            )

        results["conditional_umaps"] = conditional_results

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, emb: np.ndarray, meta: Dict[str, np.ndarray], faiss_index: Any = None) -> Dict[str, Any]:  # type: ignore[override]  # base class run() signature differs across tasks
        """Run the UMAP visualisation task.

        Args:
            emb: Embeddings array of shape (N, D).
            meta: Metadata dictionary (extracted upstream in data processor).
            faiss_index: Not used.

        Returns:
            Dictionary with saved PNG paths, quality metrics, and timing info.
        """
        import umap

        start_time = time.time()
        rng = np.random.default_rng(self.random_state)

        # 1. Subsample FIRST (before any derived computation)
        emb_sampled, meta_sampled = self._subsample(emb, meta, self.max_samples, rng)
        logger.info("Sampled %d / %d embeddings for UMAP", len(emb_sampled), len(emb))

        # 2. Compute spectral properties on the subsample only
        self._compute_spectral_properties(meta_sampled)

        # 2b. Compute annotation properties (requires theoretical_spectrum.enabled=True)
        self._compute_annotation_properties(meta_sampled)

        # 2c. Compute top duplicate peptide coloring
        self._compute_top_duplicate_peptides(meta_sampled)

        # 3. Note pre-computed sequence clusters if present
        if "seq_cluster_id" in meta_sampled:
            valid_cl = meta_sampled["seq_cluster_id"]
            n_cl = len(np.unique(valid_cl[valid_cl >= 0]))
            logger.debug("Using pre-computed sequence clusters: %d clusters", n_cl)

        # 4. UMAP projection
        reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
            n_components=self.n_components,
            spread=self.spread,
            local_connectivity=self.local_connectivity,
            repulsion_strength=self.repulsion_strength,
            negative_sample_rate=self.negative_sample_rate,
            transform_queue_size=self.transform_queue_size,
            verbose=self.verbose,
            densmap=False,
            low_memory=True,
        )
        emb_2d = np.asarray(reducer.fit_transform(emb_sampled))

        # 5. Quality metrics
        quality_metrics: Dict[str, float] = {}
        if self.compute_quality_metrics:
            try:
                quality_metrics = self._compute_quality_metrics(emb_sampled, emb_2d)
                logger.info(
                    "Global UMAP quality: knn_preservation=%.3f, trustworthiness_k15=%.3f",
                    quality_metrics.get("knn_preservation_k15", 0),
                    quality_metrics.get("trustworthiness_k15", 0),
                )
            except Exception as e:
                logger.warning("Failed to compute quality metrics: %s", e)

        # 6. Create visualizations
        save_dir = self._create_unique_output_directory()
        saved_paths: List[str] = []
        visualization_stats: Dict[str, Any] = {}

        # Persist the 2D UMAP coordinates + per-spectrum metadata so that
        # custom (zoomed / recoloured) figures can be regenerated offline
        # without re-running the model. Eval read-path only; wrapped so it
        # can never block the visualisation task.
        coord_root = Path(self.output_dir) if self.output_dir else save_dir
        # (i) marker — proves this block executed in the running image
        try:
            (coord_root / "COORDS_BLOCK_REACHED.txt").write_text(f"reached run() coord-save: {len(emb_2d)} points\n")
            saved_paths.append(str(coord_root / "COORDS_BLOCK_REACHED.txt"))
        except Exception as e:
            logger.warning("UMAP_COORDS_MARKER_FAILED: %s", e)
        # (ii) build columns one at a time — a single bad column can't kill the save
        n_pts = len(emb_2d)
        data: Dict[str, Any] = {
            "umap_x": np.asarray(emb_2d[:, 0], dtype=float),
            "umap_y": np.asarray(emb_2d[:, 1], dtype=float),
        }
        for mk, mv in meta_sampled.items():
            try:
                arr = np.asarray(mv)
                if arr.ndim == 1 and len(arr) == n_pts:
                    data[mk] = arr if np.issubdtype(arr.dtype, np.number) else arr.astype(str)
            except Exception:
                continue
        # (iii) npz — rock-solid (numpy only, no dtype/parquet surprises)
        try:
            np.savez_compressed(str(coord_root / "umap_coordinates.npz"), **data)
            saved_paths.append(str(coord_root / "umap_coordinates.npz"))
            logger.info("UMAP_COORDS_NPZ_SAVED: %d points, %d cols", n_pts, len(data))
        except Exception as e:
            logger.warning("UMAP_COORDS_NPZ_FAILED: %s", e)
        # (iv) parquet — convenient for polars/pandas; per-column so one failure is skipped
        try:
            import polars as pl

            cols = {}
            for k, v in data.items():
                try:
                    cols[k] = v.tolist()
                except Exception:
                    pass
            pl.DataFrame(cols).write_parquet(str(coord_root / "umap_coordinates.parquet"))
            saved_paths.append(str(coord_root / "umap_coordinates.parquet"))
            logger.info("UMAP_COORDS_PARQUET_SAVED: %d cols", len(cols))
        except Exception as e:
            logger.warning("UMAP_COORDS_PARQUET_FAILED: %s", e)

        if self.create_multiple_plots:
            for cfg in self.visualization_configs:
                plot_path = self._create_single_visualization(emb_2d, meta_sampled, cfg, save_dir)
                viz_name = cfg["name"]
                if plot_path:
                    saved_paths.append(plot_path)
                    visualization_stats[viz_name] = {
                        "plot_path": plot_path,
                        "title": cfg["title"],
                        "colormap": cfg["cmap"],
                        "created": True,
                    }
                else:
                    visualization_stats[viz_name] = {
                        "plot_path": None,
                        "title": cfg["title"],
                        "colormap": cfg["cmap"],
                        "created": False,
                    }
        else:
            default_cfg: dict[str, Any] = {"name": "charge", "key": "precursor_charge", "title": "Precursor Charge", "cmap": "viridis"}
            plot_path = self._create_single_visualization(emb_2d, meta_sampled, default_cfg, save_dir)
            if plot_path:
                saved_paths.append(plot_path)
                visualization_stats["charge"] = {
                    "plot_path": plot_path,
                    "title": default_cfg["title"],
                    "colormap": default_cfg["cmap"],
                    "created": True,
                }

        # 7. Summary panel
        if self.create_summary_panel:
            panel_path = self._create_summary_panel(emb_2d, meta_sampled, save_dir)
            if panel_path:
                saved_paths.append(panel_path)

        # 8. Instrument-conditional UMAPs
        #    Build results dict first so _run_conditional_umaps can attach to it
        execution_time = time.time() - start_time

        results: Dict[str, Any] = {
            "task_name": self.name,
            "num_embeddings": len(emb),
            "num_sampled": len(emb_sampled),
            "save_paths": saved_paths,
            "execution_time": execution_time,
            "quality_metrics": quality_metrics,
            "visualization_stats": visualization_stats,
            "umap_parameters": {
                "n_neighbors": self.n_neighbors,
                "min_dist": self.min_dist,
                "metric": self.metric,
                "n_components": self.n_components,
                "spread": self.spread,
                "local_connectivity": self.local_connectivity,
                "repulsion_strength": self.repulsion_strength,
                "negative_sample_rate": self.negative_sample_rate,
                "transform_queue_size": self.transform_queue_size,
            },
            "visualization_config": {
                "max_samples": self.max_samples,
                "create_multiple_plots": self.create_multiple_plots,
                "create_summary_panel": self.create_summary_panel,
                "compute_quality_metrics": self.compute_quality_metrics,
                "plot_size": self.plot_size,
                "dpi": self.dpi,
                "alpha": self.alpha,
                "point_size": self.point_size,
                "max_categories": self.max_categories,
                "collapse_rare_categories": self.collapse_rare_categories,
                "ckpt_step": self.ckpt_step,
                "save_dir": str(save_dir),
            },
            "metadata_availability": {
                key: key in meta_sampled
                for key in [
                    "precursor_charge",
                    "sequence",
                    "retention_time",
                    "collision_energy",
                    "precursor_mz",
                    "hyperscore",
                    "search_acquisition",
                    "search_enzyme",
                    "search_detector",
                    "search_instrument",
                    "search_project",
                    "search_organism",
                    "search_quant",
                    "modifications",
                    "frag_type",
                    "ptm_present",
                    "modified_peptide",
                    "hydrophobicity",
                    "modification_types",
                    "seq_cluster_id",
                    "spectrum_confidence",
                    "sequence_length",
                    "unmodified_peptide",
                    "annotation_ratio",
                    "backbone_coverage",
                    "signal_intensity_ratio",
                    "n_fragment_groups_metric",
                    "median_ppm_error",
                ]
            },
        }

        if self.enable_conditional_umaps:
            logger.info("Running instrument-conditional UMAPs...")
            self._run_conditional_umaps(
                emb_sampled,
                meta_sampled,
                save_dir,
                saved_paths,
                results,
            )
            # Update execution time to include conditional UMAPs
            results["execution_time"] = time.time() - start_time
            results["save_paths"] = saved_paths

        return results

    # ------------------------------------------------------------------
    # Loggable metrics for MLflow
    # ------------------------------------------------------------------

    def get_loggable_metrics(self, task_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract UMAP quality metrics for MLflow logging.

        Logs kNN preservation only — trustworthiness is near-ceiling (>0.89)
        across all conditions and does not discriminate between models.
        """
        metrics: Dict[str, float] = {}

        qm = task_results.get("quality_metrics", {})
        if "knn_preservation_k15" in qm:
            metrics["knn_preservation_k15"] = float(qm["knn_preservation_k15"])

        for subset_name, subset_info in task_results.get("conditional_umaps", {}).items():
            if isinstance(subset_info, dict) and not subset_info.get("skipped", True):
                sq = subset_info.get("quality_metrics", {})
                if "knn_preservation_k15" in sq:
                    metrics[f"cond_{subset_name}_knn_preservation_k15"] = float(sq["knn_preservation_k15"])

        return metrics
