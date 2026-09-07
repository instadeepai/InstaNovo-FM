"""General-purpose Linear Probe evaluation task.

This task trains linear models (logistic regression for classification, ridge regression
for continuous targets) on embeddings to evaluate whether the latent space captures
various metadata attributes (e.g., charge, collision energy, hydrophobicity, etc.).

Supports project-disjoint splitting: probe-train/val/test are aligned with model splits,
and no project appears in more than one probe split. L2 regularization is tuned on the
val split, and final metrics are reported once on the test split.
"""

import logging
import re
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

try:
    import cuml
    from cuml.linear_model import LogisticRegression as _CuMLLogisticRegression
    from cuml.linear_model import Ridge as _CuMLRidge

    cuml.set_global_output_type("numpy")
    _CUML_AVAILABLE = True
except ImportError:
    _CUML_AVAILABLE = False

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from instanovo_fm.eval import embedding_io
from instanovo_fm.eval.embed_eval_tasks import BaseTask

logger = logging.getLogger(__name__)


class LinearProbeTask(BaseTask):
    """General-purpose linear probe evaluation task.

    Trains linear models on embeddings to predict metadata attributes.
    Automatically detects task type (classification vs regression) based on target data.

    When ``use_project_split=True`` (default), uses project-disjoint splitting aligned
    with model train/val/test splits. L2 regularization is tuned on val, final metrics
    reported once on test.

    When ``use_project_split=False``, falls back to legacy behaviour: internal random
    train/test split with GridSearchCV for hyperparameter tuning.
    """

    name = "Linear Probe"
    description = "Run linear probes (classification or regression) on embedding metadata"
    requires_metadata = True
    requires_faiss = False
    requires_multi_split = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Target field(s) to probe
        self.targets = kwargs.get("targets", None)
        if self.targets is not None and isinstance(self.targets, str):
            self.targets = [self.targets]

        # Optional paired within-backbone control for a binary modification target
        # (e.g. "mod_deam_n"). Groups the test split by unmodified backbone and, for
        # backbones seen both with and without the modification, checks whether the
        # probe scores the modified members higher -- isolating the modification's
        # spectral signature from sequence/motif priors. Value = target field name.
        self.paired_backbone_control = kwargs.get("paired_backbone_control", None)

        # General parameters
        self.random_state = kwargs.get("random_state", 42)
        self.probe_type = kwargs.get("probe_type", "auto")
        self.max_classes_for_classification = kwargs.get("max_classes_for_classification", 20)

        # Project-disjoint splitting parameters
        self.use_project_split = kwargs.get("use_project_split", True)
        self.project_key = kwargs.get("project_key", "search_project")
        # Optional per-target class exclusion, e.g. {"precursor_charge": ["0"]} to drop the
        # charge-0 sentinel. Charge 0 means "charge unknown" and maps exactly onto DIA
        # acquisition (100% of charge-0 spectra are DIA), so including it makes the charge
        # probe partly a DDA-vs-DIA classifier rather than a charge classifier.
        self.exclude_target_values = kwargs.get("exclude_target_values", None) or {}
        self.train_samples = kwargs.get("train_samples", 100_000)
        self.val_samples = kwargs.get("val_samples", 10_000)
        self.test_samples = kwargs.get("test_samples", 10_000)
        self.max_per_project_frac = kwargs.get("max_per_project_frac", 0.15)
        self.min_project_samples = kwargs.get("min_project_samples", 5)
        self.run_in_domain_baseline = kwargs.get("run_in_domain_baseline", False)

        # Classification parameters
        self.classification_params = {
            "max_iter": kwargs.get("max_iter", 1000),
            "solver": kwargs.get("solver", "lbfgs"),
            "class_weight": kwargs.get("class_weight", "balanced"),
            "penalty": kwargs.get("penalty", "l2"),
            "tol": kwargs.get("tol", 1e-4),
            "random_state": self.random_state,
            "n_jobs": kwargs.get("n_jobs", -1),
        }
        self.c_values = kwargs.get("c_values", [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])

        # cuML-compatible classification params: strip solver/class_weight/n_jobs/random_state (unsupported)
        if _CUML_AVAILABLE:
            self._cuml_classification_params = {
                k: v for k, v in self.classification_params.items() if k not in ("solver", "class_weight", "random_state", "n_jobs")
            }
        else:
            self._cuml_classification_params = None

        # Regression parameters
        self.alpha_values = kwargs.get("alpha_values", [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])

        # Legacy parameters (used when use_project_split=False)
        self.max_samples = kwargs.get("max_samples", 3000)
        self.test_size = kwargs.get("test_size", 0.2)
        self.use_grid_search = kwargs.get("use_grid_search", True)

    def run(
        self,
        emb: np.ndarray,
        meta: Dict[str, np.ndarray],
        faiss_index: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run linear probe evaluation.

        Args:
            emb: Embeddings array (N, D) — used for legacy mode or as fallback.
            meta: Metadata dictionary — used for legacy mode or as fallback.
            faiss_index: Not used.
            **kwargs: May contain ``splits`` dict for project-disjoint mode.

        Returns:
            Dictionary containing probe results for each target field.
        """
        if self.targets is None:
            raise ValueError("Must specify 'targets' parameter (list of metadata field names)")

        backend = "cuML (GPU)" if _CUML_AVAILABLE else "sklearn (CPU)"
        logger.info(f"Linear probe backend: {backend}")

        start_time = time.time()

        splits = kwargs.get("splits", None)
        pre_filtered = kwargs.get("pre_filtered", False)

        # Debug: log which path will be taken and metadata state
        logger.info(
            f"Linear probe: splits={'provided' if splits else 'None'}, pre_filtered={pre_filtered}, use_project_split={self.use_project_split}"
        )
        if splits is not None:
            for sname, (semb, smeta) in splits.items():
                if sname in ("train", "valid", "val", "test"):
                    proj = smeta.get(self.project_key)
                    proj_info = "missing"
                    if proj is not None:
                        proj_info = f"dtype={type(proj).__name__}, elem={type(proj[0]).__name__ if len(proj) > 0 else '?'}, n={len(proj)}"
                    logger.info(f"  {sname}: {len(semb)} embeddings, {self.project_key}={proj_info}")
        elif meta is not None:
            proj = meta.get(self.project_key)
            proj_info = "missing"
            if proj is not None:
                proj_info = f"dtype={type(proj).__name__}, elem={type(proj[0]).__name__ if len(proj) > 0 else '?'}, n={len(proj)}"
            logger.info(f"  meta: {self.project_key}={proj_info}")

        if splits is not None and (pre_filtered or not self.use_project_split):
            # Use model splits directly as probe splits (no project reassignment)
            results = self._run_pre_filtered(splits)
        elif self.use_project_split and splits is not None:
            results = self._run_project_disjoint(splits)
        else:
            if self.use_project_split and splits is None:
                logger.warning("use_project_split=True but no splits provided. Falling back to legacy internal split.")
            self.validate_inputs(emb, meta, faiss_index)
            results = self._run_legacy(emb, meta)

        results["execution_time"] = time.time() - start_time
        return results

    # ------------------------------------------------------------------
    # Project-disjoint pipeline
    # ------------------------------------------------------------------

    def _compute_probe_embedding_stats(
        self,
        probe_splits: Dict[str, Any],
        source: str,
    ) -> Dict[str, Any]:
        """Compute embedding statistics from final probe splits.

        Args:
            probe_splits: Dict with keys "train", "val", "test", each containing
                {"embeddings": np.ndarray, ...}.
            source: Label describing the data provenance (e.g. "project_disjoint_subsampled").

        Returns:
            Stats dict with source, split_counts, and embedding statistics.
        """
        split_counts = {}
        parts = []
        for name in ("train", "val", "test"):
            emb = probe_splits.get(name, {}).get("embeddings")
            if emb is not None and len(emb) > 0:
                parts.append(emb)
                split_counts[name] = len(emb)
            else:
                split_counts[name] = 0

        if not parts:
            return {"source": source, "split_counts": split_counts}

        all_emb = np.concatenate(parts, axis=0)
        stats = embedding_io.get_embedding_stats(all_emb)
        stats["source"] = source
        stats["split_counts"] = split_counts
        return stats

    @staticmethod
    def _compute_balanced_sample_weight(y: np.ndarray) -> np.ndarray:
        """Compute per-sample weights that replicate sklearn's class_weight='balanced'.

        Equivalent to: w_c = n_samples / (n_classes * count_c)
        """
        classes, counts = np.unique(y, return_counts=True)
        n_samples = len(y)
        n_classes = len(classes)
        weight_per_class = n_samples / (n_classes * counts)
        class_weight_map = dict(zip(classes, weight_per_class, strict=False))
        return np.array([class_weight_map[yi] for yi in y], dtype=np.float32)

    def _validate_splits(self, splits: Dict[str, tuple]) -> None:
        """Validate that splits dict has the expected structure.

        Raises:
            ValueError: If splits is missing required keys or contains invalid data.
        """
        if not splits:
            raise ValueError("splits dict is empty")
        for split_name, value in splits.items():
            if split_name == "assignment_info":
                continue
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError(f"splits['{split_name}'] must be a (embeddings, metadata) tuple, got {type(value)}")
            emb, meta = value
            if not isinstance(emb, np.ndarray) or emb.ndim != 2:
                raise ValueError(f"splits['{split_name}'][0] must be a 2D numpy array, got shape {getattr(emb, 'shape', 'unknown')}")

    def _run_project_disjoint(
        self,
        splits: Dict[str, tuple],
    ) -> Dict[str, Any]:
        """Run probes with project-disjoint train/val/test splits."""
        self._validate_splits(splits)
        from instanovo_fm.eval.probe_splitting import project_disjoint_split

        probe_splits = project_disjoint_split(
            splits=splits,
            project_key=self.project_key,
            train_samples=self.train_samples,
            val_samples=self.val_samples,
            test_samples=self.test_samples,
            max_per_project_frac=self.max_per_project_frac,
            random_state=self.random_state,
            min_project_samples=self.min_project_samples,
        )

        target_results = {}
        for target_field in self.targets:
            logger.debug(f"Running probe for target: {target_field}")

            if target_field == "modification_class":
                target_results[target_field] = self._run_hierarchical_ptm_probe(probe_splits)
            else:
                target_results[target_field] = self._run_single_target_probe(probe_splits, target_field)

        return {
            "task_name": self.name,
            "mode": "project_disjoint",
            "split_info": probe_splits.get("assignment_info", {}),
            "targets": target_results,
            "config": self._get_config_summary(),
            "embedding_stats": self._compute_probe_embedding_stats(probe_splits, source="project_disjoint_subsampled"),
        }

    def _run_pre_filtered(
        self,
        splits: Dict[str, tuple],
    ) -> Dict[str, Any]:
        """Run probes when embeddings are already project-disjoint filtered.

        Converts the evaluator's multi_splits format {name: (emb, meta)} into the
        probe_splits format expected by _run_single_target_probe, then runs each target.
        No further project assignment is performed.

        Args:
            splits: Dict mapping split names ("train", "valid"/"val", "test") to
                (embeddings, metadata) tuples.
        """
        self._validate_splits(splits)
        # Normalise "valid" -> "val" and convert tuple format to dict format
        probe_splits: Dict[str, Any] = {}
        n_samples: Dict[str, int] = {}
        n_projects_per_split: Dict[str, int] = {}
        all_projects: set = set()
        for split_name, (emb, meta) in splits.items():
            if split_name not in ("train", "valid", "val", "test"):
                continue
            norm = "val" if split_name == "valid" else split_name
            probe_splits[norm] = {
                "embeddings": emb,
                "metadata": meta,
                "projects": [],  # Already filtered upstream
            }
            n_samples[norm] = len(emb)
            # Count unique projects in this split from metadata if available
            proj_vals = meta.get(self.project_key, None)
            if proj_vals is not None and len(proj_vals) > 0:
                unique = set(proj_vals.tolist()) if hasattr(proj_vals, "tolist") else set(proj_vals)
                n_projects_per_split[norm] = len(unique)
                all_projects.update(unique)
            else:
                n_projects_per_split[norm] = 0

        # Report the regime that actually produced these splits rather than a fixed string.
        # "pre_filtered" was hardcoded here, so a project-assigned run and an overlapping run were
        # indistinguishable in the artefacts -- which is how two incomparable protocols went
        # unnoticed. Derive it from the data: if the same project appears in more than one split,
        # the splits were not project-assigned.
        counted = {s: n_projects_per_split.get(s, 0) for s in ("train", "val", "test")}
        overlap = sum(counted.values()) > len(all_projects) * 1.3 if all_projects else False
        probe_splits["assignment_info"] = {
            "n_projects_total": len(all_projects),
            "n_projects_per_split": counted,
            "n_samples_per_split": n_samples,
            "n_samples_dropped_invalid_project": 0,
            "targets": {"train": self.train_samples, "val": self.val_samples, "test": self.test_samples},
            "mode": "projects_overlap_splits" if overlap else "projects_assigned_to_splits",
            "use_project_split_requested": bool(self.use_project_split),
            "projects_shared_across_splits": bool(overlap),
        }
        if overlap and self.use_project_split:
            logger.warning(
                "  use_project_split=true was requested but projects appear in more than one "
                f"split ({counted} of {len(all_projects)} total) — the splits are NOT "
                "project-disjoint, so numbers are not comparable with project-assigned runs."
            )

        target_results = {}
        for target_field in self.targets:
            logger.debug(f"Running probe for target: {target_field}")
            if target_field == "modification_class":
                target_results[target_field] = self._run_hierarchical_ptm_probe(probe_splits)
            else:
                target_results[target_field] = self._run_single_target_probe(probe_splits, target_field)

        return {
            "task_name": self.name,
            "mode": "pre_filtered_project_disjoint",
            "split_info": probe_splits.get("assignment_info", {}),
            "targets": target_results,
            "config": self._get_config_summary(),
            "embedding_stats": self._compute_probe_embedding_stats(probe_splits, source="pre_filtered_project_disjoint"),
        }

    def _exclude_values(
        self,
        probe_splits: Dict[str, Any],
        target_field: str,
        drop: set,
    ) -> Dict[str, Any]:
        """Drop rows whose ``target_field`` value is in *drop*, in every split.

        Embeddings and all per-spectrum metadata arrays are masked together so that X and y
        stay aligned. Returns a copy; the input splits are left untouched.
        """
        out = dict(probe_splits)
        for split in ("train", "val", "test"):
            if split not in probe_splits or probe_splits[split] is None:
                continue
            emb = probe_splits[split]["embeddings"]
            meta = probe_splits[split]["metadata"]
            if target_field not in meta:
                continue
            vals = np.asarray([str(v) for v in np.asarray(meta[target_field]).ravel()])
            keep = ~np.isin(vals, list(drop))
            n0 = len(keep)
            if keep.all():
                continue
            new_meta = {}
            for k, v in meta.items():
                arr = np.asarray(v)
                new_meta[k] = arr[keep] if arr.ndim >= 1 and arr.shape[0] == n0 else v
            out[split] = {**probe_splits[split], "embeddings": emb[keep], "metadata": new_meta}
            logger.info(f"{target_field}: excluded {int((~keep).sum()):,} of {n0:,} {split} spectra with value in {sorted(drop)}")
        return out

    def _run_single_target_probe(
        self,
        probe_splits: Dict[str, Any],
        target_field: str,
    ) -> Dict[str, Any]:
        """Run a probe for a single target across project-disjoint splits."""
        # Optionally drop sentinel/unwanted classes for this target before anything else,
        # so they affect neither the label mapping nor the reported metrics.
        drop = self.exclude_target_values.get(target_field)
        if drop:
            drop = {str(v) for v in drop}
            probe_splits = self._exclude_values(probe_splits, target_field, drop)

        # Prepare data for each split
        train_data = self._prepare_target_data(
            probe_splits["train"]["embeddings"],
            probe_splits["train"]["metadata"],
            target_field,
        )
        val_data = self._prepare_target_data(
            probe_splits["val"]["embeddings"],
            probe_splits["val"]["metadata"],
            target_field,
        )
        test_data = self._prepare_target_data(
            probe_splits["test"]["embeddings"],
            probe_splits["test"]["metadata"],
            target_field,
        )

        # Check all splits have valid data
        for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            if data is None:
                return {"error": f"No valid data for {target_field} in {name} split"}

        train_emb, train_y, label_mapping = train_data
        val_emb, val_y, _ = val_data
        test_emb, test_y, _ = test_data

        # Use the same label mapping across splits (from train)
        if label_mapping is not None:
            val_y = self._apply_label_mapping(probe_splits["val"]["metadata"], target_field, label_mapping)
            test_y = self._apply_label_mapping(probe_splits["test"]["metadata"], target_field, label_mapping)
            if val_y is None or test_y is None:
                return {"error": f"Label mapping failed for {target_field}"}

            # Re-filter embeddings to match the valid mask from _apply_label_mapping
            # (avoids X/y length mismatch when _prepare_target_data filtered differently)
            val_valid = val_y >= 0
            val_emb = probe_splits["val"]["embeddings"][val_valid]
            val_y = val_y[val_valid]

            test_valid = test_y >= 0
            test_emb = probe_splits["test"]["embeddings"][test_valid]
            test_y = test_y[test_valid]

        # Detect task type
        task_type = self._detect_task_type(train_y, target_field)

        # Standardize embeddings
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_emb)
        X_val = scaler.transform(val_emb)
        X_test = scaler.transform(test_emb)

        if task_type == "classification":
            test_sequences = None
            if self.paired_backbone_control and target_field == self.paired_backbone_control:
                seqs = probe_splits["test"]["metadata"].get("sequence")
                if seqs is not None:
                    seqs = np.asarray(seqs)
                    # Align to the final test rows. With a label mapping, mirror the
                    # test_valid refilter above; otherwise a clean binary target drops
                    # no rows and the raw order matches (guarded by a length check).
                    if label_mapping is not None:
                        seqs = seqs[test_valid]
                    test_sequences = seqs
            return self._train_tune_eval_classification(
                X_train,
                train_y,
                X_val,
                val_y,
                X_test,
                test_y,
                target_field,
                label_mapping,
                test_sequences=test_sequences,
            )
        else:
            return self._train_tune_eval_regression(
                X_train,
                train_y,
                X_val,
                val_y,
                X_test,
                test_y,
                target_field,
            )

    def _run_hierarchical_ptm_probe(
        self,
        probe_splits: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run hierarchical PTM probing.

        Level 1: ptm_present (binary) — on all data.
        Level 2: modification_class — on PTM-positive examples only.
        """
        result = {}

        # Level 1: ptm_present (binary, all data)
        logger.debug("  Hierarchical PTM: running ptm_present (binary)")
        result["ptm_present"] = self._run_single_target_probe(probe_splits, "ptm_present")

        # Level 2: modification_class on PTM-positive only
        logger.debug("  Hierarchical PTM: running modification_class (PTM+ only)")
        ptm_pos_splits = self._filter_splits_by_ptm(probe_splits)

        if ptm_pos_splits is None:
            result["modification_class"] = {"error": "Insufficient PTM-positive samples for modification_class probe"}
        else:
            result["modification_class"] = self._run_single_target_probe(ptm_pos_splits, "modification_class")

        return result

    def _filter_splits_by_ptm(
        self,
        probe_splits: Dict[str, Any],
        min_samples: int = 50,
    ) -> Optional[Dict[str, Any]]:
        """Filter each split to only PTM-positive examples.

        Returns None if any split has too few PTM-positive samples.
        """
        filtered = {}
        for split_name in ("train", "val", "test"):
            split_data = probe_splits[split_name]
            meta = split_data["metadata"]

            if "ptm_present" not in meta:
                logger.warning(f"'ptm_present' not in {split_name} metadata. Cannot filter for PTM-positive.")
                return None

            ptm_labels = meta["ptm_present"]

            # Handle various representations of True/1; explicitly exclude NaN
            def _is_ptm_positive(v: Any) -> bool:
                try:
                    if v is None:
                        return False
                    if isinstance(v, float) and np.isnan(v):
                        return False
                    return bool(v) and v != 0 and str(v).lower() not in ("false", "0", "unmodified")
                except (TypeError, ValueError):
                    return False

            mask = np.array([_is_ptm_positive(v) for v in ptm_labels])

            if mask.sum() < min_samples:
                logger.warning(f"Only {mask.sum()} PTM-positive samples in {split_name} (min: {min_samples}). Skipping modification_class probe.")
                return None

            filtered_meta = {}
            for k, v in meta.items():
                try:
                    filtered_meta[k] = v[mask]
                except (TypeError, IndexError, KeyError):
                    logger.debug(f"Skipping non-indexable metadata key '{k}' in PTM filter")
            filtered[split_name] = {
                "embeddings": split_data["embeddings"][mask],
                "metadata": filtered_meta,
                "projects": split_data.get("projects", []),
            }

        # Carry over assignment_info
        if "assignment_info" in probe_splits:
            filtered["assignment_info"] = probe_splits["assignment_info"]

        return filtered

    # ------------------------------------------------------------------
    # Train / tune / eval (project-disjoint mode)
    # ------------------------------------------------------------------

    def _train_tune_eval_classification(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        target_field: str,
        label_mapping: Optional[Dict[int, str]] = None,
        test_sequences: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Train with L2 sweep on train, tune on val, report on test."""
        y_train = y_train.astype(int)
        y_val = y_val.astype(int)
        y_test = y_test.astype(int)

        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            return {"error": f"Fewer than 2 classes in train for {target_field}"}

        # Find classes present in both train and test
        test_classes = np.unique(y_test)
        common_classes = np.intersect1d(unique_classes, test_classes)
        if len(common_classes) < 2:
            return {"error": f"Fewer than 2 common classes between train and test for {target_field}"}

        if len(common_classes) < len(unique_classes):
            missing = set(unique_classes) - set(common_classes)
            missing_names = [label_mapping.get(c, str(c)) if label_mapping else str(c) for c in missing]
            logger.warning(
                f"  {len(missing)} classes in train absent from test: {missing_names}. Metrics computed on {len(common_classes)} common classes."
            )

        # L2 sweep: train per C, evaluate on val
        best_model = None
        best_c = None
        best_val_score = -np.inf
        val_scores = {}

        for c in self.c_values:
            if _CUML_AVAILABLE:
                clf = _CuMLLogisticRegression(C=c, **self._cuml_classification_params)
                sample_weight = self._compute_balanced_sample_weight(y_train)
                clf.fit(X_train, y_train, sample_weight=sample_weight)
            else:
                clf = LogisticRegression(C=c, **self.classification_params)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
                    clf.fit(X_train, y_train)

            val_pred = np.asarray(clf.predict(X_val))
            val_score = balanced_accuracy_score(y_val, val_pred)
            val_scores[c] = float(val_score)

            if val_score > best_val_score:
                best_val_score = val_score
                best_model = clf
                best_c = c

        logger.debug(f"  Best C={best_c} (val balanced_acc={best_val_score:.4f})")

        # Final evaluation on TEST
        y_pred = np.asarray(best_model.predict(X_test))

        # Probabilities for AUROC / AUCPR
        try:
            y_proba = np.asarray(best_model.predict_proba(X_test))
        except Exception:
            y_proba = None

        # Persist the test-set scores so ROC/PR curves can be drawn without re-running the
        # probe. Only the AUROC survives in the metrics dict, and a scalar cannot be turned
        # back into a curve.
        self._save_test_scores(target_field, y_test, y_pred, y_proba, unique_classes, common_classes, label_mapping)

        # Core metrics.
        #
        # macro-F1 MUST be averaged over `common_classes` (classes present in the test set).
        # Without an explicit `labels=`, sklearn averages over the union of labels appearing in
        # y_test OR y_pred: a class the model can predict but which has no test support then
        # enters the mean with F1 = 0 and inflates the denominator. Precursor charge hit exactly
        # this — a rare high-charge class present in train but not in the test subsample turned a
        # 7-class macro-F1 into an 8-class one. `macro_f1_over_predicted_labels` retains the old
        # (inconsistent-denominator) value so historical numbers can be reconciled.
        accuracy = accuracy_score(y_test, y_pred)
        bal_accuracy = balanced_accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, labels=common_classes, average="macro", zero_division=0)
        macro_f1_over_predicted = f1_score(y_test, y_pred, average="macro", zero_division=0)
        if not np.isclose(macro_f1, macro_f1_over_predicted):
            logger.warning(
                f"  {target_field}: macro-F1 over the {len(common_classes)} test-present classes "
                f"is {macro_f1:.4f}; averaging over every predicted label instead gives "
                f"{macro_f1_over_predicted:.4f}. Reporting the former."
            )

        # Baseline
        train_counts = np.bincount(y_train, minlength=max(unique_classes) + 1)
        majority_class = np.argmax(train_counts)
        baseline_accuracy = np.mean(y_test == majority_class)

        # AUROC
        auroc = self._compute_auroc(y_test, y_proba, unique_classes, keep_classes=common_classes)

        # AUCPR (binary only)
        aucpr = None
        if len(unique_classes) == 2 and y_proba is not None:
            try:
                aucpr = float(average_precision_score(y_test, y_proba[:, 1]))
            except Exception:
                pass

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=common_classes, zero_division=0)
        per_class = {}
        for idx, cls in enumerate(common_classes):
            cls_name = label_mapping.get(cls, str(cls)) if label_mapping else str(cls)
            per_class[cls_name] = {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
                "train_support": int(train_counts[cls]),
            }

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred, labels=common_classes)
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            conf_normalized = np.nan_to_num(conf_matrix / row_sums, nan=0.0)

        # Most confused pairs
        confused_pairs = []
        for i in range(len(common_classes)):
            for j in range(len(common_classes)):
                if i != j and conf_matrix[i, j] > 0:
                    true_name = label_mapping.get(common_classes[i], str(common_classes[i])) if label_mapping else str(common_classes[i])
                    pred_name = label_mapping.get(common_classes[j], str(common_classes[j])) if label_mapping else str(common_classes[j])
                    confused_pairs.append(
                        {
                            "true_class": true_name,
                            "predicted_as": pred_name,
                            "count": int(conf_matrix[i, j]),
                            "rate": float(conf_normalized[i, j]),
                        }
                    )
        confused_pairs.sort(key=lambda x: x["rate"], reverse=True)

        result = {
            "task_type": "classification",
            "target_field": target_field,
            "accuracy": float(accuracy),
            "balanced_accuracy": float(bal_accuracy),
            "macro_f1": float(macro_f1),
            "macro_f1_n_classes": int(len(common_classes)),
            "macro_f1_over_predicted_labels": float(macro_f1_over_predicted),
            "macro_auroc": float(auroc),
            "baseline_accuracy": float(baseline_accuracy),
            "improvement": float(accuracy - baseline_accuracy),
            "best_c": float(best_c),
            "val_scores_by_c": val_scores,
            "n_classes": int(len(common_classes)),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "per_class_metrics": per_class,
            "most_confused_pairs": confused_pairs[:10],
        }

        if aucpr is not None:
            result["aucpr"] = aucpr

        # Paired within-backbone control (only when requested for this binary target).
        if test_sequences is not None and y_proba is not None and len(common_classes) == 2 and 1 in list(common_classes):
            if len(test_sequences) != len(y_test):
                logger.warning(f"  Paired backbone control skipped for {target_field}: {len(test_sequences)} sequences vs {len(y_test)} test labels.")
            else:
                # Positive-class (deam=1) probability column. Prefer classes_ if the
                # estimator exposes it (sklearn); else fall back to the last column
                # (binary convention: classes sorted ascending -> col 1 == class 1).
                proba = np.asarray(y_proba)
                try:
                    pos_idx = list(best_model.classes_).index(1)
                except Exception:
                    pos_idx = proba.shape[1] - 1
                result["paired_backbone_control"] = self._compute_paired_backbone_control(test_sequences, y_test, proba[:, pos_idx])

        return result

    @staticmethod
    def _strip_mods(seq: str) -> str:
        """Unmodified backbone: drop [...] / (...) modification marks."""
        return re.sub(r"\[[^\]]*\]|\([^)]*\)", "", seq or "")

    def _compute_paired_backbone_control(
        self,
        sequences: np.ndarray,
        y_test: np.ndarray,
        proba_pos: np.ndarray,
    ) -> Dict[str, Any]:
        """Within-backbone paired control for a binary modification probe.

        Groups the test spectra by unmodified backbone and, for backbones observed
        both with (y=1) and without (y=0) the modification, measures whether the
        probe assigns a higher positive-class probability to the modified members.
        Because the backbone (and its sequence motifs) is identical within a pair, a
        separation reflects the modification's spectral signature rather than
        sequence/motif priors -- the strict test that the embedding encodes the
        modification chemistry. Pooled AUROC ~0.5 => the probe was riding sequence
        priors; >0.5 => it reads the modification from the spectrum.
        """
        from collections import defaultdict

        backbones = np.array([self._strip_mods(str(s)) for s in sequences])
        y = np.asarray(y_test).astype(int)
        p = np.asarray(proba_pos, dtype=float)

        groups: Dict[str, list] = defaultdict(list)
        for i, b in enumerate(backbones):
            groups[b].append(i)

        n_backbones_paired = 0
        n_pairs = 0
        concordant = 0.0  # proba(modified) > proba(unmodified); ties count 0.5
        per_backbone_auroc = []
        pos_probas: list = []
        neg_probas: list = []

        for _b, idx_list in groups.items():
            idxs = np.array(idx_list)
            pos = idxs[y[idxs] == 1]
            neg = idxs[y[idxs] == 0]
            if len(pos) == 0 or len(neg) == 0:
                continue
            n_backbones_paired += 1
            pp = p[pos][:, None]
            nn = p[neg][None, :]
            wins = float((pp > nn).sum()) + 0.5 * float((pp == nn).sum())
            total = int(len(pos) * len(neg))
            n_pairs += total
            concordant += wins
            per_backbone_auroc.append(wins / total)
            pos_probas.extend(p[pos].tolist())
            neg_probas.extend(p[neg].tolist())

        if n_pairs == 0:
            return {
                "available": False,
                "reason": "no backbone appears with both classes in the test split",
                "n_backbones_paired": 0,
            }

        return {
            "available": True,
            "n_backbones_paired": int(n_backbones_paired),
            "n_pairs": int(n_pairs),
            "pooled_within_backbone_auroc": float(concordant / n_pairs),
            "macro_within_backbone_auroc": float(np.mean(per_backbone_auroc)),
            "mean_proba_modified": float(np.mean(pos_probas)),
            "mean_proba_unmodified": float(np.mean(neg_probas)),
        }

    def _train_tune_eval_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        target_field: str,
    ) -> Dict[str, Any]:
        """Train with alpha sweep on train, tune on val, report on test."""
        best_model = None
        best_alpha = None
        best_val_score = -np.inf
        val_scores = {}

        for alpha in self.alpha_values:
            if _CUML_AVAILABLE:
                reg = _CuMLRidge(alpha=alpha)
            else:
                reg = Ridge(alpha=alpha, random_state=self.random_state)
            reg.fit(X_train, y_train)
            val_r2 = r2_score(y_val, np.asarray(reg.predict(X_val)))
            val_scores[alpha] = float(val_r2)

            if val_r2 > best_val_score:
                best_val_score = val_r2
                best_model = reg
                best_alpha = alpha

        logger.debug(f"  Best alpha={best_alpha} (val R²={best_val_score:.4f})")

        # Final evaluation on TEST
        y_pred = np.asarray(best_model.predict(X_test))

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        pearson = float(np.corrcoef(y_test, y_pred)[0, 1])

        try:
            spearman_corr, spearman_p = spearmanr(y_test, y_pred)
            spearman = float(spearman_corr)
        except Exception:
            spearman = float("nan")

        return {
            "task_type": "regression",
            "target_field": target_field,
            "r2": float(r2),
            "mae": float(mae),
            "rmse": rmse,
            "pearson_correlation": pearson,
            "spearman_correlation": spearman,
            "best_alpha": float(best_alpha),
            "val_scores_by_alpha": val_scores,
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "target_mean": float(np.mean(y_test)),
            "target_std": float(np.std(y_test)),
        }

    # ------------------------------------------------------------------
    # Legacy pipeline (internal random split + GridSearchCV)
    # ------------------------------------------------------------------

    def _run_legacy(self, emb: np.ndarray, meta: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Legacy probe pipeline with internal random split."""
        # Sample data if needed
        if len(emb) > self.max_samples:
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(len(emb), self.max_samples, replace=False)
            emb_sampled = emb[indices]
            meta_sampled = {}
            for k, v in meta.items():
                try:
                    meta_sampled[k] = v[indices]
                except (TypeError, IndexError, KeyError):
                    pass
        else:
            emb_sampled = emb
            meta_sampled = meta

        target_results = {}
        for target_field in self.targets:
            if target_field not in meta_sampled:
                logger.warning(f"Target field '{target_field}' not found in metadata. Available: {list(meta_sampled.keys())}")
                target_results[target_field] = {
                    "error": f"Target field '{target_field}' not found in metadata",
                }
                continue

            logger.debug(f"Running probe for target: {target_field}")
            probe_result = self._run_legacy_probe(emb_sampled, meta_sampled[target_field], target_field)
            target_results[target_field] = probe_result

        return {
            "task_name": self.name,
            "mode": "legacy",
            "num_embeddings": int(len(emb)),
            "num_sampled": int(len(emb_sampled)),
            "targets": target_results,
            "config": self._get_config_summary(),
        }

    def _run_legacy_probe(self, emb: np.ndarray, y: np.ndarray, target_field: str) -> Dict[str, Any]:
        """Run a single probe in legacy mode (internal split + GridSearchCV)."""
        data = self._prepare_target_data_from_array(emb, y, target_field)
        if data is None:
            return {"error": f"Cannot process target field {target_field}"}

        emb_valid, y_valid, label_mapping = data
        task_type = self._detect_task_type(y_valid, target_field)

        scaler = StandardScaler()
        emb_scaled = scaler.fit_transform(emb_valid)

        if task_type == "classification":
            return self._run_legacy_classification(emb_scaled, y_valid, target_field, label_mapping)
        else:
            return self._run_legacy_regression(emb_scaled, y_valid, target_field)

    def _run_legacy_classification(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_field: str,
        label_mapping: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        """Legacy classification probe with internal split + GridSearchCV."""
        y = y.astype(int)
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) < 2:
            return {"error": f"Insufficient classes: {len(unique_classes)} < 2"}

        min_samples_per_class = int(np.min(counts))
        use_stratify = min_samples_per_class >= 2

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if use_stratify else None,
        )

        # Baseline
        unique_train, train_counts = np.unique(y_train, return_counts=True)
        majority_class = unique_train[np.argmax(train_counts)]
        baseline_accuracy = float(np.mean(y_test == majority_class))

        # Train
        if self.use_grid_search and min_samples_per_class >= 2:
            cv_folds = max(2, min(3, min_samples_per_class))
            base_clf = LogisticRegression(**self.classification_params)
            grid_search = GridSearchCV(
                base_clf,
                {"C": self.c_values},
                cv=cv_folds,
                scoring="accuracy",
                n_jobs=-1,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
                grid_search.fit(X_train, y_train)
            clf = grid_search.best_estimator_
            best_c = grid_search.best_params_["C"]
        else:
            clf = LogisticRegression(**self.classification_params)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
                clf.fit(X_train, y_train)
            best_c = self.classification_params.get("C", 1.0)

        y_pred = clf.predict(X_test)
        # See the note in _train_tune_eval_classification: average macro-F1 over the classes
        # present in the test split, not over every label the model happens to predict.
        common_classes = np.intersect1d(np.unique(y_train), np.unique(y_test))
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, labels=common_classes, average="macro", zero_division=0)
        auroc = self._compute_auroc(y_test, clf.predict_proba(X_test), unique_classes, keep_classes=common_classes)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_test,
            y_pred,
            labels=unique_classes,
            zero_division=0,
        )
        train_counts = np.bincount(y_train, minlength=max(unique_classes) + 1)
        per_class = {}
        for idx, cls in enumerate(unique_classes):
            cls_name = label_mapping.get(cls, str(cls)) if label_mapping else str(cls)
            per_class[cls_name] = {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
                "train_support": int(train_counts[cls]),
            }

        return {
            "task_type": "classification",
            "target_field": target_field,
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "macro_f1_n_classes": int(len(common_classes)),
            "macro_auroc": float(auroc),
            "baseline_accuracy": baseline_accuracy,
            "improvement": float(accuracy - baseline_accuracy),
            "n_classes": int(len(unique_classes)),
            "best_c": float(best_c),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "per_class_metrics": per_class,
        }

    def _run_legacy_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_field: str,
    ) -> Dict[str, Any]:
        """Legacy regression probe with internal split + GridSearchCV."""
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        if self.use_grid_search:
            base_reg = Ridge(random_state=self.random_state)
            grid_search = GridSearchCV(
                base_reg,
                {"alpha": self.alpha_values},
                cv=3,
                scoring="r2",
                n_jobs=-1,
            )
            grid_search.fit(X_train, y_train)
            reg = grid_search.best_estimator_
            best_alpha = grid_search.best_params_["alpha"]
        else:
            reg = Ridge(alpha=1.0, random_state=self.random_state)
            reg.fit(X_train, y_train)
            best_alpha = 1.0

        y_pred = reg.predict(X_test)
        return {
            "task_type": "regression",
            "target_field": target_field,
            "r2": float(r2_score(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "pearson_correlation": float(np.corrcoef(y_test, y_pred)[0, 1]),
            "best_alpha": float(best_alpha),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "target_mean": float(np.mean(y_test)),
            "target_std": float(np.std(y_test)),
        }

    # ------------------------------------------------------------------
    # Data preparation helpers
    # ------------------------------------------------------------------

    def _prepare_target_data(
        self,
        embeddings: np.ndarray,
        metadata: Dict[str, np.ndarray],
        target_field: str,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[Dict[int, str]]]]:
        """Extract and validate a target from metadata for one split.

        Returns:
            (embeddings_valid, y_valid, label_mapping) or None on failure.
            label_mapping maps integer codes to original string labels (or None).
        """
        if target_field not in metadata:
            logger.warning(f"Target '{target_field}' not in metadata. Available: {list(metadata.keys())}")
            return None

        y = metadata[target_field]
        return self._prepare_target_data_from_array(embeddings, y, target_field)

    def _prepare_target_data_from_array(
        self,
        embeddings: np.ndarray,
        y: np.ndarray,
        target_field: str,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[Dict[int, str]]]]:
        """Process a raw target array into clean (embeddings, y, mapping)."""
        label_mapping = None

        # Flatten object arrays
        if y.dtype == object:
            y, label_mapping = self._flatten_object_array(y, target_field)
            if y is None:
                return None

        # Handle string dtypes
        if y.dtype.kind in {"U", "S"}:
            unique_values = np.unique(y)
            value_to_code = {val: idx for idx, val in enumerate(unique_values)}
            label_mapping = {idx: str(val) for val, idx in value_to_code.items()}
            y = np.array([value_to_code.get(val, -1) for val in y], dtype=int)
            valid_mask = y >= 0
        elif y.dtype.kind not in {"i", "f", "b", "u"}:
            try:
                y = y.astype(float)
                valid_mask = np.isfinite(y)
            except (ValueError, TypeError):
                logger.warning(f"Cannot convert {target_field} to numeric.")
                return None
        else:
            valid_mask = np.isfinite(y)

        if not valid_mask.any():
            logger.warning(f"No valid values for {target_field}.")
            return None

        return embeddings[valid_mask], y[valid_mask], label_mapping

    def _flatten_object_array(self, y: np.ndarray, target_field: str) -> Tuple[Optional[np.ndarray], Optional[Dict[int, str]]]:
        """Flatten an object-dtype array to scalar values.

        Returns (flattened_array, label_mapping) or (None, None) on failure.
        """
        flattened = []
        for item in y:
            try:
                if isinstance(item, np.ndarray):
                    if item.ndim == 0:
                        flattened.append(item.item())
                    elif item.size > 0:
                        flattened.append(item.flat[0])
                    else:
                        flattened.append(np.nan)
                elif isinstance(item, (list, tuple)):
                    flattened.append(item[0] if len(item) > 0 else np.nan)
                elif isinstance(item, (int, float, np.integer, np.floating)):
                    flattened.append(item)
                elif isinstance(item, str):
                    flattened.append(item)
                elif item is None:
                    flattened.append(np.nan)
                else:
                    try:
                        flattened.append(float(item))
                    except (ValueError, TypeError):
                        flattened.append(np.nan)
            except Exception as e:
                logger.warning(f"Could not process item in {target_field}: {type(item)} - {e}")
                flattened.append(np.nan)

        y = np.array(flattened)

        # Check if strings after flattening
        label_mapping = None
        if y.dtype == object and len(y) > 0:
            first_item = y[0]
            is_string = isinstance(first_item, (str, np.str_))
            if is_string:
                unique_values = np.unique(y[y != None])  # noqa: E711
                value_to_code = {val: idx for idx, val in enumerate(unique_values)}
                label_mapping = {idx: str(val) for val, idx in value_to_code.items()}
                y = np.array([value_to_code.get(val, -1) for val in y], dtype=int)
            else:
                try:
                    y = y.astype(float)
                except (ValueError, TypeError):
                    return None, None

        return y, label_mapping

    def _apply_label_mapping(
        self,
        metadata: Dict[str, np.ndarray],
        target_field: str,
        label_mapping: Dict[int, str],
    ) -> Optional[np.ndarray]:
        """Apply an existing label mapping to a different split's target."""
        if target_field not in metadata:
            return None

        y = metadata[target_field]

        # Build reverse mapping: string -> code
        str_to_code = {v: k for k, v in label_mapping.items()}

        result = []
        for item in y:
            val = item
            if isinstance(val, np.ndarray):
                val = val.item() if val.ndim == 0 else (val.flat[0] if val.size > 0 else None)
            if isinstance(val, (list, tuple)):
                val = val[0] if len(val) > 0 else None

            s = str(val).strip() if val is not None else None
            code = str_to_code.get(s, -1)
            result.append(code)

        result = np.array(result, dtype=int)
        valid = result >= 0
        if not valid.any():
            return None

        return result

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _detect_task_type(self, y: np.ndarray, target_field: str) -> str:
        """Detect classification vs regression from target values."""
        if self.probe_type != "auto":
            return self.probe_type

        valid_mask = np.isfinite(y)
        y_valid = y[valid_mask]

        if len(y_valid) == 0:
            raise ValueError(f"No valid values for target field '{target_field}'")

        is_integer = y_valid.dtype.kind in {"i", "b", "u"}
        n_unique = len(np.unique(y_valid))

        if is_integer and n_unique <= self.max_classes_for_classification:
            task_type = "classification"
        else:
            task_type = "regression"

        logger.debug(f"  Detected task type: {task_type} (n_unique={n_unique}, dtype={y_valid.dtype})")
        return task_type

    def _save_test_scores(
        self,
        target_field: str,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        train_classes: np.ndarray,
        common_classes: np.ndarray,
        label_mapping: Optional[Dict[int, str]] = None,
    ) -> None:
        """Write per-example test scores for one target, for ROC/PR curves.

        One row per test spectrum: the true label, the predicted label, and one probability
        column per class the model was trained on (named ``p_<class>``). Classes absent from
        the test split are still written, so a reader can reproduce both the reported AUROC
        (test-present classes only) and the raw model output.
        """
        # BaseTask stashes constructor kwargs in self.config; the evaluator passes output_dir there.
        out_dir = getattr(self, "output_dir", None) or (self.config or {}).get("output_dir")
        if out_dir is None:
            logger.warning(f"  No output_dir available — not saving test scores for {target_field}")
            return
        if y_proba is None:
            return
        try:
            import polars as pl

            name = lambda c: (label_mapping.get(c, str(c)) if label_mapping else str(c))
            data = {
                "y_true": np.asarray(y_test).astype(int),
                "y_true_label": [name(int(c)) for c in y_test],
                "y_pred": np.asarray(y_pred).astype(int),
            }
            for i, c in enumerate(np.asarray(train_classes)):
                if i < y_proba.shape[1]:
                    data[f"p_{name(int(c))}"] = y_proba[:, i].astype(float)
            path = Path(out_dir) / f"test_scores_{target_field}.parquet"
            pl.DataFrame(data).write_parquet(path)
            logger.info(
                f"  Saved {len(y_test):,} test scores for {target_field} to {path.name} "
                f"({len(common_classes)} scored classes of {len(train_classes)} trained)"
            )
        except Exception as e:  # a diagnostic dump must never break the probe
            logger.warning(f"  Could not save test scores for {target_field}: {e}")

    def _compute_auroc(
        self,
        y_true: np.ndarray,
        y_proba: Optional[np.ndarray],
        unique_classes: np.ndarray,
        keep_classes: Optional[np.ndarray] = None,
    ) -> float:
        """Compute macro AUROC, handling binary and multi-class.

        `unique_classes` are the classes the model was trained on, i.e. the columns of
        `y_proba`. `keep_classes` are the classes present in `y_true`; when a trained class has
        no test support its column is dropped and the remaining probabilities renormalised, so
        the one-vs-rest average runs over the same classes as macro-F1. Without this,
        roc_auc_score rejects the shape mismatch and the metric silently becomes NaN.

        Returns float('nan') when AUROC cannot be computed (e.g. single class
        in test set, missing probabilities) so callers can distinguish a
        genuine failure from a legitimately bad score of 0.0.
        """
        if y_proba is None:
            return float("nan")
        classes = np.asarray(unique_classes)
        if keep_classes is not None and len(keep_classes) < len(classes):
            cols = np.isin(classes, keep_classes)
            y_proba = np.asarray(y_proba, dtype=float)[:, cols]
            row_sum = y_proba.sum(axis=1, keepdims=True)
            if (row_sum <= 0).any():
                return float("nan")
            y_proba = y_proba / row_sum
            classes = classes[cols]
        try:
            if len(classes) == 2:
                return float(roc_auc_score(y_true, y_proba[:, 1]))
            elif len(classes) > 2:
                return float(
                    roc_auc_score(
                        y_true,
                        y_proba,
                        multi_class="ovr",
                        average="macro",
                        labels=classes,
                    )
                )
        except Exception as e:
            logger.warning(f"Could not compute AUROC: {e}")
        return float("nan")

    def _get_config_summary(self) -> Dict[str, Any]:
        """Return serialisable config summary."""
        return {
            "targets": list(self.targets) if self.targets else [],
            "use_project_split": self.use_project_split,
            "project_key": self.project_key,
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
            "test_samples": self.test_samples,
            "max_per_project_frac": self.max_per_project_frac,
            "random_state": self.random_state,
            "probe_type": self.probe_type,
            "c_values": list(self.c_values),
            "alpha_values": list(self.alpha_values),
        }

    # ------------------------------------------------------------------
    # Loggable metrics
    # ------------------------------------------------------------------

    def get_loggable_metrics(self, task_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract headline metrics for MLflow logging (one per target).

        Logs a single primary metric per target for quick ablation comparison:
        - Regression: R²
        - Classification: balanced_accuracy
        - Multiclass: macro_f1

        Full per-target detail is always saved in the JSON results on disk.
        """
        loggable = {}

        if "error" in task_results:
            return loggable

        targets = task_results.get("targets", {})
        for target_field, target_result in targets.items():
            # Handle hierarchical PTM results — only log the multiclass level
            # (ptm_present is already logged as a standalone target)
            if target_field == "modification_class" and "ptm_present" in target_result:
                sub = target_result.get("modification_class")
                if isinstance(sub, dict) and "error" not in sub:
                    self._extract_headline_metric(loggable, target_field, sub)
                continue

            if not isinstance(target_result, dict) or "error" in target_result:
                continue

            self._extract_headline_metric(loggable, target_field, target_result)

        return loggable

    def _extract_headline_metric(
        self,
        loggable: Dict[str, float],
        target: str,
        result: Dict[str, Any],
    ) -> None:
        """Extract a single headline metric from a target result.

        Regression  → R² (normalized goodness-of-fit, 0 = mean predictor)
        Binary      → balanced_accuracy (threshold-dependent but handles imbalance)
        Multiclass  → macro_f1 (harmonic mean of per-class precision/recall)
        """
        task_type = result.get("task_type", "unknown")

        if task_type == "regression":
            loggable[f"{target}/r2"] = result.get("r2", 0.0)
        elif task_type == "classification":
            n_classes = result.get("n_classes", 0)
            if n_classes <= 2:
                loggable[f"{target}/balanced_accuracy"] = result.get("balanced_accuracy", 0.0)
            else:
                loggable[f"{target}/macro_f1"] = result.get("macro_f1", 0.0)
