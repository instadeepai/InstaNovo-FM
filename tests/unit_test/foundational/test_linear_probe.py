"""Tests for the refactored LinearProbeTask."""

import numpy as np
import pytest

from instanovo_fm.eval.embed_eval_tasks.linear_probe import LinearProbeTask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_probe_splits(
    n_train: int = 500,
    n_val: int = 100,
    n_test: int = 100,
    d: int = 32,
    n_projects: int = 5,
    n_classes: int = 3,
    seed: int = 42,
):
    """Create synthetic multi-split data for linear probe tests.

    Returns a splits dict matching what the evaluator passes.
    """
    rng = np.random.RandomState(seed)

    # Assign projects to splits (disjoint)
    projects_per_split = {
        "train": [f"PXD_{i}" for i in range(0, n_projects - 2)],
        "valid": [f"PXD_{n_projects - 2}"],
        "test": [f"PXD_{n_projects - 1}"],
    }

    splits = {}
    for split_name, (n, proj_list) in zip(
        ("train", "valid", "test"),
        ((n_train, projects_per_split["train"]), (n_val, projects_per_split["valid"]), (n_test, projects_per_split["test"])),
        strict=False,
    ):
        # Create embeddings that are linearly separable by charge
        charges = rng.randint(1, 1 + n_classes, size=n)
        emb = rng.randn(n, d).astype(np.float32)
        # Add signal: shift embedding mean by charge value
        for c in range(1, 1 + n_classes):
            mask = charges == c
            emb[mask, 0] += c * 2.0

        # Assign projects
        projects = np.array(
            [proj_list[i % len(proj_list)] for i in range(n)],
            dtype=object,
        )

        # Continuous target: correlated with first embedding dim
        precursor_mz = emb[:, 0] * 100 + rng.randn(n) * 10

        # PTM labels
        ptm_present = rng.randint(0, 2, size=n)
        mod_classes = np.array(
            ["Oxidation" if ptm_present[i] else "Unmodified" for i in range(n)],
            dtype=object,
        )

        meta = {
            "search_project": projects,
            "precursor_charge": charges,
            "precursor_mz": precursor_mz.astype(np.float32),
            "ptm_present": ptm_present,
            "modification_class": mod_classes,
            "frag_type": np.array([["HCD", "ETD"][i % 2] for i in range(n)], dtype=object),
        }

        splits[split_name] = (emb, meta)

    return splits


# ---------------------------------------------------------------------------
# Tests: Project-disjoint mode
# ---------------------------------------------------------------------------


class TestLinearProbeProjectDisjoint:
    def test_classification_probe_runs(self):
        splits = _make_probe_splits()
        task = LinearProbeTask(
            targets=["precursor_charge"],
            use_project_split=True,
            train_samples=300,
            val_samples=80,
            test_samples=80,
        )
        result = task.run(
            emb=np.zeros((1, 32)),  # unused in project-disjoint mode
            meta={},
            faiss_index=None,
            splits=splits,
        )

        assert result["mode"] == "project_disjoint"
        assert "precursor_charge" in result["targets"]
        charge_result = result["targets"]["precursor_charge"]
        assert charge_result["task_type"] == "classification"
        assert "accuracy" in charge_result
        assert "balanced_accuracy" in charge_result
        assert "macro_f1" in charge_result
        assert "best_c" in charge_result
        assert "val_scores_by_c" in charge_result
        # With small synthetic data, just check it's above random (1/3 for 3 classes)
        assert charge_result["accuracy"] >= 0.33

    def test_regression_probe_runs(self):
        splits = _make_probe_splits()
        task = LinearProbeTask(
            targets=["precursor_mz"],
            use_project_split=True,
            train_samples=300,
            val_samples=80,
            test_samples=80,
        )
        result = task.run(
            emb=np.zeros((1, 32)),
            meta={},
            faiss_index=None,
            splits=splits,
        )

        mz_result = result["targets"]["precursor_mz"]
        assert mz_result["task_type"] == "regression"
        assert "r2" in mz_result
        assert "mae" in mz_result
        assert "spearman_correlation" in mz_result
        assert "best_alpha" in mz_result

    def test_hierarchical_ptm_probe(self):
        splits = _make_probe_splits()
        task = LinearProbeTask(
            targets=["modification_class"],
            use_project_split=True,
            train_samples=300,
            val_samples=80,
            test_samples=80,
        )
        result = task.run(
            emb=np.zeros((1, 32)),
            meta={},
            faiss_index=None,
            splits=splits,
        )

        mod_result = result["targets"]["modification_class"]
        # Hierarchical: should have ptm_present and modification_class sub-results
        assert "ptm_present" in mod_result
        assert "modification_class" in mod_result

    def test_multiple_targets(self):
        splits = _make_probe_splits()
        task = LinearProbeTask(
            targets=["precursor_charge", "precursor_mz", "frag_type"],
            use_project_split=True,
            train_samples=300,
            val_samples=80,
            test_samples=80,
        )
        result = task.run(
            emb=np.zeros((1, 32)),
            meta={},
            faiss_index=None,
            splits=splits,
        )

        assert "precursor_charge" in result["targets"]
        assert "precursor_mz" in result["targets"]
        assert "frag_type" in result["targets"]

    def test_missing_target_handled(self):
        splits = _make_probe_splits()
        task = LinearProbeTask(
            targets=["nonexistent_field"],
            use_project_split=True,
            train_samples=300,
            val_samples=80,
            test_samples=80,
        )
        result = task.run(
            emb=np.zeros((1, 32)),
            meta={},
            faiss_index=None,
            splits=splits,
        )

        assert "error" in result["targets"]["nonexistent_field"]

    def test_split_info_in_results(self):
        splits = _make_probe_splits()
        task = LinearProbeTask(
            targets=["precursor_charge"],
            use_project_split=True,
            train_samples=300,
            val_samples=80,
            test_samples=80,
        )
        result = task.run(
            emb=np.zeros((1, 32)),
            meta={},
            faiss_index=None,
            splits=splits,
        )

        assert "split_info" in result
        info = result["split_info"]
        assert "n_projects_total" in info
        assert "n_samples_per_split" in info


# ---------------------------------------------------------------------------
# Tests: Legacy mode
# ---------------------------------------------------------------------------


class TestLinearProbeLegacy:
    def test_legacy_fallback_when_no_splits(self):
        rng = np.random.RandomState(42)
        n, d = 200, 16
        emb = rng.randn(n, d).astype(np.float32)
        charges = rng.randint(1, 4, size=n)
        emb[:, 0] += charges * 2.0

        meta = {"precursor_charge": charges}

        task = LinearProbeTask(
            targets=["precursor_charge"],
            use_project_split=True,  # True, but no splits kwarg
            max_samples=200,
        )
        result = task.run(emb, meta, faiss_index=None)

        # Should fall back to legacy
        assert result["mode"] == "legacy"
        assert "precursor_charge" in result["targets"]

    def test_legacy_explicit(self):
        rng = np.random.RandomState(42)
        n, d = 200, 16
        emb = rng.randn(n, d).astype(np.float32)
        meta = {"precursor_charge": rng.randint(1, 4, size=n)}

        task = LinearProbeTask(
            targets=["precursor_charge"],
            use_project_split=False,
            max_samples=200,
        )
        result = task.run(emb, meta, faiss_index=None)

        assert result["mode"] == "legacy"


# ---------------------------------------------------------------------------
# Tests: Loggable metrics
# ---------------------------------------------------------------------------


class TestLoggableMetrics:
    def test_classification_headline_metric(self):
        """Multiclass classification logs macro_f1 as headline metric."""
        task = LinearProbeTask(targets=["precursor_charge"])
        results = {
            "targets": {
                "precursor_charge": {
                    "task_type": "classification",
                    "n_classes": 7,
                    "accuracy": 0.9,
                    "balanced_accuracy": 0.88,
                    "macro_f1": 0.87,
                    "macro_auroc": 0.95,
                    "improvement": 0.5,
                }
            }
        }
        metrics = task.get_loggable_metrics(results)

        assert "precursor_charge/macro_f1" in metrics
        assert metrics["precursor_charge/macro_f1"] == 0.87
        assert len(metrics) == 1  # Only headline metric

    def test_binary_classification_headline_metric(self):
        """Binary classification logs balanced_accuracy as headline metric."""
        task = LinearProbeTask(targets=["ptm_present"])
        results = {
            "targets": {
                "ptm_present": {
                    "task_type": "classification",
                    "n_classes": 2,
                    "accuracy": 0.8,
                    "balanced_accuracy": 0.78,
                    "macro_f1": 0.77,
                    "aucpr": 0.65,
                }
            }
        }
        metrics = task.get_loggable_metrics(results)

        assert "ptm_present/balanced_accuracy" in metrics
        assert metrics["ptm_present/balanced_accuracy"] == 0.78
        assert len(metrics) == 1

    def test_regression_headline_metric(self):
        """Regression logs R² as headline metric."""
        task = LinearProbeTask(targets=["precursor_mz"])
        results = {
            "targets": {
                "precursor_mz": {
                    "task_type": "regression",
                    "r2": 0.85,
                    "mae": 10.5,
                    "rmse": 15.2,
                    "pearson_correlation": 0.92,
                    "spearman_correlation": 0.90,
                }
            }
        }
        metrics = task.get_loggable_metrics(results)

        assert "precursor_mz/r2" in metrics
        assert metrics["precursor_mz/r2"] == 0.85
        assert len(metrics) == 1

    def test_hierarchical_ptm_logs_only_multiclass(self):
        """Hierarchical PTM logs only the multiclass level (ptm_present is a standalone target)."""
        task = LinearProbeTask(targets=["modification_class"])
        results = {
            "targets": {
                "modification_class": {
                    "ptm_present": {
                        "task_type": "classification",
                        "n_classes": 2,
                        "accuracy": 0.8,
                        "balanced_accuracy": 0.78,
                        "macro_f1": 0.77,
                        "aucpr": 0.65,
                    },
                    "modification_class": {
                        "task_type": "classification",
                        "n_classes": 7,
                        "accuracy": 0.7,
                        "balanced_accuracy": 0.68,
                        "macro_f1": 0.65,
                    },
                }
            }
        }
        metrics = task.get_loggable_metrics(results)

        # Only the multiclass level is logged (ptm_present is a standalone target)
        assert "modification_class/macro_f1" in metrics
        assert metrics["modification_class/macro_f1"] == 0.65
        assert "modification_class/ptm_present/aucpr" not in metrics
        assert len(metrics) == 1

    def test_error_target_skipped(self):
        task = LinearProbeTask(targets=["bad"])
        results = {"targets": {"bad": {"error": "No data"}}}
        metrics = task.get_loggable_metrics(results)
        assert len(metrics) == 0


# ---------------------------------------------------------------------------
# Tests: _run_pre_filtered path
# ---------------------------------------------------------------------------


class TestLinearProbePreFiltered:
    """Tests for the _run_pre_filtered fast path (embeddings already split)."""

    def _make_pre_filtered_splits(
        self,
        n_train: int = 400,
        n_val: int = 80,
        n_test: int = 80,
        d: int = 32,
        n_classes: int = 3,
        seed: int = 42,
    ):
        """Build pre-filtered splits in evaluator format {name: (emb, meta)}."""
        rng = np.random.RandomState(seed)
        splits = {}
        for split_name, n in [("train", n_train), ("valid", n_val), ("test", n_test)]:
            charges = rng.randint(1, 1 + n_classes, size=n)
            emb = rng.randn(n, d).astype(np.float32)
            emb[:, 0] += charges * 2.0
            precursor_mz = emb[:, 0] * 100 + rng.randn(n) * 5
            ptm_present = rng.randint(0, 2, size=n)
            mod_classes = np.array(
                ["Oxidation" if ptm_present[i] else "Unmodified" for i in range(n)],
                dtype=object,
            )
            meta = {
                "precursor_charge": charges,
                "precursor_mz": precursor_mz.astype(np.float32),
                "ptm_present": ptm_present,
                "modification_class": mod_classes,
            }
            splits[split_name] = (emb, meta)
        return splits

    def test_pre_filtered_classification(self):
        splits = self._make_pre_filtered_splits()
        task = LinearProbeTask(
            targets=["precursor_charge"],
            train_samples=400,
            val_samples=80,
            test_samples=80,
        )
        result = task.run(
            emb=np.zeros((1, 32)),
            meta={},
            faiss_index=None,
            splits=splits,
            pre_filtered=True,
        )

        assert result["mode"] == "pre_filtered_project_disjoint"
        assert "precursor_charge" in result["targets"]
        charge_result = result["targets"]["precursor_charge"]
        assert charge_result["task_type"] == "classification"
        assert "accuracy" in charge_result
        assert "balanced_accuracy" in charge_result

    def test_pre_filtered_regression(self):
        splits = self._make_pre_filtered_splits()
        task = LinearProbeTask(
            targets=["precursor_mz"],
            train_samples=400,
            val_samples=80,
            test_samples=80,
        )
        result = task.run(
            emb=np.zeros((1, 32)),
            meta={},
            faiss_index=None,
            splits=splits,
            pre_filtered=True,
        )

        mz_result = result["targets"]["precursor_mz"]
        assert mz_result["task_type"] == "regression"
        assert "r2" in mz_result
        assert "spearman_correlation" in mz_result

    def test_pre_filtered_hierarchical_ptm(self):
        splits = self._make_pre_filtered_splits()
        task = LinearProbeTask(
            targets=["modification_class"],
            train_samples=400,
            val_samples=80,
            test_samples=80,
        )
        result = task.run(
            emb=np.zeros((1, 32)),
            meta={},
            faiss_index=None,
            splits=splits,
            pre_filtered=True,
        )

        mod_result = result["targets"]["modification_class"]
        assert "ptm_present" in mod_result
        assert "modification_class" in mod_result

    def test_pre_filtered_split_info_mode(self) -> None:
        """split_info must describe the regime that produced the splits, not a fixed string.

        The mode used to be hardcoded to "pre_filtered", which made a project-assigned run and a
        run whose projects span all three splits indistinguishable in the artefacts -- the reason
        two incomparable protocols went unnoticed.
        """
        splits = self._make_pre_filtered_splits()
        task = LinearProbeTask(targets=["precursor_charge"])
        result = task.run(
            emb=np.zeros((1, 32)),
            meta={},
            faiss_index=None,
            splits=splits,
            pre_filtered=True,
        )
        info = result["split_info"]
        assert info["mode"] in ("projects_assigned_to_splits", "projects_overlap_splits")
        assert "projects_shared_across_splits" in info
        assert "use_project_split_requested" in info
        # the fixture gives each split its own projects, so this is the assigned regime
        assert info["mode"] == "projects_assigned_to_splits"
        assert info["projects_shared_across_splits"] is False

    def test_split_info_flags_projects_shared_across_splits(self) -> None:
        """When every split draws on the same projects, that must be recorded, not hidden.

        This is the case that produced the paper's probe numbers: projects appear in train, val and
        test alike, so the scores are not project-disjoint and are not comparable with runs where
        projects were assigned to a single split.
        """
        splits = self._make_pre_filtered_splits()
        shared = np.array(["PXD000001"] * 400 + ["PXD000002"] * 400, dtype=object)
        for name, (emb, meta) in splits.items():
            meta["search_project"] = shared[: len(emb)].copy()
        task = LinearProbeTask(targets=["precursor_charge"])
        result = task.run(
            emb=np.zeros((1, 32)), meta={}, faiss_index=None, splits=splits, pre_filtered=True,
        )
        info = result["split_info"]
        assert info["mode"] == "projects_overlap_splits"
        assert info["projects_shared_across_splits"] is True

    def test_pre_filtered_valid_key_normalised(self):
        """'valid' key in splits should be normalised to 'val' without error."""
        splits = self._make_pre_filtered_splits()
        assert "valid" in splits  # _make_pre_filtered_splits uses "valid"
        task = LinearProbeTask(targets=["precursor_charge"])
        result = task.run(
            emb=np.zeros((1, 32)),
            meta={},
            faiss_index=None,
            splits=splits,
            pre_filtered=True,
        )
        assert "precursor_charge" in result["targets"]

    def test_pre_filtered_without_flag_uses_project_disjoint(self):
        """pre_filtered=False should fall through to _run_project_disjoint, which
        raises ValueError when search_project key is absent from metadata.
        """
        splits = self._make_pre_filtered_splits()
        task = LinearProbeTask(
            targets=["precursor_charge"],
            use_project_split=True,
            train_samples=200,
            val_samples=50,
            test_samples=50,
        )
        # No search_project in metadata → project_disjoint_split raises ValueError
        with pytest.raises(ValueError, match="not found"):
            task.run(
                emb=np.zeros((1, 32)),
                meta={},
                faiss_index=None,
                splits=splits,
                pre_filtered=False,
            )


# ---------------------------------------------------------------------------
# Tests: _validate_splits
# ---------------------------------------------------------------------------


class TestValidateSplits:
    def test_valid_splits_pass(self):
        rng = np.random.RandomState(0)
        splits = {
            "train": (rng.randn(100, 16).astype(np.float32), {"charge": np.ones(100)}),
            "valid": (rng.randn(20, 16).astype(np.float32), {"charge": np.ones(20)}),
            "test": (rng.randn(20, 16).astype(np.float32), {"charge": np.ones(20)}),
        }
        task = LinearProbeTask(targets=["charge"])
        task._validate_splits(splits)  # should not raise

    def test_empty_splits_raises(self):
        task = LinearProbeTask(targets=["charge"])
        with pytest.raises(ValueError, match="empty"):
            task._validate_splits({})

    def test_non_tuple_value_raises(self):
        task = LinearProbeTask(targets=["charge"])
        with pytest.raises(ValueError, match="tuple"):
            task._validate_splits({"train": "not_a_tuple"})

    def test_non_2d_embeddings_raises(self):
        task = LinearProbeTask(targets=["charge"])
        with pytest.raises(ValueError, match="2D"):
            task._validate_splits(
                {
                    "train": (np.ones(100), {"charge": np.ones(100)}),
                }
            )


# ---------------------------------------------------------------------------
# Tests: NaN handling in _filter_splits_by_ptm
# ---------------------------------------------------------------------------


class TestFilterSplitsByPtmNaN:
    def test_nan_ptm_not_treated_as_positive(self):
        """NaN in ptm_present must not be counted as PTM-positive."""
        rng = np.random.RandomState(0)
        n = 200
        emb = rng.randn(n, 16).astype(np.float32)

        # Half NaN, half genuine 0/1
        ptm = np.array([float("nan")] * (n // 2) + [1] * (n // 4) + [0] * (n // 4))
        meta = {
            "ptm_present": ptm,
            "modification_class": np.array(["Oxidation"] * (n // 4) + ["Unmodified"] * (3 * n // 4), dtype=object),
        }

        probe_splits = {
            "train": {"embeddings": emb, "metadata": meta, "projects": []},
            "val": {"embeddings": emb, "metadata": meta, "projects": []},
            "test": {"embeddings": emb, "metadata": meta, "projects": []},
        }

        task = LinearProbeTask(targets=["modification_class"])
        filtered = task._filter_splits_by_ptm(probe_splits, min_samples=10)

        # Filtered splits should only contain genuine PTM-positive samples
        if filtered is not None:
            for split_name in ("train", "val", "test"):
                ptm_vals = filtered[split_name]["metadata"]["ptm_present"]
                # No NaN should be in filtered results
                for v in ptm_vals:
                    assert not (isinstance(v, float) and np.isnan(v)), f"NaN found in filtered {split_name} split"


# ---------------------------------------------------------------------------
# Tests: AUROC returns nan on failure
# ---------------------------------------------------------------------------


class TestAUROCFailure:
    def test_auroc_nan_when_proba_none(self):
        task = LinearProbeTask(targets=["x"])
        result = task._compute_auroc(
            y_true=np.array([0, 1, 0, 1]),
            y_proba=None,
            unique_classes=np.array([0, 1]),
        )
        import math

        assert math.isnan(result)

    def test_auroc_nan_when_single_class(self):
        """AUROC should return nan if test set has only one class."""
        task = LinearProbeTask(targets=["x"])
        # All same class — roc_auc_score will raise
        result = task._compute_auroc(
            y_true=np.array([0, 0, 0, 0]),
            y_proba=np.array([[0.9, 0.1]] * 4),
            unique_classes=np.array([0, 1]),
        )
        import math

        assert math.isnan(result)


# ---------------------------------------------------------------------------
# Tests: macro-F1 / AUROC denominators use the classes present in the test set
# ---------------------------------------------------------------------------


class TestMacroDenominatorConsistency:
    """A class present in train but absent from test must not enter the macro average.

    Regression test for the precursor-charge probe: a rare high-charge class survived into
    the probe's train split but not its test split. The model could still predict it, so
    sklearn's default ``average='macro'`` (labels = y_true UNION y_pred) folded an F1 of 0
    into the mean and divided by 8 instead of 7, deflating the score by a factor 7/8.
    """

    @staticmethod
    def _fit_data(n: int = 600, n_classes: int = 7, seed: int = 0) -> tuple:
        rng = np.random.default_rng(seed)
        y = rng.integers(0, n_classes, n)
        # separable-ish embeddings so the probe is not degenerate
        X = np.eye(n_classes)[y] * 4.0 + rng.normal(0, 1.0, (n, n_classes))
        return X, y

    def test_macro_f1_excludes_train_only_class(self) -> None:
        task = LinearProbeTask(targets=["charge"], c_values=[1.0])
        X, y = self._fit_data()
        n_classes = 7
        X_train, y_train = X[:400], y[:400]
        X_val, y_val = X[400:500], y[400:500]
        X_test, y_test = X[500:], y[500:]
        # inject a train-only class: present in train, absent from val/test
        extra = np.zeros((12, n_classes))
        extra[:, 0] = 8.0  # sits on top of class 0 so the model will sometimes predict it
        X_train = np.vstack([X_train, extra])
        y_train = np.concatenate([y_train, np.full(12, n_classes)])

        res = task._train_tune_eval_classification(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            target_field="charge",
        )
        assert "error" not in res, res
        # the reported macro-F1 is averaged over test-present classes only
        assert res["macro_f1_n_classes"] == len(np.unique(y_test))
        assert res["macro_f1_n_classes"] < len(np.unique(y_train))
        # per-class F1s are the same set the macro average is taken over
        assert len(res["per_class_metrics"]) == res["macro_f1_n_classes"]
        assert np.isclose(
            res["macro_f1"],
            np.mean([m["f1"] for m in res["per_class_metrics"].values()]),
        )
        # the old (inflated-denominator) value is retained for reconciliation and is never
        # larger than the corrected one
        assert res["macro_f1_over_predicted_labels"] <= res["macro_f1"] + 1e-9

    def test_macro_auroc_not_nan_with_train_only_class(self) -> None:
        """y_proba has one column per train class; dropping the test-absent column keeps
        roc_auc_score computable instead of silently yielding NaN.
        """
        task = LinearProbeTask(targets=["charge"])
        # 3 train classes, only 0 and 1 in test
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_proba = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.2, 0.7, 0.1],
                [0.6, 0.3, 0.1],
                [0.3, 0.6, 0.1],
                [0.1, 0.8, 0.1],
                [0.8, 0.1, 0.1],
            ]
        )
        naive = task._compute_auroc(y_true, y_proba, unique_classes=np.array([0, 1, 2]))
        fixed = task._compute_auroc(
            y_true,
            y_proba,
            unique_classes=np.array([0, 1, 2]),
            keep_classes=np.array([0, 1]),
        )
        assert np.isnan(naive)  # shape mismatch -> unusable
        assert not np.isnan(fixed)
        assert fixed == pytest.approx(1.0)
