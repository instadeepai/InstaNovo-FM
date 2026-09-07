"""Tests for project-disjoint probe splitting."""

import numpy as np
import pytest

from instanovo_fm.eval.probe_splitting import (
    _build_project_index,
    _gather_project_indices,
    _is_valid_project,
    _normalise_split_keys,
    _project_capped_sample,
    _train_priority_assignment,
    project_disjoint_split,
)


# ---------------------------------------------------------------------------
# Helpers for building synthetic data
# ---------------------------------------------------------------------------


def _make_split(n: int, d: int, projects: list, seed: int = 0):
    """Create synthetic (embeddings, metadata) for a model split."""
    rng = np.random.RandomState(seed)
    emb = rng.randn(n, d).astype(np.float32)
    meta = {
        "search_project": np.array(projects[:n], dtype=object),
        "precursor_charge": rng.randint(1, 5, size=n),
    }
    return emb, meta


def _make_three_splits(
    project_counts: dict,
    d: int = 16,
):
    """Build train/val/test splits from project sample counts.

    Args:
        project_counts: {project_name: {"train": n, "val": n, "test": n}}
        d: embedding dimension
    """
    splits = {}
    for split_name in ("train", "valid", "test"):
        key = "val" if split_name == "valid" else split_name
        projects = []
        for proj, counts in project_counts.items():
            projects.extend([proj] * counts.get(key, 0))
        n = len(projects)
        rng = np.random.RandomState(hash(split_name) % 2**31)
        emb = rng.randn(n, d).astype(np.float32)
        meta = {
            "search_project": np.array(projects, dtype=object),
            "precursor_charge": rng.randint(1, 5, size=n),
        }
        splits[split_name] = (emb, meta)
    return splits


# ---------------------------------------------------------------------------
# Tests: _is_valid_project
# ---------------------------------------------------------------------------


class TestIsValidProject:
    def test_valid_strings(self):
        assert _is_valid_project("PXD001234") is True
        assert _is_valid_project("project_a") is True

    def test_invalid_values(self):
        assert _is_valid_project(None) is False
        assert _is_valid_project("") is False
        assert _is_valid_project("none") is False
        assert _is_valid_project("nan") is False
        assert _is_valid_project("unknown") is False
        assert _is_valid_project("null") is False

    def test_whitespace(self):
        assert _is_valid_project("  ") is False
        assert _is_valid_project(" PXD001234 ") is True


# ---------------------------------------------------------------------------
# Tests: _normalise_split_keys
# ---------------------------------------------------------------------------


class TestNormaliseSplitKeys:
    def test_valid_renamed(self):
        splits = {"train": "a", "valid": "b", "test": "c"}
        result = _normalise_split_keys(splits)
        assert "val" in result
        assert "valid" not in result
        assert result["train"] == "a"
        assert result["val"] == "b"

    def test_val_kept(self):
        splits = {"train": "a", "val": "b", "test": "c"}
        result = _normalise_split_keys(splits)
        assert result["val"] == "b"

    def test_val_and_valid_collision_raises(self):
        splits = {"train": "a", "val": "b", "valid": "c", "test": "d"}
        with pytest.raises(ValueError, match="Both 'val' and 'valid'"):
            _normalise_split_keys(splits)


# ---------------------------------------------------------------------------
# Tests: _build_project_index
# ---------------------------------------------------------------------------


class TestBuildProjectIndex:
    def test_basic_counts(self):
        splits = _make_three_splits({
            "PXD_A": {"train": 100, "val": 20, "test": 20},
            "PXD_B": {"train": 50, "val": 10, "test": 10},
        })
        # Normalise to use "val" key
        splits_norm = _normalise_split_keys(splits)
        info, dropped = _build_project_index(splits_norm, "search_project", 5)

        assert dropped == 0
        assert "PXD_A" in info
        assert "PXD_B" in info
        assert info["PXD_A"]["train"] == 100
        assert info["PXD_A"]["val"] == 20
        assert info["PXD_B"]["test"] == 10

    def test_min_project_samples_filter(self):
        splits = _make_three_splits({
            "PXD_A": {"train": 100, "val": 20, "test": 20},
            "PXD_tiny": {"train": 2, "val": 0, "test": 0},
        })
        splits_norm = _normalise_split_keys(splits)
        info, _ = _build_project_index(splits_norm, "search_project", 5)

        assert "PXD_A" in info
        assert "PXD_tiny" not in info

    def test_missing_project_key(self):
        emb = np.zeros((10, 4))
        meta = {"precursor_charge": np.ones(10)}
        splits = {"train": (emb, meta), "val": (emb, meta), "test": (emb, meta)}
        info, dropped = _build_project_index(splits, "search_project", 1)

        assert len(info) == 0
        assert dropped == 30  # all samples dropped

    def test_invalid_project_labels_dropped(self):
        emb = np.zeros((5, 4))
        meta = {
            "search_project": np.array(["PXD_A", None, "nan", "PXD_A", ""], dtype=object)
        }
        splits = {"train": (emb, meta), "val": (np.zeros((0, 4)), {}), "test": (np.zeros((0, 4)), {})}
        info, dropped = _build_project_index(splits, "search_project", 1)

        assert info["PXD_A"]["train"] == 2
        assert dropped == 3  # None, nan, ""

    def test_unknown_split_key_ignored(self):
        """Unknown split keys (e.g. 'holdout') must not raise KeyError."""
        emb = np.zeros((10, 4))
        meta = {"search_project": np.array(["PXD_A"] * 10, dtype=object)}
        splits = {
            "train": (emb, meta),
            "val": (np.zeros((0, 4)), {}),
            "test": (np.zeros((0, 4)), {}),
            "holdout": (emb, meta),  # unexpected key
        }
        info, dropped = _build_project_index(splits, "search_project", 1)
        # holdout samples are silently ignored, train counts are correct
        assert info["PXD_A"]["train"] == 10
        assert "holdout" not in info.get("PXD_A", {})


# ---------------------------------------------------------------------------
# Tests: _train_priority_assignment
# ---------------------------------------------------------------------------


class TestTrainPriorityAssignment:
    def test_basic_assignment(self):
        project_info = {
            "A": {"train": 50000, "val": 5000, "test": 5000},
            "B": {"train": 30000, "val": 3000, "test": 3000},
            "C": {"train": 20000, "val": 2000, "test": 2000},
            "D": {"train": 10000, "val": 6000, "test": 1000},
            "E": {"train": 5000, "val": 1000, "test": 6000},
        }
        assignment = _train_priority_assignment(
            project_info, val_target=5000, test_target=5000, random_state=42
        )

        # All projects assigned to exactly one split
        all_assigned = set(assignment["train"]) | set(assignment["val"]) | set(assignment["test"])
        assert all_assigned == set(project_info.keys())

        # No overlap
        assert len(set(assignment["train"]) & set(assignment["val"])) == 0
        assert len(set(assignment["train"]) & set(assignment["test"])) == 0
        assert len(set(assignment["val"]) & set(assignment["test"])) == 0

    def test_val_gets_enough(self):
        project_info = {
            "A": {"train": 1000, "val": 8000, "test": 1000},
            "B": {"train": 1000, "val": 5000, "test": 1000},
            "C": {"train": 1000, "val": 100, "test": 1000},
        }
        assignment = _train_priority_assignment(
            project_info, val_target=7000, test_target=500, random_state=42
        )

        # A has the most val samples, should be assigned to val first
        assert "A" in assignment["val"]
        val_total = sum(project_info[p]["val"] for p in assignment["val"])
        assert val_total >= 7000


# ---------------------------------------------------------------------------
# Tests: _project_capped_sample
# ---------------------------------------------------------------------------


class TestProjectCappedSample:
    def test_cap_enforced(self):
        n = 1000
        projects = np.array(["A"] * 800 + ["B"] * 200, dtype=object)
        meta = {"search_project": projects}
        indices = np.arange(n)
        rng = np.random.RandomState(42)

        sampled = _project_capped_sample(
            meta, indices, "search_project",
            target_n=500, max_per_project_frac=0.15, rng=rng,
        )

        # Check cap: project A should not have more than 500 * 0.15 = 75
        a_count = sum(1 for i in sampled if projects[i] == "A")
        assert a_count <= 75

    def test_small_pool_returns_all(self):
        n = 50
        projects = np.array(["A"] * 50, dtype=object)
        meta = {"search_project": projects}
        indices = np.arange(n)
        rng = np.random.RandomState(42)

        sampled = _project_capped_sample(
            meta, indices, "search_project",
            target_n=1000, max_per_project_frac=0.15, rng=rng,
        )

        assert len(sampled) == 50

    def test_target_approximately_hit(self):
        # 10 projects, each with 1000 samples, target=5000, cap=15% (=750 per project)
        # 10 * 750 = 7500 > 5000, so scaling down to hit target
        n = 10000
        projects = np.array(
            sum(([f"P{i}"] * 1000 for i in range(10)), []),
            dtype=object,
        )
        meta = {"search_project": projects}
        indices = np.arange(n)
        rng = np.random.RandomState(42)

        sampled = _project_capped_sample(
            meta, indices, "search_project",
            target_n=5000, max_per_project_frac=0.15, rng=rng,
        )

        # Should be close to target (within rounding)
        assert abs(len(sampled) - 5000) <= 10

    def test_exact_target_hit_no_undershoot(self):
        """Second-pass sampling must hit exactly target_n, not undershoot."""
        # Many projects with uneven sizes that could cause rounding undershoot
        rng_data = np.random.RandomState(0)
        sizes = [237, 183, 419, 301, 158, 92, 444, 371, 267, 128]
        projects = np.array(
            sum(([f"P{i}"] * s for i, s in enumerate(sizes)), []),
            dtype=object,
        )
        meta = {"search_project": projects}
        indices = np.arange(len(projects))
        rng = np.random.RandomState(99)

        # Cap at 15%, target 800 — forces second pass
        sampled = _project_capped_sample(
            meta, indices, "search_project",
            target_n=800, max_per_project_frac=0.15, rng=rng,
        )

        assert len(sampled) == 800


# ---------------------------------------------------------------------------
# Tests: project_disjoint_split (integration)
# ---------------------------------------------------------------------------


class TestProjectDisjointSplit:
    def test_basic_split(self):
        splits = _make_three_splits({
            "PXD_A": {"train": 5000, "val": 1000, "test": 1000},
            "PXD_B": {"train": 3000, "val": 800, "test": 800},
            "PXD_C": {"train": 2000, "val": 500, "test": 500},
            "PXD_D": {"train": 4000, "val": 1200, "test": 1200},
            "PXD_E": {"train": 6000, "val": 2000, "test": 2000},
        })

        result = project_disjoint_split(
            splits,
            train_samples=5000,
            val_samples=1000,
            test_samples=1000,
            random_state=42,
        )

        # Check structure
        assert "train" in result
        assert "val" in result
        assert "test" in result
        assert "assignment_info" in result

        # Check project disjointness
        train_proj = set(result["train"]["projects"])
        val_proj = set(result["val"]["projects"])
        test_proj = set(result["test"]["projects"])
        assert len(train_proj & val_proj) == 0
        assert len(train_proj & test_proj) == 0
        assert len(val_proj & test_proj) == 0

        # All projects accounted for
        all_proj = train_proj | val_proj | test_proj
        assert len(all_proj) == 5

        # Embeddings have correct shape
        d = 16
        for split_name in ("train", "val", "test"):
            emb = result[split_name]["embeddings"]
            assert emb.ndim == 2
            assert emb.shape[1] == d

    def test_missing_project_key_raises(self):
        emb = np.zeros((10, 4))
        meta = {"precursor_charge": np.ones(10)}
        splits = {
            "train": (emb, meta),
            "valid": (emb, meta),
            "test": (emb, meta),
        }

        with pytest.raises(ValueError, match="not found"):
            project_disjoint_split(splits)

    def test_valid_alias_accepted(self):
        """Test that 'valid' key is accepted as alias for 'val'."""
        splits = _make_three_splits({
            "PXD_A": {"train": 100, "val": 50, "test": 50},
        })
        # _make_three_splits uses "valid" key
        assert "valid" in splits

        result = project_disjoint_split(
            splits,
            train_samples=50,
            val_samples=20,
            test_samples=20,
            random_state=42,
        )

        # Should work without error
        assert result["val"]["embeddings"].shape[0] > 0 or result["train"]["embeddings"].shape[0] > 0

    def test_metadata_sliced_correctly(self):
        splits = _make_three_splits({
            "PXD_A": {"train": 200, "val": 50, "test": 50},
            "PXD_B": {"train": 200, "val": 50, "test": 50},
        })

        result = project_disjoint_split(
            splits,
            train_samples=100,
            val_samples=30,
            test_samples=30,
            random_state=42,
        )

        for split_name in ("train", "val", "test"):
            n = len(result[split_name]["embeddings"])
            for key, arr in result[split_name]["metadata"].items():
                assert len(arr) == n, f"{split_name}/{key} length mismatch"


class TestTrainTargetEscalation:
    """Regression tests for the bug where train was starved when use_project_split=True.

    Root cause: at max_per_project_frac=0.15, val/test consumed many projects
    (each contributing ≤15% of the target), leaving train with only "orphan"
    projects (0 val/0 test samples) that had very few train samples total.

    Fix: pass train_target to _train_priority_assignment so the escalation loop
    only breaks when all three targets (val, test, train) are satisfied.
    """

    def _build_lcfm_like_project_info(self) -> dict:
        """Simulate LCFM-like project distribution.

        Many projects with balanced val/test data.  At frac=0.15 each project
        contributes at most 0.15*5000=750 val samples, so ≥7 projects are needed
        to fill val — those projects are then unavailable for train.  With enough
        such projects consumed by val/test, train is left with only orphans.
        """
        project_info = {}
        # 20 "rich" projects: lots of train + val + test data
        # At frac=0.15, each contributes 750 to val → need ≥7 for val_target=5000
        for i in range(20):
            project_info[f"RICH_{i:02d}"] = {
                "train": 5000,
                "val": 5000,
                "test": 5000,
            }
        # 5 "orphan" projects: train only, no val/test
        # These are the only ones left for train at frac=0.15 if too many rich
        # projects are consumed by val/test
        for i in range(5):
            project_info[f"ORPHAN_{i}"] = {
                "train": 200,
                "val": 0,
                "test": 0,
            }
        return project_info

    def test_without_train_target_train_may_be_underfilled(self):
        """Without train_target, escalation stops as soon as val+test are met.

        At frac=0.15: val needs ≥7 rich projects, test needs ≥7 rich projects
        → up to 14 rich projects consumed, leaving only orphans for train
        → total orphan train samples = 5 * 200 = 1000 << train_target=100_000.
        """
        project_info = self._build_lcfm_like_project_info()
        assignment = _train_priority_assignment(
            project_info,
            val_target=5_000,
            test_target=5_000,
            random_state=42,
            max_per_project_frac=0.15,
            train_target=0,  # No train feasibility check
        )
        train_samples = sum(project_info[p]["train"] for p in assignment["train"])
        # Confirm the bug: train is severely under-filled
        assert train_samples < 100_000, (
            f"Expected train to be under-filled without train_target, got {train_samples}"
        )

    def test_with_train_target_escalation_fills_train(self):
        """With train_target, the loop escalates until train is also satisfied.

        At frac=0.5: val needs ≥2 rich projects, test needs ≥2 rich projects
        → only 4 rich projects consumed → 16 rich projects go to train
        → train samples = 16 * 5000 = 80_000+, which meets train_target=50_000.
        """
        project_info = self._build_lcfm_like_project_info()
        assignment = _train_priority_assignment(
            project_info,
            val_target=5_000,
            test_target=5_000,
            random_state=42,
            max_per_project_frac=0.15,
            train_target=50_000,
        )
        train_samples = sum(project_info[p]["train"] for p in assignment["train"])
        assert train_samples >= 50_000, (
            f"Expected train to be filled after escalation, got {train_samples}"
        )
        # Val and test targets must still be met
        val_samples = sum(project_info[p]["val"] for p in assignment["val"])
        test_samples = sum(project_info[p]["test"] for p in assignment["test"])
        assert val_samples >= 5_000
        assert test_samples >= 5_000

    def test_assignment_info_reports_correct_sizes(self):
        """project_disjoint_split logs n_samples_per_split in assignment_info."""
        # Build splits with enough data so all targets can be met
        project_counts = {}
        for i in range(10):
            project_counts[f"P{i:02d}"] = {"train": 3000, "val": 1000, "test": 1000}

        splits = _make_three_splits(project_counts, d=8)

        result = project_disjoint_split(
            splits,
            train_samples=10_000,
            val_samples=2_000,
            test_samples=2_000,
            max_per_project_frac=0.15,
            random_state=42,
        )

        info = result["assignment_info"]
        sizes = info["n_samples_per_split"]

        # All three sizes must be reported
        assert "train" in sizes
        assert "val" in sizes
        assert "test" in sizes

        # Sizes must be positive
        assert sizes["train"] > 0, f"Train size {sizes['train']} is zero"
        assert sizes["val"] > 0, f"Val size {sizes['val']} is zero"
        assert sizes["test"] > 0, f"Test size {sizes['test']} is zero"

        # Sizes should approach the targets (within available data)
        assert sizes["val"] >= min(2_000, sum(p["val"] for p in project_counts.values()))
        assert sizes["test"] >= min(2_000, sum(p["test"] for p in project_counts.values()))
