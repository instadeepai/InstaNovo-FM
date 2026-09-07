"""Tests for the EVōC clustering evaluation task and shared clustering helpers."""
from __future__ import annotations
import pytest as _pytest

# evoc is an optional dependency (pip install 'instanovo-fm[clustering]'), so this
# module skips rather than fails when it is absent. The task itself also degrades
# gracefully; this keeps the test suite honest about which of the two happened.
_pytest.importorskip("evoc", reason="evoc is optional; install the clustering extra")


import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from instanovo_fm.eval.embed_eval_tasks import _clustering_common as cc
from instanovo_fm.eval.embed_eval_tasks.evoc_clustering import EVoCClusteringTask


# ---------------------------------------------------------------------------
# Synthetic data: 3 well-separated blobs with metadata correlated to the blob.
# ---------------------------------------------------------------------------
def _make_data(n=900, d=64, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 8, size=(3, d))
    assign = rng.integers(0, 3, size=n)
    emb = np.stack([centers[a] + rng.normal(0, 1, size=d) for a in assign]).astype(np.float32)
    meta = {
        "frag_type": np.array(["HCD", "CID", "HCID"])[assign].astype(object),
        "precursor_charge": (assign + 2).astype(np.int32),
        "precursor_mz": (400 + assign * 200 + rng.normal(0, 10, n)).astype(np.float32),
        "collision_energy": (25 + assign * 3 + rng.normal(0, 1, n)).astype(np.float32),
        "search_detector": np.array(["Orbitrap"] * n, dtype=object),
    }
    return emb, meta, assign


_FAST_EVOC = {"base_min_cluster_size": 20, "min_samples": 10}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
class TestClusteringHelpers:
    def test_build_descriptor_matrix_standardised(self):
        _, meta, _ = _make_data()
        X, names = cc.build_descriptor_matrix(meta, n=len(meta["precursor_mz"]))
        assert X.shape[0] == len(meta["precursor_mz"])
        assert X.shape[1] == len(names) >= 2
        # standardised columns: mean ~0, std ~1
        assert np.allclose(X.mean(axis=0), 0, atol=1e-4)
        assert np.allclose(X.std(axis=0), 1, atol=1e-4)

    def test_build_descriptor_matrix_drops_all_nan_feature(self):
        _, meta, _ = _make_data(n=100)
        meta["hyperscore"] = np.full(100, np.nan, dtype=np.float32)
        _, names = cc.build_descriptor_matrix(meta, n=100)
        assert "hyperscore" not in names

    def test_compute_cluster_enrichment(self):
        _, meta, assign = _make_data()
        enr = cc.compute_cluster_enrichment(assign, meta)
        # frag_type perfectly determined by the cluster -> purity 1.0
        frag = enr["categorical"]["frag_type"]["clusters"]
        assert all(c["purity"] == pytest.approx(1.0) for c in frag.values())
        # numeric effect sizes present for precursor_mz
        assert "precursor_mz" in enr["numeric"]

    def test_effect_matrix_shape(self):
        _, meta, assign = _make_data()
        enr = cc.compute_cluster_enrichment(assign, meta)
        mat, cids, feats = cc.enrichment_effect_matrix(enr)
        assert mat.shape == (len(cids), len(feats))

    def test_select_discriminative_field_picks_frag_type(self):
        _, meta, assign = _make_data()
        sel = cc.select_discriminative_field(
            assign, meta, ["frag_type", "search_detector"]
        )
        # frag_type separates the 3 groups; detector is constant -> frag_type wins
        assert sel is not None
        assert sel["field"] == "frag_type"
        assert sel["score"] > 0.5

    def test_hierarchy_navigation(self):
        layers = [np.array([0, 0, 1, 1, 2, 2])]
        tree = {(1, 0): [(0, 0), (0, 1), (0, 2)]}
        root = cc.find_root(tree)
        assert root == (1, 0)
        # virtual root spans all points
        assert cc.node_member_mask(root, layers, 6).all()
        # largest child has the most members (all equal here -> a valid child)
        assert cc.largest_child(root, tree, layers, 6) in tree[(1, 0)]


# ---------------------------------------------------------------------------
# EVoCClusteringTask
# ---------------------------------------------------------------------------
class TestEVoCClusteringTask:
    def test_runs_and_recovers_structure(self, tmp_path):
        emb, meta, _ = _make_data()
        task = EVoCClusteringTask(output_dir=str(tmp_path), max_samples=len(emb), evoc_params=_FAST_EVOC)
        res = task.run(emb, meta)
        assert res["skipped"] is False
        assert res["n_clusters"] >= 2
        # clusters should be pure w.r.t. frag_type (blobs == frag types)
        assert res["mean_purity_frag_type"] == pytest.approx(1.0, abs=0.05)
        assert len(res["save_paths"]) >= 1
        # per-cluster zoom + recolour-by-peptide-property figure is produced
        assert any("cluster_zoom_recolor" in p for p in res["save_paths"])
        loggable = task.get_loggable_metrics(res)
        assert "n_clusters" in loggable and "noise_fraction" in loggable

    def test_zoom_story_is_data_driven(self, tmp_path):
        emb, meta, _ = _make_data()
        task = EVoCClusteringTask(output_dir=str(tmp_path), max_samples=len(emb), evoc_params=_FAST_EVOC)
        res = task.run(emb, meta)
        story = res.get("zoom_story")
        assert story, "expected at least one zoom level"
        # top split should be explained by frag_type (the true generative factor)
        assert story[0]["selected_field"] == "frag_type"
        assert story[0]["nmi"] > 0.3

    def test_graceful_skip_without_evoc(self, tmp_path, monkeypatch):
        emb, meta, _ = _make_data(n=200)
        monkeypatch.setitem(sys.modules, "evoc", None)  # force ImportError on `import evoc`
        task = EVoCClusteringTask(output_dir=str(tmp_path), max_samples=len(emb))
        res = task.run(emb, meta)
        assert res["skipped"] is True
        assert "evoc" in res["reason"]
        assert task.get_loggable_metrics(res) == {}
