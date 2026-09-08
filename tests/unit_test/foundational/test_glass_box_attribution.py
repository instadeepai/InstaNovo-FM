"""Tests for the Glass Box UMAP feature-attribution evaluation task."""
from __future__ import annotations
import pytest as _pytest

# glass_box_umap is an optional dependency (pip install 'instanovo-fm[interpret]'), so this
# module skips rather than fails when it is absent. The task itself also degrades
# gracefully; this keeps the test suite honest about which of the two happened.
_pytest.importorskip("glass_box_umap", reason="glass_box_umap is optional; install the interpret extra")


import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")

from instanovo_fm.eval.embed_eval_tasks.glass_box_attribution import GlassBoxAttributionTask


def _make_data(n=500, d=64, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 8, size=(3, d))
    assign = rng.integers(0, 3, size=n)
    emb = np.stack([centers[a] + rng.normal(0, 1, size=d) for a in assign]).astype(np.float32)
    meta = {
        "frag_type": np.array(["HCD", "CID", "HCID"])[assign].astype(object),
        "precursor_charge": (assign + 2).astype(np.int32),
        # precursor_mz strongly tracks the blob -> should be a high-importance descriptor
        "precursor_mz": (400 + assign * 400 + rng.normal(0, 5, n)).astype(np.float32),
        "collision_energy": (25 + rng.normal(0, 1, n)).astype(np.float32),
        "retention_time": rng.normal(1000, 200, n).astype(np.float32),
        "hydrophobicity": rng.normal(0, 1, n).astype(np.float32),
    }
    return emb, meta


_FAST_GLASSBOX = {"n_neighbors": 15, "epochs": 6, "n_components": 2}
_FAST_EVOC = {"base_min_cluster_size": 20, "min_samples": 10}


class TestGlassBoxAttributionTask:
    def test_runs_with_exact_reconstruction(self, tmp_path):
        emb, meta = _make_data()
        task = GlassBoxAttributionTask(
            output_dir=str(tmp_path), max_samples=len(emb),
            glassbox_params=_FAST_GLASSBOX, evoc_params=_FAST_EVOC,
        )
        res = task.run(emb, meta)
        assert res["skipped"] is False
        assert res["n_features"] >= 2
        # Glass Box's defining property: contributions sum to the embedding exactly.
        assert res["reconstruction_residual"] < 1e-3
        # Per-cluster attribution present (EVōC bridge succeeded).
        assert res["per_cluster_importance"] is not None
        # Figures: attribution set + metadata views.
        assert len(res["save_paths"]) > 3
        loggable = task.get_loggable_metrics(res)
        assert "top_feature_importance" in loggable

    def test_metadata_views_toggle_off(self, tmp_path):
        import os

        emb, meta = _make_data()
        task = GlassBoxAttributionTask(
            output_dir=str(tmp_path), max_samples=len(emb),
            glassbox_params=_FAST_GLASSBOX, evoc_params=_FAST_EVOC,
            create_metadata_views=False,
        )
        res = task.run(emb, meta)
        assert res["skipped"] is False
        # No figure should live inside a "metadata_views" subdirectory, and the dir
        # itself should not be created. (Check the parent folder name precisely — the
        # tmp_path name itself can contain the test-function substring.)
        assert not any(os.path.basename(os.path.dirname(p)) == "metadata_views"
                       for p in res["save_paths"])
        assert not (tmp_path / "glassbox_figs" / "metadata_views").exists()

    def test_skip_on_insufficient_descriptors(self, tmp_path):
        emb, _ = _make_data(n=200)
        meta = {"frag_type": np.array(["HCD"] * 200, dtype=object)}  # no numeric descriptors
        task = GlassBoxAttributionTask(output_dir=str(tmp_path), max_samples=200)
        res = task.run(emb, meta)
        assert res["skipped"] is True
        assert "descriptor" in res["reason"]

    def test_graceful_skip_without_glassbox(self, tmp_path, monkeypatch):
        emb, meta = _make_data(n=200)
        monkeypatch.setitem(sys.modules, "glass_box_umap", None)  # force ImportError
        task = GlassBoxAttributionTask(output_dir=str(tmp_path), max_samples=200)
        res = task.run(emb, meta)
        assert res["skipped"] is True
        assert "glass-box-umap" in res["reason"]
        assert task.get_loggable_metrics(res) == {}
