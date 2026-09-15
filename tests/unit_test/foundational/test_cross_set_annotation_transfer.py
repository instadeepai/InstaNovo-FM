"""Tests for cross-set annotation transfer task."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from instanovo_fm.eval.embed_eval_tasks import TASK_REGISTRY
from instanovo_fm.eval.embed_eval_tasks.cross_set_annotation_transfer import (
    CrossSetAnnotationTransferTask,
)
from instanovo_fm.eval.spectrum_metrics.sequencing import compute_sequencing_metrics


def _fixture() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    emb = np.array(
        [
            [1.00, 0.00],  # acfm query 0
            [0.90, 0.10],  # acfm query 1
            [0.98, 0.02],  # lcfm_valid library (close to query 0)
            [0.10, 0.95],  # lcfm_valid library (far)
            [0.89, 0.11],  # lcfm_valid library (close to query 1)
            [0.20, 0.80],  # lcfm_train (ignored by filter)
        ],
        dtype=np.float32,
    )
    meta = {
        "search_tier": np.array(
            ["acfm", "acfm", "lcfm_valid", "lcfm_valid", "lcfm_valid", "lcfm_train"],
            dtype=object,
        ),
        "peptides": np.array(
            ["", "", "PEPTIDEA", "PEPTIDEK", "PEPTIDEB", "PEPTIDETRAIN"],
            dtype=object,
        ),
        "unmodified_peptide": np.array(
            ["", "", "PEPTIDEA", "PEPTIDEK", "PEPTIDEB", "PEPTIDETRAIN"],
            dtype=object,
        ),
        "usi": np.array(
            [
                "mzspec:PXD:runA:scan:1",
                "mzspec:PXD:runA:scan:2",
                "mzspec:PXD:runL:scan:1",
                "mzspec:PXD:runL:scan:2",
                "mzspec:PXD:runL:scan:3",
                "mzspec:PXD:runT:scan:1",
            ],
            dtype=object,
        ),
        "overlap_id": np.array(
            [
                "runA:1",
                "runA:2",
                "runL:1",
                "runL:2",
                "runL:3",
                "runT:1",
            ],
            dtype=object,
        ),
    }
    return emb, meta


def test_task_is_registered() -> None:
    assert TASK_REGISTRY["crosssetannotationtransfertask"] is CrossSetAnnotationTransferTask
    assert TASK_REGISTRY["cross_set_annotation_transfer"] is CrossSetAnnotationTransferTask


def test_normalise_filter_values_handles_omegaconf_listconfig() -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"search_tier": ["acfm", "lcfm_valid"]})
    allowed = CrossSetAnnotationTransferTask._normalise_filter_values(cfg.search_tier)
    assert allowed == {"acfm", "lcfm_valid"}


def test_cross_set_run_outputs_candidates(tmp_path: Path) -> None:
    emb, meta = _fixture()
    task = CrossSetAnnotationTransferTask(
        query_filter={"search_tier": "acfm"},
        library_filter={"search_tier": "lcfm_valid"},
        k_values=[1, 2],
        output_dir=str(tmp_path),
        save_candidates_csv=True,
        save_matrix_artifact=False,
        topk_only=True,
        compute_evidence_metrics=False,
    )

    results = task.run(emb, meta)

    assert "error" not in results
    assert results["num_queries"] == 2
    assert results["num_library"] == 3
    assert "candidates_csv" in results
    assert "plot_paths" in results
    for key in (
        "retrieval_curves",
        "margin_distribution",
        "top1_score_distribution",
        "topk_heatmap",
        "top_peptides",
        "query_library_umap",
    ):
        assert Path(results["plot_paths"][key]).exists()

    candidates = pl.read_csv(results["candidates_csv"])
    assert candidates.height == 4  # 2 queries * top2
    top1_q0 = candidates.filter((pl.col("query_index") == 0) & (pl.col("rank") == 1))
    assert top1_q0["library_peptide"][0] == "PEPTIDEA"
    assert "embedding_score" in candidates.columns


def test_cross_set_run_with_evidence_metrics_generates_evidence_plots(tmp_path: Path) -> None:
    """Block A (MCP-independent) evidence metrics should attach to candidates and plot."""
    emb, meta = _fixture()
    n = len(meta["usi"])
    rng = np.random.RandomState(0)
    meta["mz_array"] = np.array([np.sort(rng.uniform(100, 1000, size=8)) for _ in range(n)], dtype=object)
    meta["intensity_array"] = np.array([rng.uniform(10, 1000, size=8) for _ in range(n)], dtype=object)
    meta["precursor_mz"] = np.array([500.0] * n, dtype=np.float64)
    meta["precursor_charge"] = np.array([2] * n, dtype=np.int64)

    task = CrossSetAnnotationTransferTask(
        query_filter={"search_tier": "acfm"},
        library_filter={"search_tier": "lcfm_valid"},
        k_values=[1, 2],
        output_dir=str(tmp_path),
        save_candidates_csv=True,
        save_matrix_artifact=False,
        topk_only=True,
        compute_evidence_metrics=True,
        score_blocks=["A"],
        num_workers=1,
    )

    results = task.run(emb, meta)

    assert "error" not in results
    candidates = pl.read_csv(results["candidates_csv"])
    assert "q_obs__lib_obs__cosine_similarity" in candidates.columns
    assert "evidence_block_a_cosine_distribution" in results["plot_paths"]
    assert Path(results["plot_paths"]["evidence_block_a_cosine_distribution"]).exists()
    assert "evidence_rank_curves" in results["plot_paths"]
    assert Path(results["plot_paths"]["evidence_rank_curves"]).exists()


def test_topk_vectorized_matches_full_matrix() -> None:
    emb, meta = _fixture()
    task = CrossSetAnnotationTransferTask(
        query_filter={"search_tier": "acfm"},
        library_filter={"search_tier": "lcfm_valid"},
        k_values=[1, 2],
        compute_evidence_metrics=False,
        topk_only=True,
    )
    topk_results = task.run(emb, meta)

    full_task = CrossSetAnnotationTransferTask(
        query_filter={"search_tier": "acfm"},
        library_filter={"search_tier": "lcfm_valid"},
        k_values=[1, 2],
        compute_evidence_metrics=False,
        topk_only=False,
        save_matrix_artifact=False,
    )
    full_results = full_task.run(emb, meta)
    assert topk_results["mean_top1_score"] == pytest.approx(full_results["mean_top1_score"])


def test_kostas_protocol_file_pair_top_peptides(tmp_path: Path) -> None:
    """Kostas: ACFM−LCFM diff queries; library anchors = top-N peptides from LCFM file."""
    from instanovo_fm.eval.cross_set_dataset import build_kostas_protocol_parquet

    acfm_dir = tmp_path / "acfm"
    lcfm_dir = tmp_path / "lcfm"
    acfm_dir.mkdir()
    lcfm_dir.mkdir()

    # Same basename pair. Scans 1 and 2 overlap; scan 3 is ACFM-only (diff).
    pl.DataFrame(
        {
            "mz": [[100.0, 200.0], [110.0, 210.0], [120.0, 220.0]],
            "intensity": [[1000.0, 500.0], [900.0, 400.0], [800.0, 300.0]],
            "scan": [1, 2, 3],
            "precursor_mz": [500.0, 501.0, 502.0],
            "precursor_charge": [2, 2, 2],
        }
    ).write_ipc(acfm_dir / "runA.mzML.ipc")

    pl.DataFrame(
        {
            "mz": [[100.0, 200.0], [110.0, 210.0], [130.0, 230.0], [140.0, 240.0]],
            "intensity": [[1000.0, 500.0], [900.0, 400.0], [700.0, 300.0], [600.0, 200.0]],
            "scan": [1, 2, 10, 11],
            "precursor_mz": [500.0, 501.0, 510.0, 511.0],
            "precursor_charge": [2, 2, 2, 2],
            "peptide": ["AAA", "BBB", "AAA", "CCC"],
            "modified_peptide": [None, "B[1]BB", "AAA", None],
            "hyperscore": [10.0, 50.0, 40.0, 5.0],
            "probability": [0.9, 0.99, 0.95, 0.5],
            "expectation": [0.1, 0.001, 0.01, 0.5],
        }
    ).write_ipc(lcfm_dir / "runA.mzML.ipc")

    output = tmp_path / "kostas.parquet"
    summary = build_kostas_protocol_parquet(
        acfm_dir=acfm_dir,
        lcfm_dir=lcfm_dir,
        output_path=output,
        overlap_key="scan",
        project_id="PXDTEST",
        top_n_peptides=2,
        file_name="runA.mzML.ipc",
    )

    assert summary["num_pairs"] == 1
    assert summary["num_queries"] == 1  # only scan 3
    assert summary["pairs"][0]["top_peptides"][:2]  # non-empty
    # AAA appears twice with high scores → selected; BBB once with highest single hyperscore
    top = summary["pairs"][0]["top_peptides"]
    assert "AAA" in top
    assert "B[1]BB" in top or "BBB" in top

    combined = pl.read_parquet(output)
    assert set(combined["search_tier"].unique().to_list()) == {"acfm", "lcfm"}
    anchors = combined.filter(pl.col("is_selected_anchor") == "1")
    assert anchors.height >= 1
    # No null/"None" sequence on selected anchors after coalesce
    assert anchors.filter(pl.col("sequence").is_null() | (pl.col("sequence") == "")).height == 0


def test_kostas_protocol_query_library_overlap_ids_are_disjoint(tmp_path: Path) -> None:
    """No ACFM query row may share overlap_id (scan) with any LCFM anchor row.

    Otherwise a query would trivially retrieve its own spectrum from the library.
    Checked here at the dataset-builder level (before the task-level fail_on_overlap
    re-check even runs), for both a capped top-N library and the uncapped "all" library.
    """
    from instanovo_fm.eval.cross_set_dataset import build_kostas_protocol_parquet

    acfm_dir = tmp_path / "acfm"
    lcfm_dir = tmp_path / "lcfm"
    acfm_dir.mkdir()
    lcfm_dir.mkdir()

    # Scans 1 and 2 overlap with LCFM; scan 3 is ACFM-only (diff).
    pl.DataFrame(
        {
            "mz": [[100.0, 200.0], [110.0, 210.0], [120.0, 220.0]],
            "intensity": [[1000.0, 500.0], [900.0, 400.0], [800.0, 300.0]],
            "scan": [1, 2, 3],
            "precursor_mz": [500.0, 501.0, 502.0],
            "precursor_charge": [2, 2, 2],
        }
    ).write_ipc(acfm_dir / "runA.mzML.ipc")

    pl.DataFrame(
        {
            "mz": [[100.0, 200.0], [110.0, 210.0], [130.0, 230.0], [140.0, 240.0]],
            "intensity": [[1000.0, 500.0], [900.0, 400.0], [700.0, 300.0], [600.0, 200.0]],
            "scan": [1, 2, 10, 11],
            "precursor_mz": [500.0, 501.0, 510.0, 511.0],
            "precursor_charge": [2, 2, 2, 2],
            "peptide": ["AAA", "BBB", "AAA", "CCC"],
            "modified_peptide": [None, "B[1]BB", "AAA", None],
            "hyperscore": [10.0, 50.0, 40.0, 5.0],
            "probability": [0.9, 0.99, 0.95, 0.5],
            "expectation": [0.1, 0.001, 0.01, 0.5],
        }
    ).write_ipc(lcfm_dir / "runA.mzML.ipc")

    for top_n_peptides in (2, None):  # capped library, and "all" (uncapped) library
        output = tmp_path / f"kostas_top_{top_n_peptides}.parquet"
        build_kostas_protocol_parquet(
            acfm_dir=acfm_dir,
            lcfm_dir=lcfm_dir,
            output_path=output,
            overlap_key="scan",
            project_id="PXDTEST",
            top_n_peptides=top_n_peptides,
            file_name="runA.mzML.ipc",
        )
        combined = pl.read_parquet(output)
        query_overlap_ids = set(combined.filter(pl.col("search_tier") == "acfm")["overlap_id"].to_list())
        anchor_overlap_ids = set(
            combined.filter(pl.col("is_selected_anchor") == "1")["overlap_id"].to_list()
        )
        assert query_overlap_ids, f"top_n_peptides={top_n_peptides}: expected non-empty query diff"
        assert anchor_overlap_ids, f"top_n_peptides={top_n_peptides}: expected non-empty library anchors"
        assert query_overlap_ids.isdisjoint(anchor_overlap_ids), (
            f"top_n_peptides={top_n_peptides}: query/library overlap_id leakage: "
            f"{query_overlap_ids & anchor_overlap_ids}"
        )


def test_kostas_protocol_top_n_peptides_all_selects_every_peptide(tmp_path: Path) -> None:
    """top_n_peptides=None ("all") must anchor every distinct LCFM peptide, no cap."""
    from instanovo_fm.eval.cross_set_dataset import build_kostas_protocol_parquet

    acfm_dir = tmp_path / "acfm"
    lcfm_dir = tmp_path / "lcfm"
    acfm_dir.mkdir()
    lcfm_dir.mkdir()

    pl.DataFrame(
        {
            "mz": [[100.0, 200.0]],
            "intensity": [[1000.0, 500.0]],
            "scan": [3],
            "precursor_mz": [502.0],
            "precursor_charge": [2],
        }
    ).write_ipc(acfm_dir / "runA.mzML.ipc")

    # Three distinct peptides in the LCFM file; top_n=2 would drop one of them.
    pl.DataFrame(
        {
            "mz": [[100.0, 200.0], [110.0, 210.0], [130.0, 230.0]],
            "intensity": [[1000.0, 500.0], [900.0, 400.0], [700.0, 300.0]],
            "scan": [1, 2, 10],
            "precursor_mz": [500.0, 501.0, 510.0],
            "precursor_charge": [2, 2, 2],
            "peptide": ["AAA", "BBB", "CCC"],
            "modified_peptide": [None, None, None],
            "hyperscore": [10.0, 50.0, 5.0],
            "probability": [0.9, 0.99, 0.5],
            "expectation": [0.1, 0.001, 0.5],
        }
    ).write_ipc(lcfm_dir / "runA.mzML.ipc")

    output = tmp_path / "kostas_all.parquet"
    summary = build_kostas_protocol_parquet(
        acfm_dir=acfm_dir,
        lcfm_dir=lcfm_dir,
        output_path=output,
        overlap_key="scan",
        project_id="PXDTEST",
        top_n_peptides=None,
        file_name="runA.mzML.ipc",
    )

    top = summary["pairs"][0]["top_peptides"]
    assert set(top) == {"AAA", "BBB", "CCC"}

    combined = pl.read_parquet(output)
    anchors = combined.filter(pl.col("is_selected_anchor") == "1")
    assert set(anchors["sequence"].unique().to_list()) == {"AAA", "BBB", "CCC"}


def test_task_exclude_overlap_drops_queries_not_library() -> None:
    emb = np.eye(4, dtype=np.float32)
    meta = {
        "search_tier": np.array(["acfm", "acfm", "lcfm_valid", "lcfm_valid"], dtype=object),
        "peptides": np.array(["", "", "PEPA", "PEPB"], dtype=object),
        "unmodified_peptide": np.array(["", "", "PEPA", "PEPB"], dtype=object),
        "usi": np.array(["q1", "q2", "l1", "l2"], dtype=object),
        "overlap_id": np.array(["same", "only_q", "same", "only_l"], dtype=object),
    }
    task = CrossSetAnnotationTransferTask(
        query_filter={"search_tier": "acfm"},
        library_filter={"search_tier": "lcfm_valid"},
        k_values=[1],
        exclude_overlap=True,
        overlap_key="overlap_id",
        compute_evidence_metrics=False,
        save_plots=False,
    )
    results = task.run(emb, meta)
    assert results["num_queries"] == 1  # q1 dropped, q2 kept
    assert results["num_library"] == 2  # library untouched
    assert results["overlap_check"]["n_overlapping_queries"] == 0  # re-check confirms the drop worked


def test_task_fails_hard_when_overlap_not_excluded_upstream() -> None:
    """fail_on_overlap=True (default) must catch a broken upstream diff, not silently proceed.

    exclude_overlap=False means "the caller promises the diff was already applied upstream" --
    this simulates that promise being broken (q1 still shares overlap_id with library row l1).
    """
    emb = np.eye(4, dtype=np.float32)
    meta = {
        "search_tier": np.array(["acfm", "acfm", "lcfm_valid", "lcfm_valid"], dtype=object),
        "peptides": np.array(["", "", "PEPA", "PEPB"], dtype=object),
        "unmodified_peptide": np.array(["", "", "PEPA", "PEPB"], dtype=object),
        "usi": np.array(["q1", "q2", "l1", "l2"], dtype=object),
        "overlap_id": np.array(["same", "only_q", "same", "only_l"], dtype=object),
    }
    task = CrossSetAnnotationTransferTask(
        query_filter={"search_tier": "acfm"},
        library_filter={"search_tier": "lcfm_valid"},
        k_values=[1],
        exclude_overlap=False,
        overlap_key="overlap_id",
        compute_evidence_metrics=False,
        save_plots=False,
    )
    results = task.run(emb, meta)
    assert "error" in results
    assert "overlap" in results["error"].lower()
    assert results["overlap_check"]["n_overlapping_queries"] == 1


def test_task_fail_on_overlap_false_drops_instead_of_erroring() -> None:
    emb = np.eye(4, dtype=np.float32)
    meta = {
        "search_tier": np.array(["acfm", "acfm", "lcfm_valid", "lcfm_valid"], dtype=object),
        "peptides": np.array(["", "", "PEPA", "PEPB"], dtype=object),
        "unmodified_peptide": np.array(["", "", "PEPA", "PEPB"], dtype=object),
        "usi": np.array(["q1", "q2", "l1", "l2"], dtype=object),
        "overlap_id": np.array(["same", "only_q", "same", "only_l"], dtype=object),
    }
    task = CrossSetAnnotationTransferTask(
        query_filter={"search_tier": "acfm"},
        library_filter={"search_tier": "lcfm_valid"},
        k_values=[1],
        exclude_overlap=False,
        fail_on_overlap=False,
        overlap_key="overlap_id",
        compute_evidence_metrics=False,
        save_plots=False,
    )
    results = task.run(emb, meta)
    assert "error" not in results
    assert results["num_queries"] == 1  # q1 dropped by the independent re-check, not exclude_overlap
    assert results["overlap_check"]["n_overlapping_queries"] == 1


def test_compute_sequencing_metrics_consecutive_series() -> None:
    matched = [
        {"theoretical_annotation": "b1"},
        {"theoretical_annotation": "b2"},
        {"theoretical_annotation": "b3"},
        {"theoretical_annotation": "y2"},
    ]
    metrics = compute_sequencing_metrics(matched, sequence_length=5)
    assert metrics["consecutive_ion_series"] == 3
    assert metrics["residue_evidence_coverage"] > 0


def test_evidence_metrics_with_mcp(tmp_path: Path) -> None:
    pytest.importorskip("proteomics_mcp")
    emb, meta = _fixture()
    meta["mz_array"] = np.array(
        [
            [100.0, 200.0],
            [110.0, 210.0],
            [100.0, 200.0],
            [50.0, 60.0],
            [110.0, 210.0],
            [70.0, 80.0],
        ],
        dtype=object,
    )
    meta["intensity_array"] = np.array(
        [
            [1000.0, 500.0],
            [900.0, 400.0],
            [1000.0, 500.0],
            [100.0, 50.0],
            [900.0, 400.0],
            [100.0, 50.0],
        ],
        dtype=object,
    )
    meta["precursor_mz"] = np.array([500.0, 501.0, 500.0, 400.0, 501.0, 400.0], dtype=float)
    meta["precursor_charge"] = np.array([2, 2, 2, 2, 2, 2], dtype=int)

    task = CrossSetAnnotationTransferTask(
        query_filter={"search_tier": "acfm"},
        library_filter={"search_tier": "lcfm_valid"},
        k_values=[1],
        output_dir=str(tmp_path),
        compute_evidence_metrics=True,
        score_blocks=["A", "B", "C"],
        num_workers=1,
    )
    results = task.run(emb, meta)
    assert "error" not in results, results.get("error")
    candidates = pl.read_csv(results["candidates_csv"])
    assert "q_obs__lib_theo__matched_ion_count" in candidates.columns
    assert "q_obs__lib_obs__cosine_similarity" in candidates.columns
    assert "lib_obs__lib_theo__matched_ion_count" in candidates.columns
