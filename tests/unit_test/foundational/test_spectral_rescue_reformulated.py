"""Tests for the reformulated spectral rescue evaluation task."""

from __future__ import annotations

import os

import numpy as np
import polars as pl

from instanovo_fm.eval.embed_eval_tasks import TASK_REGISTRY
from instanovo_fm.eval.embed_eval_tasks.spectral_rescue_reformulated import (
    SpectralRescueTaskReformulated,
)
from instanovo_fm.eval.embed_eval_tasks.spectral_rescue_reformulated_plots import (
    load_rescue_artifacts_from_dir,
)


def _make_rescue_fixture():
    sequences = np.array(
        [
            "LEQGQALDDLMPAQK",
            "LEQGQALDDLMPAQK",
            "LEQGQALDDLMPAQK",
            "LEQGQALDDLM[UNIMOD:35]PAQK",
            "LEQGQALDDLM[UNIMOD:21]PAQK",
            "TTTTTTTTTTTTTTT",
            "LEQGQALDDLMPAQR",
            "LEQGQALDDLMPAQK",
        ],
        dtype=object,
    )
    unmodified = np.array(
        [
            "LEQGQALDDLMPAQK",
            "LEQGQALDDLMPAQK",
            "LEQGQALDDLMPAQK",
            "LEQGQALDDLMPAQK",
            "LEQGQALDDLMPAQK",
            "TTTTTTTTTTTTTTT",
            "LEQGQALDDLMPAQR",
            "LEQGQALDDLMPAQK",
        ],
        dtype=object,
    )
    projects = np.array(["PXD047134"] * 7 + ["PXD_OTHER"], dtype=object)
    meta = {
        "peptides": sequences,
        "unmodified_peptide": unmodified,
        "search_project": projects,
        "usi": np.array([f"mzspec:{project}:run:scan:{i}" for i, project in enumerate(projects)], dtype=object),
    }
    emb = np.array(
        [
            [1.00, 0.00, 0.00],
            [0.98, 0.02, 0.00],
            [0.95, 0.01, 0.00],
            [0.96, 0.04, 0.00],
            [0.80, 0.20, 0.00],
            [0.00, 1.00, 0.00],
            [0.85, 0.15, 0.00],
            [1.00, 0.00, 0.00],
        ],
        dtype=np.float32,
    )
    return emb, meta


def test_task_is_registered():
    assert TASK_REGISTRY["spectralrescuetaskreformulated"] is SpectralRescueTaskReformulated
    assert TASK_REGISTRY["spectral_rescue_reformulated"] is SpectralRescueTaskReformulated


def test_tokenizer_keeps_internal_modification_with_amino_acid():
    tokens = SpectralRescueTaskReformulated._tokenize("LEQGQALDDLM[UNIMOD:35]PAQK")

    assert "M[UNIMOD:35]" in tokens
    assert "[UNIMOD:35]" not in tokens
    assert SpectralRescueTaskReformulated._backbone_sequence("LEQGQALDDLM[UNIMOD:35]PAQK") == "LEQGQALDDLMPAQK"


def test_selection_excludes_queries_and_same_backbone_modified_variants():
    _, meta = _make_rescue_fixture()
    task = SpectralRescueTaskReformulated(
        sample_seed=7,
        num_reference_queries=1,
        num_modified_queries=1,
        max_negatives=None,
        max_base_positives=None,
    )

    selection = task._build_selection(meta)
    library = set(selection["library_indices"])

    assert not set(selection["reference_query_indices"]) & library
    assert not set(selection["modified_query_indices"]) & library
    assert 4 not in library  # same-backbone modified variant must not leak into library
    assert 7 not in library  # same peptide from a different project must not leak into library

    for idx, role in zip(selection["library_indices"], selection["library_roles"]):
        if role == "positive_library":
            assert meta["peptides"][idx] == "LEQGQALDDLMPAQK"
        else:
            assert meta["unmodified_peptide"][idx] != "LEQGQALDDLMPAQK"


def test_near_sequence_negative_is_filtered_by_clean_edit_distance():
    _, meta = _make_rescue_fixture()
    task = SpectralRescueTaskReformulated(
        sample_seed=7,
        num_reference_queries=1,
        num_modified_queries=1,
        max_negatives=None,
        max_base_positives=None,
        min_negative_clean_edit_distance=5,
    )

    selection = task._build_selection(meta)

    assert 5 in selection["library_indices"]
    assert 6 not in selection["library_indices"]


def test_run_computes_metrics_and_writes_artifacts(tmp_path):
    emb, meta = _make_rescue_fixture()
    task = SpectralRescueTaskReformulated(
        sample_seed=7,
        num_reference_queries=1,
        num_modified_queries=1,
        max_base_positives=None,
        max_negatives=None,
        k_values=[1, 2],
        output_dir=str(tmp_path),
        save_artifacts=True,
        save_plots=True,
    )

    results = task.run(emb, meta)

    assert "error" not in results
    assert results["num_base_positives"] == 2
    assert results["num_negatives"] == 1
    assert results["num_reference_queries"] == 1
    assert results["num_modified_queries"] == 1
    assert "reference_query_best_positive_rank_mean" in results
    assert "modified_query_best_positive_rank_mean" in results
    assert "reference_query_margin_mean" in results
    assert "modified_query_margin_mean" in results
    assert not any(key.startswith("_") for key in results)
    assert os.path.exists(results["artifact_paths"]["selection_csv"])
    assert os.path.exists(results["artifact_paths"]["similarity_npz"])
    assert os.path.exists(results["artifact_paths"]["ranked_library_csv"])
    assert os.path.exists(results["artifact_paths"]["query_metrics_csv"])
    assert os.path.exists(results["artifact_paths"]["all_pair_scores_csv"])
    assert os.path.exists(results["plot_paths"]["retrieval_curves"])
    assert os.path.exists(results["plot_paths"]["distance_similarity_hexbin"])
    assert os.path.exists(results["plot_paths"]["margin_comparison"])
    assert os.path.exists(results["plot_paths"]["match_unmatch_distribution"])
    assert os.path.exists(results["plot_paths"]["margin_distribution_and_cdf"])
    assert os.path.exists(results["plot_paths"]["unified_margin_and_cdf"])
    assert os.path.exists(results["plot_paths"]["match_unmatch_cosine_values_csv"])
    assert os.path.exists(results["plot_paths"]["per_query_margin_values_csv"])

    ranked = pl.read_csv(results["artifact_paths"]["ranked_library_csv"])
    query_metrics = pl.read_csv(results["artifact_paths"]["query_metrics_csv"])
    all_pairs = pl.read_csv(results["artifact_paths"]["all_pair_scores_csv"])

    assert "query_sequence" in ranked.columns
    assert "library_sequence" in ranked.columns
    assert "library_usi" in ranked.columns
    assert "is_positive" in ranked.columns
    assert "clean_edit_distance" in ranked.columns
    assert query_metrics.height == 2
    assert "best_positive_similarity" in query_metrics.columns
    assert "query_usi" in query_metrics.columns
    assert all_pairs.height == results["num_similarity_pairs"]
    assert {"query_sequence", "library_sequence", "score", "rank", "is_positive"} <= set(all_pairs.columns)


def test_run_consumes_offline_rescue_roles():
    emb, meta = _make_rescue_fixture()
    roles = np.array(
        [
            "reference_query",
            "positive_library",
            "positive_library",
            "modified_query",
            "unused_same_backbone_variant",
            "negative_library",
            "unused_near_negative",
            "unused_other_project",
        ],
        dtype=object,
    )
    meta["rescue_role"] = roles
    task = SpectralRescueTaskReformulated(k_values=[1, 2], save_artifacts=False, save_plots=False)

    results = task.run(emb, meta)

    assert "error" not in results
    assert results["num_reference_queries"] == 1
    assert results["num_modified_queries"] == 1
    assert results["num_base_positives"] == 2
    assert results["num_negatives"] == 1
    assert results["num_similarity_pairs"] == 6


def test_load_rescue_artifacts_from_dir(tmp_path):
    emb, meta = _make_rescue_fixture()
    task = SpectralRescueTaskReformulated(
        sample_seed=7,
        num_reference_queries=1,
        num_modified_queries=1,
        max_base_positives=None,
        max_negatives=None,
        k_values=[1, 2],
        output_dir=str(tmp_path),
        save_artifacts=True,
        save_plots=False,
    )
    task.run(emb, meta)

    S, selection, query_metrics, loaded_meta = load_rescue_artifacts_from_dir(tmp_path)

    assert S.shape[0] == 2
    assert len(selection["query_indices"]) == 2
    assert len(selection["library_indices"]) == 3
    assert len(query_metrics) == 2
    assert selection["sequences"][0] == meta["peptides"][0]
    assert loaded_meta is None
