"""Tests for the Spectral Annotation Transfer evaluation task."""

from __future__ import annotations

import builtins
import os

import numpy as np
import pytest

from instanovo_fm.eval.embed_eval_tasks.spectral_annotation_transfer import (
    SpectralAnnotationTransferTask,
)


def _make_synthetic_data(n=100, d=64, n_unique=10, seed=42):
    """Create synthetic embeddings with n_unique peptides, each duplicated n//n_unique times.

    Returns (emb, meta) where emb is L2-normalised.
    """
    rng = np.random.default_rng(seed)

    # n_unique peptides, each with n // n_unique spectra
    reps = n // n_unique
    alphabet = "ABCDEFGHJKMNOPQRSTUVWXYZ"  # excludes I, L
    unique_peptides = [f"PEPTA{alphabet[i % len(alphabet)]}{alphabet[(i // len(alphabet)) % len(alphabet)]}" for i in range(n_unique)]
    peptides = np.array([unique_peptides[i // reps] for i in range(n)], dtype=object)

    # Embeddings: tight clusters per peptide (well-separated centroids)
    centroids = rng.normal(0, 1.0, size=(n_unique, d))
    emb = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        pep_id = i // reps
        emb[i] = centroids[pep_id] + rng.normal(0, 0.05, size=d)

    # L2 normalise
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.where(norms == 0, 1.0, norms)

    meta = {
        "peptides": peptides,
        "precursor_charge": np.array([2 if i % 2 == 0 else 3 for i in range(n)], dtype=np.int32),
        "frag_type": np.array(["HCD" if i % 2 == 0 else "CID" for i in range(n)], dtype=object),
        "search_detector": np.array(["Orbitrap"] * n, dtype=object),
    }

    return emb, meta


class TestQueryLibraryConstruction:
    def test_deduplication(self):
        emb, meta = _make_synthetic_data(n=100, n_unique=10)
        task = SpectralAnnotationTransferTask(peptide_key="peptides")

        query_idx, lib_idx = task._build_query_library(meta["peptides"])

        # Should have exactly 10 unique queries
        assert len(query_idx) == 10
        # Library should be all 100 spectra
        assert len(lib_idx) == 100
        # Query indices should be a subset of library indices
        assert all(q in lib_idx for q in query_idx)

    def test_all_unique_peptides_represented(self):
        emb, meta = _make_synthetic_data(n=100, n_unique=10)
        task = SpectralAnnotationTransferTask(peptide_key="peptides")

        query_idx, _ = task._build_query_library(meta["peptides"])
        query_peps = set(str(meta["peptides"][i]).strip() for i in query_idx)

        assert len(query_peps) == 10
        assert query_peps.issubset(set(meta["peptides"]))

    def test_max_samples_caps_queries(self):
        emb, meta = _make_synthetic_data(n=100, n_unique=10)
        task = SpectralAnnotationTransferTask(peptide_key="peptides", max_samples=5)

        query_idx, lib_idx = task._build_query_library(meta["peptides"])

        assert len(query_idx) == 5
        assert len(lib_idx) == 100  # Library uncapped


class TestSimilarityMatrix:
    def test_shape(self):
        emb, meta = _make_synthetic_data(n=50, d=32, n_unique=5)
        task = SpectralAnnotationTransferTask(peptide_key="peptides")

        query_idx, lib_idx = task._build_query_library(meta["peptides"])
        E_q = task._l2_normalise(emb[query_idx].astype(np.float32).copy())
        E_lib = task._l2_normalise(emb[lib_idx].astype(np.float32).copy())

        S = task._compute_similarity_batched(E_q, E_lib, batch_size=2)

        assert S.shape == (5, 50)

    def test_values_in_range(self):
        emb, meta = _make_synthetic_data(n=50, d=32, n_unique=5)
        task = SpectralAnnotationTransferTask(peptide_key="peptides")

        query_idx, lib_idx = task._build_query_library(meta["peptides"])
        E_q = task._l2_normalise(emb[query_idx].astype(np.float32).copy())
        E_lib = task._l2_normalise(emb[lib_idx].astype(np.float32).copy())

        S = task._compute_similarity_batched(E_q, E_lib, batch_size=10)

        assert np.all(S >= -1.01)
        assert np.all(S <= 1.01)

    def test_batched_matches_full(self):
        emb, meta = _make_synthetic_data(n=50, d=32, n_unique=5)
        task = SpectralAnnotationTransferTask(peptide_key="peptides")

        query_idx, lib_idx = task._build_query_library(meta["peptides"])
        E_q = task._l2_normalise(emb[query_idx].astype(np.float32).copy())
        E_lib = task._l2_normalise(emb[lib_idx].astype(np.float32).copy())

        S_batched = task._compute_similarity_batched(E_q, E_lib, batch_size=2)
        S_full = E_q @ E_lib.T

        np.testing.assert_allclose(S_batched, S_full, atol=1e-6)


class TestSelfMatchMasking:
    def test_self_match_masked(self):
        emb, meta = _make_synthetic_data(n=50, d=32, n_unique=5)
        task = SpectralAnnotationTransferTask(peptide_key="peptides")

        query_idx, lib_idx = task._build_query_library(meta["peptides"])
        E_q = task._l2_normalise(emb[query_idx].astype(np.float32).copy())
        E_lib = task._l2_normalise(emb[lib_idx].astype(np.float32).copy())

        S = task._compute_similarity_batched(E_q, E_lib, batch_size=10)

        self_col_map = task._build_self_match_map(query_idx, lib_idx)
        task._mask_self_matches(S, self_col_map)

        # Each query should have exactly one -inf entry (its self-match)
        for row, col in self_col_map.items():
            assert S[row, col] == -np.inf


class TestLevenshteinDistance:
    def test_identical(self):
        assert SpectralAnnotationTransferTask._levenshtein_distance("ABC", "ABC") == 0

    def test_empty(self):
        assert SpectralAnnotationTransferTask._levenshtein_distance("", "ABC") == 3
        assert SpectralAnnotationTransferTask._levenshtein_distance("ABC", "") == 3

    def test_substitution(self):
        assert SpectralAnnotationTransferTask._levenshtein_distance("ABC", "ABD") == 1

    def test_insertion(self):
        assert SpectralAnnotationTransferTask._levenshtein_distance("ABC", "ABCD") == 1

    def test_deletion(self):
        assert SpectralAnnotationTransferTask._levenshtein_distance("ABCD", "ABC") == 1

    def test_complex(self):
        assert SpectralAnnotationTransferTask._levenshtein_distance("PEPTIDE", "PEPTODE") == 1
        assert SpectralAnnotationTransferTask._levenshtein_distance("PEPTIDE", "REPTIDK") == 2


class TestRetrievalMetrics:
    def test_singleton_queries_are_skipped_after_self_masking(self):
        S = np.array([[-np.inf, 0.3], [-0.2, -np.inf]], dtype=np.float32)
        query_peps = np.array(["A", "B"], dtype=object)
        lib_peps = np.array(["A", "C"], dtype=object)
        self_col_map = {0: 0, 1: 1}

        metrics = SpectralAnnotationTransferTask._compute_retrieval_metrics(
            S,
            query_peps,
            lib_peps,
            self_col_map,
            [1, 5],
        )

        assert metrics["num_valid_queries"] == 0
        assert metrics["recall@1"] == 0.0
        assert metrics["prop_recall@1"] == 0.0
        assert metrics["map@5"] == 0.0

    def test_self_match_is_excluded_from_relevant_denominator(self):
        S = np.array([[-np.inf, 0.9, 0.1]], dtype=np.float32)
        query_peps = np.array(["A"], dtype=object)
        lib_peps = np.array(["A", "A", "B"], dtype=object)
        self_col_map = {0: 0}

        metrics = SpectralAnnotationTransferTask._compute_retrieval_metrics(
            S,
            query_peps,
            lib_peps,
            self_col_map,
            [1],
        )

        assert metrics["num_valid_queries"] == 1
        assert metrics["recall@1"] == 1.0
        assert metrics["prop_recall@1"] == 1.0
        assert metrics["map@1"] == 1.0


class TestRunEndToEnd:
    def test_runs_successfully(self):
        emb, meta = _make_synthetic_data(n=100, d=64, n_unique=10)

        task = SpectralAnnotationTransferTask(
            k_values=[1, 5, 10],
            peptide_key="peptides",
            max_samples=20_000,
            batch_size=10,
            n_ed_sample_pairs=1000,
            sample_seed=42,
            save_matrix_artifact=False,
        )

        results = task.run(emb, meta, faiss_index=None)

        assert "error" not in results
        assert results["num_embeddings"] == 100
        assert results["num_queries"] == 10
        assert results["num_library"] == 100

        # Retrieval metrics
        for k in [1, 5, 10]:
            assert f"recall@{k}" in results
            assert f"prop_recall@{k}" in results
        assert "map@10" in results
        assert "map@20" not in results

        # Replicate similarity
        assert "replicate_similarity_mean" in results
        assert results["replicate_similarity_mean"] > 0.5  # Tight clusters → high sim

        # Edit distance profile
        assert "edit_distance_profile" in results
        assert "d=0" in results["edit_distance_profile"]

        # Correlation
        assert "spearman_r" in results
        assert "pearson_r" in results

        # AUPRC / AUROC
        assert "auprc" in results
        assert "auroc" in results
        assert results["auroc"] > 0.5  # Better than random

    def test_high_recall_with_tight_clusters(self):
        emb, meta = _make_synthetic_data(n=100, d=64, n_unique=10)

        task = SpectralAnnotationTransferTask(
            k_values=[1, 5, 10],
            peptide_key="peptides",
            batch_size=50,
            n_ed_sample_pairs=500,
            save_matrix_artifact=False,
        )

        results = task.run(emb, meta, faiss_index=None)

        # With tight clusters (noise=0.05), recall@10 should be very high
        assert results["recall@10"] > 0.8

    def test_missing_peptide_key(self):
        emb, meta = _make_synthetic_data(n=50, d=32, n_unique=5)
        del meta["peptides"]

        task = SpectralAnnotationTransferTask(
            peptide_key="peptides",
            save_matrix_artifact=False,
        )

        results = task.run(emb, meta, faiss_index=None)
        assert "error" in results

    def test_loggable_metrics(self):
        emb, meta = _make_synthetic_data(n=100, d=64, n_unique=10)

        task = SpectralAnnotationTransferTask(
            k_values=[1, 5],
            peptide_key="peptides",
            batch_size=50,
            n_ed_sample_pairs=500,
            save_matrix_artifact=False,
        )

        results = task.run(emb, meta, faiss_index=None)
        loggable = task.get_loggable_metrics(results)

        assert "recall@1" in loggable
        assert "recall@5" in loggable
        assert "replicate_similarity_mean" in loggable
        assert "auprc" in loggable
        assert "auroc" in loggable

    def test_respects_custom_k_values(self):
        emb, meta = _make_synthetic_data(n=100, d=64, n_unique=10)

        task = SpectralAnnotationTransferTask(
            k_values=[2],
            peptide_key="peptides",
            batch_size=50,
            n_ed_sample_pairs=500,
            save_matrix_artifact=False,
        )

        results = task.run(emb, meta, faiss_index=None)

        assert "recall@2" in results
        assert "prop_recall@2" in results
        assert "map@2" in results
        assert "recall@1" not in results
        assert "recall@5" not in results
        assert "map@20" not in results

    def test_raises_if_scipy_missing_for_correlation(self, monkeypatch):
        task = SpectralAnnotationTransferTask(
            peptide_key="peptides",
            n_ed_sample_pairs=4,
            save_matrix_artifact=False,
        )
        S = np.array(
            [[0.9, 0.3, 0.1], [0.2, 0.8, 0.4]],
            dtype=np.float32,
        )
        query_peps = np.array(["AAA", "BBB"], dtype=object)
        lib_peps = np.array(["AAA", "AAC", "BBB"], dtype=object)
        real_import = builtins.__import__

        def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "scipy.stats":
                raise ImportError("blocked for test")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", blocked_import)

        with pytest.raises(ImportError, match="requires scipy"):
            task._compute_correlation(S, query_peps, lib_peps)

    def test_raises_if_sklearn_missing_for_auprc_auroc(self, monkeypatch):
        S = np.array(
            [[-np.inf, 0.8, 0.2], [0.1, 0.7, -np.inf]],
            dtype=np.float32,
        )
        query_peps = np.array(["AAA", "BBB"], dtype=object)
        lib_peps = np.array(["AAA", "AAA", "BBB"], dtype=object)
        real_import = builtins.__import__

        def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "sklearn.metrics":
                raise ImportError("blocked for test")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", blocked_import)

        with pytest.raises(ImportError, match="requires scikit-learn"):
            SpectralAnnotationTransferTask._compute_auprc_auroc(S, query_peps, lib_peps)


class TestArtifactSaving:
    def test_save_npz(self, tmp_path):
        emb, meta = _make_synthetic_data(n=50, d=32, n_unique=5)

        task = SpectralAnnotationTransferTask(
            k_values=[1, 5],
            peptide_key="peptides",
            batch_size=10,
            n_ed_sample_pairs=100,
            save_matrix_artifact=True,
            output_dir=str(tmp_path),
        )

        results = task.run(emb, meta, faiss_index=None)

        assert "artifact_path" in results
        artifact_path = results["artifact_path"]
        assert os.path.exists(artifact_path)

        # Load and verify
        data = np.load(artifact_path, allow_pickle=True)
        assert "similarity_matrix" in data
        assert "query_peptides" in data
        assert "library_peptides" in data
        assert "query_indices" in data
        assert "library_indices" in data
        assert "plot_paths" in results
        assert os.path.exists(results["plot_paths"]["similarity_heatmap"])
        assert os.path.exists(results["plot_paths"]["distance_similarity_plot"])

        assert data["similarity_matrix"].shape == (5, 50)
        assert len(data["query_peptides"]) == 5
        assert len(data["library_peptides"]) == 50


class TestConditionalEval:
    def test_conditional_eval(self, tmp_path):
        emb, meta = _make_synthetic_data(n=100, d=64, n_unique=10)

        task = SpectralAnnotationTransferTask(
            k_values=[5],
            peptide_key="peptides",
            batch_size=50,
            n_ed_sample_pairs=500,
            save_matrix_artifact=True,
            output_dir=str(tmp_path),
            enable_conditional_eval=True,
            conditional_min_samples=10,
            conditional_subsets=[
                {
                    "name": "hcd_orbitrap",
                    "frag_types": ["HCD"],
                    "detectors": ["Orbitrap"],
                    "instruments": None,
                }
            ],
        )

        results = task.run(emb, meta, faiss_index=None)

        assert "conditional_eval" in results
        assert "hcd_orbitrap" in results["conditional_eval"]

        subset_res = results["conditional_eval"]["hcd_orbitrap"]
        assert subset_res["skipped"] is False
        assert "recall@5" in subset_res
        assert "auprc" in subset_res
        assert os.path.exists(subset_res["artifact_path"])
        assert os.path.exists(subset_res["plot_paths"]["similarity_heatmap"])
        assert os.path.exists(subset_res["plot_paths"]["distance_similarity_plot"])


class TestCanonicalization:
    def test_canonicalize_preserves_ptms_and_maps_i_to_l(self):
        assert SpectralAnnotationTransferTask._canonicalize_peptide("PEPTIDE") == "PEPTLDE"
        assert SpectralAnnotationTransferTask._canonicalize_peptide("PEPTLDE") == "PEPTLDE"
        assert SpectralAnnotationTransferTask._canonicalize_peptide("ACD[UNIMOD:35]IK") == "ACD[UNIMOD:35]LK"
        assert SpectralAnnotationTransferTask._canonicalize_peptide("M[+16]IPTIDE") == "M[+16]LPTLDE"
        assert SpectralAnnotationTransferTask._canonicalize_peptide("ILIL") == "LLLL"

    def test_conditional_eval_uses_canonical_peptides_for_matching(self):
        # Two spectra differ only by I/L; canonicalization should treat them as
        # the same peptide in conditional retrieval scoring.
        emb = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        meta = {
            "peptides": np.array(["PEPTIDE", "PEPTLDE"], dtype=object),
            "frag_type": np.array(["HCD", "HCD"], dtype=object),
            "search_detector": np.array(["Orbitrap", "Orbitrap"], dtype=object),
        }

        task = SpectralAnnotationTransferTask(
            k_values=[1],
            peptide_key="peptides",
            save_matrix_artifact=False,
            save_plots=False,
            enable_conditional_eval=True,
            conditional_min_samples=2,
            conditional_subsets=[
                {
                    "name": "hcd_only",
                    "frag_types": ["HCD"],
                    "detectors": None,
                    "instruments": None,
                }
            ],
        )

        results = task.run(emb, meta, faiss_index=None)
        subset = results["conditional_eval"]["hcd_only"]

        assert subset["skipped"] is False
        assert subset["num_queries"] == 1
        assert subset["num_valid_queries"] == 1
        assert subset["recall@1"] == 1.0


class TestSequenceSimilarityNormalization:
    def test_sequence_similarity_uses_token_length_not_raw_string_length(self):
        task = SpectralAnnotationTransferTask(
            peptide_key="peptides",
            n_ed_sample_pairs=1,
            save_matrix_artifact=False,
            save_plots=False,
        )

        S = np.array([[0.5]], dtype=np.float32)
        query_peps = np.array(["A[UNIMOD:35]C"], dtype=object)
        lib_peps = np.array(["A[UNIMOD:35]D"], dtype=object)

        _, edit_distances, seq_sims = task._collect_similarity_distance_pairs(
            S,
            query_peps,
            lib_peps,
            seed_offset=123,
            max_pairs=1,
        )

        # Tokenized peptides: ["A", "[UNIMOD:35]", "C"] vs ["A", "[UNIMOD:35]", "D"]
        # Edit distance = 1, max token length = 3 => seq_sim = 1 - 1/3 = 2/3.
        assert edit_distances[0] == 1
        assert seq_sims[0] == pytest.approx(2.0 / 3.0, rel=1e-6)
