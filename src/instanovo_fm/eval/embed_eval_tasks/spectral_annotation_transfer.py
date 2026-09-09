"""Spectral Annotation Transfer evaluation task.

This task evaluates the quality of spectrum embeddings for annotation transfer
by computing the full M x N cosine similarity matrix between deduplicated query
spectra and the full library, then reporting retrieval metrics, edit distance
profiling, correlation, and AUPRC/AUROC.
"""

import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from instanovo_fm.eval.embed_eval_tasks import BaseTask

logger = logging.getLogger(__name__)


class SpectralAnnotationTransferTask(BaseTask):
    """Spectral Annotation Transfer evaluation task.

    Constructs a rectangular M x N similarity matrix where:
      - Queries (M rows): one representative spectrum per unique peptide.
      - Library (N cols): all spectra in the validation set (with duplicates).

    Reports retrieval metrics (Recall@k, MAP), replicate similarity statistics,
    edit-distance-binned similarity profiles, correlation between embedding
    similarity and sequence similarity, and AUPRC/AUROC for replicate
    identification.

    Saves the similarity matrix and peptide labels as a compressed .npz artifact
    for downstream re-analysis.
    """

    name = "Spectral Annotation Transfer"
    description = "Evaluate annotation transfer via M x N cosine similarity matrix with retrieval metrics, edit distance profiling, and AUPRC/AUROC"
    requires_metadata = True
    requires_faiss = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.k_values: List[int] = kwargs.get("k_values", [1, 5, 10])
        self.peptide_key: str = kwargs.get("peptide_key", "peptides")
        self.max_samples: int = kwargs.get("max_samples", 20_000)
        self.batch_size: int = kwargs.get("batch_size", 500)
        self.edit_distance_bins: List[int] = kwargs.get("edit_distance_bins", [0, 1, 2, 3, 5])
        self.n_ed_sample_pairs: int = kwargs.get("n_ed_sample_pairs", 500_000)
        self.sample_seed: int = kwargs.get("sample_seed", 42)
        self.save_matrix_artifact: bool = kwargs.get("save_matrix_artifact", True)
        self.save_plots: bool = kwargs.get("save_plots", True)
        self.heatmap_max_queries: int = kwargs.get("heatmap_max_queries", 256)
        self.heatmap_max_library: int = kwargs.get("heatmap_max_library", 512)
        self.plot_max_points: int = kwargs.get("plot_max_points", 50_000)
        self.plot_dpi: int = kwargs.get("plot_dpi", 160)
        self.output_dir: Optional[str] = kwargs.get("output_dir", None)

        # Instrument-conditional evaluation
        self.enable_conditional_eval: bool = kwargs.get("enable_conditional_eval", False)
        self.conditional_min_samples: int = kwargs.get("conditional_min_samples", 200)
        self.conditional_subsets: list = kwargs.get(
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        emb: np.ndarray,
        meta: Dict[str, np.ndarray],
        faiss_index: Any = None,
    ) -> Dict[str, Any]:
        """Run the annotation transfer evaluation task.

        Args:
            emb: Embeddings array of shape (N, D).
            meta: Metadata dictionary.
            faiss_index: Unused (kept for interface compatibility).

        Returns:
            Dictionary containing evaluation results.
        """
        self.validate_inputs(emb, meta, faiss_index=None)
        start_time = time.time()
        logger.info(
            "Spectral annotation transfer config: %s",
            self._get_config_summary(),
        )

        if self.peptide_key not in meta:
            return {
                "task_name": self.name,
                "error": (f"Peptide key '{self.peptide_key}' not found in metadata. Available: {list(meta.keys())}"),
                "execution_time": time.time() - start_time,
            }

        peptides = meta[self.peptide_key]
        n_total = len(emb)

        if n_total < 2:
            return {
                "task_name": self.name,
                "error": f"Insufficient embeddings (found {n_total})",
                "execution_time": time.time() - start_time,
            }

        # Build query (deduplicated) and library (all spectra) ---------
        query_indices, library_indices = self._build_query_library(peptides)

        # Canonicalize peptides for all downstream comparisons (replicates, edit distance)
        query_peps = np.array([self._canonicalize_peptide(str(peptides[i]).strip()) for i in query_indices])
        lib_peps = np.array([self._canonicalize_peptide(str(peptides[i]).strip()) for i in library_indices])

        M, N = len(query_indices), len(library_indices)

        logger.info(
            "Annotation transfer: %d queries (unique peptides), %d library spectra",
            M,
            N,
        )

        # Normalise embeddings -----------------------------------------
        E_q = emb[query_indices].astype(np.float32).copy()
        E_lib = emb[library_indices].astype(np.float32).copy()
        E_q = self._l2_normalise(E_q)
        E_lib = self._l2_normalise(E_lib)

        # Compute M x N similarity matrix (batched) --------------------
        S = self._compute_similarity_batched(E_q, E_lib, self.batch_size)
        logger.info("Similarity matrix computed: shape %s", S.shape)

        # Mask self-matches (query spectrum appearing in library) -------
        self_col_map = self._build_self_match_map(query_indices, library_indices)
        self._mask_self_matches(S, self_col_map)

        # Compute metrics ----------------------------------------------
        retrieval_metrics = self._compute_retrieval_metrics(
            S,
            query_peps,
            lib_peps,
            self_col_map,
            self.k_values,
        )
        replicate_metrics = self._compute_replicate_similarity(S, query_peps, lib_peps)
        ed_profile = self._compute_edit_distance_profile(S, query_peps, lib_peps)
        correlation_metrics = self._compute_correlation(S, query_peps, lib_peps)
        auprc_auroc_metrics = self._compute_auprc_auroc(S, query_peps, lib_peps)

        # Save .npz artifact -------------------------------------------
        artifact_path = None
        if self.save_matrix_artifact and self.output_dir:
            artifact_path = self._save_artifact(
                S,
                query_peps,
                lib_peps,
                np.array(query_indices),
                np.array(library_indices),
            )

        execution_time = time.time() - start_time

        results: Dict[str, Any] = {
            "task_name": self.name,
            "num_embeddings": n_total,
            "num_queries": M,
            "num_library": N,
            "num_unique_peptides": M,
            "execution_time": execution_time,
            **retrieval_metrics,
            **replicate_metrics,
            **ed_profile,
            **correlation_metrics,
            **auprc_auroc_metrics,
        }

        if artifact_path is not None:
            results["artifact_path"] = str(artifact_path)

        if self.save_plots and self.output_dir:
            plot_paths = self._save_visualizations(S, query_peps, lib_peps)
            if plot_paths:
                results["plot_paths"] = plot_paths

        # Conditional evaluation
        if self.enable_conditional_eval:
            results["conditional_eval"] = self._run_conditional_eval(emb, meta)

        return results

    def _get_config_summary(self) -> Dict[str, Any]:
        """Return a serializable summary of the effective task settings."""
        return {
            "k_values": list(self.k_values),
            "peptide_key": self.peptide_key,
            "max_samples": self.max_samples,
            "batch_size": self.batch_size,
            "edit_distance_bins": list(self.edit_distance_bins),
            "n_ed_sample_pairs": self.n_ed_sample_pairs,
            "sample_seed": self.sample_seed,
            "save_matrix_artifact": self.save_matrix_artifact,
            "save_plots": self.save_plots,
            "heatmap_max_queries": self.heatmap_max_queries,
            "heatmap_max_library": self.heatmap_max_library,
            "plot_max_points": self.plot_max_points,
            "plot_dpi": self.plot_dpi,
            "output_dir": self.output_dir,
            "enable_conditional_eval": self.enable_conditional_eval,
            "conditional_min_samples": self.conditional_min_samples,
            "conditional_subsets": self.conditional_subsets,
        }

    def get_loggable_metrics(self, task_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract key metrics for logging."""
        if "error" in task_results or "skipped" in task_results:
            return {}

        loggable: Dict[str, float] = {}

        # Retrieval metrics
        for k in self.k_values:
            for prefix in ("recall", "prop_recall"):
                key = f"{prefix}@{k}"
                if key in task_results:
                    loggable[key] = float(task_results[key])

        # MAP
        for key in task_results:
            if key.startswith("map@"):
                loggable[key] = float(task_results[key])

        # Replicate similarity
        if "replicate_similarity_mean" in task_results:
            loggable["replicate_similarity_mean"] = float(task_results["replicate_similarity_mean"])

        # Correlation
        for key in ("spearman_r", "pearson_r"):
            if key in task_results:
                loggable[key] = float(task_results[key])

        # AUPRC / AUROC
        for key in ("auprc", "auroc"):
            if key in task_results:
                loggable[key] = float(task_results[key])

        # Conditional metrics
        for subset_name, subset_info in task_results.get("conditional_eval", {}).items():
            if isinstance(subset_info, dict) and not subset_info.get("skipped", True):
                for k in self.k_values:
                    key = f"recall@{k}"
                    if key in subset_info:
                        loggable[f"cond_{subset_name}_{key}"] = float(subset_info[key])
                for key in ("auprc", "auroc", "replicate_similarity_mean"):
                    if key in subset_info:
                        loggable[f"cond_{subset_name}_{key}"] = float(subset_info[key])

        return loggable

    # ------------------------------------------------------------------
    # Query / Library construction
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize_peptide(seq: str) -> List[str]:
        """Tokenize a peptide into amino acids and modifications."""
        return re.findall(r"\[[^\]]*\]|[A-Z]", seq)

    @classmethod
    def _canonicalize_peptide(cls, seq: str) -> str:
        """Canonicalize a peptide for grouping and comparison.

        1. Keep PTMs intact (e.g. [UNIMOD:35], [+16])
        2. Replace I → L (isobaric equivalence)
        """
        tokens = cls._tokenize_peptide(seq)
        return "".join("L" if t == "I" else t for t in tokens)

    def _build_query_library(
        self,
        peptides: np.ndarray,
    ) -> Tuple[List[int], List[int]]:
        """Build deduplicated query set and full library.

        Queries: one representative spectrum per unique peptide (the first
        occurrence). Library: all spectra.

        Returns:
            (query_indices, library_indices) — both are lists of ints indexing
            into the original embedding / metadata arrays.
        """
        rng = np.random.RandomState(self.sample_seed)
        seen: Dict[str, int] = {}
        for i, pep in enumerate(peptides):
            pep_str = str(pep).strip()
            if pep_str:
                canon = self._canonicalize_peptide(pep_str)
                if canon not in seen:
                    seen[canon] = i

        query_indices = sorted(seen.values())
        library_indices = list(range(len(peptides)))

        # Cap queries if max_samples is set (library stays full)
        if self.max_samples and len(query_indices) > self.max_samples:
            query_indices = sorted(rng.choice(query_indices, self.max_samples, replace=False).tolist())

        return query_indices, library_indices

    # ------------------------------------------------------------------
    # Similarity matrix
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalise(E: np.ndarray) -> np.ndarray:
        """L2-normalise rows in place and return."""
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        E /= norms
        return E

    @staticmethod
    def _compute_similarity_batched(
        E_q: np.ndarray,
        E_lib: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:
        """Compute (M, N) cosine similarity matrix in batches.

        Both E_q and E_lib must already be L2-normalised.

        Args:
            E_q: Query embeddings (M, D).
            E_lib: Library embeddings (N, D).
            batch_size: Number of query rows per batch.

        Returns:
            S: Cosine similarity matrix (M, N), float32.
        """
        M = E_q.shape[0]
        N = E_lib.shape[0]
        S = np.empty((M, N), dtype=np.float32)
        for start in range(0, M, batch_size):
            end = min(start + batch_size, M)
            S[start:end] = E_q[start:end] @ E_lib.T
        return S

    @staticmethod
    def _build_self_match_map(
        query_indices: List[int],
        library_indices: List[int],
    ) -> Dict[int, int]:
        """Map query row index to the library column of its own spectrum.

        Returns:
            Dict mapping query row i → library column j where
            library_indices[j] == query_indices[i].
        """
        lib_idx_to_col = {idx: col for col, idx in enumerate(library_indices)}
        self_map: Dict[int, int] = {}
        for row, q_idx in enumerate(query_indices):
            if q_idx in lib_idx_to_col:
                self_map[row] = lib_idx_to_col[q_idx]
        return self_map

    @staticmethod
    def _mask_self_matches(S: np.ndarray, self_col_map: Dict[int, int]) -> None:
        """Set self-match entries to -inf so they are excluded from ranking."""
        for row, col in self_col_map.items():
            S[row, col] = -np.inf

    # ------------------------------------------------------------------
    # Retrieval metrics (Recall@k, MAP)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_retrieval_metrics(
        S: np.ndarray,
        query_peps: np.ndarray,
        lib_peps: np.ndarray,
        self_col_map: Dict[int, int],
        k_values: List[int],
    ) -> Dict[str, Any]:
        """Compute Recall@k and MAP from the similarity matrix.

        A "hit" for a query is any library spectrum with the same peptide
        sequence (d=0, i.e. a replicate).

        Returns dict with recall@k, prop_recall@k, and map@max_k.
        """
        max_k = max(k_values)
        M = S.shape[0]

        # Pre-compute sorted indices (descending similarity) — top max_k only
        top_k_indices = np.argsort(S, axis=1)[:, : -(max_k + 1) : -1]  # (M, max_k)

        # Build replicate sets per query
        # For each query peptide, the set of library columns with the same peptide
        pep_to_lib_cols: Dict[str, List[int]] = defaultdict(list)
        for col, pep in enumerate(lib_peps):
            pep_to_lib_cols[pep].append(col)

        binary_recall = dict.fromkeys(k_values, 0.0)
        prop_recall = dict.fromkeys(k_values, 0.0)
        aps: List[float] = []
        n_valid_queries = 0

        for i in range(M):
            q_pep = query_peps[i]
            relevant_cols = pep_to_lib_cols.get(q_pep, [])
            relevant_set = set(relevant_cols)
            self_col = self_col_map.get(i)
            if self_col is not None:
                relevant_set.discard(self_col)
            if not relevant_set:
                continue

            n_valid_queries += 1
            retrieved = top_k_indices[i]

            # Recall@k
            for k in k_values:
                top_k = retrieved[:k]
                hits = sum(1 for idx in top_k if idx in relevant_set)
                if hits > 0:
                    binary_recall[k] += 1.0
                prop_recall[k] += hits / len(relevant_set)

            # Average Precision
            ap_hits = 0
            ap_sum = 0.0
            for rank, idx in enumerate(retrieved, 1):
                if idx in relevant_set:
                    ap_hits += 1
                    ap_sum += ap_hits / rank
            denom = min(len(relevant_set), max_k)
            aps.append(ap_sum / denom if denom > 0 else 0.0)

        # Normalise
        metrics: Dict[str, Any] = {}
        if n_valid_queries > 0:
            for k in k_values:
                metrics[f"recall@{k}"] = binary_recall[k] / n_valid_queries
                metrics[f"prop_recall@{k}"] = prop_recall[k] / n_valid_queries
            metrics[f"map@{max_k}"] = float(np.mean(aps))
        else:
            for k in k_values:
                metrics[f"recall@{k}"] = 0.0
                metrics[f"prop_recall@{k}"] = 0.0
            metrics[f"map@{max_k}"] = 0.0

        metrics["num_valid_queries"] = n_valid_queries
        return metrics

    # ------------------------------------------------------------------
    # Replicate similarity (d=0)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_replicate_similarity(
        S: np.ndarray,
        query_peps: np.ndarray,
        lib_peps: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute mean and std of cosine similarity for replicate pairs (d=0).

        For each query, collects similarity values to all library spectra with
        the same peptide (excluding self-match which is masked to -inf).
        """
        replicate_sims: List[float] = []

        for i, q_pep in enumerate(query_peps):
            # Library columns matching this peptide
            cols = np.where(lib_peps == q_pep)[0]
            sims = S[i, cols]
            # Filter out -inf (self-match)
            valid = sims[np.isfinite(sims)]
            replicate_sims.extend(valid.tolist())

        if replicate_sims:
            arr = np.array(replicate_sims)
            return {
                "replicate_similarity_mean": float(np.mean(arr)),
                "replicate_similarity_std": float(np.std(arr)),
                "replicate_similarity_median": float(np.median(arr)),
                "num_replicate_pairs": len(replicate_sims),
            }
        return {
            "replicate_similarity_mean": 0.0,
            "replicate_similarity_std": 0.0,
            "replicate_similarity_median": 0.0,
            "num_replicate_pairs": 0,
        }

    # ------------------------------------------------------------------
    # Levenshtein edit distance
    # ------------------------------------------------------------------

    @classmethod
    def _levenshtein_distance(cls, s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance between two tokenized sequences (DP)."""
        t1 = cls._tokenize_peptide(s1)
        t2 = cls._tokenize_peptide(s2)
        if t1 == t2:
            return 0
        len1, len2 = len(t1), len(t2)
        if len1 == 0:
            return len2
        if len2 == 0:
            return len1

        # Use single-row DP for memory efficiency
        prev = list(range(len2 + 1))
        for i in range(1, len1 + 1):
            curr = [i] + [0] * len2
            for j in range(1, len2 + 1):
                cost = 0 if t1[i - 1] == t2[j - 1] else 1
                curr[j] = min(
                    curr[j - 1] + 1,  # insertion
                    prev[j] + 1,  # deletion
                    prev[j - 1] + cost,  # substitution
                )
            prev = curr
        return prev[len2]

    # ------------------------------------------------------------------
    # Edit distance profiling
    # ------------------------------------------------------------------

    def _compute_edit_distance_profile(
        self,
        S: np.ndarray,
        query_peps: np.ndarray,
        lib_peps: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute mean similarity binned by peptide edit distance.

        Samples pairs to keep computation tractable, then groups by edit
        distance bins (e.g. d=0, 1, 2, 3, >=5).
        """
        rng = np.random.RandomState(self.sample_seed + 1)
        M, N = S.shape

        # Determine number of pairs to sample
        total_pairs = M * N
        n_sample = min(self.n_ed_sample_pairs, total_pairs)

        # Sample random (row, col) pairs
        if n_sample < total_pairs:
            flat_indices = rng.choice(total_pairs, n_sample, replace=False)
            rows = flat_indices // N
            cols = flat_indices % N
        else:
            rows, cols = np.mgrid[0:M, 0:N]
            rows = rows.ravel()
            cols = cols.ravel()

        # Cache unique peptide pair -> edit distance
        ed_cache: Dict[Tuple[str, str], int] = {}

        # Collect (similarity, edit_distance) pairs
        bins = sorted(self.edit_distance_bins)
        bin_sims: Dict[str, List[float]] = {self._bin_label(b, bins): [] for b in bins}
        # Add overflow bin
        overflow_label = f"d>={bins[-1]}"
        bin_sims[overflow_label] = []

        for idx in range(len(rows)):
            r, c = int(rows[idx]), int(cols[idx])
            sim = S[r, c]
            if not np.isfinite(sim):
                continue

            q_pep = query_peps[r]
            l_pep = lib_peps[c]

            cache_key = (q_pep, l_pep) if q_pep <= l_pep else (l_pep, q_pep)
            if cache_key not in ed_cache:
                ed_cache[cache_key] = self._levenshtein_distance(q_pep, l_pep)
            ed = ed_cache[cache_key]

            label = self._assign_bin(ed, bins)
            bin_sims[label].append(sim)

        # Compute statistics per bin
        profile: Dict[str, Any] = {}
        for label, sims in bin_sims.items():
            if sims:
                arr = np.array(sims, dtype=np.float32)
                profile[label] = {
                    "mean_similarity": float(np.mean(arr)),
                    "std_similarity": float(np.std(arr)),
                    "count": len(sims),
                }
            else:
                profile[label] = {
                    "mean_similarity": None,
                    "std_similarity": None,
                    "count": 0,
                }

        return {
            "edit_distance_profile": profile,
            "num_ed_pairs_sampled": len(rows),
            "num_unique_ed_pairs_cached": len(ed_cache),
        }

    @staticmethod
    def _bin_label(b: int, bins: List[int]) -> str:
        """Create label for an edit distance bin value."""
        return f"d={b}"

    @staticmethod
    def _assign_bin(ed: int, bins: List[int]) -> str:
        """Assign an edit distance to the appropriate bin label."""
        for b in bins:
            if ed == b:
                return f"d={b}"
        # Overflow: >= last bin
        return f"d>={bins[-1]}"

    def _sample_pairs(
        self,
        M: int,
        N: int,
        *,
        seed_offset: int,
        max_pairs: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample row/column index pairs from an M x N matrix."""
        rng = np.random.RandomState(self.sample_seed + seed_offset)
        total_pairs = M * N
        n_sample = min(max_pairs or self.n_ed_sample_pairs, total_pairs)

        if n_sample < total_pairs:
            flat_indices = rng.choice(total_pairs, n_sample, replace=False)
            rows = flat_indices // N
            cols = flat_indices % N
        else:
            rows, cols = np.mgrid[0:M, 0:N]
            rows = rows.ravel()
            cols = cols.ravel()
        return rows, cols

    def _collect_similarity_distance_pairs(
        self,
        S: np.ndarray,
        query_peps: np.ndarray,
        lib_peps: np.ndarray,
        *,
        seed_offset: int,
        max_pairs: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collect sampled cosine similarities with sequence distances."""
        M, N = S.shape
        rows, cols = self._sample_pairs(M, N, seed_offset=seed_offset, max_pairs=max_pairs)

        ed_cache: Dict[Tuple[str, str], int] = {}
        token_len_cache: Dict[str, int] = {}
        cos_sims: List[float] = []
        edit_distances: List[int] = []
        seq_sims: List[float] = []

        for idx in range(len(rows)):
            r, c = int(rows[idx]), int(cols[idx])
            sim = S[r, c]
            if not np.isfinite(sim):
                continue

            q_pep = query_peps[r]
            l_pep = lib_peps[c]

            cache_key = (q_pep, l_pep) if q_pep <= l_pep else (l_pep, q_pep)
            if cache_key not in ed_cache:
                ed_cache[cache_key] = self._levenshtein_distance(q_pep, l_pep)
            ed = ed_cache[cache_key]

            if q_pep not in token_len_cache:
                token_len_cache[q_pep] = len(self._tokenize_peptide(q_pep))
            if l_pep not in token_len_cache:
                token_len_cache[l_pep] = len(self._tokenize_peptide(l_pep))
            max_len = max(token_len_cache[q_pep], token_len_cache[l_pep])
            seq_sim = 1.0 - (ed / max_len) if max_len > 0 else 1.0

            cos_sims.append(float(sim))
            edit_distances.append(ed)
            seq_sims.append(seq_sim)

        return (
            np.array(cos_sims, dtype=np.float32),
            np.array(edit_distances, dtype=np.int32),
            np.array(seq_sims, dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # Correlation (Spearman / Pearson)
    # ------------------------------------------------------------------

    def _compute_correlation(
        self,
        S: np.ndarray,
        query_peps: np.ndarray,
        lib_peps: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute Spearman and Pearson correlation between cosine similarity
        and normalised sequence similarity.

        Sequence similarity: 1 - levenshtein(p1, p2) / max(len(p1), len(p2)).

        Samples pairs to keep computation tractable.
        """
        cos_arr, _, seq_arr = self._collect_similarity_distance_pairs(
            S,
            query_peps,
            lib_peps,
            seed_offset=2,
        )

        if len(cos_arr) < 3:
            return {
                "spearman_r": None,
                "spearman_p": None,
                "pearson_r": None,
                "pearson_p": None,
                "num_correlation_pairs": len(cos_arr),
            }

        try:
            from scipy.stats import pearsonr, spearmanr
        except ImportError as exc:
            raise ImportError("SpectralAnnotationTransferTask requires scipy for correlation metrics.") from exc

        cos_arr = np.array(cos_arr, dtype=np.float64)
        seq_arr = np.array(seq_arr, dtype=np.float64)

        sp_r, sp_p = spearmanr(cos_arr, seq_arr)
        pe_r, pe_p = pearsonr(cos_arr, seq_arr)

        return {
            "spearman_r": float(sp_r),
            "spearman_p": float(sp_p),
            "pearson_r": float(pe_r),
            "pearson_p": float(pe_p),
            "num_correlation_pairs": len(cos_arr),
        }

    # ------------------------------------------------------------------
    # AUPRC / AUROC
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_auprc_auroc(
        S: np.ndarray,
        query_peps: np.ndarray,
        lib_peps: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute AUPRC and AUROC for replicate identification.

        Positive class: library spectrum has the same peptide as the query (d=0).
        Negative class: different peptide (d>0).

        Uses ALL entries in the M x N matrix (excluding masked self-matches).
        """
        # Build binary labels and scores
        # For efficiency, vectorise the comparison
        # query_peps: (M,), lib_peps: (N,)
        # labels[i, j] = 1 if query_peps[i] == lib_peps[j]
        M, N = S.shape

        # Flatten, filtering out -inf (self-matches)
        mask = np.isfinite(S)
        scores = S[mask]

        # Build labels matrix efficiently
        # Expand and compare
        labels_flat = np.zeros(mask.sum(), dtype=np.int32)
        flat_idx = 0
        for i in range(M):
            row_mask = mask[i]
            n_valid = row_mask.sum()
            if n_valid == 0:
                continue
            q_pep = query_peps[i]
            row_labels = (lib_peps[row_mask] == q_pep).astype(np.int32)
            labels_flat[flat_idx : flat_idx + n_valid] = row_labels
            flat_idx += n_valid

        n_pos = labels_flat.sum()
        n_neg = len(labels_flat) - n_pos

        if n_pos == 0 or n_neg == 0:
            return {
                "auprc": 0.0,
                "auroc": 0.0,
                "num_positive_pairs": int(n_pos),
                "num_negative_pairs": int(n_neg),
            }

        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
        except ImportError as exc:
            raise ImportError("SpectralAnnotationTransferTask requires scikit-learn for AUPRC/AUROC metrics.") from exc

        auprc = float(average_precision_score(labels_flat, scores))
        auroc = float(roc_auc_score(labels_flat, scores))

        return {
            "auprc": auprc,
            "auroc": auroc,
            "num_positive_pairs": int(n_pos),
            "num_negative_pairs": int(n_neg),
        }

    @staticmethod
    def _manual_auprc_auroc(
        labels: np.ndarray,
        scores: np.ndarray,
    ) -> Tuple[float, float]:
        """Fallback AUPRC/AUROC without sklearn."""
        # Sort by descending score
        order = np.argsort(scores)[::-1]
        labels_sorted = labels[order]

        # AUROC via Mann-Whitney U statistic
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        pos_ranks = np.where(labels_sorted == 1)[0]
        # Sum of ranks (0-indexed) for positives
        u_stat = pos_ranks.sum() - n_pos * (n_pos - 1) / 2
        auroc = 1.0 - u_stat / (n_pos * n_neg) if (n_pos * n_neg) > 0 else 0.0

        # AUPRC via precision-recall curve
        tp_cumsum = np.cumsum(labels_sorted)
        precision = tp_cumsum / np.arange(1, len(labels_sorted) + 1)
        recall = tp_cumsum / n_pos
        # Compute area under PR curve using trapezoidal rule on recall changes
        recall_diff = np.diff(recall, prepend=0)
        auprc = float(np.sum(precision * recall_diff))

        return auprc, auroc

    # ------------------------------------------------------------------
    # Artifact saving
    # ------------------------------------------------------------------

    def _save_artifact(
        self,
        S: np.ndarray,
        query_peps: np.ndarray,
        lib_peps: np.ndarray,
        query_indices: np.ndarray,
        library_indices: np.ndarray,
        filename_prefix: str = "",
    ) -> Path:
        """Save similarity matrix and peptide labels as compressed .npz."""
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = out_dir / f"{filename_prefix}similarity_matrix.npz"

        # Downcast to float16 to save space
        S_f16 = S.astype(np.float16)

        np.savez_compressed(
            str(artifact_path),
            similarity_matrix=S_f16,
            query_peptides=query_peps,
            library_peptides=lib_peps,
            query_indices=query_indices,
            library_indices=library_indices,
        )

        size_mb = artifact_path.stat().st_size / (1024 * 1024)
        logger.info("Saved similarity matrix artifact: %s (%.1f MB)", artifact_path, size_mb)
        return artifact_path

    def _save_visualizations(
        self,
        S: np.ndarray,
        query_peps: np.ndarray,
        lib_peps: np.ndarray,
        filename_prefix: str = "",
    ) -> Dict[str, str]:
        """Save quick-look plots for the similarity matrix and sequence relation."""
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "similarity_heatmap": str(out_dir / f"{filename_prefix}similarity_heatmap.png"),
            "distance_similarity_plot": str(out_dir / f"{filename_prefix}distance_similarity_plot.png"),
        }
        self._save_similarity_heatmap(S, Path(paths["similarity_heatmap"]))
        self._save_distance_similarity_plot(
            S,
            query_peps,
            lib_peps,
            Path(paths["distance_similarity_plot"]),
        )
        return paths

    @staticmethod
    def _make_subset_prefix(subset_name: str) -> str:
        """Create a safe filename prefix for conditional subset artifacts."""
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in subset_name.strip())
        safe = safe or "subset"
        return f"{safe}_"

    def _save_similarity_heatmap(self, S: np.ndarray, save_path: Path) -> None:
        """Save a downsampled overview heatmap of the rectangular similarity matrix."""
        row_step = max(1, int(np.ceil(S.shape[0] / self.heatmap_max_queries)))
        col_step = max(1, int(np.ceil(S.shape[1] / self.heatmap_max_library)))
        S_view = S[::row_step, ::col_step]
        S_masked = np.ma.masked_invalid(S_view)

        fig, ax = plt.subplots(figsize=(12, 6))
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="lightgray")
        im = ax.imshow(
            S_masked,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=-1.0,
            vmax=1.0,
        )
        ax.set_title("Spectral Annotation Transfer Similarity Heatmap")
        ax.set_xlabel(f"Library spectra (every {col_step} columns)")
        ax.set_ylabel(f"Query peptides (every {row_step} rows)")
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Cosine similarity")
        fig.tight_layout()
        fig.savefig(save_path, dpi=self.plot_dpi, bbox_inches="tight")
        plt.close(fig)

    def _save_distance_similarity_plot(
        self,
        S: np.ndarray,
        query_peps: np.ndarray,
        lib_peps: np.ndarray,
        save_path: Path,
    ) -> None:
        """Save a quick-look plot of edit distance versus cosine similarity."""
        cos_sims, edit_distances, _ = self._collect_similarity_distance_pairs(
            S,
            query_peps,
            lib_peps,
            seed_offset=3,
            max_pairs=self.plot_max_points,
        )
        if len(cos_sims) == 0:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        hb = ax.hexbin(
            edit_distances,
            cos_sims,
            gridsize=35,
            mincnt=1,
            cmap="plasma",
        )
        cbar = fig.colorbar(hb, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Sample count")

        unique_distances = np.unique(edit_distances)
        mean_sims = np.array(
            [cos_sims[edit_distances == d].mean() for d in unique_distances],
            dtype=np.float32,
        )
        ax.plot(unique_distances, mean_sims, color="white", linewidth=2.0, label="Mean similarity")

        ax.set_title("Sequence Edit Distance vs Cosine Similarity")
        ax.set_xlabel("Levenshtein edit distance")
        ax.set_ylabel("Cosine similarity")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(save_path, dpi=self.plot_dpi, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Conditional evaluation
    # ------------------------------------------------------------------

    def _run_conditional_eval(
        self,
        emb: np.ndarray,
        meta: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Run annotation transfer on instrument-filtered subsets."""
        conditional_results: Dict[str, Any] = {}

        for subset_cfg in self.conditional_subsets:
            subset_name = subset_cfg.get("name", "unnamed")

            emb_filt, meta_filt, desc = self.apply_conditional_filter(emb, meta, subset_cfg)
            n_filt = len(emb_filt)

            if n_filt < self.conditional_min_samples:
                conditional_results[subset_name] = {
                    "skipped": True,
                    "reason": f"insufficient samples ({n_filt} < {self.conditional_min_samples})",
                    "n_samples": n_filt,
                    "filter": desc,
                }
                continue

            if self.peptide_key not in meta_filt:
                conditional_results[subset_name] = {
                    "skipped": True,
                    "reason": f"peptide key '{self.peptide_key}' not in filtered metadata",
                    "n_samples": n_filt,
                    "filter": desc,
                }
                continue

            peptides_filt = meta_filt[self.peptide_key]

            # Build query / library on filtered data
            query_indices, library_indices = self._build_query_library(peptides_filt)
            # Keep conditional matching semantics identical to global evaluation:
            # tokenize + canonicalize (e.g. I->L while preserving PTM tokens).
            query_peps = np.array([self._canonicalize_peptide(str(peptides_filt[i]).strip()) for i in query_indices])
            lib_peps = np.array([self._canonicalize_peptide(str(peptides_filt[i]).strip()) for i in library_indices])

            E_q = emb_filt[query_indices].astype(np.float32).copy()
            E_lib = emb_filt[library_indices].astype(np.float32).copy()
            E_q = self._l2_normalise(E_q)
            E_lib = self._l2_normalise(E_lib)

            S = self._compute_similarity_batched(E_q, E_lib, self.batch_size)

            self_col_map = self._build_self_match_map(query_indices, library_indices)
            self._mask_self_matches(S, self_col_map)

            retrieval = self._compute_retrieval_metrics(
                S,
                query_peps,
                lib_peps,
                self_col_map,
                self.k_values,
            )
            replicate = self._compute_replicate_similarity(S, query_peps, lib_peps)
            auprc_auroc = self._compute_auprc_auroc(S, query_peps, lib_peps)

            subset_results = {
                "skipped": False,
                "n_samples": n_filt,
                "filter": desc,
                "num_queries": len(query_indices),
                "num_library": len(library_indices),
                **retrieval,
                **replicate,
                **auprc_auroc,
            }

            subset_prefix = self._make_subset_prefix(subset_name)
            if self.save_matrix_artifact and self.output_dir:
                subset_results["artifact_path"] = str(
                    self._save_artifact(
                        S,
                        query_peps,
                        lib_peps,
                        np.array(query_indices),
                        np.array(library_indices),
                        filename_prefix=subset_prefix,
                    )
                )
            if self.save_plots and self.output_dir:
                subset_results["plot_paths"] = self._save_visualizations(
                    S,
                    query_peps,
                    lib_peps,
                    filename_prefix=subset_prefix,
                )

            conditional_results[subset_name] = subset_results

            logger.info(
                "  %s: recall@1=%.3f, auprc=%.3f (%d queries, %d library)",
                subset_name,
                retrieval.get("recall@1", 0),
                auprc_auroc.get("auprc", 0),
                len(query_indices),
                len(library_indices),
            )

        return conditional_results
