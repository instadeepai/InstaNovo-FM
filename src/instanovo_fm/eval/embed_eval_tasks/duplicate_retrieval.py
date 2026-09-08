"""Duplicate Spectrum Retrieval evaluation task.

This task evaluates how well embeddings can retrieve spectra with identical
peptide sequences. It computes Recall@k and mAP metrics.
"""

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from instanovo_fm.eval.embed_eval_tasks import BaseTask

logger = logging.getLogger(__name__)


class DuplicateRetrievalTask(BaseTask):
    """Duplicate spectrum retrieval evaluation task.

    Evaluates the ability of embeddings to retrieve spectra with identical
    peptide sequences. This serves as a fast sanity check - if embeddings
    collapse or are poor, this task should fail.

    Reports both binary recall (hit or miss) and proportional recall (fraction
    of relevant items found) to better handle large replicate sets.
    """

    name = "Duplicate Spectrum Retrieval"
    description = "Evaluate retrieval of spectra with identical peptide sequences (binary + proportional recall)"
    requires_metadata = True
    requires_faiss = True

    def __init__(self, **kwargs: Any) -> None:
        """Initialise the input."""
        super().__init__(**kwargs)
        self.k_values = kwargs.get("k_values", [1, 5, 10])
        self.peptide_key = kwargs.get("peptide_key", "sequence")
        self.max_samples = kwargs.get("max_samples", 1000)  # Limit for performance
        self.sample_seed = kwargs.get("sample_seed", 42)

        # Instrument-conditional evaluation — same pattern as PeakTypeClassificationTask
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

    def run(self, emb: np.ndarray, meta: Dict[str, np.ndarray], faiss_index: Any) -> Dict[str, Any]:  # type: ignore[override]  # base class run() signature differs across tasks
        """Run the duplicate retrieval evaluation task.

        Args:
            emb: Embeddings array of shape (N, D)
            meta: Metadata dictionary containing peptide sequences
            faiss_index: FAISS index for similarity search

        Returns:
            Dictionary containing retrieval metrics
        """
        # Validate inputs
        self.validate_inputs(emb, meta, faiss_index)

        start_time = time.time()

        # Check if peptide information is available
        if self.peptide_key not in meta:
            return {
                "task_name": self.name,
                "error": f"Peptide information not found in metadata. Available keys: {list(meta.keys())}",
                "execution_time": time.time() - start_time,
            }

        peptides = meta[self.peptide_key]

        # Find duplicate peptides and get detailed statistics
        peptide_groups, peptide_to_indices = self._find_duplicate_peptides(peptides)

        if not peptide_groups:
            return {"task_name": self.name, "error": "No duplicate peptides found in the dataset", "execution_time": time.time() - start_time}

        # Sample peptide groups for evaluation (for performance)
        sampled_groups = self._sample_peptide_groups(peptide_groups, self.max_samples)

        # Compute retrieval metrics
        recall_metrics = self._compute_recall_at_k(emb, sampled_groups, faiss_index, self.k_values)
        max_k = max(self.k_values)
        map_score = self._compute_map(emb, sampled_groups, faiss_index, max_k)

        # Compute detailed statistics about top duplicate sequences
        top_duplicates_stats = self._compute_top_duplicates_stats(peptides, peptide_to_indices, top_n=20)

        execution_time = time.time() - start_time

        # Separate binary and proportional recall metrics for clarity
        binary_recall_metrics = {k: v for k, v in recall_metrics.items() if k.startswith("recall@")}
        prop_recall_metrics = {k: v for k, v in recall_metrics.items() if k.startswith("prop_recall@")}

        logger.info(
            "Global: %d groups, recall@1=%.3f, prop_recall@1=%.3f, map@%d=%.3f",
            len(sampled_groups),
            binary_recall_metrics.get("recall@1", 0),
            prop_recall_metrics.get("prop_recall@1", 0),
            max_k,
            map_score,
        )

        results: dict[str, Any] = {
            "task_name": self.name,
            "num_embeddings": len(emb),
            "num_duplicate_groups": len(peptide_groups),
            "num_sampled_groups": len(sampled_groups),
            "execution_time": execution_time,
            "recall_metrics": binary_recall_metrics,
            "prop_recall_metrics": prop_recall_metrics,
            f"map@{max_k}": map_score,
            "summary": {
                "total_duplicate_spectra": sum(len(group) for group in peptide_groups),
                "avg_group_size": np.mean([len(group) for group in peptide_groups]),
                "max_group_size": max(len(group) for group in peptide_groups),
            },
            "aggregate_stats": {
                # Binary recall stats
                "binary_recall_mean": np.mean(list(binary_recall_metrics.values())),
                "binary_recall_std": np.std(list(binary_recall_metrics.values())),
                # Proportional recall stats
                "prop_recall_mean": np.mean(list(prop_recall_metrics.values())),
                "prop_recall_std": np.std(list(prop_recall_metrics.values())),
                # Difference between binary and proportional recall
                "recall_difference_mean": np.mean(list(binary_recall_metrics.values())) - np.mean(list(prop_recall_metrics.values())),
            },
            "top_duplicates": top_duplicates_stats,
        }

        # Conditional evaluation — run retrieval within instrument-filtered subsets
        if self.enable_conditional_eval:
            results["conditional_eval"] = self._run_conditional_eval(emb, meta)
            results["execution_time"] = time.time() - start_time

        return results

    def get_loggable_metrics(self, task_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract key retrieval metrics for MLflow logging.

        Logs recall@1 (hit rate), prop_recall@1 (completeness), and MAP@20
        (ranking quality) — global and per-condition. These three metrics
        capture distinct aspects of retrieval and are sufficient for ablation
        comparison.
        """
        if "error" in task_results:
            return {}

        loggable: Dict[str, float] = {}

        # Global: recall@1, prop_recall@1, map@20
        recall = task_results.get("recall_metrics", {})
        prop_recall = task_results.get("prop_recall_metrics", {})
        if "recall@1" in recall:
            loggable["recall@1"] = float(recall["recall@1"])
        if "prop_recall@1" in prop_recall:
            loggable["prop_recall@1"] = float(prop_recall["prop_recall@1"])
        if "map@20" in task_results:
            loggable["map@20"] = float(task_results["map@20"])

        # Conditional: same three metrics per subset
        for subset_name, subset_info in task_results.get("conditional_eval", {}).items():
            if isinstance(subset_info, dict) and not subset_info.get("skipped", True):
                sr = subset_info.get("recall_metrics", {})
                sp = subset_info.get("prop_recall_metrics", {})
                if "recall@1" in sr:
                    loggable[f"cond_{subset_name}_recall@1"] = float(sr["recall@1"])
                if "prop_recall@1" in sp:
                    loggable[f"cond_{subset_name}_prop_recall@1"] = float(sp["prop_recall@1"])
                for key in subset_info:
                    if key.startswith("map@"):
                        loggable[f"cond_{subset_name}_{key}"] = float(subset_info[key])

        return loggable

    def _find_duplicate_peptides(self, peptides: np.ndarray) -> Tuple[List[List[int]], Dict[str, List[int]]]:
        """Find groups of indices that have identical peptide sequences.

        Args:
            peptides: Array of peptide sequences

        Returns:
            Tuple of (duplicate_groups, peptide_to_indices_dict):
                - duplicate_groups: List of lists, where each inner list contains indices of spectra with identical peptides
                - peptide_to_indices_dict: Dictionary mapping peptide sequences to their indices
        """
        # Create mapping from peptide to indices
        peptide_to_indices = defaultdict(list)
        for i, peptide in enumerate(peptides):
            # Convert to string for consistent comparison
            peptide_str = str(peptide).strip()
            if peptide_str:  # Skip empty peptides
                peptide_to_indices[peptide_str].append(i)

        # Return only groups with more than one spectrum
        duplicate_groups = [indices for indices in peptide_to_indices.values() if len(indices) > 1]

        return duplicate_groups, dict(peptide_to_indices)

    def _sample_peptide_groups(self, peptide_groups: List[List[int]], max_samples: int) -> List[List[int]]:
        """Sample peptide groups for evaluation to control performance.

        Args:
            peptide_groups: List of peptide groups
            max_samples: Maximum number of groups to sample

        Returns:
            Sampled list of peptide groups
        """
        if len(peptide_groups) <= max_samples:
            return peptide_groups

        # Use deterministic sampling for reproducibility
        rng = np.random.RandomState(self.sample_seed)
        sampled_indices = rng.choice(len(peptide_groups), max_samples, replace=False)
        sampled_groups = [peptide_groups[i] for i in sorted(sampled_indices)]

        return sampled_groups

    def _compute_top_duplicates_stats(self, peptides: np.ndarray, peptide_to_indices: Dict[str, List[int]], top_n: int = 20) -> Dict[str, Any]:
        """Compute detailed statistics about the top N most duplicated peptide sequences.

        Args:
            peptides: Array of all peptide sequences
            peptide_to_indices: Dictionary mapping peptide sequences to their indices
            top_n: Number of top duplicates to report

        Returns:
            Dictionary containing statistics about top duplicates
        """
        total_spectra = len(peptides)

        # Sort peptides by number of occurrences (descending)
        sorted_peptides = sorted(peptide_to_indices.items(), key=lambda x: len(x[1]), reverse=True)

        # Get top N duplicated peptides (those with >1 occurrence)
        top_duplicates = [(pep, indices) for pep, indices in sorted_peptides if len(indices) > 1][:top_n]

        # Compute statistics for each top duplicate
        top_duplicates_list = []
        cumulative_count = 0

        for rank, (peptide, indices) in enumerate(top_duplicates, 1):
            count = len(indices)
            cumulative_count += count
            percentage = (count / total_spectra) * 100
            cumulative_percentage = (cumulative_count / total_spectra) * 100

            # Compute peptide properties
            peptide_length = len(peptide)

            # Truncate very long peptides for display
            display_peptide = peptide if len(peptide) <= 60 else f"{peptide[:57]}..."

            top_duplicates_list.append(
                {
                    "rank": rank,
                    "peptide": display_peptide,
                    "full_peptide": peptide,  # Keep full sequence for reference
                    "peptide_length": peptide_length,
                    "count": count,
                    "percentage": round(percentage, 3),
                    "cumulative_count": cumulative_count,
                    "cumulative_percentage": round(cumulative_percentage, 3),
                }
            )

        # Compute overall statistics
        total_duplicates = sum(len(indices) for pep, indices in peptide_to_indices.items() if len(indices) > 1)
        total_unique_peptides = len(peptide_to_indices)
        total_duplicated_peptides = sum(1 for indices in peptide_to_indices.values() if len(indices) > 1)

        # Top N coverage statistics
        top_n_coverage = cumulative_count if top_duplicates else 0
        top_n_coverage_pct = (top_n_coverage / total_spectra) * 100 if total_spectra > 0 else 0

        return {
            "top_n": top_n,
            "top_duplicates_list": top_duplicates_list,
            "overall_stats": {
                "total_spectra": total_spectra,
                "total_unique_peptides": total_unique_peptides,
                "total_duplicated_peptides": total_duplicated_peptides,
                "total_duplicate_spectra": total_duplicates,
                "duplicate_spectra_percentage": round((total_duplicates / total_spectra) * 100, 2) if total_spectra > 0 else 0,
                "unique_peptide_percentage": round((total_unique_peptides / total_spectra) * 100, 2) if total_spectra > 0 else 0,
            },
            "top_n_coverage": {
                "spectra_count": top_n_coverage,
                "percentage_of_total": round(top_n_coverage_pct, 2),
                "description": (
                    f"Top {len(top_duplicates)} most duplicated peptides account for {top_n_coverage} spectra ({top_n_coverage_pct:.1f}% of dataset)"
                ),
            },
        }

    def _compute_recall_at_k(self, emb: np.ndarray, peptide_groups: List[List[int]], faiss_index: Any, k_values: List[int]) -> Dict[str, float]:
        """Compute Recall@k for each k value.

        Args:
            emb: Embeddings array
            peptide_groups: Groups of indices with identical peptides
            faiss_index: FAISS index for similarity search
            k_values: List of k values to compute recall for

        Returns:
            Dictionary mapping k values to recall scores (both binary and proportional)
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu is required for similarity search") from None

        # Fix: Assert that index vectors are properly normalized
        try:
            if hasattr(faiss_index, "reconstruct") and faiss_index.ntotal > 0:
                # Try to reconstruct a vector from the index to check normalization
                dimension = emb.shape[1]
                reconstructed_vector = np.zeros(dimension, dtype=np.float32)
                faiss_index.reconstruct(0, reconstructed_vector)
                norm = np.linalg.norm(reconstructed_vector)
                assert np.allclose(norm, 1.0, atol=1e-3), f"Index vector not normalized: norm={norm}"
        except Exception:
            # If we can't check index normalization, at least verify input embeddings
            input_norms = np.linalg.norm(emb, axis=1)
            mean_norm = np.mean(input_norms)
            assert np.allclose(mean_norm, 1.0, atol=1e-3), f"Input embeddings not normalized: mean_norm={mean_norm}"

        max_k = max(k_values)
        recall_scores = {f"recall@{k}": 0.0 for k in k_values}
        prop_recall_scores = {f"prop_recall@{k}": 0.0 for k in k_values}
        total_queries = 0

        for group in peptide_groups:
            if len(group) < 2:
                continue

            # Use each spectrum in the group as a query
            for query_idx in group:
                # Get the query embedding
                query_emb = emb[query_idx : query_idx + 1].astype(np.float32)
                # Normalize query embedding for consistent similarity computation
                # Note: Index vectors should also be L2-normalized for proper cosine similarity
                faiss.normalize_L2(query_emb)

                # Search for similar embeddings
                distances, indices = faiss_index.search(query_emb, max_k + 1)  # +1 to account for self

                # Remove the query itself from results
                retrieved_indices = indices[0][1 : max_k + 1]  # Skip first (self)

                # Other spectra in the same peptide group (excluding query)
                relevant_indices = [idx for idx in group if idx != query_idx]

                # Compute recall for each k value
                for k in k_values:
                    # Check if any relevant spectrum is in the top-k results
                    top_k_indices = retrieved_indices[:k]

                    # Binary recall: hit or miss (1 if any relevant item found, 0 otherwise)
                    # This masks progress for large replicate sets once you cross the "≥1" threshold
                    if any(idx in relevant_indices for idx in top_k_indices):
                        recall_scores[f"recall@{k}"] += 1.0

                    # Proportional recall: fraction of relevant items found
                    # This better rewards finding more relevant items in large replicate sets
                    # Formula: (#hits) / (group_size - 1) where group_size - 1 excludes the query
                    hits = sum(1 for idx in top_k_indices if idx in relevant_indices)
                    prop_hits = hits / (len(group) - 1)  # group_size - 1 (excluding query)
                    prop_recall_scores[f"prop_recall@{k}"] += prop_hits

                total_queries += 1

        # Normalize by total number of queries
        if total_queries > 0:
            for k in k_values:
                recall_scores[f"recall@{k}"] /= total_queries
                prop_recall_scores[f"prop_recall@{k}"] /= total_queries

        # Combine both metrics
        all_scores: dict[str, Any] = {**recall_scores, **prop_recall_scores}
        return all_scores

    def _compute_map(self, emb: np.ndarray, peptide_groups: List[List[int]], faiss_index: Any, max_k: int) -> float:
        """Compute Mean Average Precision (mAP).

        Args:
            emb: Embeddings array
            peptide_groups: Groups of indices with identical peptides
            faiss_index: FAISS index for similarity search
            max_k: Maximum k for retrieval

        Returns:
            mAP score
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu is required for similarity search") from None

        # Fix: Assert that index vectors are properly normalized
        try:
            if hasattr(faiss_index, "reconstruct") and faiss_index.ntotal > 0:
                # Try to reconstruct a vector from the index to check normalization
                dimension = emb.shape[1]
                reconstructed_vector = np.zeros(dimension, dtype=np.float32)
                faiss_index.reconstruct(0, reconstructed_vector)
                norm = np.linalg.norm(reconstructed_vector)
                assert np.allclose(norm, 1.0, atol=1e-3), f"Index vector not normalized: norm={norm}"
        except Exception:
            # If we can't check index normalization, at least verify input embeddings
            input_norms = np.linalg.norm(emb, axis=1)
            mean_norm = np.mean(input_norms)
            assert np.allclose(mean_norm, 1.0, atol=1e-3), f"Input embeddings not normalized: mean_norm={mean_norm}"

        aps = []

        for group in peptide_groups:
            if len(group) < 2:
                continue

            # Use each spectrum in the group as a query
            for query_idx in group:
                # Get the query embedding
                query_emb = emb[query_idx : query_idx + 1].astype(np.float32)
                # Normalize query embedding for consistent similarity computation
                # Note: Index vectors should also be L2-normalized for proper cosine similarity
                faiss.normalize_L2(query_emb)

                # Search for similar embeddings
                distances, indices = faiss_index.search(query_emb, max_k + 1)  # +1 to account for self

                # Remove the query itself from results
                retrieved_indices = indices[0][1 : max_k + 1]  # Skip first (self)

                # Other spectra in the same peptide group (excluding query)
                relevant_indices = [idx for idx in group if idx != query_idx]

                # Compute average precision for this query
                ap = self._compute_average_precision(retrieved_indices, relevant_indices)
                aps.append(ap)

        # Return mean average precision
        return float(np.mean(aps)) if aps else 0.0

    def _compute_average_precision(self, retrieved_indices: np.ndarray, relevant_indices: List[int]) -> float:
        """Compute Average Precision for a single query.

        Fixed implementation: properly handles truncation at max_k by dividing
        by min(len(relevant), len(retrieved)) instead of len(relevant).

        Args:
            retrieved_indices: Indices of retrieved items (truncated at max_k)
            relevant_indices: Indices of relevant items (same peptide)

        Returns:
            Average precision score (AP@k)
        """
        if not relevant_indices:
            return 0.0

        # Convert relevant_indices to set for O(1) lookup
        relevant_set = set(relevant_indices)

        # Count relevant items at each position
        hits = 0
        sum_precision = 0.0

        for i, idx in enumerate(retrieved_indices, 1):  # i = rank (1-based)
            if idx in relevant_set:
                hits += 1
                precision_at_k = hits / i
                sum_precision += precision_at_k

        # Normalize by min(len(relevant), len(retrieved)) for AP@k
        denominator = min(len(relevant_indices), len(retrieved_indices))
        return sum_precision / denominator if denominator > 0 else 0.0

    # ------------------------------------------------------------------
    # Instrument-conditional evaluation
    # ------------------------------------------------------------------

    def _run_conditional_eval(
        self,
        emb: np.ndarray,
        meta: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Run duplicate retrieval on instrument-filtered subsets.

        Builds a local FAISS index per subset so retrieval is strictly
        within-condition.
        """
        try:
            import faiss
        except ImportError:
            logger.warning("faiss-cpu not available — skipping conditional eval")
            return {}

        conditional_results: Dict[str, Any] = {}

        for subset_cfg in self.conditional_subsets:
            subset_name = subset_cfg.get("name", "unnamed")

            emb_filt, meta_filt, desc = self.apply_conditional_filter(emb, meta, subset_cfg)
            n_filt = len(emb_filt)

            if n_filt < self.conditional_min_samples:
                logger.warning(
                    "  Skipping '%s': only %d samples (min=%d)",
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

            # Check peptide availability
            if self.peptide_key not in meta_filt:
                conditional_results[subset_name] = {
                    "skipped": True,
                    "reason": f"peptide key '{self.peptide_key}' not in filtered metadata",
                    "n_samples": n_filt,
                    "filter": desc,
                }
                continue

            # Build local FAISS index on filtered embeddings
            emb_norm = emb_filt.astype(np.float32).copy()
            faiss.normalize_L2(emb_norm)
            local_index = faiss.IndexFlatIP(emb_norm.shape[1])
            local_index.add(emb_norm)

            # Find duplicates in filtered subset
            peptides_filt = meta_filt[self.peptide_key]
            peptide_groups, _ = self._find_duplicate_peptides(peptides_filt)

            if not peptide_groups:
                conditional_results[subset_name] = {
                    "skipped": True,
                    "reason": "no duplicate peptides in filtered subset",
                    "n_samples": n_filt,
                    "filter": desc,
                }
                continue

            sampled_groups = self._sample_peptide_groups(peptide_groups, self.max_samples)

            # Compute metrics
            recall_metrics = self._compute_recall_at_k(
                emb_norm,
                sampled_groups,
                local_index,
                self.k_values,
            )
            max_k = max(self.k_values)
            map_score = self._compute_map(emb_norm, sampled_groups, local_index, max_k)

            binary_recall = {k: v for k, v in recall_metrics.items() if k.startswith("recall@")}
            prop_recall = {k: v for k, v in recall_metrics.items() if k.startswith("prop_recall@")}

            conditional_results[subset_name] = {
                "skipped": False,
                "n_samples": n_filt,
                "filter": desc,
                "num_duplicate_groups": len(peptide_groups),
                "num_sampled_groups": len(sampled_groups),
                "recall_metrics": binary_recall,
                "prop_recall_metrics": prop_recall,
                f"map@{max_k}": map_score,
                "summary": {
                    "total_duplicate_spectra": sum(len(g) for g in peptide_groups),
                    "avg_group_size": float(np.mean([len(g) for g in peptide_groups])),
                    "max_group_size": max(len(g) for g in peptide_groups),
                },
            }

            logger.info(
                "  %s: recall@1=%.3f, recall@5=%.3f, map@%d=%.3f (%d groups, %d/%d spectra)",
                subset_name,
                binary_recall.get("recall@1", 0),
                binary_recall.get("recall@5", 0),
                max_k,
                map_score,
                len(sampled_groups),
                n_filt,
                len(emb),
            )

        return conditional_results
