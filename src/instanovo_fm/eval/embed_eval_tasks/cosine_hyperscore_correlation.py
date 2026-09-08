"""
Cosine vs. Hyperscore Correlation evaluation task.

This task computes the Spearman correlation between cosine similarity
of embeddings and Hyperscore values to check if "close in latent space"
corresponds to "good database match".
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy.stats import spearmanr
import time
from collections import defaultdict

from instanovo_fm.eval.embed_eval_tasks import BaseTask


class CosineHyperscoreCorrelationTask(BaseTask):
    """
    Cosine vs. Hyperscore correlation evaluation task.

    For each spectrum, computes cosine similarity with other spectra of the same peptide
    and correlates this with the Hyperscore value to check if embeddings reflect
    traditional database search quality.
    """

    name = "Cosine vs. Hyperscore Correlation"
    description = "Compute Spearman correlation between cosine similarity and Hyperscore values"
    requires_metadata = True
    requires_faiss = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.peptide_key = kwargs.get('peptide_key', 'sequence')
        self.hyperscore_key = kwargs.get('hyperscore_key', 'hyperscore')
        self.max_samples = kwargs.get('max_samples', 10000)  # Limit for performance
        self.min_peptide_spectra = kwargs.get('min_peptide_spectra', 2)  # Min spectra per peptide
        self.random_state = kwargs.get('random_state', 42)
        self.use_log_expectation = kwargs.get('use_log_expectation', False)  # Alternative to hyperscore
        self.expectation_key = kwargs.get('expectation_key', 'expectation')  # For log expectation
        self.cosine_metric = kwargs.get('cosine_metric', 'mean')  # 'mean', 'median', or 'max'
        self.max_peptide_spectra = kwargs.get('max_peptide_spectra', 20)  # Limit large peptides for performance

    def run(self, emb: np.ndarray, meta: Dict[str, np.ndarray], faiss_index: Any = None) -> Dict[str, Any]:
        """
        Run the cosine vs. hyperscore correlation evaluation task.

        Args:
            emb: Embeddings array of shape (N, D)
            meta: Metadata dictionary containing peptide and hyperscore information
            faiss_index: Not used for this task

        Returns:
            Dictionary containing correlation results
        """
        # Validate inputs
        self.validate_inputs(emb, meta, faiss_index)

        start_time = time.time()

        # Check if required metadata is available
        if self.peptide_key not in meta:
            return {
                'task_name': self.name,
                'error': f"Peptide information not found in metadata. Available keys: {list(meta.keys())}",
                'execution_time': time.time() - start_time
            }

        # Check for hyperscore or expectation values
        score_key = None
        if self.use_log_expectation:
            if self.expectation_key in meta:
                score_key = self.expectation_key
            else:
                return {
                    'task_name': self.name,
                    'error': f"Expectation values not found in metadata. Available keys: {list(meta.keys())}",
                    'execution_time': time.time() - start_time
                }
        else:
            if self.hyperscore_key in meta:
                score_key = self.hyperscore_key
            else:
                return {
                    'task_name': self.name,
                    'error': f"Hyperscore values not found in metadata. Available keys: {list(meta.keys())}",
                    'execution_time': time.time() - start_time
                }

        peptides = meta[self.peptide_key]
        scores = meta[score_key]

        # Subsample data if needed
        if len(emb) > self.max_samples:
            indices = np.random.choice(len(emb), self.max_samples, replace=False)
            emb_sampled = emb[indices]
            peptides_sampled = peptides[indices]
            scores_sampled = scores[indices]
        else:
            emb_sampled = emb
            peptides_sampled = peptides
            scores_sampled = scores

        # Process scores (convert expectation to -log if needed)
        if self.use_log_expectation:
            # Convert expectation to -log(expectation) for better correlation
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            processed_scores = -np.log(np.maximum(scores_sampled, epsilon))
            score_name = "negative_log_expectation"
        else:
            processed_scores = scores_sampled
            score_name = "hyperscore"

        # Compute correlations
        correlation_results = self._compute_cosine_hyperscore_correlation(
            emb_sampled, peptides_sampled, processed_scores
        )

        execution_time = time.time() - start_time

        results = {
            'task_name': self.name,
            'num_embeddings': len(emb),
            'num_sampled': len(emb_sampled),
            'num_peptides': len(np.unique(peptides_sampled)),
            'num_correlation_pairs': correlation_results['num_pairs'],
            'execution_time': execution_time,
            'correlation_results': correlation_results,
            'config': {
                'peptide_key': self.peptide_key,
                'score_key': score_key,
                'score_name': score_name,
                'max_samples': self.max_samples,
                'min_peptide_spectra': self.min_peptide_spectra,
                'use_log_expectation': self.use_log_expectation,
                'random_state': self.random_state,
                'cosine_metric': self.cosine_metric,
                'max_peptide_spectra': self.max_peptide_spectra
            }
        }

        return results

    def _compute_cosine_hyperscore_correlation(self, emb: np.ndarray, peptides: np.ndarray,
                                             scores: np.ndarray) -> Dict[str, Any]:
        """
        Compute cosine similarity vs hyperscore correlation for each spectrum.

        Fixed implementation: computes one summary statistic per spectrum instead of
        creating multiple pairs per spectrum, which was causing correlation to collapse to zero.

        Args:
            emb: Embeddings array
            peptides: Peptide labels array
            scores: Hyperscore or -log(expectation) values array

        Returns:
            Dictionary with correlation results
        """
        # Group spectra by peptide
        peptide_groups = defaultdict(list)
        for i, peptide in enumerate(peptides):
            peptide_groups[peptide].append(i)

        # Filter peptides with sufficient spectra
        valid_peptides = {
            peptide: indices for peptide, indices in peptide_groups.items()
            if len(indices) >= self.min_peptide_spectra
        }

        if not valid_peptides:
            return {
                'error': f"No peptides with at least {self.min_peptide_spectra} spectra found",
                'num_pairs': 0
            }

        # L2 normalize embeddings once for efficiency
        emb_normalized = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-12)

        # Collect one summary statistic per spectrum
        mean_cosines = []
        score_values = []

        for peptide, indices in valid_peptides.items():
            if len(indices) < 2:
                continue

            # Subsample large peptides for performance
            if len(indices) > self.max_peptide_spectra:
                rng = np.random.RandomState(self.random_state)
                indices = rng.choice(indices, self.max_peptide_spectra, replace=False)

            # Get normalized embeddings and scores for this peptide
            peptide_emb = emb_normalized[indices]
            peptide_scores = scores[indices]

            # Compute full cosine similarity matrix
            cos_matrix = peptide_emb @ peptide_emb.T

            # Remove self-similarity (set diagonal to NaN)
            np.fill_diagonal(cos_matrix, np.nan)

            # Compute summary statistic for each spectrum
            if self.cosine_metric == 'mean':
                spectrum_cosines = np.nanmean(cos_matrix, axis=1)
            elif self.cosine_metric == 'median':
                spectrum_cosines = np.nanmedian(cos_matrix, axis=1)
            elif self.cosine_metric == 'max':
                spectrum_cosines = np.nanmax(cos_matrix, axis=1)
            else:
                raise ValueError(f"Unknown cosine_metric: {self.cosine_metric}")

            # Add to our lists (one pair per spectrum)
            mean_cosines.extend(spectrum_cosines.tolist())
            score_values.extend(peptide_scores.tolist())

        if not mean_cosines:
            return {
                'error': "No valid cosine-score pairs found",
                'num_pairs': 0
            }

        # Convert to numpy arrays
        cosine_array = np.array(mean_cosines)
        score_array = np.array(score_values)

        # Compute Spearman correlation
        try:
            spearman_corr, spearman_pvalue = spearmanr(cosine_array, score_array)
        except Exception as e:
            return {
                'error': f"Failed to compute Spearman correlation: {e}",
                'num_pairs': len(mean_cosines)
            }

        # Compute Pearson correlation
        pearson_corr = np.corrcoef(cosine_array, score_array)[0, 1] if len(cosine_array) > 1 else 0.0

        # Compute additional statistics
        cosine_stats = {
            'mean': float(np.mean(cosine_array)),
            'std': float(np.std(cosine_array)),
            'min': float(np.min(cosine_array)),
            'max': float(np.max(cosine_array))
        }

        score_stats = {
            'mean': float(np.mean(score_array)),
            'std': float(np.std(score_array)),
            'min': float(np.min(score_array)),
            'max': float(np.max(score_array))
        }

        return {
            'spearman_correlation': float(spearman_corr),
            'spearman_pvalue': float(spearman_pvalue),
            'pearson_correlation': float(pearson_corr),
            'num_pairs': len(mean_cosines),
            'cosine_stats': cosine_stats,
            'score_stats': score_stats,
            'cosine_metric': self.cosine_metric,
            'max_peptide_spectra': self.max_peptide_spectra
        }
