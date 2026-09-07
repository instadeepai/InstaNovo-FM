"""
Embedding statistics evaluation task.

This task performs comprehensive statistical analysis on spectrum embeddings including:
- L2-normalization detection and adaptive metric selection
- Isotropy and anisotropy analysis (centroid norm, uniformity, effective rank)
- Dimensionality analysis (PCA energy, participation ratio, eigenvalue spectrum)
- Similarity distribution analysis (pairwise cosine similarity)
For L2-normalized embeddings (detected automatically), trivial norm metrics are skipped
and isotropic reference baselines are computed for contextualizing all metrics.
"""

import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
from sklearn.decomposition import PCA
import warnings
import matplotlib.pyplot as plt


from instanovo_fm.eval.embed_eval_tasks import BaseTask


class EmbeddingStatisticsTask(BaseTask):
    """
    Comprehensive embedding statistics task with L2-normalization awareness.

    Performs various analyses on spectrum embeddings to understand
    their quality, structure, and properties. Automatically detects
    L2-normalized embeddings and adapts metrics accordingly.
    """

    name = "embedding_statistics"
    description = "Descriptive statistics, PCA, isotropy and similarity diagnostics for embeddings."
    requires_metadata = False
    requires_faiss = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_components_pca = kwargs.get('n_components_pca', 150)
        self.compute_pca = kwargs.get('compute_pca', True)
        self.compute_similarity_dist = kwargs.get('compute_similarity_dist', True)
        self.num_pairs_similarity = kwargs.get('num_pairs_similarity', 50000)
        self.uniformity_t = kwargs.get('uniformity_t', 2.0)
        self.seed = kwargs.get('seed', 42)

        # Dashboard configuration
        self.output_dir = kwargs.get('output_dir', None)
        self.create_dashboard = kwargs.get('create_dashboard', True)
        self.dashboard_figsize = kwargs.get('dashboard_figsize', (16, 10))
        self.dashboard_dpi = kwargs.get('dashboard_dpi', 150)

    def run(self, emb: np.ndarray, meta: Dict[str, np.ndarray], faiss_index: Any) -> Dict[str, Any]:
        """
        Run the embedding statistics task.

        Args:
            emb: Embeddings array of shape (N, D)
            meta: Metadata dictionary (not used in this task)
            faiss_index: FAISS index (not used in this task)

        Returns:
            Dictionary containing analysis results
        """
        self.validate_inputs(emb, meta, faiss_index)

        D = emb.shape[1]
        results = {
            'task_name': self.name,
            'num_embeddings': len(emb),
            'embedding_dim': D,
        }

        # Detect L2 normalization
        norms = np.linalg.norm(emb, axis=1)
        is_l2_normalized = bool(np.std(norms) < 1e-6 and abs(np.mean(norms) - 1.0) < 1e-3)
        results['is_l2_normalized'] = is_l2_normalized

        # Isotropic baselines for this dimensionality
        isotropic_centroid_norm = 1.0 / np.sqrt(D)
        isotropic_per_dim_std = 1.0 / np.sqrt(D)
        results['isotropic_baselines'] = {
            'centroid_norm': float(isotropic_centroid_norm),
            'per_dim_std': float(isotropic_per_dim_std),
            'embedding_dim': D,
        }

        # Basic embedding statistics
        results.update(self._compute_basic_stats(emb, norms, is_l2_normalized, D))

        # Dimensionality analysis (PCA energy, eigenvalue spectrum)
        if self.compute_pca:
            results.update(self._compute_pca_analysis(emb, D))

        # Similarity distribution analysis (pairwise cosine)
        if self.compute_similarity_dist:
            results.update(self._compute_similarity_distribution(emb, is_l2_normalized))

        # Uniformity metric (Wang & Isola, 2020)
        results.update(self._compute_uniformity(emb, is_l2_normalized))

        # Quality validation
        validation_results = self.validate_embedding_quality(results)
        results['quality_validation'] = validation_results

        # Create quality dashboard if enabled
        if self.create_dashboard:
            try:
                dashboard_path = self._create_quality_dashboard(results)
                if dashboard_path:
                    results['dashboard_path'] = str(dashboard_path)
            except Exception as e:
                warnings.warn(f"Failed to create quality dashboard: {e}")

        return results

    def _compute_basic_stats(
        self, emb: np.ndarray, norms: np.ndarray, is_l2_normalized: bool, D: int
    ) -> Dict[str, Any]:
        """
        Compute basic embedding statistics with L2-normalization awareness.

        For L2-normalized embeddings, trivial norm metrics are skipped and
        anisotropy-focused metrics are computed instead.
        """
        stats: Dict[str, Any] = {}

        # Only include norm stats when embeddings are NOT L2-normalized
        if not is_l2_normalized:
            mean_norm = float(np.mean(norms))
            std_norm = float(np.std(norms))
            norm_cv = std_norm / mean_norm if mean_norm > 0 else 0.0
            stats['mean_norm'] = mean_norm
            stats['std_norm'] = std_norm
            stats['min_norm'] = float(np.min(norms))
            stats['max_norm'] = float(np.max(norms))
            stats['norm_coefficient_of_variation'] = float(norm_cv)

        # Per-dimension variance utilization
        emb_centered = emb - emb.mean(axis=0, keepdims=True)
        dim_std = emb_centered.std(axis=0)
        per_dim_std_min = float(np.min(dim_std)) if dim_std.size > 0 else 0.0
        per_dim_std_mean = float(np.mean(dim_std)) if dim_std.size > 0 else 0.0
        stats['per_dim_std_min'] = per_dim_std_min
        stats['per_dim_std_mean'] = per_dim_std_mean

        # Centroid norm (was collapse_indicator) — measures angular concentration
        mean_vector = emb.mean(axis=0)
        centroid_norm = float(np.linalg.norm(mean_vector))
        stats['centroid_norm'] = centroid_norm

        # Anisotropy ratio: centroid_norm relative to isotropic expectation
        isotropic_centroid = 1.0 / np.sqrt(D)
        stats['anisotropy_ratio'] = float(centroid_norm / isotropic_centroid) if isotropic_centroid > 0 else 0.0

        # Isotropy deficit: per-dim utilization relative to isotropic expectation
        isotropic_per_dim = 1.0 / np.sqrt(D)
        stats['isotropy_deficit'] = float(per_dim_std_mean / isotropic_per_dim) if isotropic_per_dim > 0 else 0.0

        # Dead dimensions: dims with near-zero variance
        dead_threshold = 0.001
        dead_dims = int(np.sum(dim_std < dead_threshold))
        stats['dead_dimensions'] = dead_dims
        stats['dead_dimensions_fraction'] = float(dead_dims / D) if D > 0 else 0.0

        return {'basic_stats': stats}

    def _compute_pca_analysis(self, emb: np.ndarray, D: int) -> Dict[str, Any]:
        """
        Compute PCA analysis with ceiling auto-increase and effective rank.

        Includes:
        - PCA energy captured by top-1 and top-5 components
        - Number of components for 90%, 95%, 99% variance (with ceiling detection)
        - Participation ratio (effective dimensionality)
        - Effective rank (entropy-based, Roy & Vetterli 2007)
        - Eigenvalue spectrum for visualization
        """
        try:
            N = emb.shape[0]
            n_components = min(self.n_components_pca, D, N - 1)
            if n_components < 2:
                return {'pca_analysis': {'error': 'Not enough dimensions for PCA'}}

            svd_solver = "randomized" if n_components < min(N, D) // 2 else "full"
            pca = PCA(n_components=n_components, svd_solver=svd_solver, random_state=self.seed)
            pca.fit(emb)

            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            lambda_ = pca.explained_variance_

            def _nth_component_for(threshold, cum):
                idx = np.searchsorted(cum, threshold)
                return int(idx + 1) if idx < len(cum) else int(len(cum))

            n_components_90 = _nth_component_for(0.90, cumulative_variance)
            n_components_95 = _nth_component_for(0.95, cumulative_variance)
            n_components_99 = _nth_component_for(0.99, cumulative_variance)

            n_computed = len(cumulative_variance)

            # Detect PCA ceiling hit (95% threshold not reached)
            pca_ceiling_hit = n_components_95 >= n_computed

            # Auto-retry with more components if ceiling hit
            if pca_ceiling_hit:
                expanded_n = min(2 * self.n_components_pca, D, N - 1)
                if expanded_n > n_components:
                    svd_solver_exp = "randomized" if expanded_n < min(N, D) // 2 else "full"
                    pca_exp = PCA(n_components=expanded_n, svd_solver=svd_solver_exp, random_state=self.seed)
                    pca_exp.fit(emb)
                    explained_variance_ratio = pca_exp.explained_variance_ratio_
                    cumulative_variance = np.cumsum(explained_variance_ratio)
                    lambda_ = pca_exp.explained_variance_
                    n_components_90 = _nth_component_for(0.90, cumulative_variance)
                    n_components_95 = _nth_component_for(0.95, cumulative_variance)
                    n_components_99 = _nth_component_for(0.99, cumulative_variance)
                    n_computed = len(cumulative_variance)
                    pca_ceiling_hit = n_components_95 >= n_computed

            # Participation ratio
            participation_ratio = float((np.sum(lambda_) ** 2) / np.sum(lambda_ ** 2))

            # Effective rank (Roy & Vetterli, 2007): entropy-based dimensionality
            p = lambda_ / np.sum(lambda_)
            # Avoid log(0) by filtering near-zero eigenvalues
            p_nonzero = p[p > 1e-12]
            effective_rank = float(np.exp(-np.sum(p_nonzero * np.log(p_nonzero))))

            pca_energy_top1 = float(explained_variance_ratio[0])
            pca_energy_top5 = float(np.sum(explained_variance_ratio[:min(5, len(explained_variance_ratio))]))

            return {
                'pca_analysis': {
                    'pca_energy_top1': pca_energy_top1,
                    'pca_energy_top5': pca_energy_top5,
                    'n_components_90_variance': int(n_components_90),
                    'n_components_95_variance': int(n_components_95),
                    'n_components_99_variance': int(n_components_99),
                    'participation_ratio': participation_ratio,
                    'effective_rank': effective_rank,
                    'effective_rank_ratio': float(np.log(effective_rank) / np.log(D)) if D > 1 and effective_rank > 0 else 0.0,
                    'cumulative_variance': cumulative_variance.tolist(),
                    'eigenvalues': lambda_.tolist(),
                    'n_components_computed': int(n_computed),
                    'pca_ceiling_hit': pca_ceiling_hit,
                }
            }
        except Exception as e:
            return {'pca_analysis': {'error': str(e)}}

    def _compute_similarity_distribution(self, emb: np.ndarray, is_l2_normalized: bool) -> Dict[str, Any]:
        """
        Compute pairwise cosine similarity distribution between embeddings.

        Uses efficient random pair sampling for large datasets.
        """
        try:
            if is_l2_normalized:
                emb_normalized = emb
            else:
                emb_normalized = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-12)

            rng = np.random.RandomState(self.seed)

            n_samples = min(2000, len(emb))
            n_pairs = min(self.num_pairs_similarity, n_samples * (n_samples - 1) // 2)

            if n_pairs < n_samples * 2:
                indices = rng.choice(len(emb), n_samples, replace=False)
                sampled_emb = emb_normalized[indices]
                similarities = np.dot(sampled_emb, sampled_emb.T)
                upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
                similarities_array = upper_triangle
            else:
                idx_i = rng.choice(len(emb), n_pairs, replace=True)
                idx_j = rng.choice(len(emb), n_pairs, replace=True)
                valid_mask = idx_i != idx_j
                idx_i = idx_i[valid_mask]
                idx_j = idx_j[valid_mask]
                if len(idx_i) > n_pairs:
                    idx_i = idx_i[:n_pairs]
                    idx_j = idx_j[:n_pairs]
                if len(idx_i) == 0:
                    return {'similarity_distribution': {'error': 'Could not generate valid pairs'}}
                similarities_array = np.sum(emb_normalized[idx_i] * emb_normalized[idx_j], axis=1)

            mean_similarity = float(np.mean(similarities_array))
            std_similarity = float(np.std(similarities_array))

            collapse_warning = None
            if mean_similarity > 0.95:
                collapse_warning = "High similarity (>0.95) suggests embeddings may be collapsed."
            elif mean_similarity > 0.90:
                collapse_warning = "Moderately high similarity (>0.90) - monitor for potential collapse."

            return {
                'similarity_distribution': {
                    'pairwise_cosine_mean': mean_similarity,
                    'pairwise_cosine_std': std_similarity,
                    'n_pairs_analyzed': int(len(similarities_array)),
                    'collapse_warning': collapse_warning,
                }
            }
        except Exception as e:
            return {'similarity_distribution': {'error': str(e)}}

    def _compute_uniformity(self, emb: np.ndarray, is_l2_normalized: bool) -> Dict[str, Any]:
        """
        Compute uniformity metric (Wang & Isola, 2020).

        Measures how uniformly embeddings are distributed on the hypersphere.
        uniformity = log(E[exp(-t * ||u - v||^2)])

        More negative = more uniform (better spread). Near 0 = collapsed.
        """
        try:
            if is_l2_normalized:
                emb_normalized = emb
            else:
                emb_normalized = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-12)

            rng = np.random.RandomState(self.seed + 1)  # Different seed from similarity
            t = self.uniformity_t

            # Sample pairs for uniformity computation
            n_pairs = min(self.num_pairs_similarity, len(emb) * (len(emb) - 1) // 2)
            idx_i = rng.choice(len(emb), n_pairs, replace=True)
            idx_j = rng.choice(len(emb), n_pairs, replace=True)
            valid_mask = idx_i != idx_j
            idx_i = idx_i[valid_mask]
            idx_j = idx_j[valid_mask]

            if len(idx_i) == 0:
                return {'uniformity': {'error': 'Could not generate valid pairs'}}

            # ||u - v||^2 for L2-normalized vectors = 2 - 2 * cos(u, v)
            sq_dists = np.sum((emb_normalized[idx_i] - emb_normalized[idx_j]) ** 2, axis=1)

            # uniformity = log(E[exp(-t * ||u-v||^2)])
            # Use log-sum-exp trick for numerical stability
            exponents = -t * sq_dists
            max_exp = np.max(exponents)
            uniformity = float(max_exp + np.log(np.mean(np.exp(exponents - max_exp))))

            # Normalize to 0-1 scale for quality score (more negative = better)
            # Typical range: -4 (excellent) to 0 (collapsed)
            # Map: -4 → 1.0, 0 → 0.0
            uniformity_normalized = float(np.clip(-uniformity / 4.0, 0.0, 1.0))

            return {
                'uniformity': {
                    'uniformity': uniformity,
                    'uniformity_normalized': uniformity_normalized,
                    'uniformity_t': t,
                    'n_pairs': int(len(idx_i)),
                }
            }
        except Exception as e:
            return {'uniformity': {'error': str(e)}}

    def _create_quality_dashboard(self, results: Dict[str, Any]) -> Optional[str]:
        """
        Create a 4-panel quality dashboard visualization.

        Panels:
        1. Similarity Distribution (with isotropic reference)
        2. Eigenvalue Spectrum (log-scale decay with effective rank annotation)
        3. PCA Cumulative Variance (with isotropic reference curve)
        4. Quality Metrics Table (with isotropic reference column)
        """
        try:
            if self.output_dir:
                save_path = Path(self.output_dir) / "embedding_stats_dashboard.png"
            else:
                save_path = Path("evaluation_results/embedding_stats/embedding_stats_dashboard.png")
                save_path.parent.mkdir(parents=True, exist_ok=True)

            fig, axes = plt.subplots(2, 2, figsize=self.dashboard_figsize)
            fig.suptitle('Embedding Quality Dashboard', fontsize=16, fontweight='bold')

            D = results.get('embedding_dim', 768)
            is_l2 = results.get('is_l2_normalized', False)

            # Panel 1 (top-left): Similarity Distribution
            ax1 = axes[0, 0]
            if 'similarity_distribution' in results and 'error' not in results['similarity_distribution']:
                sim_dist = results['similarity_distribution']
                mean_sim = sim_dist.get('pairwise_cosine_mean', 0.0)
                std_sim = sim_dist.get('pairwise_cosine_std', 0.0)

                ax1.axvspan(0.0, 0.7, alpha=0.15, color='green', label='Healthy (<0.7)')
                ax1.axvspan(0.7, 0.9, alpha=0.15, color='yellow', label='Warning (0.7-0.9)')
                ax1.axvspan(0.9, 1.0, alpha=0.15, color='red', label='Collapsed (>0.9)')

                if std_sim > 0:
                    x = np.linspace(0, 1, 200)
                    pdf = (1.0 / (std_sim * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_sim) / std_sim) ** 2)
                    ax1.plot(x, pdf, 'b-', linewidth=2, label='Approx. distribution', alpha=0.7)

                ax1.axvline(mean_sim, color='black', linestyle='--', linewidth=2, label=f'Mean: {mean_sim:.3f}')

                # Isotropic reference: for high-D L2-normalized vectors, expected mean similarity ~0
                if is_l2:
                    ax1.axvline(0.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Isotropic ref (~0)')

                ax1.set_xlim(0, 1)
                ax1.set_xlabel('Cosine Similarity')
                ax1.set_ylabel('Probability Density')
                ax1.set_title('Pairwise Similarity Distribution')
                ax1.legend(loc='upper left', fontsize=7)
                ax1.text(0.98, 0.95, f'$\\mu$={mean_sim:.3f}\n$\\sigma$={std_sim:.3f}',
                         transform=ax1.transAxes, ha='right', va='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax1.text(0.5, 0.5, 'N/A\n(Similarity distribution not computed)',
                         ha='center', va='center', transform=ax1.transAxes, fontsize=12)
                ax1.set_title('Pairwise Similarity Distribution')
            ax1.grid(True, alpha=0.3)

            # Panel 2 (top-right): Eigenvalue Spectrum
            ax2 = axes[0, 1]
            if 'pca_analysis' in results and 'error' not in results['pca_analysis']:
                pca_data = results['pca_analysis']
                eigenvalues = pca_data.get('eigenvalues', None)
                eff_rank = pca_data.get('effective_rank', None)
                part_ratio = pca_data.get('participation_ratio', None)
                n_computed = pca_data.get('n_components_computed', 0)

                if eigenvalues is not None:
                    eigenvalues = np.array(eigenvalues)
                    components = np.arange(1, len(eigenvalues) + 1)
                    ax2.semilogy(components, eigenvalues, 'b-', linewidth=2, label='Eigenvalues')

                    # Isotropic reference: flat eigenvalue line
                    mean_eigenvalue = np.mean(eigenvalues)
                    ax2.axhline(mean_eigenvalue, color='gray', linestyle=':', linewidth=1.5,
                                alpha=0.7, label=f'Mean: {mean_eigenvalue:.2e}')

                    # Annotate effective rank and participation ratio
                    annotations = []
                    if eff_rank is not None:
                        annotations.append(f'Eff. Rank: {eff_rank:.1f}')
                        # Draw vertical line at effective rank position
                        if eff_rank < n_computed:
                            ax2.axvline(eff_rank, color='orange', linestyle='--', alpha=0.6, linewidth=1)
                    if part_ratio is not None:
                        annotations.append(f'Part. Ratio: {part_ratio:.1f}')
                    annotations.append(f'Dim: {D}')

                    ax2.text(0.98, 0.95, '\n'.join(annotations),
                             transform=ax2.transAxes, ha='right', va='top', fontsize=8,
                             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

                ax2.set_xlabel('Component Index')
                ax2.set_ylabel('Eigenvalue (log scale)')
                ax2.set_title('Eigenvalue Spectrum')
                ax2.legend(loc='upper right', fontsize=8)
            else:
                ax2.text(0.5, 0.5, 'N/A\n(PCA analysis not computed)',
                         ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Eigenvalue Spectrum')
            ax2.grid(True, alpha=0.3)

            # Panel 3 (bottom-left): PCA Cumulative Variance
            ax3 = axes[1, 0]
            if 'pca_analysis' in results and 'error' not in results['pca_analysis']:
                pca_data = results['pca_analysis']
                n_90 = pca_data.get('n_components_90_variance', 0)
                n_95 = pca_data.get('n_components_95_variance', 0)
                cumulative_variance_list = pca_data.get('cumulative_variance', None)
                n_computed = pca_data.get('n_components_computed', 0)
                ceiling_hit = pca_data.get('pca_ceiling_hit', False)

                if cumulative_variance_list is not None:
                    components = np.arange(1, n_computed + 1)
                    cumulative = np.array(cumulative_variance_list)
                    ax3.plot(components, cumulative, 'b-', linewidth=2, label='Model')

                    # Isotropic reference curve: each component explains 1/D of variance
                    iso_cumvar = np.minimum(components / D, 1.0)
                    ax3.plot(components, iso_cumvar, 'gray', linestyle=':', linewidth=1.5,
                             alpha=0.7, label=f'Isotropic (D={D})')

                ax3.axhline(0.90, color='green', linestyle='--', alpha=0.5, label='90%')
                ax3.axhline(0.95, color='orange', linestyle='--', alpha=0.5, label='95%')

                if n_90 > 0:
                    at_ceiling = n_90 >= n_computed
                    label_90 = f'>=  {n_90} PCs*' if at_ceiling else f'{n_90} PCs'
                    ax3.plot(n_90, 0.90, 'go', markersize=8)
                    ax3.annotate(label_90, xy=(n_90, 0.90), xytext=(n_90 + 2, 0.85), fontsize=8, ha='left')
                if n_95 > 0:
                    at_ceiling = n_95 >= n_computed
                    label_95 = f'>= {n_95} PCs*' if at_ceiling else f'{n_95} PCs'
                    ax3.plot(n_95, 0.95, 'o', color='orange', markersize=8)
                    ax3.annotate(label_95, xy=(n_95, 0.95), xytext=(n_95 + 2, 0.92), fontsize=8, ha='left')

                if ceiling_hit:
                    ax3.text(0.02, 0.05,
                             '* threshold not reached within computed components',
                             transform=ax3.transAxes, fontsize=7, style='italic', color='red', va='bottom')

                ax3.set_xlabel('Number of Components')
                ax3.set_ylabel('Cumulative Variance Explained')
                ax3.set_title('PCA Cumulative Variance')
                ax3.legend(loc='lower right', fontsize=7)
                ax3.set_ylim(0, 1.05)
            else:
                ax3.text(0.5, 0.5, 'N/A\n(PCA analysis not computed)',
                         ha='center', va='center', transform=ax3.transAxes, fontsize=12)
                ax3.set_title('PCA Cumulative Variance')
            ax3.grid(True, alpha=0.3)

            # Panel 4 (bottom-right): Quality Metrics Table
            # Shows model value alongside typical range for masked prediction
            # transformers (BERT, DreaMS, MSBERT-class models from literature).
            ax4 = axes[1, 1]
            ax4.axis('off')

            metrics_data = []  # [Metric, Value, Typical Range]

            if 'basic_stats' in results:
                basic = results['basic_stats']

                centroid = basic.get('centroid_norm', None)
                if centroid is not None:
                    metrics_data.append(['Centroid Norm', f'{centroid:.3f}', '0.6 - 0.85'])

                iso_deficit = basic.get('isotropy_deficit', None)
                if iso_deficit is not None:
                    metrics_data.append(['Isotropy Deficit', f'{iso_deficit:.3f}', '0.4 - 0.8'])

                dead = basic.get('dead_dimensions', None)
                if dead is not None:
                    metrics_data.append(['Dead Dims', f'{dead}/{D}', '0'])

            if 'pca_analysis' in results and 'error' not in results['pca_analysis']:
                pca = results['pca_analysis']
                pca_top1 = pca.get('pca_energy_top1', None)
                if pca_top1 is not None:
                    metrics_data.append(['PCA Top-1', f'{pca_top1:.3f}', '0.10 - 0.30'])

                eff_rank = pca.get('effective_rank', None)
                if eff_rank is not None:
                    metrics_data.append(['Eff. Rank', f'{eff_rank:.1f}', '20 - 80'])

            if 'similarity_distribution' in results and 'error' not in results['similarity_distribution']:
                mean_sim = results['similarity_distribution'].get('pairwise_cosine_mean', None)
                if mean_sim is not None:
                    metrics_data.append(['Mean Similarity', f'{mean_sim:.3f}', '0.4 - 0.7'])

            if 'uniformity' in results and 'error' not in results['uniformity']:
                unif_val = results['uniformity'].get('uniformity', None)
                if unif_val is not None:
                    metrics_data.append(['Uniformity', f'{unif_val:.2f}', '-0.5 to -2.5'])

            if not metrics_data:
                ax4.text(0.5, 0.5, 'N/A\n(No metrics available)', ha='center', va='center', fontsize=12)
            else:
                table = ax4.table(cellText=metrics_data,
                                  colLabels=['Metric', 'Value', 'Typical Range'],
                                  cellLoc='center', loc='center',
                                  colWidths=[0.38, 0.25, 0.32])
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.8)

                for i in range(3):
                    table[(0, i)].set_facecolor('#4CAF50')
                    table[(0, i)].set_text_props(weight='bold', color='white')

                for i in range(1, len(metrics_data) + 1):
                    for j in range(3):
                        cell = table[(i, j)]
                        if i % 2 == 0:
                            cell.set_facecolor('#f0f0f0')

            l2_tag = ' (L2-normalized)' if is_l2 else ''
            ax4.set_title(f'Embedding Metrics{l2_tag}', fontweight='bold', pad=20)

            # Footnote explaining the typical range source
            fig.text(0.52, 0.01,
                     'Typical Range = masked prediction transformers (BERT, DreaMS, MSBERT-class models)',
                     ha='center', fontsize=8, style='italic', color='gray')

            plt.tight_layout()
            plt.savefig(save_path, dpi=self.dashboard_dpi, bbox_inches='tight')
            plt.close(fig)

            return str(save_path)

        except Exception as e:
            warnings.warn(f"Failed to create dashboard: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None

    def validate_embedding_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate embedding quality with L2-normalization-aware thresholds.

        Returns diagnostic sub-scores (each 0-1, higher = better) and
        warnings/errors for clear regressions. No composite "quality score"
        is produced — each sub-score is independently meaningful and should
        be interpreted in context of the training objective.
        """
        warnings_list = []
        errors = []
        D = results.get('embedding_dim', 768)
        is_l2 = results.get('is_l2_normalized', False)

        # Diagnostic sub-scores (all 0-1, higher = better)
        # These are NOT combined into a single score — interpret individually
        diagnostic_scores = {}

        if 'basic_stats' in results:
            basic = results['basic_stats']

            # For un-normalized embeddings, check norm consistency
            if not is_l2:
                norm_cv = basic.get('norm_coefficient_of_variation', 0.0)
                if norm_cv > 0.5:
                    warnings_list.append(f"High norm coefficient of variation ({norm_cv:.4f})")

            # Anisotropy check (L2-aware thresholds)
            aniso_ratio = basic.get('anisotropy_ratio', 0.0)
            if aniso_ratio > 50:
                warnings_list.append(f"Very high anisotropy ratio ({aniso_ratio:.1f}x isotropic)")
            elif aniso_ratio > 30:
                warnings_list.append(f"High anisotropy ratio ({aniso_ratio:.1f}x isotropic)")

            # Isotropy deficit (clamped to 0-1)
            iso_deficit = basic.get('isotropy_deficit', 0.0)
            diagnostic_scores['isotropy_deficit'] = float(np.clip(iso_deficit, 0.0, 1.0))

            # Dead dimensions check
            dead_frac = basic.get('dead_dimensions_fraction', 0.0)
            if dead_frac > 0.1:
                warnings_list.append(f"{basic.get('dead_dimensions', 0)} dead dimensions ({dead_frac:.1%} of {D})")

            # Per-dimension variance utilization
            per_dim_std_min = basic.get('per_dim_std_min', 0.0)
            if per_dim_std_min < 1e-6:
                warnings_list.append(f"Near-zero per-dimension std ({per_dim_std_min:.2e}) - dimension collapse")

        # PCA dimensionality
        if 'pca_analysis' in results and 'error' not in results['pca_analysis']:
            pca = results['pca_analysis']

            pca_energy_top1 = pca.get('pca_energy_top1', 0.0)
            if pca_energy_top1 > 0.5:
                warnings_list.append(f"High PCA energy in top-1 ({pca_energy_top1:.3f})")

            # Log-scaled effective rank ratio
            eff_rank_ratio = pca.get('effective_rank_ratio', 0.0)
            diagnostic_scores['effective_rank_ratio'] = float(np.clip(eff_rank_ratio, 0.0, 1.0))

            eff_rank = pca.get('effective_rank', 0.0)
            if eff_rank < 5:
                warnings_list.append(f"Very low effective rank ({eff_rank:.1f})")

        # Similarity distribution
        if 'similarity_distribution' in results and 'error' not in results['similarity_distribution']:
            mean_sim = results['similarity_distribution'].get('pairwise_cosine_mean', 0.0)
            if mean_sim > 0.98:
                errors.append(f"Very high similarity ({mean_sim:.4f}) - embeddings are collapsed")
            elif mean_sim > 0.95:
                warnings_list.append(f"High similarity ({mean_sim:.4f}) - monitor for collapse")

            # Diversity: lower similarity = better
            diagnostic_scores['diversity'] = float(1.0 - min(1.0, max(0.0, mean_sim)))

        # Uniformity
        if 'uniformity' in results and 'error' not in results['uniformity']:
            unif_norm = results['uniformity'].get('uniformity_normalized', 0.0)
            diagnostic_scores['uniformity'] = float(unif_norm)

        return {
            'validation_passed': len(errors) == 0,
            'warnings': warnings_list,
            'errors': errors,
            'diagnostic_scores': diagnostic_scores,
        }

    def get_loggable_metrics(self, task_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract key metrics for logging to TensorBoard/Neptune.

        Returns only the 6 essential metrics for ablation comparison and
        embedding space monitoring. Each captures a distinct aspect:

        - uniformity: distribution quality on hypersphere (more negative = better)
        - effective_rank: entropy-based dimensionality (higher = better utilization)
        - mean_similarity: pairwise cosine diversity (lower = more diverse)
        - anisotropy_ratio: angular concentration vs isotropic (lower = better)
        - pca_energy_top1: variance in 1st PC (spike = collapse early warning)
        - centroid_norm: norm of mean vector (increase = drift/collapse)
        """
        metrics = {}

        if 'basic_stats' in task_results:
            basic = task_results['basic_stats']
            metrics['centroid_norm'] = basic.get('centroid_norm', 0.0)
            metrics['anisotropy_ratio'] = basic.get('anisotropy_ratio', 0.0)

        if 'pca_analysis' in task_results and 'error' not in task_results['pca_analysis']:
            pca = task_results['pca_analysis']
            metrics['pca_energy_top1'] = pca.get('pca_energy_top1', 0.0)
            metrics['effective_rank'] = pca.get('effective_rank', 0.0)

        if 'similarity_distribution' in task_results and 'error' not in task_results['similarity_distribution']:
            metrics['mean_similarity'] = task_results['similarity_distribution'].get('pairwise_cosine_mean', 0.0)

        if 'uniformity' in task_results and 'error' not in task_results['uniformity']:
            metrics['uniformity'] = task_results['uniformity'].get('uniformity', 0.0)

        return metrics
