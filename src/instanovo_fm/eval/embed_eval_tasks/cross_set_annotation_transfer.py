"""Cross-set annotation transfer for unlabeled-query rescue workflows.

Queries (for example ACFM) are unlabeled; library spectra (for example LCFM-valid)
carry peptide labels. The task ranks library candidates per query, computes
embedding margins, and optionally scores spectral evidence blocks A/B/C in parallel.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from instanovo_fm.eval.embed_eval_tasks import BaseTask
from instanovo_fm.eval.spectrum_metrics.mcp_scoring import MCP_AVAILABLE
from instanovo_fm.eval.spectrum_metrics.worker import (
    LibrarySelfWorkItem,
    QueryRankWorkItem,
    SpectrumRecord,
    score_library_self,
    score_query_rank_worker,
)


class CrossSetAnnotationTransferTask(BaseTask):
    """Retrieve peptide candidates from a labeled library for unlabeled queries."""

    name = "Cross Set Annotation Transfer"
    description = "Cross-set retrieval with unlabeled queries and labeled library"
    requires_metadata = True
    requires_faiss = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.query_filter: Dict[str, Any] = kwargs.get("query_filter", {"search_tier": ["acfm"]})
        self.library_filter: Dict[str, Any] = kwargs.get(
            "library_filter",
            {"is_selected_anchor": ["1"]},
        )
        self.peptide_key: str = kwargs.get("peptide_key", "peptides")
        self.unmodified_key: str = kwargs.get("unmodified_key", "unmodified_peptide")
        self.id_key: str = kwargs.get("id_key", "usi")
        self.overlap_key: Optional[str] = kwargs.get("overlap_key", "usi")
        self.exclude_overlap: bool = kwargs.get("exclude_overlap", True)
        # Always independently re-verify no query shares overlap_key with any library row,
        # even when exclude_overlap=False because the diff was supposedly already applied
        # upstream (dataset builder). A query that IS a library spectrum would trivially
        # retrieve itself with ~1.0 similarity and silently invalidate every downstream
        # margin/hub/evidence-metric interpretation. Fail hard by default rather than
        # quietly dropping rows, so a broken upstream diff surfaces immediately.
        self.fail_on_overlap: bool = kwargs.get("fail_on_overlap", True)
        self.k_values: List[int] = kwargs.get("k_values", list(range(1, 21)))
        self.batch_size: int = kwargs.get("batch_size", 500)
        self.max_queries: Optional[int] = kwargs.get("max_queries", None)
        # Pre-retrieval query confidence gate: keep only the most confident query spectra
        # BEFORE computing embedding pair similarities, using the model's own
        # identification-free spectrum_confidence (conf_group * conf_offset). Requires
        # evaluation.compute_spectrum_confidence=true so meta["spectrum_confidence"] exists.
        # Precedence when several are set: top_n > top_frac > min. All None => no gate
        # (retrieve for every query, current default behaviour).
        self.query_confidence_key: str = kwargs.get("query_confidence_key", "spectrum_confidence")
        self.query_confidence_min: Optional[float] = kwargs.get("query_confidence_min", None)
        self.query_confidence_top_frac: Optional[float] = kwargs.get("query_confidence_top_frac", None)
        self.query_confidence_top_n: Optional[int] = kwargs.get("query_confidence_top_n", None)
        self.sample_seed: int = kwargs.get("sample_seed", 42)
        self.output_dir: Optional[str] = kwargs.get("output_dir", None)
        self.save_candidates_csv: bool = kwargs.get("save_candidates_csv", True)
        self.save_matrix_artifact: bool = kwargs.get("save_matrix_artifact", False)
        # Persist the raw query/library embedding vectors so the hero-query UMAP (and any
        # other embedding diagnostic) can be reproduced/iterated offline without re-running
        # the model. Off by default (embeddings are large); enable for diagnostic runs.
        self.save_embeddings_artifact: bool = kwargs.get("save_embeddings_artifact", False)
        self.topk_only: bool = kwargs.get("topk_only", True)
        # Same default as spectral_annotation_transfer / rescue_reformulated.
        self.save_plots: bool = kwargs.get("save_plots", True)
        self.plot_dpi: int = kwargs.get("plot_dpi", 160)
        self.plot_max_curve_queries: int = kwargs.get("plot_max_curve_queries", 5_000)
        self.plot_heatmap_max_queries: int = kwargs.get("plot_heatmap_max_queries", 512)
        self.plot_top_peptides: int = kwargs.get("plot_top_peptides", 20)
        # UMAP of the full library plus a handful of individually-labeled hero queries.
        # Directly visualises whether the "hub" peptides seen in cross_set_top_peptides.png
        # form a real embedding cluster near their own anchor, or are a symptom of
        # embedding anisotropy (see margin_distribution / topk_heatmap interpretation).
        # Deliberately few queries (not a dense cloud) so each one is individually
        # traceable back to its own rank-1 assigned anchor.
        self.plot_umap: bool = kwargs.get("plot_umap", True)
        self.umap_num_queries: int = kwargs.get("umap_num_queries", 5)
        self.umap_n_neighbors: int = kwargs.get("umap_n_neighbors", 30)
        self.umap_min_dist: float = kwargs.get("umap_min_dist", 0.1)

        self.compute_evidence_metrics: bool = kwargs.get("compute_evidence_metrics", True)
        self.plot_evidence_metrics: bool = kwargs.get("plot_evidence_metrics", True)
        self.score_blocks: List[str] = kwargs.get("score_blocks", ["A", "B", "C"])
        self.num_workers: int = kwargs.get("num_workers", 0)
        self.tolerance_da: float = kwargs.get("tolerance_da", 0.05)
        self.ion_types: str = kwargs.get("ion_types", "by")
        self.max_ion_charge: int = kwargs.get("max_ion_charge", 2)
        self.cache_library_self_scores: bool = kwargs.get("cache_library_self_scores", True)
        self.mz_key: str = kwargs.get("mz_key", "mz_array")
        self.intensity_key: str = kwargs.get("intensity_key", "intensity_array")
        self.precursor_mz_key: str = kwargs.get("precursor_mz_key", "precursor_mz")
        self.precursor_charge_key: str = kwargs.get("precursor_charge_key", "precursor_charge")

    def run(
        self,
        emb: np.ndarray,
        meta: Dict[str, np.ndarray],
        faiss_index: Any = None,
    ) -> Dict[str, Any]:
        """Rank library candidates for each unlabeled query spectrum."""
        self.validate_inputs(emb, meta, faiss_index=None)

        if self.peptide_key not in meta:
            return {"task_name": self.name, "error": f"Missing metadata key {self.peptide_key!r}"}
        if self.id_key not in meta:
            return {"task_name": self.name, "error": f"Missing metadata key {self.id_key!r}"}

        query_mask = self._build_mask(meta, self.query_filter)
        library_mask = self._build_mask(meta, self.library_filter)

        query_indices = np.where(query_mask)[0].astype(np.int64)
        library_indices = np.where(library_mask)[0].astype(np.int64)

        # Kostas / rescue protocol: drop QUERY spectra already present in the library
        # reference set. Never drop library rows here (that was the wrong direction).
        if self.exclude_overlap and self.overlap_key and self.overlap_key in meta:
            lib_overlap = set(
                np.asarray(meta[self.overlap_key], dtype=object)[library_indices].tolist()
            )
            query_values = np.asarray(meta[self.overlap_key], dtype=object)[query_indices]
            keep_queries = np.array([value not in lib_overlap for value in query_values], dtype=bool)
            query_indices = query_indices[keep_queries]

        # Independent re-verification (see fail_on_overlap docstring above): this runs
        # regardless of exclude_overlap, so it also catches a broken upstream dataset-builder
        # diff when exclude_overlap=False.
        overlap_check: Dict[str, Any] = {
            "checked": False,
            "overlap_key": self.overlap_key,
            "n_overlapping_queries": 0,
        }
        if self.overlap_key and self.overlap_key in meta:
            lib_overlap_values = set(
                np.asarray(meta[self.overlap_key], dtype=object)[library_indices].tolist()
            )
            query_overlap_values = np.asarray(meta[self.overlap_key], dtype=object)[query_indices]
            overlapping_mask = np.array(
                [value in lib_overlap_values for value in query_overlap_values], dtype=bool
            )
            n_overlapping = int(overlapping_mask.sum())
            overlap_check = {
                "checked": True,
                "overlap_key": self.overlap_key,
                "n_overlapping_queries": n_overlapping,
                "n_queries_before_overlap_check": int(len(query_indices)),
            }
            if n_overlapping > 0:
                if self.fail_on_overlap:
                    return {
                        "task_name": self.name,
                        "error": (
                            f"Query/library overlap detected: {n_overlapping} query row(s) share "
                            f"{self.overlap_key!r} with a library row. A query that is also in the "
                            "library would trivially retrieve itself (~1.0 similarity) and invalidate "
                            "every downstream margin/hub/evidence metric. Refusing to run -- fix the "
                            "upstream diff (dataset builder overlap_key/exclude), or set "
                            "fail_on_overlap=false to drop the offending queries instead."
                        ),
                        "overlap_check": overlap_check,
                    }
                query_indices = query_indices[~overlapping_mask]

        if len(query_indices) == 0:
            return {"task_name": self.name, "error": "No query rows selected by query_filter"}
        if len(library_indices) == 0:
            return {
                "task_name": self.name,
                "error": "No library rows selected by library_filter after overlap exclusion",
            }

        # Pre-retrieval confidence gate: restrict retrieval (and everything downstream) to
        # the most confident query spectra, using the model's own spectrum_confidence.
        # Applied BEFORE e_q/top-k so it changes which pair similarities are ever computed.
        confidence_gate: Dict[str, Any] = {"applied": False}
        gate_requested = (
            self.query_confidence_min is not None
            or self.query_confidence_top_frac is not None
            or self.query_confidence_top_n is not None
        )
        if gate_requested:
            if self.query_confidence_key not in meta:
                return {
                    "task_name": self.name,
                    "error": (
                        f"query confidence gate requested but meta[{self.query_confidence_key!r}] is "
                        "missing. Set evaluation.compute_spectrum_confidence=true so the model reports "
                        "per-spectrum confidence."
                    ),
                }
            conf_all = np.asarray(meta[self.query_confidence_key], dtype=np.float64)
            query_conf = conf_all[query_indices]
            n_before = int(len(query_indices))
            # Threshold gate.
            if self.query_confidence_min is not None:
                keep = query_conf >= float(self.query_confidence_min)
                query_indices = query_indices[keep]
                query_conf = query_conf[keep]
            # Top-fraction / top-N gate (top_n wins if both given). Sort by confidence desc.
            n_keep: Optional[int] = None
            if self.query_confidence_top_n is not None:
                n_keep = int(self.query_confidence_top_n)
            elif self.query_confidence_top_frac is not None:
                n_keep = int(np.ceil(float(self.query_confidence_top_frac) * len(query_indices)))
            if n_keep is not None and len(query_indices) > n_keep:
                order = np.argsort(query_conf)[::-1][:n_keep]
                query_indices = np.sort(query_indices[order])
            confidence_gate = {
                "applied": True,
                "confidence_key": self.query_confidence_key,
                "query_confidence_min": self.query_confidence_min,
                "query_confidence_top_frac": self.query_confidence_top_frac,
                "query_confidence_top_n": self.query_confidence_top_n,
                "n_queries_before_gate": n_before,
                "n_queries_after_gate": int(len(query_indices)),
            }
            if len(query_indices) == 0:
                return {
                    "task_name": self.name,
                    "error": "Query confidence gate removed all queries; relax the threshold/top-n.",
                    "confidence_gate": confidence_gate,
                }

        if self.max_queries is not None and len(query_indices) > self.max_queries:
            rng = np.random.RandomState(self.sample_seed)
            query_indices = np.sort(rng.choice(query_indices, self.max_queries, replace=False))

        e_q = self._l2_normalise(emb[query_indices].astype(np.float32).copy())
        e_lib = self._l2_normalise(emb[library_indices].astype(np.float32).copy())

        max_k = max(self.k_values)
        if self.topk_only:
            topk_local, topk_scores = self._topk_indices_batched(e_q, e_lib, max_k, self.batch_size)
            sim_matrix = None
        else:
            sim_matrix = self._compute_similarity_batched(e_q, e_lib, self.batch_size)
            topk_local = np.argsort(sim_matrix, axis=1)[:, ::-1][:, :max_k]
            topk_scores = np.take_along_axis(sim_matrix, topk_local, axis=1)

        candidates, margins, top1_scores = self._build_candidate_rows(
            topk_local,
            topk_scores,
            meta,
            query_indices,
            library_indices,
        )

        evidence_metrics: Dict[Tuple[int, int], Dict[str, Any]] = {}
        if self.compute_evidence_metrics:
            if "C" in self.score_blocks and not self._has_spectrum_arrays(meta):
                return {
                    "task_name": self.name,
                    "error": (
                        f"Evidence metrics require spectrum arrays ({self.mz_key!r}, {self.intensity_key!r}) "
                        "in metadata. Add them to dataset.metadata_columns."
                    ),
                }
            if any(block in self.score_blocks for block in ("B", "C")) and not MCP_AVAILABLE:
                return {
                    "task_name": self.name,
                    "error": "Blocks B/C require proteomics-mcp. Install with: uv sync --extra proteomics-metrics",
                }
            evidence_metrics = self._compute_evidence_metrics_parallel(
                meta=meta,
                query_indices=query_indices,
                library_indices=library_indices,
                topk_local=topk_local,
            )
            self._attach_evidence_to_candidates(candidates, evidence_metrics)

        results: Dict[str, Any] = {
            "task_name": self.name,
            "num_queries": int(len(query_indices)),
            "num_library": int(len(library_indices)),
            "max_k": int(max_k),
            "mean_top1_score": float(np.mean(top1_scores)) if len(top1_scores) else 0.0,
            "median_top1_score": float(np.median(top1_scores)) if len(top1_scores) else 0.0,
            "mean_top1_margin": float(np.mean(margins)) if len(margins) else 0.0,
            "median_top1_margin": float(np.median(margins)) if len(margins) else 0.0,
            "query_filter": self.query_filter,
            "library_filter": self.library_filter,
            "excluded_overlap": bool(self.exclude_overlap and self.overlap_key),
            "compute_evidence_metrics": bool(self.compute_evidence_metrics),
            "score_blocks": self.score_blocks,
            "num_workers": int(self._resolve_num_workers()),
            "mcp_available": bool(MCP_AVAILABLE),
            "save_plots": bool(self.save_plots),
            "overlap_check": overlap_check,
            "confidence_gate": confidence_gate,
        }

        if self.output_dir:
            out_dir = Path(self.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            if self.save_candidates_csv:
                path = out_dir / "cross_set_topk_candidates.csv"
                self._write_candidates_csv(path, candidates)
                results["candidates_csv"] = str(path)
            if sim_matrix is not None and self.save_matrix_artifact:
                matrix_path = out_dir / "cross_set_similarity_matrix.npz"
                np.savez_compressed(
                    str(matrix_path),
                    similarity_matrix=sim_matrix.astype(np.float16),
                    query_indices=query_indices,
                    library_indices=library_indices,
                )
                results["similarity_matrix_npz"] = str(matrix_path)
            if self.save_embeddings_artifact:
                emb_path = out_dir / "cross_set_embeddings.npz"
                peptides_meta = np.asarray(meta[self.peptide_key], dtype=object)
                unmodified_meta = np.asarray(meta.get(self.unmodified_key, peptides_meta), dtype=object)
                library_peptides = np.asarray(
                    [str(peptides_meta[idx] or unmodified_meta[idx] or "") for idx in library_indices.tolist()],
                    dtype=object,
                )
                top1_by_query_index: Dict[int, str] = {
                    int(row["query_index"]): str(row.get("library_peptide") or "")
                    for row in candidates
                    if int(row.get("rank", 0)) == 1
                }
                query_top1_peptides = np.asarray(
                    [top1_by_query_index.get(int(qi), "") for qi in query_indices.tolist()],
                    dtype=object,
                )
                np.savez_compressed(
                    str(emb_path),
                    query_embeddings=e_q.astype(np.float16),
                    library_embeddings=e_lib.astype(np.float16),
                    query_indices=query_indices,
                    library_indices=library_indices,
                    library_peptides=library_peptides,
                    query_top1_peptides=query_top1_peptides,
                )
                results["embeddings_npz"] = str(emb_path)
            if self.save_plots:
                plot_paths = self._save_plots(
                    out_dir=out_dir,
                    topk_scores=topk_scores,
                    margins=margins,
                    top1_scores=top1_scores,
                    candidates=candidates,
                    sim_matrix=sim_matrix,
                    e_q=e_q,
                    e_lib=e_lib,
                    query_indices=query_indices,
                    library_indices=library_indices,
                    meta=meta,
                )
                if self.compute_evidence_metrics and self.plot_evidence_metrics:
                    evidence_plot_paths = self._plot_evidence_metrics(candidates, out_dir)
                    plot_paths.update(evidence_plot_paths)
                if plot_paths:
                    results["plot_paths"] = plot_paths
            summary_path = out_dir / "cross_set_task_summary.json"
            summary_path.write_text(json.dumps(results, indent=2, default=str))
            results["task_summary_json"] = str(summary_path)

        return results

    def get_loggable_metrics(self, task_results: Dict[str, Any]) -> Dict[str, float]:
        """Return scalar metrics suitable for MLflow logging."""
        if "error" in task_results:
            return {}
        keys = ("num_queries", "num_library", "mean_top1_score", "mean_top1_margin", "median_top1_margin")
        return {key: float(task_results[key]) for key in keys if key in task_results}

    def _resolve_num_workers(self) -> int:
        if self.num_workers and self.num_workers > 0:
            return int(self.num_workers)
        import os

        return max(1, (os.cpu_count() or 2) - 1)

    def _has_spectrum_arrays(self, meta: Dict[str, np.ndarray]) -> bool:
        return self.mz_key in meta and self.intensity_key in meta

    @staticmethod
    def _l2_normalise(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings /= np.where(norms == 0, 1.0, norms)
        return embeddings

    @staticmethod
    def _compute_similarity_batched(e_q: np.ndarray, e_lib: np.ndarray, batch_size: int) -> np.ndarray:
        m = e_q.shape[0]
        sim_matrix = np.empty((m, e_lib.shape[0]), dtype=np.float32)
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            sim_matrix[start:end] = e_q[start:end] @ e_lib.T
        return sim_matrix

    @staticmethod
    def _topk_indices_batched(
        e_q: np.ndarray,
        e_lib: np.ndarray,
        max_k: int,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        m = e_q.shape[0]
        n = e_lib.shape[0]
        k = min(max_k, n)
        topk_local = np.empty((m, k), dtype=np.int64)
        topk_scores = np.empty((m, k), dtype=np.float32)
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            scores = e_q[start:end] @ e_lib.T
            if k == n:
                order = np.argsort(scores, axis=1)[:, ::-1]
            else:
                part = np.argpartition(scores, n - k, axis=1)[:, -k:]
                row_order = np.argsort(np.take_along_axis(scores, part, axis=1), axis=1)[:, ::-1]
                order = np.take_along_axis(part, row_order, axis=1)
            topk_local[start:end] = order
            topk_scores[start:end] = np.take_along_axis(scores, order, axis=1)
        return topk_local, topk_scores

    @staticmethod
    def _normalise_filter_values(values: Any) -> set[str]:
        # OmegaConf ListConfig is not a list/tuple/set; convert via list() first.
        if isinstance(values, (list, tuple, set)):
            return {str(v) for v in values}
        if hasattr(values, "__iter__") and not isinstance(values, (str, bytes)):
            return {str(v) for v in list(values)}
        return {str(values)}

    def _build_mask(self, meta: Dict[str, np.ndarray], filters: Dict[str, Any]) -> np.ndarray:
        first_key = next(iter(meta))
        length = len(meta[first_key])
        mask = np.ones(length, dtype=bool)
        for key, wanted in filters.items():
            if key not in meta:
                raise ValueError(f"Filter key {key!r} not present in metadata. Available keys: {list(meta.keys())}")
            allowed = self._normalise_filter_values(wanted)
            values = np.asarray(meta[key], dtype=object).astype(str)
            mask &= np.isin(values, list(allowed))
        return mask

    def _spectrum_record(self, meta: Dict[str, np.ndarray], index: int) -> SpectrumRecord:
        peptide = str(np.asarray(meta[self.peptide_key], dtype=object)[index])
        unmodified = str(np.asarray(meta.get(self.unmodified_key, meta[self.peptide_key]), dtype=object)[index])
        mz = np.asarray(meta[self.mz_key][index], dtype=float).flatten()
        intensity = np.asarray(meta[self.intensity_key][index], dtype=float).flatten()
        precursor_mz = None
        precursor_charge = None
        if self.precursor_mz_key in meta:
            value = meta[self.precursor_mz_key][index]
            precursor_mz = float(value) if value is not None and value == value else None
        if self.precursor_charge_key in meta:
            value = meta[self.precursor_charge_key][index]
            precursor_charge = int(value) if value is not None else None
        return SpectrumRecord(
            mz=tuple(float(v) for v in mz.tolist()),
            intensity=tuple(float(v) for v in intensity.tolist()),
            precursor_mz=precursor_mz,
            precursor_charge=precursor_charge,
            peptide=peptide,
            unmodified_peptide=unmodified,
        )

    def _build_candidate_rows(
        self,
        topk_local: np.ndarray,
        topk_scores: np.ndarray,
        meta: Dict[str, np.ndarray],
        query_indices: np.ndarray,
        library_indices: np.ndarray,
    ) -> tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        peptides = np.asarray(meta[self.peptide_key], dtype=object)
        unmodified = np.asarray(meta.get(self.unmodified_key, peptides), dtype=object)
        ids = np.asarray(meta[self.id_key], dtype=object)
        pair_files = np.asarray(meta["pair_file"], dtype=object) if "pair_file" in meta else None
        # Model's own masked-reconstruction confidence (conf_group * conf_offset, averaged
        # over valid peaks), present when evaluation.compute_spectrum_confidence=true.
        # Identification-free — meaningful for unlabeled ACFM queries, unlike the
        # library-side hyperscore/probability/expectation columns.
        spectrum_confidence = np.asarray(meta["spectrum_confidence"], dtype=np.float64) if "spectrum_confidence" in meta else None

        rows: List[Dict[str, Any]] = []
        margins: List[float] = []
        top1_scores: List[float] = []

        def _clean_peptide(value: Any, fallback: Any = "") -> str:
            text = "" if value is None else str(value)
            if text in {"", "None", "nan", "NaN"}:
                fallback_text = "" if fallback is None else str(fallback)
                return "" if fallback_text in {"", "None", "nan", "NaN"} else fallback_text
            return text

        for q_row, q_index in enumerate(query_indices.tolist()):
            order = topk_local[q_row]
            scores = topk_scores[q_row]
            if len(scores) > 1:
                margins.append(float(scores[0] - scores[1]))
            elif len(scores) == 1:
                margins.append(float(scores[0]))
            if len(scores) > 0:
                top1_scores.append(float(scores[0]))

            for rank in range(order.shape[0]):
                lib_col = int(order[rank])
                lib_index = int(library_indices[lib_col])
                pep = _clean_peptide(peptides[lib_index], unmodified[lib_index])
                unmod = _clean_peptide(unmodified[lib_index], pep)
                row = {
                    "query_index": int(q_index),
                    "query_id": str(ids[q_index]),
                    "rank": int(rank + 1),
                    "embedding_score": float(scores[rank]),
                    "embedding_margin_1_2": float(scores[0] - scores[1]) if len(scores) > 1 else None,
                    "library_index": lib_index,
                    "library_id": str(ids[lib_index]),
                    "library_peptide": pep,
                    "library_unmodified_peptide": unmod,
                    "metric_profile": "legacy_v1",
                }
                if pair_files is not None:
                    row["pair_file"] = str(pair_files[q_index])
                if spectrum_confidence is not None:
                    row["query_spectrum_confidence"] = float(spectrum_confidence[q_index])
                rows.append(row)

        return rows, np.asarray(margins, dtype=np.float64), np.asarray(top1_scores, dtype=np.float64)

    def _compute_evidence_metrics_parallel(
        self,
        *,
        meta: Dict[str, np.ndarray],
        query_indices: np.ndarray,
        library_indices: np.ndarray,
        topk_local: np.ndarray,
    ) -> Dict[Tuple[int, int], Dict[str, Any]]:
        library_records = {int(idx): self._spectrum_record(meta, int(idx)) for idx in library_indices.tolist()}
        query_records = {int(idx): self._spectrum_record(meta, int(idx)) for idx in query_indices.tolist()}

        library_self_cache: Dict[int, Dict[str, Any]] = {}
        blocks = tuple(block.upper() for block in self.score_blocks)
        workers = self._resolve_num_workers()

        if "C" in blocks and self.cache_library_self_scores:
            lib_items = [
                LibrarySelfWorkItem(
                    library_index=lib_index,
                    library=library_records[lib_index],
                    tolerance_da=self.tolerance_da,
                    ion_types=self.ion_types,
                    max_ion_charge=self.max_ion_charge,
                )
                for lib_index in library_records
            ]
            if workers <= 1 or len(lib_items) <= 1:
                for lib_item in lib_items:
                    lib_index, metrics = score_library_self(lib_item)
                    library_self_cache[lib_index] = metrics
            else:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    lib_futures = [executor.submit(score_library_self, lib_item) for lib_item in lib_items]
                    for future in as_completed(lib_futures):
                        lib_index, metrics = future.result()
                        library_self_cache[lib_index] = metrics

        work_items: list[QueryRankWorkItem] = []
        for q_row, q_index in enumerate(query_indices.tolist()):
            for rank in range(topk_local.shape[1]):
                lib_col = int(topk_local[q_row, rank])
                lib_index = int(library_indices[lib_col])
                work_items.append(
                    QueryRankWorkItem(
                        query_index=int(q_index),
                        rank=int(rank + 1),
                        library_index=lib_index,
                        query=query_records[int(q_index)],
                        library=library_records[lib_index],
                        score_blocks=tuple(block for block in blocks if block in ("A", "B")),
                        tolerance_da=self.tolerance_da,
                        ion_types=self.ion_types,
                        max_ion_charge=self.max_ion_charge,
                    )
                )

        evidence: Dict[Tuple[int, int], Dict[str, Any]] = {}
        if workers <= 1 or len(work_items) <= 1:
            for query_item in work_items:
                row = score_query_rank_worker((query_item, library_self_cache if "C" in blocks else None))
                evidence[(int(row["query_index"]), int(row["rank"]))] = row
            return evidence

        with ProcessPoolExecutor(max_workers=workers) as executor:
            query_futures = [
                executor.submit(score_query_rank_worker, (query_item, library_self_cache if "C" in blocks else None)) for query_item in work_items
            ]
            for future in as_completed(query_futures):
                row = future.result()
                evidence[(int(row["query_index"]), int(row["rank"]))] = row
        return evidence

    def _save_plots(
        self,
        *,
        out_dir: Path,
        topk_scores: np.ndarray,
        margins: np.ndarray,
        top1_scores: np.ndarray,
        candidates: List[Dict[str, Any]],
        sim_matrix: Optional[np.ndarray],
        e_q: Optional[np.ndarray] = None,
        e_lib: Optional[np.ndarray] = None,
        query_indices: Optional[np.ndarray] = None,
        library_indices: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, str]:
        """Save quick-look plots analogous to annotation-transfer / rescue reformulated."""
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, str] = {
            "retrieval_curves": str(out_dir / "cross_set_retrieval_curves.png"),
            "margin_distribution": str(out_dir / "cross_set_margin_distribution.png"),
            "top1_score_distribution": str(out_dir / "cross_set_top1_score_distribution.png"),
            "topk_heatmap": str(out_dir / "cross_set_topk_heatmap.png"),
            "top_peptides": str(out_dir / "cross_set_top_peptides.png"),
        }

        self._plot_retrieval_curves(topk_scores, Path(paths["retrieval_curves"]))
        self._plot_score_histogram(
            margins,
            Path(paths["margin_distribution"]),
            title="Top-1 − Top-2 Embedding Margin",
            xlabel="Cosine margin",
            color="#4E9AC6",
        )
        self._plot_score_histogram(
            top1_scores,
            Path(paths["top1_score_distribution"]),
            title="Top-1 Embedding Score Distribution",
            xlabel="Cosine similarity",
            color="#F5A45D",
        )
        self._plot_topk_heatmap(topk_scores, Path(paths["topk_heatmap"]))
        self._plot_top_peptides(candidates, Path(paths["top_peptides"]))

        if sim_matrix is not None:
            heatmap_path = out_dir / "cross_set_similarity_heatmap.png"
            self._plot_similarity_heatmap(sim_matrix, heatmap_path)
            paths["similarity_heatmap"] = str(heatmap_path)

        if (
            self.plot_umap
            and e_q is not None
            and e_lib is not None
            and query_indices is not None
            and library_indices is not None
            and meta is not None
        ):
            umap_path = out_dir / "cross_set_query_library_umap.png"
            saved = self._plot_query_library_umap(
                e_q=e_q,
                e_lib=e_lib,
                query_indices=query_indices,
                library_indices=library_indices,
                meta=meta,
                candidates=candidates,
                save_path=umap_path,
            )
            if saved:
                paths["query_library_umap"] = str(umap_path)

        return paths

    def _plot_retrieval_curves(self, topk_scores: np.ndarray, save_path: Path) -> None:
        if topk_scores.size == 0:
            return
        n_queries, max_k = topk_scores.shape
        ranks = np.arange(1, max_k + 1)
        mean_curve = topk_scores.mean(axis=0)
        p10 = np.percentile(topk_scores, 10, axis=0)
        p90 = np.percentile(topk_scores, 90, axis=0)

        fig, ax = plt.subplots(figsize=(8, 5))
        rng = np.random.RandomState(self.sample_seed)
        n_show = min(self.plot_max_curve_queries, n_queries)
        if n_show < n_queries:
            sample_idx = rng.choice(n_queries, n_show, replace=False)
        else:
            sample_idx = np.arange(n_queries)
        for idx in sample_idx:
            ax.plot(ranks, topk_scores[idx], color="#4E9AC6", alpha=0.04, linewidth=0.6)
        ax.fill_between(ranks, p10, p90, color="#4E9AC6", alpha=0.25, label="10–90% band")
        ax.plot(ranks, mean_curve, color="#1F4E79", linewidth=2.4, label="Mean score")
        ax.set_xlabel("Library rank")
        ax.set_ylabel("Embedding cosine similarity")
        ax.set_title("Cross-Set Retrieval Curves (ACFM → LCFM-valid)")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(save_path, dpi=self.plot_dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_score_histogram(
        self,
        values: np.ndarray,
        save_path: Path,
        *,
        title: str,
        xlabel: str,
        color: str,
    ) -> None:
        if values.size == 0:
            return
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(values, bins=60, color=color, edgecolor="white", alpha=0.9)
        ax.axvline(float(np.mean(values)), color="#333333", linestyle="--", linewidth=1.5, label=f"mean={np.mean(values):.4f}")
        ax.axvline(float(np.median(values)), color="#666666", linestyle=":", linewidth=1.5, label=f"median={np.median(values):.4f}")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Query count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_path, dpi=self.plot_dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_topk_heatmap(self, topk_scores: np.ndarray, save_path: Path) -> None:
        if topk_scores.size == 0:
            return
        n_queries = topk_scores.shape[0]
        step = max(1, int(np.ceil(n_queries / self.plot_heatmap_max_queries)))
        view = topk_scores[::step]
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(view, aspect="auto", interpolation="nearest", cmap="viridis", vmin=-1.0, vmax=1.0)
        ax.set_title("Top-k Similarity Heatmap (subsampled queries)")
        ax.set_xlabel("Rank")
        ax.set_ylabel(f"Queries (every {step} rows)")
        ax.set_xticks(np.arange(view.shape[1]))
        ax.set_xticklabels([str(i + 1) for i in range(view.shape[1])])
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Cosine similarity")
        fig.tight_layout()
        fig.savefig(save_path, dpi=self.plot_dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_top_peptides(self, candidates: List[Dict[str, Any]], save_path: Path) -> None:
        top1 = [str(row.get("library_peptide", "")) for row in candidates if int(row.get("rank", 0)) == 1]
        top1 = [pep for pep in top1 if pep and pep not in {"", "None", "nan"}]
        if not top1:
            return
        counts = Counter(top1).most_common(self.plot_top_peptides)
        labels = [pep if len(pep) <= 24 else pep[:21] + "..." for pep, _ in counts]
        values = [count for _, count in counts]
        fig, ax = plt.subplots(figsize=(9, max(4.0, 0.35 * len(labels) + 1.5)))
        y_pos = np.arange(len(labels))[::-1]
        ax.barh(y_pos, values, color="#2E78B8", alpha=0.9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Top-1 assignment count")
        ax.set_title(f"Most Frequent Top-1 Library Peptides (top {len(labels)})")
        fig.tight_layout()
        fig.savefig(save_path, dpi=self.plot_dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_query_library_umap(
        self,
        *,
        e_q: np.ndarray,
        e_lib: np.ndarray,
        query_indices: np.ndarray,
        library_indices: np.ndarray,
        meta: Dict[str, np.ndarray],
        candidates: List[Dict[str, Any]],
        save_path: Path,
    ) -> bool:
        """UMAP of the full library plus a handful of individually-labeled hero queries.

        Deliberately shows only ``umap_num_queries`` (default 5) queries, not a dense
        cloud of thousands -- the point is to be able to trace each dot back to a
        specific query and read off, by eye, whether it lands next to its own rank-1
        assigned library anchor (real embedding match) or somewhere unrelated
        (retrieval likely spurious / anisotropy artifact). Every point is text-labeled
        directly on the plot with its peptide, since there are too few points for a
        legend to be worth the indirection.
        """
        try:
            import umap
        except ImportError:
            return False

        n_queries = int(e_q.shape[0])
        n_library = int(e_lib.shape[0])
        if n_queries == 0 or n_library == 0:
            return False

        rng = np.random.RandomState(self.sample_seed)
        n_hero = min(self.umap_num_queries, n_queries)
        sample_rows = np.sort(rng.choice(n_queries, n_hero, replace=False))

        top1_peptide_by_query_index: Dict[int, str] = {
            int(row["query_index"]): str(row.get("library_peptide") or "")
            for row in candidates
            if int(row.get("rank", 0)) == 1
        }
        query_labels = [top1_peptide_by_query_index.get(int(query_indices[row]), "") for row in sample_rows]

        peptides_meta = np.asarray(meta[self.peptide_key], dtype=object)
        unmodified_meta = np.asarray(meta.get(self.unmodified_key, peptides_meta), dtype=object)
        library_labels = [
            str(peptides_meta[idx] or unmodified_meta[idx] or "") for idx in library_indices.tolist()
        ]

        # UMAP is built from the full library (the "map") plus only the hero queries,
        # so the library's manifold structure isn't distorted by thousands of query points.
        combined = np.concatenate([e_lib, e_q[sample_rows]], axis=0)
        n_neighbors = min(self.umap_n_neighbors, max(2, combined.shape[0] - 1))
        try:
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=self.umap_min_dist,
                metric="cosine",
                random_state=self.sample_seed,
            )
            coords = np.asarray(reducer.fit_transform(combined))
        except Exception:
            return False

        library_coords = coords[:n_library]
        query_coords = coords[n_library:]

        # Color hero queries distinctly; a library anchor gets the same color as any hero
        # query assigned to it (rank-1), so "does the query sit on its own color?" is a
        # direct visual check.
        hero_peptides = [pep for pep in dict.fromkeys(query_labels) if pep]
        cmap = plt.cm.get_cmap("tab10", max(len(hero_peptides), 1))
        color_map = {pep: cmap(i % 10) for i, pep in enumerate(hero_peptides)}

        fig, ax = plt.subplots(figsize=(11, 9))
        ax.scatter(
            library_coords[:, 0],
            library_coords[:, 1],
            c=[color_map.get(pep, "lightgray") for pep in library_labels],
            s=45,
            alpha=0.6,
            edgecolors="black",
            linewidths=0.3,
            zorder=2,
            label=f"Library anchors (n={n_library})",
        )

        marker_cycle = ["*", "D", "P", "X", "^", "v", "s", "o"]
        for i, (row, pep) in enumerate(zip(sample_rows.tolist(), query_labels)):
            color = color_map.get(pep, "red")
            label_text = pep if pep and len(pep) <= 22 else (pep[:19] + "..." if pep else "unassigned")
            ax.scatter(
                query_coords[i, 0],
                query_coords[i, 1],
                c=[color],
                s=320,
                marker=marker_cycle[i % len(marker_cycle)],
                edgecolors="black",
                linewidths=1.4,
                zorder=5,
            )
            ax.annotate(
                f"Q{i + 1}: {label_text}",
                (query_coords[i, 0], query_coords[i, 1]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=9,
                fontweight="bold",
                color=color if color != "red" else "black",
                zorder=6,
            )
            # Also label the query's own rank-1 anchor (if it exists among library_labels)
            # so it's visually paired with its query even when many anchors share a color.
            anchor_matches = [j for j, lib_pep in enumerate(library_labels) if lib_pep == pep]
            for j in anchor_matches[:1]:
                ax.annotate(
                    f"→Q{i + 1} anchor",
                    (library_coords[j, 0], library_coords[j, 1]),
                    textcoords="offset points",
                    xytext=(6, -10),
                    fontsize=7,
                    color=color,
                    zorder=4,
                )

        ax.set_title(
            f"UMAP: {n_hero} Hero Queries vs Full Library (n={n_library})\n"
            "Markers = queries (labeled by rank-1 assigned peptide); dots = library anchors"
        )
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        fig.tight_layout()
        fig.savefig(save_path, dpi=self.plot_dpi, bbox_inches="tight")
        plt.close(fig)
        return True

    def _plot_similarity_heatmap(self, sim_matrix: np.ndarray, save_path: Path) -> None:
        row_step = max(1, int(np.ceil(sim_matrix.shape[0] / self.plot_heatmap_max_queries)))
        col_step = max(1, int(np.ceil(sim_matrix.shape[1] / 512)))
        view = sim_matrix[::row_step, ::col_step]
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(view, aspect="auto", interpolation="nearest", cmap="viridis", vmin=-1.0, vmax=1.0)
        ax.set_title("Cross-Set Similarity Heatmap")
        ax.set_xlabel(f"Library spectra (every {col_step})")
        ax.set_ylabel(f"Query spectra (every {row_step})")
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Cosine similarity")
        fig.tight_layout()
        fig.savefig(save_path, dpi=self.plot_dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_evidence_metrics(
        self,
        candidates: List[Dict[str, Any]],
        out_dir: Path,
    ) -> Dict[str, str]:
        """Plot distributions of spectrum-level evidence metrics (blocks A/B/C).

        Complements the embedding-similarity plots with the chemical-evidence
        side: does the candidate peptide's theoretical spectrum (block B) or the
        library's own observed spectrum (block A) actually explain the query?
        """
        top1_rows = [row for row in candidates if int(row.get("rank", 0)) == 1]
        if not top1_rows:
            return {}

        paths: Dict[str, str] = {}

        block_a_cosine = self._extract_numeric_series(top1_rows, "q_obs__lib_obs__cosine_similarity")
        block_b_cosine = self._extract_numeric_series(top1_rows, "q_obs__lib_theo__cosine_similarity")
        block_a_matched = self._extract_numeric_series(top1_rows, "q_obs__lib_obs__matched_peak_count")
        block_b_matched = self._extract_numeric_series(top1_rows, "q_obs__lib_theo__matched_peak_count")
        embedding_scores = self._extract_numeric_series(top1_rows, "embedding_score")

        if block_a_cosine.size:
            path = out_dir / "cross_set_evidence_block_a_cosine_distribution.png"
            self._plot_score_histogram(
                block_a_cosine,
                path,
                title="Block A: Query Observed vs Library Observed Cosine (Top-1)",
                xlabel="Cosine similarity (binned spectra)",
                color="#6BAF92",
            )
            paths["evidence_block_a_cosine_distribution"] = str(path)

        if block_b_cosine.size:
            path = out_dir / "cross_set_evidence_block_b_cosine_distribution.png"
            self._plot_score_histogram(
                block_b_cosine,
                path,
                title="Block B: Query Observed vs Candidate Theoretical Cosine (Top-1)",
                xlabel="Cosine similarity (binned spectra)",
                color="#C77DA0",
            )
            paths["evidence_block_b_cosine_distribution"] = str(path)

        if block_b_matched.size:
            path = out_dir / "cross_set_evidence_matched_peak_count_distribution.png"
            self._plot_score_histogram(
                block_b_matched,
                path,
                title="Block B: Matched Fragment-Ion Count (Top-1)",
                xlabel="Matched peaks (query vs candidate theoretical spectrum)",
                color="#D6A24A",
            )
            paths["evidence_matched_peak_count_distribution"] = str(path)

        if embedding_scores.size and block_b_cosine.size and embedding_scores.size == block_b_cosine.size:
            path = out_dir / "cross_set_evidence_vs_embedding_scatter.png"
            self._plot_evidence_vs_embedding_scatter(embedding_scores, block_b_cosine, path)
            paths["evidence_vs_embedding_scatter"] = str(path)

        rank_curve_path = out_dir / "cross_set_evidence_rank_curves.png"
        if self._plot_evidence_rank_curves(candidates, rank_curve_path):
            paths["evidence_rank_curves"] = str(rank_curve_path)

        return paths

    @staticmethod
    def _extract_numeric_series(rows: List[Dict[str, Any]], key: str) -> np.ndarray:
        values: list[float] = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric == numeric:  # filter NaN
                values.append(numeric)
        return np.asarray(values, dtype=np.float64)

    def _plot_evidence_vs_embedding_scatter(
        self,
        embedding_scores: np.ndarray,
        evidence_cosine: np.ndarray,
        save_path: Path,
    ) -> None:
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.scatter(embedding_scores, evidence_cosine, s=10, alpha=0.35, color="#4E9AC6", edgecolors="none")
        if embedding_scores.size >= 2 and embedding_scores.std() > 0 and evidence_cosine.std() > 0:
            corr = float(np.corrcoef(embedding_scores, evidence_cosine)[0, 1])
            ax.set_title(f"Embedding Score vs Block B Evidence (Top-1, r={corr:.3f})")
        else:
            ax.set_title("Embedding Score vs Block B Evidence (Top-1)")
        ax.set_xlabel("Embedding cosine similarity")
        ax.set_ylabel("Block B cosine (query obs vs candidate theoretical)")
        fig.tight_layout()
        fig.savefig(save_path, dpi=self.plot_dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_evidence_rank_curves(self, candidates: List[Dict[str, Any]], save_path: Path) -> bool:
        by_rank: Dict[int, Dict[str, list[float]]] = {}
        for row in candidates:
            rank = int(row.get("rank", 0))
            if rank <= 0:
                continue
            bucket = by_rank.setdefault(rank, {"A": [], "B": []})
            for block, key in (("A", "q_obs__lib_obs__cosine_similarity"), ("B", "q_obs__lib_theo__cosine_similarity")):
                value = row.get(key)
                if value is None:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if numeric == numeric:
                    bucket[block].append(numeric)

        ranks = sorted(by_rank)
        mean_a = [np.mean(by_rank[r]["A"]) if by_rank[r]["A"] else np.nan for r in ranks]
        mean_b = [np.mean(by_rank[r]["B"]) if by_rank[r]["B"] else np.nan for r in ranks]
        if not ranks or (np.all(np.isnan(mean_a)) and np.all(np.isnan(mean_b))):
            return False

        fig, ax = plt.subplots(figsize=(8, 5))
        if not np.all(np.isnan(mean_a)):
            ax.plot(ranks, mean_a, marker="o", color="#6BAF92", label="Block A (query vs library observed)")
        if not np.all(np.isnan(mean_b)):
            ax.plot(ranks, mean_b, marker="s", color="#C77DA0", label="Block B (query vs candidate theoretical)")
        ax.set_xlabel("Library rank")
        ax.set_ylabel("Mean cosine similarity")
        ax.set_title("Spectral Evidence by Retrieval Rank")
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_path, dpi=self.plot_dpi, bbox_inches="tight")
        plt.close(fig)
        return True

    @staticmethod
    def _attach_evidence_to_candidates(
        candidates: List[Dict[str, Any]],
        evidence_metrics: Dict[Tuple[int, int], Dict[str, Any]],
    ) -> None:
        for row in candidates:
            key = (int(row["query_index"]), int(row["rank"]))
            metrics = evidence_metrics.get(key)
            if not metrics:
                continue
            for metric_key, metric_value in metrics.items():
                if metric_key in {"query_index", "rank", "library_index"}:
                    continue
                row[metric_key] = metric_value

    @staticmethod
    def _write_candidates_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            return
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
