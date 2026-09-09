"""Reformulated spectral rescue evaluation task.

See ``spectral_rescue_scope.py`` for ``single_base`` vs ``multi_base`` definitions,
when to use each, and how to configure dataset building + evaluation.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from scipy.stats import pearsonr, spearmanr

from instanovo_fm.eval.embed_eval_tasks import BaseTask
from instanovo_fm.eval.embed_eval_tasks.spectral_rescue_reformulated_plots import (
    save_rescue_publication_plots,
)
from instanovo_fm.eval.embed_eval_tasks.spectral_rescue_scope import (
    DEFAULT_PAIR_ID,
    RescuePairSpec,
    RescueScopeConfig,
    resolve_rescue_scope_config,
    scope_guidance_text,
)

logger = logging.getLogger(__name__)


class SpectralRescueTaskReformulated(BaseTask):
    """Controlled multi-query spectral rescue evaluation."""

    name = "Spectral Rescue Reformulated"
    description = "Evaluate controlled rescue retrieval with base and modified same-backbone queries"
    requires_metadata = True
    requires_faiss = False

    # Mirrors ResidueSet.tokenizer_regex: internal modifications stay attached
    # to the amino-acid token, e.g. M[UNIMOD:35].
    TOKENIZER_REGEX = re.compile(
        r"(\[[^\]]+\]|\([^)]+\)|[+-]?\d+(?:\.\d+)?|[+-]?\.\d+)|"
        r"([A-Z](?:\[[^\]]+\]|\([^)]+\)|[+-]?\d+(?:\.\d+)?|[+-]?\.\d+)?)"
    )
    AA_REGEX = re.compile(r"^[A-Z]")

    SAMPLING_PROFILES: Dict[str, Dict[str, Any]] = {
        "demo": {
            "num_reference_queries": 10,
            "num_modified_queries": 10,
            "num_positive_library": 20,
            "num_negative_library": 200,
            "max_queries_per_raw_file": None,
            "negative_sampling": "random",
            "seeds": [42],
        },
        "rigorous": {
            # 100 queries × 250 library = 25,000 pairs/seed (annotation-transfer
            # uses batched M×N matmul + plot subsampling beyond ~50k scores).
            "num_reference_queries": 50,
            "num_modified_queries": 50,
            "num_positive_library": 50,
            "num_negative_library": 200,
            "max_queries_per_raw_file": 1,
            "negative_sampling": "stratified_by_edit_distance",
            "seeds": [42, 43, 44],
        },
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.project_id: str = kwargs.get("project_id", "PXD047134")
        self.project_key: str = kwargs.get("project_key", "search_project")
        self.usi_key: str = kwargs.get("usi_key", "usi")
        self.role_key: str = kwargs.get("role_key", "rescue_role")
        self.sequence_key: str = kwargs.get("sequence_key", "peptides")
        self.sequence_fallback_keys: List[str] = kwargs.get("sequence_fallback_keys", ["sequence"])
        self.unmodified_sequence_key: str = kwargs.get("unmodified_sequence_key", "unmodified_peptide")
        self.base_sequence: str = kwargs.get("base_sequence", "LEQGQALDDLMPAQK")
        self.rescue_sequence: str = kwargs.get("rescue_sequence", "LEQGQALDDLM[UNIMOD:35]PAQK")
        self.k_values: List[int] = kwargs.get("k_values", [1, 5, 10, 20, 50, 100])
        self.reference_query_role: str = kwargs.get("reference_query_role", "reference_query")
        self.modified_query_role: str = kwargs.get("modified_query_role", "modified_query")
        self.positive_library_role: str = kwargs.get("positive_library_role", "positive_library")
        self.negative_library_role: str = kwargs.get("negative_library_role", "negative_library")
        self.num_reference_queries: int = kwargs.get("num_reference_queries", kwargs.get("num_base_queries", 10))
        self.num_modified_queries: int = kwargs.get("num_modified_queries", kwargs.get("num_rescue_queries", 10))
        self.max_base_positives: Optional[int] = kwargs.get("max_base_positives", 20)
        self.max_negatives: Optional[int] = kwargs.get("max_negatives", 200)
        self.min_negative_clean_edit_distance: int = kwargs.get("min_negative_clean_edit_distance", 5)
        self.sample_seed: int = kwargs.get("sample_seed", 42)
        self.max_queries_per_raw_file: Optional[int] = kwargs.get("max_queries_per_raw_file", None)
        self.negative_sampling: str = kwargs.get("negative_sampling", "random")
        self.save_artifacts: bool = kwargs.get("save_artifacts", True)
        self.save_plots: bool = kwargs.get("save_plots", True)
        self.plot_dpi: int = kwargs.get("plot_dpi", 160)
        self.similarity_batch_size: int = kwargs.get("similarity_batch_size", 500)
        self.plot_max_pair_scores: int = kwargs.get("plot_max_pair_scores", 50_000)
        self.top_n_ranked: int = kwargs.get("top_n_ranked", 200)
        self.output_dir: Optional[str] = kwargs.get("output_dir", None)
        self.scope: str = kwargs.get("scope", "single_base")
        self.pair_key: str = kwargs.get("pair_key", "rescue_pair_id")
        self.rescue_pairs: Optional[List[Dict[str, Any]]] = kwargs.get("rescue_pairs")
        self.rescue_pairs_path: Optional[str] = kwargs.get("rescue_pairs_path")
        self.plot_pair_id: Optional[str] = kwargs.get("plot_pair_id")

        self.scope_config: RescueScopeConfig = resolve_rescue_scope_config(
            scope=self.scope,
            pair_key=self.pair_key,
            project_id=self.project_id,
            base_sequence=self.base_sequence,
            rescue_sequence=self.rescue_sequence,
            rescue_pairs=self.rescue_pairs,
            rescue_pairs_path=self.rescue_pairs_path,
        )
        primary_pair = self.scope_config.pairs[0]
        self.base_sequence = primary_pair.base_sequence
        self.rescue_sequence = primary_pair.modified_sequence
        self.project_id = primary_pair.project_id
        self.base_sequence_canon = self._canonicalize_modified_sequence(self.base_sequence)
        self.rescue_sequence_canon = self._canonicalize_modified_sequence(self.rescue_sequence)
        self.base_backbone = self._backbone_sequence(self.base_sequence)

    def run(
        self,
        emb: np.ndarray,
        meta: Dict[str, np.ndarray],
        faiss_index: Any = None,
    ) -> Dict[str, Any]:
        """Run the controlled rescue experiment."""
        self.validate_inputs(emb, meta, faiss_index=None)
        start_time = time.time()

        try:
            if self.scope_config.is_multi_base:
                results = self._run_multi_base(emb, meta, start_time)
            else:
                selection = self._build_selection(meta)
                results = self._run_single_selection(emb, meta, selection, pair_id=DEFAULT_PAIR_ID)
        except ValueError as exc:
            return {
                "task_name": self.name,
                "scope": self.scope_config.scope,
                "scope_guidance": scope_guidance_text(self.scope_config.scope),
                "error": str(exc),
                "execution_time": time.time() - start_time,
            }

        results["scope"] = self.scope_config.scope
        results["scope_guidance"] = scope_guidance_text(self.scope_config.scope)
        results["execution_time"] = time.time() - start_time
        return results

    def _run_multi_base(
        self,
        emb: np.ndarray,
        meta: Dict[str, np.ndarray],
        start_time: float,
    ) -> Dict[str, Any]:
        pair_payloads: Dict[str, Dict[str, Any]] = {}
        for pair_spec in self.scope_config.pairs:
            selection = self._build_selection_for_pair(meta, pair_spec)
            pair_payloads[pair_spec.pair_id] = self._run_single_selection(
                emb,
                meta,
                selection,
                pair_id=pair_spec.pair_id,
                pair_spec=pair_spec,
                save_outputs=False,
            )

        panel_metrics = self._aggregate_panel_metrics(pair_payloads)
        results: Dict[str, Any] = {
            "task_name": self.name,
            "num_pairs": len(pair_payloads),
            "pair_ids": list(pair_payloads.keys()),
            "num_embeddings": int(len(emb)),
            "pair_results": {
                pair_id: {key: value for key, value in payload.items() if not key.startswith("_")}
                for pair_id, payload in pair_payloads.items()
            },
            **panel_metrics,
        }

        if self.output_dir:
            out_dir = Path(self.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            self._save_panel_summary(out_dir, pair_payloads, panel_metrics)
            for pair_id, payload in pair_payloads.items():
                pair_dir = out_dir / "pairs" / pair_id
                pair_dir.mkdir(parents=True, exist_ok=True)
                if self.save_artifacts:
                    payload["artifact_paths"] = self._save_artifacts(
                        pair_dir,
                        payload["_similarity_matrix"],
                        payload["_selection"],
                        payload["_query_metrics"],
                        payload["_ranked_rows"],
                    )
            hero_pair_id = self.plot_pair_id or self.scope_config.pairs[0].pair_id
            hero = pair_payloads.get(hero_pair_id)
            if self.save_plots and hero is not None:
                results["plot_paths"] = self._save_plots(
                    out_dir,
                    hero["_similarity_matrix"],
                    hero["_selection"],
                    hero["_ranked_rows"],
                    hero["_query_metrics"],
                    meta,
                )
                results["plot_pair_id"] = hero_pair_id

        return results

    def _run_single_selection(
        self,
        emb: np.ndarray,
        meta: Dict[str, np.ndarray],
        selection: Dict[str, Any],
        *,
        pair_id: str,
        pair_spec: Optional[RescuePairSpec] = None,
        save_outputs: bool = True,
    ) -> Dict[str, Any]:
        query_indices = np.array(selection["query_indices"], dtype=np.int64)
        library_indices = np.array(selection["library_indices"], dtype=np.int64)

        E_q = self._l2_normalise(emb[query_indices].astype(np.float32).copy())
        E_lib = self._l2_normalise(emb[library_indices].astype(np.float32).copy())
        S = self._compute_similarity_batched(E_q, E_lib, self.similarity_batch_size)

        library_roles = np.array(selection["library_roles"], dtype=object)
        positive_mask = library_roles == self.positive_library_role

        query_metrics, ranked_rows = self._compute_all_query_metrics(S, positive_mask, selection)
        aggregate_metrics = self._aggregate_query_metrics(query_metrics)

        active_pair = pair_spec or self.scope_config.pairs[0]
        results: Dict[str, Any] = {
            "task_name": self.name,
            "pair_id": pair_id,
            "project_id": active_pair.project_id,
            "base_sequence": self._canonicalize_modified_sequence(active_pair.base_sequence),
            "rescue_sequence": self._canonicalize_modified_sequence(active_pair.modified_sequence),
            "base_backbone": self._backbone_sequence(active_pair.base_sequence),
            "num_embeddings": int(len(emb)),
            "num_project_spectra": int(selection["num_project_spectra"]),
            "num_base_sequence_spectra": int(selection["num_base_sequence_spectra"]),
            "num_rescue_sequence_spectra": int(selection["num_rescue_sequence_spectra"]),
            "num_same_backbone_spectra": int(selection["num_same_backbone_spectra"]),
            "num_reference_queries": int(selection["num_reference_queries"]),
            "num_modified_queries": int(selection["num_modified_queries"]),
            "num_queries": int(len(query_indices)),
            "num_base_positives": int(selection["num_base_positives"]),
            "num_negative_candidates": int(selection["num_negative_candidates"]),
            "num_negatives": int(selection["num_negatives"]),
            "num_library": int(len(library_indices)),
            "num_similarity_pairs": int(S.size),
            "reference_query_indices": [int(idx) for idx in selection["reference_query_indices"]],
            "modified_query_indices": [int(idx) for idx in selection["modified_query_indices"]],
            **aggregate_metrics,
            "_similarity_matrix": S,
            "_selection": selection,
            "_query_metrics": query_metrics,
            "_ranked_rows": ranked_rows,
        }

        if save_outputs and self.output_dir:
            out_dir = Path(self.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            if self.save_artifacts:
                results["artifact_paths"] = self._save_artifacts(
                    out_dir,
                    S,
                    selection,
                    query_metrics,
                    ranked_rows,
                )
            if self.save_plots:
                results["plot_paths"] = self._save_plots(out_dir, S, selection, ranked_rows, query_metrics, meta)

            # Internal arrays are only needed while writing artifacts/plots. Keeping
            # them in task results makes task_results.json huge and can break JSON
            # serialization for object arrays.
            results = {key: value for key, value in results.items() if not key.startswith("_")}

        return results

    @staticmethod
    def _aggregate_panel_metrics(pair_payloads: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        metric_keys = [
            "reference_query_best_positive_rank_mean",
            "modified_query_best_positive_rank_mean",
            "reference_query_margin_mean",
            "modified_query_margin_mean",
            "reference_query_recall@1_mean",
            "modified_query_recall@1_mean",
            "reference_query_recall@10_mean",
            "modified_query_recall@10_mean",
        ]
        panel: Dict[str, Any] = {"panel_num_pairs": len(pair_payloads)}
        for key in metric_keys:
            values = [
                float(payload[key])
                for payload in pair_payloads.values()
                if key in payload and np.isfinite(float(payload[key]))
            ]
            if not values:
                continue
            panel[f"panel_{key}"] = float(np.mean(values))
            panel[f"panel_{key}_std"] = float(np.std(values)) if len(values) > 1 else 0.0
            panel[f"panel_{key}_per_pair"] = values
        return panel

    def _save_panel_summary(
        self,
        out_dir: Path,
        pair_payloads: Dict[str, Dict[str, Any]],
        panel_metrics: Dict[str, Any],
    ) -> None:
        rows = []
        for pair_id, payload in pair_payloads.items():
            rows.append(
                {
                    "pair_id": pair_id,
                    "project_id": payload.get("project_id", ""),
                    "base_sequence": payload.get("base_sequence", ""),
                    "rescue_sequence": payload.get("rescue_sequence", ""),
                    "num_reference_queries": payload.get("num_reference_queries", 0),
                    "num_modified_queries": payload.get("num_modified_queries", 0),
                    "num_library": payload.get("num_library", 0),
                    "reference_query_recall@1_mean": payload.get("reference_query_recall@1_mean"),
                    "modified_query_recall@1_mean": payload.get("modified_query_recall@1_mean"),
                    "reference_query_margin_mean": payload.get("reference_query_margin_mean"),
                    "modified_query_margin_mean": payload.get("modified_query_margin_mean"),
                }
            )
        summary_csv = out_dir / "rescue_pair_summary.csv"
        with summary_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        summary_json = out_dir / "rescue_panel_summary.json"
        summary_json.write_text(json.dumps({"panel_metrics": panel_metrics, "pairs": rows}, indent=2))

    def get_loggable_metrics(self, task_results: Dict[str, Any]) -> Dict[str, float]:
        if "error" in task_results:
            return {}
        keys = [
            "reference_query_best_positive_rank_mean",
            "modified_query_best_positive_rank_mean",
            "reference_query_margin_mean",
            "modified_query_margin_mean",
            "reference_query_recall@1_mean",
            "modified_query_recall@1_mean",
            "reference_query_recall@10_mean",
            "modified_query_recall@10_mean",
            "panel_reference_query_recall@1_mean",
            "panel_modified_query_recall@1_mean",
            "panel_reference_query_margin_mean",
            "panel_modified_query_margin_mean",
        ]
        return {key: float(task_results[key]) for key in keys if key in task_results}

    # ------------------------------------------------------------------
    # Tokenization and sequence helpers
    # ------------------------------------------------------------------

    @classmethod
    def _tokenize(cls, sequence: Any) -> List[str]:
        if sequence is None:
            return []
        if isinstance(sequence, list):
            return [str(token) for token in sequence]
        seq = str(sequence).strip()
        if not seq or seq.lower() in {"none", "nan", "null"}:
            return []
        return [item for groups in cls.TOKENIZER_REGEX.findall(seq) for item in groups if item]

    @classmethod
    def _token_amino_acid(cls, token: str) -> Optional[str]:
        match = cls.AA_REGEX.match(token)
        if match is None:
            return None
        aa = match.group(0)
        return "L" if aa == "I" else aa

    @classmethod
    def _token_has_modification(cls, token: str) -> bool:
        aa = cls._token_amino_acid(token)
        if aa is None:
            return True
        return len(token) > 1

    @classmethod
    def _canonicalize_token(cls, token: str) -> str:
        aa = cls._token_amino_acid(token)
        if aa is None:
            return token
        return aa + token[1:]

    @classmethod
    def _canonicalize_modified_sequence(cls, sequence: Any) -> str:
        return "".join(cls._canonicalize_token(token) for token in cls._tokenize(sequence))

    @classmethod
    def _backbone_sequence(cls, sequence: Any) -> str:
        residues = [cls._token_amino_acid(token) for token in cls._tokenize(sequence)]
        return "".join(residue for residue in residues if residue is not None)

    @classmethod
    def _clean_site_edit_distance(cls, seq_a: Any, seq_b: Any) -> int:
        tokens_a = cls._tokenize(seq_a)
        tokens_b = cls._tokenize(seq_b)
        residues_a = [cls._token_amino_acid(token) for token in tokens_a]
        residues_b = [cls._token_amino_acid(token) for token in tokens_b]
        clean_a: List[str] = []
        clean_b: List[str] = []

        for token_a, token_b, aa_a, aa_b in zip(tokens_a, tokens_b, residues_a, residues_b):
            if aa_a is None or aa_b is None:
                continue
            if cls._token_has_modification(token_a) or cls._token_has_modification(token_b):
                continue
            clean_a.append(aa_a)
            clean_b.append(aa_b)

        length_penalty = abs(len([aa for aa in residues_a if aa is not None]) - len([aa for aa in residues_b if aa is not None]))
        return cls._levenshtein_tokens(clean_a, clean_b) + length_penalty

    @staticmethod
    def _levenshtein_tokens(tokens_a: List[str], tokens_b: List[str]) -> int:
        if tokens_a == tokens_b:
            return 0
        if not tokens_a:
            return len(tokens_b)
        if not tokens_b:
            return len(tokens_a)
        prev = list(range(len(tokens_b) + 1))
        for i, token_a in enumerate(tokens_a, 1):
            curr = [i] + [0] * len(tokens_b)
            for j, token_b in enumerate(tokens_b, 1):
                cost = 0 if token_a == token_b else 1
                curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            prev = curr
        return prev[-1]

    @staticmethod
    def _l2_normalise(E: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        E /= np.where(norms == 0, 1.0, norms)
        return E

    @staticmethod
    def _compute_similarity_batched(
        E_q: np.ndarray,
        E_lib: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:
        """Compute (M, N) cosine similarity matrix in query-row batches."""
        m_queries = E_q.shape[0]
        similarity = np.empty((m_queries, E_lib.shape[0]), dtype=np.float32)
        for start in range(0, m_queries, batch_size):
            end = min(start + batch_size, m_queries)
            similarity[start:end] = E_q[start:end] @ E_lib.T
        return similarity

    @staticmethod
    def raw_file_from_usi(usi: Any) -> str:
        """Return the raw-file component from a mzSpec USI string."""
        parts = str(usi).split(":")
        if len(parts) >= 3:
            return parts[2]
        return str(usi)

    @classmethod
    def dedupe_by_raw_file(
        cls,
        records: List[Dict[str, Any]],
        max_per_raw_file: Optional[int],
        rng: np.random.RandomState,
    ) -> List[Dict[str, Any]]:
        """Cap candidate spectra per raw file before sampling queries or library rows."""
        if max_per_raw_file is None:
            return list(records)

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            raw_file = cls.raw_file_from_usi(record.get("usi", ""))
            grouped.setdefault(raw_file, []).append(record)

        deduped: List[Dict[str, Any]] = []
        for file_records in grouped.values():
            if len(file_records) <= max_per_raw_file:
                deduped.extend(file_records)
                continue
            chosen = rng.choice(len(file_records), size=max_per_raw_file, replace=False)
            deduped.extend(file_records[int(index)] for index in sorted(chosen.tolist()))
        return deduped

    @classmethod
    def edit_distance_bin(cls, distance: int, *, bin_width: int = 5) -> str:
        """Bucket clean edit distances for stratified negative sampling."""
        lower = (distance // bin_width) * bin_width
        upper = lower + bin_width - 1
        return f"{lower}-{upper}"

    @classmethod
    def sample_records(
        cls,
        records: List[Dict[str, Any]],
        n: int,
        rng: np.random.RandomState,
        label: str,
        *,
        strategy: str = "random",
        distance_key: str = "clean_site_edit_distance",
        bin_width: int = 5,
    ) -> List[Dict[str, Any]]:
        """Sample candidate records uniformly or with equal weight per edit-distance bin."""
        if len(records) < n:
            raise ValueError(f"Need {n} {label} records, found {len(records)}")

        if strategy == "random":
            chosen = rng.choice(len(records), size=n, replace=False)
            return [records[int(index)] for index in sorted(chosen.tolist())]

        if strategy != "stratified_by_edit_distance":
            raise ValueError(f"Unsupported sampling strategy {strategy!r}")

        bins: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            if distance_key not in record:
                raise ValueError(f"Record missing {distance_key!r} required for stratified sampling")
            bucket = cls.edit_distance_bin(int(record[distance_key]), bin_width=bin_width)
            bins.setdefault(bucket, []).append(record)

        bin_keys = sorted(bins.keys())
        if not bin_keys:
            raise ValueError(f"No bins available to sample {label} records")

        base_count, remainder = divmod(n, len(bin_keys))
        selected: List[Dict[str, Any]] = []
        selected_keys: set[tuple[Any, ...]] = set()

        def _record_key(record: Dict[str, Any]) -> tuple[Any, ...]:
            if "source_shard" in record and "row_in_shard" in record:
                return (record["source_shard"], record["row_in_shard"])
            return (record.get("usi", id(record)),)

        for index, bucket in enumerate(bin_keys):
            need = base_count + (1 if index < remainder else 0)
            if need == 0:
                continue
            pool = bins[bucket]
            take = min(need, len(pool))
            if take == 0:
                continue
            chosen = rng.choice(len(pool), size=take, replace=False)
            for choice in chosen.tolist():
                record = pool[int(choice)]
                key = _record_key(record)
                if key in selected_keys:
                    continue
                selected_keys.add(key)
                selected.append(record)

        if len(selected) < n:
            remaining = [record for record in records if _record_key(record) not in selected_keys]
            if len(selected) + len(remaining) < n:
                raise ValueError(f"Need {n} {label} records, found {len(records)}")
            deficit = n - len(selected)
            chosen = rng.choice(len(remaining), size=deficit, replace=False)
            selected.extend(remaining[int(i)] for i in sorted(chosen.tolist()))

        rng.shuffle(selected)
        return selected

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _discover_pair_ids(self, meta: Dict[str, np.ndarray]) -> List[str]:
        if self.pair_key in meta:
            return sorted(set(np.asarray(meta[self.pair_key], dtype=object).astype(str)))
        return [DEFAULT_PAIR_ID]

    def _pair_mask(self, meta: Dict[str, np.ndarray], pair_id: str, length: int) -> np.ndarray:
        if self.pair_key not in meta:
            if pair_id != DEFAULT_PAIR_ID:
                raise ValueError(
                    f"Metadata missing {self.pair_key!r} but pair_id={pair_id!r} was requested"
                )
            return np.ones(length, dtype=bool)
        mask = np.asarray(meta[self.pair_key], dtype=object).astype(str) == str(pair_id)
        if not mask.any():
            raise ValueError(f"No rows with {self.pair_key}={pair_id!r}")
        return mask

    def _build_selection_for_pair(self, meta: Dict[str, np.ndarray], pair_spec: RescuePairSpec) -> Dict[str, Any]:
        sequences = self._get_meta_array(meta, self.sequence_key, self.sequence_fallback_keys)
        pair_mask = self._pair_mask(meta, pair_spec.pair_id, len(sequences))
        if self.role_key in meta:
            return self._build_selection_from_roles(
                meta,
                sequences,
                self._get_meta_array(meta, self.unmodified_sequence_key, []),
                self._get_projects(meta),
                self._get_optional_meta_array(meta, self.usi_key, len(sequences)),
                self._get_optional_meta_array(meta, "rescue_source_shard", len(sequences)),
                self._get_optional_meta_array(meta, "rescue_row_in_shard", len(sequences)),
                pair_mask=pair_mask,
                pair_spec=pair_spec,
            )

        saved_project = self.project_id
        saved_base = self.base_sequence
        saved_rescue = self.rescue_sequence
        saved_base_canon = self.base_sequence_canon
        saved_rescue_canon = self.rescue_sequence_canon
        saved_backbone = self.base_backbone
        try:
            self.project_id = pair_spec.project_id
            self.base_sequence = pair_spec.base_sequence
            self.rescue_sequence = pair_spec.modified_sequence
            self.base_sequence_canon = self._canonicalize_modified_sequence(pair_spec.base_sequence)
            self.rescue_sequence_canon = self._canonicalize_modified_sequence(pair_spec.modified_sequence)
            self.base_backbone = self._backbone_sequence(pair_spec.base_sequence)
            selection = self._build_selection(meta)
        finally:
            self.project_id = saved_project
            self.base_sequence = saved_base
            self.rescue_sequence = saved_rescue
            self.base_sequence_canon = saved_base_canon
            self.rescue_sequence_canon = saved_rescue_canon
            self.base_backbone = saved_backbone

        pair_indices = set(np.where(pair_mask)[0].tolist())
        for key in ("reference_query_indices", "modified_query_indices", "library_indices"):
            outside = [idx for idx in selection[key] if idx not in pair_indices]
            if outside:
                raise ValueError(
                    f"Dynamic selection for pair {pair_spec.pair_id!r} returned rows outside pair mask: "
                    f"{outside[:5]}"
                )
        selection["pair_id"] = pair_spec.pair_id
        return selection

    def _build_selection(self, meta: Dict[str, np.ndarray]) -> Dict[str, Any]:
        if self.scope_config.is_multi_base:
            raise ValueError(
                "multi_base scope evaluates each pair separately; use offline data with rescue_pair_id "
                "and scope=multi_base"
            )
        pair_ids = self._discover_pair_ids(meta)
        if len(pair_ids) > 1:
            raise ValueError(
                f"single_base scope found multiple {self.pair_key} values {pair_ids}. "
                f"Use scope=multi_base for panels.\n{scope_guidance_text()}"
            )

        rng = np.random.RandomState(self.sample_seed)

        sequences = self._get_meta_array(meta, self.sequence_key, self.sequence_fallback_keys)
        unmodified = self._get_meta_array(meta, self.unmodified_sequence_key, [])
        projects = self._get_projects(meta)
        usis = self._get_optional_meta_array(meta, self.usi_key, len(sequences))
        source_shards = self._get_optional_meta_array(meta, "rescue_source_shard", len(sequences))
        source_rows = self._get_optional_meta_array(meta, "rescue_row_in_shard", len(sequences))
        if self.role_key in meta:
            pair_mask = None
            if self.pair_key in meta:
                pair_mask = self._pair_mask(meta, pair_ids[0], len(sequences))
            return self._build_selection_from_roles(
                meta,
                sequences,
                unmodified,
                projects,
                usis,
                source_shards,
                source_rows,
                pair_mask=pair_mask,
                pair_spec=self.scope_config.pairs[0],
            )

        modified_canon = np.array([self._canonicalize_modified_sequence(seq) for seq in sequences], dtype=object)
        backbone = np.array([self._backbone_sequence(seq) for seq in unmodified], dtype=object)

        project_mask = projects == self.project_id
        base_mask = project_mask & (modified_canon == self.base_sequence_canon)
        rescue_mask = project_mask & (modified_canon == self.rescue_sequence_canon)
        same_backbone_mask = project_mask & (backbone == self.base_backbone)

        base_indices = np.where(base_mask)[0]
        rescue_indices = np.where(rescue_mask)[0]

        if len(base_indices) < self.num_reference_queries + 1:
            raise ValueError(
                f"Need at least {self.num_reference_queries + 1} base-sequence spectra for "
                f"{self.base_sequence_canon} in {self.project_id}; found {len(base_indices)}"
            )
        if len(rescue_indices) < self.num_modified_queries:
            raise ValueError(
                f"Need at least {self.num_modified_queries} rescue-sequence spectra for "
                f"{self.rescue_sequence_canon} in {self.project_id}; found {len(rescue_indices)}"
            )

        base_records = [
            {"index": int(idx), "usi": str(usis[idx])}
            for idx in base_indices.tolist()
        ]
        rescue_records = [
            {"index": int(idx), "usi": str(usis[idx])}
            for idx in rescue_indices.tolist()
        ]
        base_records = self.dedupe_by_raw_file(base_records, self.max_queries_per_raw_file, rng)
        rescue_records = self.dedupe_by_raw_file(rescue_records, self.max_queries_per_raw_file, rng)

        reference_query_indices = np.sort(
            np.array(
                [
                    record["index"]
                    for record in self.sample_records(
                        base_records,
                        self.num_reference_queries,
                        rng,
                        "reference query",
                    )
                ],
                dtype=np.int64,
            )
        )
        modified_query_indices = np.sort(
            np.array(
                [
                    record["index"]
                    for record in self.sample_records(
                        rescue_records,
                        self.num_modified_queries,
                        rng,
                        "modified query",
                    )
                ],
                dtype=np.int64,
            )
        )

        reference_query_set = set(reference_query_indices.tolist())
        base_positive_indices = np.array([idx for idx in base_indices if idx not in reference_query_set], dtype=np.int64)
        if self.max_base_positives is not None and len(base_positive_indices) > self.max_base_positives:
            base_positive_indices = np.sort(rng.choice(base_positive_indices, self.max_base_positives, replace=False))

        negative_candidate_indices = np.where(project_mask & (backbone != self.base_backbone))[0]
        negative_records: List[Dict[str, Any]] = []
        for idx in negative_candidate_indices:
            distance = self._clean_site_edit_distance(self.base_sequence_canon, sequences[idx])
            if distance >= self.min_negative_clean_edit_distance:
                negative_records.append(
                    {
                        "index": int(idx),
                        "usi": str(usis[idx]),
                        "clean_site_edit_distance": int(distance),
                    }
                )

        if not negative_records:
            raise ValueError(
                f"No negative candidates in {self.project_id} passed min_negative_clean_edit_distance="
                f"{self.min_negative_clean_edit_distance}"
            )

        negative_records = self.dedupe_by_raw_file(negative_records, self.max_queries_per_raw_file, rng)
        negative_count = self.max_negatives if self.max_negatives is not None else len(negative_records)
        if negative_count > len(negative_records):
            raise ValueError(
                f"Need {negative_count} negative library records after dedupe, found {len(negative_records)}"
            )
        sampled_negative_records = self.sample_records(
            negative_records,
            negative_count,
            rng,
            "negative library",
            strategy=self.negative_sampling,
        )
        negative_indices = np.sort(np.array([record["index"] for record in sampled_negative_records], dtype=np.int64))

        library_indices = np.concatenate([base_positive_indices, negative_indices]).astype(np.int64)
        library_roles = [self.positive_library_role] * len(base_positive_indices) + [self.negative_library_role] * len(negative_indices)

        self._validate_same_backbone_consistency(sequences, unmodified, base_indices, rescue_indices, same_backbone_mask)

        return {
            "reference_query_indices": reference_query_indices.tolist(),
            "modified_query_indices": modified_query_indices.tolist(),
            "query_indices": np.concatenate([reference_query_indices, modified_query_indices]).astype(np.int64).tolist(),
            "query_roles": [self.reference_query_role] * len(reference_query_indices)
            + [self.modified_query_role] * len(modified_query_indices),
            "library_indices": library_indices.tolist(),
            "library_roles": library_roles,
            "sequences": sequences,
            "unmodified": unmodified,
            "projects": projects,
            "usis": usis,
            "source_shards": source_shards,
            "source_rows": source_rows,
            "modified_canon": modified_canon,
            "backbone": backbone,
            "num_project_spectra": int(project_mask.sum()),
            "num_base_sequence_spectra": int(base_mask.sum()),
            "num_rescue_sequence_spectra": int(rescue_mask.sum()),
            "num_same_backbone_spectra": int(same_backbone_mask.sum()),
            "num_reference_queries": int(len(reference_query_indices)),
            "num_modified_queries": int(len(modified_query_indices)),
            "num_base_positives": int(len(base_positive_indices)),
            "num_negative_candidates": int(len(negative_candidate_indices)),
            "num_negatives": int(len(negative_indices)),
        }

    def _build_selection_from_roles(
        self,
        meta: Dict[str, np.ndarray],
        sequences: np.ndarray,
        unmodified: np.ndarray,
        projects: np.ndarray,
        usis: np.ndarray,
        source_shards: np.ndarray,
        source_rows: np.ndarray,
        *,
        pair_mask: Optional[np.ndarray] = None,
        pair_spec: Optional[RescuePairSpec] = None,
    ) -> Dict[str, Any]:
        roles = np.asarray(meta[self.role_key], dtype=object).astype(str)
        active_pair = pair_spec or self.scope_config.pairs[0]
        base_canon = self._canonicalize_modified_sequence(active_pair.base_sequence)
        rescue_canon = self._canonicalize_modified_sequence(active_pair.modified_sequence)
        base_backbone = self._backbone_sequence(active_pair.base_sequence)
        modified_canon = np.array([self._canonicalize_modified_sequence(seq) for seq in sequences], dtype=object)
        backbone = np.array([self._backbone_sequence(seq) for seq in unmodified], dtype=object)

        if pair_mask is None:
            pair_mask = np.ones(len(sequences), dtype=bool)

        reference_query_indices = np.where((roles == self.reference_query_role) & pair_mask)[0].astype(np.int64)
        modified_query_indices = np.where((roles == self.modified_query_role) & pair_mask)[0].astype(np.int64)
        positive_indices = np.where((roles == self.positive_library_role) & pair_mask)[0].astype(np.int64)
        negative_indices = np.where((roles == self.negative_library_role) & pair_mask)[0].astype(np.int64)

        if len(reference_query_indices) == 0:
            raise ValueError(f"No rows with {self.role_key}={self.reference_query_role!r}")
        if len(positive_indices) == 0:
            raise ValueError(f"No rows with {self.role_key}={self.positive_library_role!r}")
        if len(negative_indices) == 0:
            raise ValueError(f"No rows with {self.role_key}={self.negative_library_role!r}")

        query_set = set(reference_query_indices.tolist()) | set(modified_query_indices.tolist())
        library_set = set(positive_indices.tolist()) | set(negative_indices.tolist())
        overlap = query_set & library_set
        if overlap:
            raise ValueError(f"Rows cannot be both query and library entries: {sorted(overlap)[:10]}")

        library_indices = np.concatenate([positive_indices, negative_indices]).astype(np.int64)
        library_roles = [self.positive_library_role] * len(positive_indices) + [self.negative_library_role] * len(negative_indices)
        same_backbone_mask = backbone == base_backbone

        return {
            "pair_id": active_pair.pair_id,
            "reference_query_indices": reference_query_indices.tolist(),
            "modified_query_indices": modified_query_indices.tolist(),
            "query_indices": np.concatenate([reference_query_indices, modified_query_indices]).astype(np.int64).tolist(),
            "query_roles": [self.reference_query_role] * len(reference_query_indices)
            + [self.modified_query_role] * len(modified_query_indices),
            "library_indices": library_indices.tolist(),
            "library_roles": library_roles,
            "sequences": sequences,
            "unmodified": unmodified,
            "projects": projects,
            "usis": usis,
            "source_shards": source_shards,
            "source_rows": source_rows,
            "modified_canon": modified_canon,
            "backbone": backbone,
            "num_project_spectra": int(np.sum(pair_mask)),
            "num_base_sequence_spectra": int(np.sum(modified_canon[pair_mask] == base_canon)),
            "num_rescue_sequence_spectra": int(np.sum(modified_canon[pair_mask] == rescue_canon)),
            "num_same_backbone_spectra": int(np.sum(same_backbone_mask & pair_mask)),
            "num_reference_queries": int(len(reference_query_indices)),
            "num_modified_queries": int(len(modified_query_indices)),
            "num_base_positives": int(len(positive_indices)),
            "num_negative_candidates": int(len(negative_indices)),
            "num_negatives": int(len(negative_indices)),
        }

    def _validate_same_backbone_consistency(
        self,
        sequences: np.ndarray,
        unmodified: np.ndarray,
        base_indices: np.ndarray,
        rescue_indices: np.ndarray,
        same_backbone_mask: np.ndarray,
    ) -> None:
        for idx in base_indices[: min(10, len(base_indices))]:
            if self._backbone_sequence(sequences[idx]) != self.base_backbone:
                raise ValueError(f"Base sequence row {idx} does not match base backbone after tokenization")
        for idx in rescue_indices[: min(10, len(rescue_indices))]:
            if self._backbone_sequence(sequences[idx]) != self.base_backbone:
                raise ValueError(f"Rescue sequence row {idx} does not match base backbone after tokenization")

        same_backbone_indices = np.where(same_backbone_mask)[0]
        for idx in same_backbone_indices[: min(10, len(same_backbone_indices))]:
            if self._backbone_sequence(unmodified[idx]) != self.base_backbone:
                raise ValueError(f"Unmodified peptide row {idx} is inconsistent with base backbone")

    def _get_meta_array(self, meta: Dict[str, np.ndarray], key: str, fallback_keys: Iterable[str]) -> np.ndarray:
        keys = [key, *fallback_keys]
        for candidate in keys:
            if candidate in meta:
                return np.asarray(meta[candidate], dtype=object)
        raise ValueError(f"Required metadata key '{key}' not found. Tried {keys}. Available keys: {list(meta.keys())}")

    @staticmethod
    def _get_optional_meta_array(meta: Dict[str, np.ndarray], key: str, length: int) -> np.ndarray:
        if key in meta:
            return np.asarray(meta[key], dtype=object)
        return np.array([""] * length, dtype=object)

    def _get_projects(self, meta: Dict[str, np.ndarray]) -> np.ndarray:
        if self.project_key in meta:
            projects = np.asarray(meta[self.project_key], dtype=object)
            valid = np.array([str(project).strip() not in {"", "None", "nan"} for project in projects])
            if valid.any():
                return projects.astype(str)

        if self.usi_key not in meta:
            raise ValueError(
                f"Project key '{self.project_key}' not found and cannot derive from missing USI key '{self.usi_key}'"
            )

        projects = []
        for usi in meta[self.usi_key]:
            parts = str(usi).split(":")
            projects.append(parts[1].strip() if len(parts) >= 2 else "")
        return np.array(projects, dtype=object)

    # ------------------------------------------------------------------
    # Metrics and outputs
    # ------------------------------------------------------------------

    def _compute_all_query_metrics(
        self,
        S: np.ndarray,
        positive_mask: np.ndarray,
        selection: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        query_metrics: List[Dict[str, Any]] = []
        ranked_rows: List[Dict[str, Any]] = []
        for query_ordinal, (scores, query_idx, query_role) in enumerate(
            zip(S, selection["query_indices"], selection["query_roles"], strict=False)
        ):
            metrics, ranked = self._compute_query_metrics(
                scores,
                positive_mask,
                selection,
                query_role=query_role,
                query_index=int(query_idx),
                query_ordinal=query_ordinal,
            )
            query_metrics.append(metrics)
            ranked_rows.extend(ranked)
        return query_metrics, ranked_rows

    def _compute_query_metrics(
        self,
        scores: np.ndarray,
        positive_mask: np.ndarray,
        selection: Dict[str, Any],
        *,
        query_role: str,
        query_index: int,
        query_ordinal: int,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        order = np.argsort(scores)[::-1]
        positive_cols = set(np.where(positive_mask)[0].tolist())
        ranked_rows = self._ranked_rows(scores, order, positive_mask, selection, query_role, query_index, query_ordinal)

        positive_ranks = [rank for rank, col in enumerate(order, 1) if col in positive_cols]
        if not positive_ranks:
            raise ValueError(f"No positive library entries available for {query_role}")

        best_positive = float(np.max(scores[positive_mask]))
        best_negative = float(np.max(scores[~positive_mask])) if (~positive_mask).any() else float("nan")
        margin = best_positive - best_negative if np.isfinite(best_negative) else float("nan")

        metrics: Dict[str, Any] = {
            "query_role": query_role,
            "query_index": int(query_index),
            "query_ordinal": int(query_ordinal),
            "best_positive_rank": int(min(positive_ranks)),
            "best_positive_similarity": best_positive,
            "mean_positive_similarity": float(np.mean(scores[positive_mask])),
            "median_positive_similarity": float(np.median(scores[positive_mask])),
            "best_negative_similarity": best_negative,
            "margin": margin,
        }

        for k in self.k_values:
            top_k = order[:k]
            hits = sum(1 for col in top_k if col in positive_cols)
            metrics[f"recall@{k}"] = 1.0 if hits > 0 else 0.0
            metrics[f"prop_recall@{k}"] = hits / len(positive_cols)

        max_k = max(self.k_values)
        ap_hits = 0
        ap_sum = 0.0
        for rank, col in enumerate(order[:max_k], 1):
            if col in positive_cols:
                ap_hits += 1
                ap_sum += ap_hits / rank
        denom = min(len(positive_cols), max_k)
        metrics[f"map@{max_k}"] = ap_sum / denom if denom else 0.0
        return metrics, ranked_rows

    def _aggregate_query_metrics(self, query_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        aggregate: Dict[str, Any] = {}
        metric_keys = [
            "best_positive_rank",
            "best_positive_similarity",
            "mean_positive_similarity",
            "median_positive_similarity",
            "best_negative_similarity",
            "margin",
            f"map@{max(self.k_values)}",
        ]
        for k in self.k_values:
            metric_keys.extend([f"recall@{k}", f"prop_recall@{k}"])

        for role in (self.reference_query_role, self.modified_query_role):
            rows = [row for row in query_metrics if row["query_role"] == role]
            aggregate[f"{role}_count"] = len(rows)
            for key in metric_keys:
                values = np.array([float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))], dtype=np.float64)
                if len(values) == 0:
                    continue
                aggregate[f"{role}_{key}_mean"] = float(np.mean(values))
                aggregate[f"{role}_{key}_std"] = float(np.std(values))
                aggregate[f"{role}_{key}_min"] = float(np.min(values))
                aggregate[f"{role}_{key}_max"] = float(np.max(values))
        return aggregate

    def _ranked_rows(
        self,
        scores: np.ndarray,
        order: np.ndarray,
        positive_mask: np.ndarray,
        selection: Dict[str, Any],
        query_role: str,
        query_index: int,
        query_ordinal: int,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        library_indices = selection["library_indices"]
        library_roles = selection["library_roles"]
        for rank, col in enumerate(order[: self.top_n_ranked], 1):
            idx = int(library_indices[int(col)])
            is_positive = bool(positive_mask[int(col)])
            rows.append(
                {
                    "query_role": query_role,
                    "query_index": int(query_index),
                    "query_ordinal": int(query_ordinal),
                    "query_usi": str(selection["usis"][query_index]),
                    "query_sequence": str(selection["sequences"][query_index]),
                    "query_unmodified_peptide": str(selection["unmodified"][query_index]),
                    "query_backbone": str(selection["backbone"][query_index]),
                    "query_source_shard": str(selection["source_shards"][query_index]),
                    "query_source_row_in_shard": str(selection["source_rows"][query_index]),
                    "rank": rank,
                    "score": float(scores[int(col)]),
                    "is_positive": is_positive,
                    "library_role": library_roles[int(col)],
                    "library_index": idx,
                    "library_usi": str(selection["usis"][idx]),
                    "library_project": str(selection["projects"][idx]),
                    "library_sequence": str(selection["sequences"][idx]),
                    "library_unmodified_peptide": str(selection["unmodified"][idx]),
                    "library_backbone": str(selection["backbone"][idx]),
                    "clean_edit_distance": self._clean_site_edit_distance(
                        selection["sequences"][query_index],
                        selection["sequences"][idx],
                    ),
                    "library_source_shard": str(selection["source_shards"][idx]),
                    "library_source_row_in_shard": str(selection["source_rows"][idx]),
                }
            )
        return rows

    def _save_artifacts(
        self,
        out_dir: Path,
        S: np.ndarray,
        selection: Dict[str, Any],
        query_metrics: List[Dict[str, Any]],
        ranked_rows: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        paths = {
            "selection_csv": str(out_dir / "rescue_selection.csv"),
            "similarity_npz": str(out_dir / "rescue_similarity_matrix.npz"),
            "ranked_library_csv": str(out_dir / "ranked_library_by_query.csv"),
            "query_metrics_csv": str(out_dir / "query_metrics.csv"),
            "all_pair_scores_csv": str(out_dir / "all_pair_scores.csv"),
        }

        self._write_selection_csv(Path(paths["selection_csv"]), selection)
        np.savez_compressed(
            paths["similarity_npz"],
            similarity_matrix=S.astype(np.float32),
            query_indices=np.array(selection["query_indices"], dtype=np.int64),
            query_roles=np.array(selection["query_roles"], dtype=object),
            library_indices=np.array(selection["library_indices"], dtype=np.int64),
            library_roles=np.array(selection["library_roles"], dtype=object),
        )
        self._write_ranked_csv(Path(paths["ranked_library_csv"]), ranked_rows)
        self._write_query_metrics_csv(Path(paths["query_metrics_csv"]), query_metrics, selection)
        self._write_all_pair_scores_csv(Path(paths["all_pair_scores_csv"]), S, selection)
        return paths

    def _write_selection_csv(self, path: Path, selection: Dict[str, Any]) -> None:
        rows = list(zip(selection["query_roles"], selection["query_indices"]))
        rows.extend(zip(selection["library_roles"], selection["library_indices"]))

        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "role",
                    "row_index",
                    "usi",
                    "project",
                    "sequence",
                    "unmodified_peptide",
                    "backbone",
                    "source_shard",
                    "source_row_in_shard",
                ],
            )
            writer.writeheader()
            for role, idx in rows:
                idx = int(idx)
                writer.writerow(
                    {
                        "role": role,
                        "row_index": idx,
                        "usi": str(selection["usis"][idx]),
                        "project": str(selection["projects"][idx]),
                        "sequence": str(selection["sequences"][idx]),
                        "unmodified_peptide": str(selection["unmodified"][idx]),
                        "backbone": str(selection["backbone"][idx]),
                        "source_shard": str(selection["source_shards"][idx]),
                        "source_row_in_shard": str(selection["source_rows"][idx]),
                    }
                )

    @staticmethod
    def _write_ranked_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _write_query_metrics_csv(
        self,
        path: Path,
        query_metrics: List[Dict[str, Any]],
        selection: Dict[str, Any],
    ) -> None:
        rows = []
        for metrics in query_metrics:
            query_idx = int(metrics["query_index"])
            rows.append(
                {
                    **metrics,
                    "query_usi": str(selection["usis"][query_idx]),
                    "query_sequence": str(selection["sequences"][query_idx]),
                    "query_unmodified_peptide": str(selection["unmodified"][query_idx]),
                    "query_backbone": str(selection["backbone"][query_idx]),
                    "query_source_shard": str(selection["source_shards"][query_idx]),
                    "query_source_row_in_shard": str(selection["source_rows"][query_idx]),
                }
            )
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_all_pair_scores_csv(self, path: Path, S: np.ndarray, selection: Dict[str, Any]) -> None:
        library_roles = np.array(selection["library_roles"], dtype=object)
        rows = []
        for query_ordinal, (query_idx, query_role) in enumerate(
            zip(selection["query_indices"], selection["query_roles"], strict=False)
        ):
            query_idx = int(query_idx)
            order = np.argsort(S[query_ordinal])[::-1]
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(order) + 1)
            for library_ordinal, library_idx in enumerate(selection["library_indices"]):
                library_idx = int(library_idx)
                rows.append(
                    {
                        "query_role": query_role,
                        "query_index": query_idx,
                        "query_ordinal": query_ordinal,
                        "query_usi": str(selection["usis"][query_idx]),
                        "query_sequence": str(selection["sequences"][query_idx]),
                        "query_unmodified_peptide": str(selection["unmodified"][query_idx]),
                        "query_backbone": str(selection["backbone"][query_idx]),
                        "query_source_shard": str(selection["source_shards"][query_idx]),
                        "query_source_row_in_shard": str(selection["source_rows"][query_idx]),
                        "library_role": str(library_roles[library_ordinal]),
                        "library_index": library_idx,
                        "library_ordinal": library_ordinal,
                        "library_usi": str(selection["usis"][library_idx]),
                        "library_project": str(selection["projects"][library_idx]),
                        "library_sequence": str(selection["sequences"][library_idx]),
                        "library_unmodified_peptide": str(selection["unmodified"][library_idx]),
                        "library_backbone": str(selection["backbone"][library_idx]),
                        "is_positive": bool(library_roles[library_ordinal] == self.positive_library_role),
                        "score": float(S[query_ordinal, library_ordinal]),
                        "rank": int(ranks[library_ordinal]),
                        "clean_edit_distance": self._clean_site_edit_distance(
                            selection["sequences"][query_idx],
                            selection["sequences"][library_idx],
                        ),
                        "library_source_shard": str(selection["source_shards"][library_idx]),
                        "library_source_row_in_shard": str(selection["source_rows"][library_idx]),
                    }
                )

        if not rows:
            return
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _collect_similarity_distance_arrays(
        self,
        S: np.ndarray,
        selection: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return cosine scores, clean edit distances, and positive flags for all query-library pairs."""
        sequences = selection["sequences"]
        library_roles = np.array(selection["library_roles"], dtype=object)
        edit_distances: List[float] = []
        cos_sims: List[float] = []
        is_positive: List[bool] = []

        for query_ordinal, query_idx in enumerate(selection["query_indices"]):
            query_idx = int(query_idx)
            query_sequence = sequences[query_idx]
            for library_ordinal, library_idx in enumerate(selection["library_indices"]):
                library_idx = int(library_idx)
                edit_distances.append(
                    float(self._clean_site_edit_distance(query_sequence, sequences[library_idx]))
                )
                cos_sims.append(float(S[query_ordinal, library_ordinal]))
                is_positive.append(bool(library_roles[library_ordinal] == self.positive_library_role))

        return (
            np.asarray(edit_distances, dtype=np.float64),
            np.asarray(cos_sims, dtype=np.float64),
            np.asarray(is_positive, dtype=bool),
        )

    def _save_distance_similarity_plot(
        self,
        save_path: Path,
        edit_distances: np.ndarray,
        cos_sims: np.ndarray,
        is_positive: np.ndarray,
    ) -> None:
        """Save edit-distance vs cosine hexbin with marginal histograms."""
        hist_color = "#9A94A8"

        fig = plt.figure(figsize=(10, 8.5))
        gs = gridspec.GridSpec(
            2,
            3,
            figure=fig,
            width_ratios=[4, 0.28, 0.07],
            height_ratios=[0.28, 4],
            wspace=0.06,
            hspace=0.08,
        )
        ax_top = fig.add_subplot(gs[0, :2])
        ax_main = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        cax = fig.add_subplot(gs[1, 2])

        ax_top.hist(
            edit_distances,
            bins=35,
            color=hist_color,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.3,
        )
        ax_right.hist(
            cos_sims,
            bins=35,
            orientation="horizontal",
            color=hist_color,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.3,
        )
        plt.setp(ax_top.get_xticklabels(), visible=False)
        plt.setp(ax_right.get_yticklabels(), visible=False)
        ax_top.tick_params(axis="x", length=0)
        ax_right.tick_params(axis="y", length=0)

        hb = ax_main.hexbin(
            edit_distances,
            cos_sims,
            gridsize=35,
            mincnt=1,
            cmap="plasma",
        )
        cbar = fig.colorbar(hb, cax=cax)
        cbar.set_label("Pair count")

        unique_distances = np.sort(np.unique(edit_distances))
        mean_sims = np.array([cos_sims[edit_distances == distance].mean() for distance in unique_distances])
        ax_main.plot(unique_distances, mean_sims, color="white", linewidth=2.0, label="Mean similarity")

        positive_mask = is_positive.astype(bool)
        if positive_mask.any():
            ax_main.scatter(
                edit_distances[positive_mask],
                cos_sims[positive_mask],
                s=8,
                c="#4E9AC6",
                alpha=0.35,
                edgecolors="none",
                label=f"Positive library ({positive_mask.sum():,})",
                zorder=3,
            )

        r_s, _ = spearmanr(edit_distances, cos_sims)
        r_p, _ = pearsonr(edit_distances, cos_sims)
        fig.suptitle(
            "Clean Edit Distance vs Embedding Cosine Similarity\n"
            f"(Spearman r={r_s:.3f}, Pearson r={r_p:.3f}, n={len(edit_distances):,} pairs)",
            y=0.98,
        )
        ax_main.set_xlabel("Clean-site edit distance")
        ax_main.set_ylabel("Cosine similarity")
        ax_main.legend(loc="upper right")
        fig.subplots_adjust(top=0.90, right=0.95)
        fig.savefig(save_path, dpi=self.plot_dpi, bbox_inches="tight")
        plt.close(fig)

    def _save_plots(
        self,
        out_dir: Path,
        S: np.ndarray,
        selection: Dict[str, Any],
        ranked_rows: List[Dict[str, Any]],
        query_metrics: List[Dict[str, Any]],
        meta: Dict[str, np.ndarray],
    ) -> Dict[str, str]:
        paths = {
            "retrieval_curves": str(out_dir / "rescue_retrieval_curves.png"),
            "margin_comparison": str(out_dir / "rescue_margin_comparison.png"),
            "distance_similarity_hexbin": str(out_dir / "rescue_distance_similarity_hexbin.png"),
        }
        positive_mask = np.array(selection["library_roles"], dtype=object) == self.positive_library_role

        fig, ax = plt.subplots(figsize=(8, 5))
        role_styles = {
            self.reference_query_role: ("Reference queries", "#4E9AC6"),
            self.modified_query_role: ("Modified queries", "#F5A45D"),
        }
        max_rank = min(250, S.shape[1])
        rank_axis = np.arange(1, max_rank + 1)
        query_roles = selection["query_roles"]
        for role, (label, color) in role_styles.items():
            role_curves: List[np.ndarray] = []
            for row_idx, query_role in enumerate(query_roles):
                if query_role != role:
                    continue
                order = np.argsort(S[row_idx])[::-1]
                curve = S[row_idx, order[:max_rank]]
                role_curves.append(curve)
                ax.plot(rank_axis, curve, color=color, alpha=0.18, linewidth=0.8)
                positive_ranks = [rank for rank, col in enumerate(order[:max_rank], 1) if positive_mask[col]]
                if positive_ranks:
                    ax.scatter(
                        positive_ranks,
                        S[row_idx, order[np.array(positive_ranks) - 1]],
                        s=8,
                        color=color,
                        alpha=0.35,
                        edgecolors="none",
                        zorder=3,
                    )
            if role_curves:
                mean_curve = np.mean(np.vstack(role_curves), axis=0)
                ax.plot(rank_axis, mean_curve, label=f"{label} mean", color=color, linewidth=2.2)
        ax.set_xlabel("Library rank")
        ax.set_ylabel("Embedding cosine similarity")
        ax.set_title("Controlled Rescue Retrieval")
        ax.legend()
        fig.tight_layout()
        fig.savefig(paths["retrieval_curves"], dpi=self.plot_dpi, bbox_inches="tight")
        plt.close(fig)

        labels = ["Reference queries", "Modified queries"]
        margin_values = [
            [row["margin"] for row in query_metrics if row["query_role"] == self.reference_query_role],
            [row["margin"] for row in query_metrics if row["query_role"] == self.modified_query_role],
        ]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(margin_values, labels=labels, patch_artist=True)
        for x_pos, values, color in [(1, margin_values[0], "#4E9AC6"), (2, margin_values[1], "#F5A45D")]:
            ax.scatter(
                np.full(len(values), x_pos, dtype=float),
                values,
                s=18,
                color=color,
                alpha=0.75,
                zorder=3,
            )
        ax.axhline(0, color="#999999", linestyle="--", linewidth=1.0)
        ax.set_ylabel("Best positive - best negative cosine")
        ax.set_title("Retrieval Margin Across Queries")
        fig.tight_layout()
        fig.savefig(paths["margin_comparison"], dpi=self.plot_dpi, bbox_inches="tight")
        plt.close(fig)

        edit_distances, cos_sims, is_positive = self._collect_similarity_distance_arrays(S, selection)
        self._save_distance_similarity_plot(
            Path(paths["distance_similarity_hexbin"]),
            edit_distances,
            cos_sims,
            is_positive,
        )

        publication_paths = save_rescue_publication_plots(
            out_dir,
            S,
            selection,
            query_metrics,
            positive_library_role=self.positive_library_role,
            negative_library_role=self.negative_library_role,
            modified_query_role=self.modified_query_role,
            min_negative_clean_edit_distance=self.min_negative_clean_edit_distance,
            clean_edit_distance_fn=self._clean_site_edit_distance,
            sample_seed=self.sample_seed,
            plot_dpi=self.plot_dpi,
            plot_max_pair_scores=self.plot_max_pair_scores,
            meta=meta,
        )
        paths.update(publication_paths)

        return paths
