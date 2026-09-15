"""Study sequence similarity vs embedding similarity on a validation split.

Two analysis modes:

* **trusted** — same forward path as ``embedding_io`` (``encode_mean_pooled``);
  cosine via ``SpectralAnnotationTransferTask`` normalisation helpers.
* **independent** — direct ``model.encode_mean_pooled`` + explicit NumPy
  re-normalisation; standalone Levenshtein + dot products.

Memory model (full validation set friendly):

* **Single parquet-chunk pass** — one shard at a time through the DataLoader;
  never materialises the full validation split in RAM (avoids Arrow OOM).
* Embeddings stream to a float32 ``.memmap`` file (never hold all embeddings in RAM).
* **Pair metrics** — fixed-size random samples + online (Welford) stats for
  same-/diff-peptide cosines; reservoir samples for plots.
* **Retrieval sanity** — chunked memmap scans per replicate query (no FAISS,
  no n×n matrix, no duplicate embedding index in RAM).

Outputs (under ``evaluation.output_dir/embedding_sequence_study_<mode>/``):

* ``metrics.json`` — sequence metrics first, then embedding metrics
* ``figures/`` — histograms and edit-distance profiles
* ``artifacts/embeddings.memmap`` — full embedding matrix on disk
* ``artifacts/sampled_pairs.npz`` — audit sample of pair-level values
* S3 upload when running on AIchor

Usage::

    uv run python -m instanovo_fm.eval.embedding_sequence_similarity_study \\
        --config-name foundational_eval_embedding_similarity_study \\
        study_mode=trusted

    uv run python -m instanovo_fm.eval.embedding_sequence_similarity_study \\
        --config-name foundational_eval_embedding_similarity_study \\
        study_mode=independent
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from instanovo.__init__ import console
from instanovo_fm.eval.evaluator import EmbeddingEvaluator
from instanovo.utils.colorlogging import ColorLog
from instanovo.utils.s3 import S3FileHandler

logger = ColorLog(console, __name__).logger

CONFIG_PATH = Path(__file__).parent.parent.parent / "configs"


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def _levenshtein(t1: List[str], t2: List[str]) -> int:
    if t1 == t2:
        return 0
    n, m = len(t1), len(t2)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if t1[i - 1] == t2[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


import re as _re

def _tokenize_peptide(seq: str) -> List[str]:
    """Tokenize a peptide into amino acids and modifications."""
    return _re.findall(r"\[[^\]]*\]|[A-Z]", seq)


def _canonicalize_peptide(seq: str) -> str:
    """Canonicalize a peptide for edit distance comparison.

    1. Keep PTMs intact (e.g. [UNIMOD:35], [+16])
    2. Replace I → L (isobaric equivalence)
    """
    tokens = _tokenize_peptide(seq)
    return "".join("L" if t == "I" else t for t in tokens)


def _normalized_sequence_similarity(s1: str, s2: str) -> Tuple[int, float]:
    """Compute edit distance and normalized sequence similarity.

    Sequences are canonicalized (I→L) and tokenized before comparison,
    so that modifications count as exactly 1 edit distance unit.
    """
    c1 = _canonicalize_peptide(s1)
    c2 = _canonicalize_peptide(s2)
    t1 = _tokenize_peptide(c1)
    t2 = _tokenize_peptide(c2)
    ed = _levenshtein(t1, t2)
    max_len = max(len(t1), len(t2))
    seq_sim = 1.0 - (ed / max_len) if max_len > 0 else 1.0
    return ed, float(seq_sim)


def _l2_normalize_rows(emb: np.ndarray) -> np.ndarray:
    out = emb.astype(np.float32, copy=True)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    out /= norms
    return out


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _make_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj


def _summary_stats(arr: np.ndarray) -> Dict[str, Any]:
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }


class RunningStats:
    """Online mean / variance (Welford) with min / max tracking."""

    __slots__ = ("n", "mean", "M2", "min_v", "max_v")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_v = float("inf")
        self.max_v = float("-inf")

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        self.min_v = min(self.min_v, x)
        self.max_v = max(self.max_v, x)

    def to_dict(self) -> Dict[str, Any]:
        if self.n == 0:
            return {"count": 0}
        var = self.M2 / self.n if self.n > 0 else 0.0
        std = float(np.sqrt(max(var, 0.0)))
        return {
            "count": self.n,
            "mean": float(self.mean),
            "std": std,
            "min": float(self.min_v),
            "max": float(self.max_v),
        }


class ReservoirSampler:
    """Fixed-size reservoir sample for plot-friendly subsets."""

    def __init__(self, capacity: int, rng: np.random.RandomState) -> None:
        self.capacity = capacity
        self.rng = rng
        self.values: List[float] = []
        self.seen = 0

    def add(self, x: float) -> None:
        self.seen += 1
        if len(self.values) < self.capacity:
            self.values.append(float(x))
            return
        j = int(self.rng.randint(0, self.seen))
        if j < self.capacity:
            self.values[j] = float(x)

    def to_array(self) -> np.ndarray:
        return np.array(self.values, dtype=np.float32)


def _upload_to_s3(output_dir: Path) -> None:
    if not S3FileHandler._aichor_enabled():
        logger.info("Not on AIchor — skipping S3 upload")
        return
    s3 = S3FileHandler()
    files = [f for f in output_dir.rglob("*") if f.is_file()]
    logger.info("Uploading %d files to S3...", len(files))
    for f in files:
        s3_path = S3FileHandler.convert_to_s3_output(str(f))
        s3.upload(str(f), s3_path)
        logger.info("Uploaded %s → %s", f.name, s3_path)


# ---------------------------------------------------------------------------
# Streaming embedding store (disk-backed)
# ---------------------------------------------------------------------------


class StreamingEmbeddingStore:
    """Append-only embedding store backed by numpy memmap."""

    def __init__(self, path: Path, capacity: int, dim: int) -> None:
        self.path = path
        self.capacity = capacity
        self.dim = dim
        self.offset = 0
        self.peptides: List[str] = []
        self.peptide_to_indices: Dict[str, List[int]] = defaultdict(list)
        self._mmap = np.memmap(path, dtype=np.float32, mode="w+", shape=(capacity, dim))

    def append_batch(self, emb: np.ndarray, batch_peptides: List[str]) -> None:
        bs = int(emb.shape[0])
        if self.offset + bs > self.capacity:
            raise RuntimeError(f"Embedding store overflow: offset={self.offset} batch={bs} capacity={self.capacity}")
        self._mmap[self.offset : self.offset + bs] = emb
        for i, pep in enumerate(batch_peptides):
            idx = self.offset + i
            raw = str(pep).strip()
            self.peptides.append(raw)
            # Group by canonicalized peptide (PTMs stripped, I→L) so that
            # replicate detection treats isobaric variants as identical.
            canon = _canonicalize_peptide(raw)
            self.peptide_to_indices[canon].append(idx)
        self.offset += bs

    def flush(self) -> None:
        self._mmap.flush()

    def finalize(self) -> None:
        """Trim memmap to actual row count and flush."""
        self.flush()
        if self.offset < self.capacity:
            trimmed = np.memmap(
                self.path,
                dtype=np.float32,
                mode="r+",
                shape=(self.offset, self.dim),
            )
            del self._mmap
            self._mmap = trimmed
            self.capacity = self.offset

    @property
    def n(self) -> int:
        return self.offset

    def row(self, i: int) -> np.ndarray:
        return np.asarray(self._mmap[i], dtype=np.float32)

    def rows(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray(self._mmap[indices], dtype=np.float32)

    def peptides_array(self) -> np.ndarray:
        return np.array(self.peptides, dtype=object)

    def embedding_norm_stats(self) -> Dict[str, Any]:
        """Compute norm stats in chunks without loading full matrix."""
        stats = RunningStats()
        chunk = 4096
        for start in range(0, self.n, chunk):
            block = np.asarray(self._mmap[start : start + chunk], dtype=np.float32)
            norms = np.linalg.norm(block, axis=1)
            for v in norms:
                stats.update(float(v))
        return stats.to_dict()


def _load_validation_sdf(config: DictConfig, split: str):
    """Lazy SpectrumDataFrame for the split (no in-memory collect)."""
    from instanovo_fm.utils.spectrum_dataframe import SpectrumDataFrame

    dataset_config = config.get("dataset", {})
    split_key = f"{split}_path"
    if split_key not in dataset_config:
        raise ValueError(
            f"Dataset split '{split}' not found in config. "
            f"Available keys: {list(dataset_config.keys())}"
        )
    return SpectrumDataFrame.load(
        source=dataset_config[split_key],
        source_type=dataset_config.get("source_type", "default"),
        lazy=dataset_config.get("lazy_loading", True),
        is_annotated=True,
        shuffle=False,
        partition=None,
        column_mapping=dataset_config.get("column_remapping", None),
        max_shard_size=dataset_config.get("max_shard_size", 100_000),
        add_source_file_column=True,
        preshuffle_across_shards=False,
        verbose=dataset_config.get("verbose_loading", True),
    )


def _iter_validation_parquet_chunks(
    sdf,
    max_samples: Optional[int],
) -> Iterator[Any]:
    """Yield one parquet shard at a time (same filtering as collect_chunked, no concat)."""
    import polars as pl
    from instanovo_fm.utils.spectrum_dataframe import SpectrumDataFrame

    if not sdf._is_native:
        df = sdf.df
        if max_samples is not None and df is not None and len(df) > max_samples:
            yield df.head(max_samples)
        elif df is not None:
            yield df
        return

    n_collected = 0
    for fp in sorted(sdf._file_paths):
        if max_samples is not None and n_collected >= max_samples:
            break

        mask = sdf._filter_series_per_file[fp]
        if not mask.any():
            continue

        chunk = pl.scan_parquet(fp).filter(mask).collect()
        chunk = SpectrumDataFrame._ensure_experiment_name(
            chunk, fp, add_source=sdf._add_source_file_column, force_source=True
        )

        if max_samples is not None:
            remaining = max_samples - n_collected
            if len(chunk) > remaining:
                chunk = chunk.head(remaining)

        if len(chunk) == 0:
            continue

        n_collected += len(chunk)
        yield chunk


def _chunk_dataloader(
    evaluator: EmbeddingEvaluator,
    pl_chunk: Any,
    prediction_id_start: int,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader for a single in-memory parquet chunk."""
    from datasets import Dataset as HFDataset, Value

    dataset = HFDataset.from_pandas(pl_chunk.to_pandas())
    n = len(dataset)
    dataset = dataset.add_column(
        "prediction_id",
        np.arange(prediction_id_start, prediction_id_start + n, dtype=np.int32),
        feature=Value("int32"),
    )
    evaluator.data_processor._keep_non_tensor_metadata = True
    processed = evaluator.data_processor.process_dataset(dataset, return_format="torch")
    batch_size = evaluator.eval_config.get("batch_size", 128)
    num_workers = evaluator.config.get("num_workers", 4)
    return torch.utils.data.DataLoader(
        processed,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=evaluator.data_processor.collate_fn,
    )


def _resolve_n_spectra_from_sdf(sdf, max_samples: Optional[int]) -> int:
    n = len(sdf)
    if max_samples is not None:
        n = min(n, int(max_samples))
    return n


def _batch_peptides(batch: Dict[str, Any], peptide_key: str) -> List[str]:
    if peptide_key not in batch:
        raise KeyError(f"'{peptide_key}' not in batch keys: {list(batch.keys())}")
    raw = batch[peptide_key]
    if isinstance(raw, torch.Tensor):
        return [str(x).strip() for x in raw.tolist()]
    if isinstance(raw, list):
        return [str(x).strip() for x in raw]
    return [str(x).strip() for x in raw]


def stream_embeddings_from_chunks(
    evaluator: EmbeddingEvaluator,
    store: StreamingEmbeddingStore,
    chunk_iterator: Iterator[Any],
    *,
    peptide_key: str,
    mode: str,
    pooling: str,
    max_samples: Optional[int],
) -> None:
    """Stream model inference over parquet chunks → memmap (+ peptide index)."""
    model = evaluator.model
    device = evaluator.device
    n_streamed = 0
    prediction_id_start = 0

    for chunk_idx, pl_chunk in enumerate(chunk_iterator):
        if max_samples is not None and n_streamed >= max_samples:
            break

        chunk_len = len(pl_chunk)
        if chunk_len == 0:
            continue

        logger.info(
            "Processing parquet chunk %d (%d rows, %d streamed so far)",
            chunk_idx,
            chunk_len,
            n_streamed,
        )
        dataloader = _chunk_dataloader(evaluator, pl_chunk, prediction_id_start)
        prediction_id_start += chunk_len

        with torch.inference_mode():
            for batch in dataloader:
                if max_samples is not None and n_streamed >= max_samples:
                    break

                spectra = batch["spectra"].to(device)
                meta = batch.get("meta", {})
                batch_peps = _batch_peptides(batch, peptide_key)

                if pooling in ("mean_pool", "confidence"):
                    emb = model.encode_mean_pooled(
                        spectra,
                        meta=meta,
                        pooling=pooling,
                        confidence_temperature=evaluator.eval_config.get("confidence_temperature", 1.0),
                    )
                else:
                    emb = model.encode(spectra, meta=meta)

                emb_np = emb.detach().cpu().numpy().astype(np.float32)
                if mode == "independent":
                    emb_np = _l2_normalize_rows(emb_np)

                if max_samples is not None and n_streamed + emb_np.shape[0] > max_samples:
                    emb_np = emb_np[: max_samples - n_streamed]
                    batch_peps = batch_peps[: emb_np.shape[0]]

                store.append_batch(emb_np, batch_peps)
                n_streamed += emb_np.shape[0]

                if n_streamed % 10_000 < emb_np.shape[0]:
                    store.flush()
                    logger.info("Streamed %d / %d embeddings", n_streamed, store.capacity)

        del dataloader

    store.finalize()
    logger.info(
        "Embedding stream complete: n=%d dim=%d mode=%s",
        store.n,
        store.dim,
        mode,
    )


# ---------------------------------------------------------------------------
# Sequence metrics (phase 1 — peptides only, no embedding RAM)
# ---------------------------------------------------------------------------


def compute_sequence_metrics(
    peptides: np.ndarray,
    rng: np.random.RandomState,
    n_random_pairs: int,
    max_edit_distance_plot: int,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    n = len(peptides)
    # Canonicalize for replicate counting (PTMs stripped, I→L)
    canon = np.array([_canonicalize_peptide(p) for p in peptides], dtype=object)
    unique = len(set(canon.tolist()))
    counts = Counter(canon.tolist())
    dup_peptides = sum(1 for c in counts.values() if c > 1)
    dup_spectra = sum(c for c in counts.values() if c > 1)

    multiplicity = Counter(counts.values())
    mult_dist = {int(k): int(v) for k, v in sorted(multiplicity.items())}

    d0_count = sum(cnt * (cnt - 1) // 2 for cnt in counts.values() if cnt > 1)

    n_sample = min(n_random_pairs, max(1, n * max(n - 1, 1)))
    rows = rng.randint(0, n, size=n_sample)
    cols = rng.randint(0, n, size=n_sample)
    rand_eds: List[int] = []
    rand_seq_sims: List[float] = []
    for i, j in zip(rows, cols, strict=False):
        if i == j:
            continue
        ed, seq_sim = _normalized_sequence_similarity(peptides[i], peptides[j])
        rand_eds.append(ed)
        rand_seq_sims.append(seq_sim)

    rand_eds_arr = np.array(rand_eds, dtype=np.int32)
    rand_seq_arr = np.array(rand_seq_sims, dtype=np.float32)

    ed_bins = defaultdict(list)
    for ed, ss in zip(rand_eds, rand_seq_sims, strict=False):
        ed_bins[int(ed)].append(ss)
    seq_sim_by_ed = {
        str(ed): {"count": len(vals), "mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for ed, vals in sorted(ed_bins.items())
        if ed <= max_edit_distance_plot
    }

    metrics: Dict[str, Any] = {
        "num_spectra": n,
        "num_unique_peptides": unique,
        "duplicate_peptide_count": dup_peptides,
        "duplicate_spectra_count": dup_spectra,
        "singleton_peptide_fraction": float((n - dup_spectra) / n) if n else 0.0,
        "spectra_per_peptide_distribution": mult_dist,
        "same_peptide_pair_count": d0_count,
        "random_pair_count": len(rand_eds),
        "same_peptide_sequence_similarity": {"count": d0_count, "mean": 1.0, "std": 0.0},
        "random_pair_sequence_similarity": _summary_stats(rand_seq_arr),
        "random_pair_edit_distance": _summary_stats(rand_eds_arr.astype(np.float32)),
        "sequence_similarity_by_edit_distance": seq_sim_by_ed,
    }

    arrays = {
        "rand_edit_distances": rand_eds_arr,
        "rand_sequence_similarities": rand_seq_arr,
    }
    return metrics, arrays


# ---------------------------------------------------------------------------
# Embedding pair metrics (memmap-backed, phase 2)
# ---------------------------------------------------------------------------


def sample_embedding_pairs_memmap(
    store: StreamingEmbeddingStore,
    rng: np.random.RandomState,
    n_random_pairs: int,
    cosine_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    max_same_pairs_eval: int,
    plot_reservoir_size: int,
) -> Dict[str, Any]:
    """Compute embedding pair statistics with batch reads from the memmap.

    Random pairs and same/different-peptide pairs are evaluated using bulk
    memmap reads (``store.rows(indices)``) rather than one-at-a-time reads to
    avoid per-row I/O overhead on large stores.
    """
    peptides = store.peptides_array()
    n = store.n

    n_sample = min(n_random_pairs, max(1, n * max(n - 1, 1)))
    rows = rng.randint(0, n, size=n_sample)
    cols = rng.randint(0, n, size=n_sample)

    # Remove self-pairs
    keep = rows != cols
    rows = rows[keep]
    cols = cols[keep]

    # Batch-read embeddings and compute cosines in bulk
    batch_sz = 10_000
    rand_cos: List[float] = []
    rand_eds: List[int] = []
    rand_seq: List[float] = []
    for start in range(0, len(rows), batch_sz):
        end = min(start + batch_sz, len(rows))
        r_batch = rows[start:end]
        c_batch = cols[start:end]
        emb_r = _l2_normalize_rows(store.rows(r_batch))
        emb_c = _l2_normalize_rows(store.rows(c_batch))
        cos_batch = np.sum(emb_r * emb_c, axis=1)
        rand_cos.extend(cos_batch.tolist())
        for i, j in zip(r_batch, c_batch, strict=False):
            ed, seq_sim = _normalized_sequence_similarity(peptides[i], peptides[j])
            rand_eds.append(ed)
            rand_seq.append(seq_sim)

    logger.info("  Random pair cosines computed: %d pairs", len(rand_cos))

    # Same-peptide pairs — batch per peptide group
    same_stats = RunningStats()
    same_reservoir = ReservoirSampler(plot_reservoir_size, rng)
    same_pairs_evaluated = 0

    for pep, idxs in store.peptide_to_indices.items():
        if len(idxs) < 2 or same_pairs_evaluated >= max_same_pairs_eval:
            break
        idxs_arr = np.array(idxs, dtype=np.int64)
        group_emb = _l2_normalize_rows(store.rows(idxs_arr))
        # Compute pairwise cosines within group: upper triangle
        # For groups up to ~2000 this is cheap; for larger groups, cap pairs
        group_n = len(idxs_arr)
        if group_n <= 200:
            # Full pairwise via matmul
            sim_matrix = group_emb @ group_emb.T
            for a in range(group_n):
                for b in range(a + 1, group_n):
                    if same_pairs_evaluated >= max_same_pairs_eval:
                        break
                    cos = float(sim_matrix[a, b])
                    same_stats.update(cos)
                    same_reservoir.add(cos)
                    same_pairs_evaluated += 1
                if same_pairs_evaluated >= max_same_pairs_eval:
                    break
        else:
            # Sample pairs within the group
            max_from_group = min(group_n * (group_n - 1) // 2, max_same_pairs_eval - same_pairs_evaluated)
            n_pairs = min(max_from_group, 5000)
            a_idx = rng.randint(0, group_n, size=n_pairs * 2)
            b_idx = rng.randint(0, group_n, size=n_pairs * 2)
            seen = 0
            for a, b in zip(a_idx, b_idx, strict=False):
                if a == b or same_pairs_evaluated >= max_same_pairs_eval:
                    continue
                cos = float(np.dot(group_emb[a], group_emb[b]))
                same_stats.update(cos)
                same_reservoir.add(cos)
                same_pairs_evaluated += 1
                seen += 1
                if seen >= n_pairs:
                    break

    if same_pairs_evaluated >= max_same_pairs_eval:
        logger.warning("Capped same-peptide pair evaluation at %d pairs", max_same_pairs_eval)
    logger.info("  Same-peptide pairs evaluated: %d", same_pairs_evaluated)

    # Different-peptide pairs — batch sample
    diff_stats = RunningStats()
    diff_reservoir = ReservoirSampler(plot_reservoir_size, rng)
    diff_pairs_evaluated = 0
    diff_target = min(n_random_pairs, max(1, n * max(n - 1, 1)))

    # Pre-compute canonical peptides for same/different classification
    canon_peptides = np.array([_canonicalize_peptide(p) for p in peptides], dtype=object)

    diff_rows = rng.randint(0, n, size=diff_target * 2)
    diff_cols = rng.randint(0, n, size=diff_target * 2)
    diff_keep = (diff_rows != diff_cols) & np.array(
        [canon_peptides[i] != canon_peptides[j] for i, j in zip(diff_rows, diff_cols, strict=False)]
    )
    diff_rows = diff_rows[diff_keep][:diff_target]
    diff_cols = diff_cols[diff_keep][:diff_target]

    for start in range(0, len(diff_rows), batch_sz):
        end = min(start + batch_sz, len(diff_rows))
        r_batch = diff_rows[start:end]
        c_batch = diff_cols[start:end]
        emb_r = _l2_normalize_rows(store.rows(r_batch))
        emb_c = _l2_normalize_rows(store.rows(c_batch))
        cos_batch = np.sum(emb_r * emb_c, axis=1)
        for cos in cos_batch:
            diff_stats.update(float(cos))
            diff_reservoir.add(float(cos))
            diff_pairs_evaluated += 1

    logger.info("  Different-peptide pairs evaluated: %d", diff_pairs_evaluated)

    return {
        "random_cosine": np.array(rand_cos, dtype=np.float32),
        "random_edit_distance": np.array(rand_eds, dtype=np.int32),
        "random_sequence_similarity": np.array(rand_seq, dtype=np.float32),
        "same_peptide_cosine_sample": same_reservoir.to_array(),
        "different_peptide_cosine_sample": diff_reservoir.to_array(),
        "same_peptide_cosine_stats": same_stats.to_dict(),
        "different_peptide_cosine_stats": diff_stats.to_dict(),
        "same_peptide_pairs_evaluated": same_pairs_evaluated,
        "different_peptide_pairs_evaluated": diff_pairs_evaluated,
    }


def compute_embedding_metrics(
    pair_data: Dict[str, Any],
    max_edit_distance_plot: int,
) -> Dict[str, Any]:
    rand_cos = pair_data["random_cosine"]
    rand_ed = pair_data["random_edit_distance"]
    rand_seq = pair_data["random_sequence_similarity"]
    same_stats = pair_data["same_peptide_cosine_stats"]
    diff_stats = pair_data["different_peptide_cosine_stats"]

    ed_bins = defaultdict(list)
    for ed, cos in zip(rand_ed, rand_cos, strict=False):
        if ed <= max_edit_distance_plot:
            ed_bins[int(ed)].append(float(cos))
    cosine_by_ed = {str(ed): {"count": len(vals), "mean": float(np.mean(vals)), "std": float(np.std(vals))} for ed, vals in sorted(ed_bins.items())}

    margin = None
    if same_stats.get("count", 0) and diff_stats.get("count", 0):
        margin = float(same_stats["mean"] - diff_stats["mean"])

    spearman_r = pearson_r = None
    if len(rand_cos) >= 3:
        try:
            from scipy.stats import pearsonr, spearmanr

            spearman_r, _ = spearmanr(rand_cos, rand_seq)
            pearson_r, _ = pearsonr(rand_cos, rand_seq)
            spearman_r = float(spearman_r)
            pearson_r = float(pearson_r)
        except ImportError:
            logger.warning("scipy not available — skipping correlation metrics")

    return {
        "random_pair_cosine": _summary_stats(rand_cos),
        "same_peptide_cosine": same_stats,
        "different_peptide_cosine": diff_stats,
        "same_peptide_pairs_evaluated": pair_data["same_peptide_pairs_evaluated"],
        "different_peptide_pairs_evaluated": pair_data["different_peptide_pairs_evaluated"],
        "same_minus_different_cosine_margin": margin,
        "cosine_similarity_by_edit_distance": cosine_by_ed,
        "sequence_vs_embedding_spearman": spearman_r,
        "sequence_vs_embedding_pearson": pearson_r,
        "fraction_random_cosine_above_0.9": float((rand_cos > 0.9).mean()) if len(rand_cos) else None,
        "fraction_random_cosine_above_0.95": float((rand_cos > 0.95).mean()) if len(rand_cos) else None,
    }


def compute_retrieval_chunked(
    store: StreamingEmbeddingStore,
    k_values: List[int],
    *,
    scan_chunk_size: int,
    max_retrieval_queries: int = 50_000,
) -> Dict[str, Any]:
    """Recall@k via deduplicated batched matmul (one query per unique peptide).

    Instead of scanning the full memmap per-query for every replicate spectrum
    (O(n_replicates × n) which is intractable for large n), this:
      1. Deduplicates queries: one representative per unique peptide.
      2. Scans the library in chunks, computing a batch matmul per chunk.
      3. Maintains a top-k heap per query across chunks.

    With M unique peptides and N total spectra, the cost is O(M × N / chunk)
    matmul operations — typically seconds to minutes instead of days.
    """
    n = store.n
    peptides = store.peptides_array()
    # Canonicalize for replicate grouping (PTMs stripped, I→L)
    canon_peptides = np.array([_canonicalize_peptide(p) for p in peptides], dtype=object)
    counts = Counter(canon_peptides.tolist())

    # Deduplicate: one representative per unique canonicalized peptide that has replicates
    seen: Dict[str, int] = {}
    for i, cpep in enumerate(canon_peptides):
        if counts[cpep] > 1 and cpep not in seen:
            seen[cpep] = i
    query_indices = sorted(seen.values())

    if not query_indices:
        return {"num_valid_queries": 0}

    # Cap queries if too many unique peptides
    if len(query_indices) > max_retrieval_queries:
        rng = np.random.RandomState(42)
        query_indices = sorted(
            rng.choice(query_indices, max_retrieval_queries, replace=False).tolist()
        )

    M = len(query_indices)
    max_k = max(k_values)
    logger.info(
        "Retrieval: %d deduplicated queries, %d library spectra, top-%d",
        M, n, max_k,
    )

    # Load and normalise query embeddings (M × D, fits in RAM)
    q_idx_arr = np.array(query_indices, dtype=np.int64)
    E_q = _l2_normalize_rows(store.rows(q_idx_arr))  # (M, D)

    # Maintain top-k scores and indices per query
    top_scores = np.full((M, max_k), -np.inf, dtype=np.float32)
    top_indices = np.full((M, max_k), -1, dtype=np.int64)

    # Scan library in chunks
    for start in range(0, n, scan_chunk_size):
        end = min(start + scan_chunk_size, n)
        block = _l2_normalize_rows(
            np.asarray(store._mmap[start:end], dtype=np.float32)
        )  # (chunk, D)

        # Batched matmul: (M, D) @ (D, chunk) → (M, chunk)
        sims = E_q @ block.T  # (M, chunk)

        # Mask self-matches
        for qi, orig_idx in enumerate(query_indices):
            if start <= orig_idx < end:
                sims[qi, orig_idx - start] = -np.inf

        # Merge with running top-k
        local_indices = np.arange(start, end, dtype=np.int64)
        combined_scores = np.concatenate([top_scores, sims], axis=1)  # (M, max_k + chunk)
        combined_indices = np.concatenate(
            [top_indices, np.broadcast_to(local_indices[None, :], (M, end - start))],
            axis=1,
        )  # (M, max_k + chunk)

        # Keep top-k per row
        part_idx = np.argpartition(combined_scores, -max_k, axis=1)[:, -max_k:]
        for qi in range(M):
            sel = part_idx[qi]
            top_scores[qi] = combined_scores[qi, sel]
            top_indices[qi] = combined_indices[qi, sel]

        if (start // scan_chunk_size) % 20 == 0:
            logger.info(
                "  Retrieval chunk %d / %d",
                start // scan_chunk_size + 1,
                (n + scan_chunk_size - 1) // scan_chunk_size,
            )

    # Compute hits using canonicalized peptide lookup
    hits = dict.fromkeys(k_values, 0)
    for qi, orig_idx in enumerate(query_indices):
        cpep = canon_peptides[orig_idx]
        rel = set(store.peptide_to_indices[cpep]) - {orig_idx}
        # Sort this query's top-k by score descending
        order = np.argsort(top_scores[qi])[::-1]
        retrieved = top_indices[qi, order]
        for k in k_values:
            if any(int(idx) in rel for idx in retrieved[:k]):
                hits[k] += 1

    nq = M
    return {
        "num_valid_queries": nq,
        "num_total_spectra": n,
        **{f"recall@{k}": hits[k] / nq for k in k_values},
    }


def trusted_cosine_fn_factory() -> Callable[[np.ndarray, np.ndarray], float]:
    from instanovo_fm.eval.embed_eval_tasks.spectral_annotation_transfer import (
        SpectralAnnotationTransferTask,
    )

    def fn(a: np.ndarray, b: np.ndarray) -> float:
        ea = SpectralAnnotationTransferTask._l2_normalise(a.reshape(1, -1))[0]
        eb = SpectralAnnotationTransferTask._l2_normalise(b.reshape(1, -1))[0]
        return float(np.dot(ea, eb))

    return fn


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _save_figures(
    output_dir: Path,
    seq_arrays: Dict[str, np.ndarray],
    pair_data: Dict[str, Any],
    seq_metrics: Dict[str, Any],
    emb_metrics: Dict[str, Any],
    mode: str,
    dpi: int,
) -> Dict[str, str]:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(10, 5))
    ed = seq_arrays["rand_edit_distances"]
    ax.hist(ed[ed <= 20], bins=21, color="steelblue", edgecolor="white")
    ax.set_xlabel("Levenshtein edit distance")
    ax.set_ylabel("Pair count")
    ax.set_title("Sequence edit distance (random pairs)")
    fig.tight_layout()
    p = fig_dir / "sequence_edit_distance_histogram.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths["sequence_edit_distance_histogram"] = str(p)

    fig, ax = plt.subplots(figsize=(10, 5))
    seq_by_ed = seq_metrics.get("sequence_similarity_by_edit_distance", {})
    if seq_by_ed:
        xs = [int(k) for k in seq_by_ed]
        ys = [seq_by_ed[str(x)]["mean"] for x in xs]
        ax.plot(xs, ys, "o-", color="steelblue", label="Sequence similarity")
    ax.set_xlabel("Edit distance")
    ax.set_ylabel("Mean normalised sequence similarity")
    ax.set_ylim(0, 1.05)
    ax.set_title("Sequence similarity vs edit distance (dataset reference)")
    ax.legend()
    fig.tight_layout()
    p = fig_dir / "sequence_similarity_by_edit_distance.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths["sequence_similarity_by_edit_distance"] = str(p)

    fig, ax = plt.subplots(figsize=(10, 5))
    cos = pair_data["random_cosine"]
    ax.hist(cos, bins=60, color="darkorange", edgecolor="white")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Pair count")
    ax.set_title(f"Embedding cosine similarity (random pairs) — {mode}")
    fig.tight_layout()
    p = fig_dir / "embedding_cosine_histogram.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths["embedding_cosine_histogram"] = str(p)

    fig, ax = plt.subplots(figsize=(10, 5))
    emb_by_ed = emb_metrics.get("cosine_similarity_by_edit_distance", {})
    if emb_by_ed:
        xs = [int(k) for k in emb_by_ed]
        ys = [emb_by_ed[str(x)]["mean"] for x in xs]
        ax.plot(xs, ys, "o-", color="darkorange", label="Embedding cosine")
    if seq_by_ed:
        xs2 = sorted(set(int(k) for k in seq_by_ed) & set(int(k) for k in emb_by_ed))
        if xs2:
            ys2 = [seq_by_ed[str(x)]["mean"] for x in xs2]
            ax.plot(xs2, ys2, "o--", color="steelblue", alpha=0.7, label="Sequence similarity")
    ax.set_xlabel("Edit distance")
    ax.set_ylabel("Mean similarity")
    ax.set_title("Embedding vs sequence similarity by edit distance")
    ax.legend()
    fig.tight_layout()
    p = fig_dir / "embedding_vs_sequence_by_edit_distance.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths["embedding_vs_sequence_by_edit_distance"] = str(p)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(
        [
            pair_data["same_peptide_cosine_sample"],
            pair_data["different_peptide_cosine_sample"],
        ],
        tick_labels=["Same peptide", "Different peptide"],
        showfliers=False,
    )
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"Same vs different peptide embeddings — {mode}")
    fig.tight_layout()
    p = fig_dir / "embedding_same_vs_different_boxplot.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths["embedding_same_vs_different_boxplot"] = str(p)

    fig, ax = plt.subplots(figsize=(10, 6))
    hb = ax.hexbin(
        pair_data["random_sequence_similarity"],
        pair_data["random_cosine"],
        gridsize=50,
        mincnt=1,
        cmap="viridis",
    )
    fig.colorbar(hb, ax=ax, label="Count")
    ax.set_xlabel("Normalised sequence similarity")
    ax.set_ylabel("Embedding cosine similarity")
    ax.set_title(f"Sequence vs embedding similarity — {mode}")
    fig.tight_layout()
    p = fig_dir / "sequence_vs_embedding_hexbin.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths["sequence_vs_embedding_hexbin"] = str(p)

    fig, ax = plt.subplots(figsize=(10, 5))
    mult = seq_metrics.get("spectra_per_peptide_distribution", {})
    if mult:
        xs = sorted(int(k) for k in mult)
        ys = [mult[str(x)] if str(x) in mult else mult[x] for x in xs]
        ax.bar(xs, ys, color="seagreen", edgecolor="white")
    ax.set_xlabel("Spectra per peptide")
    ax.set_ylabel("Peptide count")
    ax.set_title("Replicate structure in validation set")
    fig.tight_layout()
    p = fig_dir / "replicate_structure.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths["replicate_structure"] = str(p)

    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_study(config: DictConfig) -> Dict[str, Any]:
    study_cfg = config.get("study", {})
    mode = str(config.get("study_mode", "trusted")).lower()
    if mode not in ("trusted", "independent"):
        raise ValueError(f"study_mode must be 'trusted' or 'independent', got {mode!r}")

    peptide_key = study_cfg.get("peptide_key", "peptides")
    sample_seed = int(study_cfg.get("sample_seed", 42))
    n_random_pairs = int(study_cfg.get("n_random_pairs", 500_000))
    max_ed_plot = int(study_cfg.get("max_edit_distance_plot", 15))
    pooling = study_cfg.get("pooling", "mean_pool")
    k_values = list(study_cfg.get("k_values", [1, 5, 10, 20]))
    dpi = int(study_cfg.get("plot_dpi", 150))
    upload = bool(study_cfg.get("upload_to_s3", True))
    faiss_chunk_size = int(study_cfg.get("faiss_chunk_size", 8192))
    scan_chunk_size = int(study_cfg.get("scan_chunk_size", faiss_chunk_size))
    max_same_pairs_eval = int(study_cfg.get("max_same_pairs_eval", 2_000_000))
    plot_reservoir_size = int(study_cfg.get("plot_reservoir_size", 50_000))

    split = config.evaluation.get("split", "valid")
    max_samples = config.evaluation.get("max_samples", None)
    if max_samples is not None:
        max_samples = int(max_samples)

    base_out = Path(config.evaluation.get("output_dir", "./evaluation_results"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_out / f"embedding_sequence_study_{mode}_{timestamp}"
    art_dir = output_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Embedding vs sequence similarity study — mode=%s", mode)
    logger.info("Output: %s", output_dir)
    logger.info("=" * 80)

    evaluator = EmbeddingEvaluator(config)
    evaluator.model, evaluator.model_config = evaluator.load_model()
    evaluator.data_processor = evaluator.setup_data_processor()
    evaluator.data_processor.add_metadata_columns(["prediction_id"])

    sdf = _load_validation_sdf(config, split)
    n_total = _resolve_n_spectra_from_sdf(sdf, max_samples)
    logger.info(
        "Validation set: %d spectra (max_samples=%s, split=%s, chunked parquet load)",
        n_total,
        max_samples,
        split,
    )

    chunk_iter = _iter_validation_parquet_chunks(sdf, max_samples)
    first_chunk = next(chunk_iter, None)
    if first_chunk is None or len(first_chunk) == 0:
        raise RuntimeError(f"No spectra found for split '{split}'")

    # Probe embedding dim from a single-row batch (avoid processing chunk twice)
    probe_dl = _chunk_dataloader(evaluator, first_chunk.head(1), prediction_id_start=0)
    first_batch = next(iter(probe_dl))
    probe_spectra = first_batch["spectra"][:1].to(evaluator.device)
    probe_meta = first_batch.get("meta", {})
    with torch.inference_mode():
        if pooling in ("mean_pool", "confidence"):
            probe_emb = evaluator.model.encode_mean_pooled(
                probe_spectra, meta=probe_meta, pooling=pooling
            )
        else:
            probe_emb = evaluator.model.encode(probe_spectra, meta=probe_meta)
    emb_dim = int(probe_emb.shape[1])
    del probe_dl

    from itertools import chain

    memmap_path = art_dir / "embeddings.memmap"
    store = StreamingEmbeddingStore(memmap_path, capacity=n_total, dim=emb_dim)

    # --- Single pass: stream embeddings to disk (one parquet shard at a time) ---
    logger.info("Streaming embeddings to %s (chunked, no full-split RAM)...", memmap_path)
    stream_embeddings_from_chunks(
        evaluator,
        store,
        chain([first_chunk], chunk_iter),
        peptide_key=peptide_key,
        mode=mode,
        pooling=pooling,
        max_samples=max_samples,
    )

    peptides = store.peptides_array()
    rng = np.random.RandomState(sample_seed)

    # --- Phase 1: sequence metrics (peptides in RAM only) ---
    logger.info("Phase 1: sequence similarity metrics on %d spectra...", len(peptides))
    seq_metrics, seq_arrays = compute_sequence_metrics(peptides, rng, n_random_pairs, max_ed_plot)
    for k, v in seq_metrics["random_pair_sequence_similarity"].items():
        logger.info("  sequence.random_pair.seq_sim.%s = %s", k, v)

    # --- Phase 2: embedding pair metrics (memmap lookups) ---
    logger.info("Phase 2: embedding similarity metrics (memmap-backed)...")
    cosine_fn = trusted_cosine_fn_factory() if mode == "trusted" else _cosine_rows
    pair_data = sample_embedding_pairs_memmap(
        store,
        rng,
        n_random_pairs,
        cosine_fn,
        max_same_pairs_eval=max_same_pairs_eval,
        plot_reservoir_size=plot_reservoir_size,
    )
    emb_metrics = compute_embedding_metrics(pair_data, max_ed_plot)
    emb_metrics["embedding_shape"] = [store.n, store.dim]
    emb_metrics["embedding_norm"] = store.embedding_norm_stats()
    emb_metrics["retrieval_sanity"] = compute_retrieval_chunked(store, k_values, scan_chunk_size=scan_chunk_size)

    logger.info("Saving figures...")
    figure_paths = _save_figures(output_dir, seq_arrays, pair_data, seq_metrics, emb_metrics, mode, dpi)

    np.savez_compressed(
        art_dir / "sampled_pairs.npz",
        random_cosine=pair_data["random_cosine"],
        random_edit_distance=pair_data["random_edit_distance"],
        random_sequence_similarity=pair_data["random_sequence_similarity"],
        same_peptide_cosine_sample=pair_data["same_peptide_cosine_sample"],
        different_peptide_cosine_sample=pair_data["different_peptide_cosine_sample"],
    )

    results: Dict[str, Any] = {
        "study_mode": mode,
        "timestamp": timestamp,
        "split": split,
        "checkpoint_path": config.evaluation.get("checkpoint_path"),
        "max_samples": max_samples,
        "num_spectra": store.n,
        "pooling": pooling,
        "peptide_key": peptide_key,
        "n_random_pairs": n_random_pairs,
        "memory_strategy": "chunked_parquet_single_pass_memmap",
        "embeddings_memmap": str(memmap_path),
        "sequence_metrics": seq_metrics,
        "embedding_metrics": emb_metrics,
        "figure_paths": figure_paths,
        "output_dir": str(output_dir),
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(_make_serializable(results), f, indent=2)
    logger.info("Wrote %s", metrics_path)

    logger.info("--- Embedding (random pairs) ---")
    for k, v in emb_metrics["random_pair_cosine"].items():
        logger.info("  cos_sim.%s = %s", k, v)
    logger.info("  same_minus_diff_margin = %s", emb_metrics.get("same_minus_different_cosine_margin"))
    logger.info("  seq_vs_emb_spearman = %s", emb_metrics.get("sequence_vs_embedding_spearman"))
    for k, v in emb_metrics.get("retrieval_sanity", {}).items():
        logger.info("  retrieval.%s = %s", k, v)

    if upload:
        _upload_to_s3(output_dir)

    return results


@hydra.main(
    config_path=str(CONFIG_PATH),
    version_base=None,
    config_name="foundational_eval_embedding_similarity_study",
)
def main(config: DictConfig) -> None:
    logging.getLogger("instanovo_fm.eval.embedding_io").setLevel(logging.INFO)
    # resolve=False — model.tags use ${architecture.*} interpolations that only
    # resolve inside the model config group, not at the merged root.
    logger.info("Config:\n%s", OmegaConf.to_yaml(config, resolve=False))
    run_study(config)


if __name__ == "__main__":
    main()
