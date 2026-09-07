"""Shared pipeline helpers for the baseline embedding extractors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from datasets.utils.logging import disable_progress_bar

from instanovo.__init__ import console
from instanovo_fm.data.data import FoundationalDataProcessor
from instanovo_fm.eval._extract_metadata_common import finalize_metadata
from instanovo_fm.eval.embedding_io import save
from instanovo.utils.colorlogging import ColorLog
from instanovo.utils.s3 import S3FileHandler

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

logger = ColorLog(console, __name__).logger
disable_progress_bar()


def upload_to_s3(output_dir: Path) -> None:
    """Upload all files in ``output_dir`` to S3 if running on AIchor (no-op otherwise)."""
    if not S3FileHandler._aichor_enabled():
        return
    s3 = S3FileHandler()
    files = [f for f in output_dir.rglob("*") if f.is_file()]
    logger.info(f"Uploading {len(files)} files to S3...")
    for f in files:
        s3_path = S3FileHandler.convert_to_s3_output(str(f))
        s3.upload(str(f), s3_path)
        logger.info(f"Uploaded {f.name} → {s3_path}")
    logger.info("S3 upload complete")


def load_spectrum_dataframe(
    parquet_path: str,
    max_samples: Optional[int] = None,
    *,
    seed: int = 42,
    column_mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Load spectra from a parquet file (or glob / ``s3://`` URI) into a pandas DataFrame.

    Resolves S3 sources locally, loads via :class:`SpectrumDataFrame` (lazy, annotated) and
    collects up to ``max_samples`` rows. The returned frame holds the **raw, untransformed**
    spectra (``mz_array`` / ``intensity_array``) alongside precursor and metadata columns — no
    spectrum preprocessing is applied here, so callers can apply their own (the shared
    FoundationalDataProcessor for embeddings, or a model-native pipeline for de novo).

    Args:
        parquet_path: Path to the parquet file (or glob pattern), or an ``s3://`` URI.
        max_samples: Cap on total samples loaded (None = all).
        seed: Seed for :meth:`SpectrumDataFrame.collect_chunked` (deterministic subsampling).
        column_mapping: Optional column renaming (e.g. an inference config's ``column_map``) applied
            by :class:`SpectrumDataFrame` on load, mapping source column names to the canonical ones.

    Returns:
        A pandas DataFrame with one row per spectrum.
    """
    from instanovo_fm.utils.spectrum_dataframe import SpectrumDataFrame

    # SpectrumDataFrame reads only local files; materialise S3 parquet locally first.
    parquet_source = S3FileHandler().download_parquet(parquet_path) if parquet_path.startswith("s3://") else parquet_path

    logger.info(f"Loading dataset from {parquet_source}")
    sdf = SpectrumDataFrame.load(
        source=parquet_source,
        lazy=True,
        is_annotated=True,
        shuffle=False,
        partition=None,
        add_source_file_column=True,
        preshuffle_across_shards=False,
        column_mapping=column_mapping,
    )

    total = len(sdf)
    if max_samples is not None and max_samples < total:
        logger.info(f"Collecting up to {max_samples:,} of {total:,} samples")
    return sdf.collect_chunked(max_samples=max_samples, seed=seed).to_pandas()


# TODO reuse predict functionality?
def build_dataloader(
    parquet_path: str,
    batch_size: int,
    max_samples: Optional[int],
    num_workers: int = 4,
    *,
    max_mz: float,
    n_peaks: int = 200,
    min_mz: float = 50.0,
    min_intensity: float = 0.01,
    remove_precursor_tol: float = 0.0,
    normalize_mz: bool = True,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader from a parquet file using FoundationalDataProcessor.

    Replicates the evaluator's setup_dataloader() approach:
      SpectrumDataFrame → collect_chunked → HFDataset → process_dataset → DataLoader

    Args:
        parquet_path: Path to the parquet file (or glob pattern).
        batch_size: Number of spectra per batch.
        max_samples: Cap on total samples loaded (None = all).
        num_workers: Number of DataLoader workers.
        max_mz: Maximum m/z to retain; also the normalisation divisor when ``normalize_mz`` is
            True (must match the value the caller uses to denormalise before its encoder).
        n_peaks: Maximum number of peaks kept per spectrum.
        min_mz: Minimum m/z to retain.
        min_intensity: Minimum intensity threshold.
        remove_precursor_tol: Precursor-peak removal tolerance in Da (0.0 = no removal).
        normalize_mz: If True, m/z is divided by ``max_mz`` (caller denormalises); if False, m/z
            is left in raw Da.

    Returns:
        DataLoader yielding batches with keys: spectra, spectra_mask, precursors,
        precursor_mz, precursor_charge, precursor_mass, peptides, prediction_id.
    """
    import hashlib
    import os

    import datasets.arrow_dataset as _ds_arrow
    from datasets import Value
    from datasets.fingerprint import generate_random_fingerprint as _rand_fp

    df_pd = load_spectrum_dataframe(parquet_path, max_samples, seed=42)

    # Monkey-patch generate_fingerprint to avoid MemoryError on large datasets.
    _orig_gen_fp = _ds_arrow.generate_fingerprint
    _ds_arrow.generate_fingerprint = lambda _ds: _rand_fp()
    try:
        from datasets import Dataset as HFDataset

        dataset = HFDataset.from_pandas(df_pd)
    finally:
        _ds_arrow.generate_fingerprint = _orig_gen_fp
    logger.info(f"Loaded {len(dataset):,} samples")
    logger.info(f"Parquet columns ({len(dataset.column_names)}): {sorted(dataset.column_names)}")

    # Pass new_fingerprint to bypass update_fingerprint() on add_column()
    new_fp = hashlib.md5(os.urandom(16)).hexdigest()
    dataset = dataset.add_column(
        "prediction_id",
        np.arange(len(dataset)),
        feature=Value("int32"),
        new_fingerprint=new_fp,
    )

    processor = FoundationalDataProcessor(
        n_peaks=n_peaks,
        min_mz=min_mz,
        max_mz=max_mz,
        min_intensity=min_intensity,
        remove_precursor_tol=remove_precursor_tol,
        normalize_mz=normalize_mz,
        annotated=True,
        return_str=True,
        masking_strategy="none",
    )
    # frag_type / collision_energy are optional: QC / validation sets (e.g. helaqc) omit them.
    # Guard every optional column so extraction still runs on datasets that lack some — the batch
    # metadata collector already skips absent columns, and the linear probe skips any target field
    # missing from the metadata.
    meta_cols = ["prediction_id"]
    for optional_col in ("frag_type", "collision_energy", "header", "usi", "search_instrument", "search_project", "search_detector"):
        if optional_col in dataset.column_names:
            meta_cols.append(optional_col)
    processor.add_metadata_columns(meta_cols)
    # Keep string/list metadata (sequence, frag_type, usi, ...) through collate — by
    # default the collate_fn strips non-tensor items. Mirrors the FM evaluator
    # (evaluator.py setup_dataloader) so embedding_io / eval tasks receive full metadata.
    processor._keep_non_tensor_metadata = True
    dataset = processor.process_dataset(dataset, return_format="torch")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=processor.collate_fn,
    )
    logger.info(f"DataLoader ready: {len(dataset):,} samples, batch_size={batch_size}")
    return dataloader


def collect_batch_metadata(batch: Dict[str, Any], start_idx: int, batch_size_actual: int) -> Dict[str, np.ndarray]:
    """Build the per-batch metadata dict accumulated during the embedding loop.

    Args:
        batch: A collated batch from the dataloader.
        start_idx: Running offset (number of spectra processed before this batch),
            used to build a globally-unique ``sample_idx``.
        batch_size_actual: Number of spectra in this batch.

    Returns:
        Dict of per-spectrum metadata arrays for this batch.
    """
    batch_meta: Dict[str, np.ndarray] = {
        "precursor_mz": batch["precursor_mz"].numpy().astype(np.float32),
        "precursor_charge": batch["precursor_charge"].numpy().astype(np.int64),
        "precursor_mass": batch["precursor_mass"].numpy().astype(np.float32),
        "sample_idx": np.arange(batch_size_actual, dtype=np.int64) + start_idx,
    }

    # Sequences (present when annotated=True)
    if "peptides" in batch:
        peptides = batch["peptides"]
        if isinstance(peptides, torch.Tensor):
            peptides = [str(p) for p in peptides.tolist()]
        batch_meta["peptides"] = np.array(peptides, dtype=object)
        batch_meta["sequence"] = np.array(peptides, dtype=object)

    # Fragmentation type and collision energy (string / float metadata columns)
    if "frag_type" in batch:
        batch_meta["frag_type"] = np.array(batch["frag_type"], dtype=object)
    if "collision_energy" in batch:
        ce = batch["collision_energy"]
        batch_meta["collision_energy"] = np.array(
            [float(v) if v is not None else float("nan") for v in ce],
            dtype=np.float32,
        )
    for str_col in ("header", "usi", "search_instrument", "search_project", "search_detector"):
        if str_col in batch:
            batch_meta[str_col] = np.array(batch[str_col], dtype=object)

    return batch_meta


def _concatenate_metadata(all_metadata: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Concatenate per-batch metadata dicts into one, keeping only fully-present keys."""
    merged_meta: Dict[str, np.ndarray] = {}
    all_keys = set().union(*[m.keys() for m in all_metadata])
    for key in all_keys:
        parts = [m[key] for m in all_metadata if key in m]
        if len(parts) == len(all_metadata):
            try:
                merged_meta[key] = np.concatenate(parts, axis=0)
            except ValueError:
                merged_meta[key] = np.array([v for part in parts for v in part], dtype=object)
    return merged_meta


def build_faiss_index(embeddings: np.ndarray) -> Any:
    """Build a cosine (inner-product on L2-normalised vectors) FAISS index."""
    if not FAISS_AVAILABLE:
        raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")
    logger.info("Building FAISS index")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    emb_normalised = embeddings.astype(np.float32)
    faiss.normalize_L2(emb_normalised)
    index.add(emb_normalised)
    logger.info(f"FAISS index: {index.ntotal} vectors, dim={dim}")
    return index


def finalize_and_save(
    all_embeddings: List[np.ndarray],
    all_metadata: List[Dict[str, np.ndarray]],
    *,
    search_data_path: str,
    output_dir: str,
    pooling: str,
) -> None:
    """Concatenate, derive post-hoc metadata, build the FAISS index, save, and upload.

    Args:
        all_embeddings: Per-batch ``(B, D)`` embedding arrays.
        all_metadata: Per-batch metadata dicts (from :func:`collect_batch_metadata`).
        search_data_path: Path to ``search_data.csv`` for the USI lookup.
        output_dir: Directory to write ``embeddings.h5`` and ``index.faiss``.
        pooling: Pooling strategy label, stored as an attribute on the embedding store.
    """
    embeddings_array = np.concatenate(all_embeddings, axis=0)  # (N, D)
    merged_meta = _concatenate_metadata(all_metadata)

    finalize_metadata(merged_meta, search_data_path)

    index = build_faiss_index(embeddings_array)

    save(embeddings_array, merged_meta, index, output_dir, embedding_pooling=pooling)
    logger.info(f"Saved to {output_dir}")

    upload_to_s3(Path(output_dir))
