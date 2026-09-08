"""Extract FM de novo (DownstreamDeNovo) encoder embeddings into the foundation model HDF5 format.

Usage:
    uv run python -m instanovo_fm.eval.extract_fm_de_novo_embeddings \
        --parquet_path /path/to/data.parquet \
        --output_dir /path/to/output/fm_de_novo \
        --checkpoint_path /path/to/model_best.ckpt \
        [--batch_size 256] \
        [--device cuda] \
        [--max_samples 200000] \
        [--pooling mean_peaks]

This is the sibling of ``extract_instanovo_embeddings.py`` for the **downstream
de novo model** (FM encoder + InstaNovo decoder, class ``DownstreamDeNovo``).
The output is written in the exact same HDF5 + FAISS layout so the resulting
embeddings can be fed into the same linear-probe / duplicate-retrieval task
pipeline and compared directly against the InstaNovo / Casanovo baselines and
the foundation-model run.

Why a dedicated loader:
    The checkpoint is saved by ``DownstreamDeNovo`` and has no ``mz_head``;
    ``FoundationModel.load()`` crashes on it and would rebuild the wrong
    architecture. We therefore load via ``DownstreamDeNovo.load()``.

Encoder pooling (differs from the InstaNovo extractor):
    The DownstreamDeNovo encoder output layout is ``[latent(0), peak_1 ... peak_N]``
    (no meta tokens for this checkpoint; ``num_prepended`` is derived from the
    model at runtime, never hardcoded).
      - "mean_peaks" (default, comparable): mean over non-padding peak tokens,
        i.e. ``x[:, num_prepended:]``. Matches the FM evaluator's ``mean_pool``.
      - "last_token": the latent token at position 0 (NOT the last position —
        the FM/downstream encoder prepends the latent, unlike standard InstaNovo
        which appends it).

m/z convention (differs from the InstaNovo extractor):
    Do NOT denormalise. The FM/downstream encoder expects the normalised m/z
    that ``FoundationalDataProcessor`` produces (m/z divided by max_mz). Precursors
    are passed as ``None`` to ``_encoder``; precursor info is only used by the
    (unused here) decoder.

Preprocessing:
    The ``FoundationalDataProcessor`` is configured from the checkpoint's stored
    preprocessing values (``n_peaks``, ``min_mz``, ``max_mz``, ``min_intensity``,
    ``normalize_mz`` …) so the inputs match what the model was trained on. Note
    that ``min_intensity`` is therefore the checkpoint's value and is NOT matched
    to the baselines' ``min_intensity`` — flag this when comparing.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from instanovo.__init__ import console
from instanovo_fm.common.dataset import DataProcessor
from instanovo_fm.data.data import FoundationalDataProcessor
from instanovo_fm.data.search_data_manager import SearchDataManager
from instanovo_fm.downstream.de_novo_sequencing.model import DownstreamDeNovo
from instanovo_fm.eval.embedding_io import save
from instanovo_fm.utils.hydrophobicity import compute_hydrophobicity
from instanovo_fm.utils.modifications import compute_modification_types
from instanovo.utils.colorlogging import ColorLog
from instanovo.utils.s3 import S3FileHandler

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

logger = ColorLog(console, __name__).logger

# Compiled patterns for fragmentation-type extraction from spectrum header strings.
_FRAG_AT_PATTERN = re.compile(r"@(hcd|hcid|cid|etd|ecd|uvpd|ethcd)", re.IGNORECASE)
_FRAG_WORD_PATTERN = re.compile(r"\b(HCD|HCID|CID|ETD|ECD|UVPD|EThcD)\b")
_FRAG_NORM = {"HCID": "HCD", "ECD": "ETD", "ETHCD": "ETD"}


def _derive_frag_type_from_header(header_array: np.ndarray) -> np.ndarray:
    """Extract fragmentation type from spectrum header/scan-filter strings.

    Searches for Thermo ``@fragtype`` scan-filter patterns first (most
    reliable), then falls back to word-boundary matching for ``HCD``,
    ``CID``, ``ETD``, ``UVPD`` etc.  Returns ``None`` for headers where
    no fragmentation type can be identified.

    Args:
        header_array: Object array of header strings (elements may be None).

    Returns:
        Object array of canonical frag-type strings ("HCD", "CID", "ETD",
        "UVPD") or None where the type could not be determined.
    """
    result = np.empty(len(header_array), dtype=object)
    for i, h in enumerate(header_array):
        if h is None or not isinstance(h, str) or not h.strip():
            result[i] = None
            continue
        # Thermo @fragtype pattern is most reliable
        m = _FRAG_AT_PATTERN.search(h)
        if m:
            ft = m.group(1).upper()
            result[i] = _FRAG_NORM.get(ft, ft)
            continue
        # Word-boundary match as fallback
        m = _FRAG_WORD_PATTERN.search(h)
        if m:
            ft = m.group(1).upper()
            result[i] = _FRAG_NORM.get(ft, ft)
            continue
        result[i] = None
    return result


def _upload_to_s3(output_dir: Path) -> None:
    """Upload all files in output_dir to S3 if running on AIchor."""
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


def _build_dataloader(
    parquet_path: str,
    config: dict,
    batch_size: int,
    max_samples: Optional[int],
    num_workers: int = 4,
    min_intensity: Optional[float] = None,
    max_mz: Optional[float] = None,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader from a parquet file using FoundationalDataProcessor.

    Replicates the evaluator's setup_dataloader() approach:
      SpectrumDataFrame → collect_chunked → HFDataset → process_dataset → DataLoader

    Preprocessing parameters are read from the checkpoint ``config`` so the
    inputs match what the model was trained on, unless explicitly overridden.

    Args:
        parquet_path: Path to the parquet file (or glob pattern).
        config: The checkpoint's stored config (provides preprocessing values).
        batch_size: Number of spectra per batch.
        max_samples: Cap on total samples loaded (None = all).
        num_workers: Number of DataLoader workers.
        min_intensity: Override the checkpoint's min_intensity (None = use config).
            The baseline scripts pass this explicitly (e.g. 1e-6) for comparability.
        max_mz: Override the checkpoint's max_mz / normalisation divisor (None = use config).

    Returns:
        DataLoader yielding batches with keys: spectra, spectra_mask, precursors,
        precursor_mz, precursor_charge, precursor_mass, peptides, prediction_id.
    """
    import hashlib
    import os

    import datasets.arrow_dataset as _ds_arrow
    from datasets import Value
    from datasets.fingerprint import generate_random_fingerprint as _rand_fp

    from instanovo_fm.utils.spectrum_dataframe import SpectrumDataFrame

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
    )

    total = len(sdf)
    if max_samples is not None and max_samples < total:
        logger.info(f"Collecting up to {max_samples:,} of {total:,} samples")
    df = sdf.collect_chunked(max_samples=max_samples, seed=42)

    # Monkey-patch generate_fingerprint to avoid MemoryError on large datasets.
    _orig_gen_fp = _ds_arrow.generate_fingerprint
    _ds_arrow.generate_fingerprint = lambda _ds: _rand_fp()
    try:
        from datasets import Dataset as HFDataset

        dataset = HFDataset.from_pandas(df.to_pandas())
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

    # Configure preprocessing from the checkpoint so inputs match training,
    # unless min_intensity / max_mz are explicitly overridden (baseline scripts).
    resolved_min_intensity = min_intensity if min_intensity is not None else config.get("min_intensity", 0.01)
    resolved_max_mz = max_mz if max_mz is not None else config.get("max_mz", 2500.0)
    logger.info(f"Preprocessing: min_intensity={resolved_min_intensity}, max_mz={resolved_max_mz}")
    processor = FoundationalDataProcessor(
        n_peaks=config.get("n_peaks", 200),
        min_mz=config.get("min_mz", 50.0),
        max_mz=resolved_max_mz,
        min_intensity=resolved_min_intensity,
        use_spectrum_utils=config.get("use_spectrum_utils", False),
        normalize_mz=config.get("normalize_mz", True),
        peak_ordering=config.get("peak_ordering", "sorted"),
        annotated=True,
        return_str=True,
        masking_strategy="none",
    )
    meta_cols = ["prediction_id", "frag_type", "collision_energy"]
    for optional_col in ("header", "usi", "search_instrument", "search_project"):
        if optional_col in dataset.column_names:
            meta_cols.append(optional_col)
    processor.add_metadata_columns(meta_cols)
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


def extract(
    parquet_path: str,
    output_dir: str,
    checkpoint_path: str,
    batch_size: int = 256,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    num_workers: int = 4,
    pooling: str = "mean_peaks",
    search_data_path: str = "data/search_data.xlsx",
    checkpoint_url: Optional[str] = None,
    min_intensity: Optional[float] = None,
    max_mz: Optional[float] = None,
) -> None:
    """Extract DownstreamDeNovo encoder embeddings and save in foundation-model HDF5 format.

    Args:
        parquet_path: Path to input parquet file(s).
        output_dir: Directory to write embeddings.h5 and index.faiss.
        checkpoint_path: Local (or s3://) path to the DownstreamDeNovo checkpoint (.ckpt).
        batch_size: Inference batch size.
        device: PyTorch device string ("cuda", "cpu", "cuda:1", …).
        max_samples: Maximum number of spectra to embed.
        num_workers: DataLoader worker count.
        pooling: Pooling strategy — "mean_peaks" (mean over non-padding peak
            tokens, the comparable default) or "last_token" (the prepended latent
            token at position 0).
        search_data_path: Path to the search-data table for USI-based frag_type and
            search_instrument lookup.
        checkpoint_url: Optional URL to download the checkpoint from if
            checkpoint_path is a local path that does not exist (interface parity
            with the baseline extractors; unused for s3:// checkpoints).
        min_intensity: Override the checkpoint's min_intensity preprocessing value.
        max_mz: Override the checkpoint's max_mz / normalisation divisor.
    """
    if pooling not in ("mean_peaks", "last_token"):
        raise ValueError(f"pooling must be 'mean_peaks' or 'last_token', got {pooling!r}")
    if not FAISS_AVAILABLE:
        raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")

    # ------------------------------------------------------------------ #
    # 1. Resolve checkpoint (local or s3://)                              #
    # ------------------------------------------------------------------ #
    # NOTE: the S3FileHandler must stay referenced for the whole function.
    # get_local_path() downloads into handler.temp_dir (a TemporaryDirectory
    # owned by the instance); if the handler is garbage-collected the temp dir
    # is deleted and the just-downloaded checkpoint disappears before torch.load.
    s3_handler = S3FileHandler()
    resolved_ckpt = checkpoint_path
    if checkpoint_path.startswith("s3://"):
        logger.info(f"Downloading checkpoint from S3: {checkpoint_path}")
        downloaded = s3_handler.get_local_path(checkpoint_path)
        if downloaded is None:
            raise FileNotFoundError(f"Could not resolve checkpoint from S3: {checkpoint_path}")
        resolved_ckpt = downloaded
        logger.info(f"Downloaded to local path: {resolved_ckpt}")
    else:
        if not Path(resolved_ckpt).exists():
            if checkpoint_url:
                import urllib.request

                Path(resolved_ckpt).parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Checkpoint not found locally; downloading from {checkpoint_url}")
                urllib.request.urlretrieve(checkpoint_url, str(resolved_ckpt))
                logger.info(f"Downloaded checkpoint to {resolved_ckpt}")
            else:
                raise FileNotFoundError(f"Checkpoint not found: {resolved_ckpt}")

    # ------------------------------------------------------------------ #
    # 2. Load model (DownstreamDeNovo, NOT FoundationModel)               #
    # ------------------------------------------------------------------ #
    torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    logger.info(f"Loading DownstreamDeNovo checkpoint from {resolved_ckpt}")
    model, config = DownstreamDeNovo.load(str(resolved_ckpt))
    model = model.to(torch_device)
    model.eval()
    logger.info(
        f"Model loaded (dim_model={config.get('dim_model', '?')}, "
        f"encoder_layers={config.get('encoder_layers', config.get('n_layers', '?'))}, "
        f"use_meta_token={config.get('use_meta_token', False)})"
    )

    # ------------------------------------------------------------------ #
    # 3. Build DataLoader (preprocessing from checkpoint config)          #
    # ------------------------------------------------------------------ #
    dataloader = _build_dataloader(
        parquet_path,
        config,
        batch_size,
        max_samples,
        num_workers,
        min_intensity=min_intensity,
        max_mz=max_mz,
    )

    # ------------------------------------------------------------------ #
    # 4. Embedding loop                                                    #
    # ------------------------------------------------------------------ #
    all_embeddings: list[np.ndarray] = []
    all_metadata: list[Dict[str, np.ndarray]] = []
    total_processed = 0

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="FM de novo embeddings"):
            spectra = batch["spectra"].to(torch_device)  # (B, n_peaks, 2) normalised m/z
            spectra_mask = batch["spectra_mask"].to(torch_device)  # (B, n_peaks), True=padding

            # FM/downstream encoder expects NORMALISED m/z and precursors=None.
            # Output layout: [latent(0), peak_1 ... peak_N]
            memories, _ = model._encoder(spectra, precursors=None, spectra_mask=spectra_mask, meta=None)

            # num_prepended = latent (+ any meta tokens) = total tokens - n_peaks.
            # Derived from the model, never hardcoded.
            num_prepended = memories.shape[1] - spectra.shape[1]

            if pooling == "last_token":
                # Prepended latent token at position 0
                embedding = memories[:, 0, :]  # (B, D)
            else:
                # mean_peaks: average over non-padding peak tokens only.
                peak_tokens = memories[:, num_prepended:, :]  # (B, n_peaks, D)
                valid = (~spectra_mask).unsqueeze(-1).float()  # (B, n_peaks, 1)
                embedding = (peak_tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

            emb_np = embedding.cpu().float().numpy()
            n_in_batch = len(emb_np)

            # Accumulate metadata
            batch_meta: Dict[str, np.ndarray] = {
                "precursor_mz": batch["precursor_mz"].numpy().astype(np.float32),
                "precursor_charge": batch["precursor_charge"].numpy().astype(np.float32),
                "precursor_mass": batch["precursor_mass"].numpy().astype(np.float32),
                "sample_idx": np.arange(n_in_batch, dtype=np.int64) + total_processed,
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
            for str_col in ("header", "usi", "search_instrument", "search_project"):
                if str_col in batch:
                    batch_meta[str_col] = np.array(batch[str_col], dtype=object)

            all_embeddings.append(emb_np)
            all_metadata.append(batch_meta)
            total_processed += n_in_batch

    logger.info(f"Embedded {total_processed:,} spectra")

    # ------------------------------------------------------------------ #
    # 5. Concatenate                                                       #
    # ------------------------------------------------------------------ #
    embeddings_array = np.concatenate(all_embeddings, axis=0)  # (N, D)

    merged_meta: Dict[str, np.ndarray] = {}
    all_keys = set().union(*[m.keys() for m in all_metadata])
    for key in all_keys:
        parts = [m[key] for m in all_metadata if key in m]
        if len(parts) == len(all_metadata):
            try:
                merged_meta[key] = np.concatenate(parts, axis=0)
            except ValueError:
                merged_meta[key] = np.array([v for part in parts for v in part], dtype=object)

    # ------------------------------------------------------------------ #
    # 5b. Post-hoc metadata: frag_type + search_instrument from
    #     SearchDataManager (USI lookup), hydrophobicity, ptm_present,
    #     modification_class
    # ------------------------------------------------------------------ #

    # Primary source: search_data.csv via USI lookup (same source as mlflow-integration).
    if "usi" in merged_meta:
        sdm = SearchDataManager(search_data_path)
        sdm.load()
        if sdm.is_loaded:
            usis = [str(u) if u is not None else "" for u in merged_meta["usi"]]
            search_results = sdm.get_metadata_batch(usis, columns=["fragmentation", "instrument", "project"])

            def _is_valid(v: object) -> bool:
                return v is not None and str(v).strip() not in ("", "None", "nan")

            # frag_type: take first token of compound values (e.g. "HCD;CID" → "HCD")
            csv_frag = np.array(
                [str(r["fragmentation"]).split(";")[0].split("|")[0].strip() if _is_valid(r.get("fragmentation")) else None for r in search_results],
                dtype=object,
            )
            n_csv_frag = int(np.sum([_is_valid(v) for v in csv_frag]))
            logger.info(f"search_data frag_type: {n_csv_frag:,}/{len(csv_frag):,} spectra matched")
            if n_csv_frag > 0:
                # Where CSV has a value, use it; else keep existing parquet value (usually None)
                existing = merged_meta.get("frag_type", np.full(len(csv_frag), None, dtype=object))
                merged_meta["frag_type"] = np.where(np.array([_is_valid(v) for v in csv_frag]), csv_frag, existing)

            # search_instrument
            csv_inst = np.array(
                [r.get("instrument") if _is_valid(r.get("instrument")) else None for r in search_results],
                dtype=object,
            )
            n_csv_inst = int(np.sum([_is_valid(v) for v in csv_inst]))
            logger.info(f"search_data search_instrument: {n_csv_inst:,}/{len(csv_inst):,} spectra matched")
            if n_csv_inst > 0:
                merged_meta["search_instrument"] = csv_inst

            # search_project (needed for use_project_split=true)
            csv_proj = np.array(
                [r.get("project") if _is_valid(r.get("project")) else None for r in search_results],
                dtype=object,
            )
            n_csv_proj = int(np.sum([_is_valid(v) for v in csv_proj]))
            logger.info(f"search_data search_project: {n_csv_proj:,}/{len(csv_proj):,} spectra matched")
            if n_csv_proj > 0:
                merged_meta["search_project"] = csv_proj

    # Fallback: derive frag_type from Thermo scan-filter header strings.
    frag_type_arr = merged_meta.get("frag_type")
    frag_type_missing = frag_type_arr is None or not np.any(np.array([v is not None for v in frag_type_arr]))
    if frag_type_missing and "header" in merged_meta:
        logger.info("frag_type still empty after CSV lookup — deriving from header strings")
        derived = _derive_frag_type_from_header(merged_meta["header"])
        n_found = int(np.sum([v is not None for v in derived]))
        logger.info(f"frag_type derived from header: {n_found:,}/{len(derived):,} spectra have a type")
        if n_found > 0:
            merged_meta["frag_type"] = derived

    if "sequence" in merged_meta:
        try:
            sequences = merged_meta["sequence"]
            cleaned = np.empty(len(sequences), dtype=object)
            for i, seq in enumerate(sequences):
                if seq is not None and isinstance(seq, str) and seq.strip():
                    cleaned[i] = DataProcessor.clean_peptide_for_pyopenms(seq, keep_modifications=True)
                else:
                    cleaned[i] = None

            modification_types = compute_modification_types(cleaned, use_modified_peptide=True)
            if modification_types is not None and len(modification_types) > 0:
                merged_meta["modification_types"] = modification_types
                is_modified = modification_types != "Unmodified"
                merged_meta["ptm_present"] = is_modified.astype(np.int32)
                n_modified = int(is_modified.sum())
                logger.info(f"PTM summary: {n_modified:,}/{len(modification_types):,} modified ({100 * n_modified / len(modification_types):.1f}%)")

                has_multiple = np.array([" + " in str(m) for m in modification_types], dtype=bool)
                single_mods = modification_types.copy()
                single_mods[has_multiple] = "Other"
                unique_mods, counts = np.unique(single_mods, return_counts=True)
                logger.info(f"Modification classes found: {', '.join(f'{m}={c}' for m, c in zip(unique_mods, counts, strict=False))}")
                mask = (unique_mods != "Unmodified") & (unique_mods != "Other")
                ranked_mods = unique_mods[mask]
                if len(ranked_mods) > 0:
                    top_6 = set(ranked_mods[np.argsort(counts[mask])[::-1][:6]])
                    merged_meta["modification_class"] = np.where(
                        single_mods == "Unmodified",
                        "Unmodified",
                        np.where(np.isin(single_mods, list(top_6)), single_mods, "Other"),
                    )
                else:
                    logger.warning(
                        "No non-Unmodified modification classes found in this sample. modification_class probe will report insufficient classes."
                    )
                    merged_meta["modification_class"] = single_mods
        except Exception as exc:
            logger.warning(f"Failed to compute modification metadata: {exc}", exc_info=True)

    if "peptides" in merged_meta:
        try:
            hydrophobicity = compute_hydrophobicity(merged_meta["peptides"])
            if hydrophobicity is not None:
                merged_meta["hydrophobicity"] = hydrophobicity
        except Exception as exc:
            logger.warning(f"Failed to compute hydrophobicity: {exc}")

    # ------------------------------------------------------------------ #
    # 6. Build FAISS index                                                 #
    # ------------------------------------------------------------------ #
    logger.info("Building FAISS index")
    dim = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dim)
    emb_normalised = embeddings_array.astype(np.float32)
    faiss.normalize_L2(emb_normalised)
    index.add(emb_normalised)
    logger.info(f"FAISS index: {index.ntotal} vectors, dim={dim}")

    # ------------------------------------------------------------------ #
    # 7. Save                                                              #
    # ------------------------------------------------------------------ #
    save(embeddings_array, merged_meta, index, output_dir, embedding_pooling=pooling)
    logger.info(f"Saved to {output_dir}")

    # ------------------------------------------------------------------ #
    # 8. Upload to S3 (AIchor)                                             #
    # ------------------------------------------------------------------ #
    _upload_to_s3(Path(output_dir))


def main() -> None:
    """Parse CLI arguments and run the embedding extraction pipeline."""
    parser = argparse.ArgumentParser(description="Extract DownstreamDeNovo encoder embeddings into foundation-model HDF5 format")
    parser.add_argument("--parquet_path", required=True, help="Path to input parquet file or glob pattern")
    parser.add_argument("--output_dir", required=True, help="Directory to write embeddings.h5 and index.faiss")
    parser.add_argument("--checkpoint_path", required=True, help="Local or s3:// path to the DownstreamDeNovo checkpoint (.ckpt)")
    parser.add_argument("--batch_size", type=int, default=256, help="Inference batch size (default: 256)")
    parser.add_argument("--device", default="cuda", help="PyTorch device string: cuda, cpu, cuda:1, etc. (default: cuda)")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap on number of spectra to embed (default: all)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker count (default: 4)")
    parser.add_argument(
        "--pooling",
        default="mean_peaks",
        choices=["mean_peaks", "last_token"],
        help="Pooling strategy: 'mean_peaks' (comparable default) or 'last_token' (latent token)",
    )
    parser.add_argument(
        "--search_data_path",
        default="data/search_data.xlsx",
        help="Path to the search-data table for USI-based frag_type/search_instrument lookup",
    )
    parser.add_argument(
        "--checkpoint_url",
        default=None,
        help="Optional URL to download the checkpoint if --checkpoint_path is a missing local path "
        "(interface parity with baseline extractors; unused for s3:// checkpoints).",
    )
    parser.add_argument(
        "--max_mz",
        type=float,
        default=None,
        help="Override the checkpoint's max_mz / m/z normalisation divisor (default: from checkpoint config).",
    )
    parser.add_argument(
        "--min_intensity",
        type=float,
        default=None,
        help="Override the checkpoint's min_intensity preprocessing (default: from checkpoint config).",
    )
    args = parser.parse_args()

    # Treat empty-string CLI values (passed by the shell driver for unused options) as None.
    checkpoint_url = args.checkpoint_url or None

    extract(
        parquet_path=args.parquet_path,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        device=args.device,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        pooling=args.pooling,
        search_data_path=args.search_data_path,
        checkpoint_url=checkpoint_url,
        min_intensity=args.min_intensity,
        max_mz=args.max_mz,
    )


if __name__ == "__main__":
    main()
