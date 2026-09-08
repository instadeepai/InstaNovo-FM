"""Extract Casanovo spectrum encoder embeddings into the foundation model HDF5 format.

Usage:
    uv run python -m instanovo_fm.eval.extract_casanovo_embeddings \
        --parquet_path /path/to/data.parquet \
        --output_dir /path/to/output/casanovo \
        --checkpoint_path /path/to/casanovo.ckpt \
        [--batch_size 64] \
        [--device cuda] \
        [--max_samples 10000] \
        [--checkpoint_url https://...]

Prerequisites:
    casanovo must be installed in the active environment before running:
        uv pip install casanovo

Checkpoint download:
    Official Casanovo checkpoints are available at:
        https://github.com/Noble-Lab/casanovo/releases

Encoder pooling:
    Casanovo's Spec2Pep.encoder returns memories of shape (B, seq_len+1, D),
    where the first token (memories[:, 0, :]) is the global/CLS token used as
    the spectrum-level embedding.

m/z denormalisation:
    FoundationalDataProcessor normalises m/z to [0, 1] by dividing by max_mz=2500.
    This script multiplies spectra[:, :, 0] by 2500 before passing to the Casanovo
    encoder, which expects raw (unnormalised) Da values.
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from instanovo.__init__ import console
from instanovo_fm.eval.embedding_io import save
from instanovo_fm.data.data import FoundationalDataProcessor
from instanovo_fm.data.search_data_manager import SearchDataManager
from instanovo.utils.colorlogging import ColorLog
from instanovo.utils.s3 import S3FileHandler
from instanovo_fm.utils.hydrophobicity import compute_hydrophobicity
from instanovo_fm.utils.modifications import (
    compute_modification_types,
    compute_ptm_binary_flags,
)
from instanovo.common.dataset import DataProcessor

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

logger = ColorLog(console, __name__).logger

# FoundationalDataProcessor normalises m/z by dividing by this value.
MAX_MZ = 2500.0

# Compiled patterns for fragmentation-type extraction from spectrum header strings.
import re as _re
_FRAG_AT_PATTERN = _re.compile(r'@(hcd|hcid|cid|etd|ecd|uvpd|ethcd)', _re.IGNORECASE)
_FRAG_WORD_PATTERN = _re.compile(r'\b(HCD|HCID|CID|ETD|ECD|UVPD|EThcD)\b')
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


def _require_casanovo() -> "type":
    """Import Spec2Pep from casanovo, raising a clear error if not installed."""
    try:
        from casanovo.denovo.model import Spec2Pep  # type: ignore[import]
        return Spec2Pep
    except ImportError as exc:
        raise ImportError(
            "casanovo is not installed. Install it with:\n"
            "    uv pip install casanovo\n"
            "then re-run this script."
        ) from exc


def _build_dataloader(
    parquet_path: str,
    batch_size: int,
    max_samples: Optional[int],
    num_workers: int = 4,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader from a parquet file using FoundationalDataProcessor.

    Replicates the evaluator's setup_dataloader() approach:
      SpectrumDataFrame → collect_chunked → HFDataset → process_dataset → DataLoader

    Args:
        parquet_path: Path to the parquet file (or glob pattern).
        batch_size: Number of spectra per batch.
        max_samples: Cap on total samples loaded (None = all).
        num_workers: Number of DataLoader workers.

    Returns:
        DataLoader yielding batches with keys: spectra, spectra_mask, precursors,
        precursor_mz, precursor_charge, precursor_mass, peptides, prediction_id.
    """
    import hashlib
    import os
    import datasets.arrow_dataset as _ds_arrow
    from datasets import Value
    from datasets.fingerprint import generate_random_fingerprint as _rand_fp
    from instanovo.utils.data_handler import SpectrumDataFrame

    logger.info(f"Loading dataset from {parquet_path}")
    sdf = SpectrumDataFrame.load(
        source=parquet_path,
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

    processor = FoundationalDataProcessor(
        n_peaks=200,
        normalize_mz=True,
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


def _download_checkpoint(url: str, dest: Path) -> None:
    """Download a checkpoint file with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading checkpoint from {url} → {dest}")

    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded / total_size * 100)
            print(f"\r  {pct:.1f}%  ({downloaded // 1_048_576} / {total_size // 1_048_576} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=_reporthook)
    print()  # newline after progress
    logger.info("Download complete")


def extract(
    parquet_path: str,
    output_dir: str,
    checkpoint_path: str,
    batch_size: int = 64,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    checkpoint_url: Optional[str] = None,
    num_workers: int = 4,
    pooling: str = "cls",
    search_data_path: str = "instanovo/foundational/data/search_data.csv",
) -> None:
    """Extract Casanovo encoder embeddings and save in foundation-model HDF5 format.

    Args:
        parquet_path: Path to input parquet file(s).
        output_dir: Directory to write embeddings.h5 and index.faiss.
        checkpoint_path: Local path to casanovo checkpoint (.ckpt).
        batch_size: Inference batch size.
        device: PyTorch device string ("cuda", "cpu", "cuda:1", …).
        max_samples: Maximum number of spectra to embed.
        checkpoint_url: If provided and checkpoint_path doesn't exist, download from here.
        num_workers: DataLoader worker count.
        pooling: Pooling strategy — "cls" (global token at position 0, default) or
            "mean_peaks" (mean over non-padding peak tokens at positions 1+).
        search_data_path: Path to search_data.csv for USI-based frag_type and
            search_instrument lookup.
    """
    if pooling not in ("cls", "mean_peaks"):
        raise ValueError(f"pooling must be 'cls' or 'mean_peaks', got {pooling!r}")
    if not FAISS_AVAILABLE:
        raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")

    Spec2Pep = _require_casanovo()

    # ------------------------------------------------------------------ #
    # 1. Resolve checkpoint                                                #
    # ------------------------------------------------------------------ #
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        if checkpoint_url:
            _download_checkpoint(checkpoint_url, ckpt)
        else:
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt}\n"
                "Pass --checkpoint_url to download automatically, or download manually from:\n"
                "  https://github.com/Noble-Lab/casanovo/releases"
            )

    # ------------------------------------------------------------------ #
    # 2. Load model                                                        #
    # ------------------------------------------------------------------ #
    torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    logger.info(f"Loading Casanovo checkpoint from {ckpt}")
    model = Spec2Pep.load_from_checkpoint(str(ckpt), map_location="cpu", weights_only=False)
    model = model.to(torch_device)
    model.eval()
    logger.info("Casanovo model loaded")

    # ------------------------------------------------------------------ #
    # 3. Build DataLoader                                                  #
    # ------------------------------------------------------------------ #
    dataloader = _build_dataloader(parquet_path, batch_size, max_samples, num_workers)

    # ------------------------------------------------------------------ #
    # 4. Embedding loop                                                    #
    # ------------------------------------------------------------------ #
    all_embeddings: list[np.ndarray] = []
    all_metadata: list[Dict[str, np.ndarray]] = []
    total_processed = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Casanovo embeddings")):
            spectra = batch["spectra"].to(torch_device)           # (B, 200, 2)

            # Denormalise m/z: FoundationalDataProcessor divides by MAX_MZ
            spectra_raw = spectra.clone()
            spectra_raw[:, :, 0] = spectra_raw[:, :, 0] * MAX_MZ

            # Split into separate m/z and intensity tensors expected by Casanovo encoder.
            # Zero-padded positions have mz=0 and intensity=0; Casanovo's attention
            # layers handle them via its internal padding mask.
            mzs = spectra_raw[:, :, 0]         # (B, L)
            intensities = spectra_raw[:, :, 1]  # (B, L)

            # Run Casanovo encoder → (B, seq_len+1, D)
            # Output layout: [cls(0) | peak_1(1) ... peak_N]
            memories, _ = model.encoder(mzs, intensities)

            if pooling == "cls":
                embedding = memories[:, 0, :]  # (B, D)
            else:
                # mean_peaks: average over non-padding peak tokens (positions 1+).
                # Padding peaks have mz=0; use that to build the valid mask.
                peak_tokens = memories[:, 1:, :]  # (B, N_peaks, D)
                valid = (mzs > 0).unsqueeze(-1).float()  # (B, N_peaks, 1)
                embedding = (peak_tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

            emb_np = embedding.cpu().float().numpy()
            B_actual = len(emb_np)

            # Accumulate metadata
            batch_meta: Dict[str, np.ndarray] = {
                "precursor_mz": batch["precursor_mz"].numpy().astype(np.float32),
                "precursor_charge": batch["precursor_charge"].numpy().astype(np.float32),
                "precursor_mass": batch["precursor_mass"].numpy().astype(np.float32),
                "sample_idx": np.arange(B_actual, dtype=np.int64) + total_processed,
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
            total_processed += B_actual

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
                merged_meta[key] = np.array(
                    [v for part in parts for v in part], dtype=object
                )

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
                [str(r["fragmentation"]).split(";")[0].split("|")[0].strip()
                 if _is_valid(r.get("fragmentation")) else None
                 for r in search_results],
                dtype=object,
            )
            n_csv_frag = int(np.sum([_is_valid(v) for v in csv_frag]))
            logger.info(f"search_data frag_type: {n_csv_frag:,}/{len(csv_frag):,} spectra matched")
            if n_csv_frag > 0:
                # Where CSV has a value, use it; else keep existing parquet value (usually None)
                existing = merged_meta.get("frag_type", np.full(len(csv_frag), None, dtype=object))
                merged_meta["frag_type"] = np.where(
                    np.array([_is_valid(v) for v in csv_frag]), csv_frag, existing
                )

            # search_instrument
            csv_inst = np.array(
                [r.get("instrument") if _is_valid(r.get("instrument")) else None
                 for r in search_results],
                dtype=object,
            )
            n_csv_inst = int(np.sum([_is_valid(v) for v in csv_inst]))
            logger.info(f"search_data search_instrument: {n_csv_inst:,}/{len(csv_inst):,} spectra matched")
            if n_csv_inst > 0:
                merged_meta["search_instrument"] = csv_inst

            # search_project (needed for use_project_split=true)
            csv_proj = np.array(
                [r.get("project") if _is_valid(r.get("project")) else None
                 for r in search_results],
                dtype=object,
            )
            n_csv_proj = int(np.sum([_is_valid(v) for v in csv_proj]))
            logger.info(f"search_data search_project: {n_csv_proj:,}/{len(csv_proj):,} spectra matched")
            if n_csv_proj > 0:
                merged_meta["search_project"] = csv_proj

    # Fallback: derive frag_type from Thermo scan-filter header strings.
    frag_type_arr = merged_meta.get("frag_type")
    frag_type_missing = (
        frag_type_arr is None
        or not np.any(np.array([v is not None for v in frag_type_arr]))
    )
    if frag_type_missing and "header" in merged_meta:
        logger.info("frag_type still empty after CSV lookup — deriving from header strings")
        derived = _derive_frag_type_from_header(merged_meta["header"])
        n_found = int(np.sum([v is not None for v in derived]))
        logger.info(
            f"frag_type derived from header: {n_found:,}/{len(derived):,} spectra have a type"
        )
        if n_found > 0:
            merged_meta["frag_type"] = derived

    if "sequence" in merged_meta:
        try:
            sequences = merged_meta["sequence"]
            cleaned = np.empty(len(sequences), dtype=object)
            for i, seq in enumerate(sequences):
                if seq is not None and isinstance(seq, str) and seq.strip():
                    cleaned[i] = DataProcessor.clean_peptide_for_pyopenms(
                        seq, keep_modifications=True
                    )
                else:
                    cleaned[i] = None

            modification_types = compute_modification_types(cleaned, use_modified_peptide=True)
            if modification_types is not None and len(modification_types) > 0:
                merged_meta["modification_types"] = modification_types
                is_modified = modification_types != "Unmodified"
                merged_meta["ptm_present"] = is_modified.astype(np.int32)
                n_modified = int(is_modified.sum())
                ptm_flags = compute_ptm_binary_flags(sequences)
                merged_meta["mod_phospho"] = ptm_flags["mod_phospho"]
                merged_meta["mod_glyco"] = ptm_flags["mod_glyco"]
                logger.info(
                    f"PTM summary: {n_modified:,}/{len(modification_types):,} modified "
                    f"({100*n_modified/len(modification_types):.1f}%)"
                )

                has_multiple = np.array([" + " in str(m) for m in modification_types], dtype=bool)
                single_mods = modification_types.copy()
                single_mods[has_multiple] = "Other"
                unique_mods, counts = np.unique(single_mods, return_counts=True)
                logger.info(
                    f"Modification classes found: "
                    + ", ".join(f"{m}={c}" for m, c in zip(unique_mods, counts))
                )
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
                        "No non-Unmodified modification classes found in this sample. "
                        "modification_class probe will report insufficient classes."
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
    parser = argparse.ArgumentParser(
        description="Extract Casanovo encoder embeddings into foundation-model HDF5 format"
    )
    parser.add_argument(
        "--parquet_path", required=True,
        help="Path to input parquet file or glob pattern"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to write embeddings.h5 and index.faiss"
    )
    parser.add_argument(
        "--checkpoint_path", required=True,
        help="Local path to casanovo checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Inference batch size (default: 64)"
    )
    parser.add_argument(
        "--device", default="cuda",
        help="PyTorch device string: cuda, cpu, cuda:1, etc. (default: cuda)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Cap on number of spectra to embed (default: all)"
    )
    parser.add_argument(
        "--checkpoint_url", default=None,
        help=(
            "URL to download the checkpoint if --checkpoint_path does not exist. "
            "Find releases at: https://github.com/Noble-Lab/casanovo/releases"
        ),
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader worker count (default: 4)"
    )
    parser.add_argument(
        "--pooling", default="cls", choices=["cls", "mean_peaks"],
        help="Pooling strategy: 'cls' (global token, default) or 'mean_peaks'"
    )
    parser.add_argument(
        "--search_data_path",
        default="instanovo/foundational/data/search_data.csv",
        help="Path to search_data.csv for USI-based frag_type/search_instrument lookup",
    )
    args = parser.parse_args()

    extract(
        parquet_path=args.parquet_path,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        device=args.device,
        max_samples=args.max_samples,
        checkpoint_url=args.checkpoint_url,
        num_workers=args.num_workers,
        pooling=args.pooling,
        search_data_path=args.search_data_path,
    )


if __name__ == "__main__":
    main()
