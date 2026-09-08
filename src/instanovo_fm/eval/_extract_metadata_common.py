"""Shared post-hoc metadata helpers for the baseline embedding extractors."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from instanovo.__init__ import console
from instanovo.common.dataset import DataProcessor
from instanovo_fm.data.search_data_manager import SearchDataManager
from instanovo_fm.utils.hydrophobicity import compute_hydrophobicity
from instanovo_fm.utils.modifications import compute_modification_types
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger

_SEARCH_DATA_COLUMNS: Dict[str, str] = {
    "instrument": "search_instrument",  # conditional-subset / probe label
    "project": "search_project",  # needed for use_project_split=true
    "detector": "search_detector",  # conditional-subset detector filtering (e.g. hcd_orbitrap)
    "organism": "search_organism",  # UMAP: Search Organism visualisation
}


def _is_valid(value: object) -> bool:
    """Return True if a looked-up metadata value is a real (non-missing) string."""
    return value is not None and str(value).strip() not in ("", "None", "nan")


def apply_search_data_metadata(merged_meta: Dict[str, Any], search_data_path: str) -> None:
    """Populate search-engine metadata fields in ``merged_meta`` from ``search_data.csv``.

    Populates ``search_instrument`` / ``search_project`` / ``search_detector`` /
    ``search_organism`` via a USI lookup. ``frag_type`` is intentionally NOT set or
    overwritten here: the FM reads fragmentation straight from the parquet spectra
    metadata (no CSV, no preprocessing), so the baseline keeps the parquet value as-is.

    Args:
        merged_meta: Per-spectrum metadata dict, mutated in place. A ``"usi"``
            entry must be present for any lookup to occur.
        search_data_path: Path to ``search_data.csv``.
    """
    if "usi" not in merged_meta:
        return

    sdm = SearchDataManager(search_data_path)
    sdm.load()
    if not sdm.is_loaded:
        return

    usis = [str(u) if u is not None else "" for u in merged_meta["usi"]]
    search_results = sdm.get_metadata_batch(usis, columns=list(_SEARCH_DATA_COLUMNS.keys()))

    for column, meta_key in _SEARCH_DATA_COLUMNS.items():
        values = np.array(
            [r.get(column) if _is_valid(r.get(column)) else None for r in search_results],
            dtype=object,
        )
        n_matched = int(np.sum([_is_valid(v) for v in values]))
        logger.info(f"search_data {meta_key}: {n_matched:,}/{len(values):,} spectra matched")
        if n_matched > 0:
            merged_meta[meta_key] = values


def compute_peptide_metadata(merged_meta: Dict[str, Any]) -> None:
    """Derive peptide-level metadata fields in place from the ``sequence`` / ``peptides`` arrays.

    Populates ``modification_types``, ``ptm_present`` and ``modification_class``
    (from ``sequence``) and ``hydrophobicity`` (from ``peptides``). Each block is
    independently guarded so a failure in one does not abort the others.

    Args:
        merged_meta: Per-spectrum metadata dict, mutated in place.
    """
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
                mods_summary = ", ".join(f"{m}={c}" for m, c in zip(unique_mods, counts, strict=False))
                logger.info(f"Modification classes found: {mods_summary}")
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


def finalize_metadata(merged_meta: Dict[str, Any], search_data_path: str) -> None:
    """Run the full post-loop metadata derivation in place.

    Chains the shared steps the extractors perform after concatenating per-batch
    metadata: search-engine fields (USI lookup) → peptide-derived fields. ``frag_type``
    is taken straight from the parquet metadata and is never overwritten here — matching
    the FM, which reads fragmentation straight from the parquet (no CSV / header derivation).

    Args:
        merged_meta: Concatenated per-spectrum metadata dict, mutated in place.
        search_data_path: Path to ``search_data.csv`` for the USI lookup.
    """
    # search_data.csv via USI lookup: search_instrument / search_project / search_detector / search_organism.
    apply_search_data_metadata(merged_meta, search_data_path)

    # Peptide-derived fields: modification_types / ptm_present / modification_class / hydrophobicity.
    compute_peptide_metadata(merged_meta)
