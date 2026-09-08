# instanovo/foundational/eval/embedding_io.py
"""Utilities for generating, saving, loading, and searching foundation model embeddings.

This module provides core functionality for:
- Generating embeddings from model and dataloader
- Saving embeddings and FAISS indices to disk
- Loading embeddings and indices from disk
- Performing similarity search on embeddings

Main functions:
- generate() - Generate embeddings and build FAISS index (optionally save with save_to parameter)
- load() - Load previously saved embeddings from disk
- search_similar() - Search for similar embeddings using FAISS index
- get_embedding_stats() - Compute statistics for embeddings
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Optional imports for HDF5 and FAISS
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    h5py = None

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

from instanovo.__init__ import console
from instanovo.common.dataset import DataProcessor
from instanovo_fm.model import FoundationModel
from instanovo.utils.colorlogging import ColorLog
from instanovo_fm.utils.hydrophobicity import compute_hydrophobicity
from instanovo_fm.utils.modifications import (
    compute_modification_types,
    compute_ptm_binary_flags,
    compute_glyco_deam_targets,
)
from instanovo_fm.utils.peak_classification import (
    classify_peaks_batch,
    compute_spectrum_quality,
)
from instanovo_fm.utils.theoretical_spectra import (
    generate_theoretical_spectrum,
    match_theoretical_to_experimental,
    match_with_conditional_features,
    detect_custom_ions,
    DEFAULT_CID_DA_TOL,
    _da_tol_for_fragmentation,
)

logger = ColorLog(console, __name__).logger


def _ion_types_for_fragmentation(frag_type: Optional[str]) -> tuple[str, ...]:
    """Determine ion types based on fragmentation method.

    Args:
        frag_type: Fragmentation type string (e.g., "CID", "HCD", "ETD", "UVPD")

    Returns:
        Tuple of ion types to use for theoretical spectrum generation
    """
    if not frag_type:
        return ("b", "y")
    ft = str(frag_type).strip().upper()
    if ft in ("CID", "HCD", "HCID"):
        return ("b", "y")
    if ft in ("ETD", "ECD"):
        return ("c", "z")
    if ft == "UVPD":
        return ("a", "b", "c", "x", "y", "z")
    return ("b", "y")


# Maps detailed labels to broader categories for signal composition summary
_LABEL_TO_CATEGORY = {
    "b-ion": "fragment_base",
    "y-ion": "fragment_base",
    "a-ion": "fragment_base",
    "b-loss": "fragment_loss",
    "y-loss": "fragment_loss",
    "a-loss": "fragment_loss",
    "b-isotope": "fragment_isotope",
    "y-isotope": "fragment_isotope",
    "a-isotope": "fragment_isotope",
    "precursor": "precursor",
    "precursor-isotope": "precursor",
    "custom": "custom",
    "unannotated": "unannotated",
    "other": "unannotated",
}


def _compute_signal_composition(
    feature_type_detail_list: list,
) -> Dict[str, Any]:
    """Compute aggregate signal composition from per-spectrum detail labels.

    Returns a dict with ``detail_counts``, ``detail_fractions``,
    ``category_counts``, and ``category_fractions``.
    """
    from collections import Counter

    detail_counter: Counter = Counter()
    for labels in feature_type_detail_list:
        if labels is None:
            continue
        detail_counter.update(labels)

    total = sum(detail_counter.values())
    if total == 0:
        return {}

    detail_fractions = {k: v / total for k, v in detail_counter.items()}

    category_counter: Counter = Counter()
    for label, count in detail_counter.items():
        cat = _LABEL_TO_CATEGORY.get(label, "unannotated")
        category_counter[cat] += count

    category_fractions = {k: v / total for k, v in category_counter.items()}

    return {
        "detail_counts": dict(detail_counter),
        "detail_fractions": detail_fractions,
        "category_counts": dict(category_counter),
        "category_fractions": category_fractions,
        "total_peaks": total,
    }


def _generate_theoretical_spectra_batch(
    metadata: Dict[str, np.ndarray],
    config: Dict[str, Any],
) -> Optional[Dict[str, np.ndarray]]:
    """Generate theoretical spectra for all samples in metadata using conditional annotation.

    This function processes all spectra in the metadata and generates theoretical
    fragment ions for each peptide sequence. It uses a two-pass conditional annotation
    strategy to reduce false positives:

    Pass 1: Match base fragment ions (b, y, etc.) without losses or isotopes
    Pass 2: For each matched base ion, conditionally check for:
        - Neutral losses (if parent ion matched)
        - Isotopes (if monoisotopic peak matched and intense enough)

    This approach dramatically reduces false positives because:
    - Neutral loss peaks cannot exist without their parent ion
    - Isotope peaks are only checked for intense monoisotopic peaks
    - Far fewer theoretical peaks need to be generated and matched

    Args:
        metadata: Metadata dictionary containing sequences, spectra, charges, etc.
        config: Theoretical spectrum configuration from evaluation config

    Returns:
        Dictionary with theoretical spectrum data to add to metadata:
            - theoretical_mz: List of theoretical m/z arrays (base ions only, object array)
            - theoretical_annotations: List of annotation strings (object array)
            - theoretical_match_mask: Boolean mask for matched peaks (object array)
            - theoretical_match_idx: Indices of matched theoretical peaks (object array)
            - n_theoretical_success: Number of successfully generated spectra
            - n_theoretical_failed: Number of failed spectra
    """
    # Extract configuration parameters with defaults
    ppm_tol = config.get('ppm_tol', 10.0)
    ion_types = config.get('ion_types', None)  # None = auto-detect
    add_losses = config.get('add_losses', False)
    add_isotopes = config.get('add_isotopes', False)
    max_isotope = config.get('max_isotope', 4)
    add_precursor = config.get('add_precursor', True)
    keep_modifications = config.get('keep_modifications', True)
    max_mz = config.get('max_mz', 2500.0)
    min_peaks_per_spectrum = config.get('min_peaks_per_spectrum', 5)

    # Conditional annotation parameters
    use_conditional_annotation = config.get('use_conditional_annotation', True)
    loss_types = tuple(config.get('loss_types', ['H2O', 'NH3']))
    isotope_intensity_threshold = config.get('isotope_intensity_threshold', 0.02)

    cid_da_tol = config.get('cid_da_tol', DEFAULT_CID_DA_TOL)
    # None disables auto-selection (use ppm for everything)

    # Custom ion detection parameters (NEW)
    add_custom_ions = config.get('add_custom_ions', False)
    custom_ions = config.get('custom_ions', None)  # None = use DEFAULT_CUSTOM_IONS

    # Find sequence field
    sequence_keys = ['sequence', 'peptides', 'peptide', 'seq']
    sequences = None
    for key in sequence_keys:
        if key in metadata:
            sequences = metadata[key]
            break

    if sequences is None:
        logger.warning("No sequence data found for theoretical spectrum generation")
        return None

    # Get spectra
    if 'spectra' not in metadata:
        logger.warning("No spectra found in metadata for theoretical spectrum generation")
        return None

    spectra = metadata['spectra']

    # Get precursor charges
    charge_keys = ['precursor_charge', 'charge', 'precursor_charge_id']
    precursor_charges = None
    for key in charge_keys:
        if key in metadata:
            precursor_charges = metadata[key]
            break

    if precursor_charges is None:
        logger.warning("No precursor charge data found. Using default charge=2")
        precursor_charges = np.full(len(sequences), 2, dtype=np.int32)

    # Get fragmentation types (optional)
    frag_type_keys = ['frag_type', 'fragmentation_type', 'fragmentation_method']
    frag_types = None
    for key in frag_type_keys:
        if key in metadata:
            frag_types = metadata[key]
            break

    # Initialize result arrays
    n_spectra = len(sequences)
    theoretical_mz_list = []
    theoretical_annotations_list = []
    theoretical_match_mask_list = []
    theoretical_match_idx_list = []
    matched_annotation_list = []  # Annotations for matched experimental peaks
    feature_type_list = []  # For conditional annotation
    parent_annotation_list = []  # For conditional annotation
    ppm_error_list = []  # Per-peak ppm errors (NaN for unmatched)
    match_metrics_list = []  # Per-spectrum match summary metrics
    feature_type_detail_list = []  # Canonical peak labels (b-ion, y-loss, etc.)
    fragment_group_key_list = []  # Fragment group keys for completeness analysis
    spectrum_quality_list = []  # Per-spectrum backbone coverage & fragment group metrics

    n_success = 0
    n_failed = 0
    n_skipped_invalid_seq = 0
    n_skipped_few_peaks = 0

    # Process each spectrum
    logger.info(f"Generating theoretical spectra for {n_spectra} samples...")
    for idx in tqdm(range(n_spectra), desc="Generating theoretical spectra", disable=not sys.stderr.isatty()):
        seq = sequences[idx]

        # Skip invalid sequences
        if seq is None or not isinstance(seq, str) or len(seq.strip()) == 0:
            theoretical_mz_list.append(None)
            theoretical_annotations_list.append(None)
            theoretical_match_mask_list.append(None)
            theoretical_match_idx_list.append(None)
            matched_annotation_list.append(None)
            feature_type_list.append(None)
            parent_annotation_list.append(None)
            ppm_error_list.append(None)
            match_metrics_list.append(None)
            feature_type_detail_list.append(None)
            fragment_group_key_list.append(None)
            spectrum_quality_list.append(None)
            n_skipped_invalid_seq += 1
            continue

        # Get spectrum data
        if spectra.ndim == 3:  # (N, L, 2) format
            spectrum = spectra[idx]  # (L, 2)
            mz = spectrum[:, 0]
            intensity = spectrum[:, 1]
        else:
            logger.warning(f"Unexpected spectra shape: {spectra.shape}")
            theoretical_mz_list.append(None)
            theoretical_annotations_list.append(None)
            theoretical_match_mask_list.append(None)
            theoretical_match_idx_list.append(None)
            matched_annotation_list.append(None)
            feature_type_list.append(None)
            parent_annotation_list.append(None)
            ppm_error_list.append(None)
            match_metrics_list.append(None)
            feature_type_detail_list.append(None)
            fragment_group_key_list.append(None)
            spectrum_quality_list.append(None)
            n_failed += 1
            continue

        # Denormalize m/z values (spectra are normalized 0-1 in model)
        mz = mz * max_mz

        # Filter out zero/padding peaks
        valid_mask = (mz > 0) & (intensity > 0)
        mz = mz[valid_mask]
        intensity = intensity[valid_mask]

        # Skip if too few peaks
        if len(mz) < min_peaks_per_spectrum:
            theoretical_mz_list.append(None)
            theoretical_annotations_list.append(None)
            theoretical_match_mask_list.append(None)
            theoretical_match_idx_list.append(None)
            matched_annotation_list.append(None)
            feature_type_list.append(None)
            parent_annotation_list.append(None)
            ppm_error_list.append(None)
            match_metrics_list.append(None)
            feature_type_detail_list.append(None)
            fragment_group_key_list.append(None)
            spectrum_quality_list.append(None)
            n_skipped_few_peaks += 1
            continue

        # Get precursor charge
        charge = int(precursor_charges[idx])
        if charge <= 0:
            charge = 2  # Default

        # Get fragmentation type and determine ion types
        frag_type = None
        if frag_types is not None and idx < len(frag_types):
            frag_type = frag_types[idx]
            if isinstance(frag_type, (bytes, np.bytes_)):
                frag_type = frag_type.decode('utf-8')

        # Determine ion types (use config if set, otherwise auto-detect)
        if ion_types is not None:
            ion_types_used = tuple(ion_types) if isinstance(ion_types, list) else ion_types
        else:
            ion_types_used = _ion_types_for_fragmentation(frag_type)

        # Auto-select Da tolerance for low-res CID; None = use ppm_tol
        da_tol = _da_tol_for_fragmentation(frag_type, cid_da_tol) if cid_da_tol is not None else None

        # Clean sequence
        seq_str = str(seq).strip()
        clean_sequence = DataProcessor.clean_peptide_for_pyopenms(
            seq_str,
            keep_modifications=keep_modifications
        )

        if clean_sequence is None:
            theoretical_mz_list.append(None)
            theoretical_annotations_list.append(None)
            theoretical_match_mask_list.append(None)
            theoretical_match_idx_list.append(None)
            matched_annotation_list.append(None)
            feature_type_list.append(None)
            parent_annotation_list.append(None)
            ppm_error_list.append(None)
            match_metrics_list.append(None)
            feature_type_detail_list.append(None)
            fragment_group_key_list.append(None)
            spectrum_quality_list.append(None)
            n_failed += 1
            continue

        # Generate theoretical spectrum and match using conditional annotation
        try:
            if use_conditional_annotation:
                # Use improved two-pass conditional annotation
                match_result = match_with_conditional_features(
                    exp_mz=mz,
                    exp_intensity=intensity,
                    peptide=clean_sequence,
                    precursor_charge=charge,
                    ppm_tol=ppm_tol,
                    da_tol=da_tol,
                    ion_types=ion_types_used,
                    max_charge=max(1, charge),
                    add_losses=add_losses,
                    loss_types=loss_types,
                    add_isotopes=add_isotopes,
                    max_isotope=max_isotope,
                    isotope_intensity_threshold=isotope_intensity_threshold,
                    add_precursor=add_precursor,
                    use_closest=False,
                )

                # Extract theoretical spectrum from match results
                # Note: Conditional annotation doesn't return full theoretical spectrum,
                # only matched peaks. For consistency, we'll generate base spectrum separately.
                theo_mz, theo_annotations = generate_theoretical_spectrum(
                    peptide=clean_sequence,
                    precursor_charge=charge,
                    ion_types=ion_types_used,
                    max_charge=max(1, charge),
                    add_losses=False,
                    add_isotopes=False,
                    add_precursor=add_precursor,
                )
            else:
                # Legacy approach: generate all features upfront
                theo_mz, theo_annotations = generate_theoretical_spectrum(
                    peptide=clean_sequence,
                    precursor_charge=charge,
                    ion_types=ion_types_used,
                    max_charge=max(1, charge),
                    add_losses=add_losses,
                    add_isotopes=add_isotopes,
                    max_isotope=max_isotope if add_isotopes else 0,
                    add_precursor=add_precursor,
                )

                # Match experimental peaks to theoretical
                match_result = match_theoretical_to_experimental(
                    exp_mz=mz,
                    exp_intensity=intensity,
                    theo_mz=theo_mz,
                    ppm_tol=ppm_tol,
                    da_tol=da_tol,
                    use_closest=False,
                    theo_annotations=theo_annotations,
                )

        except Exception as e:
            # If modified sequence fails, try without modifications
            if keep_modifications:
                clean_sequence_unmod = DataProcessor.clean_peptide_for_pyopenms(
                    seq_str,
                    keep_modifications=False
                )

                if clean_sequence_unmod is not None:
                    try:
                        if use_conditional_annotation:
                            match_result = match_with_conditional_features(
                                exp_mz=mz,
                                exp_intensity=intensity,
                                peptide=clean_sequence_unmod,
                                precursor_charge=charge,
                                ppm_tol=ppm_tol,
                                da_tol=da_tol,
                                ion_types=ion_types_used,
                                max_charge=max(1, charge),
                                add_losses=add_losses,
                                loss_types=loss_types,
                                add_isotopes=add_isotopes,
                                max_isotope=max_isotope,
                                isotope_intensity_threshold=isotope_intensity_threshold,
                                add_precursor=add_precursor,
                                use_closest=False,
                            )

                            theo_mz, theo_annotations = generate_theoretical_spectrum(
                                peptide=clean_sequence_unmod,
                                precursor_charge=charge,
                                ion_types=ion_types_used,
                                max_charge=max(1, charge),
                                add_losses=False,
                                add_isotopes=False,
                                add_precursor=add_precursor,
                            )
                        else:
                            theo_mz, theo_annotations = generate_theoretical_spectrum(
                                peptide=clean_sequence_unmod,
                                precursor_charge=charge,
                                ion_types=ion_types_used,
                                max_charge=max(1, charge),
                                add_losses=add_losses,
                                add_isotopes=add_isotopes,
                                max_isotope=max_isotope if add_isotopes else 0,
                                add_precursor=add_precursor,
                            )

                            match_result = match_theoretical_to_experimental(
                                exp_mz=mz,
                                exp_intensity=intensity,
                                theo_mz=theo_mz,
                                ppm_tol=ppm_tol,
                                da_tol=da_tol,
                                use_closest=False,
                                theo_annotations=theo_annotations,
                            )

                        clean_sequence = clean_sequence_unmod
                    except Exception as e2:
                        theoretical_mz_list.append(None)
                        theoretical_annotations_list.append(None)
                        theoretical_match_mask_list.append(None)
                        theoretical_match_idx_list.append(None)
                        matched_annotation_list.append(None)
                        feature_type_list.append(None)
                        parent_annotation_list.append(None)
                        n_failed += 1
                        continue
                else:
                    theoretical_mz_list.append(None)
                    theoretical_annotations_list.append(None)
                    theoretical_match_mask_list.append(None)
                    theoretical_match_idx_list.append(None)
                    matched_annotation_list.append(None)
                    feature_type_list.append(None)
                    parent_annotation_list.append(None)
                    n_failed += 1
                    continue
            else:
                theoretical_mz_list.append(None)
                theoretical_annotations_list.append(None)
                theoretical_match_mask_list.append(None)
                theoretical_match_idx_list.append(None)
                matched_annotation_list.append(None)
                feature_type_list.append(None)
                parent_annotation_list.append(None)
                n_failed += 1
                continue

        # Detect custom ions if enabled (NEW)
        signal_mask = match_result["mask"].copy()  # Start with theoretical fragment matches
        matched_annotation = match_result.get("matched_annotation", None)
        feature_type = match_result.get("feature_type", None)

        # Initialize matched_annotation as list if not already
        if matched_annotation is None:
            matched_annotation = [None] * len(mz)
        elif not isinstance(matched_annotation, list):
            # Convert to list if it's an array
            matched_annotation = list(matched_annotation)

        # Initialize feature_type as list if not already
        if feature_type is None:
            feature_type = [None] * len(mz)
        elif not isinstance(feature_type, list):
            # Convert to list if it's an array
            feature_type = list(feature_type)

        # Get ppm_error from match result (NaN for unmatched peaks)
        ppm_error = match_result.get("ppm_error", np.full(len(mz), np.nan))
        if not isinstance(ppm_error, np.ndarray):
            ppm_error = np.full(len(mz), np.nan)

        # Get parent_annotation from match result
        parent_annotation = match_result.get("parent_annotation", None)
        if parent_annotation is None:
            parent_annotation = [None] * len(mz)
        elif not isinstance(parent_annotation, list):
            parent_annotation = list(parent_annotation)

        # Get match metrics from match result
        metrics = match_result.get("metrics", {})
        n_custom = 0

        if add_custom_ions:
            try:
                # Detect custom ions with isotope support
                custom_result = detect_custom_ions(
                    exp_mz=mz,
                    exp_intensity=intensity,
                    custom_ions=custom_ions,  # None = use DEFAULT_CUSTOM_IONS
                    ppm_tol=ppm_tol,
                    return_details=True,  # Get matched m/z values
                    add_isotopes=add_isotopes,
                    max_isotope=max_isotope,
                    isotope_intensity_threshold=isotope_intensity_threshold,
                )

                # Merge custom ion matches into signal mask, annotations, AND feature_type
                for group_name, group_result in custom_result.items():
                    if group_result.get("found", False) and "matched_mz" in group_result:
                        matched_mz_values = group_result["matched_mz"]
                        # Get target m/z values for ppm_error computation
                        target_mz_values = group_result.get("target_mz", matched_mz_values)

                        for j, matched_mz_val in enumerate(matched_mz_values):
                            idx_matches = np.where(np.abs(mz - matched_mz_val) < 1e-6)[0]
                            if len(idx_matches) > 0:
                                peak_idx = idx_matches[0]
                                if not signal_mask[peak_idx]:  # Only count new matches
                                    signal_mask[peak_idx] = True
                                    matched_annotation[peak_idx] = f"custom:{group_name}@{matched_mz_val:.4f}"
                                    feature_type[peak_idx] = "custom"
                                    n_custom += 1
                                    # Compute ppm_error for custom ion match
                                    if j < len(target_mz_values):
                                        target_mz_val = target_mz_values[j]
                                        ppm_error[peak_idx] = (matched_mz_val - target_mz_val) / target_mz_val * 1e6

                        # Merge isotope matches if present
                        isotope_matches = group_result.get("isotope_matches", [])
                        for iso_match in isotope_matches:
                            iso_mz = iso_match.get("matched_mz")
                            if iso_mz is None:
                                continue
                            idx_matches = np.where(np.abs(mz - iso_mz) < 1e-6)[0]
                            if len(idx_matches) > 0:
                                peak_idx = idx_matches[0]
                                if not signal_mask[peak_idx]:
                                    signal_mask[peak_idx] = True
                                    parent_mz = iso_match.get("parent_mz", matched_mz_values[0] if matched_mz_values else iso_mz)
                                    iso_num = iso_match.get("isotope_num", 1)
                                    mono_ann = f"custom:{group_name}@{parent_mz:.4f}"
                                    matched_annotation[peak_idx] = f"{mono_ann}[+{iso_num}]"
                                    feature_type[peak_idx] = "isotope"
                                    parent_annotation[peak_idx] = mono_ann
                                    n_custom += 1

            except Exception as e:
                logger.debug(f"Custom ion detection failed for spectrum {idx}: {e}")

        # Update match metrics with custom ion count
        spectrum_metrics = dict(metrics)  # copy
        spectrum_metrics["n_custom"] = n_custom
        spectrum_metrics["n_matched"] = spectrum_metrics.get("n_matched", 0) + n_custom

        # Run canonical peak classification
        detail_labels, frag_keys = classify_peaks_batch(
            feature_type, matched_annotation, parent_annotation,
        )

        # Compute backbone coverage quality metrics
        # clean_sequence is available from the matching block above
        sq = compute_spectrum_quality(
            feature_type, matched_annotation, len(clean_sequence),
        )

        # Store results (use updated signal_mask that includes custom ions)
        theoretical_mz_list.append(theo_mz)
        theoretical_annotations_list.append(theo_annotations)
        theoretical_match_mask_list.append(signal_mask)
        theoretical_match_idx_list.append(match_result["match_idx"])
        matched_annotation_list.append(matched_annotation)
        feature_type_list.append(feature_type)
        parent_annotation_list.append(parent_annotation)
        ppm_error_list.append(ppm_error)
        match_metrics_list.append(spectrum_metrics)
        feature_type_detail_list.append(detail_labels)
        fragment_group_key_list.append(frag_keys)
        spectrum_quality_list.append(sq)
        n_success += 1

    # Log statistics (only non-zero issues as warnings)
    if n_failed > 0:
        logger.warning(f"Theoretical generation: {n_failed} failed")
    if n_skipped_invalid_seq > 0:
        logger.warning(f"Theoretical generation: {n_skipped_invalid_seq} skipped (invalid sequence)")
    if n_skipped_few_peaks > 0:
        logger.warning(f"Theoretical generation: {n_skipped_few_peaks} skipped (too few peaks)")

    # Compute signal composition from detailed labels
    signal_composition = _compute_signal_composition(feature_type_detail_list)

    # Convert to numpy arrays (object dtype for variable-length arrays)
    result_dict = {
        'theoretical_mz': np.array(theoretical_mz_list, dtype=object),
        'theoretical_annotations': np.array(theoretical_annotations_list, dtype=object),
        'theoretical_match_mask': np.array(theoretical_match_mask_list, dtype=object),
        'theoretical_match_idx': np.array(theoretical_match_idx_list, dtype=object),
        'matched_annotation': np.array(matched_annotation_list, dtype=object),
        'feature_type': np.array(feature_type_list, dtype=object),
        'parent_annotation': np.array(parent_annotation_list, dtype=object),
        'ppm_error': np.array(ppm_error_list, dtype=object),
        'feature_type_detail': np.array(feature_type_detail_list, dtype=object),
        'fragment_group_key': np.array(fragment_group_key_list, dtype=object),
        'match_metrics': np.array(match_metrics_list, dtype=object),
        'spectrum_quality': np.array(spectrum_quality_list, dtype=object),
        'signal_composition': signal_composition,
        'theoretical_max_mz': max_mz,
        'theoretical_ppm_tol': ppm_tol,
        'n_theoretical_success': n_success,
        'n_theoretical_failed': n_failed + n_skipped_invalid_seq + n_skipped_few_peaks,
    }

    return result_dict


def generate(
    model: FoundationModel,
    dataloader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    show_progress: bool = True,
    save_to: Optional[str | Path] = None,
    max_samples: Optional[int] = None,
    compute_confidence: bool = False,
    store_per_peak_confidence: bool = False,
    # Theoretical spectrum generation parameters
    generate_theoretical: bool = False,
    theoretical_config: Optional[Dict[str, Any]] = None,
    # Peak embeddings
    store_peak_embeddings: bool = False,
    store_pretransformer_embeddings: bool = False,
    # Embedding pooling strategy
    embedding_pooling: str = "cls",
    # Confidence pooling temperature (only used when embedding_pooling="confidence")
    confidence_temperature: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Any]:
    """Generate embeddings and build FAISS index from model and dataloader.

    This is the core function for embedding generation. It:
    1. Runs inference on all batches in the dataloader
    2. Collects embeddings and metadata
    3. Builds a FAISS index for similarity search
    4. Optionally computes per-spectrum and per-peak confidence scores
    5. Optionally generates theoretical spectra and matches them to experimental peaks
    6. Optionally extracts and stores peak-level embeddings
    7. Optionally saves results to disk

    Args:
        model: The foundation model to use for encoding
        dataloader: DataLoader providing batches of spectra
        device: Device to run inference on (defaults to model device)
        batch_size: Batch size for processing (used for metadata indexing)
        show_progress: Whether to show progress bar
        save_to: Optional directory path to save embeddings and index (None = keep in memory only)
        max_samples: Maximum number of samples to process (None = process all)
        compute_confidence: Whether to compute per-spectrum confidence scores
        store_per_peak_confidence: Whether to store per-peak confidence arrays and spectra in metadata
        generate_theoretical: Whether to generate theoretical spectra for each peptide
        theoretical_config: Configuration dict for theoretical spectrum generation (see default.yaml)
        store_peak_embeddings: Whether to extract and store peak-level embeddings (memory intensive)
        embedding_pooling: Embedding extraction method: "cls", "mean_pool", "confidence", or "both"
        confidence_temperature: Softmax temperature for confidence pooling (default=1.0)

    Returns:
        Tuple of (embeddings, metadata, faiss_index):
            - embeddings: numpy array of shape (N, D)
            - metadata: dictionary mapping metadata keys to numpy arrays
            - faiss_index: FAISS index for similarity search
    """
    if not FAISS_AVAILABLE:
        raise ImportError("faiss-cpu is required for embedding generation. Install with: pip install faiss-cpu")

    if device is None:
        device = next(model.parameters()).device

    model = model.to(device)
    model.eval()

    # Lists to collect embeddings and metadata
    all_embeddings = []
    all_metadata = []

    # Track number of samples processed
    total_samples_processed = 0

    # Check if we should compute confidence (enable if store_per_peak_confidence is requested)
    if store_per_peak_confidence:
        compute_confidence = True

    # Check if model supports confidence computation
    is_classification = hasattr(model, 'mz_task') and model.mz_task == 'classification'
    if compute_confidence and not is_classification:
        logger.warning("Confidence computation requested but model is not a classification model. Skipping.")
        compute_confidence = False
        store_per_peak_confidence = False


    # Process batches
    try:
        n_batches = len(dataloader)
        logger.info(f"Generating embeddings for {n_batches} batches...")
    except TypeError:
        logger.info("Generating embeddings...")
    iterator = tqdm(dataloader, desc="Generating embeddings", disable=not sys.stderr.isatty()) if show_progress else dataloader

    with torch.inference_mode():
        for batch_idx, batch in enumerate(iterator):
            # Check if we've reached max_samples limit
            if max_samples is not None and total_samples_processed >= max_samples:
                if show_progress:
                    logger.info(f"Reached max_samples limit ({max_samples}), stopping early")
                break
            # Move batch to device
            spectra = batch["spectra"].to(device)
            meta = batch.get("meta", {})

            # Get embeddings (and optionally peak embeddings)
            if store_peak_embeddings:
                # Use forward_with_attn to get peak embeddings + CLS token
                output = model.forward_with_attn(spectra, meta=meta, return_peak_embeddings=True)
                peak_embeddings_batch = output["peak_embeddings"]  # (B, L, D)
                peak_embeddings_pre_batch = output.get("peak_embeddings_pretransformer")  # (B, L, D)

                # Compute spectrum-level embedding using the configured pooling strategy
                if embedding_pooling in ("mean_pool",):
                    # Mean-pool peak tokens (exclude padding), then L2-normalize
                    pad_mask = (spectra.sum(dim=-1) == 0)  # (B, L) True=padding
                    valid = (~pad_mask).unsqueeze(-1).float()  # (B, L, 1)
                    pooled = (peak_embeddings_batch * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
                    embeddings = F.normalize(pooled, p=2, dim=-1)
                else:
                    # Default: use CLS token (L2-normalized by forward_with_attn)
                    embeddings = output["embeddings"]
            elif embedding_pooling in ("mean_pool", "confidence"):
                embeddings = model.encode_mean_pooled(
                    spectra, meta=meta,
                    pooling=embedding_pooling,
                    confidence_temperature=confidence_temperature,
                )
                peak_embeddings_batch = None
            else:
                embeddings = model.encode(spectra, meta=meta)
                peak_embeddings_batch = None

            # Optionally compute per-spectrum and per-peak confidence scores
            spectrum_confidence = None
            per_peak_confidence = None
            per_peak_conf_group = None
            per_peak_conf_offset = None
            predicted_mz_normalized = None
            if compute_confidence:
                try:
                    from instanovo_fm.trainer.losses import compute_classifier_confidence

                    # Forward pass to get logits
                    spectra_mask = batch.get("spectra_mask")
                    if spectra_mask is not None:
                        spectra_mask = spectra_mask.to(device)

                    preds, aux_out = model(spectra, spectra_mask=spectra_mask, meta=meta)

                    if isinstance(preds, tuple):  # Classification model
                        group_logits, offset_logits = preds

                        # Get model bin parameters
                        n_groups = int(model.n_bin_groups_tensor.item()) if hasattr(model, 'n_bin_groups_tensor') else None
                        last_group_size = int(model.last_group_size_tensor.item()) if hasattr(model, 'last_group_size_tensor') else None

                        if n_groups is not None and last_group_size is not None:
                            # Compute confidence metrics
                            confidence_dict = compute_classifier_confidence(
                                group_logits=group_logits,
                                offset_logits=offset_logits,
                                n_groups=n_groups,
                                last_group_size=last_group_size,
                            )

                            # Extract per-peak confidence scores
                            conf_joint = confidence_dict["conf_joint"]  # (B, L)
                            padding_mask = batch.get("spectra_mask", None)

                            # Store per-peak confidence if requested
                            if store_per_peak_confidence:
                                per_peak_confidence = conf_joint.cpu().numpy()  # (B, L)
                                # Also store decomposed group/offset confidence
                                per_peak_conf_group = confidence_dict["conf_group"].cpu().numpy()  # (B, L)
                                per_peak_conf_offset = confidence_dict["conf_offset"].cpu().numpy()  # (B, L)

                            # Compute per-spectrum confidence by aggregating per-peak confidences
                            if padding_mask is not None:
                                # Compute mean confidence over non-padded peaks
                                valid_mask = ~padding_mask.bool()
                                spectrum_confidence = []
                                for i in range(conf_joint.shape[0]):
                                    valid_confidences = conf_joint[i, valid_mask[i]]
                                    if len(valid_confidences) > 0:
                                        spectrum_confidence.append(valid_confidences.mean().item())
                                    else:
                                        spectrum_confidence.append(0.0)
                                spectrum_confidence = np.array(spectrum_confidence, dtype=np.float32)
                            else:
                                # No padding mask, compute mean over all peaks
                                spectrum_confidence = conf_joint.mean(dim=1).cpu().numpy()

                            # Decode predictions to m/z values if needed for theoretical peak analysis
                            if store_per_peak_confidence:
                                from instanovo_fm.trainer.utils import bin_groups_to_mz

                                # Get model parameters
                                bin_size = model.binning_strategy.bin_size if hasattr(model, 'binning_strategy') and hasattr(model.binning_strategy, 'bin_size') else None
                                group_size = int(model.bin_group_size_tensor.item()) if hasattr(model, 'bin_group_size_tensor') else 50
                                min_mz = model.min_mz if hasattr(model, 'min_mz') else 0.0
                                max_mz = model.max_mz if hasattr(model, 'max_mz') else 2500.0
                                bin_edges = model.bin_edges if hasattr(model, 'bin_edges') else None

                                # Get group predictions
                                pred_groups = group_logits.argmax(dim=-1)  # (B, L)

                                # Mask invalid offsets for last group before argmax
                                offset_logits_masked = offset_logits.clone()
                                is_last = pred_groups.eq(n_groups - 1)  # (B, L)
                                if is_last.any():
                                    b_idx, l_idx = is_last.nonzero(as_tuple=True)
                                    offset_logits_masked[b_idx, l_idx, last_group_size:] = -float("inf")

                                # Get offset predictions with masking applied
                                pred_offsets = offset_logits_masked.argmax(dim=-1)  # (B, L)

                                # Convert bin predictions to m/z values in Daltons
                                pred_mz_da = bin_groups_to_mz(pred_groups, pred_offsets, bin_size, group_size, min_mz, bin_edges=bin_edges)  # (B, L) in Da

                                # Normalize to [0, 1] range for consistency with spectra format
                                predicted_mz_normalized = pred_mz_da / max_mz
                                predicted_mz_normalized = predicted_mz_normalized.cpu().numpy()  # (B, L)

                except Exception as e:
                    logger.warning(f"Failed to compute confidence scores for batch {batch_idx}: {e}")
                    spectrum_confidence = None
                    per_peak_confidence = None
                    per_peak_conf_group = None
                    per_peak_conf_offset = None
                    predicted_mz_normalized = None

            # Convert to numpy
            embeddings_np = embeddings.cpu().numpy()
            batch_size_actual = len(embeddings_np)

            # Check if we need to trim this batch to respect max_samples
            if max_samples is not None:
                samples_remaining = max_samples - total_samples_processed
                if samples_remaining < batch_size_actual:
                    # Trim the batch to only include remaining samples
                    embeddings_np = embeddings_np[:samples_remaining]
                    batch_size_actual = samples_remaining
                    # Also trim the batch tensors for metadata collection
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor) and len(batch[key]) > samples_remaining:
                            batch[key] = batch[key][:samples_remaining]
                        elif isinstance(batch[key], list) and len(batch[key]) > samples_remaining:
                            batch[key] = batch[key][:samples_remaining]

            # Collect embeddings
            all_embeddings.append(embeddings_np)
            total_samples_processed += batch_size_actual

            # Collect all metadata from batch (already processed by collate_fn)
            # This includes both data fields and metadata columns
            batch_metadata = {}

            for key, value in batch.items():
                # Store spectra if per-peak confidence is requested (needed for confidence signal analysis)
                # Otherwise skip spectra to save memory
                if key == "spectra":
                    if store_per_peak_confidence:
                        # Store spectra for confidence signal analysis
                        batch_metadata[key] = value.cpu().numpy()
                    continue

                # Convert to numpy/list as appropriate
                if isinstance(value, torch.Tensor):
                    batch_metadata[key] = value.cpu().numpy()
                elif isinstance(value, list):
                    # Handle list of strings (like peptide sequences) or list of tensors
                    # Check if list contains tensors
                    if value and isinstance(value[0], torch.Tensor):
                        # Convert each tensor to numpy
                        batch_metadata[key] = np.array([v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in value], dtype=object)
                    else:
                        # Regular list (strings, numbers, etc.)
                        batch_metadata[key] = np.array(value, dtype=object)
                elif isinstance(value, dict):
                    # Handle nested dictionaries (like 'meta' field)
                    # Flatten dictionary into separate metadata keys with prefix
                    for k, v in value.items():
                        prefixed_key = f"{key}_{k}"
                        if isinstance(v, torch.Tensor):
                            v_np = v.cpu().numpy()
                            # Ensure consistent 1D shape across batches
                            if v_np.ndim == 0:
                                # Scalar tensor - expand to 1D array
                                batch_metadata[prefixed_key] = np.full(batch_size_actual, v_np.item())
                            elif v_np.ndim == 1:
                                # 1D tensor - use as is (ensure correct batch size)
                                if len(v_np) == batch_size_actual:
                                    batch_metadata[prefixed_key] = v_np
                                else:
                                    # Size mismatch - trim or pad
                                    batch_metadata[prefixed_key] = v_np[:batch_size_actual]
                            elif v_np.ndim == 2 and v_np.shape[1] == 1:
                                # 2D tensor with shape (batch, 1) - squeeze to 1D
                                batch_metadata[prefixed_key] = v_np.squeeze(axis=1)[:batch_size_actual]
                            else:
                                # Multi-dimensional tensor - flatten first dimension only
                                # This handles cases like (batch, features) -> keep as 2D but ensure batch size
                                batch_metadata[prefixed_key] = v_np[:batch_size_actual]
                        else:
                            # Non-tensor value - broadcast to batch size
                            batch_metadata[prefixed_key] = np.full(batch_size_actual, v, dtype=object)
                else:
                    batch_metadata[key] = value

            # Add batch index for tracking
            batch_metadata["batch_idx"] = np.full(batch_size_actual, batch_idx)
            batch_metadata["sample_idx"] = np.arange(batch_size_actual) + batch_idx * batch_size

            # Add spectrum confidence if computed
            if spectrum_confidence is not None:
                # Trim confidence array if batch was trimmed due to max_samples
                if len(spectrum_confidence) > batch_size_actual:
                    spectrum_confidence = spectrum_confidence[:batch_size_actual]
                batch_metadata["spectrum_confidence"] = spectrum_confidence

            # Add per-peak confidence if computed
            if per_peak_confidence is not None:
                # Trim confidence array if batch was trimmed due to max_samples
                if len(per_peak_confidence) > batch_size_actual:
                    per_peak_confidence = per_peak_confidence[:batch_size_actual]
                batch_metadata["per_peak_confidence"] = per_peak_confidence
                # Store decomposed group/offset confidence for analysis
                if per_peak_conf_group is not None:
                    if len(per_peak_conf_group) > batch_size_actual:
                        per_peak_conf_group = per_peak_conf_group[:batch_size_actual]
                    batch_metadata["per_peak_conf_group"] = per_peak_conf_group
                if per_peak_conf_offset is not None:
                    if len(per_peak_conf_offset) > batch_size_actual:
                        per_peak_conf_offset = per_peak_conf_offset[:batch_size_actual]
                    batch_metadata["per_peak_conf_offset"] = per_peak_conf_offset

            # Add predicted m/z values if computed (for theoretical peak analysis)
            if predicted_mz_normalized is not None:
                # Trim prediction array if batch was trimmed due to max_samples
                if len(predicted_mz_normalized) > batch_size_actual:
                    predicted_mz_normalized = predicted_mz_normalized[:batch_size_actual]
                batch_metadata["predicted_mz_normalized"] = predicted_mz_normalized

            # Add peak embeddings if extracted
            if peak_embeddings_batch is not None:
                # Convert to numpy and trim if needed
                peak_emb_np = peak_embeddings_batch.cpu().numpy()
                if len(peak_emb_np) > batch_size_actual:
                    peak_emb_np = peak_emb_np[:batch_size_actual]
                batch_metadata["peak_embeddings"] = peak_emb_np

                # Store pre-transformer peak embeddings (opt-in, doubles peak memory)
                if store_pretransformer_embeddings and peak_embeddings_pre_batch is not None:
                    pre_emb_np = peak_embeddings_pre_batch.cpu().numpy()
                    if len(pre_emb_np) > batch_size_actual:
                        pre_emb_np = pre_emb_np[:batch_size_actual]
                    batch_metadata["peak_embeddings_pretransformer"] = pre_emb_np

            all_metadata.append(batch_metadata)


    # Concatenate all embeddings and metadata
    embeddings_array = np.concatenate(all_embeddings, axis=0)

    # Merge metadata dictionaries
    # Collect all unique keys across all batches
    all_keys = set()
    for md in all_metadata:
        all_keys.update(md.keys())

    merged_metadata = {}
    for key in all_keys:
        # Collect values for this key from all batches (skip if missing)
        values_to_concat = []
        for md in all_metadata:
            if key in md:
                values_to_concat.append(md[key])

        # Only concatenate if all batches have this key
        if len(values_to_concat) == len(all_metadata):
            try:
                merged_metadata[key] = np.concatenate(values_to_concat, axis=0)
            except ValueError as e:
                # If concatenation fails due to shape mismatch, store as object array
                logger.warning(f"Failed to concatenate metadata key '{key}': {e}. Storing as object array.")
                merged_metadata[key] = np.array([item for batch in values_to_concat for item in batch], dtype=object)
        else:
            # Key not present in all batches - skip it
            logger.warning(f"Metadata key '{key}' not present in all batches. Skipping.")

    # Compute hydrophobicity from peptide sequences if available.
    # Try 'peptides' first (may be tokenized integer tensors, in which case
    # compute_hydrophobicity returns None), then fall back to 'sequence' (raw
    # string column present when _keep_non_tensor_metadata=True on the processor).
    for _seq_key in ('peptides', 'sequence'):
        if _seq_key in merged_metadata and 'hydrophobicity' not in merged_metadata:
            try:
                hydrophobicity = compute_hydrophobicity(merged_metadata[_seq_key])
                if hydrophobicity is not None:
                    merged_metadata['hydrophobicity'] = hydrophobicity
                    logger.debug(f"Computed hydrophobicity from '{_seq_key}' for {len(hydrophobicity)} entries")
            except Exception as e:
                logger.warning(f"Failed to compute hydrophobicity from '{_seq_key}': {e}")

    # Extract search_project from USI if available (needed by LinearProbeTask for project-disjoint splitting)
    if 'usi' in merged_metadata and 'search_project' not in merged_metadata:
        try:
            usis = merged_metadata['usi']
            # Debug: log USI dtype and sample values to diagnose format issues
            logger.info(f"USI metadata: dtype={type(usis).__name__}, "
                        f"element_type={type(usis[0]).__name__ if len(usis) > 0 else 'empty'}, "
                        f"sample={repr(usis[0])[:120] if len(usis) > 0 else 'N/A'}")
            def _extract_project(u):
                """Extract project from USI, handling scalar strings and nested arrays."""
                if u is None:
                    return ""
                if isinstance(u, np.ndarray):
                    u = u.item() if u.ndim == 0 else u.flat[0]
                s = str(u)
                parts = s.split(":")
                return parts[1].strip() if len(parts) >= 2 else ""
            projects = np.array([_extract_project(u) for u in usis], dtype=object)
            merged_metadata['search_project'] = projects
            n_valid = int(np.sum(projects != ""))
            unique_projects = set(p for p in projects if p)
            logger.info(f"Extracted search_project: {n_valid}/{len(projects)} valid, "
                        f"{len(unique_projects)} unique projects, "
                        f"samples={list(unique_projects)[:5]}")
        except Exception as e:
            logger.warning(f"Failed to extract search_project from USI: {e}")
            import traceback
            traceback.print_exc()

    # Compute modification types from sequence field if available
    if 'sequence' in merged_metadata:
        try:
            sequences = merged_metadata['sequence']

            # Vectorized cleaning: process all sequences at once
            # Pre-allocate array for better memory efficiency
            cleaned_sequences = np.empty(len(sequences), dtype=object)

            # Vectorized check for valid sequences
            is_valid = np.array([
                seq is not None and isinstance(seq, str) and len(seq.strip()) > 0
                for seq in sequences
            ], dtype=bool)

            # Clean only valid sequences (avoid unnecessary function calls)
            for i, seq in enumerate(sequences):
                if is_valid[i]:
                    cleaned_sequences[i] = DataProcessor.clean_peptide_for_pyopenms(seq, keep_modifications=True)
                else:
                    cleaned_sequences[i] = None

            # Extract modification types from cleaned sequences
            modification_types = compute_modification_types(cleaned_sequences, use_modified_peptide=True)

            if modification_types is not None and len(modification_types) > 0:
                merged_metadata['modification_types'] = modification_types

                # Vectorized binary PTM presence field creation
                # Single vectorized comparison instead of loop
                is_modified = modification_types != 'Unmodified'
                ptm_present = is_modified.astype(np.int32)
                merged_metadata['ptm_present'] = ptm_present
                num_modified = np.sum(is_modified)

                # Per-modification binary probe targets (phospho / glyco),
                # detected from the raw sequence strings (UNIMOD IDs in LCFM,
                # residue-mass tokens as a fallback).
                ptm_flags = compute_ptm_binary_flags(sequences)
                merged_metadata['mod_phospho'] = ptm_flags['mod_phospho']
                merged_metadata['mod_glyco'] = ptm_flags['mod_glyco']
                # Glyco-subtype (multiclass) and deamidation-of-N (binary) targets
                extra = compute_glyco_deam_targets(sequences)
                merged_metadata['mod_deam_n'] = extra['mod_deam_n']
                merged_metadata['glyco_class'] = extra['glyco_class']
                logger.debug(
                    f"Per-mod flags: phospho={int(ptm_flags['mod_phospho'].sum())}, "
                    f"glyco={int(ptm_flags['mod_glyco'].sum())}, "
                    f"deam_n={int(extra['mod_deam_n'].sum())} / {len(sequences)}"
                )

                logger.debug(f"Modification types: {num_modified}/{len(modification_types)} modified")

                # Vectorized multi-class modification type field creation
                # Use numpy operations instead of Python loops

                # Vectorized check for multi-modifications (contains ' + ')
                has_multiple = np.array([' + ' in str(mod) for mod in modification_types], dtype=bool)

                # Initialize with modification types
                single_mods = modification_types.copy()

                # Assign multi-modifications to "Other" (vectorized)
                single_mods[has_multiple] = 'Other'

                # Count occurrences (excluding Unmodified and Other)
                unique_mods, counts = np.unique(single_mods, return_counts=True)

                # Vectorized filtering
                mask = (unique_mods != 'Unmodified') & (unique_mods != 'Other')
                ranked_mods = unique_mods[mask]
                ranked_counts = counts[mask]

                if len(ranked_mods) > 0:
                    # Sort by count (descending) and take top 6
                    sorted_indices = np.argsort(ranked_counts)[::-1]
                    top_6_mods = set(ranked_mods[sorted_indices[:6]])  # Use set for O(1) lookup

                    # Vectorized multi-class label assignment using numpy where
                    modification_class = np.where(
                        single_mods == 'Unmodified',
                        'Unmodified',
                        np.where(
                            np.isin(single_mods, list(top_6_mods)),
                            single_mods,
                            'Other'
                        )
                    )

                    merged_metadata['modification_class'] = modification_class

                    # Log class distribution
                    class_unique, class_counts = np.unique(modification_class, return_counts=True)
                    logger.debug(f"Modification classes: {len(class_unique)} types")
                    for cls, cnt in zip(class_unique, class_counts):
                        logger.debug(f"  - {cls}: {cnt} ({cnt/len(modification_class)*100:.1f}%)")
                else:
                    # No modifications found, only Unmodified
                    merged_metadata['modification_class'] = single_mods

        except Exception as e:
            logger.warning(f"Failed to extract modification types: {e}")

    # Generate theoretical spectra if requested
    if generate_theoretical and theoretical_config is not None:
        theoretical_results = _generate_theoretical_spectra_batch(
            metadata=merged_metadata,
            config=theoretical_config,
        )

        if theoretical_results is not None:
            # Add theoretical spectrum data to metadata
            merged_metadata.update(theoretical_results)
            n_ok = theoretical_results.get('n_theoretical_success', 0)
            n_fail = theoretical_results.get('n_theoretical_failed', 0)
            logger.info(f"Theoretical spectra: {n_ok} generated, {n_fail} failed")

    # Build FAISS index for similarity search
    dimension = embeddings_array.shape[1]

    # Create FAISS index using inner product (cosine similarity for normalized vectors)
    index = faiss.IndexFlatIP(dimension)

    # Normalize embeddings for cosine similarity
    embeddings_normalized = embeddings_array.astype(np.float32)
    faiss.normalize_L2(embeddings_normalized)

    # Verify normalization
    norms = np.linalg.norm(embeddings_normalized, axis=1)
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    assert np.allclose(mean_norm, 1.0, atol=1e-3), f"Embeddings not properly normalized: mean_norm={mean_norm}"

    # Add embeddings to index
    index.add(embeddings_normalized)

    logger.info(f"Embeddings ready: {embeddings_array.shape}, FAISS index: {index.ntotal} vectors")

    # Optionally save to disk
    if save_to is not None:
        save(embeddings_array, merged_metadata, index, save_to, embedding_pooling=embedding_pooling)

    return embeddings_array, merged_metadata, index


def save(
    embeddings: np.ndarray,
    metadata: Dict[str, np.ndarray],
    faiss_index: Any,
    out_dir: str | Path,
    embedding_pooling: str = "cls",
) -> Tuple[str, str]:
    """Save embeddings, metadata, and FAISS index to disk.

    Args:
        embeddings: Embeddings array of shape (N, D)
        metadata: Dictionary mapping metadata keys to numpy arrays
        faiss_index: FAISS index for similarity search
        out_dir: Directory to save files to
        embedding_pooling: Pooling strategy used to generate these embeddings

    Returns:
        Tuple of (embeddings_path, index_path)
    """
    if not H5PY_AVAILABLE:
        raise ImportError("h5py is required for saving embeddings. Install with: pip install h5py")
    if not FAISS_AVAILABLE:
        raise ImportError("faiss-cpu is required for saving FAISS index. Install with: pip install faiss-cpu")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = out_dir / "embeddings.h5"
    index_path = out_dir / "index.faiss"

    # Save embeddings and metadata to HDF5
    logger.info(f"Saving embeddings to {embeddings_path}")
    with h5py.File(embeddings_path, 'w') as f:
        # Save embeddings
        f.create_dataset('embeddings', data=embeddings, compression='gzip', compression_opts=9)

        # Save metadata
        meta_group = f.create_group('metadata')
        for key, value in metadata.items():
            # Handle string arrays (like peptide sequences) with special dtype
            if value.dtype == object or value.dtype.kind in ('U', 'S'):
                string_list = [str(v) for v in value]
                dt = h5py.string_dtype(encoding='utf-8')
                meta_group.create_dataset(key, data=string_list, dtype=dt, compression='gzip', compression_opts=9)
            else:
                meta_group.create_dataset(key, data=value, compression='gzip', compression_opts=9)

        # Save metadata attributes
        f.attrs['num_embeddings'] = len(embeddings)
        f.attrs['embedding_dim'] = embeddings.shape[1]
        f.attrs['metadata_keys'] = list(metadata.keys())
        f.attrs['embedding_pooling'] = embedding_pooling

    # Save FAISS index
    logger.info(f"Saving FAISS index to {index_path}")
    faiss.write_index(faiss_index, str(index_path))

    logger.info(f"Save complete:")
    logger.info(f"  - Embeddings: {embeddings_path} ({embeddings.shape})")
    logger.info(f"  - FAISS index: {index_path} ({faiss_index.ntotal} vectors)")

    return str(embeddings_path), str(index_path)


def load(out_dir: str | Path) -> Tuple[np.ndarray, Dict[str, np.ndarray], Any]:
    """Load embeddings, metadata, and FAISS index from disk.

    Args:
        out_dir: Directory containing embeddings.h5 and index.faiss

    Returns:
        Tuple of (embeddings, metadata, faiss_index):
            - embeddings: numpy array of shape (N, D)
            - metadata: dictionary mapping metadata keys to numpy arrays
            - faiss_index: FAISS index for similarity search
    """
    if not H5PY_AVAILABLE:
        raise ImportError("h5py is required for embedding loading. Install with: pip install h5py")
    if not FAISS_AVAILABLE:
        raise ImportError("faiss-cpu is required for embedding loading. Install with: pip install faiss-cpu")

    out_dir = Path(out_dir)
    embeddings_path = out_dir / "embeddings.h5"
    index_path = out_dir / "index.faiss"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index file not found: {index_path}")

    # Load embeddings and metadata
    logger.info(f"Loading embeddings from {embeddings_path}")
    with h5py.File(embeddings_path, 'r') as f:
        embeddings = f['embeddings'][:]

        # Load metadata
        metadata = {}
        meta_group = f['metadata']
        for key in meta_group.keys():
            arr = meta_group[key][:]
            # h5py returns bytes for string_dtype datasets; decode to str so
            # downstream code (which checks isinstance(v, str)) works correctly.
            if arr.dtype == object and len(arr) > 0 and isinstance(arr.flat[0], (bytes, np.bytes_)):
                arr = np.array([v.decode('utf-8') if isinstance(v, (bytes, np.bytes_)) else v for v in arr], dtype=object)
            metadata[key] = arr

    # Load FAISS index
    logger.info(f"Loading FAISS index from {index_path}")
    faiss_index = faiss.read_index(str(index_path))

    logger.info(f"Load complete:")
    logger.info(f"  - Embeddings: {embeddings.shape}")
    logger.info(f"  - Index: {faiss_index.ntotal} vectors")
    logger.info(f"  - Metadata keys: {list(metadata.keys())}")

    return embeddings, metadata, faiss_index


def search_similar(
    query_embedding: np.ndarray,
    faiss_index: Any,
    k: int = 10,
    return_distances: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Search for similar embeddings using FAISS index.

    Args:
        query_embedding: Query embedding (1D or 2D array)
        faiss_index: FAISS index to search in
        k: Number of nearest neighbors to return
        return_distances: Whether to return similarity scores

    Returns:
        Tuple of (indices, distances):
            - indices: array of shape (n_queries, k) with indices of nearest neighbors
            - distances: array of shape (n_queries, k) with similarity scores (or None)
    """
    if not FAISS_AVAILABLE:
        raise ImportError("faiss-cpu is required for similarity search. Install with: pip install faiss-cpu")

    # Ensure query is 2D
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Ensure query is float32 and normalized
    query_embedding = query_embedding.astype(np.float32)
    faiss.normalize_L2(query_embedding)

    # Search
    if return_distances:
        distances, indices = faiss_index.search(query_embedding, k)
        return indices, distances
    else:
        indices = faiss_index.search(query_embedding, k)[1]
        return indices, None


def get_embedding_stats(embeddings: np.ndarray) -> Dict[str, float]:
    """Compute statistics for embeddings.

    Args:
        embeddings: Embeddings array of shape (N, D)

    Returns:
        Dictionary containing embedding statistics (norms, dimensions, etc.)
    """
    norms = np.linalg.norm(embeddings, axis=1)

    stats = {
        'num_embeddings': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'mean_norm': float(np.mean(norms)),
        'std_norm': float(np.std(norms)),
        'min_norm': float(np.min(norms)),
        'max_norm': float(np.max(norms)),
        'mean_embedding_norm': float(np.linalg.norm(np.mean(embeddings, axis=0))),
    }

    return stats
