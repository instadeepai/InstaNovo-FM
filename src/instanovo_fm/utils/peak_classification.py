# instanovo/foundational/utils/peak_classification.py
"""Shared peak classification and spectrum quality utilities.

Provides:

1. **Peak classification** — canonical labels (b-ion, y-loss, precursor-isotope,
   custom, unannotated, etc.) and fragment group keys.
2. **Annotation parsing** — extract ion type, fragment position, and charge
   state from annotation strings.
3. **Spectrum quality metrics** — backbone cleavage coverage and fragment
   group counts, mirroring the quality gate used by
   ``TheoreticalAnalyser.analyze_quality_gate()``.

These functions are extracted from ``TheoreticalAnalyser`` so that both the
data-analysis framework and the embedding evaluation pipeline share the same
logic without duplication.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def parse_peak_label(
    feature_type: Optional[str],
    annotation: Optional[str],
    parent_annotation: Optional[str],
) -> str:
    """Classify a matched peak into a canonical detail label.

    Parameters
    ----------
    feature_type : str or None
        Feature type from conditional annotation: ``"base"``, ``"loss"``,
        ``"isotope"``, ``"precursor"``, ``"custom"``, or ``None``.
    annotation : str or None
        Matched annotation string (e.g. ``"b3+"``, ``"y5+-H2O"``,
        ``"p^2[+1]"``, ``"custom:immonium_His@110.0713"``).
    parent_annotation : str or None
        Parent annotation for losses and isotopes (e.g. ``"b3+"`` for its
        isotope ``"b3+[+1]"``).

    Returns
    -------
    str
        One of: ``"b-ion"``, ``"y-ion"``, ``"a-ion"``, ``"b-loss"``,
        ``"y-loss"``, ``"b-isotope"``, ``"y-isotope"``, ``"precursor"``,
        ``"precursor-isotope"``, ``"custom"``, ``"other"``, ``"unannotated"``.
    """
    if feature_type is None:
        return "unannotated"

    if feature_type == "custom":
        return "custom"

    inferred_type = feature_type
    ion_annotation = annotation or ""
    if feature_type in ("loss", "isotope") and parent_annotation:
        ion_annotation = parent_annotation

    if isinstance(ion_annotation, str):
        if inferred_type is None:
            if ion_annotation.startswith("p^") or ion_annotation.startswith("p-"):
                inferred_type = "precursor"
            elif "[+" in ion_annotation:
                inferred_type = "isotope"
            elif "-" in ion_annotation and any(
                loss in ion_annotation
                for loss in ("H2O", "NH3", "CO", "H3PO4")
            ):
                inferred_type = "loss"
            else:
                inferred_type = "base"
    else:
        ion_annotation = ""

    if isinstance(ion_annotation, str) and (
        ion_annotation.startswith("p^") or ion_annotation.startswith("p-")
    ):
        if inferred_type == "isotope":
            return "precursor-isotope"
        return "precursor"

    if not isinstance(ion_annotation, str) or len(ion_annotation) == 0:
        return "other"

    ion_type_char = ion_annotation[0].lower()
    suffix_map = {"loss": "-loss", "isotope": "-isotope"}
    if ion_type_char in ("b", "y"):
        suffix = suffix_map.get(inferred_type, "-ion")
        return f"{ion_type_char}{suffix}"

    if inferred_type == "loss":
        return f"{ion_type_char}-loss"
    if inferred_type == "isotope":
        return f"{ion_type_char}-isotope"
    if inferred_type == "base":
        return f"{ion_type_char}-ion"
    if inferred_type == "precursor":
        return "precursor"
    return "other"


def group_fragment_key(
    feature_type: Optional[str],
    annotation: Optional[str],
    parent_annotation: Optional[str],
) -> Optional[str]:
    """Map a peak to its fragment group key.

    Groups base peaks with their isotopes so that completeness analysis can
    evaluate whether the model captured the entire isotope envelope.

    Parameters
    ----------
    feature_type : str or None
        Feature type from conditional annotation.
    annotation : str or None
        Matched annotation string.
    parent_annotation : str or None
        Parent annotation for losses and isotopes.

    Returns
    -------
    str or None
        The group key (e.g. ``"b3+"``), or ``None`` for losses and
        unannotated peaks.
    """
    ann = annotation or ""
    ftype = feature_type

    if ftype == "custom":
        return ann if isinstance(ann, str) and len(ann) > 0 else None

    if ftype is None and isinstance(ann, str):
        if ann.startswith("p^") or ann.startswith("p-"):
            ftype = "precursor"
        elif "[+" in ann:
            ftype = "isotope"
        elif "-" in ann and any(
            loss in ann for loss in ("H2O", "NH3", "CO", "H3PO4")
        ):
            ftype = "loss"
        else:
            ftype = "base"

    if ftype == "loss":
        return None
    if ftype == "isotope":
        if parent_annotation:
            return parent_annotation
        if isinstance(ann, str) and "[+" in ann:
            return ann.split("[")[0]
        return None
    if ftype in ("base", "precursor"):
        return ann if isinstance(ann, str) and len(ann) > 0 else None
    return None


def classify_peaks_batch(
    feature_types: list,
    matched_annotations: list,
    parent_annotations: list,
) -> Tuple[List[str], List[Optional[str]]]:
    """Classify all peaks in a spectrum in one pass.

    Parameters
    ----------
    feature_types : list
        Per-peak feature types (may contain ``None``).
    matched_annotations : list
        Per-peak matched annotations (may contain ``None``).
    parent_annotations : list
        Per-peak parent annotations (may contain ``None``).

    Returns
    -------
    detail_labels : list[str]
        Canonical label for each peak.
    fragment_group_keys : list[str | None]
        Fragment group key for each peak.
    """
    n = len(feature_types)
    detail_labels: List[str] = []
    fragment_group_keys: List[Optional[str]] = []

    for i in range(n):
        ft = feature_types[i] if i < len(feature_types) else None
        ann = matched_annotations[i] if i < len(matched_annotations) else None
        pa = parent_annotations[i] if i < len(parent_annotations) else None

        detail_labels.append(parse_peak_label(ft, ann, pa))
        fragment_group_keys.append(group_fragment_key(ft, ann, pa))

    return detail_labels, fragment_group_keys


# ---------------------------------------------------------------------------
# Annotation parsing helpers
# ---------------------------------------------------------------------------

# Compiled once at module level
_ISOTOPE_BRACKET_RE = re.compile(r"\[\+\d+\]")
_POSITION_RE = re.compile(r"^[a-z](\d+)")


def extract_ion_type(annotation: Optional[str]) -> str:
    """Extract the base ion-type character from an annotation string.

    Examples::

        >>> extract_ion_type("b3+")
        'b'
        >>> extract_ion_type("y5+-NH3")
        'y'
        >>> extract_ion_type("p^2")
        'p'
        >>> extract_ion_type(None)
        'unknown'
    """
    if not annotation or not isinstance(annotation, str):
        return "unknown"
    clean = annotation.split("-")[0].split("[")[0]
    for char in clean:
        if char.isalpha():
            return char.lower()
    return "unknown"


def extract_fragment_position(annotation: Optional[str]) -> int:
    """Extract the fragment position number from an annotation string.

    Returns -1 for precursor ions, non-string inputs, or when no position
    can be parsed.

    Examples::

        >>> extract_fragment_position("b3+")
        3
        >>> extract_fragment_position("y12++")
        12
        >>> extract_fragment_position("p^2")
        -1
    """
    if not annotation or not isinstance(annotation, str):
        return -1
    if annotation.startswith("p^") or annotation.startswith("p-"):
        return -1
    match = _POSITION_RE.match(annotation.lower())
    if match:
        return int(match.group(1))
    return -1


def extract_charge_from_annotation(annotation: Optional[str]) -> int:
    """Extract the charge state from an annotation string.

    Handles two formats:

    - **Fragment ions**: count ``+`` symbols after stripping isotope
      brackets — ``"b3+"`` → 1, ``"y5++"`` → 2.
    - **Precursor ions** (PSI mzPAF ``^z`` notation): ``"p^2"`` → 2,
      ``"p-H2O^3"`` → 3.

    Returns 1 as the minimum charge.
    """
    if not annotation or not isinstance(annotation, str):
        return 1
    # PSI mzPAF '^z' notation — precursor ions
    caret_idx = annotation.rfind("^")
    if caret_idx != -1:
        charge_str = ""
        for ch in annotation[caret_idx + 1 :]:
            if ch.isdigit():
                charge_str += ch
            else:
                break
        if charge_str:
            return int(charge_str)
    # Fragment ion format — strip isotope brackets then count '+' symbols
    clean_ann = _ISOTOPE_BRACKET_RE.sub("", annotation)
    count = clean_ann.count("+")
    return max(1, count)


# ---------------------------------------------------------------------------
# Spectrum quality metrics (backbone coverage quality gate)
# ---------------------------------------------------------------------------

# N-terminal ion types: cleavage site = position
_N_TERMINAL_IONS = frozenset({"b", "a", "c"})
# C-terminal ion types: cleavage site = seq_len - position
_C_TERMINAL_IONS = frozenset({"y", "x", "z"})


def compute_spectrum_quality(
    feature_types: list,
    matched_annotations: list,
    seq_len: int,
) -> Dict[str, Any]:
    """Compute backbone coverage and fragment group metrics for a spectrum.

    This implements the same quality criteria used by
    ``TheoreticalAnalyser.calculate_fragment_group_analysis()`` and
    ``TheoreticalAnalyser.analyze_quality_gate()``:

    1. For each **base** fragment peak, extract ``(ion_type, position)``.
    2. Group by ``(ion_type, position)`` → **fragment groups**.
    3. Map each group to a **cleavage site** (N-terminal ions map to
       ``site = position``; C-terminal ions map to
       ``site = seq_len - position``).
    4. ``backbone_coverage = n_cleavage_sites / max_cleavage_sites``.

    Parameters
    ----------
    feature_types : list
        Per-peak feature types (``"base"``, ``"loss"``, ``"isotope"``,
        ``"precursor"``, ``"custom"``, or ``None``).
    matched_annotations : list
        Per-peak matched annotation strings.
    seq_len : int
        Peptide sequence length (number of amino-acid residues).

    Returns
    -------
    dict
        ``backbone_coverage`` (float, 0-1), ``n_fragment_groups`` (int),
        ``n_cleavage_sites`` (int), ``max_cleavage_sites`` (int).
    """
    max_sites = max(seq_len - 1, 0)

    # Collect base fragment groups: {(ion_type, position)}
    groups: set = set()
    n = min(len(feature_types), len(matched_annotations))
    for i in range(n):
        ft = feature_types[i]
        if ft != "base":
            continue
        ann = matched_annotations[i]
        if not ann:
            continue
        ion_type = extract_ion_type(ann)
        position = extract_fragment_position(ann)
        if position < 1 or ion_type == "unknown":
            continue
        groups.add((ion_type, position))

    # Map groups to cleavage sites
    cleavage_sites: set = set()
    for ion_type, position in groups:
        if ion_type in _N_TERMINAL_IONS:
            site = position
        elif ion_type in _C_TERMINAL_IONS:
            site = seq_len - position
        else:
            continue
        if 0 < site < seq_len:
            cleavage_sites.add(site)

    coverage = len(cleavage_sites) / max_sites if max_sites > 0 else 0.0

    return {
        "backbone_coverage": coverage,
        "n_fragment_groups": len(groups),
        "n_cleavage_sites": len(cleavage_sites),
        "max_cleavage_sites": max_sites,
    }
