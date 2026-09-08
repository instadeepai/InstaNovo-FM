"""Helper utilities for IG attribution analyses.

Provides:
- PredictionTarget: Captum-compatible forward wrapper for IG (full group MLM-masking)
- Fragment ion group building from theoretical annotations
- Top-k attribution analysis (per-peak chemical identity classification)
- PA bias synthetic sweep utilities
- Paper-quality visualization functions
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.axes import Axes

# ---------------------------------------------------------------------------
# Attribution reduction (moved from legacy xai_peaks_helper.py)
# ---------------------------------------------------------------------------


def reduce_attributions_to_peaks(
    attributions: torch.Tensor,
    reduction: str = "abs_sum",
) -> torch.Tensor:
    """Reduce per-channel attributions to per-peak importance scores.

    Args:
        attributions: Attribution tensor (B, L, C) where C is channels (m/z, intensity, ...)
        reduction: Reduction method - "abs_sum", "sum", "l2", "max"

    Returns:
        Per-peak scores (B, L)
    """
    if reduction == "abs_sum":
        return attributions.abs().sum(dim=-1)
    elif reduction == "sum":
        return attributions.sum(dim=-1)
    elif reduction == "l2":
        return attributions.norm(dim=-1)
    elif reduction == "max":
        return attributions.abs().max(dim=-1).values
    else:
        raise ValueError(f"Unknown reduction method: {reduction}")


# ---------------------------------------------------------------------------
# Constants — Chemical reference masses
# ---------------------------------------------------------------------------

PROTON_MASS = 1.007276  # Da
WATER_MASS = 18.010565  # Da
NH3_MASS = 17.026549  # Da
CO_MASS = 27.994915  # Da (b→a ion)
H3PO4_MASS = 97.976895  # Da (phosphorylation loss)
ISOTOPE_SPACING = 1.003355  # 13C - 12C mass difference

# Standard amino acid residue masses (monoisotopic)
AMINO_ACID_MASSES: Dict[str, float] = {
    "G": 57.02146,
    "A": 71.03711,
    "S": 87.03203,
    "P": 97.05276,
    "V": 99.06841,
    "T": 101.04768,
    "C": 103.00919,
    "L/I": 113.08406,
    "N": 114.04293,
    "D": 115.02694,
    "Q": 128.05858,
    "K": 128.09496,
    "E": 129.04259,
    "M": 131.04049,
    "H": 137.05891,
    "F": 147.06841,
    "R": 156.10111,
    "Y": 163.06333,
    "W": 186.07931,
}

NEUTRAL_LOSS_MASSES: Dict[str, float] = {
    "H2O": WATER_MASS,
    "NH3": NH3_MASS,
    "CO": CO_MASS,
    "H3PO4": H3PO4_MASS,
}

AA_MASS_LIST = sorted(AMINO_ACID_MASSES.values())


# ---------------------------------------------------------------------------
# Attribution categories
# ---------------------------------------------------------------------------

# Structural categories (mutually exclusive — every ranked peak is exactly one)
# Ordered from "most informative" (direct ladder) to "least informative" (unannotated).
# charge_variant_leakage is a non-chemistry shortcut; extended-chemistry types
# cover unannotated peaks matched by the novel-chemistry probe.
STRUCTURAL_CATEGORIES: Tuple[str, ...] = (
    "ladder_neighbor",
    "near_ladder",
    "distant_same_series",
    "complementary_pair",
    "opposite_series",
    "precursor",
    "other_annotated",
    "charge_variant_leakage",
    "internal_fragment",
    "d_ion",
    "w_ion",
    "side_chain_loss",
    "immonium_related",
    "precursor_combined_loss",
    "unannotated",
)

# Subset that represents a novel-chemistry match on an otherwise-unannotated peak.
EXTENDED_CHEMISTRY_CATEGORIES: Tuple[str, ...] = (
    "internal_fragment",
    "d_ion",
    "w_ion",
    "side_chain_loss",
    "immonium_related",
    "precursor_combined_loss",
)


# ---------------------------------------------------------------------------
# Tuning constants (not user-tunable — change here or add as a task kwarg)
# ---------------------------------------------------------------------------

# Shift distances (Da) for the novel-chemistry shift-null model. These are
# fixed offsets that land *between* common fragment masses, so a high match
# rate after shifting indicates the library matches anything in the m/z
# range rather than genuine chemistry.
SHIFT_NULL_DISTANCES: Tuple[float, ...] = (3.7, 7.3, 11.1)

# Charge-variant detection tolerance (Da) — at ISOTOPE_SPACING/2 ≈ 0.5017 Da
# a z=2 base and its +1 isotope are only 0.5 apart, so the tolerance uses
# ``<=`` in :func:`detect_charge_variant_indices` to catch the boundary.
CHARGE_VARIANT_TOLERANCE_DA: float = 0.5

# Top-k ranking caps: classify up to ``_CAP`` peaks per group for the top-1
# distribution; emit up to ``_EMIT`` to the hero JSON for visualization.
TOP_K_RANKED_PEAKS_CAP: int = 50
TOP_K_RANKED_PEAKS_EMIT: int = 15


# Parses b/y/a ion annotations like "b3+", "y11++", "b3+-H2O", "y5+[+1]".
# Precursor ("p^2"), custom immonium ("custom:immonium_Tyr@..."), etc. return None.
_SERIES_ANNOTATION_RE = re.compile(r"^([aby])(\d+)(\++)")


def _parse_series_info(annotation: Optional[str]) -> Optional[Tuple[str, int, int]]:
    """Parse a backbone ion annotation into ``(ion_type, position, charge)``.

    Strips isotope (``[+N]``) and neutral-loss (``-H2O`` etc.) suffixes before
    matching. Returns ``None`` for unparseable strings (precursor, immonium,
    custom labels, etc.).
    """
    if not annotation:
        return None
    ann = str(annotation).strip()
    if not ann or ann == "None":
        return None
    # Strip isotope suffix first, then neutral-loss suffix — either may follow
    # the charge-state '+' characters (e.g. "y10+[+1]", "b6+-H2O").
    ann = ann.split("[", 1)[0]
    ann = ann.split("-", 1)[0]
    match = _SERIES_ANNOTATION_RE.match(ann)
    if match is None:
        return None
    return match.group(1), int(match.group(2)), match.group(3).count("+")


def _gap_residues_from_sequence(
    sequence_residues: List[str],
    ion_type: str,
    masked_position: int,
    peak_position: int,
) -> Optional[List[str]]:
    """Return the residues between two same-series positions on the backbone.

    For b-ions: b_i covers residues[0:i], so b_j → b_i (j>i) adds residues[i:j].
    For y-ions: y_i covers residues[L-i:], so y_j → y_i (j>i) prepends residues[L-j:L-i].
    Returns ``None`` if the inputs are inconsistent with the sequence length.
    """
    if ion_type not in ("a", "b", "y"):
        return None
    L = len(sequence_residues)  # noqa: N806
    if L == 0:
        return None
    lo, hi = sorted((masked_position, peak_position))
    if hi > L or lo < 0:
        return None
    if ion_type in ("a", "b"):
        return [str(r)[0] for r in sequence_residues[lo:hi]]
    # y-ion: spans are counted from the C-terminus
    start = max(L - hi, 0)
    end = min(L - lo, L)
    return [str(r)[0] for r in sequence_residues[start:end]]


def expected_mz_at_other_charges(
    base_mz: float,
    base_charge: int,
    max_charge: int = 4,
) -> Dict[int, float]:
    """Return the expected m/z for every charge state except the base charge.

    Return ``{charge_state: expected_mz}`` for every charge in
    ``1..max_charge`` **except** ``base_charge``, using neutral-mass
    equivalence (``neutral = mz*z − z*proton``).

    Single source of truth for charge-state arithmetic — shared by the
    charge-variant detector and the hero-report charge-state-variant builder
    so they can never disagree on what a variant looks like.
    """
    if base_charge <= 0 or base_mz <= 0:
        return {}
    neutral = base_mz * base_charge - base_charge * PROTON_MASS
    return {z: (neutral + z * PROTON_MASS) / z for z in range(1, max_charge + 1) if z != base_charge}


def detect_charge_variant_indices(
    masked_base_mz: float,
    masked_charge: int,
    peak_mz_array: np.ndarray,
    tol_da: float = CHARGE_VARIANT_TOLERANCE_DA,
    max_charge: int = 4,
    extra_masked_mz: Optional[List[float]] = None,
) -> Set[int]:
    """Return visible peak indices whose m/z matches the masked group at a different charge state.

    Tolerance uses ``<=`` so peaks exactly at the boundary (e.g. a z=2 +1 isotope sitting exactly 0.5 Da off the base) are still caught. Args:
    extra_masked_mz: additional reference m/z values (same masked_charge) — e.g. the m/z of the masked group's isotopes and neutral-loss siblings.
    Without these the detector only checks the base ion, which misses variants of siblings like y8+-H2O → y8++-H2O.
    """
    if masked_charge <= 0 or masked_base_mz <= 0:
        return set()
    reference_mzs = [masked_base_mz]
    if extra_masked_mz:
        reference_mzs.extend(m for m in extra_masked_mz if m > 0)
    expected_mz: List[float] = []
    for ref_mz in reference_mzs:
        expected_mz.extend(expected_mz_at_other_charges(ref_mz, masked_charge, max_charge).values())
    out: Set[int] = set()
    for idx, peak_mz in enumerate(peak_mz_array):
        if peak_mz <= 0:
            continue
        for exp in expected_mz:
            if abs(float(peak_mz) - exp) <= tol_da:
                out.add(idx)
                break
    return out


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FragmentGroup:
    """A fragment ion group: base ion + its isotopes + neutral losses.

    The model treats these as a single chemical entity — attribution should
    be aggregated across the group.
    """

    group_key: str  # e.g. "b3", "y5" — the parent annotation
    peak_indices: List[int]  # all peak indices belonging to this group
    base_idx: Optional[int]  # index of the base (monoisotopic) peak, if identifiable
    base_mz: float  # m/z of the base peak (or mean m/z of the group)
    ion_type: str  # "b", "y", "precursor", or "unknown"


@dataclass
class PeakPrediction:
    """Prediction result for a single peak within a masked fragment group."""

    peak_idx: int
    annotation: str  # e.g. "b4+", "b4+[+1]", "b4+-H2O"
    true_mz_da: float
    predicted_mz_da: float
    correct_bin: bool
    log_prob: float  # log-prob of predicted bin (confidence)
    correct_bin_rank: int  # rank of the correct bin in the model's predictions (1-indexed, -1 if N/A)
    # Raw bin indices for tracing
    predicted_group_bin: int = -1
    predicted_offset_bin: int = -1
    true_group_bin: int = -1
    true_offset_bin: int = -1
    group_bin_distance: int = 0  # |predicted_group - true_group| (how far off)
    # Top-N candidates: list of (group_bin, offset_bin, mz, log_prob)
    top_candidates: List[Tuple[int, int, float, float]] = field(default_factory=list)


@dataclass
class PredictionInfo:
    """Model prediction quality for a masked fragment group."""

    predicted_mz_da: float  # predicted m/z of base ion in Daltons
    true_mz_da: float  # true m/z of base ion in Daltons
    error_da: float  # |predicted - true| in Da (base ion)
    error_ppm: float  # |predicted - true| / true * 1e6 (base ion)
    confidence: float  # mean log-prob across all masked positions
    correct_bin: bool = False  # whether base ion bin matches
    predicted_group: int = -1
    true_group: int = -1
    predicted_offset: int = -1
    true_offset: int = -1
    group_bin_accuracy: float = -1.0  # fraction of group members with correct bin
    peak_predictions: List[PeakPrediction] = field(default_factory=list)  # per-peak results


@dataclass
class AttributionResult:
    """Per-masked-group IG attribution result.

    MLM-masks the entire fragment group (training-aligned) and computes
    Integrated Gradients to identify which visible peaks the model uses.
    """

    spectrum_idx: int
    masked_group: FragmentGroup
    attributions: np.ndarray  # per-peak IG attribution (L,)
    prediction: Optional[PredictionInfo] = None
    convergence_delta: float = 0.0  # IG completeness axiom
    topk: Optional["TopKAnalysis"] = None
    # Peak indices that are the masked fragment observed at a different charge
    # state — classified as leakage and excluded from the novel-chemistry
    # denominator so charge shortcuts don't contaminate chemistry metrics.
    charge_variant_indices: Set[int] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Fragment ion group building
# ---------------------------------------------------------------------------


def build_fragment_groups(
    annotations: List[str],
    parent_annotations: Optional[List[str]],
    feature_types: Optional[List[str]],
    mz: np.ndarray,
    intensity: np.ndarray,
) -> List[FragmentGroup]:
    """Build fragment ion groups from peak annotations.

    Groups peaks by their base ion identity. A group contains the base ion
    plus all its isotopes and neutral losses.

    The grouping key is:
    - For base ions (feature_type="base", parent_annotation=None):
      the matched_annotation itself (e.g. "b3+")
    - For children (feature_type="isotope"/"loss", parent_annotation="b3+"):
      the parent_annotation value (e.g. "b3+")

    This ensures base + children share the same group key.

    Args:
        annotations: per-peak matched_annotation strings (length L)
        parent_annotations: per-peak parent_annotation (None for base ions,
            e.g. "b3+" for isotopes/losses of b3)
        feature_types: per-peak feature_type ("base", "isotope", "loss", etc.)
        mz: per-peak m/z values in Da (length L)
        intensity: per-peak intensity values (length L)

    Returns:
        List of FragmentGroup, one per unique base ion
    """
    group_map: Dict[str, List[int]] = defaultdict(list)
    base_ion_indices: Dict[str, int] = {}  # group_key -> base peak index

    for idx in range(len(annotations)):
        if intensity[idx] <= 0:
            continue

        ann = annotations[idx] if idx < len(annotations) else None
        parent = parent_annotations[idx] if parent_annotations and idx < len(parent_annotations) else None
        ftype = feature_types[idx] if feature_types and idx < len(feature_types) else None

        if not ann or str(ann) == "None":
            continue  # unannotated peak

        ann = str(ann)
        parent = str(parent) if parent and str(parent) != "None" else None
        ftype = str(ftype) if ftype and str(ftype) != "None" else None

        if parent is not None:
            # Child peak (isotope or loss): group under parent annotation
            group_key = parent
            group_map[group_key].append(idx)
        else:
            # Base ion (or standalone): group under its own annotation
            group_key = ann
            group_map[group_key].append(idx)
            base_ion_indices[group_key] = idx

    groups: list[Any] = []
    for key, indices in group_map.items():
        # Identify the base peak
        base_idx = base_ion_indices.get(key)
        if base_idx is None:
            # No explicit base found (orphan children?) — skip
            continue

        base_mz_val = float(mz[base_idx])

        # Determine ion type from the group key (which is the base annotation)
        ion_type = "unknown"
        if key.startswith("b") and not key.startswith("by"):
            ion_type = "b"
        elif key.startswith("y"):
            ion_type = "y"
        elif key.startswith("a"):
            ion_type = "a"
        elif key.startswith("p"):
            ion_type = "precursor"

        groups.append(
            FragmentGroup(
                group_key=key,
                peak_indices=indices,
                base_idx=base_idx,
                base_mz=base_mz_val,
                ion_type=ion_type,
            )
        )

    return groups


def _extract_parent_key(annotation: str) -> Optional[str]:
    """Extract the parent ion key from an annotation string.

    Examples:
        "b3" -> "b3"
        "b3+1" -> "b3"  (isotope)
        "b3-H2O" -> "b3"  (neutral loss)
        "y5^2" -> "y5"  (charge state variant)
    """
    if not annotation:
        return None
    # Strip isotope suffix (+1, +2, etc.)
    ann = annotation.split("+")[0]
    # Strip neutral loss suffix (-H2O, -NH3, etc.)
    ann = ann.split("-")[0]
    # Strip charge suffix (^2, ^3, etc.)
    ann = ann.split("^")[0]
    return ann if ann else None


# ---------------------------------------------------------------------------
# Captum target wrapper for prediction attribution
# ---------------------------------------------------------------------------


class PredictionTarget(nn.Module):
    """Captum-compatible forward wrapper for IG attribution.

    MLM-masks all peaks in a fragment group (base + isotopes + losses) and
    returns the mean log-prob across masked positions.  This mirrors
    signal_aware_fragment masking used during training.
    """

    def __init__(
        self,
        model: nn.Module,
        masked_indices: List[int],
        meta: Optional[Dict[str, torch.Tensor]] = None,
        spectra_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialise the input."""
        super().__init__()
        self.model = model
        self.masked_indices = masked_indices
        self.meta = meta
        self.spectra_mask = spectra_mask

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""
        B, L, _ = spectra.shape  # noqa: N806
        device = spectra.device

        # MLM-mask ALL group positions
        mlm_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for idx in self.masked_indices:
            mlm_mask[:, idx] = True

        predictions, _aux = self.model(
            spectra,
            spectra_mask=self.spectra_mask,
            mlm_mask=mlm_mask,
            meta=self.meta,
        )

        # Mean log-prob across all masked positions
        if isinstance(predictions, tuple):
            group_logits, offset_logits = predictions
            scalars: list[Any] = []
            for idx in self.masked_indices:
                g_lp = F.log_softmax(group_logits[:, idx], dim=-1)
                s = g_lp.max(dim=-1).values
                if offset_logits is not None:
                    o_lp = F.log_softmax(offset_logits[:, idx], dim=-1)
                    s = s + o_lp.max(dim=-1).values
                scalars.append(s)
            scalar = torch.stack(scalars, dim=-1).mean(dim=-1)  # (B,)
        else:
            masked_preds = torch.stack([predictions[:, idx, 0] for idx in self.masked_indices], dim=-1)
            scalar = masked_preds.mean(dim=-1)

        return scalar


@torch.no_grad()
def extract_prediction(
    model: nn.Module,
    spectra: torch.Tensor,
    masked_indices: List[int],
    base_idx: int,
    spectra_mask: torch.Tensor,
    true_mz_da: float,
    true_mz_all: Dict[int, float],
    annotations: Optional[List[Optional[str]]] = None,
    max_mz: float = 2500.0,
) -> PredictionInfo:
    """Extract prediction quality for the masked fragment group.

    Args:
        masked_indices: all indices in the group to MLM-mask
        base_idx: the base ion index (for primary error reporting)
        true_mz_all: mapping from peak index -> true m/z in Da (for all group members)
        annotations: per-peak annotation strings (for labeling per-peak predictions)
    """
    B, L, _ = spectra.shape  # noqa: N806
    device = spectra.device

    mlm_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    for idx in masked_indices:
        mlm_mask[:, idx] = True

    predictions, _aux = model(spectra, spectra_mask=spectra_mask, mlm_mask=mlm_mask)

    binning = getattr(model, "binning_strategy", None)
    pred_g, pred_o, true_g, true_o = -1, -1, -1, -1
    correct_bin = False
    group_correct: list[Any] = []
    peak_predictions: List[PeakPrediction] = []

    if isinstance(predictions, tuple):
        group_logits, offset_logits = predictions

        # Base ion prediction
        g_idx = group_logits[:, base_idx].argmax(dim=-1)
        o_idx = offset_logits[:, base_idx].argmax(dim=-1) if offset_logits is not None else torch.zeros_like(g_idx)
        pred_g = g_idx.item()
        pred_o = o_idx.item()

        # Confidence (mean across group)
        conf = 0.0
        for idx in masked_indices:
            g_lp = F.log_softmax(group_logits[:, idx], dim=-1)
            c = g_lp.max(dim=-1).values.item()
            if offset_logits is not None:
                o_lp = F.log_softmax(offset_logits[:, idx], dim=-1)
                c += o_lp.max(dim=-1).values.item()
            conf += c
        conf /= len(masked_indices)

        if binning is not None:
            pred_mz = binning.bin_groups_to_mz(g_idx, o_idx).item()
            true_mz_tensor = torch.tensor([true_mz_da], device=device)
            true_g_t, true_o_t = binning.mz_to_bin_groups(true_mz_tensor)
            true_g = true_g_t.item()
            true_o = true_o_t.item()
            correct_bin = pred_g == true_g and pred_o == true_o

            # Per-peak predictions with confidence, bin indices, and top-N candidates
            for idx in masked_indices:
                p_g = group_logits[:, idx].argmax(dim=-1)
                p_o = offset_logits[:, idx].argmax(dim=-1) if offset_logits is not None else torch.zeros_like(p_g)
                p_mz = binning.bin_groups_to_mz(p_g, p_o).item()

                # Log-prob of predicted bin
                g_lp = F.log_softmax(group_logits[:, idx], dim=-1)
                peak_logprob = g_lp[0, p_g[0]].item()
                if offset_logits is not None:
                    o_lp = F.log_softmax(offset_logits[:, idx], dim=-1)
                    peak_logprob += o_lp[0, p_o[0]].item()

                if idx in true_mz_all:
                    t_mz_val = true_mz_all[idx]
                    t_mz = torch.tensor([t_mz_val], device=device)
                    t_g, t_o = binning.mz_to_bin_groups(t_mz)
                    p_correct = p_g.item() == t_g.item() and p_o.item() == t_o.item()
                    group_correct.append(p_correct)

                    # Rank of the correct bin in the model's group predictions
                    g_sorted = g_lp[0].argsort(descending=True)
                    correct_g_rank = (g_sorted == t_g.item()).nonzero(as_tuple=True)[0]
                    correct_bin_rank = int(correct_g_rank[0].item()) + 1 if len(correct_g_rank) > 0 else -1

                    # Top-5 candidates (group_bin, offset_bin, mz, log_prob)
                    top_candidates: list[Any] = []
                    top5_g = g_sorted[:5]
                    for cand_g in top5_g:
                        # Best offset for this group candidate
                        cand_o = (
                            offset_logits[:, idx].argmax(dim=-1) if offset_logits is not None else torch.zeros(1, dtype=torch.long, device=device)
                        )
                        cand_mz = binning.bin_groups_to_mz(cand_g.unsqueeze(0), cand_o).item()
                        cand_lp = g_lp[0, cand_g].item()
                        if offset_logits is not None:
                            cand_lp += o_lp[0, cand_o[0]].item()
                        top_candidates.append((int(cand_g.item()), int(cand_o[0].item()), cand_mz, cand_lp))

                    ann = str(annotations[idx]) if annotations and idx < len(annotations) and annotations[idx] else f"m/z={t_mz_val:.2f}"
                    peak_predictions.append(
                        PeakPrediction(
                            peak_idx=idx,
                            annotation=ann,
                            true_mz_da=t_mz_val,
                            predicted_mz_da=p_mz,
                            correct_bin=p_correct,
                            log_prob=peak_logprob,
                            correct_bin_rank=correct_bin_rank,
                            predicted_group_bin=p_g.item(),
                            predicted_offset_bin=p_o.item(),
                            true_group_bin=t_g.item(),
                            true_offset_bin=t_o.item(),
                            group_bin_distance=abs(p_g.item() - t_g.item()),
                            top_candidates=top_candidates,
                        )
                    )
        else:
            pred_mz = 0.0
    else:
        pred_norm = predictions[:, base_idx, 0].item()
        pred_mz = pred_norm * max_mz
        conf = 0.0

    error_da = abs(pred_mz - true_mz_da)
    error_ppm = (error_da / true_mz_da * 1e6) if true_mz_da > 0 else 0.0
    grp_acc = float(np.mean(group_correct)) if group_correct else -1.0

    return PredictionInfo(
        predicted_mz_da=pred_mz,
        true_mz_da=true_mz_da,
        error_da=error_da,
        error_ppm=error_ppm,
        confidence=conf,
        correct_bin=correct_bin,
        predicted_group=pred_g,
        true_group=true_g,
        predicted_offset=pred_o,
        true_offset=true_o,
        group_bin_accuracy=grp_acc,
        peak_predictions=peak_predictions,
    )


def _charge_from_group_key(group_key: str) -> int:
    """Extract charge state from a group key (e.g., 'b9++' -> 2, 'y3+' -> 1)."""
    return group_key.count("+") or 1


def _neutral_mass_gap(mz1: float, z1: int, mz2: float, z2: int) -> float:
    """Compute the neutral mass difference between two ions at potentially different charge states."""
    mass1 = mz1 * z1 - z1 * PROTON_MASS
    mass2 = mz2 * z2 - z2 * PROTON_MASS
    return abs(mass1 - mass2)


def _is_ladder_neighbor(mz_diff: float, tol: float = 0.02) -> bool:
    """Check if a mass difference matches any amino acid mass."""
    return any(abs(mz_diff - aa_mass) < tol for aa_mass in AA_MASS_LIST)


def _closest_aa(mz_diff: float) -> str:
    """Find the closest amino acid to a given m/z difference."""
    best_name = "?"
    best_delta = float("inf")
    for name, mass in AMINO_ACID_MASSES.items():
        delta = abs(mz_diff - mass)
        if delta < best_delta:
            best_delta = delta
            best_name = name
    return best_name


# ---------------------------------------------------------------------------
# Top-k attribution analysis
# ---------------------------------------------------------------------------


@dataclass
class RankedPeak:
    """A peak (or fragment group) ranked by attribution magnitude."""

    rank: int  # 1-indexed
    peak_idx: int  # original peak index in the spectrum (base peak for groups)
    attribution: float  # sum of |attribution| across group members (single value for unannotated)
    category: str  # ladder_neighbor, same_series, complementary, etc.
    annotation: str  # annotation string or "unannotated"
    mz: float  # m/z value
    is_group: bool = False  # True if this represents a fragment group (max of members)


@dataclass
class TopKAnalysis:
    """Per-masked-group top-k attribution analysis."""

    # For each k, what fraction of top-k peaks are in each category
    top1_category: str  # category of the single highest-attributed peak
    top1_annotation: str  # annotation of top-1 peak (or "unannotated")
    # Top-k hit rates: fraction of top-k peaks in each category
    topk_fractions: Dict[str, Dict[str, float]]  # k -> {category: fraction}
    # Ladder neighbor analysis
    ladder_neighbors_present: int  # how many ±1 ladder neighbors exist in spectrum
    ladder_neighbors_in_top5: int  # how many are in the top-5 attributed peaks
    ladder_neighbor_best_rank: int  # rank (1-indexed) of highest-attributed ladder neighbor, -1 if none
    # Attribution concentration
    top5_concentration: float  # fraction of total attribution in top-5 peaks
    top10_concentration: float  # fraction of total attribution in top-10 peaks
    gini: float  # Gini coefficient of attribution distribution
    # Ranked peaks for visualization (group-level for annotated, individual for unannotated)
    ranked_peaks: List[RankedPeak] = field(default_factory=list)


def _count_aa_steps(mz_diff: float, tol: float = 0.02) -> int:
    """Count how many amino acid masses fit in the m/z difference (1, 2, 3, or 0).

    Returns 1 for immediate ladder neighbor, 2-3 for near ladder, 0 otherwise.
    """
    if _is_ladder_neighbor(mz_diff, tol):
        return 1
    # Check 2-AA combinations
    for m1 in AA_MASS_LIST:
        residual = mz_diff - m1
        if residual > 0 and _is_ladder_neighbor(residual, tol):
            return 2
    # Check 3-AA combinations (only check if diff is in plausible range)
    if mz_diff < 57.0 * 3 - tol or mz_diff > 186.1 * 3 + tol:
        return 0
    for m1 in AA_MASS_LIST:
        for m2 in AA_MASS_LIST:
            residual = mz_diff - m1 - m2
            if residual > 0 and _is_ladder_neighbor(residual, tol):
                return 3
    return 0


def compute_topk_attribution_analysis(
    attributions: np.ndarray,
    masked_group: FragmentGroup,
    all_groups: List[FragmentGroup],
    annotations: List[Optional[str]],
    mz: np.ndarray,
    precursor_mass: float = 0.0,
    da_tol: float = 0.02,
    k_values: Tuple[int, ...] = (1, 3, 5, 10),
    charge_variant_indices: Optional[Set[int]] = None,
    novel_chem_lookup: Optional[Dict[int, str]] = None,
) -> TopKAnalysis:
    """Analyze top-k attributed peaks by their chemical identity.

    Categories per peak (mutually exclusive):
      - ``ladder_neighbor``: same ion series, adjacent backbone position
      - ``near_ladder``: same ion series, ±2-3 positions away
      - ``distant_same_series``: same ion series, further away
      - ``complementary_pair``: opposite b/y series at the true complementary
        backbone position (neutral-mass sum equals the precursor neutral mass)
      - ``opposite_series``: opposite b/y series, not the complementary partner
      - ``precursor``: precursor ion or its isotopes/losses
      - ``other_annotated``: annotated but not in the above (a-ions, immonium…)
      - ``charge_variant_leakage``: peak is the same fragment as the masked
        group observed at a different charge state (non-chemistry shortcut)
      - ``internal_fragment``/``d_ion``/``w_ion``/``side_chain_loss``/
        ``immonium_related``/``precursor_combined_loss``: unannotated peaks
        matched by the novel-chemistry probe
      - ``unannotated``: no annotation and no extended-chemistry match

    Series-index classification: when both the masked and peak annotations
    carry parseable b/y/a indices, |Δposition| controls the ladder category
    (strict, avoids mass-coincidence aliasing like G+V ≈ R). Mass-based
    fallback is used for unparseable annotations.

    Args:
        charge_variant_indices: visible peak indices that are the masked
            fragment observed at a different charge state. These are labeled
            ``charge_variant_leakage`` regardless of their annotation.
        novel_chem_lookup: mapping ``peak_idx → extended_chem_type``. An
            otherwise-unannotated peak with a lookup entry is labeled with
            that extended-chemistry type instead of ``unannotated``.
    """
    masked_indices = set(masked_group.peak_indices)
    masked_mz = masked_group.base_mz
    masked_ion = masked_group.ion_type
    charge_variant_indices = charge_variant_indices or set()
    novel_chem_lookup = novel_chem_lookup or {}

    # Build lookup: peak_idx -> group for all groups
    idx_to_group: Dict[int, FragmentGroup] = {}
    for g in all_groups:
        for pi in g.peak_indices:
            idx_to_group[pi] = g

    # Get absolute attributions for visible (non-masked, non-padding) peaks
    abs_attr = np.abs(attributions)
    for mi in masked_indices:
        abs_attr[mi] = 0.0
    total_attr = abs_attr.sum()

    # Rank peaks by attribution (descending)
    ranked_indices = np.argsort(-abs_attr)
    ranked_indices = [int(i) for i in ranked_indices if abs_attr[i] > 0]

    masked_charge = _charge_from_group_key(masked_group.group_key)
    masked_series_info = _parse_series_info(masked_group.group_key)

    def _classify_peak(peak_idx: int) -> str:
        """Classify a single peak relative to the masked group."""
        if peak_idx in masked_indices:
            return "masked"

        # Charge-state leakage takes priority over annotation-based labels —
        # a y5++ peak when y5+ is masked is information leakage regardless of
        # whether it happens to be annotated in the reference library.
        if peak_idx in charge_variant_indices:  # type: ignore[operator]
            return "charge_variant_leakage"

        ann = annotations[peak_idx] if peak_idx < len(annotations) else None
        if ann is None:
            # Novel-chemistry match subsumes "unannotated" when present.
            chem_type = novel_chem_lookup.get(peak_idx)  # type: ignore[union-attr]
            if chem_type:
                return chem_type
            return "unannotated"

        group = idx_to_group.get(peak_idx)
        if group is None:
            return "other_annotated"

        # Precursor ion
        if group.ion_type == "precursor":
            return "precursor"

        peak_mz = mz[peak_idx] if peak_idx < len(mz) else 0.0
        peak_series_info = _parse_series_info(group.group_key)

        # Same ion series (b-b or y-y or a-a)
        if group.ion_type == masked_ion:
            # Prefer strict series-index comparison — avoids aliasing where a
            # 2-residue gap coincidentally matches a single AA mass.
            if masked_series_info is not None and peak_series_info is not None:
                delta_pos = abs(peak_series_info[1] - masked_series_info[1])
                if delta_pos == 0:
                    # Same backbone position, different charge → charge-state
                    # leakage. The upstream detector should normally catch
                    # this; this branch is a safety net for edge cases where
                    # the 0.5-Da tolerance missed (rare, but possible with
                    # noisy m/z measurements at the annotation level).
                    return "charge_variant_leakage"
                if delta_pos == 1:
                    return "ladder_neighbor"
                if delta_pos in (2, 3):
                    return "near_ladder"
                return "distant_same_series"
            # Fallback: mass-based when indices can't be parsed
            peak_charge = _charge_from_group_key(group.group_key)
            mass_gap = _neutral_mass_gap(peak_mz, peak_charge, masked_mz, masked_charge)
            steps = _count_aa_steps(mass_gap, da_tol)
            if steps == 1:
                return "ladder_neighbor"
            if steps in (2, 3):
                return "near_ladder"
            return "distant_same_series"

        # Opposite b/y series
        if masked_ion in ("b", "y") and group.ion_type in ("b", "y") and group.ion_type != masked_ion:
            # Complementary pair test: b_i_neutral + y_{L−i}_neutral = M_neutral,
            # where M_neutral is the peptide neutral mass — which by convention
            # already includes the terminal H2O. No extra WATER_MASS term.
            peak_charge = _charge_from_group_key(group.group_key)
            peak_neutral = peak_mz * peak_charge - peak_charge * PROTON_MASS
            masked_neutral = masked_mz * masked_charge - masked_charge * PROTON_MASS
            if abs((peak_neutral + masked_neutral) - precursor_mass) < da_tol * 2:
                return "complementary_pair"
            return "opposite_series"

        return "other_annotated"

    # Classify all ranked peaks
    peak_categories = [_classify_peak(i) for i in ranked_indices]

    # Top-1
    top1_cat = peak_categories[0] if peak_categories else "none"
    top1_ann = ""
    if ranked_indices:
        idx0 = ranked_indices[0]
        ann0 = annotations[idx0] if idx0 < len(annotations) else None
        top1_ann = str(ann0) if ann0 else "unannotated"

    # Top-k fractions — enumerate every structural category so downstream
    # consumers can safely index by category without KeyErrors, including
    # categories that don't appear in this particular group's top-k.
    topk_fractions: Dict[str, Dict[str, float]] = {}
    for k in k_values:
        top_cats = peak_categories[:k]
        n = len(top_cats)
        if n == 0:
            topk_fractions[str(k)] = dict.fromkeys(STRUCTURAL_CATEGORIES, 0.0)
        else:
            topk_fractions[str(k)] = {c: sum(1 for t in top_cats if t == c) / n for c in STRUCTURAL_CATEGORIES}

    # Ladder neighbor analysis: find ±1 residue neighbors in same series
    # Uses neutral mass gap to handle ions at different charge states
    ladder_neighbor_indices = set()
    for g in all_groups:
        if g.group_key == masked_group.group_key:
            continue
        if g.ion_type != masked_ion:
            continue
        g_charge = _charge_from_group_key(g.group_key)
        mass_gap = _neutral_mass_gap(g.base_mz, g_charge, masked_mz, masked_charge)
        if _is_ladder_neighbor(mass_gap, da_tol):
            ladder_neighbor_indices.update(g.peak_indices)

    ladder_present = len(
        [
            g
            for g in all_groups
            if g.group_key != masked_group.group_key
            and g.ion_type == masked_ion
            and _is_ladder_neighbor(_neutral_mass_gap(g.base_mz, _charge_from_group_key(g.group_key), masked_mz, masked_charge), da_tol)
        ]
    )

    top5_set = set(ranked_indices[:5])
    ladder_in_top5 = len(ladder_neighbor_indices & top5_set)

    # Best rank of any ladder neighbor peak
    ladder_best_rank = -1
    for rank, pi in enumerate(ranked_indices):
        if pi in ladder_neighbor_indices:
            ladder_best_rank = rank + 1  # 1-indexed
            break

    # Attribution concentration
    if total_attr > 0 and ranked_indices:
        sorted_attr = np.array([abs_attr[i] for i in ranked_indices])
        top5_conc = float(sorted_attr[:5].sum() / total_attr)
        top10_conc = float(sorted_attr[:10].sum() / total_attr)
        # Gini coefficient
        n = len(sorted_attr)
        if n > 1:
            sorted_asc = np.sort(sorted_attr)
            index = np.arange(1, n + 1)
            gini = float((2 * (index * sorted_asc).sum() / (n * sorted_asc.sum())) - (n + 1) / n)
        else:
            gini = 0.0
    else:
        top5_conc = 0.0
        top10_conc = 0.0
        gini = 0.0

    # Build ranked peaks list (group-level for annotated, individual for unannotated)
    # For annotated groups: sum attribution across all group members (captures
    # total information flow through the group, not just the strongest peak).
    # Group-level *category* is always taken from the base ion, not from the
    # first-seen child, so a y12+ group with a loss child (y12+-NH3) encountered
    # first in attribution rank still reports the base's classification. Prior
    # versions used whichever child was hit first, which silently mis-labeled
    # complement groups when their loss children failed the neutral-sum check.
    # For unannotated peaks: individual entries carrying the classifier's
    # category (may be ``charge_variant_leakage`` or an extended-chem type).
    seen_groups: Dict[str, RankedPeak] = {}
    base_category_cache: Dict[str, str] = {}
    individual_peaks: List[RankedPeak] = []

    def _group_base_category(group: FragmentGroup) -> str:
        gk = group.group_key
        if gk in base_category_cache:
            return base_category_cache[gk]
        base_pi = group.base_idx
        if base_pi is None or not (0 <= base_pi < len(annotations)):
            # Fall back to a child-peak classification if the base is missing
            child_pi = next((pi for pi in group.peak_indices if 0 <= pi < len(annotations)), None)
            cat = _classify_peak(child_pi) if child_pi is not None else "other_annotated"
        else:
            cat = _classify_peak(base_pi)
        base_category_cache[gk] = cat
        return cat

    for rank_idx, pi in enumerate(ranked_indices[:TOP_K_RANKED_PEAKS_CAP]):
        cat = peak_categories[rank_idx]
        attr_val = float(abs_attr[pi])
        group = idx_to_group.get(pi)
        ann = annotations[pi] if pi < len(annotations) and annotations[pi] else None
        peak_mz = float(mz[pi]) if pi < len(mz) else 0.0

        # Charge-state leakage short-circuits group aggregation — a y5++
        # peak should never be lumped into its "y5++" group for the purposes
        # of this ranking because we want it displayed as a distinct leakage
        # hit rather than hidden inside a group sum.
        if cat == "charge_variant_leakage":
            individual_peaks.append(
                RankedPeak(
                    rank=0,
                    peak_idx=pi,
                    attribution=attr_val,
                    category=cat,
                    annotation=str(ann) if ann else f"m/z={peak_mz:.2f}",
                    mz=peak_mz,
                    is_group=False,
                )
            )
        elif group is not None and group.group_key != masked_group.group_key:
            gk = group.group_key
            if gk in seen_groups:
                # Accumulate attribution (sum across group members)
                seen_groups[gk].attribution += attr_val
            else:
                seen_groups[gk] = RankedPeak(
                    rank=0,
                    peak_idx=pi,
                    attribution=attr_val,
                    category=_group_base_category(group),
                    annotation=gk,
                    mz=group.base_mz,
                    is_group=True,
                )
        elif ann is None:
            individual_peaks.append(
                RankedPeak(
                    rank=0,
                    peak_idx=pi,
                    attribution=attr_val,
                    category=cat,
                    annotation=f"m/z={peak_mz:.2f}",
                    mz=peak_mz,
                    is_group=False,
                )
            )

    # Merge and sort by attribution
    all_ranked = list(seen_groups.values()) + individual_peaks
    all_ranked.sort(key=lambda r: r.attribution, reverse=True)
    for i, rp in enumerate(all_ranked):
        rp.rank = i + 1
    ranked_peaks = all_ranked[:TOP_K_RANKED_PEAKS_EMIT]

    return TopKAnalysis(
        top1_category=top1_cat,
        top1_annotation=top1_ann,
        topk_fractions=topk_fractions,
        ladder_neighbors_present=ladder_present,
        ladder_neighbors_in_top5=ladder_in_top5,
        ladder_neighbor_best_rank=ladder_best_rank,
        top5_concentration=top5_conc,
        top10_concentration=top10_conc,
        gini=gini,
        ranked_peaks=ranked_peaks,
    )


# ---------------------------------------------------------------------------
# PA Bias Dissection utilities
# ---------------------------------------------------------------------------


@torch.no_grad()
@torch.no_grad()
def sweep_pa_response(
    pairwise_bias_module: nn.Module,
    dmz_min: float = -250.0,
    dmz_max: float = 250.0,
    resolution: float = 0.01,
    device: str = "cpu",
    pw_projection: Optional[nn.Module] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sweep synthetic delta-m/z values through the PA pipeline.

    If pw_projection is provided, computes the mean absolute per-head bias
    (the actual attention bias the model uses). Otherwise falls back to
    the f_pw output L2 norm (intermediate representation).

    Returns:
        (dmz_values, response_magnitude): numpy arrays
    """
    orig_device = next(pairwise_bias_module.parameters()).device

    dmz_values = torch.arange(dmz_min, dmz_max + resolution, resolution, device=orig_device)
    dmz_input = dmz_values.unsqueeze(-1)  # (N, 1)

    fourier_feats = pairwise_bias_module._fourier_encode(dmz_input)  # (N, 2*num_freqs)
    pw_output = pairwise_bias_module.f_pw(fourier_feats)  # (N, hidden_dim)

    if pw_projection is not None:
        # Full pipeline: f_pw → norm → g_pw → per-head biases
        normed = pw_projection.pw_norm(pw_output)
        if hasattr(pw_projection, "g_pw_batched"):
            per_head = pw_projection.g_pw_batched(normed)  # (N, n_heads*n_layers)
        else:
            per_head = pw_projection.g_pw(normed)
        # Mean absolute bias across all heads/layers
        response_mag = per_head.abs().mean(dim=-1)  # (N,)
    else:
        # Fallback: f_pw output L2 norm
        response_mag = pw_output.norm(dim=-1)  # (N,)

    return dmz_values.cpu().numpy(), response_mag.detach().cpu().numpy()


@torch.no_grad()
def sweep_pa_per_head_bias(
    pairwise_bias_module: nn.Module,
    pw_projection: nn.Module,
    dmz_min: float = -250.0,
    dmz_max: float = 250.0,
    resolution: float = 0.01,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Sweep synthetic delta-m/z and compute per-head, per-layer bias values.

    Does NOT mutate module device — runs on a temporary clone.

    Returns:
        (dmz_values, per_head_biases): numpy arrays
    """
    # Run on original device to avoid cloning the projection (complex structure)
    orig_device = next(pairwise_bias_module.parameters()).device

    dmz_values = torch.arange(dmz_min, dmz_max + resolution, resolution, device=orig_device)
    dmz_input = dmz_values.unsqueeze(-1)

    fourier_feats = pairwise_bias_module._fourier_encode(dmz_input)
    pw_output = pairwise_bias_module.f_pw(fourier_feats)  # (N, hidden_dim)

    normed = pw_projection.pw_norm(pw_output)

    if hasattr(pw_projection, "g_pw_batched"):
        per_head = pw_projection.g_pw_batched(normed)
    else:
        per_head = pw_projection.g_pw(normed)

    return dmz_values.cpu().numpy(), per_head.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Paper-quality plotting
# ---------------------------------------------------------------------------


def plot_pa_response_spectrum(
    dmz: np.ndarray,
    response: np.ndarray,
    save_path: Path,
    title: str = "Pairwise Attention Bias Response Spectrum",
    figsize: Tuple[float, float] = (16, 12),
    dpi: int = 300,
) -> None:
    """Plot the PA response spectrum with chemical reference overlays.

    3-panel figure:
      A: Full spectrum (0-200 Da) with 0.2 Da smoothing, detected peaks with
         inter-peak mass deltas annotated
      B: Isotope zone (0-1.2 Da) — charge-dependent isotope spacing at full resolution
      C: Per-AA response bar chart — response at each amino acid mass vs global mean
    """
    try:
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import find_peaks as _find_peaks
    except ImportError:
        try:
            import matplotlib.gridspec as gridspec
            import matplotlib.pyplot as plt

            gaussian_filter1d = None
            _find_peaks = None
        except ImportError:
            return

    pos_mask = dmz >= 0
    dmz_pos = dmz[pos_mask]
    resp_pos = response[pos_mask]

    resolution = float(dmz_pos[1] - dmz_pos[0]) if len(dmz_pos) > 1 else 0.01

    # Smooth at 0.2 Da — matches model bin width
    sigma_02 = max(1, int(0.2 / resolution))
    if gaussian_filter1d is not None:
        resp_smooth = gaussian_filter1d(resp_pos, sigma=sigma_02)
    else:
        resp_smooth = resp_pos

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # ===== Panel A: Full spectrum with detected peaks + inter-peak deltas =====
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.plot(dmz_pos, resp_pos, color="#d5d8dc", linewidth=0.3, alpha=0.3)
    ax_a.plot(dmz_pos, resp_smooth, color="#2c3e50", linewidth=1.5, alpha=0.9, label="Smoothed (σ=0.2 Da)")
    ax_a.set_xlabel("Δm/z (Da)", fontsize=11)
    ax_a.set_ylabel("Mean |attention bias|", fontsize=11)
    ax_a.set_title("A  PA Response Spectrum — detected peaks with mass deltas", fontsize=12, fontweight="bold", loc="left")
    ax_a.set_xlim(0, 200)

    # Detect peaks and annotate with chemical identity + inter-peak deltas
    if _find_peaks is not None:
        peak_idx, peak_props = _find_peaks(
            resp_smooth,
            prominence=0.02,
            distance=max(1, int(1.5 / resolution)),
        )
        sorted_by_prom = np.argsort(-peak_props["prominences"])
        # Keep top-20 peaks for annotation
        top_peak_indices = [peak_idx[si] for si in sorted_by_prom[:20] if 3 < dmz_pos[peak_idx[si]] < 200]
        top_peak_indices.sort()  # sort by m/z for delta computation

        prev_mz = None
        for pidx in top_peak_indices:
            pmz = float(dmz_pos[pidx])
            presp = float(resp_smooth[pidx])

            # Identify peak
            best_label = f"{pmz:.1f}"
            best_delta = 999.0
            for name, mass in AMINO_ACID_MASSES.items():
                d = abs(pmz - mass)
                if d < best_delta:
                    best_delta = d
                    best_label = name if d < 1.0 else f"{pmz:.1f}"
            for name, mass in NEUTRAL_LOSS_MASSES.items():
                d = abs(pmz - mass)
                if d < best_delta:
                    best_delta = d
                    best_label = name if d < 1.0 else best_label
            for n1, m1 in AMINO_ACID_MASSES.items():
                for n2, m2 in AMINO_ACID_MASSES.items():
                    d = abs(pmz - (m1 + m2))
                    if d < best_delta and d < 1.0:
                        best_delta = d
                        best_label = f"{n1}+{n2}"

            color = "#e74c3c" if best_delta < 1.0 else "#7f8c8d"
            ax_a.plot(pmz, presp, "v", color=color, markersize=5, zorder=5)
            ax_a.annotate(
                best_label,
                (pmz, presp),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=6,
                color=color,
                fontweight="bold" if best_delta < 1.0 else "normal",
            )

            # Inter-peak delta annotation
            if prev_mz is not None:
                delta = pmz - prev_mz
                # Check if delta matches an AA mass
                delta_label = f"Δ{delta:.1f}"
                for name, mass in AMINO_ACID_MASSES.items():
                    if abs(delta - mass) < 1.0:
                        delta_label = f"Δ{name}"
                        break
                mid_mz = (pmz + prev_mz) / 2
                mid_resp = min(presp, float(resp_smooth[np.argmin(np.abs(dmz_pos - prev_mz))]))
                ax_a.annotate(delta_label, (mid_mz, mid_resp - 0.02), ha="center", fontsize=5, color="#566573", fontstyle="italic")
            prev_mz = pmz

    ax_a.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax_a.grid(True, alpha=0.3)

    # ===== Panel B: Isotope zone (0-1.2 Da) at full resolution =====
    ax_b = fig.add_subplot(gs[1, 0])
    iso_mask = (dmz_pos >= 0) & (dmz_pos <= 1.2)
    ax_b.plot(dmz_pos[iso_mask], resp_pos[iso_mask], color="#2c3e50", linewidth=1.0)
    ax_b.set_xlabel("Δm/z (Da)", fontsize=10)
    ax_b.set_ylabel("Mean |attention bias|", fontsize=10)
    ax_b.set_title("B  Isotope Zone (0–1.2 Da)", fontsize=11, fontweight="bold", loc="left")

    iso_labels: dict[str, Any] = {3: ISOTOPE_SPACING / 3, 2: ISOTOPE_SPACING / 2, 1: ISOTOPE_SPACING}  # type: ignore[dict-item]
    iso_colors: dict[str, Any] = {"3": "#f1c40f", "2": "#e67e22", "1": "#e74c3c"}
    for z, spacing in iso_labels.items():
        color = iso_colors[str(z)]
        ax_b.axvline(spacing, color=color, linewidth=1.5, linestyle="--", alpha=0.8)
        ax_b.text(spacing + 0.01, ax_b.get_ylim()[1] * 0.95, f"z={z}\n{spacing:.3f}", fontsize=8, color=color, va="top")

    ax_b.axvline(PROTON_MASS, color="#9b59b6", linewidth=1.0, linestyle=":", alpha=0.7)
    ax_b.text(PROTON_MASS + 0.01, ax_b.get_ylim()[1] * 0.75, f"H⁺\n{PROTON_MASS:.3f}", fontsize=7, color="#9b59b6", va="top")
    ax_b.grid(True, alpha=0.3)

    # ===== Panel C: Per-AA response bar chart =====
    ax_c = fig.add_subplot(gs[1, 1])

    # Compute response at each AA mass and global mean
    global_mean = float(resp_smooth[(dmz_pos >= 50) & (dmz_pos <= 200)].mean())
    aa_responses: list[Any] = []
    for name, mass in sorted(AMINO_ACID_MASSES.items(), key=lambda x: x[1]):
        idx = np.argmin(np.abs(dmz_pos - mass))
        # Average response in a ±0.3 Da window
        window = (dmz_pos >= mass - 0.3) & (dmz_pos <= mass + 0.3)
        if window.sum() > 0:
            resp_at = float(resp_smooth[window].mean())
        else:
            resp_at = float(resp_smooth[idx])
        aa_responses.append((name, mass, resp_at))

    names = [r[0] for r in aa_responses]
    responses = [r[2] for r in aa_responses]
    colors = ["#27ae60" if r > global_mean else "#e74c3c" for r in responses]

    ax_c.barh(range(len(names)), responses, color=colors, alpha=0.85, edgecolor="white", height=0.7)
    ax_c.axvline(global_mean, color="#2c3e50", linewidth=1.5, linestyle="--", alpha=0.7, label=f"Global mean ({global_mean:.3f})")
    ax_c.set_yticks(range(len(names)))
    ax_c.set_yticklabels([f"{n} ({aa_responses[i][1]:.0f})" for i, n in enumerate(names)], fontsize=7)
    ax_c.set_xlabel("Mean |attention bias|", fontsize=10)
    ax_c.set_title("C  Response at Amino Acid Masses", fontsize=11, fontweight="bold", loc="left")
    ax_c.invert_yaxis()
    ax_c.legend(fontsize=7, loc="lower right", framealpha=0.8)
    ax_c.grid(axis="x", alpha=0.3)

    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_pa_per_head_heatmap(
    dmz: np.ndarray,
    per_head_biases: np.ndarray,
    n_heads: int,
    n_layers: int,
    save_path: Path,
    title: str = "Per-Head PA Bias Response",
    figsize: Tuple[float, float] = (18, 8),
    dpi: int = 300,
    dmz_range: Tuple[float, float] = (0, 200),
    downsample: int = 10,
) -> None:
    """Plot per-head PA bias as a heatmap (rows=heads, columns=Δm/z)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    pos_mask = (dmz >= dmz_range[0]) & (dmz <= dmz_range[1])
    dmz_plot = dmz[pos_mask][::downsample]
    biases_plot = per_head_biases[pos_mask][::downsample]

    n_total_heads = biases_plot.shape[1]
    is_batched = n_total_heads == n_heads * n_layers

    fig, ax = plt.subplots(figsize=figsize)
    heatmap_data = biases_plot.T

    vmax = np.percentile(np.abs(heatmap_data), 99)
    im = ax.imshow(
        heatmap_data,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=[dmz_plot[0], dmz_plot[-1], n_total_heads - 0.5, -0.5],
        interpolation="nearest",
    )

    if is_batched:
        labels = [f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)]  # noqa: E741
    else:
        labels = [f"H{h}" for h in range(n_heads)]
    ax.set_yticks(range(n_total_heads))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Δm/z (Da)", fontsize=11)
    ax.set_ylabel("Head", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    for _name, mass in AMINO_ACID_MASSES.items():
        if dmz_range[0] <= mass <= dmz_range[1]:
            ax.axvline(mass, color="black", alpha=0.2, linewidth=0.5, linestyle="--")
    for _name, mass in NEUTRAL_LOSS_MASSES.items():
        if dmz_range[0] <= mass <= dmz_range[1]:
            ax.axvline(mass, color="green", alpha=0.3, linewidth=0.8, linestyle=":")

    plt.colorbar(im, ax=ax, label="Bias value", shrink=0.8)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_attribution_summary(
    all_results: List["AttributionResult"],
    save_path: Path,
    dpi: int = 300,
) -> None:
    """Plot a 3-panel summary of IG attribution analysis across all masked groups.

    Panel A: Top-1 category distribution (what does the model attribute to most?)
    Panel B: Bin accuracy stratified by top-1 category in 3 groups
    Panel C: Ladder utilization funnel (available → top-5 → top-1)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not all_results:
        return

    category_colors: dict[str, Any] = {
        "ladder_neighbor": "#3498db",
        "near_ladder": "#5dade2",
        "distant_same_series": "#aed6f1",
        "complementary_pair": "#e74c3c",
        "opposite_series": "#f1948a",
        "precursor": "#f39c12",
        "other_annotated": "#2ecc71",
        "charge_variant_leakage": "#c0392b",
        "internal_fragment": "#1abc9c",
        "d_ion": "#16a085",
        "w_ion": "#0e8274",
        "side_chain_loss": "#27ae60",
        "immonium_related": "#8e44ad",
        "precursor_combined_loss": "#d35400",
        "unannotated": "#95a5a6",
    }
    category_labels: dict[str, Any] = {
        "ladder_neighbor": "Ladder\nneighbor",
        "near_ladder": "Near\nladder",
        "distant_same_series": "Distant\nsame",
        "complementary_pair": "Compl.\npair",
        "opposite_series": "Opposite\nseries",
        "precursor": "Precursor",
        "other_annotated": "Other\nannotated",
        "charge_variant_leakage": "Charge\nleakage",
        "internal_fragment": "Internal\nfragment",
        "d_ion": "d-ion",
        "w_ion": "w-ion",
        "side_chain_loss": "Side-chain\nloss",
        "immonium_related": "Immonium\nrelated",
        "precursor_combined_loss": "Prec.\ncomb.loss",
        "unannotated": "Unannotated",
    }

    n_total = len(all_results)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle(f"IG Attribution Summary (n={n_total} masked groups)", fontsize=13, fontweight="bold", y=1.02)

    # --- Panel A: Top-1 category distribution ---
    ax = axes[0]
    top1_cats = [r.topk.top1_category for r in all_results if r.topk]
    categories = list(category_colors.keys())
    counts = {c: sum(1 for t in top1_cats if t == c) for c in categories}
    active = [(c, counts[c]) for c in categories if counts[c] > 0]

    if active:
        labels_a = [category_labels.get(c, c) for c, _ in active]
        values_a = [v / len(top1_cats) * 100 for _, v in active]
        colors_a = [category_colors.get(c, "#bdc3c7") for c, _ in active]
        bars = ax.bar(range(len(active)), values_a, color=colors_a, edgecolor="white", alpha=0.9)
        ax.set_xticks(range(len(active)))
        ax.set_xticklabels(labels_a, fontsize=8)
        for bar, val in zip(bars, values_a, strict=False):
            if val > 3:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold"
                )
    ax.set_ylabel("% of masked groups", fontsize=10)
    ax.set_title("A  Top-1 Attributed Peak Category", fontsize=11, fontweight="bold", loc="left")
    ax.grid(axis="y", alpha=0.3)

    # --- Panel B: Bin accuracy by 3-way split ---
    ax = axes[1]
    ladder_set = {"ladder_neighbor", "near_ladder", "complementary_pair"}
    other_ann_set = {"distant_same_series", "opposite_series", "precursor", "other_annotated", "charge_variant_leakage"}
    # Extended-chemistry top-1 counts as "other annotated" for this split —
    # it's informative chemistry, just not the canonical ladder/complement.
    other_ann_set |= set(EXTENDED_CHEMISTRY_CATEGORIES)

    ladder_results = [r for r in all_results if r.topk and r.topk.top1_category in ladder_set]
    other_results = [r for r in all_results if r.topk and r.topk.top1_category in other_ann_set]
    unann_results = [r for r in all_results if r.topk and r.topk.top1_category == "unannotated"]

    groups_b: list[Any] = []
    accs_b: list[Any] = []
    colors_b: list[Any] = []
    ns_b: list[Any] = []
    for label, results_grp, color in [
        ("Ladder /\nnear-ladder", ladder_results, "#3498db"),
        ("Other\nannotated", other_results, "#2ecc71"),
        ("Unannotated", unann_results, "#95a5a6"),
    ]:
        if results_grp:
            acc = np.mean([r.prediction.correct_bin for r in results_grp if r.prediction]) * 100
            groups_b.append(label)
            accs_b.append(acc)
            colors_b.append(color)
            ns_b.append(len(results_grp))

    if groups_b:
        bars = ax.bar(range(len(groups_b)), accs_b, color=colors_b, edgecolor="white", alpha=0.9, width=0.65)
        ax.set_xticks(range(len(groups_b)))
        ax.set_xticklabels(groups_b, fontsize=9)
        for bar, val, n in zip(bars, accs_b, ns_b, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{val:.0f}%\n(n={n})",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    ax.set_ylabel("Bin accuracy (%)", fontsize=10)
    ax.set_title("B  Prediction Accuracy by Attribution Type", fontsize=11, fontweight="bold", loc="left")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel C: Ladder utilization funnel ---
    ax = axes[2]
    analyses = [r.topk for r in all_results if r.topk]
    if analyses:
        # How many have at least one ladder neighbor present in the spectrum?
        n_with_ladder = sum(1 for a in analyses if a.ladder_neighbors_present > 0)
        # How many have a ladder neighbor in top-5?
        n_ladder_top5 = sum(1 for a in analyses if a.ladder_neighbors_in_top5 > 0)
        # How many have ladder as top-1?
        n_ladder_top1 = sum(1 for a in analyses if a.top1_category in ("ladder_neighbor", "near_ladder"))

        funnel_labels = ["Ladder\npresent", "Ladder\nin top-5", "Ladder\nas top-1"]
        funnel_values = [
            n_with_ladder / len(analyses) * 100,
            n_ladder_top5 / len(analyses) * 100,
            n_ladder_top1 / len(analyses) * 100,
        ]
        funnel_colors = ["#aed6f1", "#5dade2", "#2980b9"]

        bars = ax.bar(range(3), funnel_values, color=funnel_colors, edgecolor="white", alpha=0.9, width=0.65)
        ax.set_xticks(range(3))
        ax.set_xticklabels(funnel_labels, fontsize=9)
        for bar, val, n in zip(bars, funnel_values, [n_with_ladder, n_ladder_top5, n_ladder_top1], strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{val:.0f}%\n({n}/{len(analyses)})",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        # Add median best rank annotation
        best_ranks = [a.ladder_neighbor_best_rank for a in analyses if a.ladder_neighbor_best_rank > 0]
        if best_ranks:
            median_rank = float(np.median(best_ranks))
            ax.text(
                0.95,
                0.95,
                f"Median ladder\nbest rank: {median_rank:.0f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "#eaf2f8", "alpha": 0.9},
            )

    ax.set_ylabel("% of masked groups", fontsize=10)
    ax.set_title("C  Ladder Neighbor Utilization", fontsize=11, fontweight="bold", loc="left")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_novel_chemistry_summary(
    novel_results: Dict[str, Any],
    save_path: Path,
    dpi: int = 300,
) -> None:
    """Create a summary figure for the novel chemistry discovery probe.

    4-panel figure showing per-type hit rates, null model comparisons,
    and confidence/attribution distributions for matched vs unmatched peaks.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Novel Chemistry Discovery Probe", fontsize=13, fontweight="bold", y=0.98)

    hit_rates = novel_results.get("hit_rates", {})
    nulls = novel_results.get("null_models", {})
    conf = novel_results.get("confidence_profile", {})
    attr = novel_results.get("attribution_profile", {})
    n_unann = novel_results.get("n_unannotated_peaks", 0)
    n_matched = novel_results.get("n_matched", 0)

    type_colors: dict[str, Any] = {
        "internal_fragment": "#3498db",
        "immonium_related": "#e74c3c",
        "precursor_combined_loss": "#f39c12",
        "side_chain_loss": "#2ecc71",
        "d_ion": "#9b59b6",
        "w_ion": "#1abc9c",
    }
    type_labels: dict[str, Any] = {
        "internal_fragment": "Internal\nfragment",
        "immonium_related": "Immonium\nrelated",
        "precursor_combined_loss": "Precursor\ncomb. loss",
        "side_chain_loss": "Side-chain\nloss",
        "d_ion": "d-ion",
        "w_ion": "w-ion",
    }

    # --- Panel A: Per-type hit rates ---
    ax = axes[0, 0]
    types_ordered = ["internal_fragment", "side_chain_loss", "precursor_combined_loss", "immonium_related", "w_ion", "d_ion"]
    vals = [hit_rates.get(t, 0) * 100 for t in types_ordered]
    colors = [type_colors.get(t, "#95a5a6") for t in types_ordered]
    labels = [type_labels.get(t, t) for t in types_ordered]
    bars = ax.bar(range(len(types_ordered)), vals, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, vals, strict=False):
        if val > 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(types_ordered)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Hit rate (%)")
    ax.set_title(f"A. Per-type hit rates ({n_matched}/{n_unann} matched)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel B: Null model comparison ---
    ax = axes[0, 1]
    overall = novel_results.get("overall_hit_rate", 0) * 100
    shift_rate = nulls.get("shift", {}).get("hit_rate", 0) * 100
    scramble_rate = nulls.get("scramble", {}).get("hit_rate", 0) * 100
    nulls.get("shift", {}).get("enrichment", 0)
    scramble_enrich = nulls.get("scramble", {}).get("enrichment", 0)

    bar_labels = ["Real", "Scramble\nnull", "Shift\nnull"]
    bar_vals = [overall, scramble_rate, shift_rate]
    bar_colors = ["#3498db", "#e74c3c", "#95a5a6"]
    bars = ax.bar(range(3), bar_vals, color=bar_colors, edgecolor="white", linewidth=0.5)
    for bar, val, _lbl in zip(bars, bar_vals, bar_labels, strict=False):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{val:.2f}%", ha="center", va="bottom", fontsize=9)
    # Add enrichment annotations
    if scramble_enrich > 0 and scramble_enrich != float("inf"):
        ax.annotate(
            f"{scramble_enrich:.1f}×", xy=(0.5, max(overall, scramble_rate) * 0.5), fontsize=10, fontweight="bold", ha="center", color="#e74c3c"
        )
    ax.set_xticks(range(3))
    ax.set_xticklabels(bar_labels, fontsize=9)
    ax.set_ylabel("Hit rate (%)")
    ax.set_title("B. Null model comparison", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel C: Confidence distribution ---
    ax = axes[1, 0]
    n_conf_m = conf.get("n_matched", 0)
    n_conf_u = conf.get("n_unmatched", 0)
    conf_auroc = conf.get("auroc")
    if n_conf_m > 0 and n_conf_u > 0:
        bar_labels = ["Matched", "Unmatched"]
        means = [conf.get("mean_conf_matched", 0), conf.get("mean_conf_unmatched", 0)]
        medians = [conf.get("median_conf_matched", 0), conf.get("median_conf_unmatched", 0)]
        x = np.arange(2)
        bars_mean = ax.bar(x - 0.15, means, 0.3, label="Mean", color=["#3498db", "#95a5a6"], alpha=0.8)
        bars_med = ax.bar(x + 0.15, medians, 0.3, label="Median", color=["#3498db", "#95a5a6"], alpha=0.5, hatch="//")
        for bar, val in zip(list(bars_mean) + list(bars_med), means + medians, strict=False):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Matched\n(n={n_conf_m})", f"Unmatched\n(n={n_conf_u})"], fontsize=9)
        ax.legend(fontsize=8)
        auroc_text = f"AUROC = {conf_auroc:.3f}" if conf_auroc is not None else "AUROC: N/A"
        ax.set_title(f"C. Confidence comparison ({auroc_text})", fontsize=10)
    else:
        ax.text(0.5, 0.5, "No confidence data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("C. Confidence comparison", fontsize=10)
    ax.set_ylabel("Confidence (conf_joint)")
    ax.grid(axis="y", alpha=0.3)

    # --- Panel D: Attribution distribution ---
    ax = axes[1, 1]
    n_attr_m = attr.get("n_matched", 0)
    n_attr_u = attr.get("n_unmatched", 0)
    attr_auroc = attr.get("auroc")
    if n_attr_m > 0 and n_attr_u > 0:
        bar_labels = ["Matched", "Unmatched"]
        means = [attr.get("mean_attr_matched", 0), attr.get("mean_attr_unmatched", 0)]
        medians = [attr.get("median_attr_matched", 0), attr.get("median_attr_unmatched", 0)]
        x = np.arange(2)
        bars_mean = ax.bar(x - 0.15, means, 0.3, label="Mean", color=["#3498db", "#95a5a6"], alpha=0.8)
        bars_med = ax.bar(x + 0.15, medians, 0.3, label="Median", color=["#3498db", "#95a5a6"], alpha=0.5, hatch="//")
        for bar, val in zip(list(bars_mean) + list(bars_med), means + medians, strict=False):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001, f"{val:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Matched\n(n={n_attr_m})", f"Unmatched\n(n={n_attr_u})"], fontsize=9)
        ax.legend(fontsize=8)
        auroc_text = f"AUROC = {attr_auroc:.3f}" if attr_auroc is not None else "AUROC: N/A"
        ax.set_title(f"D. Attribution comparison ({auroc_text})", fontsize=10)
    else:
        ax.text(0.5, 0.5, "No attribution data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("D. Attribution comparison", fontsize=10)
    ax.set_ylabel("Max |IG attribution|")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_hero_spectrum_attribution(
    mz: np.ndarray,
    intensity: np.ndarray,
    result: "AttributionResult",
    save_path: Path,
    title: str = "",
    annotations: Optional[List[str]] = None,
    max_mz: float = 2500.0,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    per_peak_confidence: Optional[np.ndarray] = None,
) -> None:
    """Plot a multi-panel hero spectrum with IG attribution.

    Panel 1: Index-based view with ion-type coloring + annotations. Masked group in red.
    Panel 2: IG attribution heatmap — entire fragment group MLM-masked (training-aligned).
             Shows which visible peaks the model uses for reconstruction.
    Panel 2b (if confidence provided): Per-peak confidence strip — shows the model's
             prediction confidence for each visible peak (independent of masking).
    Panel 3a: Per-peak prediction quality for the masked group.
    Panel 3b: Ranked attribution by fragment group.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    try:
        from instanovo_fm.utils.ion_visualization import (
            CATEGORY_COLORS,
            categorize_ion,
            format_annotation_display,
        )
    except ImportError:
        return

    masked_indices = result.masked_group.peak_indices

    # --- Prepare data ---
    valid_mask = intensity > 0
    n_valid = int(valid_mask.sum())
    if n_valid < 5:
        return

    valid_idx = np.where(valid_mask)[0]
    v_mz = mz[valid_idx]
    v_int = intensity[valid_idx]

    # Map original indices to valid-peak indices
    set(masked_indices)
    masked_valid_set = set()
    for orig_idx in masked_indices:
        pos = np.searchsorted(valid_idx, orig_idx)
        if pos < len(valid_idx) and valid_idx[pos] == orig_idx:
            masked_valid_set.add(pos)

    # Also identify child-only indices (isotopes/losses, not the base)
    base_orig = result.masked_group.base_idx
    child_valid_set = set()
    for orig_idx in masked_indices:
        if orig_idx != base_orig:
            pos = np.searchsorted(valid_idx, orig_idx)
            if pos < len(valid_idx) and valid_idx[pos] == orig_idx:
                child_valid_set.add(pos)

    max_int = v_int.max() if n_valid > 0 else 1.0
    norm_int = v_int / max_int if max_int > 0 else v_int

    # Build per-peak categories
    peak_categories: list[Any] = []
    peak_ann_display: list[Any] = []
    if annotations is not None:
        for vi in valid_idx:
            ann = annotations[vi] if vi < len(annotations) else ""
            ann_str = str(ann) if ann and str(ann) != "None" else ""
            peak_categories.append(categorize_ion(ann_str))
            peak_ann_display.append(format_annotation_display(ann_str) if ann_str else "")
    else:
        peak_categories = ["unannotated"] * n_valid
        peak_ann_display = [""] * n_valid

    # Attribution arrays for valid peaks
    def _prep_attr(raw_attr: np.ndarray) -> np.ndarray:
        if len(raw_attr) == 0:
            return np.zeros(n_valid)
        v = np.abs(raw_attr[valid_idx])
        # Zero out the base ion's own attribution (not meaningful)
        for vi_pos in masked_valid_set:
            v[vi_pos] = 0.0
        return v

    v_attr = _prep_attr(result.attributions)

    # Prepare per-peak confidence for valid peaks (if provided)
    v_conf = None
    if per_peak_confidence is not None and len(per_peak_confidence) >= len(mz):
        v_conf = per_peak_confidence[valid_mask]

    # --- Build figure: 3 or 4 rows, width scales with peak count ---
    has_conf_panel = v_conf is not None
    if figsize is None:
        fig_width = max(16, min(30, n_valid * 0.15))
        fig_height = 20 if has_conf_panel else 18
        figsize = (fig_width, fig_height)
    fig = plt.figure(figsize=(figsize[0], figsize[1]))

    if has_conf_panel:
        gs = fig.add_gridspec(
            4,
            2,
            height_ratios=[4, 4, 1.2, 3],
            width_ratios=[1, 0.02],
            hspace=0.32,
            wspace=0.02,
            top=0.96,
        )
    else:
        gs = fig.add_gridspec(
            3,
            2,
            height_ratios=[4, 4, 3],
            width_ratios=[1, 0.02],
            hspace=0.32,
            wspace=0.02,
            top=0.96,
        )
    fig.suptitle(title or "IG Attribution Spectrum", fontsize=13, fontweight="bold", y=0.99)

    # ===== Panel 1: Index-Based Ion-Type Coloring =====
    ax1 = fig.add_subplot(gs[0, 0])
    ax1_spare = fig.add_subplot(gs[0, 1])
    ax1_spare.axis("off")
    _draw_index_panel_xai(ax1, v_mz, norm_int, peak_categories, peak_ann_display, masked_valid_set, n_valid, CATEGORY_COLORS)

    # ===== Panel 2: IG Attribution — full group MLM-masked =====
    ax2 = fig.add_subplot(gs[1, 0])
    ax2_cbar = fig.add_subplot(gs[1, 1])
    pred = result.prediction
    if pred is not None:
        bin_str = "BIN CORRECT" if pred.correct_bin else "BIN WRONG"
        grp_acc = f"GrpAcc: {pred.group_bin_accuracy:.0%}" if pred.group_bin_accuracy >= 0 else ""
        attr_title = (
            f"IG Attribution (full group MLM-masked) | "
            f"Pred: {pred.predicted_mz_da:.2f} Da | "
            f"True: {pred.true_mz_da:.2f} Da | "
            f"Err: {pred.error_da:.3f} Da ({pred.error_ppm:.1f} ppm) | "
            f"{bin_str} | {grp_acc} | LogProb: {pred.confidence:.4f}"
        )
    else:
        attr_title = "IG Attribution (full group MLM-masked)"
    _draw_attribution_panel(
        ax2,
        ax2_cbar,
        v_mz,
        norm_int,
        v_attr,
        peak_categories,
        peak_ann_display,
        masked_valid_set,
        child_valid_set,
        n_valid,
        panel_title=attr_title,
        show_children_as="masked",
    )

    # ===== Panel 2b: Per-Peak Confidence Strip (optional) =====
    if has_conf_panel:
        ax_conf = fig.add_subplot(gs[2, 0])
        ax_conf_cbar = fig.add_subplot(gs[2, 1])
        _draw_confidence_strip(ax_conf, ax_conf_cbar, v_conf, peak_categories, masked_valid_set, n_valid)

    # ===== Panel 3: Two sub-panels (prediction + ranked attribution) =====
    bottom_row = 3 if has_conf_panel else 2
    gs_bottom = gs[bottom_row, :].subgridspec(1, 2, wspace=0.3)
    ax3a = fig.add_subplot(gs_bottom[0, 0])
    ax3b = fig.add_subplot(gs_bottom[0, 1])

    _draw_prediction_panel(ax3a, result)
    _draw_ranked_attribution_panel(ax3b, result)

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def _draw_prediction_panel(ax: Axes, result: "AttributionResult") -> None:
    """Sub-panel 3a: per-peak prediction confidence and correctness.

    X-axis is negative log-prob (0 = perfect confidence, larger = less confident).
    """
    pred = result.prediction
    if pred is None or not pred.peak_predictions:
        ax.text(0.5, 0.5, "No per-peak predictions", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#999999")
        ax.set_title("Per-Peak Prediction Confidence", fontsize=10, fontweight="bold")
        ax.axis("off")
        return

    from matplotlib.patches import Patch

    peaks = sorted(pred.peak_predictions, key=lambda p: p.true_mz_da)

    labels = [p.annotation for p in peaks]
    neg_log_probs = [-p.log_prob for p in peaks]  # negate so 0 = confident, larger = uncertain
    colors = ["#27ae60" if p.correct_bin else "#e74c3c" for p in peaks]

    y_pos = np.arange(len(peaks))
    bars = ax.barh(y_pos, neg_log_probs, color=colors, alpha=0.85, edgecolor="white", height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Negative Log-Prob (lower = more confident)", fontsize=9)
    ax.set_title("Per-Peak Prediction Confidence", fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, None)

    # Show correct-bin rank inside wrong prediction bars
    for bar, p in zip(bars, peaks, strict=False):
        if not p.correct_bin and p.correct_bin_rank > 0:
            ax.text(
                bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"rank {p.correct_bin_rank}",
                ha="left",
                va="center",
                fontsize=7,
                color="#e74c3c",
                fontstyle="italic",
            )

    ax.legend(
        handles=[Patch(color="#27ae60", label="BIN CORRECT"), Patch(color="#e74c3c", label="BIN WRONG")],
        loc="lower right",
        fontsize=7,
        framealpha=0.8,
    )
    ax.grid(axis="x", alpha=0.3)


def _draw_ranked_attribution_panel(ax: Axes, result: "AttributionResult") -> None:
    """Sub-panel 3b: horizontal bar chart of top-N ranked peaks by attribution."""
    topk = result.topk
    if topk is None or not topk.ranked_peaks:
        ax.text(0.5, 0.5, "No ranked attribution data", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#999999")
        ax.set_title("Ranked Attribution (Fragment Groups)", fontsize=10, fontweight="bold")
        ax.axis("off")
        return

    category_colors = {
        "ladder_neighbor": "#3498db",
        "near_ladder": "#5dade2",
        "distant_same_series": "#aed6f1",
        "complementary_pair": "#e74c3c",
        "opposite_series": "#f1948a",
        "precursor": "#f39c12",
        "other_annotated": "#2ecc71",
        "charge_variant_leakage": "#c0392b",
        "internal_fragment": "#1abc9c",
        "d_ion": "#16a085",
        "w_ion": "#0e8274",
        "side_chain_loss": "#27ae60",
        "immonium_related": "#8e44ad",
        "precursor_combined_loss": "#d35400",
        "unannotated": "#95a5a6",
    }

    ranked = topk.ranked_peaks
    labels = [rp.annotation for rp in ranked]
    attrs = [rp.attribution for rp in ranked]
    colors = [category_colors.get(rp.category, "#bdc3c7") for rp in ranked]

    y_pos = np.arange(len(ranked))
    ax.barh(y_pos, attrs, color=colors, alpha=0.85, edgecolor="white", height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Attribution (sum across group)", fontsize=9)
    ax.set_title("Ranked Attribution (Fragment Groups)", fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # Legend for categories
    from matplotlib.patches import Patch

    seen_cats = list(dict.fromkeys(rp.category for rp in ranked))
    legend_handles = [Patch(color=category_colors.get(c, "#bdc3c7"), label=c.replace("_", " ")) for c in seen_cats]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7, framealpha=0.8)


def _draw_confidence_strip(
    ax: Axes,
    ax_cbar: Any,
    v_conf: Any,
    peak_categories: Any,
    masked_valid_set: Any,
    n_valid: Any,
) -> None:
    """Thin strip showing per-peak model confidence for visible peaks.

    Uses plasma colormap (0=dark/low confidence, 1=bright/high confidence)
    to visually distinguish from the viridis attribution panel above.
    Annotated peaks get a subtle black edge; unannotated are edgeless.
    Masked peaks use the same colormap but with a thick red edge.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    cmap = plt.cm.plasma
    vmax = max(float(np.percentile(v_conf, 99)), 0.01)
    norm = Normalize(vmin=0, vmax=vmax)

    np.arange(n_valid)
    for i in range(n_valid):
        color = cmap(norm(v_conf[i]))
        if i in masked_valid_set:
            edge = "#ff0000"
            lw = 1.2
        elif peak_categories[i] != "unannotated":
            edge = "#333333"
            lw = 0.3
        else:
            edge = "none"
            lw = 0.3
        ax.bar(i, v_conf[i], width=1.0, color=color, edgecolor=edge, linewidth=lw)

    ax.set_xlim(-0.5, n_valid - 0.5)
    ax.set_ylim(0, vmax * 1.15)
    ax.set_ylabel("Confidence", fontsize=8)
    ax.set_title("Per-Peak Model Confidence (unmasked context)", fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=0.2)

    # Annotated vs unannotated mean confidence
    ann_conf = [v_conf[i] for i in range(n_valid) if i not in masked_valid_set and peak_categories[i] != "unannotated"]
    unann_conf = [v_conf[i] for i in range(n_valid) if i not in masked_valid_set and peak_categories[i] == "unannotated"]
    legend_parts: list[Any] = []
    if ann_conf:
        legend_parts.append(f"Annotated mean: {np.mean(ann_conf):.3f}")
    if unann_conf:
        legend_parts.append(f"Unannotated mean: {np.mean(unann_conf):.3f}")
    if legend_parts:
        ax.text(
            0.99,
            0.92,
            " | ".join(legend_parts),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8},
        )

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, cax=ax_cbar)
    cb.set_label("Confidence", fontsize=7)
    cb.ax.tick_params(labelsize=6)


def _draw_mz_panel_xai(
    ax: Axes,
    mz: Any,
    norm_int: Any,
    peak_categories: Any,
    peak_ann_display: Any,
    masked_valid_set: Any,
    n_valid: Any,
    max_mz: Any,
    cat_colors: Any,
    text_colors: Any,
) -> None:
    """Panel 1: m/z spectrum with ion-type coloring. Masked peaks in red."""
    # Draw unannotated first, annotated on top, masked last
    for layer in ("unannotated", "annotated", "masked"):
        for i in range(n_valid):
            cat = peak_categories[i]
            if layer == "unannotated" and (cat != "unannotated" or i in masked_valid_set):
                continue
            if layer == "annotated" and (cat == "unannotated" or i in masked_valid_set):
                continue
            if layer == "masked" and i not in masked_valid_set:
                continue

            if i in masked_valid_set:
                color, alpha, zorder, lw = "#ff0000", 1.0, 5, 2.5
            else:
                color, alpha = cat_colors.get(cat, ("#9467bd", 0.8))
                zorder = 2 if cat == "unannotated" else 3
                lw = 1.2

            ax.plot([mz[i], mz[i]], [0, norm_int[i]], color=color, linewidth=lw, alpha=alpha, zorder=zorder)
            marker_s = 40 if i in masked_valid_set else 20
            ax.scatter([mz[i]], [norm_int[i]], color=color, s=marker_s, alpha=alpha, zorder=zorder, marker="*" if i in masked_valid_set else "o")

    # Annotation labels (same approach as confidence task)
    max_int = norm_int.max() if n_valid > 0 else 1.0
    min_label_intensity = max_int * 0.03
    min_mz_gap = 20.0
    placed_mz: List[float] = []

    annotated = [(i, mz[i], norm_int[i], peak_ann_display[i]) for i in range(n_valid) if peak_categories[i] != "unannotated" and peak_ann_display[i]]
    annotated.sort(key=lambda t: t[2], reverse=True)

    for idx, m, inten, display in annotated:
        if inten < min_label_intensity:
            continue
        if any(abs(m - p) < min_mz_gap for p in placed_mz):
            continue
        placed_mz.append(m)
        cat = peak_categories[idx]
        text_color = text_colors.get(cat, "black")
        if idx in masked_valid_set:
            text_color = "red"
        ax.annotate(
            display,
            xy=(m, inten),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color=text_color,
            rotation=90,
            alpha=0.9,
            fontweight="bold",
        )

    # Mark masked group in legend
    if masked_valid_set:
        ax.plot([], [], "r*", markersize=10, label="Masked group")
        ax.legend(fontsize=9, loc="upper right")

    ax.set_xlabel("m/z", fontsize=11, fontweight="bold")
    ax.set_ylabel("Normalized Intensity", fontsize=11, fontweight="bold")
    ax.set_title("m/z Spectrum — Masked Fragment Group Highlighted", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(50, max_mz)
    ax.set_ylim(0, max_int * 1.45)


def _draw_index_panel_xai(
    ax: Axes,
    mz: Any,
    norm_int: Any,
    peak_categories: Any,
    peak_ann_display: Any,
    masked_valid_set: Any,
    n_valid: Any,
    cat_colors: Any,
) -> None:
    """Panel 2: index-based bar plot with ion-type coloring. Masked peaks in red."""
    x_pos = np.arange(n_valid)

    # Group by category
    category_groups: Dict[str, List[int]] = {}
    for i in range(n_valid):
        if i in masked_valid_set:
            category_groups.setdefault("__masked__", []).append(i)
        else:
            cat = peak_categories[i]
            category_groups.setdefault(cat, []).append(i)

    # Unannotated first
    for cat in ["unannotated"]:
        idxs = category_groups.get(cat, [])
        if idxs:
            arr = np.array(idxs)
            color, alpha = cat_colors.get(cat, ("#BDBDBD", 0.5))
            ax.bar(x_pos[arr], norm_int[arr], color=color, alpha=alpha, width=1.0, linewidth=0, label=cat)

    # Annotated categories
    for cat in sorted(set(peak_categories) - {"unannotated"}):
        idxs = category_groups.get(cat, [])
        if idxs:
            arr = np.array(idxs)
            color, alpha = cat_colors.get(cat, ("#9467bd", 0.8))
            ax.bar(x_pos[arr], norm_int[arr], color=color, alpha=alpha, width=1.0, linewidth=0, label=cat)

    # Masked group in red
    masked_idxs = category_groups.get("__masked__", [])
    if masked_idxs:
        arr = np.array(masked_idxs)
        ax.bar(x_pos[arr], norm_int[arr], color="#ff0000", alpha=1.0, width=1.0, linewidth=0.8, edgecolor="darkred", label="MASKED")

    # Annotation labels
    for i in range(n_valid):
        if peak_categories[i] != "unannotated" and peak_ann_display[i]:
            color = "red" if i in masked_valid_set else "black"
            ax.annotate(
                peak_ann_display[i],
                xy=(i, norm_int[i]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                rotation=90,
                alpha=0.9,
                fontweight="bold",
                color=color,
            )

    # X-ticks
    tick_step = max(1, n_valid // 20)
    tick_pos = np.arange(0, n_valid, tick_step)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"{mz[i]:.0f}" for i in tick_pos], fontsize=8, rotation=45, ha="right")
    ax.set_xlim(-1, n_valid)
    max_int = norm_int.max() if n_valid > 0 else 1.0
    ax.set_ylim(0, max_int * 1.3)

    ax.set_xlabel("m/z (at peak index)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Normalized Intensity", fontsize=11, fontweight="bold")
    ax.set_title("Index-Based View with Ion-Type Coloring", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)


def _draw_attribution_panel(
    ax: Axes,
    ax_cbar: Any,
    mz: Any,
    norm_int: Any,
    attr_abs: Any,
    peak_categories: Any,
    peak_ann_display: Any,
    masked_valid_set: Any,
    child_valid_set: Any,
    n_valid: Any,
    panel_title: str = "IG Attribution",
    show_children_as: str = "masked",
) -> None:
    """Attribution panel: bars colored by IG magnitude (viridis)."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch

    x_pos = np.arange(n_valid)

    # Base ion index within the masked set (the one that is MLM-masked)
    base_valid_set = masked_valid_set - child_valid_set

    # Normalize attribution for coloring (exclude masked positions)
    attr_clean = attr_abs.copy()
    attr_pos = attr_clean[attr_clean > 0]
    attr_max = np.percentile(attr_pos, 99) if len(attr_pos) > 0 else 1.0
    norm = Normalize(vmin=0, vmax=attr_max)
    cmap = plt.cm.viridis

    is_annotated = np.array([c != "unannotated" for c in peak_categories])

    for i in range(n_valid):
        if i in base_valid_set:
            # Base ion: solid red with star
            ax.bar(x_pos[i], norm_int[i], color="#ff0000", alpha=1.0, width=1.0, edgecolor="darkred", linewidth=1.2)
        elif i in child_valid_set:
            # Children are MLM-masked (model sees mask tokens).
            # Show as red bars with hatching to distinguish from base.
            ax.bar(x_pos[i], norm_int[i], color="#ff0000", alpha=0.6, width=1.0, edgecolor="darkred", linewidth=0.8, hatch="//")
        elif is_annotated[i]:
            ax.bar(x_pos[i], norm_int[i], color=cmap(norm(attr_abs[i])), alpha=0.85, width=1.0, edgecolor="red", linewidth=0.8)
        else:
            ax.bar(x_pos[i], norm_int[i], color=cmap(norm(attr_abs[i])), alpha=0.85, width=1.0, linewidth=0)

    # Annotation labels
    for i in range(n_valid):
        if peak_categories[i] != "unannotated" and peak_ann_display[i]:
            color = "red" if i in masked_valid_set else "black"
            ax.annotate(
                peak_ann_display[i],
                xy=(i, norm_int[i]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                rotation=90,
                alpha=0.9,
                fontweight="bold",
                color=color,
            )

    # Top-20 unannotated peaks labeled with m/z
    unann_idx = np.array([i for i in range(n_valid) if not is_annotated[i] and i not in masked_valid_set])
    if len(unann_idx) > 0:
        top_k = min(20, len(unann_idx))
        top_unann = unann_idx[np.argsort(norm_int[unann_idx])[-top_k:]]
        for i in top_unann:
            ax.annotate(
                f"{mz[i]:.2f}",
                xy=(i, norm_int[i]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=6,
                rotation=90,
                alpha=0.7,
                fontstyle="italic",
                color="#555555",
            )

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=ax_cbar)
    ax_cbar.set_ylabel("Attribution (99th pctl)", fontsize=9, rotation=270, labelpad=12)

    # Summary stats (exclude masked group)
    exclude_set = masked_valid_set | child_valid_set
    ann_idx = np.array([i for i in range(n_valid) if is_annotated[i] and i not in exclude_set])
    unann_only = np.array([i for i in range(n_valid) if not is_annotated[i] and i not in exclude_set])
    mean_ann = float(np.mean(attr_abs[ann_idx])) if len(ann_idx) > 0 else 0.0
    mean_unann = float(np.mean(attr_abs[unann_only])) if len(unann_only) > 0 else 0.0

    # Legend
    legend_elements = [
        Patch(facecolor="#ff0000", edgecolor="darkred", linewidth=1.2, label="Base ion (masked)"),
    ]
    if child_valid_set:
        if show_children_as == "blanked":
            from matplotlib.lines import Line2D

            legend_elements.append(
                Line2D([0], [0], marker="x", color="#ff0000", linestyle="None", markersize=8, label=f"Children removed ({len(child_valid_set)})")
            )
        elif show_children_as == "masked":
            legend_elements.append(
                Patch(
                    facecolor="#ff0000",
                    edgecolor="darkred",
                    linewidth=0.8,
                    alpha=0.6,
                    hatch="//",
                    label=f"Children MLM-masked ({len(child_valid_set)})",
                )
            )
        else:
            legend_elements.append(Patch(facecolor=cmap(0.5), edgecolor="#ff6666", linewidth=0.8, label=f"Children visible ({len(child_valid_set)})"))
    legend_elements.extend(
        [
            Patch(facecolor=cmap(0.7), edgecolor="red", linewidth=0.8, label=f"Annotated (mean {mean_ann:.1f})"),
            Patch(facecolor=cmap(0.3), alpha=0.85, label=f"Unannotated (mean {mean_unann:.1f})"),
        ]
    )
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

    # X-ticks
    tick_step = max(1, n_valid // 20)
    tick_pos = np.arange(0, n_valid, tick_step)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"{mz[i]:.0f}" for i in tick_pos], fontsize=8, rotation=45, ha="right")
    ax.set_xlim(-1, n_valid)
    max_int_val = norm_int.max() if n_valid > 0 else 1.0
    ax.set_ylim(0, max_int_val * 1.3)

    ax.set_title(panel_title, fontsize=11, fontweight="bold")
    ax.set_xlabel("m/z (at peak index)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Normalized Intensity", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
