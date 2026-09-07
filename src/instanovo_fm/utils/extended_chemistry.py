"""Extended chemistry library for novel ion type detection.

Provides computation of ion types NOT covered by standard b/y/a + isotope +
H2O/NH3 neutral loss annotation frameworks:

1. Immonium-related ions — fixed-mass diagnostic fragments beyond primary
   immonium ions (~40 entries, from Falick et al. 1993)
2. Combined precursor losses — p-2H2O, p-H2O-NH3, p-CO-H2O etc.
3. Internal fragments — dual backbone cleavages (b-type and a-type)
4. Residue-specific side-chain losses — Met CH3SH, Asp/Glu CO2 from backbone ions
5. d/w-ions — side-chain cleavage from a/z-dot ions (rare in HCD, common in keV CID)

References:
    - Falick et al. 1993, JASMS 4:882-893 (immonium and related ions)
    - Michalski et al. 2012, J Proteome Res 11:5479-5491 (HCD spectrum characterization)
    - Paizs & Suhai 2005, Mass Spectrom Rev 24:508-548 (internal fragments)
    - Medzihradszky & Chalkley 2015, Mass Spectrom Rev 34:43-63 (d/w-ions)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Physical constants (must match ig_attribution_helper.py)
# ---------------------------------------------------------------------------

PROTON_MASS = 1.007276  # Da
WATER_MASS = 18.010565  # Da
CO_MASS = 27.994915  # Da (b→a ion: a = b - CO)
NH3_MASS = 17.026549  # Da
H_ATOM_MASS = 1.00782503  # Da (hydrogen atom, NOT proton — used for radical z-dot ions)

# Standard amino acid residue masses (monoisotopic)
AMINO_ACID_MASSES: Dict[str, float] = {
    "G": 57.02146,
    "A": 71.03711,
    "S": 87.03203,
    "P": 97.05276,
    "V": 99.06841,
    "T": 101.04768,
    "C": 103.00919,
    "I": 113.08406,
    "L": 113.08406,
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


# ---------------------------------------------------------------------------
# Immonium-related ions (Falick et al. 1993; IonSource.com reference)
# ---------------------------------------------------------------------------
# These are diagnostic low-mass ions beyond the primary immonium ion (R-CO-NH=CH+).
# The primary immonium ions ARE already annotated by DEFAULT_CUSTOM_IONS in
# theoretical_spectra.py. The RELATED ions below are NOT — they appear as
# unannotated peaks.
#
# Each entry: list of (mz, label) tuples. Only generated for peptides
# containing the corresponding amino acid.

IMMONIUM_RELATED_IONS: Dict[str, List[Tuple[float, str]]] = {
    "V": [
        (41.039, "rel_V_41"),      # C3H5+ (allyl cation)
        (55.055, "rel_V_55"),      # C4H7+
        (69.070, "rel_V_69"),      # C5H9+
    ],
    "I": [
        (44.050, "rel_I_44"),      # CO-NH2=CH2+ (shared with Leu)
        (72.081, "rel_I_72"),      # immonium - CO
    ],
    "L": [
        (44.050, "rel_L_44"),      # CO-NH2=CH2+ (shared with Ile)
        (72.081, "rel_L_72"),      # immonium - CO
    ],
    "N": [
        (70.029, "rel_N_70"),      # immonium - NH3
    ],
    "D": [
        (70.029, "rel_D_70"),      # immonium - H2O
    ],
    "Q": [
        (56.050, "rel_Q_56"),
        (84.045, "rel_Q_84"),      # immonium - NH3
        (129.066, "rel_Q_129"),    # immonium + CO
    ],
    "K": [
        (70.066, "rel_K_70"),
        (84.081, "rel_K_84"),      # immonium - NH3
        (112.076, "rel_K_112"),
        (129.102, "rel_K_129"),    # immonium + CO
    ],
    "M": [
        (61.011, "rel_M_61"),      # immonium - CH2S (= loss of thioformaldehyde)
    ],
    "H": [
        (82.053, "rel_H_82"),      # immonium - CO
        (121.076, "rel_H_121"),
        (123.055, "rel_H_123"),
        (138.066, "rel_H_138"),    # immonium + CO
        (166.061, "rel_H_166"),
    ],
    "F": [
        (91.054, "rel_F_91"),      # tropylium C7H7+ (diagnostic for Phe/Tyr)
    ],
    "Y": [
        (91.054, "rel_Y_91"),      # tropylium C7H7+
        (107.049, "rel_Y_107"),    # hydroxytropylium C7H7O+
    ],
    "R": [
        (59.048, "rel_R_59"),
        (70.066, "rel_R_70"),
        (73.064, "rel_R_73"),
        (87.056, "rel_R_87"),
        (100.087, "rel_R_100"),
        (112.087, "rel_R_112"),
    ],
    "W": [
        (77.039, "rel_W_77"),      # phenyl cation C6H5+
        (117.058, "rel_W_117"),    # indolium C8H7N+
        (130.066, "rel_W_130"),    # 3-methylindolium C9H8N+
        (132.081, "rel_W_132"),
        (170.060, "rel_W_170"),
        (171.092, "rel_W_171"),
    ],
}


# ---------------------------------------------------------------------------
# Combined precursor neutral losses
# ---------------------------------------------------------------------------
# Single losses (H2O, NH3, H3PO4, SO3) are already in the main annotation
# framework. Combined/double losses are NOT — peaks at these offsets from
# the precursor ion appear as unannotated.

COMBINED_PRECURSOR_LOSSES: List[Tuple[str, float]] = [
    ("2xH2O", 2 * 18.010565),              # 36.021 Da — double water loss
    ("H2O+NH3", 18.010565 + 17.026549),    # 35.037 Da — water + ammonia
    ("CO+H2O", 27.994915 + 18.010565),     # 46.005 Da — CO + water
    ("CO+NH3", 27.994915 + 17.026549),     # 45.021 Da — CO + ammonia
    ("2xNH3", 2 * 17.026549),              # 34.053 Da — double ammonia loss
]


# ---------------------------------------------------------------------------
# d-ion and w-ion side-chain loss data (from residue_ion_reference.csv)
# ---------------------------------------------------------------------------

@dataclass
class SideChainLoss:
    """Side-chain loss for d-ion or w-ion computation."""
    formula: str
    mass: float
    source: str = ""


SIDE_CHAIN_LOSSES: Dict[str, List[SideChainLoss]] = {
    "C": [SideChainLoss("SH", 32.979896)],
    "D": [SideChainLoss("CO2H", 44.997654)],
    "E": [SideChainLoss("C2H3O2", 59.013304)],
    "I": [
        SideChainLoss("C2H5", 29.039125),   # primary — distinguishes from Leu
        SideChainLoss("CH3", 15.023475),     # alternative (β-branch)
    ],
    "K": [SideChainLoss("C3H8N", 58.065674)],
    "L": [SideChainLoss("C3H7", 43.054775)],  # distinguishes from Ile
    "M": [SideChainLoss("C2H5S", 61.011196)],
    "N": [SideChainLoss("CONH2", 44.013639)],
    "Q": [SideChainLoss("C2H4NO", 58.029289)],
    "V": [SideChainLoss("CH3", 15.023475)],
}

LEU_ILE_DISCRIMINATING = {"I", "L"}

# Residue-specific side-chain losses from backbone ions (beyond standard H2O/NH3)
RESIDUE_SPECIFIC_LOSSES: Dict[str, List[Tuple[str, float]]] = {
    "M": [("CH3SH", 48.003371), ("CH3SOH", 63.998285)],
    "D": [("CO2", 43.989830)],
    "E": [("CO2", 43.989830)],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExtendedChemistryMatch:
    """A match of an observed peak to an extended chemistry hypothesis."""
    hypothesis_type: str       # "d_ion", "w_ion", "internal_fragment", "side_chain_loss",
                               # "immonium_related", "precursor_combined_loss"
    hypothesis_label: str      # e.g. "d4(I,-C2H5)", "rel_H_82", "p-2xH2O^2"
    expected_mz: float
    observed_mz: float
    error_da: float
    error_ppm: float
    residue: Optional[str] = None
    position: Optional[int] = None
    discriminates_leu_ile: bool = False


@dataclass
class ExtendedChemistryLibrary:
    """Precomputed extended chemistry hypotheses for a given peptide + charge."""
    sequence: str
    precursor_charge: int
    d_ions: List[Tuple[float, str, str, int]] = field(default_factory=list)
    w_ions: List[Tuple[float, str, str, int]] = field(default_factory=list)
    internal_fragments: List[Tuple[float, str]] = field(default_factory=list)
    side_chain_losses: List[Tuple[float, str, str]] = field(default_factory=list)
    immonium_related: List[Tuple[float, str, str]] = field(default_factory=list)
    # (mz, label, residue)
    precursor_combined_losses: List[Tuple[float, str]] = field(default_factory=list)
    # (mz, label)

    @property
    def total_hypotheses(self) -> int:
        return (len(self.d_ions) + len(self.w_ions) + len(self.internal_fragments)
                + len(self.side_chain_losses) + len(self.immonium_related)
                + len(self.precursor_combined_losses))


# ---------------------------------------------------------------------------
# Ion mass computation
# ---------------------------------------------------------------------------

def _get_residue_mass(
    residue: str,
    residue_mass_fn: Optional[Callable[[str], float]] = None,
) -> float:
    if residue_mass_fn is not None:
        return residue_mass_fn(residue)
    aa = residue[0]
    return AMINO_ACID_MASSES.get(aa, 0.0)


def _has_modification(residue: str) -> bool:
    return "[" in residue


def compute_d_ions(
    residues: Sequence[str],
    precursor_charge: int = 2,
    residue_mass_fn: Optional[Callable[[str], float]] = None,
) -> List[Tuple[float, str, str, int]]:
    """Compute d-ion m/z values.

    d_i = a_i - side_chain_radical + H_atom
    where a_i = b_i - CO, b_i = sum(residue_masses[0:i])

    The +H arises because the a-ion (even-electron) loses a side-chain radical
    (homolytic cleavage), and a hydrogen migrates to stabilize the product ion.
    w-ions do NOT need this correction because z-dot is already a radical.
    """
    results: List[Tuple[float, str, str, int]] = []
    L = len(residues)
    cum_mass = 0.0
    for i in range(L - 1):
        cum_mass += _get_residue_mass(residues[i], residue_mass_fn)
        position = i + 1
        aa = residues[i][0]
        if aa not in SIDE_CHAIN_LOSSES or _has_modification(residues[i]):
            continue
        a_neutral = cum_mass - CO_MASS
        for loss in SIDE_CHAIN_LOSSES[aa]:
            d_neutral = a_neutral - loss.mass + H_ATOM_MASS  # +H from radical stabilization
            if d_neutral <= 0:
                continue
            for z in range(1, precursor_charge + 1):
                mz = (d_neutral + z * PROTON_MASS) / z
                if mz <= 0:
                    continue
                charge_str = "+" * z
                label = f"d{position}{charge_str}({aa},-{loss.formula})"
                results.append((mz, label, aa, position))
    return results


def compute_w_ions(
    residues: Sequence[str],
    precursor_charge: int = 2,
    residue_mass_fn: Optional[Callable[[str], float]] = None,
) -> List[Tuple[float, str, str, int]]:
    """Compute w-ion m/z values.

    z_j(neutral) = sum(residue_masses[L-j:L]) + H2O - NH3
    z_j(dot) = z_j(neutral) + H_atom
    w_j = z_j(dot) - side_chain_loss(residue at N-terminal edge of z-fragment)
    """
    results: List[Tuple[float, str, str, int]] = []
    L = len(residues)
    cum_mass = 0.0
    for j in range(1, L):
        cum_mass += _get_residue_mass(residues[L - j], residue_mass_fn)
        position = j
        target_idx = L - j
        aa = residues[target_idx][0]
        if aa not in SIDE_CHAIN_LOSSES or _has_modification(residues[target_idx]):
            continue
        z_dot_neutral = cum_mass + WATER_MASS - NH3_MASS + H_ATOM_MASS
        for loss in SIDE_CHAIN_LOSSES[aa]:
            w_neutral = z_dot_neutral - loss.mass
            if w_neutral <= 0:
                continue
            for z in range(1, precursor_charge + 1):
                mz = (w_neutral + z * PROTON_MASS) / z
                if mz <= 0:
                    continue
                charge_str = "+" * z
                label = f"w{position}{charge_str}({aa},-{loss.formula})"
                results.append((mz, label, aa, position))
    return results


def compute_internal_fragments(
    residues: Sequence[str],
    max_length: int = 6,
    residue_mass_fn: Optional[Callable[[str], float]] = None,
) -> List[Tuple[float, str]]:
    """Compute internal fragment ion m/z values (z=1 only).

    b-type internal: neutral_mass = sum(residue_masses[i:j])
    a-type internal: neutral_mass = sum(residue_masses[i:j]) - CO
    """
    results: List[Tuple[float, str]] = []
    L = len(residues)
    masses = [_get_residue_mass(r, residue_mass_fn) for r in residues]
    prefix = [0.0] * (L + 1)
    for i in range(L):
        prefix[i + 1] = prefix[i] + masses[i]

    for start in range(1, L - 1):
        for end in range(start + 1, min(start + max_length, L - 1)):
            cum = prefix[end + 1] - prefix[start]
            subseq = "".join(r[0] for r in residues[start:end + 1])

            b_mz = cum + PROTON_MASS
            if b_mz > 0:
                results.append((b_mz, f"int(b,{start + 1}-{end + 1},{subseq})"))

            a_mz = cum - CO_MASS + PROTON_MASS
            if a_mz > 0:
                results.append((a_mz, f"int(a,{start + 1}-{end + 1},{subseq})"))

    return results


def compute_side_chain_losses(
    residues: Sequence[str],
    precursor_charge: int = 2,
    residue_mass_fn: Optional[Callable[[str], float]] = None,
) -> List[Tuple[float, str, str]]:
    """Compute residue-specific side-chain losses from backbone b/y-ions."""
    results: List[Tuple[float, str, str]] = []
    L = len(residues)
    masses = [_get_residue_mass(r, residue_mass_fn) for r in residues]

    b_neutral = []
    cum = 0.0
    for i in range(L - 1):
        cum += masses[i]
        b_neutral.append(cum)

    y_neutral = []
    cum = 0.0
    for j in range(L - 1, 0, -1):
        cum += masses[j]
        y_neutral.append(cum)

    for pos, b_mass in enumerate(b_neutral):
        position = pos + 1
        for r_idx in range(position):
            aa = residues[r_idx][0]
            if aa not in RESIDUE_SPECIFIC_LOSSES or _has_modification(residues[r_idx]):
                continue
            for loss_name, loss_mass in RESIDUE_SPECIFIC_LOSSES[aa]:
                loss_neutral = b_mass - loss_mass
                if loss_neutral <= 0:
                    continue
                for z in range(1, precursor_charge + 1):
                    mz = (loss_neutral + z * PROTON_MASS) / z
                    charge_str = "+" * z
                    results.append((mz, f"b{position}{charge_str}-{loss_name}({aa})", "b"))

    for pos, y_mass in enumerate(y_neutral):
        position = pos + 1
        for r_idx in range(L - position, L):
            aa = residues[r_idx][0]
            if aa not in RESIDUE_SPECIFIC_LOSSES or _has_modification(residues[r_idx]):
                continue
            for loss_name, loss_mass in RESIDUE_SPECIFIC_LOSSES[aa]:
                loss_neutral = y_mass + WATER_MASS - loss_mass
                if loss_neutral <= 0:
                    continue
                for z in range(1, precursor_charge + 1):
                    mz = (loss_neutral + z * PROTON_MASS) / z
                    charge_str = "+" * z
                    results.append((mz, f"y{position}{charge_str}-{loss_name}({aa})", "y"))

    return results


def compute_immonium_related_ions(
    residues: Sequence[str],
) -> List[Tuple[float, str, str]]:
    """Compute immonium-related ion m/z values for a peptide.

    These are fixed-mass diagnostic fragments (tropylium, indolium, etc.) that
    are generated by specific amino acids. Only ions for amino acids PRESENT
    in the sequence are included — this is the key for null model comparison.

    Returns:
        List of (mz, label, residue) tuples. All singly charged.
    """
    # Get unique amino acids in the sequence (single-letter codes)
    aa_set = set(r[0] for r in residues)

    results: List[Tuple[float, str, str]] = []
    for aa in aa_set:
        if aa in IMMONIUM_RELATED_IONS:
            for mz, label in IMMONIUM_RELATED_IONS[aa]:
                results.append((mz, label, aa))
    return results


def compute_immonium_related_negative_control(
    residues: Sequence[str],
) -> List[Tuple[float, str, str]]:
    """Compute immonium-related ions for amino acids NOT in the peptide.

    This is the composition-based null model for immonium-related ions:
    scrambling the sequence doesn't change the composition, so instead we
    test against ions from amino acids that are absent.

    Returns:
        List of (mz, label, residue) tuples for ABSENT amino acids.
    """
    aa_set = set(r[0] for r in residues)
    results: List[Tuple[float, str, str]] = []
    for aa, ions in IMMONIUM_RELATED_IONS.items():
        if aa not in aa_set:
            for mz, label in ions:
                results.append((mz, label, aa))
    return results


def compute_combined_precursor_losses(
    precursor_mz: float,
    precursor_charge: int = 2,
) -> List[Tuple[float, str]]:
    """Compute combined precursor neutral loss m/z values.

    Single losses (H2O, NH3, H3PO4, SO3) are handled by the main annotation
    framework. This computes double/combined losses that are NOT annotated.

    For precursor at charge z: loss_mz = precursor_mz - (loss_mass / z)

    Returns:
        List of (mz, label) tuples.
    """
    results: List[Tuple[float, str]] = []
    for loss_name, loss_mass in COMBINED_PRECURSOR_LOSSES:
        for z in range(1, precursor_charge + 1):
            loss_mz = precursor_mz - (loss_mass / z)
            if loss_mz > 0:
                charge_str = "+" * z
                results.append((loss_mz, f"p-{loss_name}{charge_str}"))
    return results


# ---------------------------------------------------------------------------
# Library builder and matcher
# ---------------------------------------------------------------------------

def build_extended_chemistry_library(
    residues: Sequence[str],
    precursor_charge: int = 2,
    precursor_mz: float = 0.0,
    max_internal_length: int = 6,
    residue_mass_fn: Optional[Callable[[str], float]] = None,
) -> ExtendedChemistryLibrary:
    """Build a complete extended chemistry library for a peptide.

    Args:
        residues: Pre-parsed residue tokens
        precursor_charge: Precursor charge state
        precursor_mz: Precursor m/z (needed for combined precursor losses)
        max_internal_length: Maximum internal fragment subsequence length
        residue_mass_fn: Optional custom mass function for modified residues
    """
    sequence = "".join(r[0] for r in residues)

    return ExtendedChemistryLibrary(
        sequence=sequence,
        precursor_charge=precursor_charge,
        d_ions=compute_d_ions(residues, precursor_charge, residue_mass_fn),
        w_ions=compute_w_ions(residues, precursor_charge, residue_mass_fn),
        internal_fragments=compute_internal_fragments(
            residues, max_internal_length, residue_mass_fn,
        ),
        side_chain_losses=compute_side_chain_losses(
            residues, precursor_charge, residue_mass_fn,
        ),
        immonium_related=compute_immonium_related_ions(residues),
        precursor_combined_losses=compute_combined_precursor_losses(
            precursor_mz, precursor_charge,
        ) if precursor_mz > 0 else [],
    )


def match_peak_to_extended_chemistry(
    observed_mz: float,
    library: ExtendedChemistryLibrary,
    ppm_tol: float = 10.0,
    da_tol: Optional[float] = None,
) -> List[ExtendedChemistryMatch]:
    """Test a single observed m/z against all extended chemistry hypotheses."""
    matches: List[ExtendedChemistryMatch] = []

    def _within_tol(expected: float) -> Tuple[bool, float, float]:
        err_da = abs(observed_mz - expected)
        err_ppm = (err_da / expected) * 1e6 if expected > 0 else float("inf")
        if da_tol is not None:
            return err_da <= da_tol, err_da, err_ppm
        return err_ppm <= ppm_tol, err_da, err_ppm

    for mz, label, residue, position in library.d_ions:
        ok, err_da, err_ppm = _within_tol(mz)
        if ok:
            matches.append(ExtendedChemistryMatch(
                hypothesis_type="d_ion", hypothesis_label=label,
                expected_mz=mz, observed_mz=observed_mz,
                error_da=err_da, error_ppm=err_ppm,
                residue=residue, position=position,
                discriminates_leu_ile=residue in LEU_ILE_DISCRIMINATING,
            ))

    for mz, label, residue, position in library.w_ions:
        ok, err_da, err_ppm = _within_tol(mz)
        if ok:
            matches.append(ExtendedChemistryMatch(
                hypothesis_type="w_ion", hypothesis_label=label,
                expected_mz=mz, observed_mz=observed_mz,
                error_da=err_da, error_ppm=err_ppm,
                residue=residue, position=position,
                discriminates_leu_ile=residue in LEU_ILE_DISCRIMINATING,
            ))

    for mz, label in library.internal_fragments:
        ok, err_da, err_ppm = _within_tol(mz)
        if ok:
            matches.append(ExtendedChemistryMatch(
                hypothesis_type="internal_fragment", hypothesis_label=label,
                expected_mz=mz, observed_mz=observed_mz,
                error_da=err_da, error_ppm=err_ppm,
            ))

    for mz, label, parent_type in library.side_chain_losses:
        ok, err_da, err_ppm = _within_tol(mz)
        if ok:
            matches.append(ExtendedChemistryMatch(
                hypothesis_type="side_chain_loss", hypothesis_label=label,
                expected_mz=mz, observed_mz=observed_mz,
                error_da=err_da, error_ppm=err_ppm,
            ))

    for mz, label, residue in library.immonium_related:
        ok, err_da, err_ppm = _within_tol(mz)
        if ok:
            matches.append(ExtendedChemistryMatch(
                hypothesis_type="immonium_related", hypothesis_label=label,
                expected_mz=mz, observed_mz=observed_mz,
                error_da=err_da, error_ppm=err_ppm,
                residue=residue,
            ))

    for mz, label in library.precursor_combined_losses:
        ok, err_da, err_ppm = _within_tol(mz)
        if ok:
            matches.append(ExtendedChemistryMatch(
                hypothesis_type="precursor_combined_loss", hypothesis_label=label,
                expected_mz=mz, observed_mz=observed_mz,
                error_da=err_da, error_ppm=err_ppm,
            ))

    return matches


def match_peaks_batch(
    observed_mz_array: np.ndarray,
    library: ExtendedChemistryLibrary,
    ppm_tol: float = 10.0,
    da_tol: Optional[float] = None,
) -> Dict[int, List[ExtendedChemistryMatch]]:
    """Match multiple observed peaks against an extended chemistry library."""
    results: Dict[int, List[ExtendedChemistryMatch]] = {}
    for idx, mz in enumerate(observed_mz_array):
        if mz <= 0:
            continue
        matches = match_peak_to_extended_chemistry(mz, library, ppm_tol, da_tol)
        if matches:
            results[idx] = matches
    return results
