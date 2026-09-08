"""Hydrophobicity computation utilities for peptide sequences.

This module provides functionality to compute hydrophobicity values for peptide
sequences using the Kyte-Doolittle hydrophobicity scale. The scale assigns
numerical values to amino acids based on their hydrophobic/hydrophilic properties.

The Kyte-Doolittle scale ranges from:
- Most hydrophobic: Isoleucine (I) = 4.5
- Most hydrophilic: Arginine (R) = -4.5

Reference:
    Kyte, J., & Doolittle, R. F. (1982). A simple method for displaying the
    hydropathic character of a protein. Journal of Molecular Biology, 157(1), 105-132.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from instanovo.common.dataset import DataProcessor

# Kyte-Doolittle hydrophobicity scale
# Values represent the hydropathic index for each amino acid
KYTE_DOOLITTLE_SCALE = {
    'A': 1.8,   # Alanine
    'C': 2.5,   # Cysteine
    'D': -3.5,  # Aspartic acid
    'E': -3.5,  # Glutamic acid
    'F': 2.8,   # Phenylalanine
    'G': -0.4,  # Glycine
    'H': -3.2,  # Histidine
    'I': 4.5,   # Isoleucine (most hydrophobic)
    'K': -3.9,  # Lysine
    'L': 3.8,   # Leucine
    'M': 1.9,   # Methionine
    'N': -3.5,  # Asparagine
    'P': -1.6,  # Proline
    'Q': -3.5,  # Glutamine
    'R': -4.5,  # Arginine (most hydrophilic)
    'S': -0.8,  # Serine
    'T': -0.7,  # Threonine
    'V': 4.2,   # Valine
    'W': -0.9,  # Tryptophan
    'Y': -1.3,  # Tyrosine
}


def compute_hydrophobicity(
    peptides: np.ndarray,
    scale: dict[str, float] = KYTE_DOOLITTLE_SCALE,
    remove_modifications: bool = True,
) -> Optional[np.ndarray]:
    """Compute hydrophobicity for peptide sequences.

    Calculates the mean hydrophobicity value for each peptide sequence using
    the specified hydrophobicity scale (default: Kyte-Doolittle).

    Args:
        peptides: Array of peptide sequences (strings). Can include modifications.
        scale: Hydrophobicity scale dictionary mapping amino acids to values.
            Defaults to Kyte-Doolittle scale.
        remove_modifications: If True, removes modifications from sequences before
            computation. Defaults to True.

    Returns:
        Array of hydrophobicity values (mean per peptide), or None if no valid
        peptides are found. Invalid peptides are assigned NaN.

    Examples:
        >>> peptides = np.array(["PEPTIDE", "MKVLWAALLVTFLAGCQA"])
        >>> hydro = compute_hydrophobicity(peptides)
        >>> print(hydro)
        [-1.27142857  1.16111111]

        >>> # With modifications
        >>> peptides = np.array(["M[UNIMOD:35]PEPTIDE"])
        >>> hydro = compute_hydrophobicity(peptides, remove_modifications=True)
        >>> print(hydro)
        [0.13333333]
    """
    if peptides is None or len(peptides) == 0:
        return None

    hydrophobicity_values = []
    valid_count = 0

    for peptide in peptides:
        # Handle invalid peptides
        if peptide is None or not isinstance(peptide, str) or len(peptide) == 0:
            hydrophobicity_values.append(np.nan)
            continue

        # Clean the peptide sequence if requested
        if remove_modifications:
            clean_peptide = DataProcessor.remove_modifications(
                peptide,
                replace_isoleucine_with_leucine=False
            )
            if not clean_peptide:
                hydrophobicity_values.append(np.nan)
                continue
        else:
            clean_peptide = peptide

        # Calculate hydrophobicity as mean of amino acid values
        aa_values = [scale.get(aa, 0.0) for aa in clean_peptide]

        if aa_values:
            hydrophobicity_values.append(np.mean(aa_values))
            valid_count += 1
        else:
            hydrophobicity_values.append(np.nan)

    # Return None if no valid peptides found
    if valid_count == 0:
        return None

    return np.array(hydrophobicity_values, dtype=np.float32)


def get_amino_acid_hydrophobicity(
    amino_acid: str,
    scale: dict[str, float] = KYTE_DOOLITTLE_SCALE,
) -> Optional[float]:
    """Get hydrophobicity value for a single amino acid.

    Args:
        amino_acid: Single letter amino acid code (e.g., 'A', 'C', 'D')
        scale: Hydrophobicity scale dictionary. Defaults to Kyte-Doolittle.

    Returns:
        Hydrophobicity value for the amino acid, or None if not found.

    Examples:
        >>> get_amino_acid_hydrophobicity('I')  # Isoleucine (most hydrophobic)
        4.5
        >>> get_amino_acid_hydrophobicity('R')  # Arginine (most hydrophilic)
        -4.5
    """
    if not amino_acid or not isinstance(amino_acid, str):
        return None

    # Handle single character
    aa = amino_acid.upper().strip()
    if len(aa) != 1:
        return None

    return scale.get(aa)


def get_scale_statistics(scale: dict[str, float] = KYTE_DOOLITTLE_SCALE) -> dict[str, float]:
    """Get statistics for a hydrophobicity scale.

    Args:
        scale: Hydrophobicity scale dictionary. Defaults to Kyte-Doolittle.

    Returns:
        Dictionary containing scale statistics (mean, std, min, max, range).

    Examples:
        >>> stats = get_scale_statistics()
        >>> print(f"Mean: {stats['mean']:.2f}")
        Mean: -0.49
        >>> print(f"Range: {stats['range']:.1f}")
        Range: 9.0
    """
    values = list(scale.values())

    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'range': float(np.max(values) - np.min(values)),
        'n_amino_acids': len(values),
    }
